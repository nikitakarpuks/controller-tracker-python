"""HeuristicPoseFusionFilter -- cost-function-based fusion of vision pose
solutions with IMU dead-reckoning for one controller.

TWO separate mechanisms, deliberately decoupled, each doing one job:

1. TRACKING STATE (self.R, self.p) -- an IMU+vision-only weighted blend, no
   smoothing term. Vision solutions are fused with the IMU-predicted pose by
   minimizing

       J(p) = w_imu * ||p - p_imu||^2 + w_vision * ||p - p_vision||^2

   whose closed-form minimizer is a weighted average of the two candidate
   positions (rotation uses the same weights on tangent-space error vectors,
   linearized around the IMU prediction). w_vision scales with this frame's
   own match quality (n_inliers, mean reprojection error), see
   _vision_weight -- a weak vision solve contributes little to nothing, a
   strong one gets the full configured weight. w_imu decays linearly to 0
   over imu_decay_frames consecutive lost/coasted real frames (see
   frames_since_update) -- IMU dead-reckoning is validated reliable only for
   a short horizon past the last real correction, so its say fades out the
   longer it's been coasting unconstrained, regardless of how the gap looked
   in elapsed time. Once that decay hits exactly 0, vision is the ONLY term
   left, so the tracking state becomes EXACTLY the raw vision measurement,
   not "mostly vision." Below that threshold, w_vision can also be damped by
   a separate long-gap-since-last-update penalty (see _gap_vision_scale),
   itself quality-aware (skipped for a genuinely strong post-gap match).
   This state is what predict()/dead-reckoning anchor off next frame -- it
   is intentionally NEVER smoothed, so smoothing can't add lag to the
   tracking math itself.

2. REPORTED/DISPLAYED POSE (self.reported_R, self.reported_p) -- what
   downstream consumers (rerun display, pose_csv, etc.) actually see. This
   used to be a THIRD term in the cost function above (w_smooth *
   ||p - p_prev||^2, pulling toward the previous FUSED state) -- replaced
   entirely by a One Euro Filter (src/one_euro_filter.py) applied to the
   tracking state's output on every try_update() branch (bootstrap,
   fail_open, implausible_reject, fused). Why the replacement: the old
   smoothing term was entangled inside the same normalized weighted average
   as w_imu/w_vision, so its EFFECTIVE strength silently shifted every frame
   depending on what those two happened to be that frame -- not a legible,
   independently-tunable "how much am I smoothing" knob. A One Euro Filter
   is exactly the standard tool for this project's actual complaint (small
   sharp jitter at low/no motion, unacceptable lag during real motion): its
   cutoff adapts to the signal's own recent speed, so it smooths hard at
   rest and gets out of the way during fast motion, with two intuitive,
   independent parameters (min_cutoff, beta) instead of one weight that
   means something different every frame.

This is a soft, continuous stand-in for hard gating on the tracking-state
side: no self-calibration -- every vision solution that comes in gets fused
into the tracking state, just with a quality- and staleness-dependent say in
the outcome (until staleness alone settles it) -- with jitter removal
applied afterward, on the reported pose only.

The one HARD exception is the implausibility gate in try_update: a position/
rotation jump versus the IMU prediction far beyond real hand motion (a
likely identity swap -- this controller's search matched a SIBLING
controller's own blobs -- or some other degenerate mismatch) is a genuine
reject -- state is left at the IMU prediction, try_update returns False,
neither the inlier nor error ramp reliably catches a low-point fit that
happens to land a deceptively clean reprojection error against entirely the
wrong geometry, so this check runs before the cost-function fuse even sees
the candidate. This USED to still return True (fusing the candidate in at
zero vision weight but otherwise advancing state/last_update_ts_ns like any
other accepted frame) -- changed after that turned out to silently defeat
ControllerTracker._commit_fused_solution's and _mark_all_lost's loss-streak/
force-cold-start bookkeeping (both keyed off try_update's return value): a
single stray implausible detection was enough to indefinitely reset the
grace-period clock and the IMU-only coasting budget, letting a genuinely
lost controller coast on a contaminated prediction forever instead of ever
being re-declared cold.

Same public interface as PoseFusionFilter (src/pose_fusion.py): constructor
signature, try_update, predict, predict_dense, debug_snapshot, reset,
should_force_cold_start, set_siblings, .reported_R/.reported_p/
.consecutive_rejects/.last_update_ts_ns, .trust_score()/.trust() --
selectable via config.yml's fusion.filter_type (controller.py picks the
class). Reuses the same validated IMU dead-reckoning primitives
(predict_world_pose) unchanged -- only the fusion step itself is new.
"""
from collections import deque

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

from src.imu_data import predict_world_pose, slice_imu_to_window, dead_reckon_dense, \
    predict_headset_relative_pose, MOCAP_ROOM_G_WORLD, peak_gyro_accel_over_window, \
    integrate_gyro_segment, gyro_window_clip_flags, headset_rot_coast_budget_s, \
    effective_coast_budget_s as _shared_effective_coast_budget_s
from src.mocap_data import world_pose, headset_angular_velocity, headset_linear_velocity
from src.one_euro_filter import OneEuroFilter, OneEuroRotationFilter

_log = logger.bind(cat="pose_fusion")

# Below this elapsed gap, a try_update call is an ordinary frame-to-frame update --
# no gap penalty applies. Same value/rationale as PoseFusionFilter's own
# _WARM_FRAME_DT_CUTOFF_S (src/pose_fusion.py): this recording's real inter-update
# dt during continuous tracking clusters tightly around ~0.0221s (p50 through p95
# within ~0.0002s of each other), then jumps straight to >=0.05s for any real
# occlusion/reacquisition gap -- 0.04s sits in that empty gap with headroom on
# both sides. Kept as a separate literal (not imported from pose_fusion.py) since
# the two filter implementations are deliberately independent.
_GAP_DT_NORMAL_S = 0.04


# Returned instead of "infinity" once the state is too old for the continuous rotation gate to mean
# anything: just under the 180 deg maximum possible rotation, so every consumer that scales with the
# ceiling (Case A's pushback ramp, _agreement_weak_bounds, _vision_weight) stays finite and monotone.
_ROT_GATE_EXPIRED_DEG = 179.0


def rot_gate_threshold_deg(dt_s: float, headset_active: bool, saturated: bool, base_deg: float = 40.0,
                            sat_allow_deg: float = 25.0, nohs_allow_dps: float = 500.0,
                            cap_hs_deg: float = 75.0, cap_nohs_deg: float = 150.0,
                            max_state_age_s: float = 3.0) -> float:
    """Continuous rotation-innovation threshold (deg) between a vision candidate and the gyro-
    propagated prediction, valid at EVERY elapsed time dt_s since the last accepted state -- there is
    no "dt <= budget" on/off switch (that fail-open switch let a 171 deg mirror lock through 0.244 s
    after the last accept: walk_medium frame 348; see analysis/imu_thresholds_2026-09-25/FINAL_PLAN.md).

        headset ego-motion correction ACTIVE :  T = min(cap_hs,   base + sat_allow*saturated)
        NOT available for this prediction    :  T = min(cap_nohs, base + sat_allow*saturated + nohs_allow_dps*dt)

    The gyro-integrated rotation prediction is accurate (p99 rotation error ~14-33 deg out to 1.5 s once
    the headset's own rotation is compensated), so the headset-active gate is flat. Without the headset
    correction the prediction ignores headset rotation, whose rate reaches ~390 deg/s in these
    recordings, hence the additive nohs_allow_dps*dt term. sat_allow_deg covers gyro CLIPPING (see
    src.imu_data.gyro_window_clip_flags): a burst pinned at the sensor ceiling under-integrates by
    30-45 deg. The old fitted 0.05*peak_gyro*dt term is deliberately NOT used: it was fitted to ~3
    saturation events, is not physical, and is unbounded.

    Beyond max_state_age_s (3 s) the state is too stale to gate against: returns _ROT_GATE_EXPIRED_DEG,
    which no real innovation exceeds meaningfully. Between ~0.35 s and 3 s the gate stays at its finite
    value (measured on 588 good and 81 bad candidates with dt > 0.35 s: good headset-corrected innovation
    max 30 deg, bad >= 108 deg; T=40 had 0 false positives when the state was itself good). Negative dt_s
    is clamped to 0."""
    dt = max(0.0, float(dt_s))
    if dt > float(max_state_age_s):
        return _ROT_GATE_EXPIRED_DEG
    t = float(base_deg) + (float(sat_allow_deg) if saturated else 0.0)
    if headset_active:
        return min(float(cap_hs_deg), t)
    return min(float(cap_nohs_deg), t + float(nohs_allow_dps) * dt)


def _fmt_v(v) -> str:
    """Compact (x, y, z) meters formatting for the diagnostic prints below."""
    if v is None:
        return "None"
    return f"({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})"


def _ramp_up(value: float, weak: float, strong: float) -> float:
    """0 at/below `weak`, 1 at/above `strong`, linear in between -- for a
    signal where HIGHER is better (e.g. n_inliers)."""
    if strong <= weak:
        return 1.0
    return float(np.clip((value - weak) / (strong - weak), 0.0, 1.0))


def _ramp_down(value: float, weak: float, strong: float) -> float:
    """0 at/above `weak`, 1 at/below `strong`, linear in between -- for a
    signal where LOWER is better (e.g. mean reprojection error)."""
    if weak <= strong:
        return 1.0
    return float(np.clip((weak - value) / (weak - strong), 0.0, 1.0))


class HeuristicPoseFusionFilter:
    """Fuses vision pose solutions with IMU dead-reckoning for one controller
    via a 2-term weighted cost (IMU prediction, vision measurement) for the
    internal tracking state, plus a One Euro Filter applied separately to
    the reported/displayed pose -- see module docstring for why these are
    two decoupled mechanisms, not one blend. No covariance matrix -- state
    is just (R, p, v, last_update_ts_ns).

    fail-open, same contract as PoseFusionFilter: any missing precondition
    (no IMU coverage, no converged g_world, no prior state yet) makes
    predict()/try_update() behave as if the filter weren't there."""

    def __init__(self, gyro_data, accel_data, lever_arm, g_world_estimator, cfg: dict, ctrl_name: str = "?",
                 headset_mocap=None, g_world_estimator_abs=None):
        self._gyro_data = gyro_data
        self._accel_data = accel_data
        self._lever_arm = lever_arm
        self._g_world_estimator = g_world_estimator
        # Headset-ego-motion correction (see predict()'s _predict_with_headset_correction) --
        # both None unless mocap.enabled and a headset mocap trajectory actually loaded
        # (main.py/ControllerTracker). Either being None/unconverged makes predict() fall back
        # to the plain predict_world_pose path below, same fail-open contract as everything else
        # in this class.
        self._headset_mocap = headset_mocap
        self._g_world_estimator_abs = g_world_estimator_abs
        self._cfg = cfg                                   # shared top-level fusion: block (max_coast_s,
                                                            # trust_gap_tau_s -- same keys/semantics PoseFusionFilter reads)
        self._hc = cfg.get("fusion_heuristic", {}) or {}   # this filter's own namespaced knobs
        self._ctrl_name = ctrl_name                        # label only, for the diagnostic prints below

        self.R = None
        self.p = None
        # Last real accepted position + its timestamp, updated alongside
        # self.p at every genuine accept (fail-open, ordinary blend,
        # _accept_cold_reacquire) but, UNLIKE self.p, deliberately NOT
        # cleared by reset() (2026-09-16) -- see reset()'s own comment.
        # Sole consumer: TrackingSystem._detect_cold_identity_swap, which
        # needs "where did this controller last look like it genuinely
        # was, before things went wrong" as a fallback reference once
        # self.p itself has been wiped by a proven-wrong reset -- exactly
        # the gap that let a real identity swap go undetected (static_dark,
        # right_controller: this controller was fully reset, then re-
        # bootstrapped near where the OTHER controller used to be, with no
        # surviving reference for the swap-check to compare against).
        self._last_known_p: np.ndarray | None = None
        self._last_known_p_ts_ns: int | None = None
        self.v = np.zeros(3)
        # True once self.v has been set from an ACTUAL measured displacement
        # -- a real two-point finite difference over a SHORT, known gap, in
        # try_update's normal fused branch -- never from a bare reset/
        # bootstrap/fail-open zero-init, and (2026-09-13) never from
        # _try_cold_reacquire either: that method's own reacquisition happens
        # after an unknown-length loss against a stale pre-loss position, so
        # a straight-line difference there is not a real velocity measurement
        # (see that method's own comment for the real case this caused --
        # a ~3 m/s fabricated velocity from exactly this computation dead-
        # reckoned the search anchor further from the controller's actual
        # position on the very next vision-less frame). predict()'s own
        # dead-reckoning is gated on this (see its own docstring) --
        # integrating real accel/gyro data forward from a FABRICATED v=0 (or
        # equally fabricated long-gap difference) starting point produces a
        # position "prediction" with zero actual basis for what the
        # controller's true velocity is, not a genuine (if uncertain)
        # estimate -- there is nothing here for a subsequent vision candidate
        # to plausibly be measured against.
        self.velocity_established = False
        # >0 while self.R's own history, going back to the last
        # bootstrap/fail_open/_try_cold_reacquire seed, is still considered
        # an untrustworthy gate REFERENCE. coverage_fallback is
        # pose_search.py's own confidence=0.0 marker -- too few of the
        # geometrically-expected-visible LEDs were actually matched, e.g.
        # controller near the frame edge. Unlike position, rotation's
        # implausibility/Case-A-pushback checks have NO velocity_established-
        # style precondition (see try_update's own comment: "gyro integration
        # measures real angular rate directly from the first frame" -- true
        # in general, but that reasoning silently assumes the STARTING
        # orientation it integrates from is itself trustworthy). When the
        # seed orientation is instead a coverage-starved, likely-wrong P3P
        # solve, gyro-propagating it forward and then penalizing a good
        # vision candidate for disagreeing with that bad propagation is
        # exactly backwards -- confirmed on a real case (2026-09-13): a
        # 5-inlier coverage-fallback bootstrap's rotation was simply wrong,
        # and later good (13-inlier) vision candidates kept getting
        # hard-rejected against it.
        #
        # Decremented by one on every subsequent accept whose OWN candidate
        # is NOT itself coverage_fallback (see try_update's own blend-path
        # comment), reset back to rotation_seed_grace_frames on one that IS
        # -- an earlier one-shot version (clear unconditionally after exactly
        # one exempted frame) was found broken on a real reproduction: the
        # frame right after the bad bootstrap was ITSELF another
        # coverage-fallback candidate, so self.R never actually got
        # corrected and the exemption was wasted on it.
        #
        # SECOND real case (also 2026-09-13, after the fix above already
        # shipped): the frame ACCEPTED WHILE this counter was >0 has its OWN
        # rotation gate-exempted (that's the whole point), so ITS resulting
        # self.R is itself unconfirmed -- yet the old code cleared the
        # counter to 0 the instant that one non-fallback frame landed,
        # immediately arming the hard gate against that still-unvetted
        # rotation for the VERY NEXT candidate. Real trace: cold_reacquired
        # (coverage_fallback) -> next frame accepted at full exemption
        # (rot_innov=159deg, correct) -> gate re-armed same frame -> the
        # frame after THAT rejected as "implausible" (rot_innov=61deg) purely
        # because gyro had only one un-vetted, reacquisition-fresh frame's
        # worth of orientation to integrate from, during genuinely fast/
        # jerky real rotation (confirmed via mocap: vision was correct the
        # whole time). Fixed by starting the counter at
        # rotation_seed_grace_frames (default 2, config-tunable) instead of
        # 1 -- the accept consumed under exemption now leaves one MORE
        # grace frame before the gate re-arms, giving self.R one full real
        # accept to settle before it's trusted as a hard-gate reference.
        self._rotation_seed_grace_frames = 0
        self.last_update_ts_ns = None
        self.consecutive_rejects = 0   # count of consecutive implausible-jump rejects (see try_update)

        # Consecutive REAL FRAMES (deduped by timestamp, not elapsed time) since
        # the last genuinely trustworthy update -- bumped once per distinct
        # target_ts_ns seen by predict() (see there), reset to 0 on bootstrap/
        # fail_open/a real fused accept in try_update. Frame-COUNT based rather
        # than time-based deliberately: this recording's own cadence isn't
        # uniform (alternates ~11ms/~22ms), so "4 frames" and "4 frames' worth
        # of seconds" aren't the same threshold, and a user's own mental model
        # of "IMU shouldn't matter after a few LOST FRAMES" is frame-counted,
        # not time-based. Drives w_imu's decay in try_update -- see there.
        self.frames_since_update = 0
        self._last_predict_seen_ts_ns = None
        # (last_update_ts_ns, target_ts_ns, headset_used) stamped by predict() -- lets the continuous
        # rotation gate reuse predict()'s own headset/no-headset decision instead of repeating the
        # (0.83 ms) mocap lookups. headset_used: True / False (plain path) / None (predict() returned None).
        self._pred_stamp = None
        # one-entry memo of _rot_gate_info, keyed (last_update_ts_ns, frame_ts_ns)
        self._rot_gate_cache = None

        # A single buffered WEAK cold-reacquire candidate (dict with R/p/
        # confidence/coverage_fallback/frame_ts_ns/n_inliers), or None -- see
        # _try_cold_reacquire's own docstring. A "weak" candidate (few
        # inliers/coverage_fallback, e.g. controller at the edge of camera
        # view with only 4-5 of its LEDs visible) reaching _try_cold_reacquire
        # is no longer trusted immediately: it's held here for one frame so
        # the NEXT candidate can either confirm it (two independent weak
        # solves landing close together is real signal a strong single solve
        # doesn't need) or get superseded by a strong match. Direct fix for a
        # very common real failure shape (2026-09-13, user-reported, "64->65"
        # and "66->67"): a weak/coverage_fallback reacquisition off a barely-
        # visible edge-of-frame LED subset gets trusted on the spot, then the
        # very next frame's strong, correct match reads as a huge "jump" vs.
        # that bad anchor and eats a rejection/grace-window cost that a
        # cleaner anchor would never have needed.
        self._cold_pending: dict | None = None
        self._reject_streak_prev: dict | None = None   # see _check_reject_streak_override
        self._cand_rotation_from_prior = False   # set per try_update call, read by _try_cold_reacquire
        self._reject_streak_n = 0

        self.reported_R = None
        self.reported_p = None

        # One Euro filters for the REPORTED pose only -- see module docstring
        # for why this replaces the old w_smooth cost term. Position/rotation
        # get independent parameters since their natural units/scales differ
        # (meters vs. radians of tangent-space delta per second).
        self._pos_filter = OneEuroFilter(
            min_cutoff=float(self._hc_get("one_euro_pos_min_cutoff_hz", 1.0)),
            beta=float(self._hc_get("one_euro_pos_beta", 0.3)),
            d_cutoff=float(self._hc_get("one_euro_pos_d_cutoff_hz", 1.0)),
        )
        self._rot_filter = OneEuroRotationFilter(
            min_cutoff=float(self._hc_get("one_euro_rot_min_cutoff_hz", 1.0)),
            beta=float(self._hc_get("one_euro_rot_beta", 0.3)),
            d_cutoff=float(self._hc_get("one_euro_rot_d_cutoff_hz", 1.0)),
        )

        self._sibling_filters: list = []
        self._trust_window: deque = deque(maxlen=int(self._hc.get("trust_window_len", 8)))

        # Rolling history of recent normal-cadence FUSED accepts' own innovation
        # vs. p_pred -- feeds _agreement_weak_bounds' adaptive thresholds (see
        # try_update's Case A correction step). Appended only at normal cadence
        # (dt_s <= _GAP_DT_NORMAL_S) so a gap-inflated innovation can't pollute
        # the "typical single-frame jump size" baseline.
        self._pos_innov_hist: deque = deque(maxlen=int(self._hc.get("agreement_hist_len", 20)))
        self._rot_innov_hist: deque = deque(maxlen=int(self._hc.get("agreement_hist_len", 20)))

        self._last: dict = {}

    def set_siblings(self, siblings: list) -> None:
        self._sibling_filters = list(siblings)

    def _hc_get(self, key: str, default):
        """fusion_heuristic.per_controller.<ctrl_name>.<key> if present, else
        the shared fusion_heuristic.<key>, else `default`. Lets a controller
        with a structurally different typical match quality (e.g. one
        controller's own reprojection error running consistently higher
        across the whole recording, confirmed in this project's own
        data/vision_pose_log.csv) use its own weak/strong thresholds without
        duplicating every other knob."""
        per_ctrl = (self._hc.get("per_controller") or {}).get(self._ctrl_name, {})
        if key in per_ctrl:
            return per_ctrl[key]
        return self._hc.get(key, default)

    def _seed_rotation_grace(self, coverage_fallback: bool, high_risk: bool = False) -> None:
        """(Re)seeds self._rotation_seed_grace_frames after a fresh anchor
        (bootstrap/fail_open/_try_cold_reacquire) -- see that field's own
        __init__ comment for the full rationale. rotation_seed_grace_frames
        (default 2) is how many consecutive non-coverage_fallback accepts
        the hard gate stays exempted for after an untrustworthy seed, not
        just the one immediately following it.

        high_risk (2026-09-16): True when the seeding solution was itself
        winner_was_contested or swap_suspected -- a contested cross-
        controller conflict winner, or a candidate suspected of belonging
        to the OTHER controller. The ordinary coverage_fallback exemption's
        whole justification is "this seed is PROBABLY right, just under-
        evidenced -- don't let the hard gate false-reject the next good
        candidate on top of it." That doesn't hold when the seed itself
        already carries an independent "this might belong to the wrong
        controller entirely" flag -- granting the same blanket exemption
        there is exactly what let a real ~129deg identity-swap jump sail
        through the hard gate one frame later (found investigating a real
        case: static_dark, right_controller). high_risk always forces zero
        grace, overriding coverage_fallback -- the next candidate's
        rotation gets checked normally, no exemption."""
        if high_risk:
            self._rotation_seed_grace_frames = 0
        elif coverage_fallback:
            self._rotation_seed_grace_frames = int(self._hc_get("rotation_seed_grace_frames", 2))
        else:
            self._rotation_seed_grace_frames = 0

    def _peak_gyro_accel(self, frame_ts_ns: int) -> tuple:
        """(peak_gyro_dps, peak_dynamic_accel_mps2) over [last_update_ts_ns,
        frame_ts_ns] -- the SAME window predict() dead-reckons across (see
        slice_imu_to_window's own call in predict()). Added 2026-09-13:
        _implausible_jump_thresholds' ceiling used to scale ONLY with
        |self.v|, with no way to tell "real violent motion" apart from "a
        noisy/short-baseline velocity estimate" -- found investigating a
        real case (right_controller, walk_hard recording) where 3 genuinely
        good vision candidates (confirmed via mocap ground truth -- as
        accurate or MORE accurate than the accepted frames around them) got
        rejected as implausible during a real ~14g/~2900deg/s swing that a
        velocity-only budget was never going to cover, regardless of how
        self.v happened to be established.

        peak_dynamic_accel_mps2: raw |accel_body| magnitude minus |g_world|
        (this controller's own estimated gravity magnitude, ~9.81 -- falls
        back to a literal 9.81 if no g_world estimate exists yet) --
        DELIBERATELY a coarse proxy, not a properly R(t)-rotated
        gravity subtraction like integrate_accel_to_position's internal one
        (that would need re-running gyro integration redundantly with
        predict()'s own). By the reverse triangle inequality
        (||a|-|g|| <= |a-g|), this is always a LOWER bound on the true
        dynamic acceleration magnitude -- it can understate how violent the
        window was for some orientations, never overstate it, which is the
        safe direction for a term that's only meant to WIDEN a rejection
        gate. Mirrors controller.py's _log_jump_stats_row diagnostic
        (peak_accel_mps2 column) exactly, so the empirical fit that
        produced this method's own rate constants (see
        _implausible_jump_thresholds) and that diagnostic's real recorded
        values are directly comparable.

        Returns (0.0, 0.0) -- inert, not NaN -- when gyro/accel/prior state
        aren't available, matching predict()'s own fail-open contract:
        this term should default to contributing nothing, not blocking the
        velocity-only terms that still work fine without it.

        2026-09-14: thin wrapper over src.imu_data.peak_gyro_accel_over_window
        (factored out so ControllerTracker's sibling imu_only_propagation_max_s
        coasting budget can share the exact same signal -- see that
        function's own docstring)."""
        if self.last_update_ts_ns is None:
            return 0.0, 0.0
        g_world_mag = 9.81
        if self._g_world_estimator is not None and self._g_world_estimator.g_world is not None:
            g_world_mag = float(np.linalg.norm(self._g_world_estimator.g_world))
        return peak_gyro_accel_over_window(
            self._gyro_data, self._accel_data, self.last_update_ts_ns, frame_ts_ns, g_world_mag)

    def _effective_coast_budget_s(self, base_budget_s: float, frame_ts_ns: int,
                                   axis: str = "both", rate_prefix: str = "coast_trust") -> float:
        """base_budget_s shrunk by how violent real measured gyro/accel was
        over [last_update_ts_ns, frame_ts_ns] (see _peak_gyro_accel), down
        to a floor of coast_trust_min_budget_s -- never widened, only ever
        shrunk (unlike _implausible_jump_thresholds' accel/gyro term, which
        only ever widens a ceiling; here we're shrinking a TIME budget for
        how long a pure-IMU coast is still credible to show as "the
        model").

        Added 2026-09-14 (user-directed, "prove it with stats"): found
        investigating a real case (right_controller, walk_hard recording)
        where a flat cold_pending_report_max_gap_s=0.25s let a freshly
        recomputed IMU-only coast display over a meter from vision, well
        inside that flat budget -- the elapsed time was short, but the real
        motion during it was violent enough that the flat "short time =
        still credible" assumption didn't hold. calm floors keep this a
        no-op for ordinary motion (same shape as every other accel/gyro
        term in this file); *_shrink_s_per_* and the floor are fit from
        real dt_s/peak_gyro/peak_accel-vs-actual-prediction-error data
        across both available recordings (COAST_BUDGET_STATS_CSV, temporary
        env-gated diagnostic -- see this session's own project memory for
        the fit).

        axis (2026-09-15, user-directed, same real case revisited: frame 71,
        3-frame coast + a zero-quality candidate hitting try_update's
        degenerate w_sum<=0 fallback -- see that branch's own comment):
        "both" (default, unchanged behavior for the two original callers --
        _report_if_still_usable's display budget and ControllerTracker.
        _mark_all_lost's search-anchor budget) includes both shrink terms,
        same as before. "accel"/"gyro" include only that one term -- lets a
        caller ask two separate questions ("how long is p_pred's POSITION
        still credible" vs "how long is its ROTATION still credible")
        instead of one combined answer, mirroring how mocap ground truth on
        the frame-71 case showed the two axes were WRONG BY DIFFERENT
        AMOUNTS (406mm position error, only 4.24deg rotation error) -- a
        single combined budget can't express that a coast's rotation can
        still be trustworthy well after its position no longer is (gyro
        integration has no velocity precondition; position dead-reckoning
        does, see velocity_established's own docstring).

        rate_prefix (2026-09-15, user-directed, same real case's own
        follow-up: cold_pending_report_max_gap_s's flat 0.25s base was
        "crazy big" -- a 4-frame-lost/moderately-elevated-motion gap
        [right_controller, frame 39: 1363deg/s peak gyro, 38.6 m/s^2 peak
        accel -- "fast" motion, not extreme] still displayed a coast 291mm
        off mocap ground truth, well past the ~10cm bound this budget's
        own fit targeted). Lets _report_if_still_usable's DISPLAY-only
        budget use its OWN, separately-tunable shrink rate
        (cold_pending_shrink_s_per_mps2/_dps) instead of sharing
        coast_trust_shrink_s_per_mps2/_dps with the other two, higher-
        stakes consumers (the degenerate-fallback CORE-STATE gate in
        try_update, and ControllerTracker._mark_all_lost's own SEARCH-
        ANCHOR budget) -- gentling the display's own rate (so "fast" lands
        at a real ~35ms ceiling instead of snapping straight past it to
        the floor) must not also gentle those two already-validated,
        already-tighter budgets (see this session's own real case,
        right_controller frame 71, which the degenerate-fallback gate was
        specifically built and validated against). calm floors and
        min_budget_s stay SHARED across all three on purpose -- the target
        floor (0.025s, unchanged, user-directed: "min budget should stay
        at 25ms") is identical either way, and the calm floors (0.0 for
        both axes) don't change under a pure rate rescaling, so there was
        nothing case-specific left to duplicate for those two.

        min_budget_s is ALSO now rate_prefix-aware (2026-09-16, "big IMU
        investigation"): reads f"{rate_prefix}_min_budget_s" first, falling
        back to the shared coast_trust_min_budget_s when that key is unset
        -- lets a caller give ONE axis its own floor without touching the
        shared one every other caller still reads. Added because a broad,
        systematic sweep (real mocap ground truth vs blind gyro-only
        rotation dead-reckoning, ~217k anchor/dt samples across all 8
        available recordings, not just one hand-picked case) found the
        shared 0.035s floor was silently overriding an already-correctly-
        shrunk ROTATION budget back UP for any moderate-or-faster gyro
        burst: at ~1000deg/s the raw shrink formula already computes
        ~8ms of remaining budget, but max(min_budget_s, ...) was clamping
        that back to 35ms -- 3-4x longer than gyro dead-reckoning is
        actually still credible there (median rotation error already
        >=10deg by just 11ms once peak gyro clears ~350-550deg/s). See
        coast_trust_rot_min_budget_s/coast_trust_rot_shrink_s_per_dps in
        config.yml for the specific fit and the degenerate-fallback call
        site below for where this is actually wired in (axis="gyro" only
        -- the POSITION/accel axis's existing floor/rate already tracked
        the same sweep's accel-vs-position findings reasonably well and
        was left unchanged).

        Calm-motion extension (2026-09-16, same-day follow-up): axis="gyro"
        now ALSO reads f"{rate_prefix}_calm_extend_ceiling_s"/"_max_dps"
        (both default 0.0, fully inert unless configured) and delegates to
        the shared src.imu_data.effective_coast_budget_s -- see that
        function's own docstring for the full formula/derivation. Only
        axis="gyro" gets this: the shrink-only formula was already found to
        track the position/accel axis's own sweep data reasonably well, so
        no extend term was added there. axis="both" (the two original,
        pre-axis-split callers) is computed exactly as before -- a combined
        additive shrink with NO extend term, since the two per-axis shrinks
        don't distribute through max()/floor the same way a per-axis
        extend-then-sum would; kept as its own explicit branch rather than
        forcing it through the single-axis shared helper."""
        peak_gyro_dps, peak_accel_mps2 = self._peak_gyro_accel(frame_ts_ns)
        accel_calm_floor = float(self._hc_get("coast_trust_accel_calm_floor_mps2", 40.0))
        gyro_calm_floor = float(self._hc_get("coast_trust_gyro_calm_floor_dps", 900.0))
        per_accel = float(self._hc_get(f"{rate_prefix}_shrink_s_per_mps2", 0.0))
        per_gyro = float(self._hc_get(f"{rate_prefix}_shrink_s_per_dps", 0.0))
        min_budget_s = float(self._hc_get(f"{rate_prefix}_min_budget_s", self._hc_get("coast_trust_min_budget_s", 0.035)))
        if axis == "accel":
            return _shared_effective_coast_budget_s(base_budget_s, peak_accel_mps2, accel_calm_floor,
                                                      per_accel, min_budget_s)
        elif axis == "gyro":
            calm_extend_ceiling_s = float(self._hc_get(f"{rate_prefix}_calm_extend_ceiling_s", 0.0))
            calm_extend_max_dps = float(self._hc_get(f"{rate_prefix}_calm_extend_max_dps", 0.0))
            return _shared_effective_coast_budget_s(base_budget_s, peak_gyro_dps, gyro_calm_floor,
                                                      per_gyro, min_budget_s,
                                                      calm_extend_ceiling_s, calm_extend_max_dps)
        else:
            accel_shrink = per_accel * max(0.0, peak_accel_mps2 - accel_calm_floor)
            gyro_shrink = per_gyro * max(0.0, peak_gyro_dps - gyro_calm_floor)
            return max(min_budget_s, base_budget_s - accel_shrink - gyro_shrink)

    def _implausible_jump_thresholds(self, frame_ts_ns: int, quality: float = None) -> tuple:
        """(pos_thresh_m, rot_thresh_deg) for the hard implausibility gate --
        speed-scaled 2026-09-13, replacing a flat implausible_jump_pos_m
        (0.3m)/implausible_jump_rot_deg (60deg) that was "first-cut
        defaults, not yet validated" per its own old config comment. Ports
        the SHAPE of controller.py's already-validated, empirically-fit
        vs-predicted_pose formula (base + per_speed * speed) rather than
        reusing its exact per-axis numbers outright -- that fit was derived
        from per-camera PIXEL-space residuals, which don't obviously
        transfer to this world-frame FUSED-pose's own residual
        distribution, so this stays scalar (not per-axis) until/unless a
        real fit against fused-pose data justifies otherwise. See
        config.yml's implausible_jump_*_thresh_base/per_speed comment for
        the full numeric derivation.

        Speed input is |self.v|, this filter's OWN measured velocity.
        POSITION's own gate already requires velocity_established before it
        runs at all (see try_update's _pos_implausible), so self.v is
        always real whenever that result actually gets used. ROTATION's
        hard-gate has NO such precondition (gyro integration is valid from
        frame one -- see try_update's own comment on _rot_implausible) --
        to avoid REMOVING that existing coverage, speed_m_s falls back to
        0.0 (base-only, no speed term) whenever velocity_established is
        False, rather than gating this whole method on it.

        Accel/gyro-aware widening (2026-09-13, see _peak_gyro_accel's own
        docstring for the real case this closes): ADDITIVE on top of the
        velocity term, same base+rate shape, using peak real |accel|/|gyro|
        over this same call's window instead of self.v. *_calm_floor keeps
        this term at ~0 extra budget for ordinary motion (near the calmer
        recording's own typical range) -- it should only activate during a
        genuinely violent real event, not inflate the ceiling every frame.
        frame_ts_ns is this call's own frame timestamp (the window's other
        endpoint, alongside self.last_update_ts_ns).

        Stale-time widening (2026-09-13, second real case): self.v/p_pred's
        own dead-reckoning uses self.v as its ONE-STEP-STALE initial
        velocity -- at a genuine motion reversal (deceleration through
        zero, e.g. the turning point of a swing), that stale v0 makes
        predict() overshoot in the OLD direction, and every subsequent
        REJECTED frame compounds this: last_update_ts_ns doesn't advance on
        a reject, so the window predict() integrates over keeps growing
        while still starting from the SAME stale v0, and pos_innov_m grows
        with it (confirmed on a real 4-frame reject cascade: 123/272/373/
        576mm, tracking elapsed time since the last real accept almost
        exactly). Mirrors controller.py's own already-validated,
        consolidated "cold-start staleness" widening (max_plausible_hand_
        speed_m_s/_ang_speed_deg_s * elapsed seconds, matching: section) --
        reused directly rather than re-deriving a THIRD copy of this same
        "how far/fast could a hand plausibly have moved since we last knew
        for sure" concept (see that key's own config.yml comment on the
        consolidation this session already did once). stale_s is 0 right
        after a real accept (last_update_ts_ns == frame_ts_ns's own prior),
        growing only across consecutive rejects -- inert in the common
        (tracking normally) case.

        Quality-aware POSITION shrink (2026-09-17, real case: right_
        controller, walk_dark, frame 251 -- mocap-confirmed: a moderate-
        quality (n_inliers=7, error_px=0.36, quality~0.53) vision candidate
        landed 22mm off mocap ground truth while this filter's OWN p_pred
        was only ~3-5mm off, but cost_weight_imu=0.0 means the ordinary
        blend can't lean toward p_pred no matter how much better it is --
        the HARD gate above is the only remaining protection, and its flat
        base (61.5mm) has never scaled with how much vision's own quality
        signal (n_inliers/error_px, the SAME ramp _vision_weight already
        computes -- see _vision_quality) should be trusted. Fit (2026-09-17)
        against this session's own fresh sweep of real pos_innov_m (vision
        vs p_pred, the EXACT quantity this gate checks) binned by quality,
        warm-tracking-only frames from a full 5400-frame walk_dark run:
        p99 pos_innov_m falls from ~50-80mm at quality<0.6 down to ~20mm at
        quality>=0.7 (both controllers). Calibrated CONSERVATIVELY
        (user-directed): the shrink floor sits near each band's own p99,
        not p90 -- this reduces false-reject risk on legitimate candidates
        at the cost of NOT catching a single-frame case as mild as frame
        251's own 17.8mm pos_innov_m at quality~0.53 (that would need a
        p90-level cutoff, a materially higher false-reject risk this
        first cut deliberately avoids). Applied multiplicatively to the
        FULL pos_thresh_m computed above (including the speed/accel/stale
        widening) -- a genuinely fast-moving OR long-stale candidate still
        gets its full deserved allowance at high quality; only low-to-
        moderate quality narrows it. quality=None (callers that don't have
        a vision candidate to judge, e.g. none currently) skips this
        entirely -- byte-identical to before this fix."""
        speed_m_s = float(np.linalg.norm(self.v)) if self.velocity_established else 0.0
        pos_base_mm = float(self._hc_get("implausible_jump_pos_thresh_base_mm", 61.5))
        pos_per_speed_mm_s = float(self._hc_get("implausible_jump_pos_thresh_per_speed_mm_s", 33.0))
        rot_base_deg = float(self._hc_get("implausible_jump_rot_thresh_base_deg", 8.3))
        rot_per_speed_deg_s = float(self._hc_get("implausible_jump_rot_thresh_per_speed_deg_s", 0.86))
        pos_thresh_m = (pos_base_mm + pos_per_speed_mm_s * speed_m_s) / 1000.0
        rot_thresh_deg = rot_base_deg + rot_per_speed_deg_s * speed_m_s

        peak_gyro_dps, peak_accel_mps2 = self._peak_gyro_accel(frame_ts_ns)
        accel_calm_floor_mps2 = float(self._hc_get("implausible_jump_accel_calm_floor_mps2", 40.0))
        gyro_calm_floor_dps = float(self._hc_get("implausible_jump_gyro_calm_floor_dps", 900.0))
        pos_per_accel_mm_per_mps2 = float(self._hc_get("implausible_jump_pos_thresh_per_accel_mm_per_mps2", 0.0))
        rot_per_gyro_deg_per_dps = float(self._hc_get("implausible_jump_rot_thresh_per_gyro_deg_per_dps", 0.0))
        pos_thresh_m += pos_per_accel_mm_per_mps2 * max(0.0, peak_accel_mps2 - accel_calm_floor_mps2) / 1000.0
        rot_thresh_deg += rot_per_gyro_deg_per_dps * max(0.0, peak_gyro_dps - gyro_calm_floor_dps)

        stale_s = 0.0
        if self.last_update_ts_ns is not None:
            stale_s = max(0.0, (frame_ts_ns - self.last_update_ts_ns) / 1e9)
        _matching_cfg = self._cfg.get("matching", {}) or {}
        max_speed_m_s = float(_matching_cfg.get("max_plausible_hand_speed_m_s", 7.5))
        max_ang_speed_deg_s = float(_matching_cfg.get("max_plausible_hand_ang_speed_deg_s", 2200.0))
        pos_thresh_m += max_speed_m_s * stale_s
        rot_thresh_deg += max_ang_speed_deg_s * stale_s
        if self._rot_gate_continuous():
            # continuous mode: the rotation ceiling is the physically-grounded gate T (see
            # rot_gate_threshold_deg) -- flat for headset-corrected prediction, NOT widening with
            # 2200 deg/s * stale. Position part above is untouched.
            rot_thresh_deg = self._rot_gate_info(frame_ts_ns)["T"]

        if quality is not None and bool(self._hc_get("implausible_jump_pos_quality_shrink_enabled", True)):
            full_trust_at = float(self._hc_get("implausible_jump_pos_quality_full_trust_at", 0.7))
            min_scale = float(self._hc_get("implausible_jump_pos_quality_min_scale", 0.6))
            if full_trust_at > 0.0:
                ramp = float(np.clip(quality / full_trust_at, 0.0, 1.0))
                pos_thresh_m *= min_scale + (1.0 - min_scale) * ramp

        return pos_thresh_m, rot_thresh_deg

    def _report(self, frame_ts_ns: int, R_out: np.ndarray, p_out: np.ndarray) -> None:
        """Sets self.reported_R/self.reported_p by pushing (R_out, p_out)
        through this controller's One Euro filters (see module docstring) --
        called from EVERY try_update() branch (bootstrap, fail_open,
        implausible_reject, fused) so the reported/displayed pose is
        uniformly jitter-filtered no matter which branch produced it, and so
        the filters' own internal derivative-estimate state stays continuous
        across branches rather than skipping frames arbitrarily. Deliberately
        separate from (self.R, self.p) -- the tracking state predict() dead-
        reckons off next frame stays unfiltered/immediate, so smoothing adds
        lag only to what's displayed, never to the tracking math itself.

        fusion_heuristic.one_euro_enabled: false bypasses both filters
        entirely -- reported_R/reported_p become exactly (R_out, p_out), the
        same value the tracking-state math already computed, with no added
        smoothing lag. For isolating vision/IMU investigation from smoothing
        behavior (same spirit as vision_only_debug): the filters' own
        internal state is left untouched while bypassed (not reset), so
        turning this back to true resumes them rather than losing history --
        expect one glitched dt on the very first frame after re-enabling if
        real time elapsed while it was off, exactly like re-enabling after
        any other pause would."""
        if not bool(self._hc_get("one_euro_enabled", True)):
            self.reported_p, self.reported_R = p_out, R_out
            return
        t_s = frame_ts_ns / 1e9
        self.reported_p = self._pos_filter(t_s, p_out)
        self.reported_R = self._rot_filter(t_s, R_out)

    # ------------------------------------------------------------------
    # Dead-reckoning (unchanged from PoseFusionFilter -- reuses the same
    # validated src.imu_data primitives)
    # ------------------------------------------------------------------
    def note_real_frame(self, frame_ts_ns: int) -> None:
        """Bump frames_since_update once per distinct real frame (deduped by
        timestamp, not call count) -- the count of consecutive REAL frames this
        filter's last accepted state has gone uncorrected, which imu_frame_scale
        (see try_update) decays over imu_decay_frames of.

        Called from TWO places that must both keep this accurate: predict()
        itself below (covers try_update's own gating call, and _mark_all_lost's
        IMU-only search-anchor coast WHILE that coast is still within its own
        time budget, imu_only_propagation_max_s), and directly from
        ControllerTracker._mark_all_lost on every real lost frame regardless of
        that budget. The second call site matters: _mark_all_lost stops calling
        predict() at all once the IMU-only coast's time budget is exhausted (a
        deliberate cost cap -- an ever-growing dead-reckoning window on every
        lost frame of a long loss would be real, unbounded extra work) -- if
        THIS bump lived only inside predict(), frames_since_update would freeze
        at whatever it was at that point, for the REST of an arbitrarily long
        real loss. That silently kept imu_frame_scale > 0 (and the hard
        implausibility gate armed at its full, undecayed threshold) on a false
        rejection found 2026-09-09: a clean 6-inlier/0.07px vision reacquisition
        candidate, rejected as "implausible" (rot_innov=157.77deg) against a
        p_pred that was actually ~400ms+ stale -- frames_since_update had only
        reached ~2-3 by then, nowhere near imu_decay_frames=4, purely because
        _mark_all_lost had stopped bumping it once its own ~66ms coast budget
        ran out. Exactly the class of bug this project's own imu_decay_frames
        config comment already reasons about (frame-counted, not time-based, by
        deliberate design) -- but that reasoning assumed frames_since_update
        keeps advancing for the FULL duration of any loss, an invariant
        _mark_all_lost's own cost-capped coast silently violated once a loss
        outlasted its budget."""
        if frame_ts_ns != self._last_predict_seen_ts_ns:
            self.frames_since_update += 1
            self._last_predict_seen_ts_ns = frame_ts_ns

    def predict(self, target_ts_ns: int):
        """Dead-reckon the last ACCEPTED state forward to target_ts_ns.

        Returns (R_pred, p_pred) or None if prediction isn't possible yet (no
        prior state, no IMU coverage, g_world not converged). Still computes
        a real (gyro-integrated) p_pred even when self.velocity_established
        is False -- callers that gate specifically on POSITION plausibility
        (try_update's hard implausibility gate) check velocity_established
        themselves and skip only that criterion; R_pred stays usable as a
        rotation-only sanity check regardless (gyro integration has no
        velocity precondition -- it's a direct measurement of angular rate,
        not something extrapolated from an assumed linear velocity). See
        velocity_established's own comment in __init__ for why POSITION
        specifically needs this distinction: dead-reckoning position from a
        fabricated v=0 has no real basis, but integrating rotation from real
        gyro samples does, from the very first frame."""
        if self.R is None or self.last_update_ts_ns is None:
            return None
        self.note_real_frame(target_ts_ns)
        if self._gyro_data is None or self._accel_data is None or self._lever_arm is None:
            return None

        t_gyro, gyro_body = slice_imu_to_window(*self._gyro_data, self.last_update_ts_ns, target_ts_ns)
        t_accel, accel_body = slice_imu_to_window(*self._accel_data, self.last_update_ts_ns, target_ts_ns)

        # Mocap-corrected path uses MOCAP_ROOM_G_WORLD (a fixed, pre-validated
        # constant -- see _predict_with_headset_correction), NOT
        # self._g_world_estimator, so it must be tried BEFORE the rig-frame
        # g_world convergence gate below -- otherwise an early-run frame with
        # full headset mocap coverage but a not-yet-converged rig-frame
        # estimator (needs g_world_min_samples low-motion samples) would
        # fail-open even though a mocap-corrected prediction was available.
        _ts0 = self.last_update_ts_ns
        if self._headset_possible():
            corrected = self._predict_with_headset_correction(t_gyro, gyro_body, t_accel, accel_body,
                                                                target_ts_ns)
            if corrected is not None:
                self._pred_stamp = (_ts0, target_ts_ns, True)
                return corrected
            # else: real mocap coverage gap -- fall through to the plain
            # (uncorrected) path below, same fail-open contract as every
            # other missing precondition in this class.

        if self._g_world_estimator is None:
            self._pred_stamp = (_ts0, target_ts_ns, None)
            return None
        g_world = self._g_world_estimator.g_world
        if g_world is None:
            self._pred_stamp = (_ts0, target_ts_ns, None)
            return None

        predicted = predict_world_pose(t_gyro, gyro_body, t_accel, accel_body, g_world,
                                        self._lever_arm, self.last_update_ts_ns, target_ts_ns,
                                        self.R, self.p, self.v)
        if predicted is None:
            self._pred_stamp = (_ts0, target_ts_ns, None)
            return None
        self._pred_stamp = (_ts0, target_ts_ns, False)
        return predicted  # (R_pred, p_pred)

    def _predict_with_headset_correction(self, t_gyro, gyro_body, t_accel, accel_body, target_ts_ns: int):
        """Headset-ego-motion-corrected counterpart to the plain predict_world_pose call in
        predict() -- see src.imu_data.predict_headset_relative_pose for the physics. Returns
        (R_pred, p_pred), or None if ANY required ingredient is unavailable (real mocap coverage
        gap at either endpoint) -- predict() falls back to the uncorrected path on None, never
        raises. Absolute-frame gravity is MOCAP_ROOM_G_WORLD, a validated constant (see
        src/imu_data.py), not a live/converging estimate -- self._g_world_estimator_abs is no
        longer read here (see its own commented-out feed site in src/controller.py)."""
        ts0, ts1 = self.last_update_ts_ns, target_ts_ns
        inputs = self._headset_inputs(ts0, ts1)
        g_world_abs = MOCAP_ROOM_G_WORLD
        if inputs is None:
            return None
        T_wh0, T_wh1, omega_h0, v_wh0 = inputs
        return predict_headset_relative_pose(
            t_gyro, gyro_body, t_accel, accel_body, g_world_abs, self._lever_arm,
            ts0, ts1, self.R, self.p, self.v,
            T_wh0.R, T_wh0.t, omega_h0, v_wh0, T_wh1.R, T_wh1.t,
        )

    def _headset_possible(self) -> bool:
        """Headset ego-motion correction can be attempted at all: a headset mocap trajectory is loaded and
        fusion_heuristic.disable_headset_ego_motion (dev switch, default false: force the no-headset
        prediction path while mocap stays loaded, e.g. to validate the no-headset gate constants on
        recordings that do have headset mocap) is not set."""
        return (self._headset_mocap is not None
                and not bool(self._hc_get("disable_headset_ego_motion", False)))

    def _headset_inputs(self, ts0, ts1):
        """(T_wh0, T_wh1, omega_h0, v_wh0) headset-ego-motion inputs for the window [ts0, ts1], or None
        when headset correction is not possible (no headset mocap / disabled / ts0 unknown) or ANY of the
        four mocap lookups has a coverage gap (DeviceMocap.pose_at returns None when the two bracketing
        samples are > max_interp_gap_ms apart -- it never interpolates across a gap).

        All four lookups are always evaluated, in this fixed order, before the None check (tests patch
        world_pose with an ordered side_effect list and count calls) -- this is the single source of the
        lookup set predict() and every rule that needs to know "headset active?" share."""
        if not self._headset_possible() or ts0 is None:
            return None
        T_wh0 = world_pose(self._headset_mocap, ts0)
        T_wh1 = world_pose(self._headset_mocap, ts1)
        omega_h0 = headset_angular_velocity(self._headset_mocap, ts0)
        v_wh0 = headset_linear_velocity(self._headset_mocap, ts0)
        if T_wh0 is None or T_wh1 is None or omega_h0 is None or v_wh0 is None:
            return None
        return T_wh0, T_wh1, omega_h0, v_wh0

    # ------------------------------------------------------------------
    # Continuous rotation gate (fusion_heuristic.rot_gate_mode: continuous) -- see rot_gate_threshold_deg
    # ------------------------------------------------------------------
    def rot_coast_budget_s(self, frame_ts_ns: int, legacy_budget_s: float) -> float:
        """ROTATION-only coast/anchor budget (fusion_heuristic.coast_rot_budget_mode). legacy -> returns
        legacy_budget_s unchanged. headset -> when the headset ego-motion correction is available for
        [last_update_ts_ns, frame_ts_ns] returns max(legacy_budget_s, headset_rot_coast_budget_s(peak gyro)),
        else legacy_budget_s (no headset data => the prediction ignores head rotation => keep the steep
        legacy shrink). Position budgets never go through this."""
        if str(self._hc_get("coast_rot_budget_mode", "legacy")).lower() != "headset":
            return legacy_budget_s
        ts0 = self.last_update_ts_ns
        if ts0 is None or self._headset_inputs(ts0, frame_ts_ns) is None:
            return legacy_budget_s
        peak_gyro_dps, _ = self._peak_gyro_accel(frame_ts_ns)
        return max(legacy_budget_s, headset_rot_coast_budget_s(
            peak_gyro_dps, float(self._hc_get("coast_rot_budget_hs_full_s", 0.30)),
            float(self._hc_get("coast_rot_budget_hs_full_until_dps", 1000.0)),
            float(self._hc_get("coast_rot_budget_hs_floor_s", 0.035)),
            float(self._hc_get("coast_rot_budget_hs_floor_at_dps", 1500.0))))

    def _rot_gate_continuous(self) -> bool:
        """rot_gate_mode: legacy | continuous | auto. auto = continuous whenever headset ego-motion data
        is available to this filter (headset mocap loaded and not disabled), else legacy -- the
        no-headset case is not validated yet (see config.yml future-work note)."""
        mode = str(self._hc_get("rot_gate_mode", "legacy")).lower()
        if mode == "auto":
            return self._headset_possible()
        return mode == "continuous"

    def _rot_gate_params(self) -> dict:
        return dict(base_deg=float(self._hc_get("rot_gate_base_deg", 40.0)),
                    sat_allow_deg=float(self._hc_get("rot_gate_sat_allow_deg", 25.0)),
                    nohs_allow_dps=float(self._hc_get("rot_gate_nohs_allow_dps", 500.0)),
                    cap_hs_deg=float(self._hc_get("rot_gate_cap_hs_deg", 75.0)),
                    cap_nohs_deg=float(self._hc_get("rot_gate_cap_nohs_deg", 150.0)),
                    max_state_age_s=float(self._hc_get("rot_gate_max_state_age_s", 3.0)))

    def _rot_gate_saturated(self, ts0, ts1):
        """(saturated, peak_dps): gyro clip within the padded window, OR peak gyro >= rot_gate_sat_peak_dps
        (a clipped window can read as low as ~1250 dps because of the mix, so both are used)."""
        ceil_dps = self._hc_get("rot_gate_clip_ceiling_dps", None)
        ceil = None if ceil_dps is None else np.radians(np.asarray(ceil_dps, dtype=float))
        clipped, peak_dps = gyro_window_clip_flags(
            self._gyro_data, ts0, ts1, ceil_rad_s=ceil, frac=float(self._hc_get("rot_gate_clip_frac", 0.95)),
            pad_ns=int(float(self._hc_get("rot_gate_pad_s", 0.010)) * 1e9))
        return (clipped or peak_dps >= float(self._hc_get("rot_gate_sat_peak_dps", 1800.0))), peak_dps

    def _rot_gate_info(self, frame_ts_ns: int) -> dict:
        """{'T', 'dt_s', 'headset', 'saturated', 'peak_dps'} for a candidate at frame_ts_ns against the
        state last committed at last_update_ts_ns. Mode (headset vs not) is read from predict()'s own
        stamp when it matches this window, else re-derived from the lookups; memoised per (ts0, ts1)."""
        ts0 = self.last_update_ts_ns
        key = (ts0, frame_ts_ns)
        c = self._rot_gate_cache
        if c is not None and c[0] == key:
            return c[1]
        dt_s = 0.0 if ts0 is None else max(0.0, (frame_ts_ns - ts0) / 1e9)
        st = self._pred_stamp
        if st is not None and st[0] == ts0 and st[1] == frame_ts_ns:
            hs = st[2] is True
        else:
            hs = self._headset_inputs(ts0, frame_ts_ns) is not None
        sat, peak_dps = self._rot_gate_saturated(ts0, frame_ts_ns) if ts0 is not None else (False, 0.0)
        info = dict(T=rot_gate_threshold_deg(dt_s, hs, sat, **self._rot_gate_params()), dt_s=dt_s,
                    headset=hs, saturated=sat, peak_dps=peak_dps)
        self._rot_gate_cache = (key, info)
        return info

    def _gyro_only_rot_veto(self, R_meas, frame_ts_ns: int):
        """Continuous-gate check for the frames where predict() returned None (rig-frame g_world not
        converged, or accel/lever missing) and try_update would otherwise FAIL OPEN and accept every
        candidate unchecked -- a rotation gate needs only gyro. Returns (rot_innov_deg, T, info) if the
        candidate contradicts the gyro-only (no headset correction) rotation prediction R_state @ R_gyro by
        more than the no-headset threshold, else None (also None when the gate cannot be evaluated: no
        gyro coverage, rotation seed untrustworthy, state too stale)."""
        if (self._gyro_data is None or self.R is None or self.last_update_ts_ns is None
                or self._rotation_seed_grace_frames > 0):
            return None
        ts0 = self.last_update_ts_ns
        t_gyro, gyro_body = slice_imu_to_window(*self._gyro_data, ts0, frame_ts_ns)
        R_rel = integrate_gyro_segment(t_gyro, gyro_body, ts0, frame_ts_ns)
        if R_rel is None:
            return None
        R_pred = self.R @ R_rel
        innov = float(np.degrees(np.linalg.norm(Rotation.from_matrix(R_pred.T @ R_meas).as_rotvec())))
        dt_s = max(0.0, (frame_ts_ns - ts0) / 1e9)
        sat, peak_dps = self._rot_gate_saturated(ts0, frame_ts_ns)
        T = rot_gate_threshold_deg(dt_s, False, sat, **self._rot_gate_params())
        info = dict(T=T, dt_s=dt_s, headset=False, saturated=sat, peak_dps=peak_dps)
        return (innov, T, info) if innov > T else None

    def predict_dense(self, target_ts_ns: int, sample_every_n: int = 1):
        """Dense dead-reckoned trajectory from last_update_ts_ns to target_ts_ns,
        for debug-visualization only -- see src.imu_data.dead_reckon_dense.

        Same self.velocity_established gate as predict() (see its own
        docstring) -- a dense path dead-reckoned from a fabricated v=0 is
        exactly as unfounded here as a single-point prediction would be, even
        though this is debug-visualization-only: showing it would still
        misrepresent a real gap as "IMU knows where the controller went"."""
        if self.R is None or self.last_update_ts_ns is None:
            return None
        if not self.velocity_established:
            return None
        if self._gyro_data is None or self._accel_data is None or self._lever_arm is None:
            return None
        if self._g_world_estimator is None:
            return None
        g_world = self._g_world_estimator.g_world
        if g_world is None:
            return None
        t_gyro, gyro_body = slice_imu_to_window(*self._gyro_data, self.last_update_ts_ns, target_ts_ns)
        t_accel, accel_body = slice_imu_to_window(*self._accel_data, self.last_update_ts_ns, target_ts_ns)
        ts, p, R_list = dead_reckon_dense(t_gyro, gyro_body, t_accel, accel_body, g_world,
                                           self._lever_arm, self.last_update_ts_ns, target_ts_ns,
                                           self.R, self.p, self.v, sample_every_n=sample_every_n)
        if len(ts) == 0:
            return None
        return ts, p, R_list

    # ------------------------------------------------------------------
    # Trust reporting (display/debug only -- never gates this filter's own
    # accepts, since this filter always accepts)
    # ------------------------------------------------------------------
    def trust_score(self) -> float:
        if not self._trust_window:
            return 0.0
        return float(np.mean(self._trust_window))

    def trust(self, frame_ts_ns: int):
        if self.last_update_ts_ns is None:
            return None
        dt_s = max((frame_ts_ns - self.last_update_ts_ns) / 1e9, 0.0)
        tau = float(self._cfg.get("trust_gap_tau_s", 0.45))
        return self.trust_score() * float(np.exp(-dt_s / tau))

    # ------------------------------------------------------------------
    # Debug diagnostics -- same _LAST_FIELDS/_set_last/debug_snapshot shape as
    # PoseFusionFilter so visualization.py needs no changes (reads via .get(),
    # already None-tolerant).
    # ------------------------------------------------------------------
    _LAST_FIELDS = ("outcome", "pos_innov_m", "rot_innov_deg", "confidence",
                     "alpha_pos", "alpha_rot", "kalman_gain_pos", "kalman_gain_rot",
                     "pos_pred", "R_pred", "agreement")

    def _set_last(self, **kwargs) -> None:
        self._last = {k: kwargs.get(k) for k in self._LAST_FIELDS}

    def debug_snapshot(self, frame_ts_ns: int) -> dict:
        return {
            **self._last,
            "has_state": self.R is not None,
            "consecutive_rejects": self.consecutive_rejects,
            "trust_score": self.trust_score(),
            "trust": self.trust(frame_ts_ns),
            "pos_sigma_m": None, "rot_sigma_deg": None, "vel_sigma_m_s": None,  # no covariance in this filter
            "gate": None, "d2": None,   # no gate in this filter
        }

    # ------------------------------------------------------------------
    # Quality-dependent vision weight
    # ------------------------------------------------------------------
    def _vision_weight(self, n_inliers: int, error_px: float, pos_innov_m: float, rot_innov_deg: float,
                        frame_ts_ns: int, gate_active: bool = True, rot_gate_active: bool = True):
        """cost_weight_vision (the base/max value) scaled down for a weak
        match -- few inlier pairs and/or high mean reprojection error -- up
        to its full base value for a strong one.

        The two factors AVERAGE rather than multiply: a strict product was
        tried first and found to zero the entire vision weight whenever
        EITHER factor alone saturated to 0, even with the other at a
        perfect 1.0 -- confirmed on this project's own recording two
        different ways (right_controller's n_inliers running structurally
        lower over a several-frame window despite a near-perfect
        error_px, AND left_controller hitting an occasional error_px
        outlier despite plenty of inliers). Averaging means one strong
        axis always keeps SOME vision trust alive instead of an all-or-
        nothing AND; only a match that's bad on BOTH axes still drops
        near zero.

        weak/strong thresholds are per-controller-overridable (see
        _hc_get) because this recording's own error_px distribution
        (frames 0-1000, data/vision_pose_log.csv) differs meaningfully
        between controllers (left p50=0.25/p90=0.39, right p50=0.49/
        p90=1.04) even though n_inliers does not (both ~p50=15/p90=19) --
        see config.yml's fusion_heuristic.per_controller block.

        HARD implausibility gate on top of the two continuous ramps above:
        a position or rotation jump far beyond anything real hand motion
        produces is either an identity swap (this controller's own search
        matched a SIBLING controller's blobs) or some other geometrically-
        degenerate mismatch -- and neither the inlier nor the error ramp
        catches it, because a low-point-count fit can land a deceptively
        TINY reprojection error against entirely the wrong geometry (the
        same "clean-but-wrong" trap the inlier/error ramps already guard
        against individually, just not in combination). Confirmed on this
        recording: right_controller's vision solve landed almost exactly on
        left_controller's own tracked position -- n_inliers=6, error=0.13px
        (error_factor=1.0!), pos_innov=998mm, rot_innov=125.5deg -- and the
        averaged ramp alone still gave it vision share=0.40, corrupting the
        fused anchor by ~40% of a full 1m jump in one frame. When either
        threshold fires, vision weight is zeroed outright regardless of how
        good the other two factors look.

        NOTE: try_update now checks this exact same pair of thresholds itself,
        BEFORE calling this method, and rejects outright rather than reaching
        here (see its own docstring) -- so in practice this branch is no
        longer reachable via that call site. Left in place as a defensive
        backstop for any other caller of this method.

        gate_active=False skips this distance check entirely (still runs the
        two quality ramps below) -- used once try_update's own IMU frame-count
        decay has fully decayed p_pred's trust to 0: at that point p_pred is
        stale-by-construction and must not get to veto vision (whether via the
        top-level gate OR this one) purely by being far from it -- vision
        should be judged on its own match quality alone. Without this, a
        candidate could pass the top-level gate's skip but still get zeroed
        right back here by the exact same distance check, silently freezing
        the reported pose (accepted=True, so should_force_cold_start's
        elapsed-time clock never even starts) instead of actually re-anchoring.

        POSITION criterion here ALSO requires self.velocity_established, same
        as try_update's own top-level gate this docstring says duplicates
        (see that gate's own comment) -- found reachable in practice despite
        the "no longer reachable" note above: once velocity_established made
        the top-level gate skip its position criterion, THIS backup check
        still fired on the exact same pos_innov_m (it never inherited that
        exception), zeroing w_vision outright and leaving the state pinned to
        p_pred -- silently reintroducing the same bug from a second angle.

        rot_gate_active is the same exception for ROTATION, mirroring
        try_update's own _rot_seed_untrustworthy exemption (see that field's
        own comment): the exact same reachable-backstop bug repeated a THIRD
        time -- once try_update's top-level gate started skipping its
        rotation criterion right after a coverage-fallback seed, this backup
        check still fired on the same rot_innov_deg, zeroing w_vision outright
        (confirmed on a real case, 2026-09-13: the top-level gate correctly
        let a good post-bootstrap candidate through, but this method still
        zeroed inlier_factor/error_factor to 0.0 on the exact same
        rot_innov_deg=61.4deg, pinning the reported pose to the IMU
        prediction anyway).

        NOTE: an earlier version also added a force_full_trust mode here
        (skip the quality ramps too, mirroring _try_cold_reacquire's "vision
        is trusted directly" policy) for the same _rot_seed_untrustworthy
        regime, applied to BOTH position and rotation. Reverted (2026-09-13):
        found actively worse on the real reproduction -- a coverage-fallback
        candidate's own POSITION can be nearly as unreliable as its
        rotation, and fully trusting it moved self.p to that noisy value AND
        set velocity_established=True from the resulting noisy step, which
        then hard-rejected several genuinely good subsequent candidates as
        implausible. See try_update's own comment at its _vision_weight call
        site for the full account. Rotation-only stays because rotation
        specifically (not position) is what's structurally unstable for a
        low/degenerate point count."""
        quality, inlier_factor, error_factor = self._vision_quality(n_inliers, error_px)
        _pos_ceil_m, _rot_ceil_deg = self._implausible_jump_thresholds(frame_ts_ns, quality=quality)
        if (gate_active and self.velocity_established
                and pos_innov_m > _pos_ceil_m):
            return 0.0, 0.0, 0.0
        if (rot_gate_active
                and rot_innov_deg > _rot_ceil_deg):
            return 0.0, 0.0, 0.0

        base = float(self._hc.get("cost_weight_vision", 2.0))
        return base * 0.5 * (inlier_factor + error_factor), inlier_factor, error_factor

    def _vision_quality(self, n_inliers: int, error_px: float) -> tuple:
        """(quality, inlier_factor, error_factor) -- the SAME inlier/error
        ramps _vision_weight blends into its own w_vision, factored out here
        (2026-09-17) so `_implausible_jump_thresholds`' quality-aware ceiling
        shrink (see that method's own docstring) can use the identical
        signal instead of a second, driftable copy. quality is their simple
        average, same as try_update's own `quality = (inlier_factor +
        error_factor) / 2.0`."""
        inlier_factor = _ramp_up(
            float(n_inliers),
            float(self._hc_get("vision_weight_weak_inliers", 6)),
            float(self._hc_get("vision_weight_strong_inliers", 18)),
        )
        error_factor = _ramp_down(
            float(error_px),
            float(self._hc_get("vision_weight_weak_error_px", 0.5)),
            float(self._hc_get("vision_weight_strong_error_px", 0.15)),
        )
        return (inlier_factor + error_factor) / 2.0, inlier_factor, error_factor

    def _gap_vision_scale(self, dt_s: float, quality: float) -> float:
        """Multiplier applied to w_vision when this update follows an
        unusually long gap since the last one (a coast/reject streak, not a
        normal frame-to-frame step). Used to have a smooth_scale counterpart
        that boosted w_smooth over the same ramp -- removed along with the
        w_smooth cost term itself (see module docstring); w_imu already has
        its own frame-count decay to represent "how stale is this coast," so
        nothing needs to fill the gap this used to fill on the other side.

        Motivation: _vision_weight's inlier/error ramps judge a candidate purely
        on its OWN match quality, with no notion of how long the IMU side has
        been dead-reckoning unconstrained. A weak-but-clean-looking match (low
        n_inliers, tiny reprojection error -- the same "clean-but-wrong" shape
        _vision_weight's own docstring warns about) landing right after a long
        gap can still earn a large w_vision and get blended in at close to full
        strength against an IMU term that has itself had the least-constrained
        stretch to drift -- confirmed on this recording (frame with dt=66.6ms
        after two rejected frames: n_inliers=6, error=0.05px, vision_share=0.40,
        producing a ~115mm single-step jump vs. the ~3-7mm/frame typical of
        stable tracking). Scaling vision down specifically when dt_s is large
        damps exactly that case, while leaving ordinary frame-to-frame updates
        (the overwhelming majority) untouched.

        `quality` (0..1, the same (inlier_factor+error_factor)/2 _vision_weight
        computes) makes this QUALITY-AWARE: the time-based fraction below is
        scaled by (1-quality), so a genuinely STRONG post-gap match gets little
        or no penalty regardless of how long the gap was, while a weak one gets
        the full time-based penalty. Added after a second real case on a
        different recording: an 8-frame-lost reacquisition (dt=222ms, fully
        past gap_dt_full_s) with a SOLID match (n_inliers=12, error=0.08px) got
        vision_share knocked down to 0.17 anyway (pre-quality-aware behavior),
        producing a ~197mm fused-vs-vision divergence -- and that wrong fused
        anchor then rejected the next several genuinely-correct vision frames
        as "implausible" until the persistent-reject escape hatch had to force
        a cold-start. The original (weak-match) case this mechanism was built
        for still gets damped: a quality-aware gate turns off the SAME thing
        that made it wrong here.

        Linear ramp (before the quality scaling above) from 1.0 at
        dt_s <= gap_dt_normal_s to gap_vision_weight_floor at
        dt_s >= gap_dt_full_s. First-cut defaults, like every other threshold
        in this module -- tune during the verification pass."""
        dt_normal = float(self._hc.get("gap_dt_normal_s", _GAP_DT_NORMAL_S))
        dt_full   = float(self._hc.get("gap_dt_full_s", 0.15))
        if dt_full <= dt_normal:
            frac = 1.0 if dt_s >= dt_full else 0.0
        else:
            frac = float(np.clip((dt_s - dt_normal) / (dt_full - dt_normal), 0.0, 1.0))
        frac *= (1.0 - float(np.clip(quality, 0.0, 1.0)))
        vision_floor = float(self._hc.get("gap_vision_weight_floor", 0.3))
        return 1.0 - frac * (1.0 - vision_floor)

    # ------------------------------------------------------------------
    # Soft agreement ramp -- warm-state pushback vs. p_pred (see try_update's
    # Case A correction step below). Cold-state reacquisition used to be a
    # second consumer of this (a 3-frame confirmation window checking a
    # velocity-extrapolated prediction) -- removed, see _try_cold_reacquire's
    # own docstring; vision is trusted directly there now, no agreement ramp
    # involved.
    # ------------------------------------------------------------------
    def _agreement_factor(self, pos_innov_m: float, rot_innov_deg: float,
                           pos_weak: float, pos_strong: float,
                           rot_weak: float, rot_strong: float) -> float:
        """1.0 when both innovations are at/below their `weak` bound (candidate
        fully agrees with the reference), 0.0 once EITHER reaches its `strong`
        bound, linear in between. min() (not average) on purpose -- same
        reasoning as the pre-existing hard implausibility gate this ramp
        extends: a badly-disagreeing rotation shouldn't be maskable by a
        good position match or vice versa. Callers pass the existing hard-gate
        constants (implausible_jump_pos_m/_rot_deg) as pos_strong/rot_strong,
        so this ramp hands off to the untouched hard reject exactly where
        today's cliff already sits -- an extreme jump still gets agreement=0.0
        and is rejected/discarded exactly as before, this only adds pushback
        for what used to be "under the cliff, so fully trusted".

        NOTE on argument order: _ramp_down itself (module-level, shared with
        _vision_weight) expects (value, weak, strong) with weak > strong --
        weak is the HIGH/bad end (ramp -> 0), strong is the LOW/good end
        (ramp -> 1), matching _vision_weight's own error_px convention
        (vision_weight_weak_error_px=0.5 is the bad/high one). This method's
        OWN pos_weak/rot_weak/pos_strong/rot_strong use the opposite sense on
        purpose, matching how every caller below actually thinks about them:
        weak = the small "still fully trusted" bound, strong = the large
        "zero trust" ceiling -- so the calls below swap the order when
        forwarding to _ramp_down.

        Returns (agreement_pos, agreement_rot) SEPARATELY (not their min) --
        the caller applies agreement_pos only to the position pushback and
        agreement_rot only to the rotation pushback, same reasoning as the
        hard implausibility gate's own position/rotation split (see
        try_update's own comment on that): a position pushback toward p_pred
        is exactly as unfounded as the hard gate's position criterion when
        self.velocity_established is False (p_pred's position component is
        dead-reckoned from a fabricated v=0), while rotation's pushback
        toward R_pred stays valid regardless (gyro integration, no velocity
        precondition). The caller is responsible for forcing agreement_pos to
        1.0 when velocity isn't established -- this method only computes the
        innovation-based ramps themselves."""
        return (_ramp_down(pos_innov_m, pos_strong, pos_weak),
                _ramp_down(rot_innov_deg, rot_strong, rot_weak))

    def _agreement_weak_bounds(self, frame_ts_ns: int):
        """(pos_weak_m, rot_weak_deg) for _agreement_factor -- adaptive to this
        controller's OWN recent typical single-frame innovation (see
        _pos_innov_hist/_rot_innov_hist), not a fixed constant: a controller
        that's been very consistent gets a tight weak bound (small disagreement
        already counts as a soft jump), one with a noisier recent history gets
        a looser one, same shape as _vision_weight's per-controller overrides.
        Falls back to a thin ramp hugging the hard-gate ceiling itself (i.e.
        approximately no extra pushback beyond today's cliff) while history is
        too thin to be meaningful -- min_samples default chosen so a fresh
        bootstrap/reset doesn't start the very next frame with a hair-trigger
        threshold from a single sample. NOT exactly the ceiling itself: passing
        weak==strong==ceiling would hit _ramp_down's own degenerate-input
        branch ("if strong <= weak: return 1.0" unconditionally, regardless of
        value) -- silently disabling gating entirely rather than replicating
        today's cliff, which is the opposite of this fallback's intent
        (confirmed the hard way: a 5m cold-reacquisition candidate confirmed
        with agreement=1.0 under exactly this condition before the 0.9x
        margin below was added). A 10%-of-ceiling-wide ramp keeps the
        fallback a close approximation of today's binary gate while staying a
        valid (non-degenerate) ramp.

        Always clipped to [floor, 0.9*ceiling] once history IS populated: the
        floor stops a dead-still controller's own near-zero recent noise
        floor from making ordinary vision jitter look like a soft jump; the
        ceiling is the same hard-gate threshold _implausible_jump_thresholds
        computes (evaluated at THIS moment's own speed -- self.v if
        velocity_established, else base-only, same as the hard gate itself),
        so this can only ever make the gate MORE sensitive than today, never
        less. The SAME 0.9x margin as the thin-history fallback above --
        2026-09-15, real case: right_controller, walk_hard, frame 34 --
        confirmed this exact branch (populated history, len=10) can ALSO hit
        the identical weak==ceiling degenerate condition the thin-history
        fallback's own comment already documents, whenever this controller's
        recent history has been noisy enough that agreement_k_pos/_rot *
        percentile90(hist) reaches the raw (unmargined) ceiling -- a 69mm/
        11ms, quality=0.17 candidate got agreement=1.0 (zero pushback) purely
        from this clip landing exactly at pos_ceil_m, not because the
        candidate was actually trustworthy. The original 0.9x margin was
        only ever applied to the THIN-history return above; this branch's
        own clip used the raw ceiling unmargined, missing the exact fix its
        sibling branch already got for the exact same underlying bug."""
        pos_ceiling, rot_ceiling = self._implausible_jump_thresholds(frame_ts_ns)
        min_samples = int(self._hc.get("agreement_hist_min_samples", 5))
        if len(self._pos_innov_hist) < min_samples:
            return 0.9 * pos_ceiling, 0.9 * rot_ceiling
        pos_weak = float(np.clip(
            float(self._hc.get("agreement_k_pos", 3.0)) * float(np.percentile(self._pos_innov_hist, 90)),
            float(self._hc.get("agreement_floor_pos_m", 0.01)), 0.9 * pos_ceiling))
        rot_weak = float(np.clip(
            float(self._hc.get("agreement_k_rot", 3.0)) * float(np.percentile(self._rot_innov_hist, 90)),
            float(self._hc.get("agreement_floor_rot_deg", 2.0)), 0.9 * rot_ceiling))
        return pos_weak, rot_weak

    def _looks_like_a_sibling(self, p_meas: np.ndarray, frame_ts_ns: int) -> bool:
        """True if p_meas sits implausibly close to a SIBLING controller's own
        still-live, trustworthy position at this same instant -- position-only,
        same rationale as PoseFusionFilter._looks_like_a_sibling (src/pose_fusion.py)
        which this mirrors: the two controllers' LED constellations are documented
        near-mirror-images, so orientation alone can look locally plausible for
        either identity, but the two physical controllers are never in the same
        place. Unlike that method (bootstrap-only, chi2/Mahalanobis-gated against
        the sibling's own covariance), this filter has no covariance to test
        against, so this uses a flat radius (sibling_collision_dist_m) instead --
        and is checked in TWO places, not just bootstrap: try_update's own
        bootstrap branch, and every candidate seen while vision_only (see
        _try_cold_reacquire) -- both are cases where THIS filter has no
        trustworthy distance-based reference of its own to gate against,
        which is exactly the gap _vision_weight's own docstring documents this
        project's real identity-swap failure mode slipping through."""
        if not self._sibling_filters:
            return False
        trust_min = float(self._cfg.get("sibling_trust_min", 0.3))
        collision_dist_m = float(self._hc.get("sibling_collision_dist_m", 0.15))
        for sib in self._sibling_filters:
            sib_trust = sib.trust(frame_ts_ns)
            if sib_trust is None or sib_trust < trust_min:
                continue  # no live sibling state, or its state isn't currently trustworthy
            predicted = sib.predict(frame_ts_ns)
            if predicted is None:
                continue  # sibling has no live prediction either -- no signal available
            # FRESHNESS gate: only trust the sibling's position as a collision
            # reference if IT was itself vision-confirmed on this exact frame
            # (frames_since_update reset to 0 by every real-accept branch in
            # try_update/_try_cold_reacquire -- predict()'s own note_real_frame
            # call just above already bumped it for this frame if the sibling
            # is currently coasting, regardless of controller processing order
            # this frame). trust() alone isn't enough: it decays with elapsed
            # TIME since the sibling's last real update, not with how far its
            # IMU-extrapolated POSITION may have already drifted in that same
            # window -- a sibling only 2-3 real frames into a coast can still
            # read trust=0.6+ while its own dead-reckoned position has already
            # moved 100mm+ if the real hand is moving at a realistic few m/s
            # (confirmed on this project's own recording: frame_range 3850-
            # 3950 relative frame 17 -- right_controller's own vision candidate
            # was excellent (7+7 inliers, cam0/cam1, sub-0.15px both) but got
            # rejected as a false "identity swap" purely because it landed
            # within sibling_collision_dist_m of left_controller's own
            # 2-frames-stale, IMU-coasted position, discarding a correct
            # reacquisition in favor of leaving right pinned at its own long-
            # stale reported pose). A sibling that's genuinely tracking
            # normally has frames_since_update==0 on every successful frame,
            # so this only ever excludes a sibling that is ITSELF currently
            # having trouble -- exactly the case where its position is least
            # trustworthy as a reference to reject someone else against.
            if sib.frames_since_update != 0:
                continue
            _, sib_p_pred = predicted
            if float(np.linalg.norm(p_meas - sib_p_pred)) <= collision_dist_m:
                return True
        return False

    # ------------------------------------------------------------------
    # Core update -- cost-function fusion, plus one hard implausibility reject
    # ------------------------------------------------------------------
    def try_update(self, solution: dict, frame_ts_ns: int) -> bool:
        """Fuse a new vision solution against the IMU prediction and the
        previous state -- see module docstring for the cost function."""
        T_world_ctrl = solution["T_world_ctrl"]
        R_meas, p_meas = T_world_ctrl.R, T_world_ctrl.t
        confidence = max(float(solution.get("confidence", 1.0)), 0.05)
        # Computed here (not just later, at the FUSED-branch's own use below)
        # so _try_cold_reacquire's weak/strong routing can see it too.
        n_inliers = (len(solution.get("assignment") or [])
                     + sum(len(v) for v in (solution.get("aux_assignments") or {}).values()))
        # Also hoisted early (2026-09-15, same reason as n_inliers above) so
        # _try_cold_reacquire's own contested-winner exemption (see that
        # method's own docstring) can judge "genuinely strong" the same way
        # _vision_weight does -- inliers alone isn't enough to rule out a
        # clean-but-wrong low-point fit.
        error_px = float(solution.get("error", 0.0))
        # True when EVERY camera solution behind this candidate came from the
        # prior-constrained (P2P/P1P) solver, which copies its rotation from the
        # prediction and only solves translation -- see _try_cold_reacquire's
        # thin-candidate check for why that is not independent evidence.
        _cam_methods = [m for m in (solution.get("camera_method") or {}).values() if m]
        self._cand_rotation_from_prior = bool(_cam_methods) and all(
            str(m).startswith("prior_constrained") for m in _cam_methods)
        # Hoisted early for the same reason as n_inliers/error_px above --
        # the along-track/cross-track diagnostic further down needs it (the
        # gravity_implausible veto that originally motivated threading this
        # through was reverted 2026-09-18, see _try_cold_reacquire's own
        # comment; this plumbing was left in place for the diagnostic).
        # None when no accel stream is available (matches controller.py's
        # own accel_now contract).
        accel_now = solution.get("accel_now")

        # ── Bootstrap (no prior state) ──────────────────────────────────
        # Reuses _try_cold_reacquire's own weak/strong buffering wholesale
        # (2026-09-13, user-reported: an uncorroborated single weak/
        # coverage_fallback bootstrap candidate, followed by several more
        # weak frames in a row, was becoming the trusted anchor immediately
        # with nothing to gate it -- see that method's own docstring for the
        # full "why" and a real trace-through). Safe to reuse unmodified:
        # _try_cold_reacquire's weak-routing/buffer/confirm logic never reads
        # self.R/self.p (only writes them, inside _accept_cold_reacquire),
        # and R_pred/p_pred/dt_s=None makes _report_if_still_usable a clean
        # no-op -- there IS no prior anchor to dead-reckon a coasted estimate
        # from at bootstrap anyway, so "nothing to report while pending" is
        # exactly the right behavior here, not a special case.
        if self.R is None:
            return self._try_cold_reacquire(
                R_meas, p_meas, frame_ts_ns, confidence,
                coverage_fallback=bool(solution.get("coverage_fallback", False)), n_inliers=n_inliers,
                error_px=error_px,
                winner_was_contested=bool(solution.get("winner_was_contested", False)),
                swap_suspected=bool(solution.get("swap_suspected", False)),
                accel_now=accel_now,
                log_prefix="BOOTSTRAP", pending_outcome="bootstrap_pending", accept_outcome="bootstrap",
            )

        # ── Predict; fail-open if IMU/g_world unavailable ───────────────
        predicted = self.predict(frame_ts_ns)
        if predicted is None and self._rot_gate_continuous():
            # predict() had no full (position+rotation) prediction, which used to mean "accept everything".
            # Rotation only needs the gyro, so still veto a candidate that contradicts it.
            _veto = self._gyro_only_rot_veto(R_meas, frame_ts_ns)
            if _veto is not None:
                _innov, _T, _info = _veto
                self.consecutive_rejects += 1
                self._set_last(outcome="implausible_reject", rot_innov_deg=_innov, confidence=confidence)
                _log.info(
                    f"[{self._ctrl_name}] IMPLAUSIBLE vs gyro-only prediction (no full IMU prediction) "
                    f"ts={frame_ts_ns} rule=nohs_gyro_only mode=continuous rot_innov={_innov:.2f}deg "
                    f"T={_T:.1f}deg dt={_info['dt_s']:.4f}s peak_gyro={_info['peak_dps']:.0f}dps "
                    f"sat={_info['saturated']} rejects={self.consecutive_rejects} -- REJECTED, "
                    f"vision candidate was {_fmt_v(p_meas)}")
                return False
        if predicted is None:
            self.R, self.p = R_meas, p_meas
            self.v = np.zeros(3)
            self.velocity_established = False
            self._seed_rotation_grace(
                bool(solution.get("coverage_fallback", False)),
                high_risk=bool(solution.get("winner_was_contested", False))
                          or bool(solution.get("swap_suspected", False)))
            self.last_update_ts_ns = frame_ts_ns
            self._last_known_p, self._last_known_p_ts_ns = self.p.copy(), frame_ts_ns
            self._report(frame_ts_ns, self.R, self.p)
            self._trust_window.append(confidence)
            self.frames_since_update = 0
            self._set_last(outcome="fail_open", confidence=confidence)
            _log.debug(f"[{self._ctrl_name}] FAIL-OPEN ts={frame_ts_ns} — no IMU prediction available "
                       f"(no coverage / g_world not converged yet), reporting raw vision pos={_fmt_v(p_meas)} "
                       f"conf={confidence:.2f}")
            return True
        R_pred, p_pred = predicted
        dt_s = (frame_ts_ns - self.last_update_ts_ns) / 1e9

        # Innovation vs. the IMU prediction -- computed BEFORE the weight so
        # the implausibility gate below can see it: a raw distance check on
        # vision's own candidate, independent of how many inliers/how low an
        # error IT reported.
        rot_innov = Rotation.from_matrix(R_pred.T @ R_meas).as_rotvec()
        pos_innov_m = float(np.linalg.norm(p_meas - p_pred))
        rot_innov_deg = float(np.degrees(np.linalg.norm(rot_innov)))

        # rot_pred_implausible (2026-09-16, real case: right_controller,
        # static_dark, a 4-frame/~78ms calm-motion loss where two
        # consecutive cam1-only 6-inlier P3P solves AGREED with each other
        # (drot=1.65deg) but were BOTH wrong (150mm/158deg off mocap) --
        # the same systematic single-camera mirror-branch ambiguity,
        # repeated, which _try_cold_reacquire's own "two weak candidates
        # agreeing is corroboration" logic can't distinguish from real
        # corroboration since it only ever compares the two candidates
        # against EACH OTHER, never against this rot_innov_deg/R_pred
        # computed just above -- rot_innov_deg IS the exact right signal
        # (rot_innov_deg=158deg here), it's just discarded once execution
        # falls into the imu_frame_scale<=0 branch below (or the
        # degenerate w_sum<=0 fallback further down), neither of which the
        # warm-path's own rot_innov_deg-based hard gate reaches (that gate
        # is deliberately imu_frame_scale>0-gated -- see the comment on
        # _rot_implausible further down).
        #
        # Gated on its OWN gyro-credibility budget (_effective_coast_
        # budget_s, axis="gyro", rate_prefix="coast_trust_rot" -- the SAME
        # empirically-fit rate/floor the degenerate-fallback gate already
        # uses, see that config key's own comment) rather than imu_frame_
        # scale's flat 4-frame count: the real case's own dt_s (88.7-
        # 99.8ms, elapsed since the LAST REAL ACCEPT, not since the loss
        # started) is already past degenerate_fallback_max_s's own 0.066s
        # base at zero shrink, so THIS veto needs its own, more generous
        # base (cold_reacquire_rot_veto_max_s, default 0.25s) -- checked
        # numerically, not assumed, during design review. Threshold
        # (cold_reacquire_rot_veto_thresh_deg, default 100.0) is NOT
        # _implausible_jump_thresholds' own rotation ceiling -- that one
        # widens proportionally to stale_s (elapsed time), which for a
        # cold-reacquire call (by construction always stale) would swamp
        # any realistic rot_innov_deg and make the check inert; chosen
        # instead directly from the real sweep's own false-positive rate
        # (calm/slow gyro, dt<=100ms: rot_err_deg exceeds 100deg only
        # 0.013% of the time -- comfortably below this real case's 158deg
        # with a checked margin). _rotation_seed_grace_frames exemption
        # mirrors the existing warm-path hard gate's own _rot_implausible
        # reasoning (don't veto off a just-seeded, itself-untrustworthy
        # orientation).
        _rot_pred_budget_s = self._effective_coast_budget_s(
            float(self._hc_get("cold_reacquire_rot_veto_max_s", 0.25)), frame_ts_ns,
            axis="gyro", rate_prefix="coast_trust_rot")
        _rot_veto_thresh_deg = float(self._hc_get("cold_reacquire_rot_veto_thresh_deg", 100.0))
        rot_pred_implausible = (dt_s <= _rot_pred_budget_s
                                 and self._rotation_seed_grace_frames <= 0
                                 and rot_innov_deg > _rot_veto_thresh_deg)
        if self._rot_gate_continuous():
            # continuous mode replaces the (dt <= budget) & (innov > 100 deg) switch wholesale
            _gi = self._rot_gate_info(frame_ts_ns)
            rot_pred_implausible = (self._rotation_seed_grace_frames <= 0 and rot_innov_deg > _gi["T"])
            if rot_pred_implausible:
                _log.info(f"[{self._ctrl_name}] rot gate VETO ts={frame_ts_ns} rule=rot_pred_implausible "
                          f"mode=continuous headset={_gi['headset']} rot_innov={rot_innov_deg:.2f}deg "
                          f"T={_gi['T']:.1f}deg dt={_gi['dt_s']:.4f}s peak_gyro={_gi['peak_dps']:.0f}dps "
                          f"sat={_gi['saturated']} rejects={self.consecutive_rejects}")

        # IMU frame-count decay: w_imu (and, below, the hard implausibility
        # gate's trust in p_pred) fades linearly to 0 over imu_decay_frames
        # consecutive lost/coasted REAL FRAMES, frame-counted (not time-based,
        # since this recording's cadence isn't uniform) via frames_since_update.
        # -1 below corrects an off-by-one: predict() just bumped
        # frames_since_update for THIS frame's own ts (see its own docstring --
        # every real frame bumps it once, including a normal immediate-next-
        # frame accept with no gap at all), so the count of frames LOST BEFORE
        # this one excludes that just-added increment (confirmed broken:
        # without the -1, a perfectly stable back-to-back-frame recording
        # showed frames_lost=1/imu_frame_scale=0.75 on EVERY single frame, not
        # just after a real gap). Direct fix for "IMU shouldn't affect
        # anything after a few lost frames" -- previously w_imu was a fixed
        # 1.0 forever, no matter how long predict() had been dead-reckoning
        # unconstrained.
        imu_decay_frames = float(self._hc.get("imu_decay_frames", 4))
        _frames_lost = max(0, self.frames_since_update - 1)
        imu_frame_scale = (max(0.0, 1.0 - _frames_lost / imu_decay_frames)
                            if imu_decay_frames > 0 else 1.0)

        # Cold-state routing decision -- computed from the REAL (pre-debug-override)
        # decay state, deliberately BEFORE vision_only_debug can zero imu_frame_scale
        # below. If this used the debug-overridden value instead, turning on
        # vision_only_debug would silently route every single frame through
        # _try_cold_reacquire instead of the normal per-frame blend --
        # defeating the flag's whole documented purpose (isolating raw
        # per-frame vision behavior with zero IMU involvement,
        # gate included) and making debug mode behave nothing like what it's
        # named for.
        if imu_frame_scale <= 0.0:
            # Speed-aware hard reject past imu_decay_frames (2026-09-16,
            # same-day follow-up to rot_pred_implausible above): the warm-
            # path's own hard implausibility gate further down (_rot_
            # implausible) is deliberately imu_frame_scale > 0.0-gated, so
            # once frames_lost >= imu_decay_frames (flat 4-frame count,
            # unrelated to real elapsed time or motion violence) this
            # branch returns BEFORE that gate is ever reached -- a lone
            # wildly-wrong candidate right at/after the frame-4 cliff was
            # never hard-rejected the way the identical candidate would
            # have been one frame earlier, even while gyro is still fully
            # credible. rot_pred_implausible already answers exactly this
            # ("does this candidate disagree with a still-credible gyro
            # prediction") independent of imu_frame_scale -- reused here
            # verbatim as a genuine hard-reject trigger, not just the
            # weak-buffer-forcing signal it already is below. Same reject
            # shape as the warm gate's own _rot_implausible branch further
            # down -- zero new config, tuning cold_reacquire_rot_veto_*
            # tunes both this reject and _try_cold_reacquire's own weak/
            # CONFIRM handling at once. As dt_s grows past the credibility
            # budget, rot_pred_implausible naturally goes False and control
            # falls through to ordinary cold routing -- no new cliff.
            if rot_pred_implausible:
                if self._check_reject_streak_override(R_meas, p_meas, frame_ts_ns, confidence, solution, n_inliers,
                                                      error_px, R_pred, p_pred, dt_s, rot_innov_deg, pos_innov_m,
                                                      accel_now):
                    return True
                self.consecutive_rejects += 1
                self._report_coast_if_usable(frame_ts_ns, R_pred, p_pred, dt_s)
                self._set_last(outcome="implausible_reject", pos_innov_m=pos_innov_m, rot_innov_deg=rot_innov_deg,
                                confidence=confidence, pos_pred=p_pred, R_pred=R_pred)
                _log.info(
                    f"[{self._ctrl_name}] IMPLAUSIBLE vs still-credible gyro prediction "
                    f"(past imu_frame_scale decay) ts={frame_ts_ns} — rot_innov={rot_innov_deg:.2f}deg -- "
                    f"REJECTED (state left at IMU prediction), vision candidate was {_fmt_v(p_meas)}"
                )
                return False
            return self._try_cold_reacquire(R_meas, p_meas, frame_ts_ns, confidence,
                                             coverage_fallback=bool(solution.get("coverage_fallback", False)),
                                             n_inliers=n_inliers, error_px=error_px, R_pred=R_pred, p_pred=p_pred,
                                             dt_s=dt_s,
                                             winner_was_contested=bool(solution.get("winner_was_contested", False)),
                                             swap_suspected=bool(solution.get("swap_suspected", False)),
                                             rot_pred_implausible=rot_pred_implausible,
                                             accel_now=accel_now)

        # Manual-debug override: fusion_heuristic.vision_only_debug: true forces
        # imu_frame_scale to 0 on EVERY frame, regardless of frames_since_update --
        # everything downstream already keys off imu_frame_scale (the hard
        # implausibility gate, w_imu below), so this one override makes every
        # accepted frame's TRACKING STATE exactly the raw vision candidate,
        # permanently (still passes through the One Euro Filter like any other
        # reported pose -- see _report). For isolating "is this IMU
        # contamination or is this vision itself" -- put this back to false (or
        # remove it) once done; it is NOT a tuning knob, it's a debug bypass
        # (bootstrap/fail_open frames are still whatever they always were --
        # this only affects the weighted-blend branch).
        if bool(self._hc.get("vision_only_debug", False)):
            imu_frame_scale = 0.0

        # ── HARD implausibility gate: a genuine reject, not a zero-weighted fuse ──
        # A position/rotation jump versus the IMU prediction far beyond real hand
        # motion is almost certainly an identity swap (this controller's search
        # matched a SIBLING controller's own blobs) or some other geometrically
        # degenerate mismatch -- see _vision_weight's own docstring for how this
        # can slip past the inlier/error ramps with a deceptively clean
        # reprojection error. This USED to still return True here (module
        # docstring used to say "this still isn't a reject"), fusing the
        # candidate in at zero vision weight but otherwise treating the frame as
        # a normal accepted update: last_update_ts_ns still advanced to this
        # frame and every camera's consecutive_failures still got reset to 0 by
        # ControllerTracker._commit_fused_solution's `if accepted:` branch. Since
        # this filter has no other reject path (see should_force_cold_start's own
        # elapsed-time/consecutive-rejects triggers), a single stray implausible
        # detection was enough to indefinitely reset BOTH the grace-period clock
        # and the IMU-only coasting budget, letting a genuinely lost controller
        # coast on a contaminated prediction forever instead of ever being
        # re-declared cold (found investigating a report of a controller
        # "freezing" in place well past when tracking should have gone lost --
        # vision kept hallucinating occasional wrong-geometry candidates against
        # the sibling controller, each one silently re-arming everything). Now a
        # real reject: state is left at the IMU prediction, exactly like
        # PoseFusionFilter's own gated_reject.
        #
        # GATED ON imu_frame_scale > 0: this check's whole premise is "trust
        # p_pred enough to use it as a reference for what's plausible" -- but
        # p_pred is exactly what imu_frame_scale says has decayed to worthless
        # after a long-enough coast. Found on a real case: a weak (n_inliers=4)
        # pre-loss accept fed an 8-real-frame-lost coast, and the resulting
        # stale p_pred then hard-rejected several genuinely-correct
        # reacquisition candidates as "implausible" purely because they were
        # far from that stale prediction -- neither the quality-aware gap
        # damping nor the IMU decay above ever got a chance to run, because
        # this gate returns before either is even reached. Once IMU has fully
        # decayed, p_pred can no longer veto a candidate either -- fall through
        # to the normal quality-based blend instead, which by then has zero
        # IMU weight anyway.
        # POSITION criterion additionally requires self.velocity_established:
        # p_pred is dead-reckoned from self.v, which after a bootstrap/reset/
        # fail-open is a FABRICATED v=0, not a measurement -- real accel
        # integration from that fabricated start stays small over a short gap
        # regardless of the controller's true (unknown) velocity, so a p_pred
        # this close to the last accepted position carries no actual evidence
        # about where the controller can plausibly be found next. Confirmed on
        # a real recording: BOOTSTRAP (v=0, confidence=0.05) followed by 3 lost
        # frames then a genuine ~324mm-away vision candidate, rejected as
        # "implausible" purely because nothing had ever measured this
        # controller's real velocity in the first place. ROTATION has no such
        # gap in general -- gyro integration measures real angular rate
        # directly from the first frame, independent of any velocity/
        # translation state, so rot_innov_deg stays a valid sanity check
        # regardless of velocity_established (still gated on imu_frame_scale
        # > 0 same as before, for the existing long-coast-decay reason
        # described above) -- EXCEPT while _rotation_seed_grace_frames > 0:
        # that reasoning assumes the orientation gyro is integrating FROM is
        # itself trustworthy, which isn't true right after a coverage-
        # fallback seed (see that field's own __init__ comment for both real
        # cases this exemption exists for -- the coverage-fallback seed
        # itself, AND the one genuinely-good frame accepted immediately
        # after it, which had no chance to have ITS OWN rotation vetted
        # either).
        #
        # NOT decremented here -- only once a frame whose OWN candidate isn't
        # ALSO coverage_fallback actually lands (below, after the blend),
        # and reset back to full on one that IS. See _rotation_seed_grace_
        # frames' own __init__ comment for the two real bugs this exact
        # sequencing was built to close.
        _rot_seed_untrustworthy = self._rotation_seed_grace_frames > 0
        # _pos_ceil_m/_rot_ceil_deg: the UNSHRUNK (quality-independent)
        # ceiling -- kept exactly as before so Case A's own reuse of these
        # two values below (~line 1614) stays untouched, per this fix's own
        # scope (quality-aware shrink applies to the hard implausibility
        # gate only, see _implausible_jump_thresholds' docstring). The hard
        # gate itself checks a SEPARATE, quality-shrunk POSITION ceiling
        # (_pos_ceil_m_quality) computed alongside it.
        _pos_ceil_m, _rot_ceil_deg = self._implausible_jump_thresholds(frame_ts_ns)
        _candidate_quality, _, _ = self._vision_quality(n_inliers, error_px)
        _pos_ceil_m_quality, _ = self._implausible_jump_thresholds(frame_ts_ns, quality=_candidate_quality)
        _pos_implausible = (imu_frame_scale > 0.0 and self.velocity_established
                             and pos_innov_m > _pos_ceil_m_quality)
        _rot_implausible = (imu_frame_scale > 0.0 and not _rot_seed_untrustworthy
                             and rot_innov_deg > _rot_ceil_deg)
        _implausible = _pos_implausible or _rot_implausible
        if _implausible:
            if self._check_reject_streak_override(R_meas, p_meas, frame_ts_ns, confidence, solution, n_inliers,
                                                  error_px, R_pred, p_pred, dt_s, rot_innov_deg, pos_innov_m,
                                                  accel_now):
                return True
            self.consecutive_rejects += 1
            self._report_coast_if_usable(frame_ts_ns, R_pred, p_pred, dt_s)
            self._set_last(outcome="implausible_reject", pos_innov_m=pos_innov_m, rot_innov_deg=rot_innov_deg,
                            confidence=confidence, pos_pred=p_pred, R_pred=R_pred)
            _log.info(
                f"[{self._ctrl_name}] IMPLAUSIBLE vision jump ts={frame_ts_ns} — "
                f"pos_innov={pos_innov_m * 1000:.1f}mm rot_innov={rot_innov_deg:.2f}deg -- "
                f"likely identity swap or a degenerate low-point fit; REJECTED "
                f"(state left at IMU prediction), vision candidate was {_fmt_v(p_meas)}"
            )
            return False

        # ── Cost function: J(p) = w_imu*||p-p_pred||^2 + w_vision*||p-p_meas||^2.
        # Closed-form minimizer is a weighted average of the two candidates. No
        # smoothing term here any more -- see module docstring: jitter removal
        # moved to a One Euro Filter applied to the REPORTED pose only (_report,
        # called below), decoupled from this tracking-state blend entirely.
        # w_vision itself is quality-dependent -- see _vision_weight. ──
        # (error_px itself is now hoisted to the top of this method, next to
        # n_inliers -- see that computation's own comment.)

        # vision_only: past imu_decay_frames lost/coasted real frames, the IMU
        # prediction is no longer a trustworthy reference (a long unconstrained
        # dead-reckon from a stale anchor) -- past this point vision gets the
        # WHOLE say (w_imu = 0), so the tracking state becomes EXACTLY the raw
        # vision measurement, not "mostly vision."
        #
        # NOTE: _rot_seed_untrustworthy deliberately does NOT get this same
        # w_imu=0/force_full_trust treatment for POSITION -- tried, and found
        # actively worse on the real reproduction: a coverage-fallback
        # candidate's OWN position estimate can be nearly as unreliable as
        # its rotation (confirmed: bootstrap pos vs. the very next
        # coverage-fallback frame's own vision pos already disagreed by
        # 165mm), so fully trusting it moved self.p to that noisy value AND
        # set velocity_established=True from the resulting noisy step --
        # which then hard-rejected the next THREE genuinely good candidates
        # as "implausible" (452-890mm off a now-wrong anchor), worse than the
        # original complaint. Rotation's exemption stays because rotation
        # specifically (not position) is what's structurally unstable for a
        # low/degenerate point count -- position is comparatively stable
        # enough that the existing quality ramp is the safer default.
        w_imu = float(self._hc.get("cost_weight_imu", 1.0)) * imu_frame_scale
        vision_only = imu_frame_scale <= 0.0
        w_vision, inlier_factor, error_factor = self._vision_weight(
            n_inliers, error_px, pos_innov_m, rot_innov_deg, frame_ts_ns, gate_active=not vision_only,
            rot_gate_active=(not vision_only) and not _rot_seed_untrustworthy)
        quality = (inlier_factor + error_factor) / 2.0

        if vision_only:
            vision_gap_scale = 1.0  # nothing left to damp against once IMU has fully decayed
        else:
            # Long-gap damping -- see _gap_vision_scale's own docstring. Quality-
            # aware: a genuinely strong post-gap match (high quality) gets little
            # or none of this penalty, only a weak one does.
            vision_gap_scale = self._gap_vision_scale(dt_s, quality)
            w_vision *= vision_gap_scale

        w_sum = w_imu + w_vision
        # EXPERIMENTAL (2026-09-18, testing a user-proposed idea, not yet
        # validated): route a candidate with a bottomed-out inlier_factor
        # (n_inliers<=weak_inliers -- the same "clean-but-wrong" shape the
        # degenerate branch below already guards against) into the SAME
        # weak-candidate buffer EARLIER than w_sum<=0.0 requires -- that
        # condition needs error_factor ALSO at floor, which a deceptively
        # low reprojection error on a near-degenerate point count can dodge
        # (real case: frame 82/static_easy, right_controller, n_inliers=5
        # exactly at weak_inliers, error_px=0.495 well under its own
        # weak_error_px=1.1 override -- inlier_factor=0.0 but
        # error_factor=0.672 keeps w_sum=0.60, nowhere near 0). Gated on
        # imu_frame_scale already having decayed below a threshold well
        # short of full trust (NOT "any imu_frame_scale," which an earlier,
        # reverted attempt tried at the wrong call site -- vision_only's own
        # gate flag -- and which broke 3 tests by firing on fresh/warm
        # single-frame weak matches too); only a candidate arriving after
        # real coasting, where p_pred itself already carries much less
        # weight, gets the earlier hold.
        # 0.75 was tried (2026-09-19, static_medium frame_idx 2870) and
        # REVERTED to 0.5: it fixed that case (65mm->18mm) but regressed the
        # right controller on static_easy (the earlier frame-82 fix came back
        # as 174mm: an earlier weak candidate got buffered and then "agreed"
        # with the wrong one) and walk_easy (bad frames 5->7, from pure-IMU
        # holds during slow drift).
        _weak_imu_scale_thresh = float(self._hc_get("degenerate_fallback_weak_inlier_imu_scale_thresh", 0.5))
        _weak_inlier_partial_decay = inlier_factor <= 0.0 and imu_frame_scale <= _weak_imu_scale_thresh
        if w_sum <= 0.0 or _weak_inlier_partial_decay:
            # Degenerate: vision's own quality ramps both bottomed out (rare --
            # needs both n_inliers AND error_px at/below their weak floors at
            # once, since gate_active above already stops the distance check
            # alone from zeroing w_vision here) -- AND, in this project's real
            # config, w_imu is ALSO always 0 here regardless of imu_frame_scale
            # (cost_weight_imu=0.0, "matches this project's current real
            # config" -- see this class's own module docstring), not because
            # IMU "has fully decayed" as an earlier version of this comment
            # claimed. Nothing trustworthy to blend by the normal weights --
            # but rather than falling back to the bare IMU prediction
            # UNCONDITIONALLY (as this branch used to), check whether p_pred
            # ITSELF is still credible on each axis first.
            #
            # Added 2026-09-15 (user-directed, real case: right_controller,
            # walk_hard, frame 71, cam3): a 3-frame-lost (imu_frame_scale=0.25,
            # dt_s=66.7ms) reacquisition candidate hit this exact branch
            # (n_inliers=5/error_px=1.20, both at/below their weak floors) and
            # got the bare IMU prediction unconditionally -- no time/accel/gyro
            # decay of any kind, unlike every other "how long is a coast still
            # trustworthy" question in this file. Mocap ground truth: that pure
            # -IMU pose was 406mm off in POSITION but only 4.24deg off in
            # ROTATION; the rejected vision candidate was 103.8mm off in
            # position but 135deg off in rotation -- each source wrong on a
            # DIFFERENT axis, which a single blind trust-or-not decision can't
            # express. Reuses _effective_coast_budget_s (already-validated
            # accel/gyro shrink formula, see that method's own docstring) with
            # a NEW, axis-split base budget: degenerate_fallback_max_s mirrors
            # matching.imu_only_propagation_max_s's own value (this is a
            # search-anchor-tier decision, not the more lenient display-tier
            # cold_pending_report_max_gap_s) rather than a re-derived fit --
            # see that config key's own comment. Past EITHER axis budget,
            # p_pred is no longer trustworthy enough to blindly accept as the
            # new tracked state -- route through _try_cold_reacquire instead
            # (same weak-candidate buffer/confirm machinery the imu_frame_scale
            # <=0 case above already uses): n_inliers<=weak_inliers is
            # guaranteed true here (inlier_factor==0 requires exactly that, and
            # quality==0 requires inlier_factor==0), so this ALWAYS takes that
            # method's own weak-buffer path, never its immediate-accept path --
            # "0 imu impact, a weak pose from vision that requires
            # verification," not a NEW trust decision bolted on top of an
            # existing one.
            _degenerate_base_s = float(self._hc_get("degenerate_fallback_max_s", 0.066))
            _pos_budget_s = self._effective_coast_budget_s(_degenerate_base_s, frame_ts_ns, axis="accel")
            # rate_prefix="coast_trust_rot" (2026-09-16, "big IMU investigation"):
            # the ROTATION axis gets its own, steeper shrink rate + lower floor,
            # fit from a systematic real-mocap-vs-blind-gyro-dead-reckoning sweep
            # across all 8 recordings -- see _effective_coast_budget_s's own
            # docstring for why the shared coast_trust_min_budget_s floor was
            # silently overriding an already-correctly-shrunk rotation budget
            # back up for moderate-or-faster gyro bursts. axis="gyro" here means
            # only per_gyro/coast_trust_rot_shrink_s_per_dps is actually read;
            # coast_trust_rot_shrink_s_per_mps2 stays unused.
            _rot_budget_s = self._effective_coast_budget_s(_degenerate_base_s, frame_ts_ns,
                                                             axis="gyro", rate_prefix="coast_trust_rot")
            _rot_budget_s = self.rot_coast_budget_s(frame_ts_ns, _rot_budget_s)
            if dt_s > _pos_budget_s or dt_s > _rot_budget_s:
                return self._try_cold_reacquire(
                    R_meas, p_meas, frame_ts_ns, confidence,
                    coverage_fallback=bool(solution.get("coverage_fallback", False)),
                    n_inliers=n_inliers, error_px=error_px, R_pred=R_pred, p_pred=p_pred, dt_s=dt_s,
                    winner_was_contested=bool(solution.get("winner_was_contested", False)),
                    swap_suspected=bool(solution.get("swap_suspected", False)),
                    rot_pred_implausible=rot_pred_implausible,
                    accel_now=accel_now)
            # BUG FIX (2026-09-18, real case: right_controller, static_easy,
            # frame_idx 1340): w_vision was never zeroed here -- only w_imu/
            # w_sum got overridden to (1.0, 1.0), meaning p_new below silently
            # became an UNNORMALIZED SUM (p_pred + w_vision*p_meas) instead of
            # a weighted average, whenever w_vision was nonzero on entry.
            # Dormant since this branch's original 2026-09-15 introduction:
            # the only way in used to be w_sum<=0.0, which (cost_weight_imu=0
            # in this project's real config) already forced w_vision<=0 too,
            # so w_vision was always ~0 here and the bug never showed.
            # _weak_inlier_partial_decay (2026-09-18, see its own comment
            # above) is the first path that can reach this branch with
            # w_vision still meaningfully nonzero (inlier_factor=0 but
            # error_factor high) -- confirmed on the real case: imu=(-0.067,
            # 0.340, 0.241) vision=(-0.074, 0.358, 0.254) (the two agree to
            # within ~15mm) produced tracking=(-0.137, 0.678, 0.481) -- almost
            # exactly imu+vision added together, not blended. This whole
            # branch's own intent is "p_pred alone is still trustworthy
            # enough within budget, use it" (see the dt_s-budget check just
            # above), so vision must be fully excluded, not just outweighed.
            w_imu, w_sum, w_vision = 1.0, 1.0, 0.0

        p_new = (w_imu * p_pred + w_vision * p_meas) / w_sum

        # Rotation: same weights applied to tangent-space error vectors,
        # linearized around the IMU prediction (its own error against itself is 0).
        e = (w_vision * rot_innov) / w_sum
        R_new = R_pred @ Rotation.from_rotvec(e).as_matrix()

        # ── Case A: soft pushback for a disagreement too small to hit the hard
        # implausibility gate above but still inconsistent with a trustworthy
        # IMU prediction -- e.g. a few cm/mm-scale disagreement that today gets
        # zero pushback since it's nowhere near implausible_jump_pos_m. Skipped
        # when imu_frame_scale is 0 here (debug bypass only -- genuine cold-state
        # already returned via _try_cold_reacquire above): pulling toward p_pred
        # while vision_only_debug is deliberately isolating pure-vision behavior
        # would reintroduce exactly the IMU influence that flag exists to remove.
        agreement_pos = agreement_rot = 1.0
        if imu_frame_scale > 0.0:
            pos_weak, rot_weak = self._agreement_weak_bounds(frame_ts_ns)
            # _pos_ceil_m/_rot_ceil_deg: same hard-gate threshold already
            # computed above via _implausible_jump_thresholds() for this
            # exact frame -- reused here rather than recomputed so Case A's
            # ceiling and the hard gate's own ceiling can never drift apart
            # within a single call.
            agreement_pos, agreement_rot = self._agreement_factor(
                pos_innov_m, rot_innov_deg,
                pos_weak, _pos_ceil_m,
                rot_weak, _rot_ceil_deg)
            # Scale the PUSHBACK (not the trust itself) by imu_frame_scale --
            # same decay factor the cost-blend's w_imu above already applies.
            # Added 2026-09-14 (real case: right_controller, walk_hard,
            # frame 38 of a 34-lost-38 sequence): a reacquisition landing
            # after several real TRACKING-LOST frames (imu_frame_scale
            # partially decayed, e.g. 0.25 after 3 lost frames) can disagree
            # with p_pred by a LARGE amount (421mm here) purely because
            # p_pred is itself a multi-frame BLIND coast through that same
            # violent gap -- exactly the scenario _effective_coast_budget_s
            # (see that method's own docstring) already found makes IMU-only
            # dead-reckoning unreliable. The hard gate's own ceiling widens
            # for real violent motion (correctly, to avoid a false reject),
            # but that same widened ceiling ALSO widens THIS ramp's span
            # (_agreement_weak_bounds clips pos_weak/rot_weak to it) -- so a
            # large, real disagreement can land mid-ramp (agreement_pos=
            # 0.4847 in the real case) and pull the accepted state ~52% of
            # the way back toward a p_pred nobody should trust that much
            # right now, corrupting the anchor for several subsequent real
            # (and CORRECT) vision frames, which then get hard-rejected as
            # "implausible" against that now-wrong anchor. Without this
            # scaling, imu_frame_scale only ever gated w_imu in the cost
            # blend above (already fully decayed there for cost_weight_imu=0
            # configs) -- this ramp had no equivalent decay at all, just a
            # binary imu_frame_scale>0.0 gate. Scaling the pushback itself
            # (not agreement_pos/_rot directly, so the ramp's own shape/
            # velocity_established/_rot_seed_untrustworthy overrides below
            # are unaffected -- 1.0 stays 1.0 either way) means a multi-frame
            # lost stretch now correctly leans toward trusting vision fully,
            # same direction the cost blend already leans.
            #
            # ALSO scaled by (1-quality) (2026-09-14, user-directed follow-up
            # on the same real case): imu_frame_scale alone still left a
            # real, strong candidate (right_controller, this same ts --
            # n_inliers=14, error=0.27px, matched across 3 cameras) with a
            # visible ~14% pull toward the bad p_pred (agreement=0.86, not
            # 1.0) purely because imu_frame_scale hadn't fully decayed yet
            # (3 of 4 imu_decay_frames elapsed). But _agreement_factor's own
            # ramp is blind to vision QUALITY entirely -- it only weighs
            # disagreement magnitude against the (already accel/gyro-
            # widened) ceiling, never how many inliers/how clean the
            # reprojection actually was, unlike w_vision's own quality ramp
            # above. A 14-inlier/3-camera candidate is about as strong
            # evidence as this pipeline produces that VISION is right and
            # p_pred (a multi-frame blind coast) is what's wrong -- pulling
            # it toward p_pred at all under-weights that evidence. `quality`
            # (already computed above from the SAME inlier/error ramps
            # w_vision itself uses) folds into the same pushback-scale
            # multiplicatively alongside imu_frame_scale: a high-quality
            # candidate now gets little/no pushback regardless of
            # imu_frame_scale, and imu_frame_scale=1.0 (ordinary warm-state
            # smoothing, the mechanism's original purpose) still applies
            # its old strength for a LOW-quality candidate, only tapering
            # off as quality rises -- not a behavior change for the fully
            # warm+high-quality case's own existing tests, since those were
            # calibrated tight enough to land at agreement<1.0 regardless;
            # see CaseAImuFrameScaleDecayTests/CaseAWarmSoftTrustTests' own
            # updated expectations for the exact quality inputs assumed.
            _pushback_scale = imu_frame_scale * (1.0 - quality)
            agreement_pos = 1.0 - _pushback_scale * (1.0 - agreement_pos)
            agreement_rot = 1.0 - _pushback_scale * (1.0 - agreement_rot)
            # Position pushback toward p_pred needs the same
            # velocity_established precondition as the hard gate above (see
            # its own comment) -- p_pred's position has no real basis without
            # a measured velocity, so there is nothing here to soft-pull
            # vision toward. Rotation's own pushback is unaffected in general
            # (gyro-based, no velocity precondition) -- EXCEPT the same
            # _rot_seed_untrustworthy exemption as the hard gate above: no
            # pulling this frame's good vision rotation back toward a
            # gyro-propagated continuation of last frame's coverage-fallback
            # (likely wrong) orientation.
            if not self.velocity_established:
                agreement_pos = 1.0
            if _rot_seed_untrustworthy:
                agreement_rot = 1.0
            if agreement_pos < 1.0:
                p_new = agreement_pos * p_new + (1.0 - agreement_pos) * p_pred
            if agreement_rot < 1.0:
                R_new = R_pred @ Rotation.from_rotvec(
                    agreement_rot * Rotation.from_matrix(R_pred.T @ R_new).as_rotvec()).as_matrix()
        agreement = min(agreement_pos, agreement_rot)  # single scalar kept for debug_snapshot's own field

        if dt_s > 0:
            if _rot_seed_untrustworthy:
                # self.p (pre-update) is still the untrustworthy coverage-
                # fallback seed's own position -- p_new here may be a
                # legitimate correction AWAY from it (exactly what the
                # _rot_seed_untrustworthy exemptions above just let through),
                # but the STEP itself (seed -> corrected) isn't real motion,
                # it's an artifact of catching up to a wrong prior. Computing
                # a velocity from it fabricates a speed with no physical
                # basis. Confirmed on a real case (2026-09-13): a coverage-
                # fallback cold-reacquire's position, corrected one frame
                # later by a genuinely good ~200mm/110deg jump, produced
                # v=(-5.45,-12.33,-12.23) m/s (~18 m/s) from that single
                # step -- extrapolating even briefly from THAT then rejected
                # every real subsequent candidate as "implausible" (429mm,
                # then 1074mm, then 1290mm -- growing each frame, since
                # rejects leave state at the ever-further garbage
                # extrapolation) and visibly moved the reported pose toward
                # the headset. Treated exactly like bootstrap/fail_open/
                # _try_cold_reacquire instead: not established from this
                # step -- the FIRST subsequent real step (now measured
                # between two genuinely corrected positions) establishes a
                # real velocity normally.
                self.v = np.zeros(3)
                self.velocity_established = False
            else:
                self.v = (p_new - self.p) / dt_s
                self.velocity_established = True
        # else: degenerate/duplicate timestamp -- leave v (and velocity_established) unchanged
        # rather than divide by ~0.

        self.R, self.p = R_new, p_new
        # Only decrement _rotation_seed_grace_frames once a candidate that
        # isn't ITSELF coverage_fallback has actually been blended in -- see
        # that field's own __init__ comment for why an unconditional
        # one-shot clear is wrong (a still-degenerate next candidate would
        # consume the exemption without ever correcting self.R), and why a
        # single decrement (not an immediate drop to 0) is ALSO needed: this
        # candidate's own rotation was itself exempted whenever
        # _rot_seed_untrustworthy was true, so it hasn't been vetted either
        # -- one more grace frame lets it settle before becoming a trusted
        # gate reference.
        # high_risk (2026-09-16): a contested-conflict-winner or swap-
        # suspected accept forces grace to 0 immediately -- even if it's
        # ALSO coverage_fallback, and even overriding any residual grace
        # still counting down from an earlier seed -- see _seed_rotation_
        # grace's own docstring for why granting/leaving an exemption on
        # top of that independent "might belong to the wrong controller"
        # flag is unsafe (real case: a contested coverage_fallback accept
        # here let a ~129deg identity-swap jump through the very next
        # frame, unexempted by this fix).
        _high_risk = (bool(solution.get("winner_was_contested", False))
                      or bool(solution.get("swap_suspected", False)))
        if _high_risk:
            self._seed_rotation_grace(bool(solution.get("coverage_fallback", False)), high_risk=True)
        elif solution.get("coverage_fallback", False):
            self._seed_rotation_grace(True)
        elif self._rotation_seed_grace_frames > 0:
            self._rotation_seed_grace_frames -= 1
        self.last_update_ts_ns = frame_ts_ns
        self._last_known_p, self._last_known_p_ts_ns = self.p.copy(), frame_ts_ns
        self._report(frame_ts_ns, self.R, self.p)
        self.frames_since_update = 0

        # Rolling innovation history for _agreement_weak_bounds' adaptive
        # thresholds above -- normal cadence only (see its own docstring / the
        # deque's own docstring in __init__).
        if dt_s <= _GAP_DT_NORMAL_S:
            self._pos_innov_hist.append(pos_innov_m)
            self._rot_innov_hist.append(rot_innov_deg)

        # NOTE: the hard implausibility case (w_vision/inlier_factor/error_factor
        # all zeroed by _vision_weight AND a large innovation) can no longer reach
        # this point -- it's now caught and rejected above, before this cost-
        # function fuse ever runs. _vision_weight's own gate stays in place as a
        # defensive backstop (still correct, just unreachable via this call site
        # given the two thresholds match).
        self._trust_window.append(confidence)
        self._set_last(outcome="fused", pos_innov_m=pos_innov_m, rot_innov_deg=rot_innov_deg,
                        confidence=confidence, alpha_pos=w_vision / w_sum, alpha_rot=w_vision / w_sum,
                        kalman_gain_pos=w_vision / w_sum, kalman_gain_rot=w_vision / w_sum,  # reuse existing debug-viz field names
                        pos_pred=p_pred, R_pred=R_pred, agreement=agreement)
        _log.debug(
            f"[{self._ctrl_name}] FUSED ts={frame_ts_ns} dt={dt_s * 1000:.1f}ms conf={confidence:.2f} | "
            f"match: n_inliers={n_inliers} error={error_px:.2f}px "
            f"(inlier_factor={inlier_factor:.2f} error_factor={error_factor:.2f}) | "
            f"weights imu={w_imu:.2f} vision={w_vision:.2f} (vision share={w_vision / w_sum:.2f}, "
            f"quality={quality:.2f}, gap_scale vision={vision_gap_scale:.2f}, "
            f"frames_lost={_frames_lost} imu_frame_scale={imu_frame_scale:.2f} vision_only={vision_only}) | "
            f"agreement={agreement:.2f} | "
            f"pos: imu={_fmt_v(p_pred)} vision={_fmt_v(p_meas)} -> tracking={_fmt_v(p_new)} "
            f"reported={_fmt_v(self.reported_p)} | "
            f"innov: pos={pos_innov_m * 1000:.1f}mm rot={rot_innov_deg:.2f}deg | "
            f"trust={self.trust_score():.2f}"
        )
        return True

    # ------------------------------------------------------------------
    # Cold-state (vision_only) reacquisition
    # ------------------------------------------------------------------
    def _report_coast_if_usable(self, frame_ts_ns: int, R_pred: np.ndarray, p_pred: np.ndarray,
                                 dt_s: float) -> None:
        """Display the coasted IMU prediction after a hard-rejected vision
        candidate, but only while the loss so far is short enough to trust for
        display -- otherwise clear the reported pose, exactly like
        _try_cold_reacquire's _report_if_still_usable does for a buffered
        candidate.

        2026-09-20, real case: static_medium, left_controller, relative frame 85
        (frame_idx 4085): frames 75-84 correctly showed no pose (loss past
        cold_pending_report_max_gap_s), then a hard-rejected vision candidate at
        189ms since the last accept re-displayed the stale IMU prediction --
        243mm off mocap -- because both reject branches called _report()
        unconditionally. Their veto (cold_reacquire_rot_veto_max_s, 0.25s base)
        is deliberately more generous than the DISPLAY budget, so a candidate
        can be legitimately rejected while the coast is already too old to show.
        reject_report_respect_display_budget: false restores the old behaviour."""
        if not bool(self._hc_get("reject_report_respect_display_budget", True)):
            self._report(frame_ts_ns, R_pred, p_pred)
            return
        base = float(self._hc_get("cold_pending_report_max_gap_s", 0.25))
        budget = self._effective_coast_budget_s(base, frame_ts_ns, rate_prefix="cold_pending")
        if dt_s <= budget:
            self._report(frame_ts_ns, R_pred, p_pred)
        else:
            self.reported_R = self.reported_p = None

    def _check_reject_streak_override(self, R_meas, p_meas, frame_ts_ns, confidence, solution,
                                       n_inliers, error_px, R_pred, p_pred, dt_s, rot_innov_deg,
                                       pos_innov_m, accel_now) -> bool:
        """Called at each hard-reject site (candidate disagrees with the IMU/gyro
        prediction). Returns True if it re-anchored on this candidate instead.

        reject_streak_override (2026-09-19, real case: static_medium, right_
        controller, frames 4720-4727): a cold reacquire accepted at frame 4720
        was a mirror flip (17mm, 157.7deg off mocap). The IMU/gyro state then
        coasted from that wrong anchor, and three consecutive STRONG vision
        solutions (11 inliers, 0.10px, correct orientation) were each rejected
        as "implausible vs still-credible gyro prediction" (rot_innov=155deg) --
        the disagreement WAS the flip, i.e. the prediction was wrong, not
        vision. Two strong candidates in a row that are rejected against the
        prediction but AGREE WITH EACH OTHER are much better evidence than one
        wrong prediction, so the second one re-anchors (through
        _try_cold_reacquire's strong path, which still runs its sibling-
        collision check first). Deliberately requires STRONG candidates
        (n_inliers >= vision_weight_strong_inliers, error below the weak floor,
        not coverage_fallback/contested/swap_suspected): weak solves sharing one
        systematic error also "agree" with each other, see _try_cold_reacquire.
        State resets on any accept in between (last_update_ts_ns changes)."""
        if not bool(self._hc_get("reject_streak_override_enabled", True)):
            return False
        strong_inliers = float(self._hc_get("vision_weight_strong_inliers", 18))
        weak_error_px = float(self._hc_get("vision_weight_weak_error_px", 0.5))
        strong = (n_inliers >= strong_inliers and error_px < weak_error_px
                  and not solution.get("coverage_fallback", False)
                  and not solution.get("winner_was_contested", False)
                  and not solution.get("swap_suspected", False))
        prev = self._reject_streak_prev
        if prev is not None and prev["last_update_ts_ns"] != self.last_update_ts_ns:
            prev = None
            self._reject_streak_n = 0
        if not strong:
            self._reject_streak_prev = None
            self._reject_streak_n = 0
            return False
        this = {"R": R_meas, "p": p_meas, "frame_ts_ns": frame_ts_ns, "last_update_ts_ns": self.last_update_ts_ns}
        self._reject_streak_prev = this
        if prev is None:
            self._reject_streak_n = 1
            return False
        gap_s = (frame_ts_ns - prev["frame_ts_ns"]) / 1e9
        if gap_s <= 0.0 or gap_s > float(self._hc_get("reject_streak_override_max_gap_s", 0.25)):
            self._reject_streak_n = 1
            return False
        dpos_m = float(np.linalg.norm(p_meas - prev["p"]))
        drot_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(prev["R"].T @ R_meas).as_rotvec())))
        jump_pos_m = (float(self._hc_get("weak_confirm_pos_thresh_base_m", 0.135))
                      + float(self._hc_get("weak_confirm_max_speed_m_s", 3.0)) * gap_s)
        jump_rot_deg = (float(self._hc_get("weak_confirm_rot_thresh_base_deg", 15.6))
                        + float(self._hc_get("weak_confirm_max_ang_speed_deg_s", 2000.0)) * gap_s)
        if dpos_m > jump_pos_m or drot_deg > jump_rot_deg:
            self._reject_streak_n = 1
            return False
        self._reject_streak_n += 1
        if self._reject_streak_n < int(self._hc_get("reject_streak_override_min_agreeing", 2)):
            return False
        _log.info(
            f"[{self._ctrl_name}] REJECT-STREAK OVERRIDE ts={frame_ts_ns} — {self._reject_streak_n} consecutive strong "
            f"candidates (n_inliers={n_inliers}, err={error_px:.2f}px) disagree with the IMU prediction "
            f"(pos_innov={pos_innov_m * 1000:.0f}mm rot_innov={rot_innov_deg:.0f}deg) but agree with each other "
            f"(dpos={dpos_m * 1000:.1f}mm drot={drot_deg:.2f}deg) -- treating the prediction/anchor as wrong, re-anchoring on vision"
        )
        self._reject_streak_prev = None
        self._reject_streak_n = 0
        return self._try_cold_reacquire(
            R_meas, p_meas, frame_ts_ns, confidence, coverage_fallback=False, n_inliers=n_inliers,
            error_px=error_px, R_pred=R_pred, p_pred=p_pred, dt_s=dt_s, rot_pred_implausible=False,
            accel_now=accel_now, accept_outcome="reject_streak_override")

    def _try_cold_reacquire(self, R_meas: np.ndarray, p_meas: np.ndarray,
                             frame_ts_ns: int, confidence: float,
                             coverage_fallback: bool = False, n_inliers: int = 0,
                             error_px: float = 0.0,
                             winner_was_contested: bool = False,
                             swap_suspected: bool = False,
                             rot_pred_implausible: bool = False,
                             accel_now: np.ndarray | None = None,
                             R_pred: np.ndarray | None = None, p_pred: np.ndarray | None = None,
                             dt_s: float | None = None, log_prefix: str = "COLD REACQUIRE",
                             pending_outcome: str = "cold_pending",
                             accept_outcome: str = "cold_reacquired") -> bool:
        """ALSO reused directly by try_update's bootstrap branch (self.R is
        None) -- see that branch's own comment. log_prefix/pending_outcome/
        accept_outcome let that caller get bootstrap-appropriate log text and
        _set_last outcome strings instead of "COLD REACQUIRE"/"cold_pending"/
        "cold_reacquired", which would be confusing for what's actually a
        first-ever detection, not a reacquisition after a real loss. Every
        comment below describing "cold reacquire" behavior applies equally
        to a bootstrap call -- there's no state-dependent branching here on
        which caller this is, only the label strings differ.

        Past imu_decay_frames with no accepted vision, p_pred is no longer a
        trustworthy reference -- this is exactly why _vision_weight's
        gate_active=False skips the hard implausibility gate entirely in this
        regime: vision is trusted directly here, the same as everywhere else
        in this filter (cost_weight_imu=0 in this project's actual config
        already makes the normal warm blend pure-vision too -- this is that
        same trust extended to the reacquisition case, not a separate,
        more cautious policy) -- EXCEPT for a single carve-out below for
        genuinely WEAK candidates.

        Sibling-collision (a genuine identity-swap defense -- an independent
        signal, unrelated to motion prediction) is checked first, same as
        always, regardless of weak/strong routing below.

        REMOVED (explicit direction, 2026-09-06): a 3-frame confirmation
        window that held even a STRONG reacquisition candidate "pending" for
        2 extra frames before trusting it as the new anchor. Reasoning for
        removal: frame1->frame2 of that window never had a velocity
        reference to check against anyway (the one gap the window could
        never actually close, called out explicitly when it was built) --
        and this filter already isn't the only thing standing between a cold
        controller and a search retry: ControllerTracker._mark_all_lost's own
        IMU-only-propagation budget keeps prev_pose/pose_history warm for
        several frames past a single loss, so even "just lost 1 frame ago"
        already has a real prior to warm-start a normal per-camera search
        from, without needing this filter to ALSO run its own slower,
        parallel confirmation gate on top of that. A STRONG candidate here
        still gets none of that -- trusted immediately, exactly as removed.

        ADDED (2026-09-13, user-proposed): a narrower, WEAK-only version of
        that same idea. A "weak" candidate -- coverage_fallback (too few of
        the geometrically-expected LEDs matched, pose_search.py's own
        confidence=0.0 marker) or n_inliers at/below
        vision_weight_weak_inliers (code fallback default 6, this project's
        own config.yml sets 5 -- the same "this match's quality ramp is
        already at its floor" boundary _vision_weight uses) -- is
        exactly the shape of candidate the removed window's own reasoning
        doesn't cover: unlike a strong match, a weak one very often comes
        from a controller sitting at the edge of camera view with only 4-5
        LEDs physically visible, where a degenerate low-point-count P3P fit
        can land a deceptively plausible-looking but wrong pose (the same
        failure class _vision_weight's own docstring documents for the warm
        path). Trusting THAT immediately as the new anchor, unconditionally,
        is what let a real bad reacquisition stand while the very next
        frame's strong, correct match then read as a huge "jump" against it
        and paid a rejection/grace-window cost a cleaner anchor would never
        have needed (confirmed on this recording: frames 64->65 and,
        downstream of that same shape, 66->67).

        So: a weak candidate is buffered in self._cold_pending instead of
        being accepted -- NOT reported, NOT trusted, this frame returns
        False exactly like a genuine reject, but WITHOUT touching
        consecutive_rejects (this is a deliberate wait, not a bad-match
        signal; letting it accumulate risks should_force_cold_start firing on
        a normal multi-frame edge-of-view reacquisition and discarding a
        perfectly fine one) -- and is resolved on the NEXT call:
          - a STRONG candidate now arrives -> the buffered weak one is
            discarded outright, strong is trusted immediately (same as
            always).
          - another WEAK candidate arrives and agrees with the buffered one
            (within implausible_jump_pos_m/rot_deg -- the exact same
            "is this a real single-frame motion or a jump" thresholds used
            everywhere else in this filter) -> two independent weak solves
            landing in the same place IS real corroborating signal a single
            weak solve doesn't have on its own; accept this (newer) one as
            the anchor, discard the buffered one (it was only ever needed to
            corroborate, not to itself become the anchor -- using the newer
            frame keeps last_update_ts_ns/rotation-grace seeding keyed to
            the candidate actually being trusted).
          - another WEAK candidate arrives and DISAGREES (a jump) -> the
            first one was apparently noise; discard it entirely and buffer
            this newer one instead, continuing to wait rather than compound
            two mismatched weak guesses into anything trusted.
        A buffered candidate older than cold_confirm_max_gap_s (default
        0.25s) is dropped as stale before any of the above, so an old weak
        guess can't sit around indefinitely waiting to corroborate against
        an arbitrarily later, unrelated candidate.

        THREE REFINEMENTS (2026-09-13):

        1) (second pass, after a critical review) The weak-pair AGREE/
        DISAGREE check above no longer uses a flat floor alone -- two
        buffered weak candidates can legitimately be up to
        cold_confirm_max_gap_s (0.25s) apart, ~10-20x this recording's own
        normal ~11-22ms frame cadence a flat threshold would be calibrated
        against. A fast-moving hand can cover real distance in 0.25s without
        it being a bad candidate -- without scaling, two genuinely-correct
        weak solves could spuriously fail to "agree," discard-and-rebuffer
        repeatedly, and stall recovery for up to max_coast_s (1.0s) instead
        of the ~2 frames this whole mechanism is meant to take.

        1b) (third pass, user-directed, same day) Both axes now use
        weak_confirm_pos/rot_thresh_base + weak_confirm_max_speed_m_s/
        _ang_speed_deg_s * confirm_dt_s -- the SAME "how far could a human
        hand plausibly have moved in this much real time" formula as
        controller.py's own vs-prev_pose (check 1) and vs-last_good_pose
        (check 3) checks, ported here as a fusion_heuristic-scoped mirror
        of those matching.* values (this class can't reach the matching:
        config block directly -- see this check's own inline comment).
        Previously position alone was scaled (max(implausible_jump_pos_m,
        cold_confirm_max_speed_m_s * confirm_dt_s), rotation left flat at
        implausible_jump_rot_deg=60deg) -- both axes now scale, and the OLD
        ungrounded cold_confirm_max_speed_m_s=5.0 constant is retired in
        favor of the same shared, research-grounded rate used everywhere
        else this question gets asked (see matching.max_plausible_hand_
        speed_m_s/_ang_speed_deg_s's own config.yml comment for the full
        derivation, including the human hand-speed research it's grounded
        in). One formula for both the bootstrap and genuine-cold-reacquire
        subcases -- see this check's own inline comment for why no
        log_prefix-based split is needed once checks 1 and 3 already share
        this exact shape themselves.

        2) A buffered ("cold_pending") frame used to never call self._report,
        so reported_p/reported_R (what's actually displayed/logged/CSV'd)
        silently froze at their pre-loss values for as long as the buffer
        kept getting refilled -- even though predict() had already computed
        a live, real IMU-coasted (R_pred, p_pred) for this exact frame one
        call earlier in try_update, simply discarded. Now reported if the
        loss so far (dt_s, elapsed time since self.last_update_ts_ns -- the
        LAST REAL ANCHOR before this whole loss, NOT the gap between the two
        buffered candidates that (1) above uses) is still short enough
        (cold_pending_report_max_gap_s, default 0.25s) that the coasted
        estimate is credible to show; past that, the loss has been going on
        long enough that dead-reckoning from a stale anchor is no longer
        usable even just to display.

        2b) (2026-09-13, second refinement, real case) Past that same
        report_max_gap_s, reported_R/reported_p are now explicitly CLEARED
        to None -- not left "frozen, unchanged" as (2) above originally
        shipped. A real trace found this exact gap: a controller lost real
        tracking, coasted via ControllerTracker's OWN separate imu_only_
        predicted_pose mechanism for a few frames (a different code path,
        matching.imu_only_propagation_max_s), then that budget ALSO ran out
        and the controller was correctly hidden -- but reported_R/reported_p
        here were never touched during any of that (try_update is only
        called when a vision candidate exists; a pure "nothing found"
        stretch never calls it at all), so they stayed at their pre-loss
        value. When a genuinely weak candidate finally arrived several
        hundred ms later, past report_max_gap_s, _report_if_still_usable
        correctly declined to report the fresh coast -- but the STALE
        pre-loss pose was still sitting there un-cleared, so
        ControllerTracker/main.py read it as "a real pose exists" and drew
        the 3D model at a position from well before the loss even started,
        with nothing to indicate it was stale. Clearing here instead makes
        main.py's own existing None-check ("T_world_ctrl is None -> nothing
        to report this frame") do the right thing automatically -- no
        change needed on that side. Same "short lost track, IMU still
        usable" vs. "long loss, IMU no longer usable" distinction this
        project already draws elsewhere for a different mechanism (see
        matching.imu_only_propagation_max_s, ControllerTracker._mark_all_
        lost). 0.25s was sized against a real case: the recording's own
        "64->65" event (see above) had dt_s ~= 0.166s at exactly this point
        -- a tighter default (e.g. 0.15s, matching this file's existing
        gap_dt_full_s) would have just missed reporting on it."""
        if self._looks_like_a_sibling(p_meas, frame_ts_ns):
            self.consecutive_rejects += 1
            self._set_last(outcome="sibling_rejected", confidence=confidence)
            _log.info(
                f"[{self._ctrl_name}] {log_prefix} REJECTED ts={frame_ts_ns} pos={_fmt_v(p_meas)} "
                f"— collides with a live sibling controller's position; likely identity swap"
            )
            return False

        pending = self._cold_pending
        if pending is not None:
            max_age_s = float(self._hc_get("cold_confirm_max_gap_s", 0.25))
            if (frame_ts_ns - pending["frame_ts_ns"]) / 1e9 > max_age_s:
                _log.info(
                    f"[{self._ctrl_name}] {log_prefix}: discarding stale buffered weak candidate "
                    f"from ts={pending['frame_ts_ns']} pos={_fmt_v(pending['p'])} (older than "
                    f"cold_confirm_max_gap_s={max_age_s:.2f}s)"
                )
                pending = None
                self._cold_pending = None

        def _report_if_still_usable() -> None:
            """(2)/(2b) above: shows the coasted IMU prediction while
            buffering, but only if the loss so far is still short enough to
            trust for display -- see this method's own docstring. Past that
            budget, explicitly CLEARS reported_R/reported_p (2b) rather than
            leaving them at whatever stale value they last held -- R_pred/
            p_pred/dt_s all being real here (unlike the bootstrap call,
            which passes None for all three and returns above before ever
            reaching this) means this IS a genuine cold-reacquire with a
            real prior anchor, so there's something to clear."""
            if R_pred is None or p_pred is None or dt_s is None:
                return
            base_report_max_gap_s = float(self._hc_get("cold_pending_report_max_gap_s", 0.25))
            report_max_gap_s = self._effective_coast_budget_s(base_report_max_gap_s, frame_ts_ns,
                                                                rate_prefix="cold_pending")
            if dt_s <= report_max_gap_s:
                self._report(frame_ts_ns, R_pred, p_pred)
            else:
                self.reported_R = self.reported_p = None

        weak_inliers = float(self._hc_get("vision_weight_weak_inliers", 6))
        # weak_error_px (2026-09-18, real case: right_controller, static_easy,
        # frame_idx 6469, ts=100852465980200): this weak/strong split used to
        # check n_inliers ONLY -- a single-camera p3p_systematic solve with
        # n_inliers=6 (just above weak_inliers=5) and error_px=1.12
        # (WORSE than this same controller's own vision_weight_weak_error_px
        # override of 1.1 -- i.e. already at the error ramp's floor by
        # _vision_weight's own standard) counted as "strong enough," skipped
        # the weak buffer/confirm step entirely, and got trusted immediately
        # as the new anchor. Mocap-confirmed: that reacquisition was 44.2mm/
        # 141.2deg off truth, and every frame it anchored while coasting
        # (implausible_reject keeps the reported pose pinned to the bad
        # p_pred) drifted further (up to 210mm/142deg) before a forced
        # cold-start finally recovered ~10 frames later. Symmetric with the
        # n_inliers check above (error_px >= weak_error_px is exactly
        # _ramp_down's own "at/above weak" floor, see that function's own
        # docstring) -- a candidate this bad on EITHER axis alone gets
        # buffered, matching _vision_weight's "clean-but-wrong" reasoning
        # for why the normal warm path never trusts either axis in
        # isolation either.
        weak_error_px = float(self._hc_get("vision_weight_weak_error_px", 0.5))
        # winner_was_contested (2026-09-13): a THIRD trigger for "weak,"
        # alongside coverage_fallback/low-inlier-count -- see
        # TrackingSystem._resolve_cold_conflicts' own contested_winners
        # docstring for the real case (a 6-inlier/0.32px bootstrap, an
        # otherwise perfectly ordinary-looking confidence/error/inlier
        # profile, that won a shared-blob conflict with a physically-close
        # sibling controller and turned out 132.7deg/1.63m wrong vs mocap
        # ground truth). Neither coverage_fallback nor n_inliers caught this
        # one -- the solve itself looked fine in isolation; only the fact
        # that another controller's candidate contested the same evidence
        # this frame flagged it. Routes through the exact same weak-pair
        # buffer/confirm machinery below, no separate handling needed.
        #
        # EXEMPTED (2026-09-15, user-directed, real case: left_controller,
        # walk_hard, frame 29) if the winner is GENUINELY strong -- not just
        # "not weak" (n_inliers > weak_inliers), but clearing the SAME
        # strong_inliers/strong_error_px bar _vision_weight itself uses for
        # "full trust, no quality discount at all". Real case: an 11-inlier/
        # 0.11px winner (well past strong_inliers=8/strong_error_px=0.15)
        # won a physical-overlap conflict (1.5cm apart) and got buffered for
        # a full extra frame despite being correct (mocap: 21.5mm/5.4deg off
        # truth) -- an unnecessary display-lag cost for a candidate this
        # solid. Does NOT reopen the original 132.7deg bug above: that
        # bootstrap's own n_inliers=6 is BELOW strong_inliers (8 real-config/
        # 18 code-default) either way, so it would still be forced weak and
        # buffered under this exemption, unchanged. `_looks_like_a_sibling`
        # (this method's own very first check, run unconditionally before
        # this weak/strong routing) is a separate, position-based defense
        # against the identity-swap risk a "clean solve, wrong controller"
        # case would pose -- this exemption doesn't bypass or weaken it.
        strong_inliers = float(self._hc_get("vision_weight_strong_inliers", 18))
        strong_error_px = float(self._hc_get("vision_weight_strong_error_px", 0.15))
        _contested_but_strong = winner_was_contested and n_inliers >= strong_inliers and error_px <= strong_error_px
        # swap_suspected (2026-09-16): a FOURTH trigger, alongside coverage_
        # fallback/low-inlier-count/contested -- see TrackingSystem.
        # _detect_cold_identity_swap's own docstring for the real case (both
        # controllers simultaneously cold, each landing on the OTHER's own
        # last-known position instead of their own). Deliberately NOT given
        # the same strong-exemption _contested_but_strong gets above -- a
        # cross-controller swap can look individually clean (high inliers,
        # low error) on BOTH sides at once, since each side really is a
        # correct geometric solve, just assigned to the wrong controller;
        # letting a "strong" swap bypass buffering would defeat the whole
        # point of this check. Always forces weak/buffered, no exemption.
        #
        # rot_pred_implausible (2026-09-16): a FIFTH trigger -- see
        # try_update's own comment on this signal for the real case (two
        # single-camera weak solves that agreed with EACH OTHER but were
        # both 158deg off a still-credible gyro prediction). Same no-
        # exemption treatment as swap_suspected -- a candidate that
        # individually looks strong but contradicts an independent gyro
        # reference shouldn't bypass buffering via the strong fast-path
        # below. NOTE: forcing weak=True here is necessary but not
        # sufficient for the real bug this closes -- see the CONFIRM check
        # further down, which is the part that actually matters (two
        # candidates sharing the SAME systematic error still "agree" with
        # each other regardless of each one's own weak/strong status).
        weak = (coverage_fallback or n_inliers <= weak_inliers or error_px >= weak_error_px
                or (winner_was_contested and not _contested_but_strong)
                or swap_suspected or rot_pred_implausible)

        # bootstrap_gravity_veto (2026-09-19): the accelerometer reading rotated
        # through this candidate's R should point along world -y (down) -- see
        # the veto below for the real cases and numbers. None when there is no
        # accel sample to check against (veto then simply doesn't apply).
        _bootstrap_gravity_angle_deg = None
        if log_prefix == "BOOTSTRAP" and accel_now is not None:
            _a = np.asarray(accel_now, dtype=float)
            _a_norm = float(np.linalg.norm(_a))
            # Quiet-motion gate (2026-09-20, real case: static_medium, right_controller,
            # frames 52-56): during fast motion the accelerometer reads 13-20 m/s^2
            # (linear acceleration on top of gravity), so its direction is not
            # gravity -- correct poses there sit 84-98deg from "down" and a flipped one
            # 79deg. The check only means something while |a| is close to g; with
            # |a| within tol of 9.81 the worst-case direction error from the
            # perpendicular linear acceleration stays well under the 65deg threshold.
            _quiet = abs(_a_norm - 9.81) <= float(self._hc_get("bootstrap_gravity_veto_accel_tol_ms2", 1.5))
            if _a_norm > 1e-6 and _quiet:
                _g_world = R_meas @ (_a / _a_norm)
                _bootstrap_gravity_angle_deg = float(np.degrees(np.arccos(np.clip(-_g_world[1], -1.0, 1.0))))
        # Thin-candidate check (2026-09-20, real case: static_medium, left_
        # controller, relative frame 42 -> 47): a 2-inlier prior-constrained
        # P1P solve (rotation COPIED from the prediction, only translation
        # solved) was buffered as the "weak candidate", and a poor 5-inlier/
        # 0.69px P3P candidate five frames later "agreed" with it within a
        # 476mm/~190deg confirm budget (dt=89ms) -- 65mm/58deg off mocap, and
        # it became the anchor. A candidate with fewer than
        # cold_confirm_min_inliers inliers, or whose rotation came from the
        # prediction, carries no independent orientation evidence, so it is
        # neither buffered nor allowed to confirm/supersede anything; the
        # frame is treated as a wait (no reject counted), like any other
        # pending frame.
        _min_evidence = int(self._hc_get("cold_confirm_min_inliers", 3))
        _thin = ((bool(self._hc_get("cold_confirm_ignore_prior_constrained", True))
                  and self._cand_rotation_from_prior)
                 or n_inliers < _min_evidence)
        if _thin:
            _log.info(
                f"[{self._ctrl_name}] {log_prefix} ts={frame_ts_ns} pos={_fmt_v(p_meas)} n_inliers={n_inliers} "
                f"rotation_from_prior={self._cand_rotation_from_prior} — too thin to count as evidence "
                f"(min {_min_evidence} inliers, no prior-copied rotation), ignored"
            )
            _report_if_still_usable()
            self._set_last(outcome=pending_outcome, confidence=confidence, pos_pred=p_pred, R_pred=R_pred)
            return False

        if not weak:
            if pending is not None:
                _log.info(
                    f"[{self._ctrl_name}] {log_prefix}: strong match ts={frame_ts_ns} pos={_fmt_v(p_meas)} "
                    f"supersedes buffered weak candidate from ts={pending['frame_ts_ns']} "
                    f"pos={_fmt_v(pending['p'])} — discarding the weak one, trusting strong immediately"
                )
                self._cold_pending = None
            return self._accept_cold_reacquire(R_meas, p_meas, frame_ts_ns, confidence, coverage_fallback,
                                                high_risk=winner_was_contested or swap_suspected,
                                                log_prefix=log_prefix, accept_outcome=accept_outcome)

        if pending is None:
            self._cold_pending = {
                "R": R_meas, "p": p_meas, "confidence": confidence,
                "coverage_fallback": coverage_fallback, "frame_ts_ns": frame_ts_ns,
                "n_inliers": n_inliers, "accel": accel_now, "error_px": error_px,
            }
            _report_if_still_usable()
            self._set_last(outcome=pending_outcome, confidence=confidence, pos_pred=p_pred, R_pred=R_pred)
            _log.info(
                f"[{self._ctrl_name}] {log_prefix} PENDING ts={frame_ts_ns} pos={_fmt_v(p_meas)} "
                f"n_inliers={n_inliers} coverage_fallback={coverage_fallback} "
                f"winner_was_contested={winner_was_contested} swap_suspected={swap_suspected} "
                f"rot_pred_implausible={rot_pred_implausible} — "
                f"weak candidate, buffering for confirmation instead of trusting immediately"
            )
            return False

        dpos_m = float(np.linalg.norm(p_meas - pending["p"]))
        drot_deg = float(np.degrees(np.linalg.norm(
            Rotation.from_matrix(pending["R"].T @ R_meas).as_rotvec())))
        confirm_dt_s = (frame_ts_ns - pending["frame_ts_ns"]) / 1e9
        # weak_confirm_*: same "how far could a human hand plausibly have
        # moved in this much real time" formula as checks 1 (vs prev_pose)
        # and 3 (vs last_good_pose, matching.max_plausible_hand_speed_m_s/
        # _ang_speed_deg_s) in controller.py -- ported here as its own
        # fusion_heuristic-scoped mirror (2026-09-13, second-pass review)
        # rather than the old flat implausible_jump_pos_m/_rot_deg (+ an
        # ungrounded one-off cold_confirm_max_speed_m_s=5.0, now retired)
        # this used before. Not reachable directly from this class (self._hc
        # only sees the fusion_heuristic: config block, not matching:), so
        # these are a deliberate duplicate of the SAME values, not a
        # cross-section config read -- keep them in sync with
        # matching.pose_jump_pos_thresh_base_m/_rot_thresh_base_deg and
        # max_plausible_hand_speed_m_s/_ang_speed_deg_s if those ever change.
        # One formula for BOTH the bootstrap and genuine-cold-reacquire
        # subcases (no log_prefix-based branch needed): now that checks 1
        # and 3 share this exact same base+rate shape themselves, there's no
        # remaining difference between "what a fresh bootstrap pair should
        # agree within" and "what a post-loss reacquisition pair should
        # agree within" left to encode separately.
        _weak_confirm_pos_base_m = float(self._hc_get("weak_confirm_pos_thresh_base_m", 0.135))
        _weak_confirm_rot_base_deg = float(self._hc_get("weak_confirm_rot_thresh_base_deg", 15.6))
        _weak_confirm_speed_m_s = float(self._hc_get("weak_confirm_max_speed_m_s", 3.0))
        _weak_confirm_ang_speed_deg_s = float(self._hc_get("weak_confirm_max_ang_speed_deg_s", 2000.0))
        jump_pos_m = _weak_confirm_pos_base_m + _weak_confirm_speed_m_s * confirm_dt_s
        jump_rot_deg = _weak_confirm_rot_base_deg + _weak_confirm_ang_speed_deg_s * confirm_dt_s
        _agrees_with_pending = dpos_m <= jump_pos_m and drot_deg <= jump_rot_deg

        # REVERTED 2026-09-18 (gravity_implausible veto, shipped same day):
        # its justification -- "0% false-positive rate over 36 real firings"
        # of controller.py's gravity-direction diagnostic -- was itself
        # measured with a broken (bridge-uncomposed) mocap comparison. Every
        # frame, good or bad, shows ~178deg/~80mm "error" against mocap
        # without composing the controller's own mocap_bridge_path transform
        # first (T_est.compose(bridge).inverse().compose(T_gt), see
        # evaluate_mocap.py's own verified formula) -- that's the bridge's
        # own fixed LED-frame-to-mocap-marker-frame rotation/lever-arm
        # showing up as a false "error", not genuine candidate correctness.
        # Redone properly: only 11/36 (31%) of those firings were actually
        # bad poses; 25/36 (69%) were false positives, including several at
        # HIGH disagreement (76-153deg) sitting right next to genuine true
        # positives at similar magnitudes -- gravity disagreement magnitude
        # alone does not reliably separate good from bad here at any
        # threshold (likely needs an additional low-ANGULAR-VELOCITY gate,
        # not just the existing low-LINEAR-acceleration one, since a
        # genuinely fast rotation can pass the |accel|~9.81 gate while still
        # producing a large, legitimate gravity-direction swing). Left
        # accel_now/pending["accel"] threading in place (still used by the
        # separate along-track/cross-track diagnostic below try_update)
        # rather than ripping out the whole plumbing -- only the veto
        # decision itself is removed pending a properly re-validated design.
        # bootstrap_coverage_fallback_veto (2026-09-19, real case: static_easy,
        # left_controller frame_idx 2803/3651, right_controller frame_idx
        # 2134): coverage_fallback means too few of the geometrically-expected
        # LEDs matched -- a degenerate, near-planar point set that P3P can fit
        # to its own mirror-branch alternate just as "cleanly" as the true
        # pose. Two such candidates sharing that SAME systematic bias agree
        # with each other (this is exactly rot_pred_implausible's own real
        # case above, just with no gyro reference yet to catch it -- at
        # BOOTSTRAP there is no prior R to integrate gyro from, so that veto
        # structurally cannot run here). Empirically checked across this
        # recording's full 42 bootstraps (21 per controller): EVERY
        # coverage_fallback=True bootstrap that turned out bad (3 of 4, 75%)
        # was this exact shape; EVERY coverage_fallback=False bootstrap (38
        # of 38, 100%) was good. Scoped to BOOTSTRAP only (not mid-recording
        # COLD REACQUIRE) -- a fresh bootstrap has no other context (no
        # recent last-known-good pose, no IMU coast) to lean on if this
        # candidate is refused, unlike a cold-reacquire, so it is worth
        # being pickier specifically here; refusing costs only the 1 good
        # coverage_fallback bootstrap found (delayed until a non-fallback
        # candidate arrives), against eliminating all 3 real failures.
        # cold_confirm_require_solid_member (2026-09-20, same real case as the
        # thin-candidate check above): two weak candidates that BOTH fit poorly
        # (error_px >= vision_weight_weak_error_px) agree with each other about
        # as easily as two random guesses under the time-scaled budget, so a
        # confirmed pair needs at least one member with a clean fit.
        _pending_err = float(pending.get("error_px", 0.0))
        _no_solid_member = (bool(self._hc_get("cold_confirm_require_solid_member", True))
                            and error_px >= weak_error_px and _pending_err >= weak_error_px)
        _bootstrap_coverage_fallback_veto = log_prefix == "BOOTSTRAP" and coverage_fallback
        # bootstrap_gravity_veto (2026-09-19, real case: static_medium, left_
        # controller, frame_idx 2988: a rotation-flipped bootstrap, 7.5mm/155.9deg
        # off mocap, confirmed because two weak candidates agreed with each other
        # -- coverage_fallback was False, so the veto above didn't apply). Two
        # weak solves sharing one mirror-branch error agree with each other, but
        # a flipped orientation also flips where "down" points: at rest the
        # accelerometer reading, rotated through a correct R, lands on world -y
        # (static_easy/walk_easy first frames: (0.09,-0.97,-0.22) / (0.03,-1.00,
        # -0.03) for both controllers). Checked on every bootstrap candidate of a
        # full static_easy run (80 candidates, labelled vs mocap): good ones sit
        # 0-51deg from -y (median 15), 19 of 22 flipped ones sit >=65deg;
        # thresh 65 blocks 0/58 good. Bootstrap-only: no gyro prediction exists
        # there for rot_pred_implausible to use. Cannot see a flip about the
        # vertical axis (gravity unchanged by it).
        _grav_veto_thresh_deg = float(self._hc_get("bootstrap_gravity_veto_thresh_deg", 65.0))
        _bootstrap_gravity_veto = (_grav_veto_thresh_deg > 0.0 and _bootstrap_gravity_angle_deg is not None
                                   and _bootstrap_gravity_angle_deg > _grav_veto_thresh_deg)
        if _agrees_with_pending and not rot_pred_implausible and not _bootstrap_coverage_fallback_veto \
                and not _bootstrap_gravity_veto and not _no_solid_member:
            self._cold_pending = None
            _log.info(
                f"[{self._ctrl_name}] {log_prefix} CONFIRMED ts={frame_ts_ns} pos={_fmt_v(p_meas)} — "
                f"agrees with buffered weak candidate from ts={pending['frame_ts_ns']} "
                f"(dpos={dpos_m * 1000:.1f}mm drot={drot_deg:.2f}deg vs. budget "
                f"{jump_pos_m * 1000:.1f}mm over confirm_dt_s={confirm_dt_s * 1000:.1f}ms), accepting"
            )
            return self._accept_cold_reacquire(R_meas, p_meas, frame_ts_ns, confidence, coverage_fallback,
                                                high_risk=winner_was_contested or swap_suspected,
                                                log_prefix=log_prefix, accept_outcome=accept_outcome)

        # rot_pred_implausible veto (2026-09-16): the two weak candidates
        # AGREE with each other (would otherwise have been CONFIRMED above)
        # but this frame's own candidate independently disagrees with a
        # still-credible gyro prediction -- the real case this closes:
        # two same-camera near-degenerate P3P solves sharing the SAME
        # systematic mirror-branch error "agree" with each other (drot=
        # 1.65deg) while both being 158deg off truth. Logged distinctly
        # from an ordinary disagreement so this specific sub-case is
        # identifiable later, same spirit as swap_suspected's own log line.
        if _agrees_with_pending and rot_pred_implausible:
            _log.info(
                f"[{self._ctrl_name}] {log_prefix} PENDING ts={frame_ts_ns} pos={_fmt_v(p_meas)} — "
                f"agrees with buffered weak candidate from ts={pending['frame_ts_ns']} on position/rotation "
                f"(dpos={dpos_m * 1000:.1f}mm drot={drot_deg:.2f}deg) BUT disagrees with a still-credible "
                f"gyro prediction (rot_innov > {float(self._hc_get('cold_reacquire_rot_veto_thresh_deg', 100.0)):.0f}deg) "
                f"— refusing to confirm, buffering this one instead"
            )
        elif _agrees_with_pending and _no_solid_member:
            _log.info(
                f"[{self._ctrl_name}] {log_prefix} PENDING ts={frame_ts_ns} pos={_fmt_v(p_meas)} — "
                f"agrees with buffered weak candidate from ts={pending['frame_ts_ns']} "
                f"(dpos={dpos_m * 1000:.1f}mm drot={drot_deg:.2f}deg) BUT neither has a clean fit "
                f"(err {error_px:.2f}px / {_pending_err:.2f}px, both >= {weak_error_px:.2f}px) — "
                f"refusing to confirm, buffering this one instead"
            )
        elif _agrees_with_pending and _bootstrap_gravity_veto:
            _log.info(
                f"[{self._ctrl_name}] {log_prefix} PENDING ts={frame_ts_ns} pos={_fmt_v(p_meas)} — "
                f"agrees with buffered weak candidate from ts={pending['frame_ts_ns']} on position/rotation "
                f"(dpos={dpos_m * 1000:.1f}mm drot={drot_deg:.2f}deg) BUT its orientation puts gravity "
                f"{_bootstrap_gravity_angle_deg:.0f}deg away from world down (> {_grav_veto_thresh_deg:.0f}deg) "
                f"— refusing to establish a fresh bootstrap on a flipped-looking orientation, buffering this one instead"
            )
        elif _agrees_with_pending and _bootstrap_coverage_fallback_veto:
            _log.info(
                f"[{self._ctrl_name}] {log_prefix} PENDING ts={frame_ts_ns} pos={_fmt_v(p_meas)} — "
                f"agrees with buffered weak candidate from ts={pending['frame_ts_ns']} on position/rotation "
                f"(dpos={dpos_m * 1000:.1f}mm drot={drot_deg:.2f}deg) BUT both are coverage_fallback (too few "
                f"geometrically-expected LEDs matched) — refusing to establish a fresh bootstrap on "
                f"fallback-only evidence, buffering this one instead"
            )
        else:
            _log.info(
                f"[{self._ctrl_name}] {log_prefix} PENDING ts={frame_ts_ns} pos={_fmt_v(p_meas)} — disagrees "
                f"with buffered weak candidate from ts={pending['frame_ts_ns']} pos={_fmt_v(pending['p'])} "
                f"(dpos={dpos_m * 1000:.1f}mm drot={drot_deg:.2f}deg vs. budget {jump_pos_m * 1000:.1f}mm over "
                f"confirm_dt_s={confirm_dt_s * 1000:.1f}ms) — discarding the stale one, buffering this one instead"
            )
        self._cold_pending = {
            "R": R_meas, "p": p_meas, "confidence": confidence,
            "coverage_fallback": coverage_fallback, "frame_ts_ns": frame_ts_ns,
            "n_inliers": n_inliers, "accel": accel_now, "error_px": error_px,
        }
        _report_if_still_usable()
        self._set_last(outcome=pending_outcome, confidence=confidence, pos_pred=p_pred, R_pred=R_pred)
        return False

    def _accept_cold_reacquire(self, R_meas: np.ndarray, p_meas: np.ndarray,
                                frame_ts_ns: int, confidence: float,
                                coverage_fallback: bool, high_risk: bool = False,
                                log_prefix: str = "COLD REACQUIRE",
                                accept_outcome: str = "cold_reacquired") -> bool:
        """Actually commits a cold-reacquire (or, via _try_cold_reacquire's
        bootstrap reuse, a bootstrap) candidate as the new tracking anchor --
        factored out of _try_cold_reacquire so both the immediate-strong-
        match path and the confirmed-weak-pair path share one place that
        does this. log_prefix/accept_outcome are passed through unchanged
        from the caller -- see _try_cold_reacquire's own docstring.

        high_risk: winner_was_contested or swap_suspected on the ORIGINAL
        candidate (threaded through unchanged from _try_cold_reacquire's own
        params) -- passed straight into _seed_rotation_grace below so a
        contested/swap-suspected reacquire never grants the next frame's
        rotation gate a blanket exemption, even once it's reached this
        "trusted, committing now" point. See _seed_rotation_grace's own
        docstring for the real case this closes.

        Position is trusted immediately once this is reached (see
        _try_cold_reacquire's own docstring for the weak/strong routing that
        gates getting here), but velocity is NOT computed from this
        reacquisition -- self.p before this point is stale (wherever the
        controller was last confirmed, before however long this loss ran:
        _try_cold_reacquire is only ever reached once imu_frame_scale has
        fully decayed, i.e. imu_decay_frames+ real lost frames minimum), so a
        straight-line finite difference against it says nothing about the
        controller's actual velocity -- the true path during an unknown-
        length gap could have gone anywhere. Used to compute and trust this
        anyway ("noisy over a long gap, but no worse than starting from
        zero, and self-corrects on the very next accepted frame either way")
        -- wrong on a real case (2026-09-13): a cold reacquisition computed
        v=(2.68,-0.84,0.97) m/s from exactly this kind of stale-anchor long
        gap, and the VERY NEXT frame had NO vision at all (both cameras
        rejected) -- _mark_all_lost's IMU-only propagation then dead-
        reckoned the search anchor ~60mm away using that garbage velocity,
        moving it further from the controller's actual (just-reacquired)
        position instead of "no worse than starting from zero." Treated
        exactly like bootstrap/fail_open now: v=0, not established -- the
        FIRST subsequent real accepted frame (whether a normal blend or
        another cold reacquire) establishes a fresh velocity from a real,
        immediate two-point difference instead."""
        self.v = np.zeros(3)
        self.velocity_established = False

        self.R, self.p = R_meas, p_meas
        self._seed_rotation_grace(coverage_fallback, high_risk=high_risk)
        self.last_update_ts_ns = frame_ts_ns
        self._last_known_p, self._last_known_p_ts_ns = self.p.copy(), frame_ts_ns
        self.frames_since_update = 0
        self.consecutive_rejects = 0
        self._report(frame_ts_ns, self.R, self.p)
        self._trust_window.append(confidence)
        self._set_last(outcome=accept_outcome, confidence=confidence)
        _log.info(
            f"[{self._ctrl_name}] {log_prefix} ts={frame_ts_ns} pos={_fmt_v(p_meas)} — "
            f"{'resuming warm tracking' if log_prefix == 'COLD REACQUIRE' else 'establishing initial tracking'}"
        )
        return True

    def should_force_cold_start(self, frame_ts_ns: int) -> bool:
        """Same semantics as PoseFusionFilter's own method, reading the shared
        top-level fusion: max_coast_s / max_consecutive_rejects keys. Now has
        the same two triggers as PoseFusionFilter -- try_update's hard
        implausibility gate is a genuine reject (see its own docstring), so
        this filter can now stall on consecutive implausible detections the
        same way the Kalman filter can stall on consecutive statistical
        rejects, and needs the same escape hatch."""
        if self.last_update_ts_ns is None:
            return False
        elapsed_s = (frame_ts_ns - self.last_update_ts_ns) / 1e9
        max_coast_s = float(self._cfg.get("max_coast_s", 1.0))
        if elapsed_s > max_coast_s:
            _log.info(
                f"[{self._ctrl_name}] TRACKING LOST: no accepted vision update in "
                f"{elapsed_s:.2f}s (max_coast_s={max_coast_s:.2f}s) — forcing cold-start. "
                f"Last known pos={_fmt_v(self.p)} at ts={self.last_update_ts_ns}"
            )
            return True
        max_consecutive_rejects = int(self._cfg.get("max_consecutive_rejects", 5))
        if self.consecutive_rejects > max_consecutive_rejects:
            _log.info(
                f"[{self._ctrl_name}] TRACKING LOST: {self.consecutive_rejects} consecutive "
                f"implausible vision jumps (max_consecutive_rejects={max_consecutive_rejects}) "
                f"— forcing cold-start. Last known pos={_fmt_v(self.p)} at ts={self.last_update_ts_ns}"
            )
            return True
        return False

    def reset(self) -> None:
        """Full clear of per-track state -- called when should_force_cold_start fires."""
        _log.info(f"[{self._ctrl_name}] RESET — clearing fused state "
                   f"(was pos={_fmt_v(self.p)}, last_update_ts={self.last_update_ts_ns})")
        self.R = self.p = self.last_update_ts_ns = None
        self.v = np.zeros(3)
        self.velocity_established = False
        self._rotation_seed_grace_frames = 0
        self.consecutive_rejects = 0
        self.frames_since_update = 0
        self._last_predict_seen_ts_ns = None
        self._pred_stamp = None
        self._rot_gate_cache = None
        self._cold_pending = None
        self.reported_R = self.reported_p = None
        # Otherwise the next _report() call after re-bootstrapping would compute
        # a "derivative" between the fresh bootstrap pose and this now-stale
        # pre-reset value, producing a spurious speed/lag transient right at
        # the moment a clean re-anchor is what's actually wanted.
        self._pos_filter.reset()
        self._rot_filter.reset()
        self._trust_window.clear()
        self._pos_innov_hist.clear()
        self._rot_innov_hist.clear()
        self._last = {}
        # _last_known_p/_last_known_p_ts_ns deliberately NOT cleared here
        # (2026-09-16) -- unlike self.p/reported_p above, which correctly get
        # wiped because "the recent belief was proven wrong" for THEIR
        # purpose (seeding re-acquisition search / display), this field's
        # sole consumer is TrackingSystem._detect_cold_identity_swap, which
        # specifically wants "where did this controller last look like it
        # genuinely was, before things went wrong" as a fallback once self.p
        # itself is gone -- a stale-but-real reference is exactly what's
        # needed there, gated by its own staleness check in the consumer
        # (cold_swap_stale_reference_max_s) rather than being wiped outright.
