"""PoseFusionFilter -- the unified vision+IMU+smoothing pose filter (see
/home/nikitakarpuks/.claude/plans/wild-fluttering-dolphin.md for the full design
rationale, including why this is an error-state filter rather than a full UKF, and
the comparison against Monado's real shipped controller-fusion filter that
motivated the design choices here).

One instance per controller. Nominal state (R, p, v) is propagated by calling
src.imu_data.predict_world_pose -- already-validated dead-reckoning, zero new
integration math. A separate diagonal error covariance P (position/rotation/velocity,
9 entries, no cross-terms -- a deliberate simplification, see try_update's docstring)
grows between updates and shrinks via a per-axis Kalman correction when a vision
measurement passes the Mahalanobis gate. No bias state (matches both Monado's own
precedent for this problem and this project's own empirical finding -- see README
finding 7/11 -- that bias doesn't measurably matter at these horizons).

Every number here (process noise rates, gate threshold, coast/reject limits) is a
first-cut default meant to be tuned during the verification pass the plan describes,
not a validated final value.
"""
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from src.imu_data import predict_world_pose, slice_imu_to_window, dead_reckon_dense

# Position/rotation/velocity index blocks within the 9-entry diagonal covariance.
_POS = slice(0, 3)
_ROT = slice(3, 6)
_VEL = slice(6, 9)

# try_update's absolute-position sanity check (see there) only applies to a
# genuinely "warm", true frame-to-frame update -- NOT to an update following a
# coast/occlusion gap, where a large real displacement (or an already-loose
# P_pred) is expected and legitimate. Derived empirically from this project's
# own data/vision_pose_log.csv (both controllers, frame_range 2500-3000):
# inter-update dt during real continuous tracking is extremely tightly
# clustered at ~0.0221s (p50 through p95 all within ~0.0002s of each other --
# this dataset's cameras alternate ~11ms/~22ms capture gaps), then jumps
# straight to >=0.05s for every gap caused by a real occlusion/reacquisition
# (no dt values observed in [0.023, 0.05) at all). 0.04s sits in that empty
# gap: comfortably above the worst normal single/double-frame cadence (with
# ~1.8x headroom over the tightest-clustered p95), while still well below
# every observed coast gap.
_WARM_FRAME_DT_CUTOFF_S = 0.04


class PoseFusionFilter:
    """Fuses vision pose solutions with IMU dead-reckoning for one controller.

    fail-open throughout: any missing precondition (no IMU coverage, no converged
    g_world, no prior state yet) makes predict()/try_update() behave as if the
    filter weren't there -- never worse than today's raw-vision-every-frame
    behavior, per the plan's master fusion.enabled switch living one level up
    (this class itself has no enabled flag; the caller constructs one or not)."""

    def __init__(self, gyro_data, accel_data, lever_arm, g_world_estimator, cfg: dict):
        self._gyro_data = gyro_data            # (t_gyro, gyro_body) or None
        self._accel_data = accel_data          # (t_accel, accel_body) or None
        self._lever_arm = lever_arm            # (3,) or None
        self._g_world_estimator = g_world_estimator  # LiveGravityEstimator or None
        self._cfg = cfg

        self.R = None                # (3,3) last ACCEPTED orientation
        self.p = None                # (3,) last ACCEPTED position
        self.v = np.zeros(3)         # (3,) last ACCEPTED velocity estimate
        self.P = None                # (9,) diagonal error covariance at last_update_ts_ns
        self.last_update_ts_ns = None
        self.consecutive_rejects = 0
        # Separate from consecutive_rejects: counts consecutive _looks_like_a_sibling
        # rejects at a NEVER-yet-bootstrapped state (self.R is None), where
        # should_force_cold_start's normal elapsed-time/consecutive_rejects checks
        # can't apply (there's no last_update_ts_ns to measure from). Without this,
        # a controller repeatedly flagged as looking like a live, trustworthy sibling
        # can livelock forever -- it never bootstraps its own state, and neither this
        # nor _mark_all_lost's escape hatch can fire (found in review).
        self.consecutive_sibling_rejects = 0

        # What main.py/_commit_fused_solution should actually report this frame --
        # set by every try_update() call, whether accepted or rejected (see its
        # docstring: a rejected frame reports the IMU-only prediction, not a stale
        # last-accepted pose).
        self.reported_R = None
        self.reported_p = None

        # Other enabled controllers' own PoseFusionFilter instances -- wired post-
        # construction via set_siblings() (TrackingSystem builds every ControllerTracker
        # before any of them can reference the others). Used ONLY by try_update's
        # bootstrap branch, see _looks_like_a_sibling.
        self._sibling_filters: list = []

        # Rolling (comfort, confidence) pairs from recent GATED correction accepts
        # ONLY -- NOT bootstrap, NOT the predict()-unavailable fail-open accept. See
        # trust_score()/trust() below and _record_trust_sample's docstring for why
        # that population rule is deliberate, not an oversight.
        self._trust_window = deque(maxlen=int(cfg.get("trust_window_len", 8)))

        # Per-call debug diagnostics (see _set_last/debug_snapshot) -- purely
        # for an external debug-visualization layer, never read by this
        # class's own control flow. Empty until the first try_update() call.
        self._last: dict = {}

    def set_siblings(self, siblings: list) -> None:
        self._sibling_filters = list(siblings)

    def note_real_frame(self, frame_ts_ns: int) -> None:
        """No-op here -- this filter has no frame-counted decay (P_pred already
        grows continuously with REAL elapsed time via _process_noise(dt_s), not
        a frames_since_update analogue), so it doesn't need ControllerTracker.
        _mark_all_lost's per-lost-frame bump. Present only so callers can treat
        HeuristicPoseFusionFilter/PoseFusionFilter uniformly (see
        HeuristicPoseFusionFilter.note_real_frame's own docstring for why THAT
        filter needs this called unconditionally on every lost frame)."""
        pass

    def _initial_P(self) -> np.ndarray:
        c = self._cfg
        pos0 = float(c.get("initial_pos_sigma_m", 0.02))
        rot0 = np.radians(float(c.get("initial_rot_sigma_deg", 2.0)))
        vel0 = float(c.get("initial_vel_sigma_m_per_s", 0.1))
        return np.array([pos0, pos0, pos0, rot0, rot0, rot0, vel0, vel0, vel0]) ** 2

    def _process_noise(self, dt_s: float) -> np.ndarray:
        """Diagonal variance ADDED over an elapsed gap of dt_s seconds -- a simple,
        tunable growth model (std grows linearly with elapsed time), not a
        rigorously-derived stochastic process. See class docstring."""
        c = self._cfg
        pos_rate = float(c.get("process_noise_pos_m_per_s", 0.3))
        rot_rate = np.radians(float(c.get("process_noise_rot_deg_per_s", 20.0)))
        vel_rate = float(c.get("process_noise_vel_m_per_s2", pos_rate))
        dt_s = max(dt_s, 0.0)
        return np.array([
            (pos_rate * dt_s) ** 2, (pos_rate * dt_s) ** 2, (pos_rate * dt_s) ** 2,
            (rot_rate * dt_s) ** 2, (rot_rate * dt_s) ** 2, (rot_rate * dt_s) ** 2,
            (vel_rate * dt_s) ** 2, (vel_rate * dt_s) ** 2, (vel_rate * dt_s) ** 2,
        ])

    def predict(self, target_ts_ns: int):
        """Dead-reckon the last ACCEPTED state forward to target_ts_ns.

        Returns (R_pred, p_pred, P_pred (9,)) or None if prediction isn't possible
        yet (no prior state, no IMU coverage, g_world not converged) -- callers must
        treat None as "fall back to today's behavior for this frame", not an error."""
        if self.R is None or self.last_update_ts_ns is None:
            return None
        if self._gyro_data is None or self._accel_data is None or self._lever_arm is None:
            return None
        if self._g_world_estimator is None:
            return None
        g_world = self._g_world_estimator.g_world
        if g_world is None:
            return None

        # Sliced to this call's own [last_update_ts_ns, target_ts_ns] span -- without
        # this, integrate_gyro_segment/integrate_accel_to_position mask/interpolate
        # over the FULL (live-session-length, not just this gap's) gyro/accel arrays
        # on every predict() call, i.e. every frame once wired into live tracking
        # (found in code review).
        t_gyro, gyro_body = slice_imu_to_window(*self._gyro_data, self.last_update_ts_ns, target_ts_ns)
        t_accel, accel_body = slice_imu_to_window(*self._accel_data, self.last_update_ts_ns, target_ts_ns)
        predicted = predict_world_pose(t_gyro, gyro_body, t_accel, accel_body, g_world,
                                        self._lever_arm, self.last_update_ts_ns, target_ts_ns,
                                        self.R, self.p, self.v)
        if predicted is None:
            return None
        R_pred, p_pred = predicted
        dt_s = (target_ts_ns - self.last_update_ts_ns) / 1e9
        P_pred = self.P + self._process_noise(dt_s)
        return R_pred, p_pred, P_pred

    def _record_trust_sample(self, d2: float, gate: float, confidence: float) -> None:
        """Called ONLY from try_update's gated-correction accept branch (d2 <= gate)
        -- NOT from bootstrap, NOT from the predict()-unavailable fail-open accept.
        This means a freshly-(re)bootstrapped filter starts with an EMPTY window,
        so trust_score() reads 0.0 until it has survived at least one real gate
        check against its OWN prediction. That's deliberate: the documented failure
        this exists to catch (see class/_looks_like_a_sibling docstrings) involved
        BOTH controllers having recently, independently bootstrapped shortly before
        one drifted -- letting a just-bootstrapped filter immediately vouch for a
        sibling would reproduce the same shape of bug, not fix it."""
        comfort = float(np.clip(1.0 - d2 / gate, 0.0, 1.0))
        self._trust_window.append((comfort, confidence))

    def trust_score(self) -> float:
        """Geometric mean of recent GATED-accept comfort (how far inside the gate,
        not merely inside it) and vision confidence, over the rolling window --
        either input being weak drags the score down. 0.0 with an empty window.

        Self-referential caveat: comfort is computed against this filter's OWN
        prediction, so a prediction that has quietly drifted together with its
        measurements (the documented failure mode) can still show comfortable d2
        values -- confidence is the only externally-grounded ingredient here, which
        is why it's part of the product rather than an afterthought. This is a
        real ceiling on what this signal can catch, not a solved problem."""
        if not self._trust_window:
            return 0.0
        comforts, confidences = zip(*self._trust_window)
        return float(np.sqrt(np.mean(comforts) * np.mean(confidences)))

    def trust(self, frame_ts_ns: int):
        """Composite trust in this filter's CURRENT state, meant for a SIBLING's
        bootstrap cross-check (_looks_like_a_sibling) -- NOT used to gate this
        filter's own accepts, which stay fail-open exactly as before. Combines
        trust_score() with an explicit decay for elapsed time since the last real
        accepted update -- separate from, and in addition to, P's own covariance
        growth (see class docstring); tau default calibrated against this
        project's real gap-vs-drift data (error stays ~0.3-0.7m under ~0.8s gaps
        then cliffs to 6x+ actual displacement by ~1.3s -- see
        visualization/pose_fusion_phase5_trust_signal.html), so trust is already
        deeply discounted well before max_coast_s's hard cutoff, not just after.
        Returns None if there's no live state at all (nothing to trust or not)."""
        if self.last_update_ts_ns is None:
            return None
        dt_s = max((frame_ts_ns - self.last_update_ts_ns) / 1e9, 0.0)
        tau = float(self._cfg.get("trust_gap_tau_s", 0.45))
        gap_factor = float(np.exp(-dt_s / tau))
        return self.trust_score() * gap_factor

    # Every field debug_snapshot() can surface from one try_update() call --
    # kept as one fixed schema so a branch that doesn't produce a given field
    # (e.g. a reject has no Kalman gain) reports None for it instead of
    # leaking a stale value left over from a DIFFERENT branch's previous call.
    #
    # abs_reject: True only when the SECOND, independent absolute-distance
    # sanity check (dt_s <= _WARM_FRAME_DT_CUTOFF_S and pos_innov_m exceeds
    # fusion.absolute_reject_pos_m -- see try_update) is what fired, regardless
    # of whether the ordinary statistical d2 > gate check also happened to be
    # true. False (not None) on gated_accept/gated_reject, since the check was
    # actually evaluated there; None on bootstrap/fail_open/sibling_rejected,
    # where there was no P_pred to evaluate it against at all. Lets a debug-viz
    # consumer tell an identity-swap-shaped reject (large absolute jump, caught
    # even though loose post-coast covariance let d2 alone through) apart from
    # an ordinary statistical gated_reject.
    #
    # pos_pred/R_pred: the PRE-correction predict()ed pose -- set ONLY by the
    # two branches that actually called predict() and got a real answer
    # (gated_accept, gated_reject), deliberately None on bootstrap/fail_open/
    # sibling_rejected, where there was no prior state to predict FROM at all.
    # A debug-viz consumer wanting "what did IMU alone believe this frame"
    # must use these, not reported_p/reported_R -- reported_p IS the raw
    # vision candidate on bootstrap/fail_open (see try_update's docstring),
    # so treating it as "IMU's prediction" would silently relabel vision as
    # IMU at exactly the frames IMU has no opinion at all (found by the user
    # spotting IMU appearing to "drift toward" a wrong vision match at a
    # reacquisition bootstrap -- it wasn't drifting, it was just displaying
    # that wrong candidate's own position).
    _LAST_FIELDS = ("outcome", "d2", "gate", "confidence", "comfort",
                     "pos_innov_m", "rot_innov_deg", "kalman_gain_pos", "kalman_gain_rot",
                     "pos_pred", "R_pred", "abs_reject")

    def _set_last(self, **kwargs) -> None:
        """Record this try_update()/_looks_like_a_sibling() call's diagnostics
        for debug_snapshot() -- see _LAST_FIELDS. Debug-visualization only;
        never consumed by this class's own control flow."""
        self._last = {k: kwargs.get(k) for k in self._LAST_FIELDS}

    def debug_snapshot(self, frame_ts_ns: int) -> dict:
        """Everything a debug-visualization layer would want to plot per
        frame: the last try_update() call's per-step diagnostics (_last)
        plus live-queryable state (trust, consecutive_rejects, P-derived
        sigmas). Read-only -- config.yml's visualization.visualize_pose_fusion
        gates whether anything actually calls this; a disabled filter or one
        with no state yet still returns a valid (mostly-None) dict."""
        if self.P is not None:
            pos_sigma_m   = float(np.sqrt(np.mean(self.P[_POS])))
            rot_sigma_deg = float(np.degrees(np.sqrt(np.mean(self.P[_ROT]))))
            vel_sigma_m_s = float(np.sqrt(np.mean(self.P[_VEL])))
        else:
            pos_sigma_m = rot_sigma_deg = vel_sigma_m_s = None
        return {
            **self._last,
            "has_state":           self.R is not None,
            "consecutive_rejects": self.consecutive_rejects,
            "trust_score":         self.trust_score(),
            "trust":               self.trust(frame_ts_ns),
            "pos_sigma_m":         pos_sigma_m,
            "rot_sigma_deg":       rot_sigma_deg,
            "vel_sigma_m_s":       vel_sigma_m_s,
        }

    def predict_dense(self, target_ts_ns: int, sample_every_n: int = 1):
        """Dense dead-reckoned trajectory from last_update_ts_ns to
        target_ts_ns, for a debug-visualization curve -- NOT used by
        predict()/try_update's own control flow (those only need the
        endpoint). See src.imu_data.dead_reckon_dense for the O(samples^2)
        cost tradeoff; a caller logging this every frame across a long
        reject streak should raise sample_every_n to bound the per-frame cost.

        Returns (ts (N,) int64, positions (N,3), rotations: list of (3,3))
        or None under the same unavailability conditions predict() returns
        None for (including an empty result if no sample in the window has
        IMU coverage)."""
        if self.R is None or self.last_update_ts_ns is None:
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

    def _meas_pos_sigma_m(self, solution: dict) -> float:
        """Position-measurement-noise sigma, shared by try_update's per-controller
        gate and _looks_like_a_sibling's bootstrap cross-check -- pulled out so a
        future tuning change to one doesn't silently diverge from the other
        (found in review; both call sites previously duplicated this formula)."""
        c = self._cfg
        pos_err_px = float(solution.get("error", 0.0))
        confidence = max(float(solution.get("confidence", 1.0)), 0.05)
        base_pos_sigma_m = float(c.get("meas_base_pos_sigma_m", 0.01))
        px_to_m = float(c.get("meas_error_px_to_m", 0.002))
        return (base_pos_sigma_m + pos_err_px * px_to_m) / confidence

    def _looks_like_a_sibling(self, solution: dict, p_meas: np.ndarray, frame_ts_ns: int) -> bool:
        """True if p_meas -- a candidate THIS controller's filter has no prior state
        to gate -- sits implausibly close to a SIBLING controller's own still-live
        predicted position at this same instant. Position-only (3 dof): the two
        physical controllers' LED constellations are documented near-mirror-images,
        so a swapped candidate's ORIENTATION can still look locally plausible for
        either identity, but the two controllers are not standing in the same place.
        Read-only: sib.predict() doesn't mutate the sibling's own state, so calling
        it here has no side effect on the sibling filter itself.

        Only ever consulted from try_update's bootstrap branch (self.R is None) --
        a controller with its own live state is gated by its own prediction instead,
        which is the ORIGINAL (per-controller-only) mechanism; this is what closes
        the gap that mechanism has at a fresh bootstrap, see try_update's docstring.

        A sibling is only used as a reference when ITS OWN trust() clears
        sibling_trust_min -- closes the specific gap found validating this
        mechanism against this project's real documented swap: a live-but-
        silently-drifted sibling (individually gate-passing accepts that walked
        its state ~0.29m from ground truth) was previously trusted as-is, so the
        cross-check couldn't see the collision. See trust()'s own caveat: this
        still doesn't help when the sibling is untrustworthy for a DIFFERENT
        reason (e.g. it's also mid-reacquisition after a long simultaneous
        occlusion) -- trust() reads near-zero there too, so the loop just finds
        no usable reference, same as today's "no live prediction" case."""
        if not self._sibling_filters:
            return False
        c = self._cfg
        pos_sigma_m = self._meas_pos_sigma_m(solution)
        meas_pos_var = np.array([pos_sigma_m, pos_sigma_m, pos_sigma_m]) ** 2

        gate = float(c.get("bootstrap_cross_ctrl_gate_chi2_threshold", 7.8))
        trust_min = float(c.get("sibling_trust_min", 0.3))
        for sib in self._sibling_filters:
            sib_trust = sib.trust(frame_ts_ns)
            if sib_trust is None or sib_trust < trust_min:
                continue  # no live sibling state, or its state isn't currently trustworthy
            predicted = sib.predict(frame_ts_ns)
            if predicted is None:
                continue  # sibling has no live prediction either -- no signal available
            _, sib_p_pred, sib_P_pred = predicted
            combined_var = np.maximum(sib_P_pred[:3] + meas_pos_var, 1e-10)
            d2 = float(np.sum((p_meas - sib_p_pred) ** 2 / combined_var))
            if d2 <= gate:
                self._set_last(outcome="sibling_rejected", d2=d2, gate=gate,
                                confidence=float(solution.get("confidence", 1.0)))
                return True
        return False

    def try_update(self, solution: dict, frame_ts_ns: int) -> bool:
        """Gate + (if accepted) correct against a new vision solution.

        DIAGONAL/DECOUPLED simplification: position, rotation, and velocity error
        are tracked as independent variances (no cross-covariance terms). This means
        a position correction does NOT automatically inform velocity the way a full
        joint-covariance EKF would -- velocity is instead updated by a direct
        finite-difference against the correction, a pragmatic stand-in. This is a
        deliberate first-cut simplification (see the plan doc), not an oversight;
        revisit if validation shows velocity estimates drifting stale across a long
        string of accepted updates.

        Always call this instead of unconditionally trusting solution -- the SAME
        method is used for both warm-frame commits and cold-batch re-acquisition
        candidates, which is what makes the original identity-swap bug a special
        case of this general mechanism rather than a separate check.

        Sets self.reported_R/self.reported_p to whatever should actually be
        reported this frame (the correction if accepted, a fresh prediction if
        rejected or if no prior state exists yet to gate against).

        Returns True if the measurement was accepted (state updated), False if
        rejected (state left at prediction, consecutive_rejects incremented) or if
        this is the very first observation (bootstrapped unconditionally, counts as
        accepted) -- UNLESS _looks_like_a_sibling flags it first, see there."""
        T_world_ctrl = solution["T_world_ctrl"]
        R_meas, p_meas = T_world_ctrl.R, T_world_ctrl.t

        if self.R is None:
            # No prior state at all for THIS controller -- the per-controller gate
            # below has nothing to compare against, which is exactly the swap-bug
            # blind spot the plan's original design missed (confirmed against this
            # project's real, documented swap: both controllers had independently
            # bootstrapped, one onto the other's actual position, well before the
            # visibly-swapped frame -- see the Phase 4 validation writeup). Before
            # bootstrapping blindly, check whether this candidate instead looks
            # like it's actually a SIBLING controller's own current position --
            # a strong, checkable swap signal that requires no state of THIS
            # controller's own, only a still-live sibling.
            if self._looks_like_a_sibling(solution, p_meas, frame_ts_ns):
                self.consecutive_rejects += 1
                self.consecutive_sibling_rejects += 1
                # Do NOT report the flagged candidate (R_meas/p_meas) here --
                # _commit_fused_solution (src/controller.py) unconditionally feeds
                # reported_R/reported_p into every camera's next-frame warm-start
                # prior, even on reject. Reporting the exact position just judged
                # to probably belong to a SIBLING controller would bias this
                # controller's own next search toward the sibling's location --
                # the opposite of the intended effect (found in review). Coast on
                # whatever was last reported instead; only fall back to the raw
                # candidate if there's truly no prior reported pose at all (this
                # controller's very first-ever observation, nothing to coast on).
                if self.reported_R is None:
                    self.reported_R, self.reported_p = R_meas, p_meas
                return False
            # Bootstrap directly from this solution.
            self.R, self.p = R_meas, p_meas
            self.v = np.zeros(3)
            self.P = self._initial_P()
            self.last_update_ts_ns = frame_ts_ns
            self.consecutive_rejects = 0
            self.consecutive_sibling_rejects = 0
            self.reported_R, self.reported_p = self.R, self.p
            self._set_last(outcome="bootstrap", confidence=float(solution.get("confidence", 1.0)))
            return True

        predicted = self.predict(frame_ts_ns)
        if predicted is None:
            # Can't gate yet (IMU/g_world not ready) -- fail open: accept as-is,
            # same as today's unconditional-accept behavior.
            self.R, self.p = R_meas, p_meas
            self.v = np.zeros(3)
            self.P = self._initial_P()
            self.last_update_ts_ns = frame_ts_ns
            self.consecutive_rejects = 0
            self.reported_R, self.reported_p = self.R, self.p
            self._set_last(outcome="fail_open", confidence=float(solution.get("confidence", 1.0)))
            return True
        R_pred, p_pred, P_pred = predicted
        # Computed here (not just later, for the accept branch's v_new finite-
        # difference) because the new absolute-distance reject check below also
        # needs it, and it must gate the REJECT branch too -- self.last_update_ts_ns
        # hasn't changed since predict() used the same value, so this is exactly
        # predict()'s own elapsed gap.
        dt_s = (frame_ts_ns - self.last_update_ts_ns) / 1e9

        c = self._cfg
        confidence = max(float(solution.get("confidence", 1.0)), 0.05)  # floor -- never divide by ~0
        pos_sigma_m = self._meas_pos_sigma_m(solution)
        base_rot_sigma_deg = float(c.get("meas_base_rot_sigma_deg", 1.0))
        rot_sigma_rad = np.radians(base_rot_sigma_deg) / confidence
        R_meas_noise = np.array([pos_sigma_m, pos_sigma_m, pos_sigma_m,
                                  rot_sigma_rad, rot_sigma_rad, rot_sigma_rad]) ** 2

        pos_innov = p_meas - p_pred
        rot_innov = Rotation.from_matrix(R_pred.T @ R_meas).as_rotvec()
        innov = np.concatenate([pos_innov, rot_innov])
        # Epsilon floor: all thresholds/sigmas above are explicitly first-cut config
        # defaults meant to be tuned (see class docstring) -- a future tuning pass
        # (or dt_s==0 on a same-timestamp predict) setting any of them to 0 would
        # otherwise divide by exactly 0 here, producing NaN/inf that permanently
        # poisons self.P (never recovers, since P only ever multiplies/adds from
        # its own previous value) with no crash and no visible symptom until much
        # later. Found in review.
        combined_var = np.maximum(P_pred[:6] + R_meas_noise, 1e-10)  # diagonal -> Mahalanobis decomposes into a sum
        d2 = float(np.sum(innov ** 2 / combined_var))

        gate = float(c.get("gate_chi2_threshold", 7.8))
        pos_innov_m   = float(np.linalg.norm(pos_innov))
        rot_innov_deg = float(np.degrees(np.linalg.norm(rot_innov)))

        # SECOND, independent sanity check, OR'd with the statistical d2 > gate
        # test above: an absolute position jump too large to be real motion
        # within one normal warm-tracking frame interval, even when d2 alone
        # would pass because P_pred is loose (e.g. just after a coast/reject
        # streak) -- exactly the loophole that would let an identity-swapped
        # candidate (a real position, just the WRONG controller's) slip past
        # the purely-statistical gate. Deliberately scoped to true frame-to-
        # frame dt (_WARM_FRAME_DT_CUTOFF_S, see its own comment) -- NOT during/
        # after a long coast, where legitimately large real motion (or an
        # already-expected-large P_pred) is plausible and shouldn't trip this.
        abs_reject_thresh = float(c.get("absolute_reject_pos_m", 0.03))
        abs_reject = dt_s <= _WARM_FRAME_DT_CUTOFF_S and pos_innov_m > abs_reject_thresh

        if d2 > gate or abs_reject:
            self.consecutive_rejects += 1
            self.reported_R, self.reported_p = R_pred, p_pred
            # State/P deliberately NOT advanced to frame_ts_ns -- next predict()
            # keeps dead-reckoning from the same last-accepted anchor, so
            # uncertainty keeps compounding over the FULL elapsed time since the
            # last real acceptance, not reset by a rejected frame.
            self._set_last(outcome="gated_reject", d2=d2, gate=gate, confidence=confidence,
                            pos_innov_m=pos_innov_m, rot_innov_deg=rot_innov_deg,
                            pos_pred=p_pred, R_pred=R_pred, abs_reject=abs_reject)
            return False

        K = P_pred[:6] / combined_var  # per-axis Kalman gain, diagonal simplification

        # Long-gap reacquisition floor: past this many missed frames' worth of real
        # elapsed time, the IMU-dead-reckoned prediction should have very little say
        # in the correction, REGARDLESS of what confidence/P_pred would otherwise
        # compute -- a low-confidence brute-force recovery after a long blind gap
        # was measured landing at K~0.4 (this project's own "frame 314" case,
        # confidence=0.128 inflating R to ~118mm, comparable to P_pred's own ~93mm
        # growth over a 155ms/8-missed-frame gap), i.e. the filter kept ~60% weight
        # on a prediction that had been dead-reckoning blind the whole time. That's
        # a real, decided trust-in-IMU choice made every such frame, not a display-
        # only artifact -- trust_score()/trust_window_len (used elsewhere, e.g. the
        # sibling cross-check) never enters this correction at all. Floored here,
        # in the actual gain used for p_new/R_new below, not left to confidence
        # alone. long_gap_reacquire_s default (0.08s) ~= 5 frames at this
        # recording's real median cadence (~16.6ms, see _WARM_FRAME_DT_CUTOFF_S's
        # own derivation) -- tune per-recording if cadence differs substantially.
        _long_gap_s = float(c.get("long_gap_reacquire_s", 0.08))
        if dt_s >= _long_gap_s:
            _min_vision_trust = float(c.get("long_gap_min_vision_trust", 0.9))
            K = np.clip(np.maximum(K, _min_vision_trust), 0.0, 1.0)

        # NOTE: a low-reprojection-error trust floor was tried here (twice, same
        # day: once on error_px alone, once band-gated on confidence + pair
        # count to fix a documented tautology in error_px-alone reasoning) and
        # REVERTED BOTH TIMES. It fixed a targeted low-confidence-but-correct
        # frame ("frame ~859", left_controller), but depended on a companion
        # change in ControllerTracker._commit_fused_solution (seeding search-
        # time pose_history from raw vision on accepted updates instead of the
        # fused pose) that an adversarial, mocap-validated audit found caused a
        # SEVERE regression elsewhere in the same recording (~600ms/840mm
        # mistrack, 34 dropped frames) -- see that method's own comment for the
        # full account. That companion change was reverted, which removes the
        # reason this floor could safely fire (its whole premise was "fusion's
        # trust decision can no longer corrupt next frame's search, so it's
        # safe to floor trust here again"). Re-adding this floor without that
        # companion change would reopen the ORIGINAL feedback-loop regression
        # this floor caused the first time it was tried. Don't re-add either
        # piece in isolation -- if revisited, needs both a safe way to let
        # search anchor on vision (e.g. only when confidence is high, not
        # unconditionally) AND this floor, validated together against mocap
        # ground truth across the WHOLE recording, not a single spot-check.

        correction = K * innov

        correction = K * innov
        p_new = p_pred + correction[_POS]
        R_new = R_pred @ Rotation.from_rotvec(correction[3:6]).as_matrix()
        P_new = P_pred.copy()
        # General (Joseph-form-equivalent, per-axis-diagonal) posterior variance for
        # x_new = (1-K)*x_pred + K*z, valid for ANY K -- NOT the (1-K)*P_pred shortcut,
        # which is only correct when K is the OPTIMAL gain P_pred/(P_pred+R). Both
        # floors above can force K above optimal (found in code review, 2026-09-04):
        # with K floored, (1-K)*P_pred underestimates the true posterior variance
        # (measured ~11x at a K=0.9 floor against this recording's steady-state P/R),
        # which matters downstream via _looks_like_a_sibling's cross-controller d2
        # check (reads a sibling's P_pred) and any future gate/reject-threshold tuning.
        # Reduces to the textbook (1-K)*P_pred exactly when K IS optimal, so this is
        # strictly more correct, not just a special-case fix.
        P_new[:6] = (1.0 - K) ** 2 * P_pred[:6] + K ** 2 * R_meas_noise

        # dt_s already computed above (right after predict()'s R_pred/p_pred/
        # P_pred, so the new absolute-distance reject check could use it too).
        v_new = (p_new - self.p) / dt_s if dt_s > 0 else self.v

        self.R, self.p, self.v, self.P = R_new, p_new, v_new, P_new
        self.last_update_ts_ns = frame_ts_ns
        self.consecutive_rejects = 0
        self.reported_R, self.reported_p = self.R, self.p
        self._record_trust_sample(d2, gate, confidence)
        self._set_last(outcome="gated_accept", d2=d2, gate=gate, confidence=confidence,
                        comfort=float(np.clip(1.0 - d2 / gate, 0.0, 1.0)),
                        pos_innov_m=pos_innov_m, rot_innov_deg=rot_innov_deg,
                        kalman_gain_pos=float(np.mean(K[:3])), kalman_gain_rot=float(np.mean(K[3:6])),
                        pos_pred=p_pred, R_pred=R_pred, abs_reject=False)
        return True

    def should_force_cold_start(self, frame_ts_ns: int) -> bool:
        c = self._cfg
        # Immediate trigger: the absolute-distance identity-swap check (see
        # try_update's abs_reject) is a narrow, high-precision signature of a
        # genuine identity mismatch -- fire the escape hatch on the SAME frame
        # it's caught, rather than waiting for several more consecutive
        # rejects or a long coast to accumulate via the checks below.
        if self._last.get("abs_reject"):
            return True
        if self.last_update_ts_ns is None:
            # Never bootstrapped (or freshly reset) -- the elapsed-time/
            # consecutive_rejects checks below don't apply (nothing to measure
            # from), but a controller repeatedly flagged as looking like a
            # trustworthy sibling at every bootstrap attempt can still livelock
            # forever with no other escape (found in review) -- reuses
            # max_consecutive_rejects as its own threshold rather than adding
            # a new config key for what's the same underlying tolerance.
            return self.consecutive_sibling_rejects > int(c.get("max_consecutive_rejects", 5))
        elapsed_s = (frame_ts_ns - self.last_update_ts_ns) / 1e9
        if elapsed_s > float(c.get("max_coast_s", 1.0)):
            return True
        if self.consecutive_rejects > int(c.get("max_consecutive_rejects", 5)):
            return True
        return False

    def reset(self) -> None:
        """Full clear -- called when should_force_cold_start fires and the
        controller falls through to brute-force re-detection; the filter
        re-bootstraps unconditionally (see try_update's self.R is None branch)
        whenever that re-detection next produces a candidate."""
        self.R = self.p = self.P = self.last_update_ts_ns = None
        self.v = np.zeros(3)
        self.consecutive_rejects = 0
        self.consecutive_sibling_rejects = 0
        self.reported_R = self.reported_p = None
        self._trust_window.clear()
        self._last = {}
