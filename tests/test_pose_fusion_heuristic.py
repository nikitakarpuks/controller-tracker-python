"""Unit coverage for the jump-detection additions to HeuristicPoseFusionFilter
(src/pose_fusion_heuristic.py):

  - Case A: _agreement_factor / _agreement_weak_bounds -- the soft ramp that
    pulls the warm-state tracking pose back toward the IMU prediction for a
    disagreement too small to hit the pre-existing hard implausible_jump_pos_m/
    _rot_deg gate.
  - _try_cold_reacquire -- once imu_frame_scale has fully decayed
    (vision_only), the first candidate past the sibling-collision check is
    trusted immediately and resumes normal warm tracking (an earlier 3-frame
    confirmation window was removed 2026-09-06, explicit direction -- see
    that method's own docstring).
  - The sibling-collision check (_looks_like_a_sibling), ported from
    PoseFusionFilter's own bootstrap-only mechanism but also consulted on
    every cold-reacquisition candidate here.

Real IMU dead-reckoning (predict()) is stubbed out on the instance under test
so these tests exercise the fusion/gating logic in isolation, the same
approach test_cold_conflict_resolution.py uses for TrackingSystem -- no pool,
no images, no real gyro/accel arrays.

Uses stdlib unittest (pytest is not a declared dependency of this project).
Run with:  python3 -m unittest tests.test_pose_fusion_heuristic
"""
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from src.controller import ControllerTracker
from src.imu_data import MOCAP_ROOM_G_WORLD
from src.pose_fusion import PoseFusionFilter
from src.pose_fusion_heuristic import HeuristicPoseFusionFilter, _ramp_down, _ramp_up
from src.transformations import Transform

_NS = 1_000_000_000


def _solution(R, p, confidence=1.0, n_inliers=20, error_px=0.1):
    return {
        "T_world_ctrl": Transform(R, p),
        "confidence": confidence,
        "assignment": [(i, i) for i in range(n_inliers)],
        "aux_assignments": {},
        "error": error_px,
    }


def _quality(n_inliers, error_px, weak_inliers=5, strong_inliers=8, weak_error_px=0.5, strong_error_px=0.15):
    """(inlier_factor+error_factor)/2 -- the SAME formula _vision_weight uses
    for its own `quality`, reproduced here (against _make_filter's own
    weak/strong config values) so Case A pushback-scaling tests can compute
    the exact expected (1-quality) discount rather than assume a fixed
    quality=0/1."""
    inlier_factor = _ramp_up(float(n_inliers), weak_inliers, strong_inliers)
    error_factor = _ramp_down(float(error_px), weak_error_px, strong_error_px)
    return (inlier_factor + error_factor) / 2.0


def _make_filter(cfg_overrides=None, ctrl_name="test"):
    cfg = {
        "max_coast_s": 1.0,
        "max_consecutive_rejects": 5,
        "sibling_trust_min": 0.3,
    }
    hc = {
        "cost_weight_imu": 0.0,          # matches this project's current real config
        "cost_weight_vision": 2.0,
        "vision_weight_weak_inliers": 5,
        "vision_weight_strong_inliers": 8,
        "vision_weight_weak_error_px": 0.5,
        "vision_weight_strong_error_px": 0.15,
        # Reproduces the old flat implausible_jump_pos_m=0.3/_rot_deg=60.0
        # exactly (per_speed=0 makes _implausible_jump_thresholds' speed
        # term a no-op regardless of self.v) -- most existing tests in this
        # file assert against those specific numbers and aren't about the
        # 2026-09-13 speed-scaling itself; see ImplausibleJumpThresholdsTests
        # for tests of the speed-scaled formula specifically.
        "implausible_jump_pos_thresh_base_mm": 300.0,
        "implausible_jump_pos_thresh_per_speed_mm_s": 0.0,
        "implausible_jump_rot_thresh_base_deg": 60.0,
        "implausible_jump_rot_thresh_per_speed_deg_s": 0.0,
        "imu_decay_frames": 4,
        "gap_dt_normal_s": 0.04,
        "gap_dt_full_s": 0.15,
        "gap_vision_weight_floor": 0.3,
        "agreement_hist_len": 20,
        "agreement_hist_min_samples": 5,
        "agreement_k_pos": 3.0,
        "agreement_k_rot": 3.0,
        "agreement_floor_pos_m": 0.001,
        "agreement_floor_rot_deg": 0.5,
        "sibling_collision_dist_m": 0.15,
        # weak_confirm_max_speed_m_s/_ang_speed_deg_s intentionally left at
        # their code defaults (3.0 m/s / 2000deg/s) -- generous enough on
        # their own for this file's whole-second (_NS) timestamp convention
        # (confirmed: budget=3135mm/2015.6deg at confirm_dt_s=1s), same
        # spirit as cold_confirm_max_gap_s below.
        # Generous defaults -- this file's own tests advance timestamps in
        # whole-second (_NS) steps for convenience, unrelated to real frame
        # cadence; tests that specifically exercise staleness/reporting-gap
        # behavior override these.
        "cold_confirm_max_gap_s": 10.0,
        "cold_pending_report_max_gap_s": 10.0,
    }
    if cfg_overrides:
        hc.update(cfg_overrides)
    cfg["fusion_heuristic"] = hc
    # matching.max_plausible_hand_*: zeroed here for the same reason the
    # implausible_jump_*_per_speed rates above are -- this whole file
    # advances timestamps in whole-second (_NS) steps for convenience, not
    # realistic ~11-70ms camera gaps, so the real (nonzero) production
    # values would inject enormous, unrealistic stale-time widening into
    # every multi-frame test in this file. See _implausible_jump_thresholds'
    # own docstring for the mechanism; StaleTimeWideningTests below tests it
    # directly with realistic short (ms-scale) gaps instead.
    cfg["matching"] = {"max_plausible_hand_speed_m_s": 0.0, "max_plausible_hand_ang_speed_deg_s": 0.0}
    f = HeuristicPoseFusionFilter(gyro_data=None, accel_data=None, lever_arm=None,
                                   g_world_estimator=None, cfg=cfg, ctrl_name=ctrl_name)
    return f


def _stub_predict(R_pred, p_pred):
    """Replaces .predict on an instance with a fixed-return stub -- real
    dead-reckoning (src.imu_data) is out of scope for these tests, and its own
    frame-count side effect on frames_since_update is set directly by tests
    instead (see each test's own comment)."""
    return lambda target_ts_ns: (R_pred, p_pred)


class AgreementFactorTests(unittest.TestCase):
    """_agreement_factor returns (agreement_pos, agreement_rot) SEPARATELY
    (changed 2026-09-11, see its own docstring) -- try_update applies each to
    its own axis independently (position pushback additionally gated on
    velocity_established), rather than the method itself collapsing them to
    a single min() the way it used to. The "min, not average" behavior these
    tests originally covered is now the CALLER's responsibility (try_update
    computes `agreement = min(agreement_pos, agreement_rot)` purely for its
    own single debug field) -- still checked here for completeness."""
    def test_full_trust_at_or_below_weak(self):
        f = _make_filter()
        self.assertEqual(f._agreement_factor(0.05, 1.0, 0.1, 0.3, 5.0, 60.0), (1.0, 1.0))
        self.assertEqual(f._agreement_factor(0.1, 1.0, 0.1, 0.3, 5.0, 60.0), (1.0, 1.0))

    def test_zero_trust_at_or_above_strong(self):
        f = _make_filter()
        # rot_innov (1.0) stays well within rot_weak (5.0) in both cases, so
        # only the position component hits zero trust here.
        self.assertEqual(f._agreement_factor(0.3, 1.0, 0.1, 0.3, 5.0, 60.0), (0.0, 1.0))
        self.assertEqual(f._agreement_factor(0.5, 1.0, 0.1, 0.3, 5.0, 60.0), (0.0, 1.0))

    def test_linear_between(self):
        f = _make_filter()
        # pos at the midpoint of [0.1, 0.3] -> 0.5; rotation fully agreeing -> 1.0
        agreement_pos, agreement_rot = f._agreement_factor(0.2, 0.0, 0.1, 0.3, 5.0, 60.0)
        self.assertAlmostEqual(agreement_pos, 0.5, places=6)
        self.assertAlmostEqual(agreement_rot, 1.0, places=6)

    def test_min_not_average_a_bad_rotation_cant_hide_behind_a_good_position(self):
        f = _make_filter()
        # position fully agrees (1.0), rotation fully disagrees (0.0) -- min(...) of
        # the two components is 0.0, NOT the 0.5 an average would give (same
        # reasoning as the hard gate this extends), even though each axis is now
        # applied independently by the caller rather than pre-collapsed here.
        agreement_pos, agreement_rot = f._agreement_factor(0.0, 60.0, 0.1, 0.3, 5.0, 60.0)
        self.assertEqual(agreement_pos, 1.0)
        self.assertEqual(agreement_rot, 0.0)
        self.assertEqual(min(agreement_pos, agreement_rot), 0.0)


class AgreementWeakBoundsTests(unittest.TestCase):
    def test_falls_back_to_a_ramp_hugging_the_hard_gate_ceiling_when_history_thin(self):
        # Not exactly the ceiling itself -- see _agreement_weak_bounds' own
        # docstring on why weak==strong==ceiling would silently disable gating.
        f = _make_filter()
        pos_weak, rot_weak = f._agreement_weak_bounds(0)
        self.assertAlmostEqual(pos_weak, 0.27, places=6)
        self.assertAlmostEqual(rot_weak, 54.0, places=6)

    def test_adapts_to_recent_history_once_populated(self):
        f = _make_filter()
        for _ in range(10):
            f._pos_innov_hist.append(0.01)
            f._rot_innov_hist.append(1.0)
        pos_weak, rot_weak = f._agreement_weak_bounds(0)
        # k_pos=3.0 * p90(0.01) == 0.03, well under the 0.3 ceiling and above the 0.001 floor
        self.assertAlmostEqual(pos_weak, 0.03, places=6)
        self.assertAlmostEqual(rot_weak, 3.0, places=6)

    def test_never_exceeds_hard_gate_ceiling(self):
        f = _make_filter()
        for _ in range(10):
            f._pos_innov_hist.append(10.0)  # absurdly noisy recent history
            f._rot_innov_hist.append(500.0)
        pos_weak, rot_weak = f._agreement_weak_bounds(0)
        self.assertEqual(pos_weak, 0.3)
        self.assertEqual(rot_weak, 60.0)


class CaseAWarmSoftTrustTests(unittest.TestCase):
    def test_small_disagreement_pulls_tracking_state_toward_imu_prediction(self):
        """Case A's whole point: even with cost_weight_imu=0 (this project's
        actual current config -- the cost-function fuse alone would otherwise
        reduce to exactly p_meas, see the module docstring's own note on this),
        a disagreement between the hard gate's weak/strong bounds must still
        visibly pull the tracking state away from raw vision."""
        f = _make_filter()
        # Bootstrap.
        p0 = np.array([0.0, 0.0, 0.0])
        f.try_update(_solution(np.eye(3), p0), 0)

        # Position pushback is gated on velocity_established (added 2026-09-11,
        # see its own comment on HeuristicPoseFusionFilter) -- a bare bootstrap
        # never establishes it (v=0 by construction), so this test poking the
        # filter's internals directly to isolate Case A's own pushback math
        # (same spirit as the frames_since_update poke below) also needs to
        # declare that precondition explicitly rather than simulate a whole
        # extra real accepted frame just to obtain it.
        f.velocity_established = True

        # Give it a tight recent-history baseline so the adaptive weak bound is
        # small (~3mm) -- otherwise thin-history fallback (0.3m ceiling) would
        # swallow the 5cm test disagreement below as "fully trusted".
        for _ in range(10):
            f._pos_innov_hist.append(0.001)
            f._rot_innov_hist.append(0.1)

        R_pred, p_pred = np.eye(3), np.array([0.10, 0.0, 0.0])
        f.predict = _stub_predict(R_pred, p_pred)
        f.frames_since_update = 1  # keeps _frames_lost=0 -> imu_frame_scale=1.0 (warm)

        p_meas = np.array([0.15, 0.0, 0.0])  # 0.05m from p_pred -- between weak (~3mm) and strong (0.3m)
        # n_inliers/error_px kept LOW-quality but nonzero (quality=0 would
        # zero w_vision entirely via _vision_weight's own base*quality
        # formula, collapsing to the w_sum<=0 degenerate fallback -- p_new
        # would start AT p_pred already, before Case A even runs, defeating
        # this test's own setup). The pushback is ALSO scaled by
        # (1-quality) now (2026-09-14) -- see CandidateQualityPushback
        # ScaleTests for that behavior in isolation; this test folds the
        # resulting (nonzero-but-low) quality into its own expected value
        # below instead of assuming it away.
        n_inliers, error_px = 6, 0.4
        ok = f.try_update(_solution(np.eye(3), p_meas, n_inliers=n_inliers, error_px=error_px), 1 * _NS)
        self.assertTrue(ok)

        pos_innov_m = float(np.linalg.norm(p_meas - p_pred))
        pos_weak = float(np.clip(3.0 * np.percentile([0.001] * 10, 90), 0.001, 0.3))
        # _ramp_down(value, weak, strong) wants weak=high/bad, strong=low/good --
        # opposite sense from this test's own pos_weak (small/full-trust bound),
        # see _agreement_factor's own docstring note on this.
        raw_agreement = _ramp_down(pos_innov_m, 0.3, pos_weak)
        self.assertTrue(0.0 < raw_agreement < 1.0, "test setup should land strictly between weak/strong")
        quality = _quality(n_inliers, error_px)
        pushback_scale = 1.0 * (1.0 - quality)  # imu_frame_scale=1.0 (warm) here
        expected_agreement = 1.0 - pushback_scale * (1.0 - raw_agreement)

        expected_p = expected_agreement * p_meas + (1.0 - expected_agreement) * p_pred
        np.testing.assert_allclose(f.p, expected_p, atol=1e-9)
        # And, crucially, the corrected state must differ from raw vision --
        # this is exactly the pushback that didn't exist before Case A.
        self.assertGreater(float(np.linalg.norm(f.p - p_meas)), 1e-6)

    def test_tiny_disagreement_below_weak_bound_is_fully_trusted(self):
        f = _make_filter()
        p0 = np.array([0.0, 0.0, 0.0])
        f.try_update(_solution(np.eye(3), p0), 0)
        for _ in range(10):
            f._pos_innov_hist.append(0.01)
            f._rot_innov_hist.append(1.0)

        R_pred, p_pred = np.eye(3), np.array([0.10, 0.0, 0.0])
        f.predict = _stub_predict(R_pred, p_pred)
        f.frames_since_update = 1

        p_meas = np.array([0.1005, 0.0, 0.0])  # 0.5mm from p_pred -- well under the ~30mm weak bound
        f.try_update(_solution(np.eye(3), p_meas), 1 * _NS)
        np.testing.assert_allclose(f.p, p_meas, atol=1e-9)

    def test_extreme_disagreement_still_hard_rejects_exactly_as_before(self):
        f = _make_filter()
        p0 = np.array([0.0, 0.0, 0.0])
        f.try_update(_solution(np.eye(3), p0), 0)
        # The hard gate's POSITION criterion is gated on velocity_established
        # (added 2026-09-11) -- a bare bootstrap never establishes it, and
        # this test is specifically about the position criterion (rotation is
        # identity throughout), so it needs to declare that precondition
        # explicitly, same as the pushback test above.
        f.velocity_established = True

        R_pred, p_pred = np.eye(3), np.array([0.0, 0.0, 0.0])
        f.predict = _stub_predict(R_pred, p_pred)
        f.frames_since_update = 1

        p_meas = np.array([1.0, 0.0, 0.0])  # 1m -- far past implausible_jump_pos_m=0.3
        ok = f.try_update(_solution(np.eye(3), p_meas), 1 * _NS)
        self.assertFalse(ok)
        np.testing.assert_allclose(f.p, p_pred)  # state left at the IMU prediction, untouched by p_meas


class CaseAImuFrameScaleDecayTests(unittest.TestCase):
    """Regression for scaling Case A's soft pushback by imu_frame_scale
    (2026-09-14, user-directed after a real case: right_controller,
    walk_hard, a reacquisition landing after 3 real TRACKING-LOST frames).
    p_pred there was a multi-frame BLIND coast through a violent gap
    (imu_frame_scale=0.25), yet the hard gate's own accel/gyro-widened
    ceiling (correctly avoiding a false reject) ALSO widened this ramp's
    span enough that a genuinely large (421mm), CORRECT vision disagreement
    landed mid-ramp (agreement_pos=0.4847) and pulled the accepted state
    ~52% of the way back toward that untrustworthy p_pred -- corrupting the
    anchor for several subsequent, correct vision frames (which then
    hard-rejected as "implausible" against the now-wrong anchor). Before
    this fix, imu_frame_scale only ever gated w_imu in the cost blend
    (already 0 for cost_weight_imu=0 configs) -- this ramp had no equivalent
    decay, just a binary imu_frame_scale>0.0 gate. Now the PUSHBACK fraction
    (1-agreement) itself is scaled by imu_frame_scale, so a multi-frame lost
    stretch leans toward trusting vision fully, same direction the cost
    blend already leans -- see _agreement_factor's own call site comment in
    src/pose_fusion_heuristic.py for the exact formula."""

    def _fused_pos(self, frames_since_update):
        f = _make_filter()
        p0 = np.array([0.0, 0.0, 0.0])
        f.try_update(_solution(np.eye(3), p0), 0)
        f.velocity_established = True
        for _ in range(10):
            f._pos_innov_hist.append(0.001)
            f._rot_innov_hist.append(0.1)
        R_pred, p_pred = np.eye(3), np.array([0.10, 0.0, 0.0])
        f.predict = _stub_predict(R_pred, p_pred)
        f.frames_since_update = frames_since_update
        p_meas = np.array([0.15, 0.0, 0.0])  # same 0.05m disagreement as CaseAWarmSoftTrustTests
        # n_inliers/error_px kept LOW-quality but nonzero -- quality=0 would
        # zero w_vision entirely (_vision_weight's own base*quality formula)
        # and collapse to the w_sum<=0 degenerate fallback (p_new starting
        # AT p_pred already, before Case A even runs). The pushback is ALSO
        # scaled by (1-quality) now (2026-09-14) -- this class's own tests
        # fold the resulting quality into their expected values via
        # _quality() below instead of assuming it away.
        ok = f.try_update(_solution(np.eye(3), p_meas, n_inliers=self.N_INLIERS, error_px=self.ERROR_PX), 1 * _NS)
        self.assertTrue(ok)
        return f.p.copy(), p_pred, p_meas

    N_INLIERS, ERROR_PX = 6, 0.4

    def _raw_agreement(self, p_pred, p_meas):
        pos_innov_m = float(np.linalg.norm(p_meas - p_pred))
        pos_weak = float(np.clip(3.0 * np.percentile([0.001] * 10, 90), 0.001, 0.3))
        return _ramp_down(pos_innov_m, 0.3, pos_weak)

    def test_warm_frame_scale_1_matches_unscaled_pushback(self):
        """frames_since_update=1 -> frames_lost=0 -> imu_frame_scale=1.0 --
        the scaling formula (1 - imu_frame_scale*(1-quality)*(1-agreement))
        is a pure imu_frame_scale no-op here (still discounted by quality),
        so behavior must match CaseAWarmSoftTrustTests' own quality-scaled
        expectation -- re-derived here to anchor the imu_frame_scale=1.0
        baseline this test class's own decay tests are relative to."""
        p, p_pred, p_meas = self._fused_pos(frames_since_update=1)
        raw_agreement = self._raw_agreement(p_pred, p_meas)
        self.assertTrue(0.0 < raw_agreement < 1.0, "test setup should land strictly between weak/strong")
        quality = _quality(self.N_INLIERS, self.ERROR_PX)
        pushback_scale = 1.0 * (1.0 - quality)  # imu_frame_scale=1.0
        expected_agreement = 1.0 - pushback_scale * (1.0 - raw_agreement)
        expected_p = expected_agreement * p_meas + (1.0 - expected_agreement) * p_pred
        np.testing.assert_allclose(p, expected_p, atol=1e-9)

    def test_after_three_lost_frames_pushback_shrinks_toward_vision(self):
        """frames_since_update=4 -> frames_lost=3 -> imu_frame_scale=0.25
        (imu_decay_frames=4, matching the real case's own frame-count) --
        the pushback fraction must shrink to 1/4 of its unscaled (but still
        quality-discounted) size, not stay at the same ~50% split a flat
        (imu_frame_scale-blind) ramp would have produced."""
        p, p_pred, p_meas = self._fused_pos(frames_since_update=4)
        raw_agreement = self._raw_agreement(p_pred, p_meas)
        quality = _quality(self.N_INLIERS, self.ERROR_PX)
        imu_frame_scale = 0.25
        pushback_scale = imu_frame_scale * (1.0 - quality)
        scaled_agreement = 1.0 - pushback_scale * (1.0 - raw_agreement)
        expected_p = scaled_agreement * p_meas + (1.0 - scaled_agreement) * p_pred
        np.testing.assert_allclose(p, expected_p, atol=1e-9)
        # The whole point: must land closer to vision than the OLD,
        # imu_frame_scale-blind formula would have.
        unscaled_p = raw_agreement * p_meas + (1.0 - raw_agreement) * p_pred
        self.assertLess(float(np.linalg.norm(p - p_meas)), float(np.linalg.norm(unscaled_p - p_meas)))


class CandidateQualityPushbackScaleTests(unittest.TestCase):
    """Regression for scaling Case A's pushback by candidate quality too
    (2026-09-14, user-directed follow-up on the exact same real case as
    CaseAImuFrameScaleDecayTests, same ts): imu_frame_scale alone still left
    a real, STRONG candidate (right_controller, walk_hard -- n_inliers=14,
    error=0.27px, matched across 3 cameras) with a visible ~14% pull toward
    a demonstrably-bad p_pred (agreement=0.86, not 1.0) purely because
    imu_frame_scale hadn't fully decayed yet (3 of 4 imu_decay_frames
    elapsed). _agreement_factor's own ramp was blind to vision quality
    entirely -- a rock-solid candidate and a marginal one with the same
    pos_innov_m got identical pushback. The pushback fraction is now ALSO
    scaled by (1-quality), using the SAME `quality` _vision_weight itself
    already computes from n_inliers/error_px -- a high-quality candidate
    now gets little/no pushback regardless of imu_frame_scale, while
    imu_frame_scale=1.0 (ordinary warm-state smoothing) still applies its
    old strength for a low-quality candidate."""

    def _fused_pos(self, n_inliers, error_px, frames_since_update=4):
        f = _make_filter()
        p0 = np.array([0.0, 0.0, 0.0])
        f.try_update(_solution(np.eye(3), p0), 0)
        f.velocity_established = True
        for _ in range(10):
            f._pos_innov_hist.append(0.001)
            f._rot_innov_hist.append(0.1)
        R_pred, p_pred = np.eye(3), np.array([0.10, 0.0, 0.0])
        f.predict = _stub_predict(R_pred, p_pred)
        f.frames_since_update = frames_since_update  # 4 -> imu_frame_scale=0.25 (imu_decay_frames=4)
        p_meas = np.array([0.15, 0.0, 0.0])  # same 0.05m disagreement as the sibling test classes
        ok = f.try_update(_solution(np.eye(3), p_meas, n_inliers=n_inliers, error_px=error_px), 1 * _NS)
        self.assertTrue(ok)
        return f.p.copy(), p_pred, p_meas

    def _raw_agreement(self, p_pred, p_meas):
        pos_innov_m = float(np.linalg.norm(p_meas - p_pred))
        pos_weak = float(np.clip(3.0 * np.percentile([0.001] * 10, 90), 0.001, 0.3))
        return _ramp_down(pos_innov_m, 0.3, pos_weak)

    def test_high_quality_candidate_lands_closer_to_vision_than_low_quality(self):
        """Same imu_frame_scale (0.25), same raw disagreement -- a strong
        candidate (the real case's own n_inliers=14/error=0.27px) must land
        measurably closer to vision than a weak one (n_inliers=6/error=0.4)."""
        p_weak, p_pred, p_meas = self._fused_pos(n_inliers=6, error_px=0.4)
        p_strong, _, _ = self._fused_pos(n_inliers=14, error_px=0.27)
        self.assertLess(float(np.linalg.norm(p_strong - p_meas)),
                         float(np.linalg.norm(p_weak - p_meas)))

    def test_matches_the_real_case_own_quality_scaled_formula(self):
        """Direct regression using the real frame's own n_inliers=14/
        error_px=0.27 at imu_frame_scale=0.25 -- ties this test to the exact
        real event this fix was built for, not just a synthetic shape."""
        n_inliers, error_px = 14, 0.27
        p, p_pred, p_meas = self._fused_pos(n_inliers=n_inliers, error_px=error_px)
        raw_agreement = self._raw_agreement(p_pred, p_meas)
        quality = _quality(n_inliers, error_px)
        imu_frame_scale = 0.25
        pushback_scale = imu_frame_scale * (1.0 - quality)
        expected_agreement = 1.0 - pushback_scale * (1.0 - raw_agreement)
        expected_p = expected_agreement * p_meas + (1.0 - expected_agreement) * p_pred
        np.testing.assert_allclose(p, expected_p, atol=1e-9)

    def test_max_quality_candidate_gets_zero_pushback_regardless_of_imu_frame_scale(self):
        """quality=1.0 (past BOTH strong thresholds, this file's own
        _solution default) -> pushback_scale=0 no matter how decayed
        imu_frame_scale is -- full vision trust."""
        n_inliers, error_px = 20, 0.1
        quality = _quality(n_inliers, error_px)
        self.assertAlmostEqual(quality, 1.0, places=6)
        p, p_pred, p_meas = self._fused_pos(n_inliers=n_inliers, error_px=error_px, frames_since_update=4)
        np.testing.assert_allclose(p, p_meas, atol=1e-9)


class SiblingCollisionTests(unittest.TestCase):
    class _FakeSibling:
        def __init__(self, trust_value, predicted, frames_since_update=0):
            self._trust_value = trust_value
            self._predicted = predicted
            # Defaults to 0 (freshly vision-confirmed this exact frame) --
            # matches every real HeuristicPoseFusionFilter sibling on any
            # frame it's tracking normally. See _looks_like_a_sibling's own
            # comment (2026-09-12) for why this freshness gate exists: trust()
            # alone decays with elapsed TIME, not with how far an IMU-coasting
            # sibling's own POSITION may have already drifted in that time.
            self.frames_since_update = frames_since_update

        def trust(self, frame_ts_ns):
            return self._trust_value

        def predict(self, frame_ts_ns):
            return self._predicted

    def test_bootstrap_rejected_when_candidate_collides_with_live_sibling(self):
        f = _make_filter()
        sib = self._FakeSibling(trust_value=0.9, predicted=(np.eye(3), np.array([1.0, 2.0, 3.0])))
        f.set_siblings([sib])

        p_meas = np.array([1.02, 2.0, 3.0])  # 2cm from the sibling -- inside sibling_collision_dist_m=0.15
        ok = f.try_update(_solution(np.eye(3), p_meas), 0)
        self.assertFalse(ok)
        self.assertIsNone(f.R)  # never bootstrapped
        self.assertEqual(f.consecutive_rejects, 1)

    def test_bootstrap_unaffected_by_an_untrustworthy_sibling(self):
        f = _make_filter()
        sib = self._FakeSibling(trust_value=0.05, predicted=(np.eye(3), np.array([1.0, 2.0, 3.0])))
        f.set_siblings([sib])

        p_meas = np.array([1.0, 2.0, 3.0])
        ok = f.try_update(_solution(np.eye(3), p_meas), 0)
        self.assertTrue(ok)
        np.testing.assert_allclose(f.p, p_meas)

    def test_bootstrap_unaffected_by_a_stale_coasting_sibling(self):
        """Real case (frame_range 3850-3950 relative frame 17): a trustworthy
        (trust=0.9) but currently-coasting sibling (frames_since_update > 0,
        i.e. NOT vision-confirmed this exact frame) must not block a
        candidate colliding with its stale, IMU-extrapolated position --
        trust() alone doesn't capture that the sibling's own position may
        have already drifted during the coast."""
        f = _make_filter()
        sib = self._FakeSibling(trust_value=0.9, predicted=(np.eye(3), np.array([1.0, 2.0, 3.0])),
                                 frames_since_update=2)
        f.set_siblings([sib])

        p_meas = np.array([1.02, 2.0, 3.0])  # 2cm from the sibling -- would collide if the sibling were fresh
        ok = f.try_update(_solution(np.eye(3), p_meas), 0)
        self.assertTrue(ok)
        np.testing.assert_allclose(f.p, p_meas)


class ColdReacquireTests(unittest.TestCase):
    def _bootstrapped_cold_filter(self, cfg_overrides=None):
        f = _make_filter(cfg_overrides)
        f.try_update(_solution(np.eye(3), np.array([0.0, 0.0, 0.0])), 0)
        # Stub predict() so try_update doesn't fail-open before reaching the
        # cold-routing check -- its return value is otherwise unused by
        # _try_cold_reacquire (it never calls self.predict()).
        f.predict = _stub_predict(np.eye(3), np.array([0.0, 0.0, 0.0]))
        f.frames_since_update = 1000  # forces imu_frame_scale == 0 -> cold routing
        return f

    def test_single_candidate_is_trusted_immediately(self):
        """No confirmation window any more (removed 2026-09-06) -- a cold
        reacquisition candidate is trusted on the very first frame, same as
        this filter trusts vision everywhere else, once past the sibling
        check."""
        f = self._bootstrapped_cold_filter()
        p_new = np.array([0.5, 0.0, 0.0])
        ok = f.try_update(_solution(np.eye(3), p_new), 1 * _NS)
        self.assertTrue(ok)
        np.testing.assert_allclose(f.p, p_new)
        self.assertEqual(f.last_update_ts_ns, 1 * _NS)
        self.assertEqual(f.frames_since_update, 0)
        self.assertIsNotNone(f.reported_p)  # One Euro filter smooths it, so not necessarily == p_new exactly

    def test_even_a_large_single_frame_jump_is_trusted_when_not_a_sibling_collision(self):
        """Vision is trusted directly in the cold regime -- gate_active=False
        in _vision_weight already means no implausible_jump_pos_m/_rot_deg
        check applies here, only the sibling-collision check does."""
        f = self._bootstrapped_cold_filter()
        p_far = np.array([5.0, 0.0, 0.0])  # far beyond implausible_jump_pos_m -- irrelevant here
        ok = f.try_update(_solution(np.eye(3), p_far), 1 * _NS)
        self.assertTrue(ok)
        np.testing.assert_allclose(f.p, p_far)

    def test_sibling_collision_still_rejects(self):
        f = self._bootstrapped_cold_filter()
        sib = SiblingCollisionTests._FakeSibling(trust_value=0.9,
                                                  predicted=(np.eye(3), np.array([9.0, 9.0, 9.0])))
        f.set_siblings([sib])

        ok = f.try_update(_solution(np.eye(3), np.array([9.01, 9.0, 9.0])), 1 * _NS)
        self.assertFalse(ok)
        self.assertEqual(f.consecutive_rejects, 1)
        # Rejected -- tracking state stays at whatever it was before (the
        # bootstrap value from _bootstrapped_cold_filter), not the colliding candidate.
        np.testing.assert_allclose(f.p, [0.0, 0.0, 0.0])

    def test_max_coast_s_keeps_ticking_until_a_candidate_is_trusted(self):
        f = self._bootstrapped_cold_filter()
        self.assertFalse(f.should_force_cold_start(int(0.05 * _NS)))  # max_coast_s=1.0, well within
        self.assertTrue(f.should_force_cold_start(int(2.0 * _NS)))    # 2s since last_update_ts_ns=0
        # Once a candidate is trusted, last_update_ts_ns advances and the
        # clock resets.
        f.try_update(_solution(np.eye(3), np.array([0.5, 0.0, 0.0])), int(0.05 * _NS))
        self.assertFalse(f.should_force_cold_start(int(0.9 * _NS)))


class RotationSeedGraceFramesTests(unittest.TestCase):
    """Regression for a real false-reject found 2026-09-13 (a real
    recording's frame 66->67): a coverage_fallback seed's rotation-
    implausibility exemption used to clear the INSTANT the very next accept
    landed (as long as THAT accept wasn't itself coverage_fallback) -- but
    that accept's own rotation was itself gate-exempted (never checked
    against anything, since the gate was off for it too), so promoting it
    to a trusted hard-gate REFERENCE one frame later hard-rejected a
    genuinely correct vision candidate purely because gyro had only one
    un-vetted, reacquisition-fresh frame's worth of orientation to
    integrate from (confirmed via real mocap ground truth: vision was
    correct the whole time). Fix: rotation_seed_grace_frames (default 2,
    was implicitly 1) gives that frame one more grace frame before the hard
    gate re-arms against it."""

    def _cold_reacquired_filter(self, cfg_overrides=None):
        """Bootstrapped, then forced into the cold-reacquire regime and
        reacquired via a coverage_fallback candidate -- mirrors the real
        sequence (a normal bootstrap, later a coverage-fallback cold
        reacquisition) that seeds _rotation_seed_grace_frames.

        coverage_fallback is, by construction, always "weak" under
        ColdReacquireWeakBufferTests' new buffering (see that class) -- so
        reaching the accepted cold-reacquire state now takes the SAME weak
        candidate twice (buffered, then self-confirmed) rather than one
        immediate accept. This shifts every subsequent frame in this class's
        own tests by one full step (was 1*_NS/2*_NS/..., now 2*_NS/3*_NS/...)
        -- deliberate, matches the real sequence one frame later, not a
        loosening of what's under test."""
        f = _make_filter(cfg_overrides)
        f.try_update(_solution(np.eye(3), np.array([0.0, 0.0, 0.0])), 0)
        f.predict = _stub_predict(np.eye(3), np.array([0.0, 0.0, 0.0]))
        f.frames_since_update = 1000  # forces imu_frame_scale == 0 -> cold routing
        cold_solution = {**_solution(np.eye(3), np.array([1.0, 0.0, 0.0])),
                          "coverage_fallback": True}
        ok0 = f.try_update(cold_solution, 1 * _NS)
        self.assertFalse(ok0, "a weak/coverage_fallback candidate must be buffered, not trusted immediately")
        ok = f.try_update(cold_solution, 2 * _NS)  # self-confirming (same pose) -> accepted
        self.assertTrue(ok)
        return f

    def test_grace_frames_default_to_two_after_a_coverage_fallback_reacquire(self):
        f = self._cold_reacquired_filter()
        self.assertEqual(f._rotation_seed_grace_frames, 2)
        self.assertFalse(f.velocity_established)

    def test_second_frame_after_seed_is_also_exempted_not_just_the_first(self):
        """The actual regression: frame N+1 (right after the coverage-
        fallback seed) is exempted (pre-existing behavior, unchanged) -- but
        frame N+2 (evaluated against N+1's own, still-unvetted rotation)
        must ALSO be exempted under the fix, not hard-rejected."""
        f = self._cold_reacquired_filter()
        R_wild = Rotation.from_euler('z', 150, degrees=True).as_matrix()
        # Frame N+1: gyro-only prediction stub (unchanged from the cold-
        # reacquire anchor) vs a wildly different rotation -- must be
        # accepted (grace=2, matches pre-existing behavior).
        f.predict = _stub_predict(np.eye(3), np.array([1.0, 0.0, 0.0]))
        ok1 = f.try_update(_solution(R_wild, np.array([1.0, 0.0, 0.0])), 3 * _NS)
        self.assertTrue(ok1, "frame right after the seed must still be exempted")
        self.assertEqual(f._rotation_seed_grace_frames, 1)
        self.assertFalse(f.velocity_established, "this step's own velocity must not be trusted either")

        # Frame N+2: predict() now gyro-integrates "forward" from R_wild --
        # another wildly different rotation must STILL be accepted
        # (grace=1 -> 0): this is the actual fix under test.
        R_wild_2 = Rotation.from_euler('z', -150, degrees=True).as_matrix()
        f.predict = _stub_predict(R_wild, np.array([1.0, 0.0, 0.0]))
        ok2 = f.try_update(_solution(R_wild_2, np.array([1.0, 0.0, 0.0])), 4 * _NS)
        self.assertTrue(ok2, "the fix: one more frame of grace before the gate re-arms")
        self.assertEqual(f._rotation_seed_grace_frames, 0)

    def test_gate_re_arms_once_grace_is_exhausted(self):
        """After exactly rotation_seed_grace_frames genuinely-blended-in
        accepts, the hard gate must re-arm and reject a real implausible
        jump -- the exemption is temporary, not permanent."""
        f = self._cold_reacquired_filter()
        f.predict = _stub_predict(np.eye(3), np.array([1.0, 0.0, 0.0]))
        f.try_update(_solution(Rotation.from_euler('z', 150, degrees=True).as_matrix(),
                                np.array([1.0, 0.0, 0.0])), 3 * _NS)
        f.predict = _stub_predict(np.eye(3), np.array([1.0, 0.0, 0.0]))
        f.try_update(_solution(Rotation.from_euler('z', -150, degrees=True).as_matrix(),
                                np.array([1.0, 0.0, 0.0])), 4 * _NS)
        self.assertEqual(f._rotation_seed_grace_frames, 0)

        p_before = f.p.copy()
        f.predict = _stub_predict(np.eye(3), np.array([1.0, 0.0, 0.0]))
        ok = f.try_update(_solution(Rotation.from_euler('z', 150, degrees=True).as_matrix(),
                                     np.array([1.0, 0.0, 0.0])), 5 * _NS)
        self.assertFalse(ok, "grace exhausted -- the hard gate must be fully active again")
        np.testing.assert_allclose(f.p, p_before)  # state left at the IMU prediction, unchanged

    def test_a_coverage_fallback_frame_within_the_grace_window_resets_it(self):
        """If the frame consumed during grace is ITSELF coverage_fallback
        (still too few LEDs), the countdown must reset to full, not
        continue counting down toward a self.R that's STILL unvetted --
        this is the ORIGINAL (already-fixed, 2026-09-13 before this
        session) bug this mechanism protects against, still covered under
        the new counter-based implementation."""
        f = self._cold_reacquired_filter()
        f.predict = _stub_predict(np.eye(3), np.array([1.0, 0.0, 0.0]))
        cov_fb_2 = {**_solution(Rotation.from_euler('z', 150, degrees=True).as_matrix(),
                                 np.array([1.0, 0.0, 0.0])), "coverage_fallback": True}
        ok = f.try_update(cov_fb_2, 3 * _NS)
        self.assertTrue(ok)
        self.assertEqual(f._rotation_seed_grace_frames, 2,
                          "a still-fallback candidate must reset, not decrement, the grace window")

    def test_rotation_seed_grace_frames_is_config_tunable(self):
        f = self._cold_reacquired_filter({"rotation_seed_grace_frames": 5})
        self.assertEqual(f._rotation_seed_grace_frames, 5)


class ColdReacquireWeakBufferTests(unittest.TestCase):
    """Regression for the most common real failure shape (user-reported,
    2026-09-13, "64->65" / "66->67"): a controller at the edge of camera
    view, with only 4-5 of its LEDs physically visible, produces a weak
    (coverage_fallback and/or low-n_inliers) reacquisition candidate after a
    long track loss. _try_cold_reacquire used to trust that immediately as
    the new anchor -- so the very next frame's strong, correct match then
    read as a huge "jump" against a bad anchor and paid a rejection/grace-
    window cost a cleaner anchor would never have needed.

    Fix (user-proposed design): a weak candidate is buffered in
    self._cold_pending instead of accepted. Resolved on the NEXT candidate:
    strong supersedes and discards the buffered weak one; a second weak
    candidate that AGREES with the buffered one (no jump) confirms and gets
    accepted; a second weak candidate that DISAGREES (a jump) discards the
    first and re-buffers the second, still not accepting anything."""

    def _cold_regime_filter(self, cfg_overrides=None):
        """Bootstrapped, then forced into the cold-reacquire regime (no
        reacquisition yet) -- every test below drives _try_cold_reacquire
        itself from here."""
        f = _make_filter(cfg_overrides)
        f.try_update(_solution(np.eye(3), np.array([0.0, 0.0, 0.0])), 0)
        f.predict = _stub_predict(np.eye(3), np.array([0.0, 0.0, 0.0]))
        f.frames_since_update = 1000  # forces imu_frame_scale == 0 -> cold routing
        return f

    def test_weak_candidate_is_buffered_not_accepted(self):
        f = self._cold_regime_filter()
        p_before = f.p.copy()
        weak = _solution(np.eye(3), np.array([1.0, 0.0, 0.0]), n_inliers=4)
        ok = f.try_update(weak, 1 * _NS)
        self.assertFalse(ok, "a weak (low-n_inliers) reacquisition candidate must not be trusted immediately")
        self.assertIsNotNone(f._cold_pending)
        np.testing.assert_allclose(f._cold_pending["p"], np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(f.p, p_before, err_msg="tracking state must be untouched while buffering")
        self.assertEqual(f._last.get("outcome"), "cold_pending")
        self.assertEqual(f.consecutive_rejects, 0, "buffering is a deliberate wait, not a reject")

    def test_coverage_fallback_also_counts_as_weak_even_with_many_inliers(self):
        """coverage_fallback (pose_search.py's own confidence=0.0 marker) is
        weak regardless of n_inliers -- too few of the GEOMETRICALLY
        EXPECTED LEDs matched is the signal, not raw inlier count alone."""
        f = self._cold_regime_filter()
        weak = {**_solution(np.eye(3), np.array([1.0, 0.0, 0.0]), n_inliers=20), "coverage_fallback": True}
        ok = f.try_update(weak, 1 * _NS)
        self.assertFalse(ok)
        self.assertIsNotNone(f._cold_pending)

    def test_strong_candidate_supersedes_buffered_weak_one(self):
        f = self._cold_regime_filter()
        weak = _solution(np.eye(3), np.array([1.0, 0.0, 0.0]), n_inliers=4)
        f.try_update(weak, 1 * _NS)
        self.assertIsNotNone(f._cold_pending)

        strong = _solution(np.eye(3), np.array([5.0, 0.0, 0.0]), n_inliers=20)
        ok = f.try_update(strong, 2 * _NS)
        self.assertTrue(ok, "a strong match must still be trusted immediately, same as before this fix")
        np.testing.assert_allclose(f.p, np.array([5.0, 0.0, 0.0]),
                                    err_msg="the strong candidate's own pose must win, not the discarded weak one")
        self.assertIsNone(f._cold_pending, "the superseded weak candidate must be discarded")
        self.assertEqual(f.last_update_ts_ns, 2 * _NS)

    def test_two_agreeing_weak_candidates_confirm_and_accept(self):
        f = self._cold_regime_filter()
        weak1 = _solution(np.eye(3), np.array([1.000, 0.0, 0.0]), n_inliers=4)
        ok1 = f.try_update(weak1, 1 * _NS)
        self.assertFalse(ok1)

        weak2 = _solution(np.eye(3), np.array([1.010, 0.0, 0.0]), n_inliers=5)  # 10mm away -- agrees
        ok2 = f.try_update(weak2, 2 * _NS)
        self.assertTrue(ok2, "two independent weak solves landing close together must confirm each other")
        np.testing.assert_allclose(f.p, np.array([1.010, 0.0, 0.0]),
                                    err_msg="the newer (confirming) candidate's own pose is the anchor")
        self.assertIsNone(f._cold_pending)
        self.assertEqual(f.last_update_ts_ns, 2 * _NS)
        self.assertFalse(f.velocity_established, "still no real two-point velocity measurement, same as any cold reacquire")

    def test_two_disagreeing_weak_candidates_discard_first_and_rebuffer_second(self):
        f = self._cold_regime_filter()
        weak1 = _solution(np.eye(3), np.array([1.0, 0.0, 0.0]), n_inliers=4)
        ok1 = f.try_update(weak1, 1 * _NS)
        self.assertFalse(ok1)

        weak2 = _solution(np.eye(3), np.array([9.0, 0.0, 0.0]), n_inliers=4)  # 8m jump vs weak1
        ok2 = f.try_update(weak2, 2 * _NS)
        self.assertFalse(ok2, "disagreeing weak candidates must not compound into an accept")
        self.assertIsNotNone(f._cold_pending)
        np.testing.assert_allclose(f._cold_pending["p"], np.array([9.0, 0.0, 0.0]),
                                    err_msg="weak1 must be discarded outright, weak2 buffered in its place")
        self.assertEqual(f.consecutive_rejects, 0)

        # A third weak candidate agreeing with weak2 (not weak1) must now confirm.
        weak3 = _solution(np.eye(3), np.array([9.010, 0.0, 0.0]), n_inliers=4)
        ok3 = f.try_update(weak3, 3 * _NS)
        self.assertTrue(ok3, "weak2 must still be the live buffer entry, not weak1")
        np.testing.assert_allclose(f.p, np.array([9.010, 0.0, 0.0]))

    def test_stale_buffered_candidate_expires_instead_of_confirming(self):
        f = self._cold_regime_filter({"cold_confirm_max_gap_s": 0.1})
        weak1 = _solution(np.eye(3), np.array([1.000, 0.0, 0.0]), n_inliers=4)
        ok1 = f.try_update(weak1, 1 * _NS)
        self.assertFalse(ok1)

        # Same position (would trivially "agree"), but 0.2s later -- past the
        # 0.1s max age -- must be treated as a fresh buffer start, not a confirm.
        weak2 = _solution(np.eye(3), np.array([1.000, 0.0, 0.0]), n_inliers=4)
        late_ts = 1 * _NS + int(0.2 * _NS)
        ok2 = f.try_update(weak2, late_ts)
        self.assertFalse(ok2, "an expired buffered candidate must not confirm a new one")
        self.assertIsNotNone(f._cold_pending)
        self.assertEqual(f._cold_pending["frame_ts_ns"], late_ts)

    def test_weak_buffer_is_cleared_on_reset(self):
        f = self._cold_regime_filter()
        f.try_update(_solution(np.eye(3), np.array([1.0, 0.0, 0.0]), n_inliers=4), 1 * _NS)
        self.assertIsNotNone(f._cold_pending)
        f.reset()
        self.assertIsNone(f._cold_pending)

    def test_cold_confirm_max_gap_s_is_config_tunable(self):
        f = self._cold_regime_filter({"cold_confirm_max_gap_s": 999.0})
        weak1 = _solution(np.eye(3), np.array([1.000, 0.0, 0.0]), n_inliers=4)
        f.try_update(weak1, 1 * _NS)
        weak2 = _solution(np.eye(3), np.array([1.000, 0.0, 0.0]), n_inliers=4)
        late_ts = 1 * _NS + int(10 * _NS)  # would have expired under the 0.1s/0.25s defaults
        ok = f.try_update(weak2, late_ts)
        self.assertTrue(ok, "a much larger configured max age must keep the buffered candidate alive this long")

    def test_confirm_check_is_speed_scaled_not_flat(self):
        """Regression for the second-pass review finding: two buffered weak
        candidates can legitimately be well apart in time (up to
        cold_confirm_max_gap_s), so a jump too big for the flat
        implausible_jump_pos_m floor alone must still confirm if it's
        plausible for the ELAPSED time between them at cold_confirm_max_speed_m_s."""
        f = self._cold_regime_filter()
        weak1 = _solution(np.eye(3), np.array([1.0, 0.0, 0.0]), n_inliers=4)
        ok1 = f.try_update(weak1, 1 * _NS)
        self.assertFalse(ok1)

        # 0.5s later, 0.4m away -- fails the flat 0.3m floor alone, but well
        # within the speed budget (max_speed_m_s=5.0 * 0.5s = 2.5m).
        weak2 = _solution(np.eye(3), np.array([1.4, 0.0, 0.0]), n_inliers=4)
        ts2 = 1 * _NS + int(0.5 * _NS)
        ok2 = f.try_update(weak2, ts2)
        self.assertTrue(ok2, "a jump too large for the flat floor alone must still confirm if plausible for the elapsed gap")
        np.testing.assert_allclose(f.p, np.array([1.4, 0.0, 0.0]))

    def test_confirm_check_still_rejects_a_jump_too_fast_even_scaled(self):
        f = self._cold_regime_filter()
        weak1 = _solution(np.eye(3), np.array([1.0, 0.0, 0.0]), n_inliers=4)
        ok1 = f.try_update(weak1, 1 * _NS)
        self.assertFalse(ok1)

        # 0.1s later, 10m away -- budget is max(0.3, 5.0*0.1)=0.5m, nowhere close.
        weak2 = _solution(np.eye(3), np.array([11.0, 0.0, 0.0]), n_inliers=4)
        ts2 = 1 * _NS + int(0.1 * _NS)
        ok2 = f.try_update(weak2, ts2)
        self.assertFalse(ok2, "a jump implausible even at the max configured speed must still be rejected")
        self.assertIsNotNone(f._cold_pending)
        np.testing.assert_allclose(f._cold_pending["p"], np.array([11.0, 0.0, 0.0]),
                                    err_msg="weak1 discarded, weak2 buffered in its place, same as any disagreement")

    def test_reported_pose_updates_to_coasted_prediction_while_buffering_a_short_loss(self):
        f = self._cold_regime_filter({"cold_pending_report_max_gap_s": 0.25})
        f.predict = _stub_predict(np.eye(3), np.array([0.5, 0.5, 0.5]))
        reported_before = f.reported_p.copy()

        weak = _solution(np.eye(3), np.array([1.0, 0.0, 0.0]), n_inliers=4)
        ok = f.try_update(weak, int(0.1 * _NS))  # dt_s=0.1s since bootstrap's ts=0 anchor -- within budget
        self.assertFalse(ok)
        self.assertFalse(np.allclose(f.reported_p, reported_before),
                          "a short loss's coasted IMU prediction must still be reported while buffering")

    def test_reported_pose_cleared_while_buffering_a_long_loss(self):
        """Regression (2026-09-13, second refinement, real case): past
        cold_pending_report_max_gap_s, reported_R/reported_p are now
        explicitly CLEARED to None, not left frozen at their pre-loss value
        (the original, now-superseded behavior this test used to assert --
        see _report_if_still_usable's own docstring point 2b for the real
        trace this closes: a controller lost for long enough that even
        ControllerTracker's OWN separate imu_only_predicted_pose budget had
        expired and correctly hidden it, then a weak candidate arrived past
        report_max_gap_s -- the stale pre-loss reported pose was still
        sitting there un-cleared, so main.py drew a "trusted-looking" 3D
        model at a position from well before the loss even started, with
        nothing indicating it was stale). Clearing lets main.py's own
        existing T_world_ctrl-is-None check hide the controller instead."""
        f = self._cold_regime_filter({"cold_pending_report_max_gap_s": 0.25})
        f.predict = _stub_predict(np.eye(3), np.array([0.5, 0.5, 0.5]))
        self.assertIsNotNone(f.reported_p, "sanity: a real pre-loss reported pose exists before this update")

        weak = _solution(np.eye(3), np.array([1.0, 0.0, 0.0]), n_inliers=4)
        ok = f.try_update(weak, int(0.5 * _NS))  # dt_s=0.5s since bootstrap's ts=0 anchor -- past budget
        self.assertFalse(ok)
        self.assertIsNone(f.reported_p, "a long-enough loss's stale pre-loss pose must be cleared, not kept")
        self.assertIsNone(f.reported_R)


class ImplausibleJumpThresholdsTests(unittest.TestCase):
    """Regression for 2026-09-13 (full jump-check audit, user-directed):
    _implausible_jump_thresholds() replaces the old flat
    implausible_jump_pos_m(0.3)/implausible_jump_rot_deg(60.0) with a
    speed-scaled formula (base + per_speed * |self.v|), ported from
    controller.py's own already-validated vs-predicted_pose shape. Position
    is only ever consulted once velocity_established is True (see
    try_update's own _pos_implausible), so it's fine for this method itself
    to just report base-only there -- what matters is rotation's own
    explicit fallback, since ITS hard-gate has no velocity precondition and
    must not silently lose coverage."""

    def _filter_with_rates(self):
        return _make_filter({
            "implausible_jump_pos_thresh_base_mm": 100.0,
            "implausible_jump_pos_thresh_per_speed_mm_s": 50.0,
            "implausible_jump_rot_thresh_base_deg": 10.0,
            "implausible_jump_rot_thresh_per_speed_deg_s": 5.0,
        })

    def test_base_only_when_velocity_not_established(self):
        f = self._filter_with_rates()
        self.assertFalse(f.velocity_established)
        pos_m, rot_deg = f._implausible_jump_thresholds(0)
        self.assertAlmostEqual(pos_m, 0.1)
        self.assertAlmostEqual(rot_deg, 10.0)

    def test_thresholds_grow_with_established_velocity(self):
        f = self._filter_with_rates()
        f.velocity_established = True
        f.v = np.array([3.0, 4.0, 0.0])  # |v| = 5.0 m/s
        pos_m, rot_deg = f._implausible_jump_thresholds(0)
        self.assertAlmostEqual(pos_m, (100.0 + 50.0 * 5.0) / 1000.0)
        self.assertAlmostEqual(rot_deg, 10.0 + 5.0 * 5.0)

    def test_rotation_still_gets_a_real_ceiling_without_velocity(self):
        """The precondition-split this method exists for: rotation's own
        hard-gate runs even when velocity_established is False (gyro
        integration has no velocity precondition) -- confirms that regime
        still gets a real, non-zero, non-infinite ceiling (base-only), not
        one that's silently zeroed or skipped just because speed_m_s falls
        back to 0."""
        f = self._filter_with_rates()
        _, rot_deg = f._implausible_jump_thresholds(0)
        self.assertGreater(rot_deg, 0.0)
        self.assertEqual(rot_deg, 10.0)  # exactly the base -- no speed term leaking in


class WeakBootstrapBufferTests(unittest.TestCase):
    """Regression for the user-reported "frames 2-5, huge jumps" real case
    (2026-09-13): a controller barely visible at true session start produced
    5+ consecutive coverage_fallback candidates. The FIRST one became the
    trusted bootstrap anchor immediately and unconditionally (try_update's
    old bootstrap branch had no weak/strong routing at all -- see git
    history), and because every subsequent frame was ALSO coverage_fallback,
    rotation_seed_grace_frames kept resetting to full forever, permanently
    suppressing both hard gates -- so vision's own frame-to-frame rotation
    swung 147-173deg, fully accepted, for several consecutive frames.

    Fix: try_update's bootstrap branch (self.R is None) now calls
    _try_cold_reacquire directly instead of accepting unconditionally --
    confirmed safe to reuse as-is (its weak-routing/buffer/confirm logic
    never reads self.R/self.p, only _accept_cold_reacquire writes them), with
    R_pred=p_pred=dt_s=None (nothing to coast-report -- there's no prior
    anchor at bootstrap) and bootstrap-specific log_prefix/outcome labels so
    debug output doesn't misleadingly say "COLD REACQUIRE" for a first-ever
    detection."""

    def _fresh_filter(self, cfg_overrides=None):
        return _make_filter(cfg_overrides)

    def test_weak_bootstrap_candidate_is_buffered_not_accepted(self):
        f = self._fresh_filter()
        weak = _solution(np.eye(3), np.array([1.0, 2.0, 3.0]), n_inliers=4)
        ok = f.try_update(weak, 1 * _NS)
        self.assertFalse(ok, "a weak first-ever candidate must not be trusted immediately as the bootstrap anchor")
        self.assertIsNone(f.R, "no tracking state should exist yet")
        self.assertIsNotNone(f._cold_pending)
        np.testing.assert_allclose(f._cold_pending["p"], np.array([1.0, 2.0, 3.0]))
        self.assertEqual(f._last.get("outcome"), "bootstrap_pending",
                          "outcome must be bootstrap-labeled, not the cold-reacquire one")

    def test_coverage_fallback_bootstrap_also_counts_as_weak(self):
        f = self._fresh_filter()
        weak = {**_solution(np.eye(3), np.array([1.0, 0.0, 0.0]), n_inliers=20), "coverage_fallback": True}
        ok = f.try_update(weak, 1 * _NS)
        self.assertFalse(ok)
        self.assertIsNone(f.R)
        self.assertIsNotNone(f._cold_pending)

    def test_strong_bootstrap_candidate_still_accepted_immediately(self):
        """Unchanged behavior for the common case -- a strong first-ever
        candidate bootstraps exactly as before this fix."""
        f = self._fresh_filter()
        strong = _solution(np.eye(3), np.array([1.0, 2.0, 3.0]), n_inliers=20)
        ok = f.try_update(strong, 1 * _NS)
        self.assertTrue(ok)
        np.testing.assert_allclose(f.p, np.array([1.0, 2.0, 3.0]))
        self.assertEqual(f._last.get("outcome"), "bootstrap")

    def test_strong_candidate_supersedes_a_buffered_weak_bootstrap(self):
        f = self._fresh_filter()
        weak = _solution(np.eye(3), np.array([1.0, 0.0, 0.0]), n_inliers=4)
        f.try_update(weak, 1 * _NS)
        self.assertIsNone(f.R)

        strong = _solution(np.eye(3), np.array([5.0, 0.0, 0.0]), n_inliers=20)
        ok = f.try_update(strong, 2 * _NS)
        self.assertTrue(ok, "a strong match must still bootstrap immediately, discarding the buffered weak guess")
        np.testing.assert_allclose(f.p, np.array([5.0, 0.0, 0.0]))
        self.assertIsNone(f._cold_pending)
        self.assertEqual(f._last.get("outcome"), "bootstrap")

    def test_two_agreeing_weak_bootstrap_candidates_confirm_and_bootstrap(self):
        f = self._fresh_filter()
        weak1 = _solution(np.eye(3), np.array([1.000, 0.0, 0.0]), n_inliers=4)
        ok1 = f.try_update(weak1, 1 * _NS)
        self.assertFalse(ok1)
        self.assertIsNone(f.R)

        weak2 = _solution(np.eye(3), np.array([1.010, 0.0, 0.0]), n_inliers=5)  # 10mm away -- agrees
        ok2 = f.try_update(weak2, 2 * _NS)
        self.assertTrue(ok2, "two independent weak solves landing close together must confirm each other")
        np.testing.assert_allclose(f.p, np.array([1.010, 0.0, 0.0]))
        self.assertIsNone(f._cold_pending)
        self.assertEqual(f._last.get("outcome"), "bootstrap")
        self.assertFalse(f.velocity_established)

    def test_two_disagreeing_weak_bootstrap_candidates_discard_first_and_rebuffer(self):
        """Direct regression for the real "frames 2-5" shape: repeated
        mutually-disagreeing weak candidates must keep waiting (self.R stays
        None), not each get trusted as a fresh ungated anchor."""
        f = self._fresh_filter()
        weak1 = _solution(np.eye(3), np.array([1.0, 0.0, 0.0]), n_inliers=4)
        f.try_update(weak1, 1 * _NS)
        self.assertIsNone(f.R)

        weak2 = _solution(np.eye(3), np.array([9.0, 0.0, 0.0]), n_inliers=4)  # 8m jump vs weak1
        ok2 = f.try_update(weak2, 2 * _NS)
        self.assertFalse(ok2)
        self.assertIsNone(f.R, "still no committed state after a second disagreeing weak candidate")
        np.testing.assert_allclose(f._cold_pending["p"], np.array([9.0, 0.0, 0.0]))

        weak3 = _solution(np.eye(3), np.array([9.010, 0.0, 0.0]), n_inliers=4)
        ok3 = f.try_update(weak3, 3 * _NS)
        self.assertTrue(ok3, "the third candidate, agreeing with the second (not the first), must finally bootstrap")
        np.testing.assert_allclose(f.p, np.array([9.010, 0.0, 0.0]))

    def test_bootstrap_sibling_rejection_still_uses_bootstrap_wording(self):
        """The sibling-collision check is shared with _try_cold_reacquire,
        but the resulting outcome must still read as a bootstrap event, not
        get silently repurposed by the reuse."""
        f = self._fresh_filter()
        sib = SiblingCollisionTests._FakeSibling(trust_value=0.9, predicted=(np.eye(3), np.array([1.0, 2.0, 3.0])))
        f.set_siblings([sib])
        colliding = _solution(np.eye(3), np.array([1.02, 2.0, 3.0]))  # within sibling_collision_dist_m=0.15
        ok = f.try_update(colliding, 0)
        self.assertFalse(ok)
        self.assertIsNone(f.R)
        self.assertEqual(f._last.get("outcome"), "sibling_rejected")


class ContestedWinnerWeakRoutingTests(unittest.TestCase):
    """Regression for winner_was_contested as a THIRD "weak" trigger
    (2026-09-13), alongside coverage_fallback/low-inlier-count -- see
    TrackingSystem._resolve_cold_conflicts' own contested_winners docstring
    for the real case: a 6-inlier/0.32px bootstrap (NOT coverage_fallback,
    inliers comfortably above vision_weight_weak_inliers) won a shared-blob
    conflict with a physically-close sibling controller and was 132.7deg/
    1.63m wrong vs mocap ground truth -- the solve itself looked completely
    ordinary; only the fact that another controller's candidate contested
    the same evidence this frame flagged it. Routes through the exact same
    weak-pair buffer/confirm machinery WeakBootstrapBufferTests covers."""

    def _fresh_filter(self, cfg_overrides=None):
        return _make_filter(cfg_overrides)

    def test_contested_bootstrap_is_buffered_despite_strong_inlier_count(self):
        f = self._fresh_filter()
        contested = {**_solution(np.eye(3), np.array([1.0, 2.0, 3.0]), n_inliers=20),
                     "winner_was_contested": True}
        ok = f.try_update(contested, 1 * _NS)
        self.assertFalse(ok, "a contested winner must not be trusted immediately, even with a strong inlier count")
        self.assertIsNone(f.R, "no tracking state should exist yet")
        self.assertIsNotNone(f._cold_pending)
        self.assertEqual(f._last.get("outcome"), "bootstrap_pending")

    def test_uncontested_strong_bootstrap_unaffected(self):
        """winner_was_contested defaults to False when absent from the
        solution dict (e.g. every ControllerTracker.update()/update_warm_
        batch call site, which never runs cross-controller conflict
        resolution at all) -- must not change today's common-case
        behavior."""
        f = self._fresh_filter()
        strong = _solution(np.eye(3), np.array([1.0, 2.0, 3.0]), n_inliers=20)
        self.assertNotIn("winner_was_contested", strong)
        ok = f.try_update(strong, 1 * _NS)
        self.assertTrue(ok)
        self.assertEqual(f._last.get("outcome"), "bootstrap")

    def test_two_agreeing_contested_candidates_still_confirm_and_bootstrap(self):
        f = self._fresh_filter()
        c1 = {**_solution(np.eye(3), np.array([1.000, 0.0, 0.0]), n_inliers=20),
              "winner_was_contested": True}
        ok1 = f.try_update(c1, 1 * _NS)
        self.assertFalse(ok1)

        c2 = {**_solution(np.eye(3), np.array([1.010, 0.0, 0.0]), n_inliers=20),  # 10mm away -- agrees
              "winner_was_contested": True}
        ok2 = f.try_update(c2, 2 * _NS)
        self.assertTrue(ok2, "two independent contested-but-agreeing solves must still confirm each other")
        np.testing.assert_allclose(f.p, np.array([1.010, 0.0, 0.0]))
        self.assertEqual(f._last.get("outcome"), "bootstrap")


class OneEuroBypassTests(unittest.TestCase):
    def test_disabled_reports_raw_tracking_state_unsmoothed(self):
        f = _make_filter({"one_euro_enabled": False})
        p0 = np.array([1.0, 2.0, 3.0])
        f.try_update(_solution(np.eye(3), p0), 0)
        np.testing.assert_allclose(f.reported_p, p0)

        R_pred, p_pred = np.eye(3), np.array([1.0, 2.0, 3.0])
        f.predict = _stub_predict(R_pred, p_pred)
        f.frames_since_update = 1
        p_meas = np.array([1.001, 2.0, 3.0])  # tiny step -- One Euro would normally lag/smooth this
        f.try_update(_solution(np.eye(3), p_meas), 1 * _NS)
        np.testing.assert_allclose(f.reported_p, f.p)  # reported == tracking state exactly, no filter lag

    def test_enabled_by_default_still_smooths(self):
        f = _make_filter()  # one_euro_enabled defaults to True
        p0 = np.array([0.0, 0.0, 0.0])
        f.try_update(_solution(np.eye(3), p0), 0)
        R_pred, p_pred = np.eye(3), np.array([0.0, 0.0, 0.0])
        f.predict = _stub_predict(R_pred, p_pred)
        f.frames_since_update = 1
        # A modest, plausible step (5cm/20ms = 2.5 m/s) -- well under
        # implausible_jump_pos_m=0.3 (so it's genuinely accepted, not
        # hard-rejected back to p_pred, which would trivially make
        # reported_p == f.p for an unrelated reason).
        p_meas = np.array([0.05, 0.0, 0.0])
        dt_ns = int(0.02 * _NS)  # a realistic ~20ms frame-to-frame gap
        f.try_update(_solution(np.eye(3), p_meas), dt_ns)
        # A One Euro filter never overshoots to exactly the new raw value on
        # the very first real step -- some lag should be visible.
        self.assertFalse(np.allclose(f.reported_p, f.p))


class HeadsetCorrectionTests(unittest.TestCase):
    """Wiring/branching coverage for the headset-ego-motion correction added to predict()
    (see src.imu_data.predict_headset_relative_pose / src.mocap_data.world_pose). The underlying
    physics/math is covered exhaustively in tests.test_predict_headset_relative_pose and
    tests.test_mocap_data -- these tests only check that HeuristicPoseFusionFilter reaches that
    code correctly and falls back safely, patching mocap-facing calls as imported into
    src.pose_fusion_heuristic to isolate that from mocap_data's own (separately tested)
    correctness."""

    @staticmethod
    def _imu_streams(target_ts_ns: int, gyro_val, accel_val):
        t = np.array([0, target_ts_ns], dtype=np.int64)
        gyro = np.tile(np.asarray(gyro_val, dtype=np.float64), (2, 1))
        accel = np.tile(np.asarray(accel_val, dtype=np.float64), (2, 1))
        return (t, gyro), (t, accel)

    def _make_filter_with_imu(self, headset_mocap=None, g_world_estimator_abs=None,
                               g_world_estimator=None,
                               gyro_val=(0.1, 0.0, 0.0), accel_val=(0.0, 0.0, 9.81),
                               target_ts_ns=int(0.02 * _NS), p0=None):
        gyro_data, accel_data = self._imu_streams(target_ts_ns, gyro_val, accel_val)
        cfg = {"max_coast_s": 1.0, "max_consecutive_rejects": 5, "sibling_trust_min": 0.3,
               "fusion_heuristic": {}}
        f = HeuristicPoseFusionFilter(
            gyro_data=gyro_data, accel_data=accel_data, lever_arm=np.zeros(3),  # predict() treats
            # lever_arm=None as "not ready yet", a pre-existing gate unrelated to this feature --
            # a real (if zero-offset) lever arm is needed here so predict() actually runs.
            g_world_estimator=(g_world_estimator if g_world_estimator is not None
                                else Mock(g_world=np.array([0.0, 0.0, -9.81]))),
            cfg=cfg, ctrl_name="test",
            headset_mocap=headset_mocap, g_world_estimator_abs=g_world_estimator_abs,
        )
        f.R = np.eye(3)
        f.p = np.zeros(3) if p0 is None else np.array(p0, dtype=np.float64)
        f.v = np.zeros(3)
        f.last_update_ts_ns = 0
        return f

    def test_default_none_is_a_true_no_op(self):
        """Unmodified default (no headset_mocap) -- predict() must take the plain,
        already-existing path with no new branch reached."""
        f = self._make_filter_with_imu(headset_mocap=None)
        self.assertIsNone(f._headset_mocap)
        target_ts = int(0.02 * _NS)
        with patch("src.pose_fusion_heuristic.world_pose") as mock_world_pose:
            result = f.predict(target_ts)
            mock_world_pose.assert_not_called()
        self.assertIsNotNone(result)

    def test_fails_open_to_uncorrected_path_on_any_missing_ingredient(self):
        """A real, partial mocap-coverage gap (world_pose succeeds at ts0 but not ts1) must fall
        back to the plain predict_world_pose path, not raise or return None outright."""
        target_ts = int(0.02 * _NS)
        sentinel = (np.eye(3), np.array([9.0, 9.0, 9.0]))
        f = self._make_filter_with_imu(headset_mocap=Mock(),
                                        g_world_estimator_abs=Mock(g_world=np.array([0.0, 0.0, -9.81])),
                                        target_ts_ns=target_ts)
        with patch("src.pose_fusion_heuristic.predict_world_pose", return_value=sentinel) as mock_pwp, \
             patch("src.pose_fusion_heuristic.world_pose") as mock_world_pose, \
             patch("src.pose_fusion_heuristic.headset_angular_velocity", return_value=np.zeros(3)), \
             patch("src.pose_fusion_heuristic.headset_linear_velocity", return_value=np.zeros(3)):
            mock_world_pose.side_effect = [Mock(R=np.eye(3), t=np.zeros(3)), None]  # ts0 ok, ts1 gap
            result = f.predict(target_ts)
        self.assertEqual(result, sentinel)
        mock_pwp.assert_called_once()

    def test_nonzero_correction_changes_output_in_expected_direction(self):
        """Headset rotating fast, controller rigidly co-rotating (genuinely stationary relative
        to it) -- mirrors tests.test_predict_headset_relative_pose.ProveTheBugProveTheFixTests,
        but exercised through the real predict() wiring instead of calling the wrapper directly.

        g_abs must match MOCAP_ROOM_G_WORLD exactly (not an arbitrary axis) --
        _predict_with_headset_correction now hardcodes that constant rather than reading
        g_world_estimator_abs (see src/pose_fusion_heuristic.py), so the synthetic accel
        fixture has to be built against the SAME value the code under test actually uses,
        or this test would be checking a physics scenario the code no longer implements."""
        omega_h0 = np.array([0.0, 2.0, 0.0])  # parallel to MOCAP_ROOM_G_WORLD's own axis (Y) --
        # preserves the original test's rotation-parallel-to-gravity special case (previously
        # both were Z-axis); an arbitrary non-parallel axis choice here changes which cross
        # terms vanish and desensitizes the corrected-vs-uncorrected gap this test checks.
        p_hc0 = np.array([0.4, 0.0, 0.0])
        g_abs = MOCAP_ROOM_G_WORLD
        T = 0.02
        target_ts = int(T * _NS)
        accel_val = np.cross(omega_h0, np.cross(omega_h0, p_hc0)) - g_abs
        R_wh1 = Rotation.from_rotvec(omega_h0 * T).as_matrix()

        f_uncorrected = self._make_filter_with_imu(
            headset_mocap=None, gyro_val=omega_h0, accel_val=accel_val,
            target_ts_ns=target_ts, p0=p_hc0)
        _, p_uncorrected = f_uncorrected.predict(target_ts)

        f_corrected = self._make_filter_with_imu(
            headset_mocap=Mock(), gyro_val=omega_h0, accel_val=accel_val, target_ts_ns=target_ts, p0=p_hc0)
        with patch("src.pose_fusion_heuristic.world_pose") as mock_world_pose, \
             patch("src.pose_fusion_heuristic.headset_angular_velocity", return_value=omega_h0), \
             patch("src.pose_fusion_heuristic.headset_linear_velocity", return_value=np.zeros(3)):
            mock_world_pose.side_effect = lambda device, ts: (
                Mock(R=np.eye(3), t=np.zeros(3)) if ts == 0 else Mock(R=R_wh1, t=np.zeros(3))
            )
            _, p_corrected = f_corrected.predict(target_ts)

        corrected_err = float(np.linalg.norm(p_corrected - p_hc0))
        uncorrected_err = float(np.linalg.norm(p_uncorrected - p_hc0))
        self.assertLess(corrected_err, 1e-4, "corrected prediction should show no spurious motion")
        self.assertGreater(uncorrected_err, 100 * corrected_err,
                            "uncorrected prediction should be decisively worse -- confirms the "
                            "wiring actually reaches the fix, not just the isolated math function")

    def test_mocap_path_tried_before_rig_frame_g_world_convergence_gate(self):
        """Regression for a bug found 2026-09-09: predict() used to check
        self._g_world_estimator.g_world (the rig-frame LiveGravityEstimator, needing
        g_world_min_samples low-motion samples to converge) BEFORE ever attempting the
        mocap-corrected path -- even though _predict_with_headset_correction doesn't read
        that estimator at all (it uses the fixed MOCAP_ROOM_G_WORLD constant). Net effect:
        an early-run frame with full headset mocap coverage but a not-yet-converged
        rig-frame estimator fail-opened (raw vision, no IMU prediction) when a
        mocap-corrected prediction was actually available. This is the literal FAIL-OPEN
        scenario reported against a real run at an early timestamp.

        predict_world_pose (the rig-frame fallback) must not even be called here -- the
        mocap path must win outright, not just produce the right answer via both paths
        coincidentally agreeing."""
        target_ts = int(0.02 * _NS)
        sentinel = (np.eye(3), np.array([9.0, 9.0, 9.0]))
        f = self._make_filter_with_imu(
            headset_mocap=Mock(),
            g_world_estimator=Mock(g_world=None),  # rig-frame estimator NOT converged yet
            target_ts_ns=target_ts)
        with patch("src.pose_fusion_heuristic.predict_headset_relative_pose",
                   return_value=sentinel) as mock_phrp, \
             patch("src.pose_fusion_heuristic.predict_world_pose") as mock_pwp, \
             patch("src.pose_fusion_heuristic.world_pose") as mock_world_pose, \
             patch("src.pose_fusion_heuristic.headset_angular_velocity", return_value=np.zeros(3)), \
             patch("src.pose_fusion_heuristic.headset_linear_velocity", return_value=np.zeros(3)):
            # Full mocap coverage at both endpoints -- no real coverage gap.
            mock_world_pose.side_effect = [Mock(R=np.eye(3), t=np.zeros(3)),
                                            Mock(R=np.eye(3), t=np.zeros(3))]
            result = f.predict(target_ts)
        self.assertEqual(result, sentinel)
        mock_phrp.assert_called_once()
        mock_pwp.assert_not_called()

    def test_still_fails_open_when_neither_mocap_nor_rig_frame_g_world_available(self):
        """Genuine unavailability (no headset_mocap AND rig-frame estimator not converged)
        must still return None -- the reorder must not accidentally make predict() lenient
        beyond restoring the mocap path's own precondition-only gating."""
        target_ts = int(0.02 * _NS)
        f = self._make_filter_with_imu(headset_mocap=None, g_world_estimator=Mock(g_world=None),
                                        target_ts_ns=target_ts)
        self.assertIsNone(f.predict(target_ts))


class ControllerTrackerHeadsetMocapThreadingTests(unittest.TestCase):
    """Confirms ControllerTracker.__init__ actually threads headset_mocap/g_world_estimator_abs
    down to a constructed HeuristicPoseFusionFilter, and -- just as important -- that the
    conditional _filter_kwargs branch correctly EXCLUDES them when filter_type is "kalman"
    (PoseFusionFilter.__init__ takes no such params; passing them unconditionally would raise
    TypeError). No pool/images/real cameras needed -- cameras/trackers are never touched by
    __init__ beyond being stored."""

    def test_heuristic_filter_receives_headset_mocap_and_abs_estimator(self):
        sentinel_mocap = object()
        sentinel_abs = object()
        tracker = ControllerTracker(
            "test", {}, {}, fusion_cfg={"enabled": True, "filter_type": "heuristic"},
            headset_mocap=sentinel_mocap, g_world_estimator_abs=sentinel_abs,
        )
        self.assertIsInstance(tracker._fusion_filter, HeuristicPoseFusionFilter)
        self.assertIs(tracker._fusion_filter._headset_mocap, sentinel_mocap)
        self.assertIs(tracker._fusion_filter._g_world_estimator_abs, sentinel_abs)

    def test_kalman_filter_construction_does_not_raise_with_headset_mocap_set(self):
        tracker = ControllerTracker(
            "test", {}, {}, fusion_cfg={"enabled": True, "filter_type": "kalman"},
            headset_mocap=object(), g_world_estimator_abs=object(),
        )
        self.assertIsInstance(tracker._fusion_filter, PoseFusionFilter)


def _imu_arrays(gyro_samples, accel_samples):
    """(t_ns, values) tuples for gyro/accel, matching slice_imu_to_window's
    expected shape -- gyro_samples/accel_samples: list of (t_ns, [x,y,z])."""
    t_g = np.array([t for t, _ in gyro_samples], dtype=np.int64)
    g = np.array([v for _, v in gyro_samples], dtype=np.float64)
    t_a = np.array([t for t, _ in accel_samples], dtype=np.int64)
    a = np.array([v for _, v in accel_samples], dtype=np.float64)
    return (t_g, g), (t_a, a)


class PeakGyroAccelTests(unittest.TestCase):
    """Regression for HeuristicPoseFusionFilter._peak_gyro_accel (2026-09-13,
    added so _implausible_jump_thresholds can tell "real violent motion"
    apart from "a noisy/short-baseline velocity estimate" -- see that
    method's own docstring for the real case (mocap-confirmed-good vision
    rejected during a real ~14g/~2900deg/s swing) this closes."""

    def test_returns_zero_without_gyro_or_accel_data(self):
        f = _make_filter()  # gyro_data=accel_data=None
        f.last_update_ts_ns = 0
        self.assertEqual(f._peak_gyro_accel(1_000_000), (0.0, 0.0))

    def test_returns_zero_without_prior_state(self):
        gyro_data, accel_data = _imu_arrays(
            [(0, [0, 0, 100.0])], [(0, [0.0, 0.0, 9.81])])
        f = HeuristicPoseFusionFilter(gyro_data=gyro_data, accel_data=accel_data,
                                       lever_arm=None, g_world_estimator=None, cfg={"fusion_heuristic": {}})
        self.assertIsNone(f.last_update_ts_ns)
        self.assertEqual(f._peak_gyro_accel(1_000_000), (0.0, 0.0))

    def test_peak_magnitude_over_the_window_gravity_subtracted(self):
        # gyro peak: sqrt(0^2+0^2+(pi rad/s)^2) = pi rad/s = 180 deg/s, at t=5ms
        # (the largest sample inside the window; a later, larger sample at
        # t=20ms sits OUTSIDE [0, 10ms] and must not be picked up).
        gyro_data, accel_data = _imu_arrays(
            [(0, [0.0, 0.0, 0.0]), (5_000_000, [0.0, 0.0, np.pi]), (20_000_000, [0.0, 0.0, 100.0])],
            [(0, [0.0, 0.0, 9.81]), (5_000_000, [30.0, 0.0, 9.81]), (20_000_000, [500.0, 0.0, 9.81])],
        )
        f = HeuristicPoseFusionFilter(gyro_data=gyro_data, accel_data=accel_data,
                                       lever_arm=None, g_world_estimator=None, cfg={"fusion_heuristic": {}})
        f.last_update_ts_ns = 0
        peak_gyro_dps, peak_accel_mps2 = f._peak_gyro_accel(10_000_000)
        self.assertAlmostEqual(peak_gyro_dps, 180.0, places=3)
        # |[30,0,9.81]| = sqrt(900+96.24)=31.55, minus the 9.81 fallback gravity magnitude
        self.assertAlmostEqual(peak_accel_mps2, np.hypot(30.0, 9.81) - 9.81, places=3)

    def test_dynamic_accel_never_negative_even_below_gravity_baseline(self):
        gyro_data, accel_data = _imu_arrays(
            [(0, [0.0, 0.0, 0.0])], [(0, [0.0, 0.0, 1.0])],  # |a|=1.0 << 9.81
        )
        f = HeuristicPoseFusionFilter(gyro_data=gyro_data, accel_data=accel_data,
                                       lever_arm=None, g_world_estimator=None, cfg={"fusion_heuristic": {}})
        f.last_update_ts_ns = 0
        _, peak_accel_mps2 = f._peak_gyro_accel(1_000_000)
        self.assertEqual(peak_accel_mps2, 0.0)  # clipped, not negative

    def test_uses_g_world_estimator_magnitude_when_available(self):
        gyro_data, accel_data = _imu_arrays(
            [(0, [0.0, 0.0, 0.0])], [(0, [0.0, 0.0, 20.0])],
        )
        g_est = Mock()
        g_est.g_world = np.array([0.0, 0.0, 15.0])  # magnitude 15, not the 9.81 fallback
        f = HeuristicPoseFusionFilter(gyro_data=gyro_data, accel_data=accel_data,
                                       lever_arm=None, g_world_estimator=g_est, cfg={"fusion_heuristic": {}})
        f.last_update_ts_ns = 0
        _, peak_accel_mps2 = f._peak_gyro_accel(1_000_000)
        self.assertAlmostEqual(peak_accel_mps2, 20.0 - 15.0, places=6)


class ImplausibleJumpThresholdsAccelAwareTests(unittest.TestCase):
    """_implausible_jump_thresholds' new accel/gyro-aware widening term
    (additive on top of the existing velocity term)."""

    _RATE_CFG = {
        "implausible_jump_pos_thresh_base_mm": 100.0,
        "implausible_jump_pos_thresh_per_speed_mm_s": 0.0,
        "implausible_jump_rot_thresh_base_deg": 10.0,
        "implausible_jump_rot_thresh_per_speed_deg_s": 0.0,
        "implausible_jump_accel_calm_floor_mps2": 40.0,
        "implausible_jump_gyro_calm_floor_dps": 900.0,
        "implausible_jump_pos_thresh_per_accel_mm_per_mps2": 2.0,
        "implausible_jump_rot_thresh_per_gyro_deg_per_dps": 0.01,
    }

    def _filter_with_imu(self, peak_accel_dynamic_mps2, peak_gyro_dps):
        # One sample at t=0 and one at t=window end -- _peak_gyro_accel takes
        # the max over the window, so a single extreme sample is enough.
        accel_z = peak_accel_dynamic_mps2 + 9.81  # + gravity fallback baseline
        _, accel_data = _imu_arrays([(0, [0.0, 0.0, 0.0])], [(0, [0.0, 0.0, accel_z])])
        gyro_data = (np.array([0, 10_000_000]),
                     np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.radians(peak_gyro_dps)]]))
        # matching.max_plausible_hand_*: zeroed here to isolate this test class's
        # own accel/gyro-peak mechanism from the separate stale-time widening
        # term (_implausible_jump_thresholds' own docstring) -- both terms use
        # the same [last_update_ts_ns, frame_ts_ns] window, so a nonzero stale
        # rate would otherwise leak a second widening source into these
        # assertions; that term has its own dedicated tests below.
        f = HeuristicPoseFusionFilter(gyro_data=gyro_data, accel_data=accel_data,
                                       lever_arm=None, g_world_estimator=None,
                                       cfg={"fusion_heuristic": self._RATE_CFG,
                                            "matching": {"max_plausible_hand_speed_m_s": 0.0,
                                                         "max_plausible_hand_ang_speed_deg_s": 0.0}})
        f.last_update_ts_ns = 0
        return f

    def test_calm_motion_no_extra_widening(self):
        f = self._filter_with_imu(peak_accel_dynamic_mps2=10.0, peak_gyro_dps=500.0)  # both under calm floor
        pos_m, rot_deg = f._implausible_jump_thresholds(10_000_000)
        self.assertAlmostEqual(pos_m, 0.100, places=6)   # base only
        self.assertAlmostEqual(rot_deg, 10.0, places=6)  # base only

    def test_violent_motion_widens_both_ceilings(self):
        # 60 m/s^2 over the 40 floor -> +20 excess; 2000 deg/s over the 900 floor -> +1100 excess
        f = self._filter_with_imu(peak_accel_dynamic_mps2=60.0, peak_gyro_dps=2000.0)
        pos_m, rot_deg = f._implausible_jump_thresholds(10_000_000)
        self.assertAlmostEqual(pos_m, (100.0 + 2.0 * 20.0) / 1000.0, places=6)
        self.assertAlmostEqual(rot_deg, 10.0 + 0.01 * 1100.0, places=6)

    def test_zero_rate_constants_are_a_pure_no_op(self):
        """Rate constants default to 0.0 (see pose_fusion_heuristic.py's own
        _hc_get fallbacks) until the empirical fit lands -- must not change
        behaviour at all until then, regardless of how violent the window
        was."""
        cfg = {**self._RATE_CFG, "implausible_jump_pos_thresh_per_accel_mm_per_mps2": 0.0,
               "implausible_jump_rot_thresh_per_gyro_deg_per_dps": 0.0}
        gyro_data, accel_data = _imu_arrays(
            [(0, [0.0, 0.0, 0.0])], [(0, [0.0, 0.0, 500.0])],
        )
        f = HeuristicPoseFusionFilter(
            gyro_data=gyro_data, accel_data=accel_data, lever_arm=None, g_world_estimator=None,
            cfg={"fusion_heuristic": cfg,
                 "matching": {"max_plausible_hand_speed_m_s": 0.0, "max_plausible_hand_ang_speed_deg_s": 0.0}})
        f.last_update_ts_ns = 0
        pos_m, rot_deg = f._implausible_jump_thresholds(10_000_000)
        self.assertAlmostEqual(pos_m, 0.100, places=6)
        self.assertAlmostEqual(rot_deg, 10.0, places=6)


class EffectiveCoastBudgetTests(unittest.TestCase):
    """Regression for _effective_coast_budget_s (2026-09-14, user-directed:
    "prove it with stats"). Found investigating a real case (right_controller,
    walk_hard recording): _report_if_still_usable's flat
    cold_pending_report_max_gap_s=0.25s let a freshly recomputed IMU-only
    coast display over a meter from vision, well inside that flat budget --
    real violent motion (gyro/accel) had made the coast unreliable despite
    the elapsed time alone looking "still short enough to trust." This
    shrinks the budget by how violent real measured gyro/accel was over the
    same window _peak_gyro_accel already measures, down to a floor
    (coast_trust_min_budget_s, default matches the user's own "~11ms"
    suggestion) -- never widened, only ever shrunk, unlike every other
    accel/gyro term in this file."""

    def _filter_with_imu(self, peak_accel_dynamic_mps2, peak_gyro_dps, **cfg_overrides):
        accel_z = peak_accel_dynamic_mps2 + 9.81
        _, accel_data = _imu_arrays([(0, [0.0, 0.0, 0.0])], [(0, [0.0, 0.0, accel_z])])
        gyro_data = (np.array([0, 10_000_000]),
                     np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.radians(peak_gyro_dps)]]))
        cfg = {
            "coast_trust_accel_calm_floor_mps2": 40.0,
            "coast_trust_gyro_calm_floor_dps": 900.0,
            "coast_trust_shrink_s_per_mps2": 0.0,
            "coast_trust_shrink_s_per_dps": 0.0,
            "coast_trust_min_budget_s": 0.011,
            **cfg_overrides,
        }
        f = HeuristicPoseFusionFilter(gyro_data=gyro_data, accel_data=accel_data,
                                       lever_arm=None, g_world_estimator=None,
                                       cfg={"fusion_heuristic": cfg})
        f.last_update_ts_ns = 0
        return f

    def test_calm_window_returns_base_budget_unchanged(self):
        f = self._filter_with_imu(peak_accel_dynamic_mps2=10.0, peak_gyro_dps=500.0,
                                   coast_trust_shrink_s_per_mps2=5.0, coast_trust_shrink_s_per_dps=0.001)
        budget = f._effective_coast_budget_s(0.25, 10_000_000)
        self.assertAlmostEqual(budget, 0.25, places=6)

    def test_violent_window_shrinks_the_budget(self):
        # gyro excess = 2000-900 = 1100 -> shrink = 0.0001*1100 = 0.11s
        f = self._filter_with_imu(peak_accel_dynamic_mps2=10.0, peak_gyro_dps=2000.0,
                                   coast_trust_shrink_s_per_dps=0.0001)
        budget = f._effective_coast_budget_s(0.25, 10_000_000)
        self.assertAlmostEqual(budget, 0.25 - 0.11, places=6)

    def test_shrink_clamps_at_the_configured_floor(self):
        f = self._filter_with_imu(peak_accel_dynamic_mps2=10.0, peak_gyro_dps=3000.0,
                                   coast_trust_shrink_s_per_dps=10.0,  # absurdly large
                                   coast_trust_min_budget_s=0.011)
        budget = f._effective_coast_budget_s(0.25, 10_000_000)
        self.assertAlmostEqual(budget, 0.011, places=6)

    def test_zero_rates_are_a_pure_no_op(self):
        f = self._filter_with_imu(peak_accel_dynamic_mps2=500.0, peak_gyro_dps=5000.0)
        budget = f._effective_coast_budget_s(0.25, 10_000_000)
        self.assertAlmostEqual(budget, 0.25, places=6)

    def test_used_by_report_if_still_usable_to_clear_reported_pose(self):
        """Integration check: a violent window whose real elapsed dt_s is
        inside the FLAT budget but outside the SHRUNK one must still clear
        reported_R/reported_p via _report_if_still_usable's own (2b)
        clearing branch -- ties this method into the real bug this was
        added for, not just testing it in isolation."""
        f = self._filter_with_imu(peak_accel_dynamic_mps2=10.0, peak_gyro_dps=3000.0,
                                   coast_trust_shrink_s_per_dps=0.001,  # (3000-900)*0.001=2.1s -- wipes 0.25s
                                   coast_trust_min_budget_s=0.005,  # below the real dt_s=0.01s so the
                                   # shrunk budget (not just the floor) is what's actually exercised here
                                   cold_pending_report_max_gap_s=0.25)
        f.R, f.p, f.v = np.eye(3), np.array([9.0, 9.0, 9.0]), np.zeros(3)
        f.velocity_established = True
        f._lever_arm = np.zeros(3)
        f._g_world_estimator = Mock(g_world=np.array([0.0, 0.0, -9.81]))
        # predict() needs gyro/accel coverage spanning all the way to ts1=
        # 10_000_000 (integrate_gyro_segment/integrate_accel_to_position both
        # refuse to extrapolate past their own last sample) -- _filter_with_imu's
        # accel_data only has the one sample at t=0, so extend it to also cover
        # ts1 (gyro_data already does, built with exactly that span).
        f._accel_data = (np.array([0, 10_000_000]),
                          np.array([[0.0, 0.0, 9.81], [0.0, 0.0, 9.81]]))
        f._report(0, f.R, f.p)  # seed a real pre-loss reported pose, same as a genuine prior accept would
        self.assertIsNotNone(f.reported_p, "sanity: a real pre-loss reported pose exists")
        # Force try_update's cold-state routing (imu_frame_scale <= 0) so this
        # goes through _try_cold_reacquire/_report_if_still_usable instead of
        # the normal per-frame blend -- same condition a real several-
        #-frames-lost stretch would have produced.
        f.frames_since_update = 10

        weak = _solution(np.eye(3), np.array([1.0, 0.0, 0.0]), n_inliers=4)
        ok = f.try_update(weak, 10_000_000)  # dt_s=0.01s -- well inside the FLAT 0.25s budget
        self.assertFalse(ok)
        self.assertIsNone(f.reported_p, "the accel/gyro-shrunk budget must have cleared it despite the short elapsed time")


class StaleTimeWideningTests(unittest.TestCase):
    """Regression for _implausible_jump_thresholds' stale-time widening term
    (2026-09-13, second real case) -- reuses matching.max_plausible_hand_
    speed_m_s/_ang_speed_deg_s (this session's own already-validated,
    consolidated constants) to widen the ceiling by elapsed time since
    last_update_ts_ns, on top of the existing |self.v|-scaled term. Found
    investigating a real reject cascade at a genuine motion reversal (swing
    turning point): self.v/predict()'s dead-reckoning uses a ONE-STEP-STALE
    velocity as its integration's initial condition, so it overshoots past
    the reversal, and each subsequent REJECTED frame's growing dt (since
    last_update_ts_ns doesn't advance on a reject) compounds the divergence
    -- confirmed on a real 4-frame cascade where pos_innov_m grew 123/272/
    373/576mm tracking elapsed time since the last real accept almost
    exactly. Uses realistic millisecond-scale gaps (not this file's usual
    whole-second _NS convention -- see _make_filter's own comment on why
    that convention needs matching.max_plausible_hand_* zeroed elsewhere)."""

    def _filter_with_stale_rates(self, max_speed_m_s=7.5, max_ang_speed_deg_s=2200.0):
        f = HeuristicPoseFusionFilter(
            gyro_data=None, accel_data=None, lever_arm=None, g_world_estimator=None,
            cfg={"fusion_heuristic": {
                     "implausible_jump_pos_thresh_base_mm": 0.0,
                     "implausible_jump_pos_thresh_per_speed_mm_s": 0.0,
                     "implausible_jump_rot_thresh_base_deg": 0.0,
                     "implausible_jump_rot_thresh_per_speed_deg_s": 0.0,
                 },
                 "matching": {"max_plausible_hand_speed_m_s": max_speed_m_s,
                              "max_plausible_hand_ang_speed_deg_s": max_ang_speed_deg_s}})
        f.last_update_ts_ns = 0
        return f

    def test_zero_stale_s_right_after_a_real_accept(self):
        """frame_ts_ns == last_update_ts_ns (no elapsed time yet) -- the new
        term must contribute exactly 0, not just "small"."""
        f = self._filter_with_stale_rates()
        pos_m, rot_deg = f._implausible_jump_thresholds(0)
        self.assertEqual(pos_m, 0.0)
        self.assertEqual(rot_deg, 0.0)

    def test_widens_proportionally_to_elapsed_time(self):
        f = self._filter_with_stale_rates(max_speed_m_s=7.5, max_ang_speed_deg_s=2200.0)
        # 22.15ms elapsed (a real inter-frame gap, not a whole second)
        pos_m, rot_deg = f._implausible_jump_thresholds(22_150_000)
        self.assertAlmostEqual(pos_m, 7.5 * 0.02215, places=6)
        self.assertAlmostEqual(rot_deg, 2200.0 * 0.02215, places=6)

    def test_compounds_across_a_consecutive_reject_cascade(self):
        """last_update_ts_ns does NOT advance on a reject (try_update never
        touches it outside the accepted branches) -- so calling this again
        at a LATER frame_ts_ns without an intervening accept must widen
        FURTHER, not reset. Mirrors the real cascade this term was added
        for: elapsed time since the last real accept keeps growing across
        consecutive rejects."""
        f = self._filter_with_stale_rates()
        pos_1, _ = f._implausible_jump_thresholds(11_000_000)
        pos_2, _ = f._implausible_jump_thresholds(33_000_000)  # later frame, same last_update_ts_ns
        pos_3, _ = f._implausible_jump_thresholds(66_000_000)
        self.assertLess(pos_1, pos_2)
        self.assertLess(pos_2, pos_3)
        self.assertAlmostEqual(pos_3, 7.5 * 0.066, places=6)

    def test_negative_elapsed_time_clamped_not_negative_widening(self):
        """Defensive: frame_ts_ns before last_update_ts_ns (shouldn't happen
        in real use, but must not silently NARROW the ceiling if it did)."""
        f = self._filter_with_stale_rates()
        f.last_update_ts_ns = 50_000_000
        pos_m, rot_deg = f._implausible_jump_thresholds(10_000_000)
        self.assertEqual(pos_m, 0.0)
        self.assertEqual(rot_deg, 0.0)


class ResetClearsNewStateTests(unittest.TestCase):
    def test_reset_clears_agreement_history(self):
        f = _make_filter()
        f.try_update(_solution(np.eye(3), np.array([0.0, 0.0, 0.0])), 0)
        f._pos_innov_hist.append(0.01)
        f._rot_innov_hist.append(1.0)

        f.reset()

        self.assertEqual(len(f._pos_innov_hist), 0)
        self.assertEqual(len(f._rot_innov_hist), 0)
        self.assertIsNone(f.R)


if __name__ == "__main__":
    unittest.main()
