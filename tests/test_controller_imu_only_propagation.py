"""Regression coverage for ControllerTracker._mark_all_lost's IMU-only
search-anchor propagation budget (src/controller.py), converted 2026-09-09
from a frame-counted budget (matching.imu_only_propagation_max_frames) to a
real-elapsed-time one (matching.imu_only_propagation_max_s).

Why this matters: this recording's own camera frame rate oscillates
(~11ms/~22ms gaps), so "N consecutive lost frames" and "N frames' worth of
seconds" are NOT the same threshold. CameraTracker.finalize_search's own
cold-start-widening tolerance (the OTHER half of the re-acquisition guard)
already used real elapsed time (last_good_pose_ts_ns) -- a frame-counted
budget here and a time-based tolerance there could disagree about "how
stale is this" for the exact same loss streak, and their values weren't
even in comparable units in the resulting log line ("consecutive_failures=9,
budget=4" told you nothing about real elapsed seconds). Both now read off
the same clock: self._fusion_filter.last_update_ts_ns, the same event
last_good_pose_ts_ns is set from.

These tests exercise _mark_all_lost directly on a real ControllerTracker
with fusion enabled but no real gyro/accel data (predict() is stubbed
directly on the constructed filter instance, same approach
tests/test_pose_fusion_heuristic.py uses) and an empty `trackers` dict (safe
-- both _mark_all_lost and _propagate_pose_history's per-camera loops are
no-ops with nothing in it), so no cameras/pool/images are needed.

Uses stdlib unittest (pytest is not a declared dependency of this project).
Run with:  python3 -m unittest tests.test_controller_imu_only_propagation
"""
import unittest
from unittest.mock import Mock

import numpy as np

from src.controller import ControllerTracker
from src.transformations import Transform

_NS = 1_000_000_000


def _make_tracker(imu_only_propagation_max_s=0.066):
    matching_cfg = {"imu_only_propagation_max_s": imu_only_propagation_max_s,
                     "tracking_lost_grace_frames": 1}
    tracker = ControllerTracker(
        "test", {}, {}, matching_cfg=matching_cfg,
        fusion_cfg={"enabled": True, "filter_type": "heuristic"},
    )
    # This file tests the TIME BUDGET gate specifically (imu_only_propagation_max_s),
    # not the separate velocity_established gate added 2026-09-11 (see
    # HeuristicPoseFusionFilter.velocity_established's own comment) -- a fresh
    # filter starts with velocity_established=False, which would make every
    # predict() call below unreachable regardless of the time budget, for a
    # reason unrelated to what these tests are actually exercising.
    tracker._fusion_filter.velocity_established = True
    return tracker


def _stub_predict(calls: list, R=None, p=None):
    """Records every call (so tests can assert predict() was/wasn't reached
    -- distinct from merely checking the OUTCOME, since the gate must
    short-circuit BEFORE calling predict() once past budget, not call it and
    discard the result) and returns a fixed (R, p)."""
    R = np.eye(3) if R is None else R
    p = np.array([1.0, 2.0, 3.0]) if p is None else p

    def _predict(target_ts_ns):
        calls.append(target_ts_ns)
        return R, p
    return _predict


class TimeBasedBudgetTests(unittest.TestCase):
    def test_propagates_within_budget(self):
        """A loss well inside imu_only_propagation_max_s must still dead-reckon
        a search-anchor pose from the fusion filter."""
        tracker = _make_tracker(imu_only_propagation_max_s=0.066)
        calls = []
        tracker._fusion_filter.last_update_ts_ns = 0
        tracker._fusion_filter.predict = _stub_predict(calls)

        tracker._mark_all_lost(frame_ts_ns=int(0.03 * _NS))  # 30ms < 66ms budget

        self.assertEqual(len(calls), 1, "predict() should have been reached")
        self.assertIsNotNone(tracker._last_imu_only_pose)
        np.testing.assert_allclose(tracker._last_imu_only_pose.t, [1.0, 2.0, 3.0])

    def test_stops_past_budget(self):
        """Once real elapsed time since the last real accept exceeds the
        budget, predict() must not even be called -- not just its result
        discarded (mirrors CameraTracker.finalize_search's cold-start guard
        treating 'past budget' as 'no predicted pose available', not as a
        computed-and-rejected comparison)."""
        tracker = _make_tracker(imu_only_propagation_max_s=0.066)
        calls = []
        tracker._fusion_filter.last_update_ts_ns = 0
        tracker._fusion_filter.predict = _stub_predict(calls)

        tracker._mark_all_lost(frame_ts_ns=int(0.10 * _NS))  # 100ms > 66ms budget

        self.assertEqual(len(calls), 0, "predict() must not be called past the time budget")
        self.assertIsNone(tracker._last_imu_only_pose)

    def test_budget_is_real_time_not_call_count(self):
        """The core regression: many _mark_all_lost calls at a fast, sub-budget
        cadence must all still propagate (old frame-counted code would have
        stopped after exactly 4 calls, regardless of how little real time had
        elapsed) -- this is the frame-rate-independence the conversion to
        imu_only_propagation_max_s exists to guarantee."""
        tracker = _make_tracker(imu_only_propagation_max_s=0.066)
        calls = []
        tracker._fusion_filter.last_update_ts_ns = 0
        tracker._fusion_filter.predict = _stub_predict(calls)

        # 10 consecutive lost frames at a fast ~5ms cadence = 50ms total,
        # still under the 66ms budget -- every single one should propagate,
        # not just the first 4 (the old frame-counted behavior).
        dt_ns = int(0.005 * _NS)
        for i in range(1, 11):
            tracker._last_imu_propagate_ts_ns = None  # new distinct frame each call
            tracker._mark_all_lost(frame_ts_ns=i * dt_ns)

        self.assertEqual(len(calls), 10, "all 10 fast-cadence frames should be within the time budget")
        self.assertIsNotNone(tracker._last_imu_only_pose)

    def test_few_calls_at_slow_cadence_still_exhausts_budget(self):
        """Converse of the above: even just the 2nd _mark_all_lost call (which
        an old frame-counted budget of 4 would still happily grant) must be
        denied once real elapsed time alone has already exceeded the budget
        -- proves the gate tracks time, not "how many calls so far"."""
        tracker = _make_tracker(imu_only_propagation_max_s=0.066)
        calls = []
        tracker._fusion_filter.last_update_ts_ns = 0
        tracker._fusion_filter.predict = _stub_predict(calls)

        tracker._mark_all_lost(frame_ts_ns=int(0.02 * _NS))   # call #1, 20ms -- within budget
        self.assertEqual(len(calls), 1)

        tracker._mark_all_lost(frame_ts_ns=int(0.08 * _NS))   # call #2, 80ms -- past budget
        self.assertEqual(len(calls), 1, "the 2nd call must NOT propagate -- real elapsed time, not call count, governs")
        self.assertIsNone(tracker._last_imu_only_pose)

    def test_no_reset_bookkeeping_needed_after_a_real_accept(self):
        """A real accept advances self._fusion_filter.last_update_ts_ns (this is
        the fusion filter's own existing contract, unaffected by this change);
        the very next _mark_all_lost call must get a fresh budget measured
        from THAT new timestamp automatically, with no separate counter to
        reset anywhere in ControllerTracker itself."""
        tracker = _make_tracker(imu_only_propagation_max_s=0.066)
        calls = []
        tracker._fusion_filter.last_update_ts_ns = 0
        tracker._fusion_filter.predict = _stub_predict(calls)

        tracker._mark_all_lost(frame_ts_ns=int(0.10 * _NS))  # past budget -- exhausted
        self.assertEqual(len(calls), 0)

        # Simulate a real vision accept landing at ts=0.10s (advances the
        # fusion filter's own clock -- exactly what try_update does).
        tracker._fusion_filter.last_update_ts_ns = int(0.10 * _NS)

        tracker._mark_all_lost(frame_ts_ns=int(0.13 * _NS))  # only 30ms past the NEW anchor
        self.assertEqual(len(calls), 1, "budget should be fresh again, measured from the new accept")
        self.assertIsNotNone(tracker._last_imu_only_pose)

    def test_no_fusion_filter_last_update_yet_never_propagates(self):
        """last_update_ts_ns is None (fusion filter never accepted anything at
        all, e.g. a brand-new controller with no bootstrap yet) -- must not
        propagate (no anchor to measure elapsed time from), and must not
        crash on the None arithmetic."""
        tracker = _make_tracker(imu_only_propagation_max_s=0.066)
        calls = []
        tracker._fusion_filter.predict = _stub_predict(calls)
        self.assertIsNone(tracker._fusion_filter.last_update_ts_ns)

        tracker._mark_all_lost(frame_ts_ns=int(0.01 * _NS))

        self.assertEqual(len(calls), 0)
        self.assertIsNone(tracker._last_imu_only_pose)


class FramesSinceUpdateDoesNotFreezeTests(unittest.TestCase):
    """Regression for a false IMPLAUSIBLE-vision-jump rejection found 2026-09-09
    (real run, frame 178, left_controller): HeuristicPoseFusionFilter.
    frames_since_update (which imu_frame_scale, and the hard implausibility
    gate it guards, decay over imu_decay_frames of) used to be bumped ONLY
    inside predict() -- and _mark_all_lost only calls predict() while still
    inside the IMU-only coast's own time budget (imu_only_propagation_max_s,
    a deliberate cost cap). Past that budget, _mark_all_lost stopped calling
    predict() at all, so frames_since_update silently FROZE for the rest of
    an arbitrarily long real loss -- keeping imu_frame_scale > 0 and the hard
    gate armed at full (undecayed) strength against a p_pred that had
    actually gone stale over hundreds of ms, long after imu_decay_frames'
    intended horizon. The real case: a ~166ms + ~222ms real gap (well past
    the ~66ms budget) left frames_since_update at only ~2-3 by the time a
    clean 6-inlier/0.07px vision reacquisition candidate arrived, wrongly
    rejecting it as an "identity swap or degenerate low-point fit."

    Fix: ControllerTracker._mark_all_lost now calls
    HeuristicPoseFusionFilter.note_real_frame() unconditionally on every real
    lost frame (cheap -- a timestamp-deduped counter bump, not the actual
    dead-reckoning math), separately from the still-budget-gated predict()
    call. These tests confirm frames_since_update keeps growing across many
    real lost frames spanning far more real time than the IMU-only budget,
    not just the handful that fell within it."""

    def test_frames_since_update_keeps_growing_past_the_imu_only_budget(self):
        tracker = _make_tracker(imu_only_propagation_max_s=0.066)
        f = tracker._fusion_filter
        # Minimal bootstrap -- enough for frames_since_update to be meaningful,
        # no real IMU data needed since this test only checks the counter.
        f.R, f.p, f.v = np.eye(3), np.zeros(3), np.zeros(3)
        f.last_update_ts_ns = 0
        f.frames_since_update = 0

        # 20 consecutive real lost frames at a ~50ms cadence (1s total) --
        # only the first one (50ms < 66ms budget) would have called predict()
        # under the old code; every one of the other 19 is well past budget.
        dt_ns = int(0.05 * _NS)
        for i in range(1, 21):
            tracker._last_imu_propagate_ts_ns = None  # distinct real frame each call
            tracker._mark_all_lost(frame_ts_ns=i * dt_ns)

        self.assertEqual(f.frames_since_update, 20,
                          "frames_since_update must count every real lost frame, "
                          "not freeze once the IMU-only coast budget is exhausted")

    def test_imu_frame_scale_actually_decays_to_zero_after_a_long_real_loss(self):
        """End-to-end version of the same regression, through the real
        imu_frame_scale formula (imu_decay_frames=4 default): after enough
        real lost frames to exceed that horizon, a fresh vision candidate far
        from a (now provably very stale) p_pred must NOT hit the hard
        implausibility gate -- imu_frame_scale must have reached 0, routing
        through cold-reacquisition instead, exactly as intended for a long
        loss. Regression would show this candidate wrongly REJECTED instead
        (outcome=implausible_reject) if frames_since_update had frozen."""
        tracker = _make_tracker(imu_only_propagation_max_s=0.066)
        f = tracker._fusion_filter
        f.R, f.p, f.v = np.eye(3), np.array([0.0, 0.0, 0.0]), np.zeros(3)
        f.last_update_ts_ns = 0
        f.frames_since_update = 0
        # Real (if trivial) IMU streams + a converged g_world -- otherwise
        # predict() always fail-opens (self._gyro_data is None short-circuit)
        # before ever reaching imu_frame_scale/the implausibility gate this
        # test is actually exercising, regardless of frames_since_update.
        n_lost_frames = 10  # >> imu_decay_frames=4
        dt_ns = int(0.05 * _NS)
        t_end = (n_lost_frames + 1) * dt_ns
        t_imu = np.array([0, t_end], dtype=np.int64)
        f._gyro_data = (t_imu, np.zeros((2, 3)))
        f._accel_data = (t_imu, np.tile([0.0, 0.0, -9.81], (2, 1)))
        f._lever_arm = np.zeros(3)
        f._g_world_estimator = Mock(g_world=np.array([0.0, 0.0, -9.81]))

        for i in range(1, n_lost_frames + 1):
            tracker._last_imu_propagate_ts_ns = None
            tracker._mark_all_lost(frame_ts_ns=i * dt_ns)

        target_ts = (n_lost_frames + 1) * dt_ns
        # A candidate implausibly far from the stale (0,0,0) state under the
        # OLD, fully-armed gate (implausible_jump_pos_m default 0.3m) -- must
        # now be accepted (routed to cold-reacquisition) since imu_frame_scale
        # should already be 0 after 10 real lost frames.
        far_candidate = {
            "T_world_ctrl": Transform(np.eye(3), np.array([5.0, 0.0, 0.0])),
            "confidence": 1.0,
            "assignment": [(i, i) for i in range(20)],
            "aux_assignments": {},
            "error": 0.1,
        }
        ok = f.try_update(far_candidate, target_ts)
        self.assertTrue(ok, "a far candidate after a long real loss must be accepted via "
                             "cold-reacquisition, not hard-rejected as implausible")
        self.assertEqual(f._last.get("outcome"), "cold_reacquired")


class AccelGyroAwareBudgetShrinkTests(unittest.TestCase):
    """Regression for the accel/gyro-aware shrink of imu_only_propagation_max_s
    (2026-09-14, user-directed: "prove it with stats"). Found investigating a
    real case (right_controller, walk_hard recording): a flat elapsed-time
    budget assumes "short time since the last real accept -> IMU coast still
    credible," which real violent motion (measured: ~144 m/s^2 accel, ~2985
    deg/s gyro during this recording's own violent segments) can violate well
    inside that flat budget. Reuses src.imu_data.peak_gyro_accel_over_window
    (the same signal HeuristicPoseFusionFilter._effective_coast_budget_s uses
    for the sibling cold_pending_report_max_gap_s budget) rather than a
    second, drifting copy of this exact concept."""

    def _make_tracker_with_imu(self, gyro_samples, accel_samples, **cfg_overrides):
        matching_cfg = {
            "imu_only_propagation_max_s": 0.066,
            "tracking_lost_grace_frames": 1,
            "coast_trust_accel_calm_floor_mps2": 40.0,
            "coast_trust_gyro_calm_floor_dps": 900.0,
            "coast_trust_shrink_s_per_mps2": 0.0,
            "coast_trust_shrink_s_per_dps": 0.0,
            "coast_trust_min_budget_s": 0.011,
            **cfg_overrides,
        }
        tracker = ControllerTracker(
            "test", {}, {}, matching_cfg=matching_cfg,
            fusion_cfg={"enabled": True, "filter_type": "heuristic"},
        )
        tracker._fusion_filter.velocity_established = True
        tracker._fusion_filter.last_update_ts_ns = 0
        t_g = np.array([t for t, _ in gyro_samples], dtype=np.int64)
        g = np.array([v for _, v in gyro_samples], dtype=np.float64)
        t_a = np.array([t for t, _ in accel_samples], dtype=np.int64)
        a = np.array([v for _, v in accel_samples], dtype=np.float64)
        tracker._gyro_data = (t_g, g)
        tracker._accel_data = (t_a, a)
        tracker._g_world_estimator = Mock(g_world=np.array([0.0, 0.0, 9.81]))
        return tracker

    def test_calm_window_keeps_full_budget(self):
        """Low gyro/accel (well under the calm floors) -- budget stays at
        the flat default, unchanged from today's behavior."""
        tracker = self._make_tracker_with_imu(
            [(0, [0.0, 0.0, 0.0]), (int(0.05 * _NS), [0.0, 0.0, 0.0])],
            [(0, [0.0, 0.0, 9.81]), (int(0.05 * _NS), [0.0, 0.0, 9.81])],
            coast_trust_shrink_s_per_mps2=2.0, coast_trust_shrink_s_per_dps=0.001,
        )
        calls = []
        tracker._fusion_filter.predict = _stub_predict(calls)

        tracker._mark_all_lost(frame_ts_ns=int(0.05 * _NS))  # 50ms < 66ms budget, calm window

        self.assertEqual(len(calls), 1, "a calm window's real elapsed time is still under the un-shrunk budget")

    def test_violent_window_shrinks_budget_below_a_gap_the_flat_budget_would_allow(self):
        """A real gap (50ms) that the FLAT 66ms budget would happily grant --
        but real gyro during it (3000 deg/s, far past the 900 calm floor)
        shrinks the effective budget below 50ms, so predict() must not be
        reached at all (not just have its result discarded)."""
        tracker = self._make_tracker_with_imu(
            [(0, [0.0, 0.0, 0.0]), (int(0.05 * _NS), [0.0, 0.0, np.radians(3000.0)])],
            [(0, [0.0, 0.0, 9.81]), (int(0.05 * _NS), [0.0, 0.0, 9.81])],
            coast_trust_shrink_s_per_dps=0.0001,  # (3000-900)*0.0001 = 0.21s shrink -- wipes the 66ms budget
        )
        calls = []
        tracker._fusion_filter.predict = _stub_predict(calls)

        tracker._mark_all_lost(frame_ts_ns=int(0.05 * _NS))

        self.assertEqual(len(calls), 0, "predict() must not be called once real violent motion shrinks the budget below the elapsed gap")
        self.assertIsNone(tracker._last_imu_only_pose)

    def test_shrink_never_goes_below_the_configured_floor(self):
        """An extreme shrink rate must still clamp at coast_trust_min_budget_s,
        not go negative or to zero outright -- defensive, matches
        _effective_coast_budget_s's own floor semantics."""
        tracker = self._make_tracker_with_imu(
            [(0, [0.0, 0.0, 0.0]), (int(0.005 * _NS), [0.0, 0.0, np.radians(5000.0)])],
            [(0, [0.0, 0.0, 9.81]), (int(0.005 * _NS), [0.0, 0.0, 9.81])],
            coast_trust_shrink_s_per_dps=10.0,  # absurdly large -- would go deeply negative unclamped
            coast_trust_min_budget_s=0.003,
        )
        calls = []
        tracker._fusion_filter.predict = _stub_predict(calls)

        # 3ms elapsed -- inside the 3ms floor, so predict() should still fire
        # despite the enormous nominal shrink.
        tracker._mark_all_lost(frame_ts_ns=int(0.003 * _NS))
        self.assertEqual(len(calls), 1, "the floor must still grant a tiny budget, not clamp to zero")

    def test_default_rates_are_a_pure_no_op(self):
        """coast_trust_shrink_s_per_* default to 0.0 (see the real
        matching.coast_trust_* config comment) -- must not change behaviour
        at all until fitted, regardless of how violent the window was."""
        tracker = self._make_tracker_with_imu(
            [(0, [0.0, 0.0, 0.0]), (int(0.05 * _NS), [0.0, 0.0, np.radians(3000.0)])],
            [(0, [0.0, 0.0, 9.81]), (int(0.05 * _NS), [0.0, 0.0, 9.81])],
        )  # no coast_trust_shrink_* override -- stays at the 0.0 default
        calls = []
        tracker._fusion_filter.predict = _stub_predict(calls)

        tracker._mark_all_lost(frame_ts_ns=int(0.05 * _NS))  # 50ms < 66ms budget

        self.assertEqual(len(calls), 1, "zero shrink rates must reproduce today's flat-budget behavior exactly")


if __name__ == "__main__":
    unittest.main()
