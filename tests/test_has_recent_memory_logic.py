"""Regression coverage for main.py's _is_recent_memory_usable -- the pure
boolean logic behind static_lamp_mask's has_recent_memory feature (skip
lamp exclusion on a cold re-detect if the controller has a recent, strong
last_good_pose, since ControllerTracker's own vs-last_good_pose jump-gate
already independently corroborates that kind of candidate).

This logic was found unsafe once already (a 3-agent critique of a first
version that checked ONLY "is last_good_pose set at all"): last_good_pose
is retained indefinitely across loss events by design, while the jump-gate
it leans on has no cap on its own staleness-based widening and reaches a
full no-op within roughly 70ms at this project's real shipped config. It
earns direct, isolated unit tests rather than only incidental exercise
through the full pipeline.

Uses stdlib unittest (pytest is not a declared dependency of this project).
Run with:  python3 -m unittest tests.test_has_recent_memory_logic
"""
import unittest

from main import _is_recent_memory_usable

_MAX_STALE_S = 0.05
_MIN_INLIERS = 6
_MAX_ERROR_PX = 0.5
_SOME_POSE = (object(), object())  # content is never inspected, only is-None-ness matters


class HasRecentMemoryLogicTests(unittest.TestCase):
    def _call(self, last_good_pose=_SOME_POSE, last_good_pose_ts_ns=1_000_000_000,
               last_good_pose_quality=(10, 0.1), frame_ts_ns=1_010_000_000):
        return _is_recent_memory_usable(
            last_good_pose, last_good_pose_ts_ns, last_good_pose_quality, frame_ts_ns,
            _MAX_STALE_S, _MIN_INLIERS, _MAX_ERROR_PX)

    def test_fresh_strong_pose_is_usable(self):
        """10ms stale, well within the 50ms budget; 10 inliers/0.1px, well
        within the strong bar -- the intended common case."""
        self.assertTrue(self._call())

    def test_no_last_good_pose_is_not_usable(self):
        self.assertFalse(self._call(last_good_pose=None))

    def test_missing_timestamp_is_not_usable(self):
        """Can't judge freshness without a timestamp -- conservative default."""
        self.assertFalse(self._call(last_good_pose_ts_ns=None))

    def test_missing_quality_is_not_usable(self):
        """Can't judge strength without a quality record -- conservative default."""
        self.assertFalse(self._call(last_good_pose_quality=None))

    def test_exactly_at_stale_budget_is_usable(self):
        """Boundary: exactly max_stale_s of elapsed time is still usable (<=, not <)."""
        ts = 1_000_000_000
        frame_ts = ts + int(_MAX_STALE_S * 1e9)
        self.assertTrue(self._call(last_good_pose_ts_ns=ts, frame_ts_ns=frame_ts))

    def test_just_past_stale_budget_is_not_usable(self):
        """THE core regression this whole test file exists for: a reference
        that is technically non-None but too old must be rejected -- this is
        exactly the condition the first (unsafe) version of this logic never
        checked at all."""
        ts = 1_000_000_000
        frame_ts = ts + int(_MAX_STALE_S * 1e9) + 1_000_000  # 1ms past the budget
        self.assertFalse(self._call(last_good_pose_ts_ns=ts, frame_ts_ns=frame_ts))

    def test_arbitrarily_stale_pose_is_not_usable(self):
        """A controller merely set down or out of view for a long time (not
        cleared by any identity-swap/contested-winner reset) must NOT read
        as "recent memory" just because last_good_pose still happens to be
        set -- the exact failure mode the critique found."""
        ts = 1_000_000_000
        frame_ts = ts + int(60 * 1e9)  # 60 real seconds later
        self.assertFalse(self._call(last_good_pose_ts_ns=ts, frame_ts_ns=frame_ts))

    def test_negative_staleness_is_not_usable(self):
        """Defensive: a last_good_pose timestamp AFTER frame_ts_ns (clock
        skew / out-of-order call) must not accidentally read as "fresh"
        just because the raw difference is a small-magnitude negative
        number less than max_stale_s."""
        ts = 1_000_000_000
        frame_ts = ts - 1_000_000  # 1ms "before" the reference was captured
        self.assertFalse(self._call(last_good_pose_ts_ns=ts, frame_ts_ns=frame_ts))

    def test_weak_reference_quality_is_not_usable(self):
        """THE second core regression: even a perfectly fresh reference must
        be rejected if its own quality doesn't clear the strong bar --
        otherwise ControllerTracker's _quality_rescue could bypass the
        jump-gate outright against exactly this reference."""
        self.assertFalse(self._call(last_good_pose_quality=(3, 1.2)))  # below both bars

    def test_inliers_just_below_bar_is_not_usable(self):
        self.assertFalse(self._call(last_good_pose_quality=(_MIN_INLIERS - 1, 0.1)))

    def test_error_just_above_bar_is_not_usable(self):
        self.assertFalse(self._call(last_good_pose_quality=(10, _MAX_ERROR_PX + 0.01)))

    def test_exactly_at_quality_bars_is_usable(self):
        """Boundary: exactly meeting both bars (>=, <=) is usable."""
        self.assertTrue(self._call(last_good_pose_quality=(_MIN_INLIERS, _MAX_ERROR_PX)))


if __name__ == "__main__":
    unittest.main()
