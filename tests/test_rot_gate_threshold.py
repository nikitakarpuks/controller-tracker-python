"""Pure-function tests for the continuous rotation gate: rot_gate_threshold_deg and gyro_window_clip_flags."""
import unittest

import numpy as np

from src.imu_data import gyro_window_clip_flags
from src.pose_fusion_heuristic import rot_gate_threshold_deg, _ROT_GATE_EXPIRED_DEG

MS = 1_000_000


class RotGateThresholdTests(unittest.TestCase):
    def test_headset_active_is_flat_base(self):
        for dt in (0.0, 0.01, 0.1, 0.25, 0.35):
            self.assertAlmostEqual(rot_gate_threshold_deg(dt, True, False), 40.0)

    def test_headset_active_saturated_adds_allowance_and_caps(self):
        self.assertAlmostEqual(rot_gate_threshold_deg(0.1, True, True), 65.0)
        self.assertAlmostEqual(rot_gate_threshold_deg(0.1, True, True, sat_allow_deg=60.0), 75.0)  # cap_hs

    def test_no_headset_grows_linearly_with_dt(self):
        self.assertAlmostEqual(rot_gate_threshold_deg(0.0, False, False), 40.0)
        self.assertAlmostEqual(rot_gate_threshold_deg(0.0555, False, False), 40.0 + 500.0 * 0.0555)
        self.assertAlmostEqual(rot_gate_threshold_deg(0.1, False, True), 40.0 + 25.0 + 50.0)

    def test_no_headset_caps_at_150(self):
        self.assertAlmostEqual(rot_gate_threshold_deg(0.2441, False, False), 150.0)
        self.assertAlmostEqual(rot_gate_threshold_deg(0.35, False, True), 150.0)

    def test_expired_state_returns_179(self):
        self.assertEqual(rot_gate_threshold_deg(3.01, True, False), _ROT_GATE_EXPIRED_DEG)
        self.assertEqual(rot_gate_threshold_deg(5.0, False, True), _ROT_GATE_EXPIRED_DEG)

    def test_finite_limit_past_0p35s(self):
        # a wrong 135 deg pair was confirmed at 0.455 s (walk_medium right, 93.70 s): still gated now
        self.assertAlmostEqual(rot_gate_threshold_deg(0.455, True, False), 40.0)
        self.assertGreater(135.0, rot_gate_threshold_deg(0.455, True, False))
        self.assertAlmostEqual(rot_gate_threshold_deg(1.5, False, False), 150.0)   # no headset: capped
        self.assertAlmostEqual(rot_gate_threshold_deg(3.0, True, True), 65.0)

    def test_negative_dt_clamped(self):
        self.assertAlmostEqual(rot_gate_threshold_deg(-1.0, False, False), 40.0)

    def test_monotone_in_dt_without_headset(self):
        ts = np.linspace(0.0, 0.35, 50)
        v = [rot_gate_threshold_deg(t, False, False) for t in ts]
        self.assertTrue(all(b >= a for a, b in zip(v, v[1:])))

    def test_real_case_anchors(self):
        # walk_medium right 348: 172 deg at dt 0.2441 (no headset) is vetoed
        self.assertGreater(172.0, rot_gate_threshold_deg(0.2441, False, False))
        # walk_medium left 5333: 87.3 deg at dt 0.0555 vetoed without headset, and with it
        self.assertGreater(87.3, rot_gate_threshold_deg(0.0555, False, False))
        self.assertGreater(87.3, rot_gate_threshold_deg(0.0555, True, False))
        # GOOD guards: walk_easy 2046 (25.2 deg, dt 0.155), walk_hard 4873/4874 (41.9 deg, saturated window)
        self.assertLess(25.2, rot_gate_threshold_deg(0.155, True, False))
        self.assertLess(25.2, rot_gate_threshold_deg(0.155, False, False))
        self.assertLess(41.9, rot_gate_threshold_deg(0.02, True, True))


class GyroWindowClipFlagsTests(unittest.TestCase):
    def _data(self, w_dps_rows, step_ns=5 * MS):
        t = np.arange(len(w_dps_rows), dtype=np.int64) * step_ns
        return t, np.radians(np.asarray(w_dps_rows, dtype=float))

    def test_missing_inputs(self):
        self.assertEqual(gyro_window_clip_flags(None, 0, 10), (False, 0.0))

    def test_calm_not_clipped(self):
        d = self._data([[100, 0, 0]] * 40)
        c, peak = gyro_window_clip_flags(d, 0, 100 * MS)
        self.assertFalse(c)
        self.assertAlmostEqual(peak, 100.0, places=3)

    def test_x_clip_detected(self):
        rows = [[100, 0, 0]] * 40
        rows[20] = [1950, 0, 0]  # >= 0.95 * 2000
        c, _ = gyro_window_clip_flags(self._data(rows), 90 * MS, 110 * MS)
        self.assertTrue(c)

    def test_z_ceiling_is_higher(self):
        rows = [[0, 0, 0]] * 40
        rows[20] = [0, 0, 1950]  # below 0.95 * 2722
        c, _ = gyro_window_clip_flags(self._data(rows), 90 * MS, 110 * MS)
        self.assertFalse(c)
        rows[20] = [0, 0, 2650]
        c, _ = gyro_window_clip_flags(self._data(rows), 90 * MS, 110 * MS)
        self.assertTrue(c)

    def test_padding_extends_window(self):
        rows = [[100, 0, 0]] * 40
        rows[20] = [2000, 0, 0]  # at t=100 ms
        d = self._data(rows)
        self.assertTrue(gyro_window_clip_flags(d, 105 * MS, 150 * MS)[0])      # within 10 ms pad
        self.assertFalse(gyro_window_clip_flags(d, 115 * MS, 150 * MS)[0])     # outside pad


if __name__ == "__main__":
    unittest.main()


class HeadsetRotCoastBudgetTests(unittest.TestCase):
    def test_shape(self):
        from src.imu_data import headset_rot_coast_budget_s as f
        self.assertAlmostEqual(f(0.0), 0.30)
        self.assertAlmostEqual(f(1000.0), 0.30)
        self.assertAlmostEqual(f(1250.0), 0.30 + 0.5 * (0.035 - 0.30))
        self.assertAlmostEqual(f(1500.0), 0.035)
        self.assertAlmostEqual(f(4000.0), 0.035)

    def test_monotone_non_increasing(self):
        from src.imu_data import headset_rot_coast_budget_s as f
        v = [f(g) for g in np.linspace(0, 3000, 200)]
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(v, v[1:])))
