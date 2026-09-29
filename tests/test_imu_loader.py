"""Exact-value tests for load_and_calibrate_controller_imu's 2026-09-23 change (recorded controller
IMU CSVs are already factory-corrected by the Monado driver -> do not apply mix+bias again; accel
driver-scale fix). Expected values are computed here from the raw inputs / factory JSON arrays, NOT by
calling the code under test's own helpers, so a convention slip in either place would fail.

Run with: python3 -m unittest tests.test_imu_loader
"""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.imu_data import (ACCEL_DRIVER_SCALE, ACCEL_QUIET_MAGNITUDE_BAND, _DIAG_FLIP, imu_loader_kwargs,
                           load_and_calibrate_controller_imu, load_imu_csv, median_quiet_accel_magnitude)
from src.load_config import load_json_config, load_yaml_config

_REPO = Path(__file__).resolve().parent.parent
_LEFT_JSON = _REPO / "data" / "controllers" / "left_controller_A85K5091630091L.json"
_REAL_IMU1 = Path("/home/nikitakarpuks/Downloads/recordings-aug26/euroc_recording_20260826173103_static_dark/mav0/imu1/data.csv")


def _write_imu_csv(path, rows):
    """rows: list of (t_ns, gx, gy, gz, ax, ay, az) -- EuRoC layout with one header line."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["#timestamp [ns]", "w_x", "w_y", "w_z", "a_x", "a_y", "a_z"])
        w.writerows(rows)


def _factory_arrays(cfg, entry_index=1):
    """(gyro_mix0 (3,3), gyro_bias0 (3,), accel_mix0, accel_bias0) straight from the JSON: constant (T=0)
    coefficient of each temperature polynomial is every 4th element -- matches what Monado reads."""
    sensors = cfg["CalibrationInformation"]["InertialSensors"]
    g = [s for s in sensors if s["SensorType"] == "CALIBRATION_InertialSensorType_Gyro"][entry_index]
    a = [s for s in sensors if s["SensorType"] == "CALIBRATION_InertialSensorType_Accelerometer"][entry_index]
    return (np.array(g["MixingMatrixTemperatureModel"][0::4], dtype=np.float64).reshape(3, 3),
            np.array(g["BiasTemperatureModel"][0::4], dtype=np.float64),
            np.array(a["MixingMatrixTemperatureModel"][0::4], dtype=np.float64).reshape(3, 3),
            np.array(a["BiasTemperatureModel"][0::4], dtype=np.float64))


class LoaderNewDefaultTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "imu.csv"
        # values exactly representable in float32 so the file->float32->float64 path adds no error
        self.raw = np.array([[0.125, 0.25, 0.5, 1.0, 2.0, 3.0],
                             [-0.5, 0.0, 0.75, -4.0, 8.0, 0.5]])
        _write_imu_csv(self.path, [(1_000 + i, *r) for i, r in enumerate(self.raw)])

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_is_axis_flip_plus_accel_scale_no_factory_correction(self):
        t, gyro, accel = load_and_calibrate_controller_imu(self.path, cfg_unused := {}, lag_ns=0)
        np.testing.assert_allclose(gyro, self.raw[:, 0:3] * np.array([1.0, -1.0, -1.0]), atol=1e-15)
        np.testing.assert_allclose(accel, 0.980665 * self.raw[:, 3:6] * np.array([1.0, -1.0, -1.0]), atol=1e-13)
        # exact worked numbers, hand-derived: gyro (0.125,0.25,0.5) -> (0.125,-0.25,-0.5);
        # accel (1,2,3)-row1 = (1.0,2.0,3.0) -> 0.980665*(1,-2,-3)
        np.testing.assert_allclose(gyro[0], [0.125, -0.25, -0.5], atol=1e-15)
        np.testing.assert_allclose(accel[0], [0.980665, -1.96133, -2.941995], atol=1e-12)

    def test_default_never_touches_the_controller_config(self):
        """factory_corrected_input=True must not even read InertialSensors (a raw {} would KeyError otherwise)."""
        load_and_calibrate_controller_imu(self.path, {}, lag_ns=0)  # must not raise

    def test_accel_scale_constant_value(self):
        self.assertEqual(ACCEL_DRIVER_SCALE, 9.80665 / 10.0)
        self.assertAlmostEqual(ACCEL_DRIVER_SCALE, 0.980665, places=15)

    def test_accel_scale_one_is_pure_axis_flip(self):
        _, _, accel = load_and_calibrate_controller_imu(self.path, {}, accel_scale=1.0)
        np.testing.assert_allclose(accel, self.raw[:, 3:6] * np.array([1.0, -1.0, -1.0]), atol=1e-15)

    def test_gyro_is_never_scaled(self):
        _, gyro_a, _ = load_and_calibrate_controller_imu(self.path, {}, accel_scale=1.0)
        _, gyro_b, _ = load_and_calibrate_controller_imu(self.path, {}, accel_scale=0.5)
        np.testing.assert_array_equal(gyro_a, gyro_b)

    def test_timestamps_get_lag_and_stay_int64(self):
        t, _, _ = load_and_calibrate_controller_imu(self.path, {}, lag_ns=-7_650_000)
        self.assertEqual(t.dtype, np.int64)
        np.testing.assert_array_equal(t, np.array([1_000, 1_001]) - 7_650_000)

    def test_axis_flip_matrix_is_the_confirmed_one(self):
        np.testing.assert_array_equal(_DIAG_FLIP, np.diag([1.0, -1.0, -1.0]))


@unittest.skipUnless(_LEFT_JSON.exists(), "factory controller JSON not present")
class LoaderLegacySwitchTests(unittest.TestCase):
    """factory_corrected_input=False + accel_scale=1.0 must reproduce the pre-2026-09-23 chain EXACTLY:
    out = D @ (mix0 @ raw + bias0), expected computed here from the JSON arrays independently."""

    def test_legacy_matches_independent_computation(self):
        cfg = load_json_config(str(_LEFT_JSON))
        gmix, gbias, amix, abias = _factory_arrays(cfg, entry_index=1)
        rng = np.random.default_rng(3)
        raw = rng.normal(size=(50, 6)).astype(np.float32).astype(np.float64)   # float32-exact inputs
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "imu.csv"
            _write_imu_csv(path, [(i, *r) for i, r in enumerate(raw)])
            # the CSV round trip stores 17-sig-digit text; the loader reads float32, so build expectation from that
            _, g32, a32 = load_imu_csv(path)
            t, gyro, accel = load_and_calibrate_controller_imu(path, cfg, lag_ns=123, factory_corrected_input=False,
                                                                accel_scale=1.0)
        exp_gyro = ((np.diag([1.0, -1.0, -1.0]) @ ((gmix @ g32.astype(np.float64).T).T + gbias).T)).T
        exp_accel = ((np.diag([1.0, -1.0, -1.0]) @ ((amix @ a32.astype(np.float64).T).T + abias).T)).T
        np.testing.assert_allclose(gyro, exp_gyro, atol=1e-13)
        np.testing.assert_allclose(accel, exp_accel, atol=1e-13)
        np.testing.assert_array_equal(t, np.arange(50) + 123)

    def test_legacy_differs_from_new_default_by_exactly_the_factory_terms(self):
        """Sanity that the two paths are really different: for zero raw input the legacy output is the
        flipped factory bias (nonzero), the new default is exactly zero."""
        cfg = load_json_config(str(_LEFT_JSON))
        _, gbias, _, abias = _factory_arrays(cfg, entry_index=1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "imu.csv"
            _write_imu_csv(path, [(0, 0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0)])
            _, g_new, a_new = load_and_calibrate_controller_imu(path, cfg)
            _, g_old, a_old = load_and_calibrate_controller_imu(path, cfg, factory_corrected_input=False, accel_scale=1.0)
        np.testing.assert_array_equal(g_new, 0.0)
        np.testing.assert_array_equal(a_new, 0.0)
        np.testing.assert_allclose(g_old[0], np.array([1, -1, -1]) * gbias, atol=1e-15)
        np.testing.assert_allclose(a_old[0], np.array([1, -1, -1]) * abias, atol=1e-15)
        self.assertGreater(np.linalg.norm(a_old[0]), 1e-3)   # left accel factory bias is nonzero


@unittest.skipUnless(_REAL_IMU1.exists(), "real recording not present")
class LoaderRealFileTests(unittest.TestCase):
    def test_real_slice_matches_flip_and_scale_bit_for_bit(self):
        t_raw, gyro_raw, accel_raw = load_imu_csv(_REAL_IMU1)
        t, gyro, accel = load_and_calibrate_controller_imu(_REAL_IMU1, {}, lag_ns=0)
        np.testing.assert_array_equal(t, t_raw)
        np.testing.assert_allclose(gyro, gyro_raw.astype(np.float64) * np.array([1.0, -1.0, -1.0]), atol=0, rtol=0)
        np.testing.assert_allclose(accel, ACCEL_DRIVER_SCALE * (accel_raw.astype(np.float64) * np.array([1.0, -1.0, -1.0])),
                                   atol=0, rtol=1e-15)


class ConfigKwargsTests(unittest.TestCase):
    def test_defaults_when_unset(self):
        for cfg in (None, {}, {"recorded_stream_factory_corrected": None, "accel_driver_scale": None}):
            self.assertEqual(imu_loader_kwargs(cfg), {"factory_corrected_input": True, "accel_scale": ACCEL_DRIVER_SCALE})

    def test_overrides(self):
        kw = imu_loader_kwargs({"recorded_stream_factory_corrected": False, "accel_driver_scale": 1.0})
        self.assertEqual(kw, {"factory_corrected_input": False, "accel_scale": 1.0})

    def test_shipped_config_uses_the_fix(self):
        cfg = load_yaml_config(str(_REPO / "config" / "config.yml"))
        kw = imu_loader_kwargs(cfg["imu"])
        self.assertTrue(kw["factory_corrected_input"])
        self.assertAlmostEqual(kw["accel_scale"], 0.980665, places=12)


class QuietAccelMagnitudeGuardTests(unittest.TestCase):
    def test_hand_values_and_quiet_mask(self):
        n = 300
        gyro = np.zeros((n, 3)); accel = np.tile([0.0, 0.0, 9.81], (n, 1))
        self.assertAlmostEqual(median_quiet_accel_magnitude(gyro, accel), 9.81, places=12)
        # samples above the rotation threshold are excluded, whatever their accel
        gyro2 = np.vstack([gyro, np.tile([0.0, 0.0, 3.0], (500, 1))])
        accel2 = np.vstack([accel, np.tile([0.0, 0.0, 50.0], (500, 1))])
        self.assertAlmostEqual(median_quiet_accel_magnitude(gyro2, accel2), 9.81, places=12)
        # 3-4-5 triangle: |(3,4,0)| = 5 exactly
        self.assertAlmostEqual(median_quiet_accel_magnitude(gyro, np.tile([3.0, 4.0, 0.0], (n, 1))), 5.0, places=12)

    def test_too_few_quiet_samples_returns_none(self):
        gyro = np.tile([0.0, 0.0, 3.0], (1000, 1)); accel = np.tile([0.0, 0.0, 9.81], (1000, 1))
        self.assertIsNone(median_quiet_accel_magnitude(gyro, accel))
        self.assertIsNone(median_quiet_accel_magnitude(gyro[:99] * 0, accel[:99]))    # 99 < min_samples=100
        self.assertIsNotNone(median_quiet_accel_magnitude(gyro[:100] * 0, accel[:100]))   # exactly 100 is enough

    def test_band_separates_correct_double_corrected_and_unscaled(self):
        lo, hi = ACCEL_QUIET_MAGNITUDE_BAND
        self.assertTrue(lo <= 9.80665 <= hi)                         # true gravity is inside
        self.assertTrue(lo <= 9.80665 * 1.01 <= hi)                  # +1 % (real per-session bias) still inside
        self.assertFalse(lo <= 10.13 <= hi)                          # smallest old double-corrected reading is outside
        self.assertFalse(lo <= 10.0 * 1.0 + 0.1 <= hi)               # unscaled driver units (10.0 + bias) are outside

    def test_real_recordings_all_inside_band_both_controllers(self):
        root = Path("/home/nikitakarpuks/Downloads/recordings-aug26")
        recs = sorted(root.glob("euroc_recording_*")) if root.exists() else []
        if not recs:
            self.skipTest("recordings not present")
        checked = 0
        for rec in recs:
            for imu in ("imu1", "imu2"):
                path = rec / "mav0" / imu / "data.csv"
                if not path.exists():
                    continue
                _, gyro, accel = load_and_calibrate_controller_imu(path, {}, lag_ns=0)
                v = median_quiet_accel_magnitude(gyro, accel)
                self.assertIsNotNone(v, f"{rec.name}/{imu}: no quiet samples")
                self.assertTrue(ACCEL_QUIET_MAGNITUDE_BAND[0] <= v <= ACCEL_QUIET_MAGNITUDE_BAND[1],
                                f"{rec.name}/{imu}: median quiet |accel| = {v:.3f} outside {ACCEL_QUIET_MAGNITUDE_BAND}")
                # and the OLD chain must fall outside it (that is the whole point of the guard)
                cfg_path = _LEFT_JSON if imu == "imu1" else _LEFT_JSON.with_name("right_controller_A85K6081930636R.json")
                _, g_old, a_old = load_and_calibrate_controller_imu(path, load_json_config(str(cfg_path)),
                                                                      factory_corrected_input=False, accel_scale=1.0)
                v_old = median_quiet_accel_magnitude(g_old, a_old)
                self.assertGreater(v_old, ACCEL_QUIET_MAGNITUDE_BAND[1], f"{rec.name}/{imu}: old chain {v_old:.3f}")
                checked += 1
        self.assertGreaterEqual(checked, 8)


if __name__ == "__main__":
    unittest.main()
