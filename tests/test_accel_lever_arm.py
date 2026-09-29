"""Exact-value tests for accel_lever_arm_body (src/imu_data.py): the accelerometer position in the
controller BODY frame. Background: the tracker used the raw factory translation t_acc (sensor frame) as
the lever arm -- right length, ~95 deg wrong direction; the correct body-frame position is the
translation of Rt's inverse, -R^T t (see fix_design.md, analysis/imu_bias/REPORT.md, and the independent
review that caught the bridge-direction slip in the first design).

Run with: python3 -m unittest tests.test_accel_lever_arm
"""
import json
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.imu_data import (accel_lever_arm_body, create_imu_calib_from_config, integrate_accel_to_position,
                           _lever_arm_correction)
from src.load_config import load_json_config
from src.mocap_data import load_mocap_bridge
from src.transformations import Transform

_REPO = Path(__file__).resolve().parent.parent
_JSON = {"left": _REPO / "data" / "controllers" / "left_controller_A85K5091630091L.json",
         "right": _REPO / "data" / "controllers" / "right_controller_A85K6081930636R.json"}
_BRIDGE = {"left": _REPO / "data" / "mocap_calib" / "controller_left_mocap_bridge_basalt01.json",
           "right": _REPO / "data" / "mocap_calib" / "controller_right_mocap_bridge_basalt01.json"}
_HAVE_FILES = all(p.exists() for p in list(_JSON.values()) + list(_BRIDGE.values()))


def _raw_accel_rt(cfg, entry_index=1):
    """(R (3,3) row-major as stored, t (3,)) of the accelerometer entry, read straight from the JSON."""
    sensors = cfg["CalibrationInformation"]["InertialSensors"]
    a = [s for s in sensors if s["SensorType"] == "CALIBRATION_InertialSensorType_Accelerometer"][entry_index]
    return (np.array(a["Rt"]["Rotation"], dtype=np.float64).reshape(3, 3),
            np.array(a["Rt"]["Translation"], dtype=np.float64))


def _angle_deg(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))


@unittest.skipUnless(_HAVE_FILES, "controller JSONs / bridges not present")
class LeverArmValueTests(unittest.TestCase):
    def test_equals_minus_RT_t_from_raw_json_and_gyro_translation_is_zero(self):
        for side in ("left", "right"):
            cfg = load_json_config(str(_JSON[side]))
            calib = create_imu_calib_from_config(cfg)
            R, t = _raw_accel_rt(cfg)
            np.testing.assert_array_equal(calib.gyro.T_rt.t, 0.0)          # precondition of the whole formulation
            np.testing.assert_allclose(accel_lever_arm_body(calib), -R.T @ t, atol=1e-15)

    def test_shape_dtype_and_magnitude(self):
        for side in ("left", "right"):
            r = accel_lever_arm_body(create_imu_calib_from_config(load_json_config(str(_JSON[side]))))
            self.assertEqual(r.shape, (3,))
            self.assertEqual(r.dtype, np.float64)
            self.assertTrue(0.080 < np.linalg.norm(r) < 0.090, f"{side}: {np.linalg.norm(r) * 1000:.1f} mm")

    def test_it_is_not_the_old_raw_factory_translation(self):
        """The old expression's value (accel.T_rt.compose(gyro.T_rt.inverse()).t == raw t_acc here) points
        ~95 deg away; pin that the new vector is genuinely different so a silent revert fails."""
        for side in ("left", "right"):
            calib = create_imu_calib_from_config(load_json_config(str(_JSON[side])))
            old = calib.accel.T_rt.compose(calib.gyro.T_rt.inverse()).t
            new = accel_lever_arm_body(calib)
            self.assertAlmostEqual(np.linalg.norm(old), np.linalg.norm(new), delta=2e-3)   # same length
            self.assertGreater(_angle_deg(old, new), 85.0)                                # very different direction

    def test_expected_axes_and_sign_pattern(self):
        """Body-frame accelerometer position is dominated by -z (~ -84 mm) with small x/y (x sign is pinned by the bridge test)."""
        rl = accel_lever_arm_body(create_imu_calib_from_config(load_json_config(str(_JSON["left"]))))
        rr = accel_lever_arm_body(create_imu_calib_from_config(load_json_config(str(_JSON["right"]))))
        for r in (rl, rr):
            self.assertLess(r[2], -0.075)
            self.assertLess(abs(r[0]), 0.010)
            self.assertLess(abs(r[1]), 0.015)


@unittest.skipUnless(_HAVE_FILES, "controller JSONs / bridges not present")
class LeverArmVsBridgeTests(unittest.TestCase):
    """Independent, MOCAP-FREE factory derivation vs the bridge fitted against vision+mocap. Bridge
    semantics (compare_vision_mocap.py:165 residual = T_vision.compose(bridge).inverse().compose(mocap_rel);
    mocap_data.load_mocap_bridge docstring): the bridge maps accel-IMU-frame points to LED-frame points, so
    the accel origin in the LED frame is bridge.t itself (NOT -R^T t -- the first design's slip)."""

    def test_agrees_with_bridge_translation(self):
        for side in ("left", "right"):
            r = accel_lever_arm_body(create_imu_calib_from_config(load_json_config(str(_JSON[side]))))
            b = load_mocap_bridge(str(_BRIDGE[side]))
            self.assertLess(_angle_deg(r, b.t), 3.0, f"{side} angle")            # measured 1.7 / 1.2 deg
            np.testing.assert_allclose(r, b.t, atol=0.005, err_msg=f"{side} per-axis (mm-level)")   # measured <= ~4 mm

    def test_sign_guard_against_the_wrong_first_design(self):
        """bridge.t x-sign is (+, -) for (left, right); the wrong -R_b^T t_b has the opposite x sign. The
        factory-derived vector must side with bridge.t, not with the wrong one."""
        for side, sx in (("left", +1.0), ("right", -1.0)):
            b = load_mocap_bridge(str(_BRIDGE[side]))
            wrong = -b.R.T @ b.t
            r = accel_lever_arm_body(create_imu_calib_from_config(load_json_config(str(_JSON[side]))))
            self.assertGreater(sx * b.t[0], 0.0)
            self.assertLess(sx * wrong[0], 0.0)                                   # the trap really flips x
            self.assertGreater(sx * r[0], 0.0, f"{side}: lever arm x sign follows bridge.t, not the wrong design")
            self.assertLess(np.linalg.norm(r - b.t), np.linalg.norm(r - wrong))    # strictly closer to the right one


class BridgeConventionTests(unittest.TestCase):
    """Fixes the numeric convention of the bridge with a hand-built transform: with R = diag(1,-1,-1) and
    t = (a,b,c), the IMU-frame ORIGIN maps to the LED-frame point (a,b,c) -- exactly bridge.t, sign preserved --
    and T_world_ctrl.compose(bridge) places that point in the world at T_world_ctrl(bridge.t)."""

    def test_imu_origin_in_led_frame_is_bridge_t_not_minus_RT_t(self):
        a, b, c = 0.0055, 0.0069, -0.0828
        bridge = Transform(np.diag([1.0, -1.0, -1.0]), np.array([a, b, c]))
        origin_led = bridge.apply(np.zeros((1, 3)))[0]
        np.testing.assert_allclose(origin_led, [a, b, c], atol=1e-15)
        np.testing.assert_allclose(-bridge.R.T @ bridge.t, [-a, b, c], atol=1e-15)     # the x flip trap
        self.assertNotAlmostEqual(origin_led[0], (-bridge.R.T @ bridge.t)[0], places=6)

    def test_compose_places_imu_origin_at_world_pose_of_bridge_t(self):
        T_world_ctrl = Transform(Rotation.from_euler("xyz", [0.3, -0.5, 1.1]).as_matrix(), np.array([0.2, -0.1, 0.7]))
        bridge = Transform(np.diag([1.0, -1.0, -1.0]), np.array([0.0055, 0.0069, -0.0828]))
        T_world_imu = T_world_ctrl.compose(bridge)                 # what compare_vision_mocap equates to relative_pose
        np.testing.assert_allclose(T_world_imu.t, T_world_ctrl.R @ bridge.t + T_world_ctrl.t, atol=1e-15)


class SpinningBodyPhysicsTests(unittest.TestCase):
    """'Prove the bug, prove the fix' through the project's own integrate_accel_to_position. Body origin at
    rest (a_origin = 0), rotating at constant omega about a fixed body axis: R(t) = exp(t [omega]x). An
    accelerometer at body-frame position r_true reads f = omega x (omega x r_true) - R(t)^T g. Integrating with
    r = r_true must give zero position change; any other r leaves omega x (omega x (r_true - r)) uncancelled."""

    G = np.array([0.0, -9.81, 0.0])            # gravity vector; project's ADDITIVE convention a_world = R f + g_world, g_world = G
    OMEGA = np.array([2.0, 3.0, 4.0])          # rad/s, exercises all three components
    T0 = 5_000_000_000
    N = 101                                    # 0..0.5 s at 200 Hz
    DT_NS = 5_000_000

    def _stream(self, r_true):
        ts = self.T0 + np.arange(self.N, dtype=np.int64) * self.DT_NS
        t_s = (ts - self.T0) / 1e9
        Rs = np.stack([Rotation.from_rotvec(self.OMEGA * t).as_matrix() for t in t_s])
        centripetal = np.cross(self.OMEGA, np.cross(self.OMEGA, r_true))
        accel = centripetal[None, :] - np.einsum("nji,j->ni", Rs, self.G)        # omega x (omega x r) - R^T g
        gyro = np.tile(self.OMEGA, (self.N, 1))
        return ts, accel, gyro, Rs

    def _dp(self, r_true, r_used):
        ts, accel, gyro, Rs = self._stream(r_true)
        return integrate_accel_to_position(ts, accel, ts[0], ts[-1], Rs[0], Rs[-1], np.zeros(3), self.G,
                                           t_gyro=ts, gyro_body=gyro, r=r_used)

    def test_correct_lever_arm_gives_zero_position_change(self):
        r_true = np.array([0.0056, 0.0074, -0.0839])
        np.testing.assert_allclose(self._dp(r_true, r_true), 0.0, atol=1e-9)

    def test_wrong_direction_lever_arm_leaves_a_large_error_the_bug_case(self):
        r_true = np.array([0.0056, 0.0074, -0.0839])                     # body-frame accel position (bridge-like)
        r_old = np.array([0.0346, -0.0775, 0.0028])                       # what main.py used: raw factory t (sensor frame)
        err_old = np.linalg.norm(self._dp(r_true, r_old))
        err_zero = np.linalg.norm(self._dp(r_true, np.zeros(3)))
        err_fix = np.linalg.norm(self._dp(r_true, r_true))
        self.assertLess(err_fix, 1e-9)
        self.assertGreater(err_old, 0.05)          # tens of cm-scale error from omega^2 * ~125 mm mismatch over 0.5 s
        self.assertGreater(err_zero, 0.01)         # 'no lever arm at all' is also clearly wrong at this rate
        # exact analytic check of the residual acceleration: mismatch = omega x (omega x (r_true - r_used));
        # its double integral over the SAME rotating frame -> compare with an independent high-resolution sum
        ts, _, _, Rs = self._stream(r_true)
        d = np.cross(self.OMEGA, np.cross(self.OMEGA, r_true - r_old))
        a_w = np.einsum("nij,j->ni", Rs, d)
        v = np.zeros((self.N, 3)); p = np.zeros(3)
        for i in range(self.N - 1):
            dt = self.DT_NS / 1e9
            v[i + 1] = v[i] + 0.5 * (a_w[i] + a_w[i + 1]) * dt
            p += 0.5 * (v[i] + v[i + 1]) * dt
        np.testing.assert_allclose(self._dp(r_true, r_old), p, atol=1e-9)

    def test_correction_is_zero_without_rotation_any_r_identical(self):
        """omega = 0, alpha = 0: the lever arm has no effect at all (invariance / no-op reduction)."""
        ts = self.T0 + np.arange(21, dtype=np.int64) * self.DT_NS
        gyro = np.zeros((21, 3))
        for r in (np.zeros(3), np.array([0.03, -0.08, 0.003]), np.array([0.0056, 0.0074, -0.0839])):
            np.testing.assert_array_equal(_lever_arm_correction(ts, ts, gyro, r), 0.0)

    def test_centripetal_term_matches_hand_value(self):
        """omega = (0,0,6) rad/s, r = (0.084,0,0): omega x (omega x r) = -omega^2 r = (-3.024, 0, 0) exactly."""
        ts = self.T0 + np.arange(21, dtype=np.int64) * self.DT_NS
        gyro = np.tile([0.0, 0.0, 6.0], (21, 1))
        corr = _lever_arm_correction(ts, ts, gyro, np.array([0.084, 0.0, 0.0]))
        np.testing.assert_allclose(corr, np.tile([-3.024, 0.0, 0.0], (21, 1)), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
