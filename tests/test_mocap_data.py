"""Unit tests for src.mocap_data's two new headset-ego-motion helpers
(headset_angular_velocity, headset_linear_velocity) added for the headset-ego-motion-corrected
dead-reckoning feature (see src.imu_data.predict_headset_relative_pose). No test file existed for
src/mocap_data.py before this.

All DeviceMocap fixtures here are built directly from hand-picked analytic curves (no CSV/JSON
loading) -- matching this project's convention of synthetic, exact-where-possible numeric
fixtures for pure-math unit tests. Uses stdlib unittest (pytest is not a declared dependency of
this project). Run with:
    python3 -m unittest tests.test_mocap_data
"""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.mocap_data import DeviceMocap, headset_angular_velocity, headset_linear_velocity, \
    load_vision_offset_ns, relative_pose, controller_imu_lag_ns, controller_imu_files
from src.transformations import Transform

_NS = 1_000_000_000
_NOMINAL_CADENCE_NS = 8_300_000  # ~8.3ms, this module's own documented nominal mocap cadence


def _sample_times(t0_ns: int, duration_s: float, cadence_ns: int = _NOMINAL_CADENCE_NS):
    n = int(duration_s * _NS / cadence_ns)
    return t0_ns + np.arange(n + 1) * cadence_ns


def _make_device(t_ns, R_of_t, p_of_t, T_imu_marker=None, max_interp_gap_ns=None):
    """Build a DeviceMocap whose raw MARKER trajectory is exactly (R_of_t(t), p_of_t(t)) at each
    t in t_ns, with fine_offset_ns=0 (clock-offset handling is a pre-existing, separately-used
    mechanism -- see mocap_data.DeviceMocap.pose_at -- not something these tests need to exercise;
    that's covered implicitly by this feature reusing the SAME already-offset-corrected
    DeviceMocap object main.py already builds, per the implementation plan)."""
    positions = np.array([p_of_t(t) for t in t_ns], dtype=np.float64)
    quats = np.array([Rotation.from_matrix(R_of_t(t)).as_quat() for t in t_ns], dtype=np.float64)
    kwargs = {}
    if max_interp_gap_ns is not None:
        kwargs["max_interp_gap_ns"] = max_interp_gap_ns
    return DeviceMocap(t_ns, positions, quats, fine_offset_ns=0.0,
                        T_imu_marker=T_imu_marker or Transform(np.eye(3), np.zeros(3)), **kwargs)


class ConstantAngularVelocityTests(unittest.TestCase):
    """Identity T_imu_marker (marker IS the IMU) -- isolates the rotation-differencing math.
    A constant-angular-velocity analytic curve is reproduced EXACTLY by SLERP interpolation
    between any two points on it (geodesics compose exactly under SLERP for constant-rate
    rotation), so headset_angular_velocity should recover the true rate to near machine
    precision through the REAL pose_at -> world_pose -> headset_angular_velocity path -- not
    just as an idealized direct-formula check."""

    def test_recovers_known_constant_rate(self):
        omega_true = np.array([0.3, -1.1, 0.7])
        t0 = 10 * _NS
        t_ns = _sample_times(t0, duration_s=1.0)

        def R_of_t(t):
            return Rotation.from_rotvec(omega_true * (t - t0) / 1e9).as_matrix()

        device = _make_device(t_ns, R_of_t, lambda t: np.zeros(3))
        query_ts = t0 + int(0.5 * _NS)
        omega_est = headset_angular_velocity(device, query_ts)
        self.assertIsNotNone(omega_est)
        np.testing.assert_allclose(omega_est, omega_true, atol=1e-8)

    def test_zero_rate_recovers_zero(self):
        t0 = 0
        t_ns = _sample_times(t0, duration_s=0.5)
        device = _make_device(t_ns, lambda t: np.eye(3), lambda t: np.zeros(3))
        omega_est = headset_angular_velocity(device, t0 + int(0.25 * _NS))
        self.assertIsNotNone(omega_est)
        np.testing.assert_allclose(omega_est, np.zeros(3), atol=1e-10)


class ConstantLinearVelocityTests(unittest.TestCase):
    """Identity T_imu_marker, pure straight-line constant-velocity translation -- linear
    interpolation of a linear curve is exact, so this should also match to near machine
    precision."""

    def test_recovers_known_constant_velocity(self):
        v_true = np.array([1.5, -0.4, 0.2])
        t0 = 0
        t_ns = _sample_times(t0, duration_s=1.0)
        p0 = np.array([5.0, -2.0, 1.0])

        def p_of_t(t):
            return p0 + v_true * (t - t0) / 1e9

        device = _make_device(t_ns, lambda t: np.eye(3), p_of_t)
        v_est = headset_linear_velocity(device, t0 + int(0.5 * _NS))
        self.assertIsNotNone(v_est)
        np.testing.assert_allclose(v_est, v_true, atol=1e-8)


class LeverArmInducedVelocityTests(unittest.TestCase):
    """Marker fixed in place (rotation center), constant rotation, NONZERO T_imu_marker
    translational offset -- the IMU origin traces a circle around the marker as the rigid body
    spins, exactly the lever-arm effect this codebase's existing controller-side
    _lever_arm_correction handles for accelerometers. headset_linear_velocity must capture this
    (differencing world_pose's IMU-origin position, not the raw marker position).

    Only approximately exact (unlike the two tests above): world_pose itself is computed exactly
    at the two finite-difference sample points (SLERP is exact for constant-rate rotation, and
    the marker position here is trivially constant/exact), but the CENTRAL-DIFFERENCE estimate of
    a genuinely curved (circular) velocity has a real, small O(window_s^2) truncation error --
    hence the looser (but still tight) tolerance."""

    def test_matches_omega_cross_r_formula(self):
        omega_true = np.array([0.0, 0.0, 2.0])  # single-axis, matches this project's own
                                                 # gravity-aligned-yaw convention elsewhere
        t_off = np.array([0.05, 0.0, 0.0])      # 5cm marker->IMU lever arm, in marker-local coords
        p_marker = np.array([1.0, 1.0, 1.0])    # rotation center, fixed
        t0 = 0
        t_ns = _sample_times(t0, duration_s=1.0)

        def R_of_t(t):
            return Rotation.from_rotvec(omega_true * (t - t0) / 1e9).as_matrix()

        T_imu_marker = Transform(np.eye(3), t_off)
        device = _make_device(t_ns, R_of_t, lambda t: p_marker, T_imu_marker=T_imu_marker)

        query_ts = t0 + int(0.5 * _NS)
        v_est = headset_linear_velocity(device, query_ts)
        self.assertIsNotNone(v_est)

        R_q = R_of_t(query_ts)
        r_q = R_q @ (-t_off)  # world-frame vector from rotation center to the IMU origin
        v_expected = np.cross(omega_true, r_q)
        np.testing.assert_allclose(v_est, v_expected, atol=1e-5)


class CoverageGapTests(unittest.TestCase):
    """None-propagation: a real, expected outcome (marker-occlusion gaps up to ~158ms are
    documented in DeviceMocap's own docstring), not an error case."""

    def test_none_when_finite_diff_endpoint_falls_outside_covered_range(self):
        t0 = 0
        t_ns = _sample_times(t0, duration_s=1.0)
        device = _make_device(t_ns, lambda t: np.eye(3), lambda t: np.zeros(3))
        # 5ms from the very start of coverage -- well within range itself, but a +-10ms window
        # pushes the LOWER endpoint before t_ns[0].
        query_ts = t0 + int(0.005 * _NS)
        self.assertIsNone(headset_angular_velocity(device, query_ts))
        self.assertIsNone(headset_linear_velocity(device, query_ts))

    def test_none_when_finite_diff_endpoint_falls_inside_a_real_occlusion_gap(self):
        t0 = 0
        before = _sample_times(t0, duration_s=0.3)
        gap_ns = 158_000_000  # the documented real-world worst case
        after_start = before[-1] + gap_ns
        after = _sample_times(after_start, duration_s=0.3)
        t_ns = np.concatenate([before, after])
        device = _make_device(t_ns, lambda t: np.eye(3), lambda t: np.zeros(3))
        # Query sits just after the gap opens -- comfortably within the array's overall covered
        # RANGE (t_ns[0] <= query <= t_ns[-1]), but a +-10ms window reaches back across the gap.
        query_ts = int(before[-1] + gap_ns / 2)
        self.assertIsNone(headset_angular_velocity(device, query_ts))
        self.assertIsNone(headset_linear_velocity(device, query_ts))

class VisionOffsetTests(unittest.TestCase):
    """DeviceMocap.vision_offset_ns: lookup is frame_ts + vision_offset_ns + fine_offset_ns.
    Analytic constant-velocity / constant-rate curves are reproduced exactly by linear/SLERP
    interpolation, so expected values are exact (not approximate) up to float precision."""

    V = np.array([0.8, -0.3, 0.5])          # m/s
    W = np.array([0.4, 1.2, -0.9])          # rad/s
    T0 = 50 * _NS

    def _device(self, fine_ns, vision_ns):
        t_ns = _sample_times(self.T0, duration_s=2.0)
        R = lambda t: Rotation.from_rotvec(self.W * (t - self.T0) / 1e9).as_matrix()
        p = lambda t: self.V * (t - self.T0) / 1e9
        positions = np.array([p(t) for t in t_ns])
        quats = np.array([Rotation.from_matrix(R(t)).as_quat() for t in t_ns])
        return DeviceMocap(t_ns, positions, quats, fine_offset_ns=fine_ns,
                           T_imu_marker=Transform(np.eye(3), np.zeros(3)), vision_offset_ns=vision_ns), R, p

    def test_lookup_is_frame_plus_vision_plus_fine(self):
        fine, vis = 100_000_000, 7_600_000
        dev, R, p = self._device(fine, vis)
        q = self.T0 + 300_000_000
        R_got, p_got = dev.pose_at(q)
        t_true = q + fine + vis
        np.testing.assert_allclose(p_got, p(t_true), atol=1e-9)
        np.testing.assert_allclose(R_got, R(t_true), atol=1e-9)

    def test_default_is_zero_offset_old_behavior(self):
        dev, R, p = self._device(100_000_000, 0.0)
        q = self.T0 + 300_000_000
        np.testing.assert_allclose(dev.pose_at(q)[1], p(q + 100_000_000), atol=1e-9)
        self.assertEqual(DeviceMocap(np.array([0, 1]), np.zeros((2, 3)), np.array([[0, 0, 0, 1.0]] * 2),
                                     0.0, Transform(np.eye(3), np.zeros(3))).vision_offset_ns, 0.0)

    def test_lag_removed_relative_pose_residual(self):
        """Controller moving at V, headset static: vision (stamped at true time) vs mocap looked up
        WITHOUT the offset is off by exactly |V| * 7.6ms; WITH it the position error is ~0."""
        fine, vis = 0, 7_600_000
        t_ns = _sample_times(self.T0, duration_s=2.0)
        I = Transform(np.eye(3), np.zeros(3))
        head = DeviceMocap(t_ns, np.zeros((len(t_ns), 3)), np.tile([0, 0, 0, 1.0], (len(t_ns), 1)), 0.0, I)
        ctrl_pos = np.array([self.V * (t - self.T0) / 1e9 for t in t_ns])
        quats = np.tile([0, 0, 0, 1.0], (len(t_ns), 1))
        # vision stamp t_cam sees the controller at ITS true mocap time t_cam + vis
        q = self.T0 + 500_000_000
        truth_at_vision = self.V * (q + vis - self.T0) / 1e9
        with_off = DeviceMocap(t_ns, ctrl_pos, quats, fine, I, vision_offset_ns=vis)
        without = DeviceMocap(t_ns, ctrl_pos, quats, fine, I)
        err_with = np.linalg.norm(relative_pose(head, with_off, q).t - truth_at_vision)
        err_without = np.linalg.norm(relative_pose(head, without, q).t - truth_at_vision)
        self.assertLess(err_with, 1e-9)
        self.assertAlmostEqual(err_without, np.linalg.norm(self.V) * 7.6e-3, places=9)

    def test_load_vision_offset_ns(self):
        self.assertEqual(load_vision_offset_ns(None), 0.0)
        self.assertEqual(load_vision_offset_ns({}), 0.0)
        self.assertEqual(load_vision_offset_ns({"mocap_vision_offset_ns": None}), 0.0)
        self.assertEqual(load_vision_offset_ns({"mocap_vision_offset_ns": 7600000}), 7_600_000.0)


class ControllerImuLagTests(unittest.TestCase):
    """lag_ns (IMU stream) is the negation of mocap_vision_offset_ns (mocap lookup): one physical link,
    one source of truth. Legacy -5ms/-7ms only when the key is unset."""

    def test_lag_is_negated_vision_offset(self):
        cfg = {"controllers": {"left_controller": {"mocap_vision_offset_ns": 7_650_000},
                               "right_controller": {"mocap_vision_offset_ns": 7_600_000.0}}}
        self.assertEqual(controller_imu_lag_ns("left_controller", cfg), -7_650_000)
        self.assertEqual(controller_imu_lag_ns("right_controller", cfg), -7_600_000)
        self.assertIsInstance(controller_imu_lag_ns("right_controller", cfg), int)

    def test_legacy_fallback_when_unset(self):
        cfg = {"controllers": {"left_controller": {}, "right_controller": {"mocap_vision_offset_ns": None}}}
        self.assertEqual(controller_imu_lag_ns("left_controller", cfg), -5_000_000)
        self.assertEqual(controller_imu_lag_ns("right_controller", cfg), -7_000_000)

    def test_files_table_matches_lag_and_paths(self):
        cfg = {"controllers": {"left_controller": {"mocap_vision_offset_ns": 7_650_000},
                               "right_controller": {"mocap_vision_offset_ns": 7_600_000}}}
        self.assertEqual(controller_imu_files(cfg),
                         {"left_controller": ("imu1/data.csv", -7_650_000),
                          "right_controller": ("imu2/data.csv", -7_600_000)})

    def test_shipped_config_agrees_with_vision_offset(self):
        """The real config.yml: lag_ns must be exactly -mocap_vision_offset_ns for both controllers."""
        from src.load_config import load_yaml_config
        cfg = load_yaml_config("config/config.yml")
        for k in ("left_controller", "right_controller"):
            self.assertEqual(controller_imu_lag_ns(k), -int(load_vision_offset_ns(cfg["controllers"][k])))


if __name__ == "__main__":
    unittest.main()
