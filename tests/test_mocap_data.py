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

from src.mocap_data import DeviceMocap, headset_angular_velocity, headset_linear_velocity
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


if __name__ == "__main__":
    unittest.main()
