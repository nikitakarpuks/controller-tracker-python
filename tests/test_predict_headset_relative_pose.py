"""Unit tests for src.imu_data.predict_headset_relative_pose -- the frame-conversion wrapper
that corrects controller dead-reckoning (predict_world_pose) for headset ego-motion. This
pipeline's own "world" frame is really the headset-IMU rig frame (Camera.T_world_cam is a fixed
rig extrinsic, see src/camera.py), not an inertial one, so blindly integrating a controller's own
gyro/accel against that frame silently assumes the headset itself is non-rotating/non-accelerating
during the prediction window -- false while walking (measured this session on walk_medium: mean
headset gyro 86 deg/s, median 56 deg/s, 72% of samples above 30 deg/s).

All fixtures are synthetic/analytic (no real recording data), matching this project's convention
for pure-math unit tests (see tests/test_pose_jump_rotation_convention.py). Uses stdlib unittest
(pytest is not a declared dependency of this project). Run with:
    python3 -m unittest tests.test_predict_headset_relative_pose
"""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.imu_data import predict_world_pose, predict_headset_relative_pose

_NS = 1_000_000_000
_TS0 = 0
_TS1 = int(0.02 * _NS)  # 20ms -- a typical single warm-tracking-frame gap in this pipeline
_G_ABS = np.array([0.0, 0.0, -9.81])  # arbitrary fixed absolute-frame gravity vector


def _const_stream(ts0: int, ts1: int, value: np.ndarray):
    """(t (2,), values (2,2)) constant-value gyro/accel stream spanning exactly [ts0, ts1] --
    with samples only at the two endpoints, integrate_gyro_segment/integrate_accel_to_position's
    own mid_mask selects no interior samples, so the resulting trapezoidal integration is over a
    single interval -- the simplest input that still satisfies their coverage checks."""
    t = np.array([ts0, ts1], dtype=np.int64)
    v = np.tile(np.asarray(value, dtype=np.float64), (2, 1))
    return t, v


class IdentityReductionTests(unittest.TestCase):
    """No headset ego-motion at all (R_wh=I, p_wh=0, omega_h0=v_wh0=0) must reduce EXACTLY to
    calling predict_world_pose directly -- the wrapper must be a true no-op in this case."""

    def test_reduces_exactly_to_predict_world_pose(self):
        R_hc0 = Rotation.from_euler('xyz', [10, -20, 30], degrees=True).as_matrix()
        p_hc0 = np.array([0.3, -0.1, 0.5])
        v_hc0 = np.array([0.1, 0.2, -0.05])
        t_gyro, gyro_body = _const_stream(_TS0, _TS1, [0.5, -0.3, 0.2])
        t_accel, accel_body = _const_stream(_TS0, _TS1, [0.2, 0.1, 9.9])

        I3, z3 = np.eye(3), np.zeros(3)
        corrected = predict_headset_relative_pose(
            t_gyro, gyro_body, t_accel, accel_body, _G_ABS, None,
            _TS0, _TS1, R_hc0, p_hc0, v_hc0,
            I3, z3, z3, z3, I3, z3,
        )
        direct = predict_world_pose(
            t_gyro, gyro_body, t_accel, accel_body, _G_ABS, None,
            _TS0, _TS1, R_hc0, p_hc0, v_hc0,
        )
        self.assertIsNotNone(corrected)
        self.assertIsNotNone(direct)
        np.testing.assert_allclose(corrected[0], direct[0], atol=1e-10)
        np.testing.assert_allclose(corrected[1], direct[1], atol=1e-10)


class GalileanInvarianceTests(unittest.TestCase):
    """A headset that translates at CONSTANT velocity without rotating is still, itself, an
    inertial frame -- the correction must also be a no-op here, but (unlike the identity case)
    the equivalent "uncorrected" comparison call must use gravity re-expressed in the headset's
    own (fixed, non-identity) orientation: g_world_headset = R_wh.T @ g_world_abs. Using the SAME
    g_world_abs unrotated in both calls would be wrong whenever R_wh != I -- this is exactly the
    frame-mismatch bug this feature's design review caught in an early draft, so this test fixes
    R_wh to a genuinely non-identity rotation specifically to make sure a future change can't
    silently reintroduce that mismatch and still pass."""

    def _check(self, v_wh0: np.ndarray):
        R_wh = Rotation.from_euler('xyz', [45, 15, -60], degrees=True).as_matrix()
        R_hc0 = Rotation.from_euler('xyz', [5, 40, -10], degrees=True).as_matrix()
        p_hc0 = np.array([0.4, 0.05, -0.2])
        v_hc0 = np.array([-0.1, 0.15, 0.05])
        t_gyro, gyro_body = _const_stream(_TS0, _TS1, [0.3, 0.4, -0.2])
        t_accel, accel_body = _const_stream(_TS0, _TS1, [0.1, -0.2, 9.8])

        T = (_TS1 - _TS0) / 1e9
        p_wh0 = np.array([1.0, 2.0, 0.5])
        p_wh1 = p_wh0 + v_wh0 * T
        omega_h0 = np.zeros(3)

        corrected = predict_headset_relative_pose(
            t_gyro, gyro_body, t_accel, accel_body, _G_ABS, None,
            _TS0, _TS1, R_hc0, p_hc0, v_hc0,
            R_wh, p_wh0, omega_h0, v_wh0, R_wh, p_wh1,
        )
        g_world_headset = R_wh.T @ _G_ABS
        direct = predict_world_pose(
            t_gyro, gyro_body, t_accel, accel_body, g_world_headset, None,
            _TS0, _TS1, R_hc0, p_hc0, v_hc0,
        )
        self.assertIsNotNone(corrected)
        self.assertIsNotNone(direct)
        np.testing.assert_allclose(corrected[0], direct[0], atol=1e-9)
        np.testing.assert_allclose(corrected[1], direct[1], atol=1e-9)

    def test_boost_a(self):
        self._check(np.array([0.5, -0.3, 0.1]))

    def test_boost_b(self):
        self._check(np.array([-1.2, 0.0, 0.8]))


class ProveTheBugProveTheFixTests(unittest.TestCase):
    """The decisive test: a headset in pure constant-rate YAW (rotation axis aligned with
    gravity -- the common "looking around while standing/walking upright" case) with a controller
    RIGIDLY CO-ROTATING (genuinely stationary relative to the headset: fixed R_hc/p_hc, v_hc0=0).

    Choosing the rotation axis aligned with gravity makes every quantity below exactly constant
    in body-frame coordinates (both the co-rotating point's own centripetal term and gravity's
    body-frame representation are invariant under rotation about their own shared axis -- derived
    and confirmed independently during this feature's design review), so gyro_body/accel_body can
    be hand-derived in closed form with no residual modeling slack:

        gyro_body(t)  = omega_h0                                              (co-aligned frames)
        accel_body(t) = omega_h0 x (omega_h0 x p_hc) - g_world_abs            (constant specific force)

    The corrected function must recover p_hc1 ~= p_hc0 (no spurious relative motion -- there
    genuinely isn't any). Calling predict_world_pose directly on the same inputs (what today's
    UNCORRECTED code actually does) must NOT recover that -- it has no way to know the headset is
    rotating, and manufactures spurious "relative" displacement purely from the missing headset
    term. This is the concrete reproduction of the real bug (a 100ms coasting gap at fast headset
    yaw can spuriously exceed this pipeline's own 30deg pose-jump-reject threshold with zero real
    controller motion, per this feature's own empirical analysis of walk_medium)."""

    def test_corrected_sees_no_motion_uncorrected_sees_spurious_motion(self):
        omega_h0 = np.array([0.0, 0.0, 2.0])  # ~115 deg/s yaw -- within this session's measured
                                               # real headset gyro range (mean 86, p90 205 deg/s)
        p_hc0 = np.array([0.4, 0.0, 0.0])     # ~40cm offset -- plausible arm's-reach lever arm
        R_hc0 = np.eye(3)                     # co-aligned with the headset (rigidly bolted)
        v_hc0 = np.zeros(3)

        gyro_val = omega_h0
        accel_val = np.cross(omega_h0, np.cross(omega_h0, p_hc0)) - _G_ABS
        t_gyro, gyro_body = _const_stream(_TS0, _TS1, gyro_val)
        t_accel, accel_body = _const_stream(_TS0, _TS1, accel_val)

        T = (_TS1 - _TS0) / 1e9
        R_wh0 = np.eye(3)
        R_wh1 = Rotation.from_rotvec(omega_h0 * T).as_matrix()
        p_wh0 = p_wh1 = np.zeros(3)
        v_wh0 = np.zeros(3)

        corrected = predict_headset_relative_pose(
            t_gyro, gyro_body, t_accel, accel_body, _G_ABS, None,
            _TS0, _TS1, R_hc0, p_hc0, v_hc0,
            R_wh0, p_wh0, omega_h0, v_wh0, R_wh1, p_wh1,
        )
        self.assertIsNotNone(corrected)
        R_hc1_corrected, p_hc1_corrected = corrected
        corrected_err = np.linalg.norm(p_hc1_corrected - p_hc0)

        # g_world_headset0 = R_wh0.T @ _G_ABS = _G_ABS here since R_wh0 = I -- what today's real,
        # unmodified code path would use at this exact instant.
        uncorrected = predict_world_pose(
            t_gyro, gyro_body, t_accel, accel_body, _G_ABS, None,
            _TS0, _TS1, R_hc0, p_hc0, v_hc0,
        )
        self.assertIsNotNone(uncorrected)
        _, p_hc1_uncorrected = uncorrected
        uncorrected_err = np.linalg.norm(p_hc1_uncorrected - p_hc0)

        # Corrected: no spurious relative motion, to a tolerance consistent with the trapezoidal
        # scheme's discretization error over a small (~0.04 rad) rotation angle across 20ms.
        self.assertLess(corrected_err, 1e-4, f"corrected path shows spurious motion: {corrected_err} m")
        # Uncorrected: real, measurably wrong drift -- at least 2 orders of magnitude worse than
        # the corrected path's residual discretization noise, not just "some nonzero drift".
        self.assertGreater(uncorrected_err, 100 * corrected_err,
                            f"uncorrected err {uncorrected_err} not decisively worse than corrected {corrected_err}")
        # Direction check: the spurious displacement should point INWARD (the uncorrected path
        # integrates the co-rotating point's own centripetal-reaction specific force as if it
        # were real relative acceleration, which pulls p_hc1 back toward the rotation axis/origin,
        # i.e. opposite p_hc0's own outward direction).
        outward = p_hc0 / np.linalg.norm(p_hc0)
        delta = p_hc1_uncorrected - p_hc0
        self.assertLess(np.dot(delta, outward), 0.0,
                         "expected the uncorrected spurious displacement to point inward")


class GeneralRoundTripTests(unittest.TestCase):
    """General case: headset AND controller each undergo independent constant angular velocity
    (about the shared gravity axis, for the same exactness reasons as above) plus constant linear
    velocity, entirely decoupled from each other. Ground truth is derived by evaluating the
    controller's own closed-form ABSOLUTE trajectory directly (not via the function under test),
    then projecting through the headset's own closed-form trajectory -- exercising the real
    integration in predict_world_pose (not just the lift/project algebra already covered above)
    against an independently-derived, exact answer."""

    def test_independent_headset_and_controller_motion(self):
        omega_h0 = np.array([0.0, 0.0, -1.5])   # headset yawing one way
        omega_c_world = np.array([0.0, 0.0, 3.0])  # controller spinning the other way, faster
        v_wh0 = np.array([0.6, -0.2, 0.0])
        v_wc_world = np.array([-0.3, 0.4, 0.1])
        p_wh0 = np.array([2.0, -1.0, 1.6])
        R_wh0 = Rotation.from_euler('xyz', [0, 0, 20], degrees=True).as_matrix()
        p_wc0 = np.array([2.3, -0.9, 1.5])
        R_wc0 = Rotation.from_euler('xyz', [0, 0, -50], degrees=True).as_matrix()

        T = (_TS1 - _TS0) / 1e9
        R_wh1 = Rotation.from_rotvec(omega_h0 * T).as_matrix() @ R_wh0
        p_wh1 = p_wh0 + v_wh0 * T
        R_wc1 = Rotation.from_rotvec(omega_c_world * T).as_matrix() @ R_wc0
        p_wc1 = p_wc0 + v_wc_world * T

        # Headset-relative state at t0 (what the vision pipeline would have measured/tracked).
        R_hc0 = R_wh0.T @ R_wc0
        p_hc0 = R_wh0.T @ (p_wc0 - p_wh0)
        v_hc0 = R_wh0.T @ (v_wc_world - v_wh0) - np.cross(omega_h0, p_hc0)

        # Ground truth at t1, via the SAME projection, independent of the function under test.
        R_hc1_true = R_wh1.T @ R_wc1
        p_hc1_true = R_wh1.T @ (p_wc1 - p_wh1)

        # Controller's own gyro/accel: rotation is about the (gravity-aligned) z-axis in world
        # frame, so body-frame gyro is exactly constant and equal to omega_c_world (see
        # ProveTheBugProveTheFixTests' docstring for the underlying invariance); translating at
        # constant velocity means zero true world acceleration, so accel_body is just -g rotated
        # into the (also z-axis-rotating) body frame, which is likewise exactly constant.
        gyro_val = omega_c_world
        accel_val = -_G_ABS
        t_gyro, gyro_body = _const_stream(_TS0, _TS1, gyro_val)
        t_accel, accel_body = _const_stream(_TS0, _TS1, accel_val)

        predicted = predict_headset_relative_pose(
            t_gyro, gyro_body, t_accel, accel_body, _G_ABS, None,
            _TS0, _TS1, R_hc0, p_hc0, v_hc0,
            R_wh0, p_wh0, omega_h0, v_wh0, R_wh1, p_wh1,
        )
        self.assertIsNotNone(predicted)
        R_hc1_pred, p_hc1_pred = predicted
        np.testing.assert_allclose(p_hc1_pred, p_hc1_true, atol=1e-9)
        np.testing.assert_allclose(R_hc1_pred, R_hc1_true, atol=1e-9)


class NonePropagationTests(unittest.TestCase):
    """If the wrapped predict_world_pose call can't produce a result (gyro/accel coverage doesn't
    span [ts0, ts1]), the wrapper must propagate None, not raise or fabricate a partial answer."""

    def test_none_when_underlying_integration_fails(self):
        I3, z3 = np.eye(3), np.zeros(3)
        # accel coverage stops short of ts1 -- integrate_accel_to_position's own coverage check
        # (ts1 > t_accel[-1]) must fail, and that None must propagate through unchanged.
        t_gyro, gyro_body = _const_stream(_TS0, _TS1, [0.1, 0.0, 0.0])
        short_ts1 = _TS1 - int(0.005 * _NS)
        t_accel, accel_body = _const_stream(_TS0, short_ts1, [0.0, 0.0, 9.8])

        result = predict_headset_relative_pose(
            t_gyro, gyro_body, t_accel, accel_body, _G_ABS, None,
            _TS0, _TS1, I3, z3, z3,
            I3, z3, z3, z3, I3, z3,
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
