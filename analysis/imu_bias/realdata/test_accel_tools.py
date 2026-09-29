"""Exact-value tests of the windowed accel-bias measurement (sign, frame, gravity convention, telescoping)."""
import sys, types, unittest
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from accel_tools import *
from src.imu_data import _lever_arm_correction
from scipy.spatial.transform import Slerp

_RUN = Run("static_dark"); CTRL = "left_controller"
d0 = _RUN.ctrl[CTRL]
T0 = int(d0["vision"].ts[0]) + 3_000_000_000; DUR = 45.0


def synth(b_a, lever=np.array([0.0346, -0.0775, 0.0028]), pos_noise_mm=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t_imu = d0["t"][(d0["t"] >= T0 - 1_000_000_000) & (d0["t"] <= T0 + int(DUR * 1e9) + 1_000_000_000)]
    v_ts = d0["vision"].ts[(d0["vision"].ts >= T0) & (d0["vision"].ts <= T0 + int(DUR * 1e9))]
    t_fine = np.arange(T0 - 1_000_000_000, T0 + int(DUR * 1e9) + 1_000_001_000, 500_000, dtype=np.int64)
    s = (t_fine - t_fine[0]) / 1e9
    # world trajectory (m): sum of sinusoids; analytic velocity/acceleration
    A = np.array([[0.25, 0.12, 0.08], [0.10, 0.20, 0.05], [0.06, 0.04, 0.12]]); F = np.array([0.4, 0.9, 1.7]); Ph = np.array([[0, 1, 2], [0.5, 1.5, 2.5], [1, 0.2, 3]])
    p = lambda t: sum(A[k] * np.sin(2 * np.pi * F[k] * t[:, None] + Ph[k]) for k in range(3))
    v = lambda t: sum(A[k] * 2 * np.pi * F[k] * np.cos(2 * np.pi * F[k] * t[:, None] + Ph[k]) for k in range(3))
    a = lambda t: sum(-A[k] * (2 * np.pi * F[k]) ** 2 * np.sin(2 * np.pi * F[k] * t[:, None] + Ph[k]) for k in range(3))
    # body angular velocity (rad/s) and exact orientation on the fine grid
    wf = lambda t: np.stack([2.0 * np.sin(2 * np.pi * 0.31 * t + 0.1) + 0.7 * np.sin(2 * np.pi * 1.4 * t), 1.5 * np.sin(2 * np.pi * 0.83 * t + 1.3), 1.0 * np.sin(2 * np.pi * 0.5 * t + 2.1)], axis=1)
    R = [np.eye(3)]
    for k in range(len(s) - 1):
        R.append(R[-1] @ Rotation.from_rotvec(wf(np.array([0.5 * (s[k] + s[k + 1])]))[0] * (s[k + 1] - s[k])).as_matrix())
    R = np.array(R)
    sl = Slerp(t_fine.astype(float), Rotation.from_matrix(R))
    Rk_imu = sl(t_imu.astype(float)).as_matrix()
    s_imu = (t_imu - t_fine[0]) / 1e9
    gyro = wf(s_imu)
    lev = _lever_arm_correction(t_imu, t_imu, gyro, lever)
    f = np.einsum("nji,nj->ni", Rk_imu, a(s_imu) - G_ABS) + lev + b_a          # R^T (a - g) + lever + b
    Rv = sl(v_ts.astype(float)).as_matrix(); s_v = (v_ts - t_fine[0]) / 1e9
    Pv = p(s_v) + pos_noise_mm * 1e-3 * rng.standard_normal((len(v_ts), 3))
    run_like = types.SimpleNamespace(ctrl={CTRL: dict(t=t_imu, gyro=gyro, accel=f, lever=lever)})
    return run_like, AbsTrack(v_ts, Rv, Pv)


class T(unittest.TestCase):
    def test_static_gravity_sign(self):
        """R=I, a_w=0 -> f = -g = [0, +9.81, 0] (+ bias): the Q integral must vanish for the true bias, sign +."""
        t = np.arange(0, 20_000_000_000, 5_000_000, dtype=np.int64) + 10**12
        ts = np.arange(0, 20_000_000_000, 15_000_000, dtype=np.int64) + 10**12
        b = np.array([0.1, -0.05, 0.2]); lever = np.zeros(3)
        f = np.tile(-G_ABS + b, (len(t), 1)); gyro = np.zeros((len(t), 3))
        trk = AbsTrack(ts, np.tile(np.eye(3), (len(ts), 1, 1)), np.zeros((len(ts), 3)))
        run = types.SimpleNamespace(ctrl={"x": dict(t=t, gyro=gyro, accel=f, lever=lever)})
        W = AccelWindows(run, "x", trk)
        est, cov, nseg = W.measure(t[0] + 2_000_000_000, t[0] + 18_000_000_000)
        np.testing.assert_allclose(est, b, atol=1e-4)   # ridge=1e-3 -> 6e-5 relative shrink

    def test_general_motion_recovers_bias(self):
        b = np.array([0.15, -0.10, 0.08])
        run, trk = synth(b)
        W = AccelWindows(run, CTRL, trk)
        for (lo, hi) in ((3, 13), (5, 25), (8, 38)):
            est, cov, nseg = W.measure(T0 + int(lo * 1e9), T0 + int(hi * 1e9))
            print(f"   [accel synth, window {hi-lo}s] est {np.round(est,4)} true {b}  err {np.linalg.norm(est-b):.4f}")
            self.assertLess(np.linalg.norm(est - b), 0.02)

    def test_sign_flip_and_orientation_matter(self):
        """Wrong-orientation (R transposed) must NOT recover the bias -> proves the frame/sign logic is exercised."""
        b = np.array([0.15, -0.10, 0.08]); run, trk = synth(b)
        bad = AbsTrack(trk.ts, np.transpose(trk.R, (0, 2, 1)), trk.P)
        W = AccelWindows(run, CTRL, bad)
        est, cov, nseg = W.measure(T0 + int(5e9), T0 + int(35e9))
        self.assertGreater(np.linalg.norm(est - b), 0.3)

    def test_position_noise_scaling(self):
        """5 mm position noise -> window noise ~ velocity noise*sqrt2/T (telescoping); not 1/dt^2 blow-up."""
        b = np.array([0.15, -0.10, 0.08]); errs = []
        for seed in range(5):
            run, trk = synth(b, pos_noise_mm=5.0, seed=seed); W = AccelWindows(run, CTRL, trk)
            est, _, _ = W.measure(T0 + int(5e9), T0 + int(25e9)); errs.append(np.linalg.norm(est - b))
        print(f"   [accel synth, 5 mm pos noise, 20 s window] rms err {np.sqrt(np.mean(np.square(errs))):.4f} m/s^2")
        self.assertLess(np.mean(errs), 0.05)

if __name__ == "__main__":
    unittest.main(verbosity=2)
