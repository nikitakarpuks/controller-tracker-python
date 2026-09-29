"""Exact-value unit tests for the residual -> bias-measurement mapping and the estimators (sign, frame, causality).
Synthetic data is built on the REAL static_dark timestamps (controller IMU stamps + vision stamps), so cadence,
alternating 11/22 ms gaps and IMU sampling are realistic, but the truth is known exactly.

Run:  python3 analysis/imu_bias/realdata/test_estimators.py
"""
import sys, types, unittest
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from common import *
from estimators import *
from pipeline import make_steps, run_estimator

_RUN = Run("static_dark")
CTRL = "left_controller"
_T0 = int(_RUN.ctrl[CTRL]["vision"].ts[0]) + 3_000_000_000
_DUR_S = 30.0


def _omega(t_s, amp, freqs, phases):
    """smooth body-frame angular velocity (rad/s), (N,3)."""
    t_s = np.atleast_1d(t_s)[:, None]
    return sum(amp * np.sin(2 * np.pi * f * t_s + p) for f, p in zip(freqs, phases))


def _integrate_rotation(t_fine_s, w_fn):
    """Exact-ish body-frame rotation trajectory: R(t_{k+1}) = R(t_k) Exp(w(mid) dt) on a fine grid."""
    R = [np.eye(3)]
    for k in range(len(t_fine_s) - 1):
        dt = t_fine_s[k + 1] - t_fine_s[k]
        R.append(R[-1] @ Rotation.from_rotvec(w_fn(0.5 * (t_fine_s[k] + t_fine_s[k + 1]))[0] * dt).as_matrix())
    return np.array(R)


def synth(b_true, w_amp=np.array([2.0, 1.5, 1.0]), headset_amp=0.0, seed=0, vis_noise_deg=0.0, zero_motion=False):
    """Returns (run_like, prep, truth). Controller gyro = w_true(t) + b_true sampled at REAL IMU stamps; vision frames at
    REAL vision stamps carry R_hc = R_wh^T R_wc (headset rotating if headset_amp>0)."""
    rng = np.random.default_rng(seed)
    d = _RUN.ctrl[CTRL]
    t_imu = d["t"][(d["t"] >= _T0 - 1_000_000_000) & (d["t"] <= _T0 + int(_DUR_S * 1e9) + 1_000_000_000)]
    v_ts = d["vision"].ts[(d["vision"].ts >= _T0) & (d["vision"].ts <= _T0 + int(_DUR_S * 1e9))]
    f_c = [0.31, 0.83, 1.7]; p_c = [0.0, 1.3, 2.1]
    f_h = [0.17, 0.5]; p_h = [0.4, 2.2]
    wc = (lambda t: np.zeros((np.atleast_1d(t).size, 3))) if zero_motion else (lambda t: _omega(t, w_amp / 2, f_c[:2], p_c[:2]) + _omega(t, w_amp / 4, f_c[2:], p_c[2:]))
    wh = lambda t: _omega(t, np.array([1.0, 0.7, 0.4]) * headset_amp, f_h, p_h)
    t_fine = np.arange(_T0 - 1_000_000_000, _T0 + int(_DUR_S * 1e9) + 1_000_000_001, 500_000, dtype=np.int64)
    tf_s = (t_fine - t_fine[0]) / 1e9
    off = lambda w: (lambda t: w(t + 0.0))
    Rc = _integrate_rotation(tf_s, lambda t: wc(t))
    Rh = _integrate_rotation(tf_s, lambda t: wh(t))
    interp = lambda R, t_ns: np.array([Rotation.from_matrix(R[np.searchsorted(t_fine, t)]).as_matrix() for t in t_ns])
    # vision frames exactly on the fine grid? stamps are 1 ns resolution: snap to nearest 0.5 ms sample (Slerp would
    # also do; at 0.5 ms and <3 rad/s the snap error is <0.09 deg -- too big for exact tests, so SLERP instead)
    from scipy.spatial.transform import Slerp
    slc = Slerp(t_fine.astype(float), Rotation.from_matrix(Rc)); slh = Slerp(t_fine.astype(float), Rotation.from_matrix(Rh))
    R_wc = slc(v_ts.astype(float)).as_matrix()
    R_wh = slh(v_ts.astype(float)).as_matrix()
    noise = np.array([Rotation.from_rotvec(np.radians(vis_noise_deg) * rng.standard_normal(3) / np.sqrt(3)).as_matrix()
                      for _ in v_ts]) if vis_noise_deg > 0 else np.array([np.eye(3)] * len(v_ts))
    R_wc_meas = np.einsum("nij,njk->nik", R_wc, noise)            # right-perturbation = body-frame noise
    gyro = wc((t_imu - t_fine[0]) / 1e9) + b_true
    run_like = types.SimpleNamespace(ctrl={CTRL: dict(t=t_imu, gyro=gyro)})
    R_hc = np.einsum("nji,njk->nik", R_wh, R_wc_meas)             # R_wh^T R_wc_meas
    prep = dict(ts=v_ts, R_wc=R_wc_meas, R_hc=R_hc, R_wh=R_wh, ok=np.ones(len(v_ts), bool), ctrl=CTRL)
    return run_like, prep, dict(R_wc_true=R_wc, R_wh=R_wh)


class TestResidualMapping(unittest.TestCase):
    def test_exact_zero_motion_sign_and_value(self):
        b = np.array([0.011, -0.007, 0.004])
        run, prep, _ = synth(b, zero_motion=True)
        steps, _ = make_steps(run, CTRL, prep)
        for s in steps[:200]:
            np.testing.assert_allclose(s["e0"], -b * s["dt"], atol=1e-12)       # e0 = -b dt EXACTLY, sign negative
        self.assertGreater(len(steps), 500)

    def test_exact_axis_parallel_motion(self):
        b = np.array([0.0, 0.02, 0.0])
        run, prep, _ = synth(b, w_amp=np.array([0.0, 2.0, 0.0]))
        steps, _ = make_steps(run, CTRL, prep)
        err = [np.linalg.norm(s["e0"] + b * s["dt"]) / (np.linalg.norm(b) * s["dt"]) for s in steps]
        self.assertLess(np.median(err), 0.01)                                     # same-axis rotations commute -> ~exact

    def test_general_motion_first_order(self):
        b = np.array([0.01, -0.008, 0.006])
        run, prep, _ = synth(b)
        steps, _ = make_steps(run, CTRL, prep)
        rel = [np.linalg.norm(s["e0"] + b * s["dt"]) / (np.linalg.norm(b) * s["dt"]) for s in steps]
        self.assertLess(np.median(rel), 0.03)                                     # noncommuting terms ~ |w| dt ~ 3%
        print(f"   [general motion] median relative deviation of e0 from -b dt: {np.median(rel):.4f}")

    def test_first_order_feedback_equivalence(self):
        """e(b) = e0 + b dt to second order: compare with an exact re-integration using bias b."""
        b_true = np.array([0.01, -0.008, 0.006]); b_apply = np.array([0.004, 0.002, -0.003])
        run, prep, _ = synth(b_true)
        d = run.ctrl[CTRL]
        worst = 0.0
        for a in range(0, 600, 37):
            ta, tb = int(prep["ts"][a]), int(prep["ts"][a + 1])
            dt = (tb - ta) / 1e9
            e0 = Rotation.from_matrix(integrate_gyro_segment(d["t"], d["gyro"], ta, tb).T @ (prep["R_wc"][a].T @ prep["R_wc"][a + 1])).as_rotvec()
            eb = Rotation.from_matrix(integrate_gyro_segment(d["t"], d["gyro"] - b_apply, ta, tb).T @ (prep["R_wc"][a].T @ prep["R_wc"][a + 1])).as_rotvec()
            worst = max(worst, np.linalg.norm(eb - (e0 + b_apply * dt)))
        self.assertLess(worst, 5e-6)
        print(f"   [first-order feedback] worst |e(b) - (e0 + b dt)| = {worst:.2e} rad")


class TestEstimatorsNoiseFree(unittest.TestCase):
    b = np.array([0.012, -0.009, 0.006])

    def _run(self, est, **kw):
        run, prep, _ = synth(self.b, **kw)
        steps, _ = make_steps(run, CTRL, prep)
        T, B = run_estimator(est, steps)
        return B

    def test_all_estimators_recover_bias_sign_and_value(self):
        for est in (FeedbackEMA(tau_s=3.0), RatioEMA(tau_s=3.0), WorldRLS(tau_s=8.0), WindowMedian(5.0)):
            B = self._run(est)
            final = B[-1]
            rel = np.linalg.norm(final - self.b) / np.linalg.norm(self.b)
            print(f"   [noise-free, static headset] {est.name}: final b = {np.round(final, 5)} (true {self.b})  rel err {rel:.3f}")
            self.assertLess(rel, 0.30 if isinstance(est, WorldRLS) else 0.08, est.name)   # WorldRLS is prior-shrunk (mu) at short tau
            self.assertGreater(final @ self.b, 0)                                  # sign preserved

    def test_rotating_headset_needs_ego_removal(self):
        """Frame test. Headset rotates (0.6 rad/s scale). With ego-motion removed (R_wc = R_wh R_hc) bias is recovered;
        pretending the headset is static (using rig-frame R_hc as the world orientation) is measurably wrong."""
        run, prep, _ = synth(self.b, headset_amp=1.0)
        steps_ok, _ = make_steps(run, CTRL, prep)                                    # ego removed (prep['R_wc'] is inertial)
        prep_bad = dict(prep); prep_bad["R_wc"] = prep["R_hc"]                       # rig-frame orientation: WRONG
        steps_bad, _ = make_steps(run, CTRL, prep_bad)
        b_ok = run_estimator(WorldRLS(tau_s=5.0), steps_ok)[1][-1]
        b_bad = run_estimator(WorldRLS(tau_s=5.0), steps_bad)[1][-1]
        e_ok = np.linalg.norm(b_ok - self.b); e_bad = np.linalg.norm(b_bad - self.b)
        print(f"   [rotating headset] error with ego removed {e_ok:.5f} rad/s, without {e_bad:.5f} rad/s")
        self.assertLess(e_ok, 0.25 * np.linalg.norm(self.b))
        self.assertGreater(e_bad, 5 * e_ok)

    def test_causality_bit_exact(self):
        """Corrupting every frame after K must not change any estimate up to K."""
        run, prep, _ = synth(self.b)
        steps, _ = make_steps(run, CTRL, prep)
        K = 800
        steps2 = [dict(s) for s in steps]
        for s in steps2[K + 1:]:
            s["e0"] = s["e0"] + np.array([0.3, -0.2, 0.1])
        for mk in (lambda: FeedbackEMA(3.0), lambda: RatioEMA(3.0), lambda: WorldRLS(3.0), lambda: WindowMedian(5.0)):
            B1 = run_estimator(mk(), steps)[1]; B2 = run_estimator(mk(), steps2)[1]
            np.testing.assert_array_equal(B1[:K + 1], B2[:K + 1])
            self.assertFalse(np.array_equal(B1[K + 1:], B2[K + 1:]))


class TestEstimatorsNoisy(unittest.TestCase):
    def test_noise_scaling_matches_telescoping_theory(self):
        """With 0.3 deg white vision noise the world-frame estimator's steady-state error should be ~ sigma_rot/tau
        (endpoint noise only), NOT sigma/(dt*sqrt(N)). Report and check the order of magnitude."""
        b = np.array([0.01, -0.008, 0.006]); tau = 8.0
        errs = []
        for seed in range(6):
            run, prep, _ = synth(b, vis_noise_deg=0.3, seed=seed)
            steps, _ = make_steps(run, CTRL, prep)
            T, B = run_estimator(WorldRLS(tau_s=tau), steps)
            tail = B[T > T[0] + int(3 * tau * 1e9)]
            errs.append(np.sqrt(np.mean(np.sum((tail - b) ** 2, axis=1))))
        sigma_rad = np.radians(0.3)
        predicted = sigma_rad / tau * np.sqrt(3) * 0.5     # order-of-magnitude endpoint-noise estimate
        print(f"   [noise 0.3deg, tau {tau}s] worldRLS rms bias error {np.mean(errs):.5f} rad/s   (order-of-magnitude theory ~{predicted:.5f}; "
              f"naive per-frame 1/sqrt(N) theory would be ~{sigma_rad/0.0112/np.sqrt(30/0.0112):.5f})")
        self.assertLess(np.mean(errs), 0.006)


if __name__ == "__main__":
    unittest.main(verbosity=2)
