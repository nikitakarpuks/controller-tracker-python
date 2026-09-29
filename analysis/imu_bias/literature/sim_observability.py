#!/usr/bin/env python3
"""sim_observability.py -- numerical check of the closed-form accuracy claims in REPORT.md.

Everything here is a synthetic Monte-Carlo with the noise numbers quoted in the report (vision
orientation sigma, vision position sigma, gyro/accel noise densities, frame cadence alternating
11.1/22.2 ms as in the project's controller-camera stream). It does NOT touch project data.

Checks:
  A. Gyro-bias from vision orientation (1 axis, error-state KF: state [d_theta, b_g]):
     A1 constant bias, iid vision noise: KF/LS-slope std vs the closed form sqrt(12)*sig*sqrt(dt)/T^1.5
     A2 same but vision orientation error correlated (AR(1), tau_c = 1 s)
     A3 drifting bias (random walk q): steady-state std vs Kalman-Bucy closed form 2^(1/4) q^(3/4) R^(1/8)
     A4 the naive estimator "unweighted mean of per-frame residuals e_k/dt_k" vs the correct
        time-weighted one sum(e_k)/sum(dt_k), with alternating frame intervals and iid noise
     A5 robustness: one 5 deg vision outlier per 2 s
  B. Accel-bias from vision position (1 axis, KF state [d_p, d_v, b_a]): std vs 26.8*sig_p*sqrt(dt)/T^2.5
  C. Gravity leakage table: apparent accel bias = g * sin(orientation error)
"""
import numpy as np

rng = np.random.default_rng(1)
DTS = np.array([1 / 90.0, 2 / 90.0])          # alternating 11.1 ms / 22.2 ms (mean 16.7 ms)
DT_MEAN = DTS.mean()


def frame_times(T):
    t, out, i = 0.0, [0.0], 0
    while t < T:
        t += DTS[i % 2]; out.append(t); i += 1
    return np.array(out)


def ar1_noise(n, sigma, tau_c, dts):
    """AR(1)/OU noise with stationary std sigma and correlation time tau_c sampled at irregular dts."""
    x = np.empty(n); x[0] = rng.normal(0, sigma)
    for k in range(1, n):
        a = np.exp(-dts[k - 1] / tau_c)
        x[k] = a * x[k - 1] + np.sqrt(1 - a * a) * sigma * rng.normal()
    return x


def kf_gyro_bias(t, z, q_b, sig_phi, n_g=4.8e-5, P0=(1e-2, 3e-2)):
    """error-state KF, state [d_theta, b], measurement z = d_theta + v. Returns b_hat trajectory."""
    x = np.zeros(2); P = np.diag([P0[0] ** 2, P0[1] ** 2]); bh = np.empty(len(t))
    R = sig_phi ** 2
    bh[0] = 0.0
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        F = np.array([[1, dt], [0, 1]])
        Q = np.diag([n_g ** 2 * dt, q_b ** 2 * dt])
        x = F @ x; P = F @ P @ F.T + Q
        S = P[0, 0] + R; K = P[:, 0] / S
        x = x + K * (z[k] - x[0]); P = P - np.outer(K, P[0])
        bh[k] = x[1]
    return bh


def run_A(sig_phi_deg=0.3, T=120.0, n_mc=20):
    sig = np.radians(sig_phi_deg)
    t = frame_times(T); dts = np.diff(t); n = len(t)
    print(f"\n== A. gyro bias from vision orientation (sigma_phi={sig_phi_deg} deg = {sig:.2e} rad, "
          f"{n} frames over {T:.0f}s, mean dt {DT_MEAN * 1e3:.1f} ms)")

    # A1/A2: constant true bias 0.01 rad/s, filter q tiny (=nearly constant model); std of b_hat at T_eval
    for label, tau_c in (("A1 iid", None), ("A2 corr tau_c=1s", 1.0)):
        errs = {W: [] for W in (1, 3, 10, 30, 100)}
        for _ in range(n_mc):
            v = rng.normal(0, sig, n) if tau_c is None else ar1_noise(n, sig, tau_c, dts)
            # true d_theta grows linearly with the constant bias (b*t); gyro white noise adds a random walk
            b_true = 0.01
            dth = b_true * t + np.concatenate([[0], np.cumsum(rng.normal(0, 4.8e-5 * np.sqrt(dts)))])
            z = dth - v * (-1)  # vision is theta_true + v, gyro-integrated is theta_true + dth -> z = dth - v; sign irrelevant for std
            z = dth + v
            bh = kf_gyro_bias(t, z, q_b=1e-7, sig_phi=sig)
            for W in errs:
                k = np.searchsorted(t, W)
                if k < n: errs[W].append(bh[k] - b_true)
        print(f"  {label}: std(b_hat - b) after W seconds (KF, q=1e-7):")
        for W, e in errs.items():
            cf = np.sqrt(12) * sig * np.sqrt(DT_MEAN) / W ** 1.5
            print(f"     W={W:>4}s  MC {np.std(e):.2e} rad/s   closed-form iid LS-slope {cf:.2e}")

    # A3: drifting bias with random walk q, steady-state error
    for q in (1e-5, 1e-4, 1e-3):
        errs = []
        for _ in range(n_mc):
            b = 0.005 + np.concatenate([[0], np.cumsum(rng.normal(0, q * np.sqrt(dts)))])
            dth = np.concatenate([[0], np.cumsum(0.5 * (b[1:] + b[:-1]) * dts)])
            v = rng.normal(0, sig, n)
            bh = kf_gyro_bias(t, dth + v, q_b=q, sig_phi=sig)
            errs.append(bh[n // 2:] - b[n // 2:])
        Rc = sig ** 2 * DT_MEAN
        closed = 2 ** 0.25 * q ** 0.75 * Rc ** 0.125
        wn = (q ** 2 / Rc) ** 0.25
        print(f"  A3 drifting bias q={q:.0e} rad/s/sqrt(s): MC steady-state std {np.std(np.concatenate(errs)):.2e} rad/s | "
              f"Kalman-Bucy closed form {closed:.2e} | natural time-const 1/wn = {1 / wn:.1f} s")

    # A4 naive unweighted mean of per-frame residual r_k = e_k/dt_k vs time-weighted, constant bias, iid noise
    naive, correct = [], []
    for _ in range(200):
        Tn = 30.0
        tt = frame_times(Tn); d = np.diff(tt)
        v = rng.normal(0, sig, len(tt))
        th_err = 0.01 * tt + v                         # d_theta(t) + vision noise
        e = np.diff(th_err)                            # per-frame residual rotation (vision noise telescopes)
        naive.append(np.mean(e / d) - 0.01)
        correct.append(np.sum(e) / np.sum(d) - 0.01)
    print(f"  A4 30s window, alternating dt: std of naive mean(e_k/dt_k) = {np.std(naive):.2e} rad/s "
          f"vs time-weighted sum(e)/sum(dt) = {np.std(correct):.2e} rad/s  (ratio {np.std(naive) / np.std(correct):.0f}x)")

    # A5 outliers: one 5 deg vision error every 2 s; naive per-frame EMA vs KF with 3-sigma gate
    errs_nogate, errs_gate = [], []
    for _ in range(n_mc):
        v = rng.normal(0, sig, n)
        k_out = np.arange(int(2.0 / DT_MEAN), n, int(2.0 / DT_MEAN))
        v[k_out] += np.radians(5.0) * rng.choice([-1, 1], len(k_out))
        dth = 0.01 * t
        bh = kf_gyro_bias(t, dth + v, q_b=1e-5, sig_phi=sig)
        errs_nogate.append(bh[n // 2:] - 0.01)
        # gated: drop measurements whose innovation vs 3-sigma is violated (crude: |z - (dth prediction)| via residual to running fit)
        keep = np.abs(v) < 4 * sig
        t2, z2 = t[keep], (dth + v)[keep]
        bh2 = kf_gyro_bias(t2, z2, q_b=1e-5, sig_phi=sig)
        errs_gate.append(bh2[len(t2) // 2:] - 0.01)
    print(f"  A5 5deg outlier every 2 s: KF no gate std {np.std(np.concatenate(errs_nogate)):.2e} rad/s | "
          f"with |v|<4sigma gate {np.std(np.concatenate(errs_gate)):.2e} rad/s")


def run_B(sig_p=4e-3, n_mc=20):
    """Accel bias from vision position. Effective window T = W + 5 s warm start (KF has seen T seconds).
    v_walk = accel white-noise density (velocity random walk, m/s/sqrt(s)): factory-derived 5.4e-4
    (7.6e-3 per-sample @200 Hz -> *sqrt(dt)) vs Basalt's default 1.6e-2 (deliberately inflated)."""
    print(f"\n== B. accel bias from vision position (sigma_p={sig_p * 1e3:.0f} mm, PERFECT orientation, true b_a=0.1 m/s^2)")
    for v_walk, tag in ((5.4e-4, "factory-derived velocity walk 5.4e-4"), (1.6e-2, "Basalt default velocity walk 1.6e-2")):
        print(f"  -- {tag}")
        for Wtot in (3, 6, 10, 15, 35):
            errs = []
            for _ in range(n_mc):
                t = frame_times(Wtot); n = len(t)
                b_true = 0.1
                z = 0.5 * b_true * t ** 2 + rng.normal(0, sig_p, n)
                x = np.zeros(3); P = np.diag([1e-2, 1e-1, 0.5 ** 2]); bh = np.empty(n); bh[0] = 0
                for k in range(1, n):
                    dt = t[k] - t[k - 1]
                    F = np.array([[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]])
                    Q = np.diag([0, v_walk ** 2 * dt, (1e-6) ** 2 * dt])
                    x = F @ x; P = F @ P @ F.T + Q
                    S = P[0, 0] + sig_p ** 2; K = P[:, 0] / S
                    x = x + K * (z[k] - x[0]); P = P - np.outer(K, P[0])
                    bh[k] = x[2]
                errs.append(bh[-1] - b_true)
            cf = 26.8 * sig_p * np.sqrt(DT_MEAN) / Wtot ** 2.5
            print(f"     data span T={Wtot:>3}s: MC std(b_a_hat - b_a) {np.std(errs):.2e} m/s^2 | closed form (iid position, no velocity noise) {cf:.2e}")


def run_C():
    g = 9.80665
    print("\n== C. gravity leakage: apparent accel bias = g*sin(orientation error)")
    for deg in (0.1, 0.3, 0.5, 1.0, 2.0, 5.0):
        print(f"  orientation error {deg:>4} deg -> {g * np.sin(np.radians(deg)):.3f} m/s^2")
    print("  factory-uncertainty for reference: accel 0.01 m/s^2 (=0.06 deg of tilt), gyro 1e-4 rad/s")


if __name__ == "__main__":
    run_A()
    run_B()
    run_C()
