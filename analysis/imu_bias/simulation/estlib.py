"""estlib.py -- causal bias estimators, all consuming a simlib.SimData (or real data adapted to it).

  (A) frame-level residual -> bias measurement -> running average (the mentor's scheme), with
      variants: naive per-frame EMA, dt-weighted (telescoping) exponential sum, windowed median.
  (B) error-state Kalman filter (15 states: dtheta, dv, dp, b_g, b_a) with vision pose updates.
  (C) sliding-window generalised least squares (gyro: window-constant b_g with nuisance initial
      attitude; accel: window-constant b_a with nuisance p0, v0).

Bias convention: sensor = truth + b  =>  corrected = sensor - b_est.  b_g in rad/s, b_a in m/s^2, body frame.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from simlib import Exp, Log, skew

G_MAX_GAP_S = 3.0


# ------------------------------------------------------------------ gyro interval (+ Jacobian)
def _interp3(t, x, q):
    return np.stack([np.interp(q, t, x[:, i]) for i in range(3)], axis=1)


def gyro_prefix(t_ns, gyro, ts0, ts1, b=None):
    """Same sample construction & midpoint rule as src.imu_data.integrate_gyro_segment, but returns
    all prefix rotations. times (n+1,) ns float, P (n+1,3,3) with P[0]=I, dt (n,), a=(w_mid*dt) (n,3)."""
    i0 = np.searchsorted(t_ns, ts0, "right"); i1 = np.searchsorted(t_ns, ts1, "left")
    times = np.concatenate(([ts0], t_ns[i0:i1], [ts1])).astype(np.float64)
    g = gyro if b is None else gyro - b
    om = _interp3(t_ns.astype(np.float64), g, times)
    dt = np.diff(times) / 1e9
    a = 0.5 * (om[:-1] + om[1:]) * dt[:, None]
    Rs = Rot.from_rotvec(a).as_matrix()
    P = np.empty((len(times), 3, 3)); P[0] = np.eye(3)
    for i in range(len(Rs)):
        P[i + 1] = P[i] @ Rs[i]
    return times, P, dt, a


def _jr(a):
    """first-order right Jacobian of Exp at a (n,3): I - 0.5[a]x + (1/6)[a]x^2"""
    n = len(a); S = np.zeros((n, 3, 3))
    S[:, 0, 1] = -a[:, 2]; S[:, 0, 2] = a[:, 1]; S[:, 1, 0] = a[:, 2]; S[:, 1, 2] = -a[:, 0]
    S[:, 2, 0] = -a[:, 1]; S[:, 2, 1] = a[:, 0]
    return np.eye(3) - 0.5 * S + (1 / 6.0) * np.einsum("nij,njk->nik", S, S)


def gyro_interval(t_ns, gyro, ts0, ts1):
    """(dR0, J): dR0 = bias-free integrated rotation, and dR(b) ~= dR0 Exp(-J b) to first order
    in b (J ~= dt*I for slow rotation)."""
    times, P, dt, a = gyro_prefix(t_ns, gyro, ts0, ts1)
    Pn = P[-1]
    Jr = _jr(a)
    inner = np.einsum("n,nij,njk->ik", dt, P[1:], Jr) if False else np.einsum("n,nij,njk->ik", dt, P[1:], Jr)
    return Pn, Pn.T @ inner


class Intervals:
    """Per consecutive-node gyro intervals, computed once and shared by the estimators."""
    def __init__(self, d, max_gap=G_MAX_GAP_S):
        K = len(d.t_v_ns)
        self.dt = np.zeros(K); self.dR = [None] * K; self.J = [None] * K; self.ok = np.zeros(K, bool)
        for k in range(1, K):
            dt = (d.t_v_ns[k] - d.t_v_ns[k - 1]) / 1e9
            self.dt[k] = dt
            if dt <= 0 or dt > max_gap or d.t_v_ns[k - 1] < d.t_imu_ns[0] or d.t_v_ns[k] > d.t_imu_ns[-1]:
                continue
            self.dR[k], self.J[k] = gyro_interval(d.t_imu_ns, d.gyro, d.t_v_ns[k - 1], d.t_v_ns[k])
            self.ok[k] = True


# ------------------------------------------------------------------ (A) frame-level running average
def run_A_gyro(d, iv, mode="dt_weighted", tau=20.0, gate=0.12):
    """Returns b_est (K,3) valid AFTER processing node k. mode: naive_ema | dt_weighted | median."""
    K = len(d.t_v_ns)
    out = np.zeros((K, 3)); b = np.zeros(3)
    S_r = np.zeros(3); S_J = np.zeros((3, 3)); hist = []
    tv = d.t_v_s
    for k in range(1, K):
        if iv.ok[k]:
            r0 = Log((d.R_wc_meas[k - 1] @ iv.dR[k]).T @ d.R_wc_meas[k])
            if np.linalg.norm(r0) < gate:
                dt = iv.dt[k]
                if mode == "naive_ema":                       # literal: per-frame measurement, EMA gain dt/tau
                    b_meas = -np.linalg.solve(iv.J[k], r0)
                    b = b + min(1.0, dt / tau) * (b_meas - b)
                elif mode == "dt_weighted":                   # exponential sum of residuals / sum of J: telescopes
                    lam = np.exp(-dt / tau)
                    S_r = lam * S_r + r0; S_J = lam * S_J + iv.J[k]
                    b = -np.linalg.solve(S_J, S_r)
                elif mode == "median":
                    hist.append((tv[k], -np.linalg.solve(iv.J[k], r0)))
                    while hist and tv[k] - hist[0][0] > tau: hist.pop(0)
                    b = np.median(np.array([h[1] for h in hist]), axis=0)
                else:
                    raise ValueError(mode)
        out[k] = b
    return out


def _lever_terms(t_s, gyro, r):
    """omega x (omega x r) + alpha x r at IMU samples (alpha: 3-sample central difference of gyro,
    bias-independent). Same construction as src.imu_data._lever_arm_correction but on the IMU grid."""
    dt = np.diff(t_s)
    al = np.zeros_like(gyro)
    al[1:-1] = (gyro[2:] - gyro[:-2]) / (t_s[2:] - t_s[:-2])[:, None]
    al[0] = al[1]; al[-1] = al[-2]
    return np.cross(al, r) + np.cross(gyro, np.cross(gyro, r))


def accel_series(d, k0, k1, b_g, lever_terms, ts_ns_list=None):
    """Concatenated sample series (times s, R_i, f_o (lever-corrected accel, bias-free), node indices)
    over nodes k0..k1, attitudes from gyro (bias-corrected by b_g) propagated forward from each node
    and backward from the next, blended linearly in time."""
    t_imu = d.t_imu_ns
    times_all, R_all, fo_all, node_pos = [], [], [], {}
    for k in range(k0, k1):
        ts0, ts1 = d.t_v_ns[k], d.t_v_ns[k + 1]
        times, P, dt, a = gyro_prefix(t_imu, d.gyro, ts0, ts1, b=b_g)
        R0, R1 = d.R_wc_meas[k], d.R_wc_meas[k + 1]
        Rf = np.einsum("ij,njk->nik", R0, P)
        Rb = np.einsum("ij,jk,nkl->nil", R1, P[-1].T, P)
        lam = ((times - times[0]) / (times[-1] - times[0]))[:, None]
        rv = Rot.from_matrix(np.einsum("nji,njk->nik", Rf, Rb)).as_rotvec()
        Ri = np.einsum("nij,njk->nik", Rf, Rot.from_rotvec(lam * rv).as_matrix())
        f = _interp3(t_imu.astype(np.float64), d.accel, times) - _interp3(t_imu.astype(np.float64), lever_terms, times)
        sl = slice(0, None) if k == k0 else slice(1, None)
        if k == k0: node_pos[k0] = 0
        times_all.append(times[sl]); R_all.append(Ri[sl]); fo_all.append(f[sl])
        node_pos[k + 1] = sum(len(x) for x in times_all) - 1
    return (np.concatenate(times_all) / 1e9, np.concatenate(R_all), np.concatenate(fo_all), node_pos)


def _cumtrapz(t, x):
    out = np.zeros_like(x)
    out[1:] = np.cumsum(0.5 * (x[1:] + x[:-1]) * np.diff(t).reshape((-1,) + (1,) * (x.ndim - 1)), axis=0)
    return out


def accel_terms(t_s, R, fo, g):
    """P(t) = double integral of (R fo + g);  Q(t) = double integral of R  (3x3) -- so that
    position(t) = p0 + v0 tau + P(t) - Q(t) b_a  for a constant body bias b_a."""
    a_w = np.einsum("nij,nj->ni", R, fo) + g
    v = _cumtrapz(t_s, a_w); P = _cumtrapz(t_s, v)
    Qv = _cumtrapz(t_s, R); Q = _cumtrapz(t_s, Qv)
    return P, Q


def run_A_accel(d, iv, b_g_est, mode="dt_weighted", tau=30.0, max_gap=0.12, lever=None):
    """Frame-level accel-bias scheme (velocity from the previous vision difference, position residual
    against IMU prediction). Returns b_a_est (K,3)."""
    K = len(d.t_v_ns); out = np.zeros((K, 3)); b = np.zeros(3)
    S_r = np.zeros(3); S_Q = np.zeros((3, 3)); tv = d.t_v_s
    lv = _lever_terms(d.t_imu_s, d.gyro, d.lever_est)
    for k in range(2, K):
        dt1, dt0 = tv[k] - tv[k - 1], tv[k - 1] - tv[k - 2]
        if 0 < dt1 <= max_gap and 0 < dt0 <= max_gap and d.strong[k] and d.strong[k - 1] and d.strong[k - 2]:
            v_prev = (d.p_wc_meas[k - 1] - d.p_wc_meas[k - 2]) / dt0
            t_s, R, fo, npos = accel_series(d, k - 1, k, b_g_est[k - 1], lv)
            P, Q = accel_terms(t_s, R, fo, d.g_est)
            p_pred0 = d.p_wc_meas[k - 1] + v_prev * dt1 + P[-1]
            d0 = d.p_wc_meas[k] - p_pred0                    # = -Q b* + noise
            Qe = Q[-1]
            if np.linalg.norm(d0) < 0.05:
                if mode == "naive_ema":
                    b_meas = -np.linalg.solve(Qe, d0)
                    b = b + min(1.0, dt1 / tau) * (b_meas - b)
                else:
                    lam = np.exp(-dt1 / tau)
                    S_r = lam * S_r + Qe.T @ (-d0); S_Q = lam * S_Q + Qe.T @ Qe
                    b = np.linalg.solve(S_Q + 1e-12 * np.eye(3), S_r)
        out[k] = b
    return out


# ------------------------------------------------------------------ (C) sliding-window GLS
def gls_gyro_window(d, iv, k0, k1, sig_rot, prior_b=1.0, gate_sig=4.0, iters=3, b_lin=None, n_gn=2):
    """Window nodes k0..k1. Gauss-Newton around b_lin: increments dR(b_lin) = dR0 Exp(-J b_lin) (first-order,
    error O(b^2 dt) -- see test), then Z_j = Log((R_0 G_j)^T R_j) = -J_j db - G_j^T e0 + e_j, b = b_lin + db.
    e0 = nuisance initial-attitude error (prior sigma = sig_rot).  Returns (b, cov, n_used) or None."""
    n = k1 - k0
    b_cur = np.zeros(3) if b_lin is None else np.asarray(b_lin, float).copy()
    for k in range(k0 + 1, k1 + 1):
        if not iv.ok[k]:
            return None
    for _gn in range(n_gn):
        G = [np.eye(3)]; Jc = [np.zeros((3, 3))]
        for j in range(1, n + 1):
            k = k0 + j
            dR = iv.dR[k] @ Exp(-iv.J[k] @ b_cur)
            G.append(G[-1] @ dR); Jc.append(dR.T @ Jc[-1] + iv.J[k])
        Z = np.array([Log((d.R_wc_meas[k0] @ G[j]).T @ d.R_wc_meas[k0 + j]) for j in range(1, n + 1)])
        A = np.zeros((3 * n, 6))
        for j in range(1, n + 1):
            A[3 * (j - 1):3 * j, :3] = -Jc[j]; A[3 * (j - 1):3 * j, 3:] = -G[j].T
        y = Z.ravel(); w = np.ones(3 * n)
        Ap = np.vstack([A, np.hstack([np.zeros((3, 3)), np.eye(3)]), np.hstack([np.eye(3) * (sig_rot / prior_b), np.zeros((3, 3))])])
        for _ in range(iters):
            yp = np.concatenate([y, np.zeros(6)]); wp = np.concatenate([w, np.ones(6)])
            Aw = Ap * wp[:, None]
            x, *_ = np.linalg.lstsq(Aw / sig_rot, yp * wp / sig_rot, rcond=None)
            res = (y - A @ x).reshape(n, 3)
            bad = np.linalg.norm(res, axis=1) > gate_sig * np.sqrt(3) * sig_rot * 1.5
            w = np.repeat((~bad).astype(float), 3)
        b_cur = b_cur + x[:3]
    Aw = Ap * np.concatenate([w, np.ones(6)])[:, None]
    cov = np.linalg.inv(Aw.T @ Aw / sig_rot ** 2)
    return b_cur, cov[:3, :3], int((~bad).sum())


def run_C_gyro(d, iv, W=30.0, every=5, sig_rot=None, min_nodes=8):
    K = len(d.t_v_ns); tv = d.t_v_s
    sig_rot = sig_rot or d.sig_rot
    out = np.zeros((K, 3)); std = np.full((K, 3), np.nan); b = np.zeros(3); s = np.full(3, np.nan)
    for k in range(K):
        if k % every == 0 and k >= min_nodes:
            k0 = int(np.searchsorted(tv, tv[k] - W))
            if k - k0 >= min_nodes:
                r = gls_gyro_window(d, iv, k0, k, sig_rot, b_lin=b)
                if r is not None:
                    b, cov, _ = r; s = np.sqrt(np.diag(cov))
        out[k] = b; std[k] = s
    return out, std


def gls_accel_window(d, k0, k1, b_g, lv, sig_pos, prior_ba=0.3):
    t_s, R, fo, npos = accel_series(d, k0, k1, b_g, lv)
    P, Q = accel_terms(t_s, R, fo, d.g_est)
    nodes = list(range(k0, k1 + 1))
    tau = np.array([t_s[npos[k]] - t_s[0] for k in nodes])
    n = len(nodes)
    A = np.zeros((3 * n + 3, 9)); y = np.zeros(3 * n + 3)
    for j, k in enumerate(nodes):
        i = npos[k]
        A[3 * j:3 * j + 3, 0:3] = np.eye(3); A[3 * j:3 * j + 3, 3:6] = tau[j] * np.eye(3); A[3 * j:3 * j + 3, 6:9] = -Q[i]
        y[3 * j:3 * j + 3] = d.p_wc_meas[k] - P[i]
    A[3 * n:, 6:9] = np.eye(3) * (sig_pos / prior_ba)
    x, *_ = np.linalg.lstsq(A / sig_pos, y / sig_pos, rcond=None)
    cov = np.linalg.inv(A.T @ A / sig_pos ** 2)
    return x[6:9], cov[6:9, 6:9]


def run_C_accel(d, b_g_est, W=30.0, every=20, sig_pos=None, prior_ba=0.3, min_nodes=10, max_nodes=400):
    K = len(d.t_v_ns); tv = d.t_v_s
    sig_pos = sig_pos or d.sig_pos
    lv = _lever_terms(d.t_imu_s, d.gyro, d.lever_est)
    out = np.zeros((K, 3)); std = np.full((K, 3), np.nan); b = np.zeros(3); s = np.full(3, np.nan)
    for k in range(K):
        if k % every == 0 and k >= min_nodes:
            k0 = max(int(np.searchsorted(tv, tv[k] - W)), k - max_nodes)
            if k - k0 >= min_nodes:
                b, cov = gls_accel_window(d, k0, k, b_g_est[k], lv, sig_pos, prior_ba)
                s = np.sqrt(np.diag(cov))
        out[k] = b; std[k] = s
    return out, std


# ------------------------------------------------------------------ (B) error-state Kalman filter
def run_B(d, q_bg=1e-5, q_ba=1e-3, sig_bg0=0.02, sig_ba0=0.3, meas_infl=1.5, gate_chi2=22.5,
          sig_rot=None, sig_pos=None, estimate_accel=True, update_pos=True):
    """15-state ESKF on the IMU grid; vision pose update at each node. Returns dict with b_g,b_a at nodes."""
    sr = (sig_rot or d.sig_rot) * meas_infl; sp = (sig_pos or d.sig_pos) * meas_infl
    tI = d.t_imu_s; gI = d.gyro; aI = d.accel; g = d.g_est; r = d.lever_est
    tV = d.t_v_s; K = len(tV)
    # event list: IMU samples + nodes
    lo = max(tI[0], tV[0]); ev_t = np.concatenate([tI[(tI >= lo)], tV])
    kind = np.concatenate([np.zeros((tI >= lo).sum(), int), np.ones(K, int)])
    order = np.lexsort((kind, ev_t)); ev_t = ev_t[order]; kind = kind[order]
    node_idx = np.concatenate([np.full((tI >= lo).sum(), -1), np.arange(K)])[order]
    om = np.stack([np.interp(ev_t, tI, gI[:, i]) for i in range(3)], 1)
    ac = np.stack([np.interp(ev_t, tI, aI[:, i]) for i in range(3)], 1)
    lvI = _lever_terms(tI, gI, r)
    lv = np.stack([np.interp(ev_t, tI, lvI[:, i]) for i in range(3)], 1)
    sg2, sa2 = 7e-4 ** 2, 6.5e-3 ** 2
    Rn = np.eye(3); p = np.zeros(3); v = np.zeros(3); bg = np.zeros(3); ba = np.zeros(3)
    P = np.zeros((15, 15)); started = False
    outg = np.zeros((K, 3)); outa = np.zeros((K, 3)); stdg = np.full((K, 3), np.nan); stda = np.full((K, 3), np.nan)
    n_rej = 0; H = np.zeros((6, 15)); H[0:3, 0:3] = np.eye(3); H[3:6, 6:9] = np.eye(3)
    Rm = np.diag([sr ** 2] * 3 + [sp ** 2] * 3)
    I15 = np.eye(15); t_prev = None; om_prev = None; ac_prev = None; lv_prev = None
    for i in range(len(ev_t)):
        t = ev_t[i]
        if started:
            dt = t - t_prev
            if dt > 0:
                w = 0.5 * (om[i] + om_prev) - bg
                f = 0.5 * (ac[i] + ac_prev) - ba - 0.5 * (lv[i] + lv_prev)
                dRm = Exp(0.5 * w * dt); Rmid = Rn @ dRm
                a_w = Rmid @ f + g
                p = p + v * dt + 0.5 * a_w * dt ** 2
                v = v + a_w * dt
                Rn = Rn @ Exp(w * dt)
                F = I15.copy()
                F[0:3, 9:12] = -Rmid * dt
                F[3:6, 0:3] = -skew(Rmid @ f) * dt
                F[3:6, 12:15] = -Rmid * dt
                F[6:9, 3:6] = np.eye(3) * dt
                Q = np.zeros((15, 15))
                Q[0:3, 0:3] = Rmid @ Rmid.T * sg2 * dt
                Q[3:6, 3:6] = Rmid @ Rmid.T * sa2 * dt
                Q[9:12, 9:12] = np.eye(3) * q_bg ** 2 * dt
                Q[12:15, 12:15] = np.eye(3) * (q_ba ** 2 * dt if estimate_accel else 0.0)
                P = F @ P @ F.T + Q
        t_prev = t; om_prev = om[i]; ac_prev = ac[i]; lv_prev = lv[i]
        if kind[i] == 1:
            k = node_idx[i]
            if not started:
                Rn = d.R_wc_meas[k].copy(); p = d.p_wc_meas[k].copy(); v = np.zeros(3)
                P = np.diag(np.array([sr] * 3 + [5.0] * 3 + [sp] * 3 + [sig_bg0] * 3 + [sig_ba0 if estimate_accel else 1e-6] * 3) ** 2)
                started = True
            else:
                z = np.concatenate([Log(d.R_wc_meas[k] @ Rn.T), d.p_wc_meas[k] - p])
                Hh = H if update_pos else H[:3]
                zz = z if update_pos else z[:3]
                Rr = Rm if update_pos else Rm[:3, :3]
                S = Hh @ P @ Hh.T + Rr
                chi2 = zz @ np.linalg.solve(S, zz)
                if chi2 < gate_chi2 * (1.0 if update_pos else 0.5):
                    Kg = P @ Hh.T @ np.linalg.inv(S)
                    dx = Kg @ zz
                    IKH = I15 - Kg @ Hh
                    P = IKH @ P @ IKH.T + Kg @ Rr @ Kg.T
                    Rn = Exp(dx[0:3]) @ Rn; v = v + dx[3:6]; p = p + dx[6:9]
                    bg = bg + dx[9:12]
                    if estimate_accel: ba = ba + dx[12:15]
                else:
                    n_rej += 1
            outg[k] = bg; outa[k] = ba
            stdg[k] = np.sqrt(np.diag(P)[9:12]); stda[k] = np.sqrt(np.diag(P)[12:15])
    return dict(b_g=outg, b_a=outa, std_g=stdg, std_a=stda, n_rej=n_rej)
