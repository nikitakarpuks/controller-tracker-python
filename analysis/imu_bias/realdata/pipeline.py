"""pipeline.py -- turns a Run + controller into per-step residuals for the bias estimators, plus the mocap oracle
and the payoff (rotation-prediction) evaluator. Strictly uses only data at/before each step for the estimators."""
import numpy as np
from common import *
from estimators import *


def prepare(run, ctrl, ego="mocap", strong=(8, 0.5), imu0_bias=None, imu0_lag_ns=0, use_all_frames=False):
    """Frame table for one controller.
    ego: 'mocap'  -> R_wc = R_wh(mocap) @ R_hc  (dev-only ground-truth headset orientation)
         'imu0'   -> headset ego rotation chained from imu0 gyro (minus imu0_bias, shifted by imu0_lag_ns)
         'none'   -> R_wc = R_hc (rig frame, headset motion NOT removed)
    Returns dict(ts, R_wc, R_hc, ok, conf, err, nin, p_hc)."""
    d = run.ctrl[ctrl]; v = d["vision"]
    ts = v.ts
    ok = np.ones(len(ts), bool) if use_all_frames else v.strong_mask(*strong)
    R_wc = np.empty_like(v.R)
    if ego == "mocap":
        for i, t in enumerate(ts):
            Rh = run.R_wh(t)
            if Rh is None:
                ok[i] = False; R_wc[i] = v.R[i]
            else:
                R_wc[i] = Rh @ v.R[i]
    elif ego == "none":
        R_wc[:] = v.R
    elif ego == "imu0":
        t0, g0, _ = run.imu0
        bias = np.zeros(3) if imu0_bias is None else np.asarray(imu0_bias)
        Rh = np.eye(3); prev = None
        for i, t in enumerate(ts):
            if prev is not None:
                Rg = integrate_gyro_segment(t0 + imu0_lag_ns, g0 - bias, int(ts[prev]), int(t))
                if Rg is None:
                    ok[i] = False
                else:
                    Rh = Rh @ Rg
            R_wc[i] = Rh @ v.R[i]
            prev = i
    else:
        raise ValueError(ego)
    return dict(ts=ts, R_wc=R_wc, R_hc=v.R, ok=ok, conf=v.conf, err=v.err, nin=v.nin, p_hc=v.p, ctrl=ctrl)


def make_steps(run, ctrl, prep, max_dt=0.05, outlier_deg=3.0, gyro_bias=np.zeros(3), gyro=None):
    """Consecutive strong-frame pairs -> step dicts. `outlier_deg`: any step whose zero-bias residual exceeds this
    (implausible jump / swap / wrong solve) NEVER feeds an estimator. Returned list is time-ordered; each step also
    carries indices and time for bookkeeping."""
    d = run.ctrl[ctrl]
    ts, R_wc, ok = prep["ts"], prep["R_wc"], prep["ok"]
    idx = np.flatnonzero(ok)
    steps = []
    n_out = 0
    for a, b in zip(idx[:-1], idx[1:]):
        dt = (ts[b] - ts[a]) / 1e9
        if dt <= 0 or dt > max_dt:
            continue
        Rg = integrate_gyro_segment(d["t"], (d["gyro"] if gyro is None else gyro) - gyro_bias, int(ts[a]), int(ts[b]))
        if Rg is None:
            continue
        dRv = R_wc[a].T @ R_wc[b]
        e0 = Rotation.from_matrix(Rg.T @ dRv).as_rotvec()
        if np.linalg.norm(e0) > np.radians(outlier_deg):
            n_out += 1
            continue
        om = Rotation.from_matrix(dRv).as_rotvec() / dt
        om_g = Rotation.from_matrix(Rg).as_rotvec() / dt      # gyro-side rate (no vision noise) -- the right regressor
        steps.append(dict(i=a, j=b, t=ts[b], dt=dt, e0=e0, R_end=R_wc[b], omega=om, omega_g=om_g, rate=float(np.linalg.norm(om))))
    return steps, n_out


def run_estimator(est, steps):
    """Feed steps in order; returns (t_ns (N,), b (N,3)) = bias estimate AFTER absorbing each step."""
    T, B = [], []
    for s in steps:
        b = est.update(s)
        T.append(s["t"]); B.append(np.array(b))
    return np.array(T, dtype=np.int64), np.array(B)


def b_at(T, B, t_query, default=None):
    """Causal lookup: latest estimate with time <= t_query (estimate available AT the anchor frame)."""
    k = np.searchsorted(T, t_query, side="right") - 1
    if k < 0:
        return np.zeros(3) if default is None else default
    return B[k]


# ------------------------------- mocap oracle -------------------------------------------------------------
def oracle_bias(run, ctrl, window_s=10.0, hop_s=2.0, ridge=1e-2, max_dt=0.05):
    """Centered-window (NON-causal) gyro-bias truth from the CONTROLLER MOCAP orientation (bridge-composed to the
    LED/body frame), at every vision timestamp (not only strong ones -- mocap needs no vision quality).
    Same math as WorldRLS but with mocap orientation instead of vision. Returns (t_centers_ns, b (N,3), n_steps)."""
    d = run.ctrl[ctrl]; ts = d["vision"].ts
    R = []
    keep = []
    for t in ts:
        Rm = run.R_w_ctrl_mocap_led(ctrl, t)
        keep.append(Rm is not None)
        R.append(np.eye(3) if Rm is None else Rm)
    R = np.array(R); keep = np.array(keep)
    idx = np.flatnonzero(keep)
    st = []
    for a, b in zip(idx[:-1], idx[1:]):
        dt = (ts[b] - ts[a]) / 1e9
        if dt <= 0 or dt > max_dt:
            continue
        Rg = integrate_gyro_segment(d["t"], d["gyro"], int(ts[a]), int(ts[b]))
        if Rg is None:
            continue
        e0 = Rotation.from_matrix(Rg.T @ (R[a].T @ R[b])).as_rotvec()
        if np.linalg.norm(e0) > np.radians(3.0):
            continue
        st.append((ts[b], dt, e0, R[b]))
    tt = np.array([s[0] for s in st])
    centers = np.arange(tt[0] + int(window_s / 2 * 1e9), tt[-1] - int(window_s / 2 * 1e9) + 1, int(hop_s * 1e9))
    out_t, out_b, out_n = [], [], []
    for c in centers:
        lo, hi = np.searchsorted(tt, c - int(window_s / 2 * 1e9)), np.searchsorted(tt, c + int(window_s / 2 * 1e9))
        if hi - lo < 50:
            continue
        A = np.zeros((3, 3)); S = np.zeros(3)
        for s in st[lo:hi]:
            A += s[1] * s[3]; S += -(s[3] @ s[2])
        out_t.append(c); out_b.append(np.linalg.solve(A.T @ A + (0.008 / 0.02) ** 2 * np.eye(3), A.T @ S)); out_n.append(hi - lo)
    return np.array(out_t, dtype=np.int64), np.array(out_b), np.array(out_n)


def oracle_lookup(t_or, b_or, t_query):
    """Nearest oracle window centre (for evaluating estimator error against truth)."""
    k = np.clip(np.searchsorted(t_or, t_query), 1, len(t_or) - 1)
    k = np.where(np.abs(t_or[k] - t_query) < np.abs(t_or[k - 1] - t_query), k, k - 1)
    return b_or[k]


# ------------------------------- payoff: prediction error over a gap ---------------------------------------
def gap_errors(run, ctrl, prep, T_est, B_est, gap_s, tol_frac=0.35, bias_fn=None, t_min_ns=None, anchors=None):
    """For every strong ANCHOR frame a (time >= t_min_ns) pick the strong TARGET frame nearest to t_a + gap_s
    (within tol_frac*gap_s, min 6 ms); predict the target rotation by integrating the gyro over [t_a, t_b] minus
    the bias AVAILABLE AT THE ANCHOR (causal) -- bias_fn(t_a) -> (3,) rad/s -- and return the error angles (deg),
    anchor times and gap durations. The reference rotation is the vision relative rotation with ego-motion removed.
    """
    d = run.ctrl[ctrl]
    ts, R_wc, ok = prep["ts"], prep["R_wc"], prep["ok"]
    idx = np.flatnonzero(ok)
    tt = ts[idx]
    errs, tas, gaps = [], [], []
    tol = max(0.006, tol_frac * gap_s) * 1e9
    for k, a in enumerate(idx):
        if t_min_ns is not None and ts[a] < t_min_ns:
            continue
        target = ts[a] + gap_s * 1e9
        kk = np.searchsorted(tt, target)
        cand = [c for c in (kk - 1, kk) if 0 <= c < len(tt) and c != k and tt[c] > ts[a]]
        if not cand:
            continue
        c = min(cand, key=lambda c: abs(tt[c] - target))
        if abs(tt[c] - target) > tol:
            continue
        b = idx[c]
        bias = bias_fn(ts[a])
        Rg = integrate_gyro_segment(d["t"], d["gyro"] - bias, int(ts[a]), int(ts[b]))
        if Rg is None:
            continue
        errs.append(rot_deg(Rg.T @ (R_wc[a].T @ R_wc[b])))
        tas.append(ts[a]); gaps.append((ts[b] - ts[a]) / 1e9)
    return np.array(errs), np.array(tas), np.array(gaps)


def mocap_steps(run, ctrl, max_dt=0.05, outlier_deg=3.0, use_strong_only=False, prep_vision=None, gyro=None):
    """Per-step residuals built from the CONTROLLER MOCAP orientation (bridge-composed LED frame) at the vision
    timestamps -- the low-noise reference used for the oracle and for residual-decomposition regressions."""
    d = run.ctrl[ctrl]; ts = d["vision"].ts
    R = []; keep = []
    for t in ts:
        Rm = run.R_w_ctrl_mocap_led(ctrl, t)
        keep.append(Rm is not None); R.append(np.eye(3) if Rm is None else Rm)
    R = np.array(R); keep = np.array(keep)
    if use_strong_only:
        keep &= d["vision"].strong_mask()
    idx = np.flatnonzero(keep)
    st = []
    for a, b in zip(idx[:-1], idx[1:]):
        dt = (ts[b] - ts[a]) / 1e9
        if dt <= 0 or dt > max_dt:
            continue
        Rg = integrate_gyro_segment(d["t"], d["gyro"] if gyro is None else gyro, int(ts[a]), int(ts[b]))
        if Rg is None:
            continue
        dR = R[a].T @ R[b]
        e0 = Rotation.from_matrix(Rg.T @ dR).as_rotvec()
        if np.linalg.norm(e0) > np.radians(outlier_deg):
            continue
        om = Rotation.from_matrix(dR).as_rotvec() / dt
        om_g = Rotation.from_matrix(Rg).as_rotvec() / dt
        st.append(dict(i=a, j=b, t=ts[b], dt=dt, e0=e0, R_end=R[b], omega=om, omega_g=om_g, rate=float(np.linalg.norm(om))))
    return st


def window_bias(st, t_lo, t_hi, mu=(0.008 / 0.02) ** 2):
    """World-frame bias over steps with t in [t_lo, t_hi):  b = (A^T A + mu I)^-1 A^T S  with A = sum dt R_end, S = sum -R_end e0
    (prior-regularised LS -- the plain solve A b = S is ill-conditioned on fast/large-angle motion; mu = (sigma_s/sigma_b)^2, see estimators.WorldRLS)."""
    tt = np.array([s["t"] for s in st])
    lo, hi = np.searchsorted(tt, t_lo), np.searchsorted(tt, t_hi)
    if hi - lo < 30:
        return None
    A = sum(s["dt"] * s["R_end"] for s in st[lo:hi])
    S = sum(-(s["R_end"] @ s["e0"]) for s in st[lo:hi])
    return np.linalg.solve(A.T @ A + mu * np.eye(3), A.T @ S)
