"""Step 14b: is the unexplained accel error a LEVER-ARM (delta r), scale/misalignment (K_a) or bias (b) problem? Regress first-60 s mocap windows
Q = M b + F K_a + O dr, evaluate on held-out windows (velocity residual) and held-out position dead-reckoning (v0/R from mocap to isolate the accel model)."""
import sys
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from accel_tools import *
from src.imu_data import _lever_arm_correction
from pipeline import *
from step3f_scale_misalign import fit_body
from step4a_payoff_KB import corrected
name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
run = Run(name)
def windows(AW, t_lo, t_hi, W=5, hop=2.0, min_cov=0.6):
    out = []
    for a in np.arange(t_lo, t_hi - int(W * 1e9), int(hop * 1e9)):
        r = AW.components(int(a), int(a + W * 1e9))
        if r is not None and r[3] > min_cov: out.append((r[0], r[1], r[2], AW.last_O.copy()))
    return out
def fit(win, use):                                   # use subset of {"b","K","r"}
    X, y = [], []
    for Q, M, F, O in win:
        blocks = [M] * ("b" in use) + [F] * ("K" in use) + [O] * ("r" in use)
        X.append(np.hstack(blocks)); y.append(Q)
    X = np.vstack(X); y = np.concatenate(y); lam = 1e-2 * np.mean(np.diag(X.T @ X))
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
def unpack(beta, use):
    k = 0; b = np.zeros(3); K = np.zeros((3, 3)); dr = np.zeros(3)
    if "b" in use: b = beta[k:k + 3]; k += 3
    if "K" in use: K = beta[k:k + 9].reshape(3, 3); k += 9
    if "r" in use: dr = beta[k:k + 3]
    return b, K, dr
for c in CTRLS:
    d = run.ctrl[c]; prep = prepare(run, c, ego="mocap"); trk_v = track_vision(run, c, prep); trk_m = track_mocap(run, c)
    t0 = prep["ts"][prep["ok"]][0]; split = t0 + int(60e9); t_end = prep["ts"][-1]
    st0, _ = make_steps(run, c, prep); bk, _, _ = fit_body([s for s in st0 if s["t"] < split], key="omega_g")
    gyro_c = corrected(d["gyro"], np.zeros(3), bk[3:].reshape(3, 3))
    AW = AccelWindows(run, c, trk_m, gyro=gyro_c)
    tr = windows(AW, t0, split); te = windows(AW, split, t_end)
    print(f"\n[{name}/{c}] factory-derived lever arm r = {np.round(d['lever']*1000,1)} mm; train windows {len(tr)}, test windows {len(te)}")
    sets = ["", "b", "K", "r", "br", "bK", "Kr", "bKr"]
    print("   held-out velocity residual per 5 s window [m/s] (median / mean):")
    fits = {}
    for use in sets:
        if use == "":
            res = [np.linalg.norm(Q) for Q, M, F, O in te]; fits[use] = (np.zeros(3), np.zeros((3, 3)), np.zeros(3))
            print(f"      {'zero':6s}: {np.median(res):.3f} / {np.mean(res):.3f}"); continue
        beta = fit(tr, use); b, K, dr = unpack(beta, use); fits[use] = (b, K, dr)
        res = []
        for Q, M, F, O in te:
            pred = M @ b + F @ K.ravel() + O @ dr
            res.append(np.linalg.norm(Q - pred))
        extra = f"  b={np.round(b,3)}" if "b" in use else ""
        extra += f"  dr={np.round(dr*1000,1)} mm" if "r" in use else ""
        extra += f"  diag(K)={np.round(np.diag(K),3)}" if "K" in use else ""
        print(f"      {use:6s}: {np.median(res):.3f} / {np.mean(res):.3f}{extra}")
    # held-out position dead-reckoning with R,v0 from mocap (isolates the accelerometer model)
    tsv = trk_v.ts; t = d["t"]
    def positions(gap, use):
        b, K, dr = fits[use]; lever = d["lever"] + dr; errs = []
        for i in range(0, len(tsv), 3):
            ta = tsv[i]
            if ta < split: continue
            kk = np.searchsorted(tsv, ta + gap * 1e9); cs = [k for k in (kk - 1, kk) if 0 <= k < len(tsv) and tsv[k] > ta]
            if not cs: continue
            k = min(cs, key=lambda k: abs(tsv[k] - (ta + gap * 1e9)))
            if abs(tsv[k] - (ta + gap * 1e9)) > max(0.006, 0.35 * gap) * 1e9 or np.max(np.diff(tsv[i:k + 1])) > 0.1e9: continue
            tb = tsv[k]; i0, i1 = np.searchsorted(t, ta), np.searchsorted(t, tb)
            ts_ = np.concatenate(([ta], t[i0:i1][(t[i0:i1] > ta) & (t[i0:i1] < tb)], [tb])).astype(np.int64)
            if len(ts_) < 4: continue
            g_ = np.stack([np.interp(ts_, t, gyro_c[:, q]) for q in range(3)], 1); f_ = np.stack([np.interp(ts_, t, d["accel"][:, q]) for q in range(3)], 1)
            dts = np.diff(ts_) / 1e9; R = trk_m.R_at(ts_)
            v0 = trk_m.v_at(ta, 0.04, 3)
            if v0 is None: continue
            fc = f_ - _lever_arm_correction(ts_, ts_, g_, lever)
            fu = (np.linalg.inv(np.eye(3) + K) @ (fc - b).T).T
            aw = np.einsum("nij,nj->ni", R, fu) + G_ABS
            v = np.empty_like(aw); v[0] = v0; p = trk_v.P[i].copy()
            for q in range(len(dts)):
                v[q + 1] = v[q] + 0.5 * (aw[q] + aw[q + 1]) * dts[q]; p = p + 0.5 * (v[q] + v[q + 1]) * dts[q]
            errs.append(np.linalg.norm(p - trk_v.P[k]))
        return np.array(errs)
    print("   held-out position dead-reckoning (R, v0 from mocap), median error (m):        gap:      0.5 s     1.0 s")
    for use in sets:
        print(f"      model {use if use else 'zero':6s}" + " " * 50 + "".join(f"{np.median(positions(g, use)):>10.3f}" for g in (0.5, 1.0)))
