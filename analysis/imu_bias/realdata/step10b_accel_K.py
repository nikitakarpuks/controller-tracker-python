"""Step 10b: accel scale/misalignment K_a. Fit (b), (K_a) and (b,K_a) on FIRST-60 s mocap windows (overlapping 5/10 s windows), apply as
f_used = (I+K_a)^-1 (f_c - b) in position dead-reckoning, report held-out (t>60 s) errors. Same anchors/protocol as step10."""
import sys
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from accel_tools import *
from src.imu_data import _lever_arm_correction
from pipeline import *
from step3f_scale_misalign import fit_body
from step4a_payoff_KB import corrected
from step6_gyro_estimators import block_bootstrap_ci

GAPS = [0.25, 0.5, 1.0]

def fit_bK(AW, t_lo, t_hi, W, use_b, use_K, hop=2.0, lam_rel=1e-2, min_cov=0.6):
    X, y = [], []
    for a in np.arange(t_lo, t_hi - int(W * 1e9), int(hop * 1e9)):
        r = AW.components(int(a), int(a + W * 1e9))
        if r is None or r[3] < min_cov: continue
        blocks = ([r[1]] if use_b else []) + ([r[2]] if use_K else [])
        X.append(np.hstack(blocks)); y.append(r[0])
    X = np.vstack(X); y = np.concatenate(y)
    lam = lam_rel * np.mean(np.diag(X.T @ X))
    beta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    b = beta[:3] if use_b else np.zeros(3)
    K = beta[(3 if use_b else 0):(3 if use_b else 0) + 9].reshape(3, 3) if use_K else np.zeros((3, 3))
    return b, K, len(y) // 3

def predict(run, ctrl, anchors, gyro_c, accel, lever, vel_fn, params):
    d = run.ctrl[ctrl]; t = d["t"]
    out = {k: [] for k in ["naive"] + list(params)}
    for (ta, tb, Ra, pa, pb) in anchors:
        i0, i1 = np.searchsorted(t, ta), np.searchsorted(t, tb)
        ts_ = np.concatenate(([ta], t[i0:i1][(t[i0:i1] > ta) & (t[i0:i1] < tb)], [tb])).astype(np.int64)
        if len(ts_) < 4:
            for k in out: out[k].append(np.nan)
            continue
        g_ = np.stack([np.interp(ts_, t, gyro_c[:, k]) for k in range(3)], 1); f_ = np.stack([np.interp(ts_, t, accel[:, k]) for k in range(3)], 1)
        dts = np.diff(ts_) / 1e9
        R = np.empty((len(ts_), 3, 3)); R[0] = Ra
        for k in range(len(dts)): R[k + 1] = R[k] @ Rotation.from_rotvec(0.5 * (g_[k] + g_[k + 1]) * dts[k]).as_matrix()
        fc = f_ - _lever_arm_correction(ts_, ts_, g_, lever)
        v0 = vel_fn(ta)
        if v0 is None:
            for k in out: out[k].append(np.nan)
            continue
        out["naive"].append(np.linalg.norm(pa + v0 * (tb - ta) / 1e9 - pb))
        for nm, (b, K) in params.items():
            fu = (np.linalg.inv(np.eye(3) + K) @ (fc - b).T).T
            aw = np.einsum("nij,nj->ni", R, fu) + G_ABS
            v = np.empty_like(aw); v[0] = v0; p = pa.copy()
            for k in range(len(dts)):
                v[k + 1] = v[k] + 0.5 * (aw[k] + aw[k + 1]) * dts[k]; p = p + 0.5 * (v[k] + v[k + 1]) * dts[k]
            out[nm].append(np.linalg.norm(p - pb))
    return {k: np.array(v) for k, v in out.items()}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
    run = Run(name)
    for c in CTRLS:
        d = run.ctrl[c]; prep = prepare(run, c, ego="mocap")
        trk_v = track_vision(run, c, prep); trk_m = track_mocap(run, c)
        t0 = prep["ts"][prep["ok"]][0]; split = t0 + int(60e9); t_end = prep["ts"][-1]
        st0, _ = make_steps(run, c, prep)
        bk, _, _ = fit_body([s for s in st0 if s["t"] < split], key="omega_g")
        gyro_c = corrected(d["gyro"], np.zeros(3), bk[3:].reshape(3, 3))
        AW_m = AccelWindows(run, c, trk_m, gyro=gyro_c)
        params = {"accel b=0": (np.zeros(3), np.zeros((3, 3)))}
        for W in (5, 10):
            for lab, ub, uk in ((f"b only (W={W})", True, False), (f"K_a only (W={W})", False, True), (f"b+K_a (W={W})", True, True)):
                b, K, n = fit_bK(AW_m, t0, split, W, ub, uk)
                params[lab] = (b, K)
                if W == 10: print(f"[{name}/{c}] fit {lab}: n_windows {n}  b={np.round(b,3)}  diag(K)={np.round(np.diag(K),4)}  |K|max={np.abs(K).max():.4f}")
        b_all, K_all, n = fit_bK(AW_m, t0, t_end, 10, True, True)
        params["b+K_a all-time (W=10) [UB]"] = (b_all, K_all)
        tsv = trk_v.ts
        def anchors(gap):
            out = []
            for i in range(0, len(tsv), 3):
                ta = tsv[i]; kk = np.searchsorted(tsv, ta + gap * 1e9)
                cs = [k for k in (kk - 1, kk) if 0 <= k < len(tsv) and tsv[k] > ta]
                if not cs: continue
                k = min(cs, key=lambda k: abs(tsv[k] - (ta + gap * 1e9)))
                if abs(tsv[k] - (ta + gap * 1e9)) > max(0.006, 0.35 * gap) * 1e9 or np.max(np.diff(tsv[i:k + 1])) > 0.1e9: continue
                out.append((ta, tsv[k], trk_v.R[i], trk_v.P[i], trk_v.P[k]))
            return out
        for v0src in ("vision", "mocap"):
            vel_fn = (lambda t: trk_v.v_at(t - int(0.03e9), 0.03, 3)) if v0src == "vision" else (lambda t: trk_m.v_at(t, 0.04, 3))
            print(f"\n   --- [{c}] v0 from {v0src}: TEST t>60 s median / p90 position error (m)      gap(s): " + "".join(f"{g:>16.2f}" for g in GAPS))
            store = {}
            for g in GAPS:
                anc = anchors(g); times = np.array([a[0] for a in anc])
                store[g] = (times, predict(run, c, anc, gyro_c, d["accel"], d["lever"], vel_fn, params))
            for nm in ["naive"] + list(params):
                cells = []
                for g in GAPS:
                    times, errs = store[g]; m = (times >= split) & np.isfinite(errs[nm]); e = errs[nm][m]
                    cells.append(f"{np.median(e):.3f}/{np.percentile(e,90):.3f}")
                print(f"   {nm[:50]:50s}" + "".join(f"{x:>16s}" for x in cells))
            print("   paired mean-error difference vs 'accel b=0' (m) [95% CI], negative = better:")
            for nm in [k for k in params if k != "accel b=0"]:
                line = []
                for g in (0.5, 1.0):
                    times, errs = store[g]; m = (times >= split) & np.isfinite(errs[nm]) & np.isfinite(errs["accel b=0"])
                    diff = errs[nm][m] - errs["accel b=0"][m]; lo, hi = block_bootstrap_ci(diff, times[m])
                    line.append(f"{g}s: {diff.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")
                print(f"     {nm[:50]:50s} " + "   ".join(line))
