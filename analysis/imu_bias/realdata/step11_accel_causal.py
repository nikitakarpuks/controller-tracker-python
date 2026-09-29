"""Step 11: CAUSAL accel-bias estimator via recursive least squares over overlapping hop-1s windows (W=5 s):
    A <- lam A + M_w^T M_w ,  y <- lam y + M_w^T Q_w ,  b = (A + ridge I)^-1 y     [optionally with K_a: X_w=[M_w | F_w] , 12 params]
lam = exp(-hop/tau). Vision track (mocap headset ego: dev-only). tau chosen on 20-60 s anchors, reported on t>60 s.
Compared to: zero, constant fit on first 60 s (mocap windows, oracle-ish), all-time constant (upper bound)."""
import sys, csv
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from accel_tools import *
from src.imu_data import _lever_arm_correction
from pipeline import *
from step3f_scale_misalign import fit_body
from step4a_payoff_KB import corrected
from step6_gyro_estimators import block_bootstrap_ci
from step10b_accel_K import fit_bK, predict

TAUS = [20, 40, 80, 160]
GAPS = [0.25, 0.5, 1.0]

class AccelRLS:
    def __init__(self, tau_s, use_K, hop_s=1.0, ridge_rel=1e-2, min_windows=3):
        self.lam = np.exp(-hop_s / tau_s); self.use_K = use_K; self.ridge_rel = ridge_rel; self.minw = min_windows
        n = 12 if use_K else 3
        self.A = np.zeros((n, n)); self.y = np.zeros(n); self.nw = 0; self.beta = np.zeros(n)
    def update(self, Q, M, F):
        X = np.hstack([M, F]) if self.use_K else M
        self.A = self.lam * self.A + X.T @ X; self.y = self.lam * self.y + X.T @ Q; self.nw += 1
        if self.nw >= self.minw:
            lam_r = self.ridge_rel * np.mean(np.diag(self.A))
            self.beta = np.linalg.solve(self.A + lam_r * np.eye(len(self.y)), self.y)
        return self.current()
    def current(self):
        b = self.beta[:3]; K = self.beta[3:12].reshape(3, 3) if self.use_K else np.zeros((3, 3))
        return b, K

def series(AW, t0, t_end, tau, use_K, W=5, hop=1.0):
    est = AccelRLS(tau, use_K); T, P = [], []
    for t in np.arange(t0 + int(W * 1e9), t_end, int(hop * 1e9)):
        r = AW.components(int(t - W * 1e9), int(t))
        if r is not None and r[3] > 0.7:
            est.update(r[0], r[1], r[2])
        T.append(t); P.append(est.current())
    return np.array(T, dtype=np.int64), P

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
    run = Run(name); rows = []
    for c in CTRLS:
        d = run.ctrl[c]; prep = prepare(run, c, ego="mocap")
        trk_v = track_vision(run, c, prep); trk_m = track_mocap(run, c)
        t0 = prep["ts"][prep["ok"]][0]; split = t0 + int(60e9); sel_lo = t0 + int(20e9); t_end = prep["ts"][-1]
        st0, _ = make_steps(run, c, prep)
        bk, _, _ = fit_body([s for s in st0 if s["t"] < split], key="omega_g")
        gyro_c = corrected(d["gyro"], np.zeros(3), bk[3:].reshape(3, 3))
        AW_v = AccelWindows(run, c, trk_v, gyro=gyro_c); AW_m = AccelWindows(run, c, trk_m, gyro=gyro_c)
        b_tr, K_tr, _ = fit_bK(AW_m, t0, split, 5, True, False); bK_tr_b, bK_tr_K, _ = fit_bK(AW_m, t0, split, 5, True, True)
        b_all, K_all, _ = fit_bK(AW_m, t0, t_end, 5, True, True)
        cand = {}
        for tau in TAUS:
            for use_K in (False, True):
                cand[f"RLS-vision tau={tau}s {'b+K_a' if use_K else 'b'}"] = series(AW_v, t0, t_end, tau, use_K)
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
            store = {}
            for g in GAPS:
                anc = anchors(g); times = np.array([a[0] for a in anc])
                # time-varying params: evaluate each anchor with the estimator state AVAILABLE at that anchor -> need per-anchor params
                # implement by grouping: params dict per method is a function of t; predict() takes fixed (b,K), so loop anchors individually
                errs = {k: [] for k in ["naive", "accel b=0", "const b (first 60 s)", "const b+K_a (first 60 s)", "b+K_a all-time [UB]"] + list(cand)}
                for a_ in anc:
                    ta = a_[0]
                    prm = {"accel b=0": (np.zeros(3), np.zeros((3, 3))), "const b (first 60 s)": (b_tr, np.zeros((3, 3))),
                           "const b+K_a (first 60 s)": (bK_tr_b, bK_tr_K), "b+K_a all-time [UB]": (b_all, K_all)}
                    for cn, (T, P) in cand.items():
                        k = np.searchsorted(T, ta, side="right") - 1
                        prm[cn] = P[k] if k >= 0 else (np.zeros(3), np.zeros((3, 3)))
                    o = predict(run, c, [a_], gyro_c, d["accel"], d["lever"], vel_fn, prm)
                    for k in errs: errs[k].append(o[k][0])
                store[g] = (times, {k: np.array(v) for k, v in errs.items()})
            def sel(cn):
                sc = []
                for g in GAPS:
                    times, e = store[g]; m = (times >= sel_lo) & (times < split) & np.isfinite(e[cn]); sc.append(np.median(e[cn][m]))
                return np.mean(sc)
            best_b = min([k for k in cand if k.endswith(" b")], key=sel); best_bK = min([k for k in cand if k.endswith("b+K_a")], key=sel)
            print(f"\n===== [{name}/{c}] v0 from {v0src}   (chosen on 20-60 s: {best_b} | {best_bK})")
            print("   TEST t>60 s median/p90 position error (m)      gap(s): " + "".join(f"{g:>16.2f}" for g in GAPS))
            show = ["naive", "accel b=0", "const b (first 60 s)", "const b+K_a (first 60 s)", best_b, best_bK, "b+K_a all-time [UB]"]
            for nm in show:
                cells = []
                for g in GAPS:
                    times, e = store[g]; m = (times >= split) & np.isfinite(e[nm]); x = e[nm][m]
                    cells.append(f"{np.median(x):.3f}/{np.percentile(x,90):.3f}")
                    rows.append([name, c, v0src, nm, g, int(m.sum()), round(float(np.median(x)), 4), round(float(np.percentile(x, 90)), 4), round(float(np.mean(x)), 4)])
                print(f"   {nm[:50]:50s}" + "".join(f"{x:>16s}" for x in cells))
            print("   paired mean-error difference vs 'accel b=0' (m) [95% CI], negative = better:")
            for nm in show[2:]:
                line = []
                for g in (0.5, 1.0):
                    times, e = store[g]; m = (times >= split) & np.isfinite(e[nm]) & np.isfinite(e["accel b=0"])
                    diff = e[nm][m] - e["accel b=0"][m]; lo, hi = block_bootstrap_ci(diff, times[m]); line.append(f"{g}s: {diff.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")
                print(f"     {nm[:50]:50s} " + "   ".join(line))
            # trajectory of chosen causal b for the record
            if v0src == "vision":
                T, P = cand[best_b]
                with open(OUT_DIR / f"traj_accel_{name}_{c}{LEVER_TAG}.csv", "w", newline="") as f:
                    w = csv.writer(f); w.writerow(["t_ns", "t_s", "bx", "by", "bz"])
                    for t, (b, K) in list(zip(T, P))[::2]: w.writerow([int(t), round((t - t0) / 1e9, 2), *[round(float(x), 5) for x in b]])
    with open(OUT_DIR / f"accel_causal_{name}{LEVER_TAG}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["recording", "ctrl", "v0_source", "method", "gap_s", "n_test", "median_m", "p90_m", "mean_m"]); w.writerows(rows)
    print("saved")
