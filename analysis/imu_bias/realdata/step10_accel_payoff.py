"""Step 10: accelerometer-bias payoff = position dead-reckoning error over gaps, on HELD-OUT time.

Prediction (absolute inertial frame, ego from mocap headset -- dev-only; same convention as predict_headset_relative_pose):
   anchor a (strong vision frame): R_a, p_a from vision (ego-lifted); v_a from a CAUSAL 60-ms LS line fit over frames <= t_a
   (variant 'v0=mocap': controller-mocap velocity, to isolate the accel term from the v0 noise);
   R(t) by integrating the (K-corrected) gyro sample-by-sample from R_a;  a_w(t) = R(t) (f_c(t) - b_a) + g;  p_pred(t_b) by double
   trapezoid integration.  Error = |p_pred - p_vision(t_b)| (m).
Methods: naive constant velocity | accel b=0 | accel + constant b (fit on first 60 s, mocap-windows oracle) | accel + causal windowed
estimator (vision, W, EMA beta) | non-causal windowed oracle (mocap 20 s centred).  Protocol: hyper-parameters on 20-60 s, report t>60 s."""
import sys, csv
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from accel_tools import *
from src.imu_data import _lever_arm_correction
from pipeline import *
from step3f_scale_misalign import fit_body
from step4a_payoff_KB import corrected
from step6_gyro_estimators import block_bootstrap_ci

GAPS = [0.1, 0.25, 0.5, 1.0]


def const_from_windows(AW, t_lo, t_hi, W, hop=2.0, min_cov=0.6):
    Ms, Qs = [], []
    for a in np.arange(t_lo, t_hi - int(W * 1e9), int(hop * 1e9)):
        r = AW.components(int(a), int(a + W * 1e9))
        if r is not None and r[3] > min_cov:
            Qs.append(r[0]); Ms.append(r[1])
    if len(Qs) < 3:
        return np.zeros(3), 0
    M = sum(Ms); Q = sum(Qs)
    return np.linalg.solve(M + 1e-3 * np.eye(3), Q), len(Qs)


def causal_series(AW, t_lo, t_hi, W, beta, hop=1.0, min_cov=0.7):
    """Sliding-window estimator: every `hop` s take the window [t-W, t]; b <- (1-beta) b + beta b_win. Only data <= t used."""
    T, B = [], []
    b = np.zeros(3)
    for t in np.arange(t_lo + int(W * 1e9), t_hi, int(hop * 1e9)):
        r = AW.measure(int(t - W * 1e9), int(t))
        if r is not None and r[1] > min_cov:
            b = (1 - beta) * b + beta * r[0] if np.any(b) else r[0]
        T.append(t); B.append(b.copy())
    return np.array(T, dtype=np.int64), np.array(B)


def predict_positions(run, ctrl, anchors, gyro_c, accel, track_abs, lever, vel_fn, gap_s, bias_fns):
    """anchors: list of (a_idx, b_idx, R_a(3x3), p_a, t_a, t_b, p_b_target). Returns dict name -> errors (m)."""
    d = run.ctrl[ctrl]; t = d["t"]
    out = {k: [] for k in ["naive"] + list(bias_fns)}
    for (ta, tb, Ra, pa, pb) in anchors:
        i0, i1 = np.searchsorted(t, ta), np.searchsorted(t, tb)
        if i1 - i0 < 3:
            for k in out: out[k].append(np.nan)
            continue
        ts_ = np.concatenate(([ta], t[i0:i1][(t[i0:i1] > ta) & (t[i0:i1] < tb)], [tb])).astype(np.int64)
        g_ = np.stack([np.interp(ts_, t, gyro_c[:, k]) for k in range(3)], 1)
        f_ = np.stack([np.interp(ts_, t, accel[:, k]) for k in range(3)], 1)
        dts = np.diff(ts_) / 1e9
        R = np.empty((len(ts_), 3, 3)); R[0] = Ra
        for k in range(len(dts)):
            R[k + 1] = R[k] @ Rotation.from_rotvec(0.5 * (g_[k] + g_[k + 1]) * dts[k]).as_matrix()
        lev = _lever_arm_correction(ts_, ts_, g_, lever) if len(ts_) > 2 else np.zeros_like(f_)
        fc = f_ - lev
        v0 = vel_fn(ta)
        if v0 is None:
            for k in out: out[k].append(np.nan)
            continue
        out["naive"].append(np.linalg.norm(pa + v0 * (tb - ta) / 1e9 - pb))
        for name, bf in bias_fns.items():
            b = bf(ta)
            aw = np.einsum("nij,nj->ni", R, fc - b) + G_ABS
            v = np.empty_like(aw); v[0] = v0; p = pa.copy()
            for k in range(len(dts)):
                v[k + 1] = v[k] + 0.5 * (aw[k] + aw[k + 1]) * dts[k]
                p = p + 0.5 * (v[k] + v[k + 1]) * dts[k]
            out[name].append(np.linalg.norm(p - pb))
    return {k: np.array(v) for k, v in out.items()}


def main(name="static_dark"):
    run = Run(name)
    rows = []
    for c in CTRLS:
        d = run.ctrl[c]
        prep = prepare(run, c, ego="mocap")
        trk_v = track_vision(run, c, prep); trk_m = track_mocap(run, c)
        t0 = prep["ts"][prep["ok"]][0]; split = t0 + int(60e9); sel_lo = t0 + int(20e9); t_end = prep["ts"][-1]
        st0, _ = make_steps(run, c, prep)
        beta_k, _, _ = fit_body([s for s in st0 if s["t"] < split], key="omega_g")
        gyro_c = corrected(d["gyro"], np.zeros(3), beta_k[3:].reshape(3, 3))
        AW_v = AccelWindows(run, c, trk_v, gyro=gyro_c); AW_m = AccelWindows(run, c, trk_m, gyro=gyro_c)
        # constants / references
        b_train_mocap, n1 = const_from_windows(AW_m, t0, split, 10)
        b_all_mocap, n2 = const_from_windows(AW_m, t0, t_end, 10)
        b_train_vis, n3 = const_from_windows(AW_v, t0, split, 10)
        print(f"\n===== [{name}/{c}]  constant accel bias:  fit first-60s mocap windows ({n1}) {np.round(b_train_mocap,3)} | vision windows ({n3}) {np.round(b_train_vis,3)} | all-time mocap ({n2}) {np.round(b_all_mocap,3)}  m/s^2")
        cen = np.arange(t0 + int(10e9), t_end - int(10e9), int(2e9))
        Bor = []
        for cc in cen:
            r = AW_m.measure(int(cc - 10e9), int(cc + 10e9))
            Bor.append(r[0] if (r is not None and r[1] > 0.6) else np.full(3, np.nan))
        Bor = np.array(Bor); okc = np.all(np.isfinite(Bor), axis=1)
        bias_or = lambda t, cen=cen[okc], Bor=Bor[okc]: Bor[np.clip(np.searchsorted(cen, t), 0, len(cen) - 1)]
        # candidate causal estimators
        cand = {}
        for W in (10, 20, 40):
            for beta in (1.0, 0.5, 0.2):
                T, B = causal_series(AW_v, t0, t_end, W, beta)
                cand[f"causal vis-window W={W}s beta={beta}"] = (T, B)
        # anchor sets per gap (paired)
        tsv = trk_v.ts
        def make_anchors(gap):
            idx = np.arange(0, len(tsv), 3); out = []
            for i in idx:
                ta = tsv[i]; kk = np.searchsorted(tsv, ta + gap * 1e9)
                cand_ = [k for k in (kk - 1, kk) if 0 <= k < len(tsv) and tsv[k] > ta]
                if not cand_: continue
                k = min(cand_, key=lambda k: abs(tsv[k] - (ta + gap * 1e9)))
                if abs(tsv[k] - (ta + gap * 1e9)) > max(0.006, 0.35 * gap) * 1e9: continue
                # require the whole interval inside ONE hole-free stretch (vision frames between a..b spaced <= 0.1 s)
                if np.max(np.diff(tsv[i:k + 1])) > 0.1e9: continue
                out.append((ta, tsv[k], trk_v.R[i], trk_v.P[i], trk_v.P[k]))
            return out
        for v0src in ("vision", "mocap"):
            vel_fn = (lambda t: trk_v.v_at(t - int(0.03e9), 0.03, 3)) if v0src == "vision" else (lambda t: trk_m.v_at(t, 0.04, 3))
            print(f"   --- v0 from {v0src}: TEST (t>60s) median / p90 position error (m) by gap")
            print("   gap(s):                                        " + "".join(f"{g:>16.2f}" for g in GAPS))
            store = {}
            for g in GAPS:
                anc = make_anchors(g)
                times = np.array([a[0] for a in anc])
                fns = {"accel b=0": lambda t: np.zeros(3),
                       "accel + const(train, mocap windows)": lambda t: b_train_mocap,
                       "accel + const(all-time mocap) [UB]": lambda t: b_all_mocap,
                       "accel + windowed oracle (mocap, NON-causal) [UB]": bias_or}
                for cn, (T, B) in cand.items():
                    fns[cn] = (lambda t, T=T, B=B: b_at(T, B, t))
                errs = predict_positions(run, c, anc, gyro_c, d["accel"], trk_v, d["lever"], vel_fn, g, fns)
                store[g] = (times, errs)
            # choose causal candidate on 20-60 s (mean over gaps 0.25,0.5,1 of median error)
            def sel(cn):
                sc = []
                for g in (0.25, 0.5, 1.0):
                    times, errs = store[g]; m = (times >= sel_lo) & (times < split) & np.isfinite(errs[cn])
                    sc.append(np.median(errs[cn][m]) if m.sum() > 10 else np.inf)
                return np.mean(sc)
            best = min(cand, key=sel)
            show = ["naive", "accel b=0", "accel + const(train, mocap windows)", best, "accel + const(all-time mocap) [UB]", "accel + windowed oracle (mocap, NON-causal) [UB]"]
            for nm in show:
                cells = []
                for g in GAPS:
                    times, errs = store[g]; m = (times >= split) & np.isfinite(errs[nm])
                    e = errs[nm][m]
                    cells.append(f"{np.median(e):.3f}/{np.percentile(e,90):.3f}" if len(e) > 5 else "n/a")
                    rows.append([name, c, v0src, nm, g, int(m.sum()), round(float(np.median(e)), 4), round(float(np.percentile(e, 90)), 4), round(float(np.mean(e)), 4)])
                print(f"   {nm[:50]:50s}" + "".join(f"{x:>16s}" for x in cells))
            print("   paired mean-error difference vs 'accel b=0' (m), 95% block-bootstrap CI [negative = better]:")
            for nm in show[2:4]:
                line = []
                for g in (0.5, 1.0):
                    times, errs = store[g]; m = (times >= split) & np.isfinite(errs[nm]) & np.isfinite(errs["accel b=0"])
                    diff = errs[nm][m] - errs["accel b=0"][m]
                    lo, hi = block_bootstrap_ci(diff, times[m])
                    line.append(f"{g}s: {diff.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")
                print(f"     {nm[:50]:50s} " + "   ".join(line))
            print(f"   (best causal candidate on 20-60 s: {best})")
    with open(OUT_DIR / f"accel_payoff_{name}{LEVER_TAG}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["recording", "ctrl", "v0_source", "method", "gap_s", "n_test", "median_m", "p90_m", "mean_m"]); w.writerows(rows)
    print("saved")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "static_dark")
