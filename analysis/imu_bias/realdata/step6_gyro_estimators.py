"""Step 6: causal running-average gyro-bias estimators (3 params), evaluated on held-out time.

Protocol (no peeking):
  * K (static scale/misalignment) is fit ONLY on the first 60 s (gyro regressor, vision steps) and optionally applied first.
  * Estimator hyper-parameter (tau) is chosen ONLY on anchors in [t0+20 s, t0+60 s]  (objective: mean over gaps {0.25,0.5,1.0} s of the
    MEDIAN rotation-prediction error with the causal bias available at the anchor).
  * Everything is REPORTED on anchors with t > t0+60 s.
  * Estimators only see steps <= current time (strictly causal; unit-tested).
Outputs: CSVs of per-method/per-gap errors, bias trajectories, and a summary printout."""
import sys, csv
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from common import *
from pipeline import *
from step3f_scale_misalign import fit_body
from step4a_payoff_KB import corrected

GAPS = [0.05, 0.25, 0.5, 1.0, 2.0]
SEL_GAPS = [0.25, 0.5, 1.0]
TAUS = [2, 5, 10, 20, 40, 80]


def anchors_for_gap(prep, gap_s, stride=3):
    """List of (a_idx, b_idx) strong-frame pairs ~gap_s apart (same for every method -> paired statistics)."""
    ts, ok = prep["ts"], prep["ok"]
    idx = np.flatnonzero(ok); tt = ts[idx]
    tol = max(0.006, 0.35 * gap_s) * 1e9
    out = []
    for k in range(0, len(idx), stride):
        a = idx[k]
        kk = np.searchsorted(tt, ts[a] + gap_s * 1e9)
        cand = [c for c in (kk - 1, kk) if 0 <= c < len(tt) and tt[c] > ts[a]]
        if not cand: continue
        c = min(cand, key=lambda c: abs(tt[c] - (ts[a] + gap_s * 1e9)))
        if abs(tt[c] - (ts[a] + gap_s * 1e9)) > tol: continue
        out.append((a, idx[c]))
    return out


def eval_pairs(run, ctrl, prep, pairs, gyro, bias_fn):
    d = run.ctrl[ctrl]; ts, R = prep["ts"], prep["R_wc"]
    errs = np.full(len(pairs), np.nan)
    for n, (a, b) in enumerate(pairs):
        Rg = integrate_gyro_segment(d["t"], gyro - bias_fn(ts[a]), int(ts[a]), int(ts[b]))
        if Rg is not None:
            errs[n] = rot_deg(Rg.T @ (R[a].T @ R[b]))
    return errs


def block_bootstrap_ci(diff, times_ns, block_s=2.0, n=2000, seed=0):
    """95% CI of mean(diff) resampling 2-s blocks (anchors are strongly autocorrelated)."""
    m = np.isfinite(diff); diff, t = diff[m], times_ns[m]
    blk = ((t - t.min()) / 1e9 // block_s).astype(int)
    ub = np.unique(blk); groups = [diff[blk == u] for u in ub]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        means.append(np.mean(np.concatenate([groups[i] for i in pick])))
    return np.percentile(means, [2.5, 97.5])


def make_estimators(tau, gate=None):
    return {
        f"feedbackEMA tau={tau}": FeedbackEMA(tau_s=tau),
        f"ratioEMA tau={tau}": RatioEMA(tau_s=tau),
        f"worldRLS tau={tau}": WorldRLS(tau_s=tau),
        f"worldRLS+rateGate tau={tau}": WorldRLS(tau_s=tau, weight=rate_weight(3.0)),
        f"windowMedian W={tau}": WindowMedian(window_s=tau),
    }


def main(name, ego="mocap"):
    run = Run(name)
    all_rows = []
    summary = {}
    for c in CTRLS:
        d = run.ctrl[c]
        prep = prepare(run, c, ego=ego)
        t0 = prep["ts"][prep["ok"]][0]; split = t0 + int(60e9); sel_lo = t0 + int(20e9); t_end = prep["ts"][-1]
        st0, _ = make_steps(run, c, prep)
        beta, _, _ = fit_body([s for s in st0 if s["t"] < split], key="omega_g")
        K = beta[3:].reshape(3, 3)
        variants = {"rawgyro": d["gyro"], "gyro+K": corrected(d["gyro"], np.zeros(3), K)}
        pairs = {g: anchors_for_gap(prep, g) for g in GAPS}
        ts = prep["ts"]
        # mocap oracle (K-corrected residuals for the K variant) for bias-error metric
        for vname, gy in variants.items():
            steps, n_out = make_steps(run, c, prep, gyro=gy)
            st_m = mocap_steps(run, c, gyro=gy)
            tt_all = np.array([s["t"] for s in st_m])
            def oracle_at(t, W=20):
                b = window_bias(st_m, t - int(W / 2 * 1e9), t + int(W / 2 * 1e9))
                return np.zeros(3) if b is None else b
            methods = {"zero": ZeroBias()}
            for tau in TAUS:
                methods.update(make_estimators(tau))
            traj = {}
            for mname, est in methods.items():
                T, B = run_estimator(est, steps)
                traj[mname] = (T, B)
            # constant references
            b_train = window_bias(steps, t0, split)
            b_all_mocap = window_bias(st_m, t0, t_end)
            traj["const(train vision fit)"] = (np.array([0]), np.array([b_train]))
            traj["const(oracle all-time, mocap) [upper bound]"] = (np.array([0]), np.array([b_all_mocap]))
            # windowed non-causal mocap oracle (20 s centered) as an ideal time-varying reference
            cen = np.arange(t0, t_end, int(2e9))
            To = np.array(cen); Bo = np.array([oracle_at(t) for t in cen])
            traj["windowed oracle (mocap 20s centred, NON-causal) [upper bound]"] = (To, Bo)
            # evaluate each method on selection (20-60s) and test (>60s) anchors
            res = {}
            for mname, (T, B) in traj.items():
                if mname.startswith("windowed oracle"):
                    fn = lambda t, T=T, B=B: B[np.clip(np.searchsorted(T, t), 0, len(T) - 1)]
                else:
                    fn = lambda t, T=T, B=B: b_at(T, B, t)
                per_gap = {}
                for g in GAPS:
                    errs = eval_pairs(run, c, prep, pairs[g], gy, fn)
                    per_gap[g] = errs
                res[mname] = per_gap
            times = {g: np.array([ts[a] for a, _ in pairs[g]]) for g in GAPS}
            # pick tau per family on selection window
            fams = {}
            for mname in res:
                fam = mname.split(" tau=")[0].split(" W=")[0]
                fams.setdefault(fam, []).append(mname)
            chosen = {}
            for fam, names in fams.items():
                if fam in ("zero",) or fam.startswith("const") or fam.startswith("windowed oracle"):
                    chosen[fam] = names[0]; continue
                def sel_score(mn):
                    sc = []
                    for g in SEL_GAPS:
                        m = (times[g] >= sel_lo) & (times[g] < split) & np.isfinite(res[mn][g])
                        sc.append(np.median(res[mn][g][m]))
                    return np.mean(sc)
                chosen[fam] = min(names, key=sel_score)
            print(f"\n===== [{name}/{c}] variant={vname}  ego={ego}   K diag {np.round(np.diag(K),4)}   steps {len(steps)} (outlier-gated {n_out})")
            print("   chosen hyper-parameters on 20-60 s:", {f: m.split(' ', 1)[1] if ' ' in m else '' for f, m in chosen.items() if f not in ('zero',)})
            print(f"   TEST (t>60 s)  median / p90 error (deg)      gap(s): " + "".join(f"{g:>14.2f}" for g in GAPS))
            zero_res = res["zero"]
            for fam in ["zero"] + [f for f in fams if f != "zero"]:
                mn = chosen[fam]
                cells = []
                for g in GAPS:
                    m = (times[g] >= split) & np.isfinite(res[mn][g])
                    e = res[mn][g][m]
                    cells.append(f"{np.median(e):.2f}/{np.percentile(e,90):.2f}")
                    all_rows.append([name, c, vname, ego, fam, mn, g, int(m.sum()), round(float(np.median(e)), 4), round(float(np.percentile(e, 90)), 4), round(float(np.mean(e)), 4)])
                print(f"   {fam[:44]:44s}" + "".join(f"{x:>14s}" for x in cells))
            # paired CI of mean-error difference vs zero, for chosen estimators at 0.5 s and 1 s
            print("   paired mean-error difference vs zero (deg), 95% block-bootstrap CI  [negative = better]")
            for fam in [f for f in fams if f != "zero"]:
                mn = chosen[fam]; line = []
                for g in (0.5, 1.0, 2.0):
                    m = (times[g] >= split) & np.isfinite(res[mn][g]) & np.isfinite(zero_res[g])
                    diff = res[mn][g][m] - zero_res[g][m]
                    lo, hi = block_bootstrap_ci(diff, times[g][m])
                    line.append(f"{g}s: {diff.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")
                print(f"     {fam[:44]:44s} " + "   ".join(line))
            # bias-estimate error vs the mocap oracle (20 s windows) over the test period + jitter
            print("   bias-estimate vs mocap oracle (test period): rms error (rad/s), mean(estimate), jitter = rms of 1-s change (rad/s)")
            tq = np.arange(split + int(10e9), t_end - int(10e9), int(2e9))
            Bo_q = np.array([oracle_at(t, 20) for t in tq])
            for fam in [f for f in fams if not f.startswith("windowed") and f != "zero"]:
                mn = chosen[fam]; T, B = traj[mn]
                Bq = np.array([b_at(T, B, t) for t in tq])
                rms = np.sqrt(np.mean(np.sum((Bq - Bo_q) ** 2, axis=1)))
                jit = np.sqrt(np.mean(np.sum(np.diff(Bq, axis=0) ** 2, axis=1))) if len(Bq) > 2 else np.nan
                print(f"     {fam[:44]:44s} rms err {rms:.4f}   mean est {np.round(Bq.mean(0),4)}   mean oracle {np.round(Bo_q.mean(0),4)}   jitter/2s {jit:.4f}")
            # dump chosen trajectories
            for fam in [f for f in fams if not f.startswith("windowed") and not f.startswith("const") and f != "zero"]:
                T, B = traj[chosen[fam]]
                with open(OUT_DIR / f"traj_{name}_{c}_{vname}_{ego}_{fam.replace(' ', '_').replace('+','_')}.csv", "w", newline="") as f:
                    w = csv.writer(f); w.writerow(["t_ns", "t_s", "bx", "by", "bz"])
                    for t, b in list(zip(T, B))[::10]:
                        w.writerow([int(t), round((t - t0) / 1e9, 3), *[round(float(x), 6) for x in b]])
    out = OUT_DIR / f"gyro_estimators_{name}_{ego}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["recording", "ctrl", "gyro_variant", "ego", "family", "chosen", "gap_s", "n_test", "median_deg", "p90_deg", "mean_deg"]); w.writerows(all_rows)
    print("\nsaved", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "static_dark", sys.argv[2] if len(sys.argv) > 2 else "mocap")
