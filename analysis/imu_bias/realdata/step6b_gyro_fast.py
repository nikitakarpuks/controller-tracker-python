"""Step 6b: fast, complete gyro-bias estimator evaluation (same protocol as step6, O(1) per method via fastgap).

Usage: python3 step6b_gyro_fast.py <recording> [ego=mocap|imu0] [K_source=own60|<recording>|none]
Protocol (no peeking): K fit ONLY on the first 60 s of the K_source recording (own60 = this recording's first 60 s); estimator tau chosen ONLY
on anchors in [t0+20 s, t0+60 s]; report on anchors with t>t0+60 s. Gap classes: fixed gaps {0.05,0.25,0.5,1,2} s and REAL tracking-loss gaps
(consecutive strong frames 0.15-1.5 s apart). Estimators see steps <= now only. Paired 95% block-bootstrap CIs vs zero."""
import sys, csv
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from common import *
from pipeline import *
from fastgap import integrate_with_C, fast_error_deg
from step3f_scale_misalign import fit_body
from step4a_payoff_KB import corrected
from step6_gyro_estimators import block_bootstrap_ci

GAPS = [0.05, 0.25, 0.5, 1.0, 2.0]
SEL = [0.25, 0.5, 1.0]
TAUS = [2, 5, 10, 20, 40, 80, 160]


def gap_pairs(prep, gap_s, stride=2):
    ts, ok = prep["ts"], prep["ok"]; idx = np.flatnonzero(ok); tt = ts[idx]
    tol = max(0.006, 0.35 * gap_s) * 1e9; out = []
    for k in range(0, len(idx), stride):
        a = idx[k]; kk = np.searchsorted(tt, ts[a] + gap_s * 1e9)
        cs = [c for c in (kk - 1, kk) if 0 <= c < len(tt) and tt[c] > ts[a]]
        if not cs: continue
        c = min(cs, key=lambda c: abs(tt[c] - (ts[a] + gap_s * 1e9)))
        if abs(tt[c] - (ts[a] + gap_s * 1e9)) <= tol: out.append((a, idx[c]))
    return out


def real_gap_pairs(prep, lo=0.15, hi=1.5):
    ts, ok = prep["ts"], prep["ok"]; idx = np.flatnonzero(ok)
    return [(a, b) for a, b in zip(idx[:-1], idx[1:]) if lo < (ts[b] - ts[a]) / 1e9 <= hi]


class Cache:
    """Per (gyro variant, pair-list): dR0, C, dR_v, anchor time  -> evaluate any constant bias in O(1)."""
    def __init__(self, run, ctrl, prep, gyro, pairs):
        d = run.ctrl[ctrl]; ts, R = prep["ts"], prep["R_wc"]
        self.items = []
        for a, b in pairs:
            dR0, C = integrate_with_C(d["t"], gyro, int(ts[a]), int(ts[b]))
            self.items.append(None if dR0 is None else (dR0, C, R[a].T @ R[b]))
        self.times = np.array([ts[a] for a, _ in pairs]); self.dur = np.array([(ts[b] - ts[a]) / 1e9 for a, b in pairs])

    def errors(self, bias_fn):
        out = np.full(len(self.items), np.nan)
        for n, it in enumerate(self.items):
            if it is not None:
                out[n] = fast_error_deg(it[0], it[1], it[2], bias_fn(self.times[n]))
        return out


def main(name, ego="mocap", K_source="own60", tag=""):
    run = Run(name); rows = []
    for c in CTRLS:
        d = run.ctrl[c]
        imu0_bias = None
        if ego == "imu0":                                    # headset gyro bias: first-10 s regression vs mocap (dev-only calibration step)
            t0i, g0, _ = run.imu0; ts_v = d["vision"].ts; prep_m = prepare(run, c, ego="mocap")
            tt0 = prep_m["ts"][prep_m["ok"]][0]
            Rh = [run.R_wh(t) for t in ts_v]; okh = [r is not None for r in Rh]; ii = [i for i in range(len(ts_v)) if okh[i] and ts_v[i] < tt0 + int(10e9)]
            stt = []
            for a_, b_ in zip(ii[:-1], ii[1:]):
                dt = (ts_v[b_] - ts_v[a_]) / 1e9
                if 0 < dt <= 0.05:
                    Rg = integrate_gyro_segment(t0i, g0, int(ts_v[a_]), int(ts_v[b_]))
                    if Rg is not None:
                        stt.append(dict(t=ts_v[b_], dt=dt, e0=Rotation.from_matrix(Rg.T @ (Rh[a_].T @ Rh[b_])).as_rotvec(), R_end=Rh[b_]))
            imu0_bias = window_bias(stt, stt[0]["t"], stt[-1]["t"] + 1)
        prep = prepare(run, c, ego=ego, imu0_bias=imu0_bias)
        t0 = prep["ts"][prep["ok"]][0]; split = t0 + int(60e9); sel_lo = t0 + int(20e9); t_end = prep["ts"][-1]
        st0, _ = make_steps(run, c, prep)
        if K_source == "none":
            K = np.zeros((3, 3))
        elif K_source == "own60":
            beta, _, _ = fit_body([s for s in st0 if s["t"] < split], key="omega_g"); K = beta[3:].reshape(3, 3)
        else:                                                # K from another recording (whole recording, mocap ego) -- offline calibration
            r2 = Run(K_source); p2 = prepare(r2, c, ego="mocap"); s2, _ = make_steps(r2, c, p2)
            beta, _, _ = fit_body(s2, key="omega_g"); K = beta[3:].reshape(3, 3)
        variants = {"rawgyro": d["gyro"]}
        if K_source != "none":
            variants["gyro+K"] = corrected(d["gyro"], np.zeros(3), K)
        pairs = {g: gap_pairs(prep, g) for g in GAPS}; pairs["real"] = real_gap_pairs(prep)
        ts = prep["ts"]
        for vname, gy in variants.items():
            steps, n_out = make_steps(run, c, prep, gyro=gy)
            st_m = mocap_steps(run, c, gyro=gy)
            caches = {g: Cache(run, c, prep, gy, pairs[g]) for g in pairs}
            def oracle_at(t, W=20):
                b = window_bias(st_m, t - int(W / 2 * 1e9), t + int(W / 2 * 1e9)); return np.zeros(3) if b is None else b
            methods = {"zero": ZeroBias()}
            for tau in TAUS:
                methods[f"feedbackEMA tau={tau}"] = FeedbackEMA(tau_s=tau); methods[f"ratioEMA tau={tau}"] = RatioEMA(tau_s=tau)
                methods[f"worldRLS tau={tau}"] = WorldRLS(tau_s=tau); methods[f"worldRLS+rateGate tau={tau}"] = WorldRLS(tau_s=tau, weight=rate_weight(3.0))
                methods[f"worldRLS(strongPrior) tau={tau}"] = WorldRLS(tau_s=tau, sigma_b=0.01, sigma_s=0.05)   # conservative: shrinks toward 0 when A is poorly conditioned / S noisy
                methods[f"windowMedian W={tau}"] = WindowMedian(window_s=min(tau, 40))
            traj = {}
            for mn, est in methods.items(): traj[mn] = run_estimator(est, steps)
            b_train = window_bias(steps, t0, split); b_all = window_bias(st_m, t0, t_end)
            traj["const(train vision fit)"] = (np.array([0]), np.array([b_train])); traj["const(all-time mocap) [UB]"] = (np.array([0]), np.array([b_all]))
            cen = np.arange(t0, t_end, int(2e9)); Bo = np.array([oracle_at(t) for t in cen])
            traj["windowed oracle (mocap 20s, NON-causal) [UB]"] = (cen, Bo)
            res = {}
            for mn, (T, B) in traj.items():
                if mn.startswith("windowed oracle"): fn = lambda t, T=T, B=B: B[np.clip(np.searchsorted(T, t), 0, len(T) - 1)]
                else: fn = lambda t, T=T, B=B: b_at(T, B, t)
                res[mn] = {g: caches[g].errors(fn) for g in caches}
            times = {g: caches[g].times for g in caches}
            oracle_ok = (name == "static_dark")          # mocap-orientation oracle validated only where mocap timing/drift was tuned (static_dark); walk_medium oracle is garbage
            if not oracle_ok:
                for k in [k for k in res if k.startswith("windowed oracle") or k.startswith("const(all-time")]: del res[k]
            fams = {}
            for mn in res: fams.setdefault(mn.split(" tau=")[0].split(" W=")[0], []).append(mn)
            chosen = {}
            for fam, names in fams.items():
                if fam == "zero" or fam.startswith("const") or fam.startswith("windowed"): chosen[fam] = names[0]; continue
                def score(mn):
                    sc = []
                    for g in SEL:
                        m = (times[g] >= sel_lo) & (times[g] < split) & np.isfinite(res[mn][g]); sc.append(np.median(res[mn][g][m]))
                    return np.mean(sc)
                chosen[fam] = min(names, key=score)
            print(f"\n===== [{name}/{c}] gyro={vname} ego={ego} K_source={K_source}  K diag {np.round(np.diag(K),4)}  steps {len(steps)} (gated {n_out})  imu0_bias {None if imu0_bias is None else np.round(imu0_bias,4)}")
            print("   chosen (20-60 s):", {f: chosen[f].split(' ', 1)[1] for f in chosen if ' ' in chosen[f] and (' tau=' in chosen[f] or ' W=' in chosen[f])})
            cls = GAPS + ["real"]
            print("   TEST t>60 s median/p90 error (deg)      gap(s): " + "".join(f"{str(g):>14s}" for g in cls))
            for fam in ["zero"] + [f for f in fams if f != "zero"]:
                mn = chosen[fam]; cells = []
                for g in cls:
                    m = (times[g] >= split) & np.isfinite(res[mn][g]); e = res[mn][g][m]
                    cells.append(f"{np.median(e):.2f}/{np.percentile(e,90):.2f}" if len(e) > 5 else "n/a")
                    if len(e) > 5: rows.append([name, c, vname, ego, K_source, fam, mn, g, int(m.sum()), round(float(np.median(e)), 4), round(float(np.percentile(e, 90)), 4), round(float(np.mean(e)), 4)])
                print(f"   {fam[:44]:44s}" + "".join(f"{x:>14s}" for x in cells))
            print("   paired mean-error difference vs zero (deg) [95%% CI], negative = better; n(real gaps test) = %d" % int(((times['real'] >= split) & np.isfinite(res['zero']['real'])).sum()))
            for fam in [f for f in fams if f != "zero"]:
                mn = chosen[fam]; line = []
                for g in (1.0, 2.0, "real"):
                    m = (times[g] >= split) & np.isfinite(res[mn][g]) & np.isfinite(res["zero"][g])
                    if m.sum() < 8: continue
                    diff = res[mn][g][m] - res["zero"][g][m]; lo, hi = block_bootstrap_ci(diff, times[g][m]); line.append(f"{g}: {diff.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")
                print(f"     {fam[:44]:44s} " + "   ".join(line))
            tq = np.arange(split + int(10e9), t_end - int(10e9), int(2e9)); Bo_q = np.array([oracle_at(t, 20) for t in tq])
            print("   bias estimate vs mocap oracle (test period): rms err (rad/s) | mean estimate | mean oracle | jitter/2s" + ("" if oracle_ok else "   [SKIPPED: mocap oracle unreliable on this recording]"))
            for fam in [f for f in fams if not f.startswith("windowed") and f != "zero" and oracle_ok]:
                T, B = traj[chosen[fam]]; Bq = np.array([b_at(T, B, t) for t in tq])
                rms = np.sqrt(np.mean(np.sum((Bq - Bo_q) ** 2, axis=1))); jit = np.sqrt(np.mean(np.sum(np.diff(Bq, axis=0) ** 2, axis=1)))
                print(f"     {fam[:44]:44s} {rms:.4f} | {np.round(Bq.mean(0),4)} | {np.round(Bo_q.mean(0),4)} | {jit:.4f}")
            for fam in [f for f in fams if f.startswith("worldRLS") or f.startswith("feedback") or f.startswith("ratio")]:
                T, B = traj[chosen[fam]]
                with open(OUT_DIR / f"traj_{name}_{c}_{vname}_{ego}_{K_source}_{fam.replace(' ', '_').replace('+', '_')}{tag}.csv", "w", newline="") as f:
                    w = csv.writer(f); w.writerow(["t_ns", "t_s", "bx", "by", "bz"])
                    for t, b in list(zip(T, B))[::10]: w.writerow([int(t), round((t - t0) / 1e9, 3), *[round(float(x), 6) for x in b]])
    out = OUT_DIR / f"gyro_estimators_fast_{name}_{ego}_{K_source}{tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["recording", "ctrl", "gyro_variant", "ego", "K_source", "family", "chosen", "gap", "n_test", "median_deg", "p90_deg", "mean_deg"]); w.writerows(rows)
    print("\nsaved", out)


if __name__ == "__main__":
    a = sys.argv
    main(a[1] if len(a) > 1 else "static_dark", a[2] if len(a) > 2 else "mocap", a[3] if len(a) > 3 else "own60")
