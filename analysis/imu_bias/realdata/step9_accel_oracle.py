"""Step 9a: accelerometer bias (and scale/misalignment) from windows. Oracle = controller MOCAP track; estimator input = VISION track
(mocap headset ego, dev-only). Windows of W seconds, non-overlapping. Reports: window bias series stats; and HELD-OUT velocity
residual  ||Q_w - M_w b - F_w K||  (m/s per window, lower is better) with (b,K) fit on the first 60 s only."""
import sys, csv
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from accel_tools import *
from pipeline import *
name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
run = Run(name)
rows = []
for c in CTRLS:
    d = run.ctrl[c]
    prep = prepare(run, c, ego="mocap")
    trk_v = track_vision(run, c, prep); trk_m = track_mocap(run, c)
    t0 = prep["ts"][prep["ok"]][0]; split = t0 + int(60e9); t_end = prep["ts"][-1]
    print(f"\n[{name}/{c}] factory accel bias0 {np.round(d['calib'].accel.bias0,4)}")
    for src, trk in (("mocap", trk_m), ("vision", trk_v)):
        AW = AccelWindows(run, c, trk)
        for W in (5, 10, 20):
            comp = []
            for a in np.arange(t0 + int(3e9), t_end - int(W * 1e9), int(W * 1e9)):
                r = AW.components(int(a), int(a + W * 1e9))
                if r is not None and r[3] > 0.7:
                    comp.append((int(a), *r))
            if not comp: continue
            bs = np.array([np.linalg.solve(x[2] + 1e-3 * np.eye(3), x[1]) for x in comp])
            tsec = np.array([(x[0] - t0) / 1e9 for x in comp])
            tr = [np.polyfit(tsec, bs[:, k], 1)[0] * 60 for k in range(3)] if len(bs) > 3 else [np.nan]*3
            print(f"   {src:6s} W={W:2d}s  windows {len(comp):3d}  per-window bias: mean {np.round(bs.mean(0),3)}  std {np.round(bs.std(0),3)} m/s^2  trend {np.round(tr,3)} /min   (per-window noise std would be ~{0.1*1.414/W:.3f} from end velocities)")
            for x, b in zip(comp, bs):
                rows.append([name, c, src, W, x[0], round((x[0]-t0)/1e9, 2), *[round(float(v), 5) for v in b]])
            # held-out velocity-residual test: fit on first 60 s windows
            tr_idx = [i for i, x in enumerate(comp) if x[0] + W * 1e9 <= split]; te_idx = [i for i, x in enumerate(comp) if x[0] >= split]
            if len(tr_idx) < 4 or len(te_idx) < 3: continue
            def stack(idx, use_b, use_K):
                X, y = [], []
                for i in idx:
                    Q, M, F = comp[i][1], comp[i][2], comp[i][3]
                    blocks = []
                    if use_b: blocks.append(M)
                    if use_K: blocks.append(F)
                    X.append(np.hstack(blocks)); y.append(Q)
                return np.vstack(X), np.concatenate(y)
            out = {}
            for lab, ub, uk in (("zero", False, False), ("b only", True, False), ("K only", False, True), ("b+K", True, True)):
                if not ub and not uk:
                    beta = None
                else:
                    Xtr, ytr = stack(tr_idx, ub, uk)
                    # ridge for stability
                    beta = np.linalg.solve(Xtr.T @ Xtr + 1e-3 * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
                Xte, yte = stack(te_idx, ub or uk and False, False) if False else (None, None)
                res = []
                for i in te_idx:
                    Q, M, F = comp[i][1], comp[i][2], comp[i][3]
                    pred = np.zeros(3)
                    if beta is not None:
                        k = 0
                        if ub: pred += M @ beta[:3]; k = 3
                        if uk: pred += F @ beta[k:k + 9]
                    res.append(np.linalg.norm(Q - pred))
                out[lab] = (np.median(res), np.mean(res))
            print(f"          held-out velocity residual per window [m/s] (median/mean):  " + "   ".join(f"{k}: {v[0]:.3f}/{v[1]:.3f}" for k, v in out.items()))
with open(OUT_DIR / f"accel_windows_{name}{LEVER_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["recording", "ctrl", "source", "W_s", "t_start_ns", "t_s", "bx", "by", "bz"]); w.writerows(rows)
print("saved")
