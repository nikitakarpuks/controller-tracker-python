"""Step 18: accel bias variability after the static terms are removed. Lever arm = bridge; K_a fixed from the ALL-time mocap fit (b,K_a); then per-window b
(mocap track, 10 s windows, hop 5 s) = M^-1 (Q - F K_a). Reports mean, std over time, linear trend, and the expected pure-noise std for context."""
import sys
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from accel_tools import *
from pipeline import *
from step3f_scale_misalign import fit_body
from step4a_payoff_KB import corrected
from step10b_accel_K import fit_bK
assert LEVER_MODE == "bridge"
name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
run = Run(name)
for c in CTRLS:
    d = run.ctrl[c]; prep = prepare(run, c, ego="mocap"); trk_m = track_mocap(run, c)
    t0 = prep["ts"][prep["ok"]][0]; t_end = prep["ts"][-1]
    st0, _ = make_steps(run, c, prep); bk, _, _ = fit_body([s for s in st0 if s["t"] < t0 + int(60e9)], key="omega_g")
    gyro_c = corrected(d["gyro"], np.zeros(3), bk[3:].reshape(3, 3)); AW = AccelWindows(run, c, trk_m, gyro=gyro_c)
    b_all, K_all, n = fit_bK(AW, t0, t_end, 5, True, True)
    print(f"\n[{name}/{c}] all-time (b,K_a) fit ({n} windows): b={np.round(b_all,3)} m/s^2  diag(K_a)={np.round(np.diag(K_all),4)}  offdiag max {np.abs(K_all-np.diag(np.diag(K_all))).max():.4f}")
    for W in (10, 20):
        bs, ts = [], []
        for a in np.arange(t0 + int(2e9), t_end - int(W * 1e9), int(5e9)):
            r = AW.components(int(a), int(a + W * 1e9))
            if r is None or r[3] < 0.6: continue
            Q, M, F = r[0], r[1], r[2]
            bs.append(np.linalg.solve(M + 1e-3 * np.eye(3), Q - F @ K_all.ravel())); ts.append((a - t0) / 1e9)
        bs = np.array(bs); ts = np.array(ts)
        tr = [np.polyfit(ts, bs[:, k], 1)[0] * 60 for k in range(3)]
        print(f"   W={W:2d}s: {len(bs)} windows  mean b {np.round(bs.mean(0),3)}  std over time {np.round(bs.std(0),3)}  trend {np.round(tr,3)} m/s^2 per min   (pure-noise std expected ~{0.1*1.414/W:.3f})")
