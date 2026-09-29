"""Step 5: after removing the static gyro scale/misalignment K (fit on first 60 s, gyro regressor), how large and how variable is
the REMAINING gyro bias in windows? (mocap oracle and vision), and is there any drift/trend? Also S(T) scaling restored?"""
import sys
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from common import *
from pipeline import *
from step3f_scale_misalign import fit_body
from step4a_payoff_KB import corrected
name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
run = Run(name)
for c in CTRLS:
    d = run.ctrl[c]; prep = prepare(run, c, ego="mocap"); st_v0, _ = make_steps(run, c, prep)
    t0 = prep["ts"][prep["ok"]][0]; split = t0 + int(60e9)
    beta, _, _ = fit_body([s for s in st_v0 if s["t"] < split], key="omega_g")
    K = beta[3:].reshape(3, 3)
    g_K = corrected(d["gyro"], np.zeros(3), K)
    st_v, _ = make_steps(run, c, prep, gyro=g_K); st_m = mocap_steps(run, c, gyro=g_K); st_m0 = mocap_steps(run, c)
    print(f"\n[{name}/{c}]  K fit on first 60 s:  diag {np.round(np.diag(K),4)}")
    # S(T) scaling with K removed
    def S_T(st, Ts=(0.25, 0.5, 1, 2, 5, 10, 20)):
        tt = np.array([s["t"] for s in st]); W = np.array([-(s["R_end"] @ s["e0"]) for s in st]); cs = np.vstack([np.zeros(3), np.cumsum(W, 0)])
        out = []
        for T in Ts:
            vals = []
            for k in range(0, len(st), 25):
                hi = np.searchsorted(tt, tt[k] + int(T * 1e9))
                if hi >= len(st): break
                if (tt[hi] - tt[k]) / 1e9 > 1.2 * T: continue
                vals.append(np.linalg.norm(cs[hi + 1] - cs[k]))
            out.append(np.sqrt(np.mean(np.square(vals))))
        return np.round(out, 4)
    print("   S(T) T=(0.25,0.5,1,2,5,10,20)s  vision  raw:", S_T(st_v0), "  K-removed:", S_T(st_v))
    print("                                    mocap   raw:", S_T(st_m0), "  K-removed:", S_T(st_m))
    for lab, st in (("mocap", st_m), ("vision", st_v)):
        tt = np.array([s["t"] for s in st])
        for W in (10, 20, 40):
            cen = np.arange(tt[0] + int(W / 2 * 1e9), tt[-1] - int(W / 2 * 1e9), int(4e9))
            bs = np.array([b for b in (window_bias(st, cc - int(W / 2 * 1e9), cc + int(W / 2 * 1e9)) for cc in cen) if b is not None])
            tsec = (cen[:len(bs)] - tt[0]) / 1e9
            tr = [np.polyfit(tsec, bs[:, k], 1)[0] * 60 for k in range(3)]
            print(f"   {lab:6s} K-removed windowed bias, W={W:2d}s: mean {np.round(bs.mean(0),4)}  std over time {np.round(bs.std(0),4)} rad/s  trend {np.round(tr,4)} rad/s/min  (n={len(bs)})")
