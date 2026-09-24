"""Step 4b: errors-in-variables check. Fit e0 = -dt (b + K w) with w = VISION/MOCAP-derived rate (noisy, correlated with e0's own noise)
versus w = GYRO-integrated rate (independent of the reference's noise). A real scale/misalignment gives the same K either way;
an artifact does not. Also fit on first 60 s and evaluate prediction on the rest for the gyro-regressor K."""
import sys
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from common import *
from pipeline import *
from step3f_scale_misalign import fit_body
from step4a_payoff_KB import gap_err_gyro, corrected, GAPS   # noqa
name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
run = Run(name)
for c in CTRLS:
    d = run.ctrl[c]; prep = prepare(run, c, ego="mocap"); st_v, _ = make_steps(run, c, prep); st_m = mocap_steps(run, c)
    print(f"\n[{name}/{c}]")
    res = {}
    for lab, st in (("vision", st_v), ("mocap", st_m)):
        for key in ("omega", "omega_g"):
            beta, se, _ = fit_body(st, key=key); res[(lab, key)] = beta
            K = beta[3:].reshape(3, 3)
            print(f"   ref={lab:6s} regressor={key:8s}: b={np.round(beta[:3],4)}  diag(K)={np.round(np.diag(K),4)}  offdiag rms={np.sqrt((K-np.diag(np.diag(K)))[~np.eye(3,dtype=bool)].var()+0):.4f}  |K|max={np.abs(K).max():.4f}")
    Kv, Km = res[("vision","omega_g")][3:], res[("mocap","omega_g")][3:]
    print(f"   K(gyro regressor) agreement vision vs mocap: corr {np.corrcoef(Kv,Km)[0,1]:.3f}  max|dK| {np.abs(Kv-Km).max():.4f}")
    t0 = prep["ts"][prep["ok"]][0]; split = t0 + int(60e9)
    tr = [s for s in st_v if s["t"] < split]
    beta, se, _ = fit_body(tr, key="omega_g")
    bK, K = beta[:3], beta[3:].reshape(3, 3)
    print(f"   fit on first 60s (gyro regressor): b={np.round(bK,4)}  K=\n{np.round(K,4)}")
    variants = {"zero": d["gyro"], "K only (gyro reg.)": corrected(d["gyro"], np.zeros(3), K), "b + K (gyro reg.)": corrected(d["gyro"], bK, K)}
    print("   TEST t>60s median/p90 error (deg):   gap(s): " + "".join(f"{g:>15.2f}" for g in GAPS))
    for lab, gy in variants.items():
        cells = []
        for g in GAPS:
            e = gap_err_gyro(run, c, prep, gy, g, split, prep["ts"][-1])
            cells.append(f"{np.median(e):.2f}/{np.percentile(e,90):.2f}")
        print(f"   {lab:22s}" + "".join(f"{x:>15s}" for x in cells))
