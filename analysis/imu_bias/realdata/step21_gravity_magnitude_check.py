"""Step 21: independent physical check of accel scale. On samples with LOW dynamics per MOCAP (|omega|<w_max, |a_world|<a_max, from mocap position 2nd difference over 40 ms)
the specific force magnitude must be g = 9.81 (lever-arm terms ~ 0 there). Report mean |f| (factory-corrected, lever-corrected) and its ratio to 9.81, per controller,
for progressively stricter 'quiet' thresholds; also the ratio along each body axis is NOT tested here (needs orientation diversity) -- this is only the isotropic magnitude."""
import os, sys
os.environ["LEVER"] = "bridge"
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from accel_tools import *
from src.imu_data import _lever_arm_correction
from pipeline import *
for name in ("static_dark", "walk_medium"):
    run = Run(name)
    for c in CTRLS:
        d = run.ctrl[c]; trk = track_mocap(run, c)
        t = d["t"]; m = trk.valid(t)
        # mocap-derived angular speed (rate) and world acceleration at IMU samples via central differences of mocap frames (+-40 ms)
        idx = np.flatnonzero(m)
        tt = t[idx]
        Rm = trk.R_at(tt); h = int(40e6)
        R_p = trk.R_at(np.clip(tt + h, trk.ts[0], trk.ts[-1])); R_m = trk.R_at(np.clip(tt - h, trk.ts[0], trk.ts[-1]))
        om = np.array([rot_deg(a.T @ b) for a, b in zip(R_m, R_p)]) * np.pi / 180 / 0.08
        P = lambda tq: np.array([np.interp(tq, trk.ts, trk.P[:, k]) for k in range(3)]).T
        aw = (P(tt + h) - 2 * P(tt) + P(tt - h)) / (0.04 ** 2)
        lev = _lever_arm_correction(t, t, d["gyro"], d["lever"])
        fc = (d["accel"] - lev)[idx]
        fn = np.linalg.norm(fc, axis=1)
        print(f"\n[{name}/{c}] |f_c| (bridge lever) on mocap-quiet samples   [g=9.81]")
        for wmax, amax in ((1.0, 1.0), (0.6, 0.6), (0.4, 0.4), (0.3, 0.3)):
            q = (om < wmax) & (np.linalg.norm(aw, axis=1) < amax)
            if q.sum() < 50: print(f"   |w|<{wmax} rad/s, |a|<{amax} m/s^2: n={q.sum()} (too few)"); continue
            print(f"   |w|<{wmax} rad/s, |a|<{amax} m/s^2: n={q.sum():5d} ({100*q.mean():.1f}%)  mean |f_c| {fn[q].mean():.3f}  median {np.median(fn[q]):.3f}   ratio to g {fn[q].mean()/9.81:.4f}  (expected 1.000 if factory scale right; fitted K_a diag ~1.02-1.035)")
