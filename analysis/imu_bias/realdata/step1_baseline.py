"""Step 1: reproduce the known-good baseline. Zero EXTRA bias (factory T=0 correction only, as loaded by
load_and_calibrate_controller_imu). Per consecutive strong vision-frame pair, compare gyro-integrated relative
rotation (controller body frame) with the vision relative rotation lifted to the inertial (mocap-world) frame
with the headset's mocap orientation. README finding 9 expects median ~0.34 deg (left) / ~0.51 deg (right)."""
import sys
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from common import *

name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
run = Run(name)
for c in CTRLS:
    d = run.ctrl[c]; v = d["vision"]
    strong = v.strong_mask()
    ts = v.ts
    Rw = [None] * len(ts)
    for i, t in enumerate(ts):
        Rh = run.R_wh(t)
        Rw[i] = None if Rh is None else Rh @ v.R[i]
    ok = strong & np.array([r is not None for r in Rw])
    pairs = step_pairs(ts, ok)
    errs, dts, rates = [], [], []
    errs_noego = []
    for i, j in pairs:
        Rg = integrate_gyro_segment(d["t"], d["gyro"], int(ts[i]), int(ts[j]))
        if Rg is None:
            continue
        dR_v = Rw[i].T @ Rw[j]
        errs.append(rot_deg(Rg.T @ dR_v))
        dR_rig = v.R[i].T @ v.R[j]   # no headset ego-motion removal (what a naive rig-frame comparison would do)
        errs_noego.append(rot_deg(Rg.T @ dR_rig))
        dts.append((ts[j] - ts[i]) / 1e9)
        rates.append(rot_deg(dR_v) / dts[-1])
    errs = np.array(errs); errs_noego = np.array(errs_noego); rates = np.array(rates)
    print(f"[{name}/{c}] strong frames {strong.sum()}/{len(ts)}  pairs {len(errs)}  "
          f"median dt {np.median(dts)*1e3:.1f}ms  median rate {np.median(rates):.1f} deg/s")
    print(f"   gyro-vs-vision rotation error (mocap headset ego-motion removed): median {np.median(errs):.3f} deg  "
          f"p90 {np.percentile(errs, 90):.3f}  p95 {np.percentile(errs, 95):.3f}")
    print(f"   same WITHOUT headset ego-motion removal: median {np.median(errs_noego):.3f}  p95 {np.percentile(errs_noego, 95):.3f}")
