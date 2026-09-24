import sys, time
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
from gyro_oracle import *
for tau in (0.5,):
    for name in REC_NAMES:
        for ctrl in CTRLS:
            t0 = time.time()
            try:
                d = build(name, ctrl, tau)
                print(f"{name}/{ctrl} tau={tau}: n={len(d['t0'])} ({time.time()-t0:.0f}s)", flush=True)
            except Exception as ex:
                print(f"{name}/{ctrl} FAILED: {ex!r}", flush=True)
