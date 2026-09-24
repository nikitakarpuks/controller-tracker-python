import sys, pickle, time
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
from accel_oracle import *
for name in REC_NAMES:
    for ctrl in CTRLS:
        t0 = time.time()
        try:
            wins = run(name, ctrl, 2.0, False)
            pickle.dump(wins, open(OUT + f"accel_windows_{name}_{ctrl}.pkl", "wb"))
            print(f"{name}/{ctrl}: {len(wins)} windows ({time.time()-t0:.0f}s)", flush=True)
        except Exception as ex:
            print(f"{name}/{ctrl} FAILED {ex!r}", flush=True)
