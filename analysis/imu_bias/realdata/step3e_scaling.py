"""Step 3e: scaling of the accumulated WORLD-frame residual S(T) = sum_{window T} -R_j e0_j with window length T.
 - telescoping endpoint noise -> S(T) saturates;  - constant body bias b -> S ~ |A b| ~ T*|b|;  - random walk -> sqrt(T).
Also the 'implied bias' |S|/T. Done for mocap-orientation steps and vision steps; a straight-line 'reference' shows what a
0.01 rad/s bias would add."""
import sys
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from common import *
from pipeline import *
name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
run = Run(name)
for c in CTRLS:
    prep = prepare(run, c, ego="mocap"); st_v, _ = make_steps(run, c, prep)
    st_m = mocap_steps(run, c)
    print(f"\n[{name}/{c}]   window T (s): rms |S(T)| rad   [implied bias |S|/T rad/s]   for mocap steps | vision steps")
    for T in (0.25, 0.5, 1, 2, 5, 10, 20):
        row = []
        for st in (st_m, st_v):
            tt = np.array([s["t"] for s in st]); W = np.array([-(s["R_end"] @ s["e0"]) for s in st])
            cs = np.vstack([np.zeros(3), np.cumsum(W, 0)])
            vals = []
            for k in range(0, len(st), 25):
                hi = np.searchsorted(tt, tt[k] + int(T * 1e9))
                if hi >= len(st): break
                # require the window to be dense (no big vision gaps)
                if (tt[hi] - tt[k]) / 1e9 > 1.2 * T: continue
                vals.append(np.linalg.norm(cs[hi + 1] - cs[k]))
            row.append((np.sqrt(np.mean(np.square(vals))) if vals else np.nan, len(vals)))
        print(f"   T={T:5.2f}: mocap {row[0][0]:.4f} [{row[0][0]/T:.4f}] (n={row[0][1]})   | vision {row[1][0]:.4f} [{row[1][0]/T:.4f}] (n={row[1][1]})   | 0.01 rad/s bias would add {0.01*T:.4f}")
