"""Step 3: mocap-derived gyro-bias oracle b_or(t) (centered windows, non-causal) for both controllers, window-size
sensitivity, comparison with the vision-only centered-window estimate, and factory-bias context."""
import sys, csv
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from common import *
from pipeline import *

name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
run = Run(name)
rows = []
for c in CTRLS:
    d = run.ctrl[c]
    print(f"\n[{name}/{c}] factory gyro bias0 (T=0, already applied by the loader, sensor frame) = {np.round(d['calib'].gyro.bias0,5)} rad/s")
    for W in (5, 10, 20):
        t_or, b_or, n = oracle_bias(run, c, window_s=W, hop_s=2.0)
        print(f"   mocap oracle, window {W:2d}s: n_windows {len(t_or)}  mean {np.round(b_or.mean(0),5)}  std over time {np.round(b_or.std(0),5)}  "
              f"min {np.round(b_or.min(0),4)} max {np.round(b_or.max(0),4)} rad/s")
    t_or, b_or, n = oracle_bias(run, c, window_s=10.0, hop_s=2.0)
    # first half vs second half, linear trend
    tsec = (t_or - t_or[0]) / 1e9
    for ax, an in enumerate("xyz"):
        p = np.polyfit(tsec, b_or[:, ax], 1)
        print(f"      axis {an}: first-60s mean {b_or[tsec<60, ax].mean():+.5f}  rest mean {b_or[tsec>=60, ax].mean():+.5f}   trend {p[0]*60:+.5f} rad/s per min")
    # vision-only centered-window estimate (world frame, strong frames, mocap ego) at the same centers
    prep = prepare(run, c, ego="mocap")
    steps, n_out = make_steps(run, c, prep)
    tt = np.array([s["t"] for s in steps])
    vb = []
    for cen in t_or:
        lo, hi = np.searchsorted(tt, cen - int(5e9)), np.searchsorted(tt, cen + int(5e9))
        A = np.zeros((3, 3)); S = np.zeros(3)
        for s in steps[lo:hi]:
            A += s["dt"] * s["R_end"]; S += -(s["R_end"] @ s["e0"])
        vb.append(np.linalg.solve(A + 1e-2 * np.eye(3), S))
    vb = np.array(vb)
    diff = vb - b_or
    print(f"   vision-only (centered 10s, strong frames, {len(steps)} steps, {n_out} outlier-gated) vs mocap oracle: "
          f"rms diff {np.round(np.sqrt((diff**2).mean(0)),5)} rad/s  mean diff {np.round(diff.mean(0),5)}   corr per axis "
          f"{[round(float(np.corrcoef(vb[:,k], b_or[:,k])[0,1]),3) for k in range(3)]}")
    for tsec_i, t in enumerate(t_or):
        rows.append([name, c, int(t), round((t - t_or[0]) / 1e9, 2), *[round(float(x), 6) for x in b_or[tsec_i]], *[round(float(x), 6) for x in vb[tsec_i]]])
out = OUT_DIR / f"oracle_gyro_bias_{name}.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["recording", "ctrl", "t_center_ns", "t_s", "or_x", "or_y", "or_z", "vis_x", "vis_y", "vis_z"]); w.writerows(rows)
print("saved", out)
