#!/usr/bin/env python3
"""check_motion_stats.py -- how often are the controller IMUs (near-)stationary in the 8 recordings?
Decides whether stationary/ZUPT-style bias updates are even possible in-session.
Reads mav0/imu1|imu2/data.csv (raw EuRoC CSV) only. 0.5 s windows (100 samples @ ~200 Hz), stride 0.25 s."""
import glob
import numpy as np

ROOT = "/home/nikitakarpuks/Downloads/recordings-aug26"
print(f"{'recording':14s}{'ctrl':6s}{'dur_s':>7s}  gyro max-axis std per 0.5 s window [rad/s]: p1 / p5 / p25 / p50   | frac windows std<0.02 / <0.05 rad/s")
for rec in sorted(glob.glob(ROOT + "/euroc_recording_*")):
    name = rec.split("_", 3)[-1]
    for side, imu in (("left", "imu1"), ("right", "imu2")):
        a = np.loadtxt(f"{rec}/mav0/{imu}/data.csv", delimiter=",", skiprows=1)
        g = a[:, 1:4]; n = 100
        s = np.array([g[i:i + n].std(0).max() for i in range(0, len(a) - n, 50)])
        dur = (a[-1, 0] - a[0, 0]) / 1e9
        p = np.percentile(s, [1, 5, 25, 50])
        print(f"{name:14s}{side:6s}{dur:7.1f}  {p[0]:.3f} / {p[1]:.3f} / {p[2]:.3f} / {p[3]:.3f}   | {np.mean(s < 0.02):.4f} / {np.mean(s < 0.05):.4f}")
