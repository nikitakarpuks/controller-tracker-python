"""Sanity gate 1: factory-corrected gyro (zero extra bias) vs mocap relative rotation over 0.1 s intervals.
Tests which frame map between mocap-IMU frame and gyro_body reproduces the data (H_sensor: conjugate by _DIAG_FLIP;
H_identity: mocap-IMU frame == body frame)."""
import sys
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
from common import *
name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
rdir = rec_dir(name)
D = DIAG_FLIP
for ctrl in CTRLS:
    t, gb, ab = load_imu(rdir, ctrl)
    mo = MocapOrientation(load_mocap_device(rdir, ctrl))
    grid = np.arange(t[0] + 2e9, t[-1] - 2e9, 0.1e9).astype(np.int64)
    ok = mo.valid(grid) & mo.valid(grid + int(0.1e9))
    grid = grid[ok]
    R0 = mo.R_world_imu(grid).as_matrix(); R1 = mo.R_world_imu(grid + int(0.1e9)).as_matrix()
    res = {"sensor": [], "identity": []}; mag = []
    for i, t0 in enumerate(grid):
        Rg = integrate_gyro_segment(t, gb, int(t0), int(t0) + int(0.1e9))
        if Rg is None: continue
        Rm = R0[i].T @ R1[i]
        mag.append(np.degrees(np.linalg.norm(Rotation.from_matrix(Rm).as_rotvec())))
        for k, Rg_k in (("sensor", D @ Rg @ D), ("identity", Rg)):
            res[k].append(np.degrees(np.linalg.norm(Rotation.from_matrix(Rm.T @ Rg_k).as_rotvec())))
    mag = np.array(mag)
    print(f"{name}/{ctrl}: n={len(mag)} intervals, mocap rotation/0.1s median {np.median(mag):.2f} deg (p90 {np.percentile(mag,90):.2f})")
    for k, v in res.items():
        v = np.array(v); print(f"   H_{k:8s}: median residual {np.median(v):.3f} deg/0.1s  p90 {np.percentile(v,90):.3f}  ({np.median(v)/0.1:.2f} deg/s)")
