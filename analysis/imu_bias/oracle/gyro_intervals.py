"""Per-interval gyro-vs-mocap residuals (sensor frame) for one recording/controller -> intervals CSV.
e = Log(Rm^T Rg)/tau  ~= b + K*omega ;  omega = Log(Rm)/tau.   tau = 0.1 s, stride = tau (independent intervals)."""
import sys
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
from common import *
import csv

TAU = 0.1
def compute(name, ctrl, tau=TAU):
    rdir = rec_dir(name)
    t, gb, ab = load_imu(rdir, ctrl)
    gs = (DIAG_FLIP @ gb.T).T
    mo = MocapOrientation(load_mocap_device(rdir, ctrl))
    dtn = int(tau * 1e9)
    grid = np.arange(t[0] + 0.5e9, t[-1] - 0.5e9, dtn).astype(np.int64)
    ok = mo.valid(grid) & mo.valid(grid + dtn) & mo.valid(grid + dtn // 2)
    grid = grid[ok]
    R0 = mo.R_world_imu(grid); R1 = mo.R_world_imu(grid + dtn)
    rows = []
    for i, t0 in enumerate(grid):
        Rg = integrate_gyro_segment(t, gs, int(t0), int(t0) + dtn)
        if Rg is None: continue
        Rm = (R0[i].inv() * R1[i]).as_matrix()
        om = Rotation.from_matrix(Rm).as_rotvec() / tau
        e = Rotation.from_matrix(Rm.T @ Rg).as_rotvec() / tau
        rows.append((int(t0), *om, *e))
    return np.array(rows, dtype=np.float64)

if __name__ == "__main__":
    for name in (sys.argv[1:] or REC_NAMES):
        for ctrl in CTRLS:
            a = compute(name, ctrl)
            np.save(f"/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle/gyro_intervals_{name}_{ctrl}.npy", a)
            om = np.linalg.norm(a[:, 1:4], axis=1); e = a[:, 4:7]
            print(f"{name}/{ctrl}: n={len(a)} |omega| median {np.median(om):.2f} rad/s; e mean {e.mean(0).round(4)} median {np.median(e,0).round(4)} rad/s; n(|omega|<0.3)={int((om<0.3).sum())}", flush=True)
