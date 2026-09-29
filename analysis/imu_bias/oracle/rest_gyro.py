"""Rest-interval gyro bias (independent of K / integration model): mean(gyro_sensor) - omega_mocap over runs where mocap says rest."""
import sys, csv
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
from common import *
OUT = "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle/"

def find_rest_runs(dev, min_len_s=1.0, om_thr=0.15, v_thr=0.03, h=6):
    t = dev.t_ns.astype(np.float64); R = Rotation.from_quat(dev.quat_xyzw.astype(np.float64))
    dtv = ((t[2*h:] - t[:-2*h]) / 1e9)[:, None]
    om = np.linalg.norm((R[:-2*h].inv() * R[2*h:]).as_rotvec() / dtv, axis=1)
    v = np.linalg.norm((dev.position[2*h:] - dev.position[:-2*h]) / dtv, axis=1)
    rest = (om < om_thr) & (v < v_thr); tc = t[h:-h]
    idx = np.flatnonzero(np.diff(np.concatenate(([0], rest.view(np.int8), [0]))))
    runs = []
    for a, b in zip(idx[::2], idx[1::2]):
        if (tc[b-1] - tc[a]) / 1e9 >= min_len_s: runs.append((tc[a] + 0.2e9, tc[b-1] - 0.2e9))   # shrink 0.2 s each side (filter edges)
    return runs

def analyse(name, ctrl):
    rdir = rec_dir(name); dev = load_mocap_device(rdir, ctrl); mo = MocapOrientation(dev)
    # runs found in mocap-track time (data.csv base == device clock + coarse); convert to camera-clock query time by inverting the lookup shift
    t, gb, ab = load_imu(rdir, ctrl); gs = (DIAG_FLIP @ gb.T).T
    out = []
    for (a, b) in find_rest_runs(dev, min_len_s=1.0):
        # mocap time -> IMU (camera-clock) time: solve lookup_times(q) = m  (shift is ~const over a run)
        shift = mo.lookup_times(np.array([a]))[0] - a
        qa, qb = a - shift, b - shift
        sel = (t >= qa) & (t <= qb)
        if sel.sum() < 100: continue
        g = gs[sel]; n = sel.sum()
        # mocap slope: regress Log(R0^T R(t)) vs time over the run
        qs = np.linspace(qa, qb, 60); Rr = mo.R_world_imu(qs); rv = (Rr[0].inv() * Rr).as_rotvec()
        tt = (qs - qs[0]) / 1e9
        om_m = np.array([np.polyfit(tt, rv[:, i], 1)[0] for i in range(3)])
        # SEM: gyro white-noise sigma / sqrt(n)
        sem = g.std(0) / np.sqrt(n)
        out.append(dict(rec=name, ctrl=ctrl, t_start_s=(a - dev.t_ns[0]) / 1e9, t_end_s=(b - dev.t_ns[0]) / 1e9, n=int(n),
                        **{f"b_{ax}": (g.mean(0) - om_m)[i] for i, ax in enumerate("xyz")}, **{f"sem_{ax}": sem[i] for i, ax in enumerate("xyz")},
                        **{f"gyro_sd_{ax}": g.std(0)[i] for i, ax in enumerate("xyz")}, **{f"om_m_{ax}": om_m[i] for i, ax in enumerate("xyz")}))
    return out

if __name__ == "__main__":
    allrows = []
    for name in REC_NAMES:
        for ctrl in CTRLS:
            rows = analyse(name, ctrl); allrows += rows
            for r in rows:
                print(f"{name:14s}{ctrl[:5]} t=[{r['t_start_s']:6.1f},{r['t_end_s']:6.1f}]s n={r['n']:4d}  b_rest=[{r['b_x']:+.4f} {r['b_y']:+.4f} {r['b_z']:+.4f}] rad/s  sem~{r['sem_x']:.4f}  |om_mocap|={np.linalg.norm([r['om_m_x'],r['om_m_y'],r['om_m_z']]):.4f}")
    with open(OUT + "gyro_bias_rest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
