import sys
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
from common import *
from scipy.ndimage import uniform_filter1d
rows = []
for name in REC_NAMES:
    rdir = rec_dir(name)
    for ctrl in CTRLS:
        dev = load_mocap_device(rdir, ctrl); mo = MocapOrientation(dev)
        t = dev.t_ns.astype(np.float64)                     # mocap track times (data.csv time base)
        R = Rotation.from_quat(dev.quat_xyzw.astype(np.float64))
        h = 6                                                # +-6 samples ~ +-50 ms
        Rm = (R[:-2*h].inv() * R[2*h:]).as_rotvec() / ((t[2*h:] - t[:-2*h])[:, None] / 1e9)
        om = np.linalg.norm(Rm, axis=1)
        v = np.linalg.norm((dev.position[2*h:] - dev.position[:-2*h]) / ((t[2*h:] - t[:-2*h])[:, None] / 1e9), axis=1)
        tt = (t[h:-h] - t[0]) / 1e9
        rest = (om < 0.3) & (v < 0.05)
        # longest contiguous rest run
        idx = np.flatnonzero(np.diff(np.concatenate(([0], rest.view(np.int8), [0]))))
        runs = [(tt[a], tt[b-1], tt[b-1]-tt[a]) for a, b in zip(idx[::2], idx[1::2])]
        runs = sorted(runs, key=lambda x: -x[2])[:3]
        print(f"{name:14s}{ctrl[:5]}: dur {tt[-1]:.0f}s omega median {np.median(om):.2f} p90 {np.percentile(om,90):.2f} rad/s | speed median {np.median(v):.2f} m/s | rest frac {rest.mean()*100:.1f}% | longest rest runs (start,end,len s): {[tuple(round(x,1) for x in r) for r in runs]}")
