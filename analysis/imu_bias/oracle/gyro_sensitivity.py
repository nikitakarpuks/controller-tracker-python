import sys
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/oracle")
from gyro_oracle import *
import gyro_oracle as go
recs = ["static_dark", "walk_dark", "static_easy"]
# (1) interval length sensitivity (tau) with K free per recording; (2) IMU timing shift sensitivity (+-4 ms) with tau=0.5
orig_load = go.load_imu
def shifted(delta_ms):
    def f(rdir, ctrl):
        t, g, a = orig_load(rdir, ctrl); return t + int(delta_ms * 1e6), g, a
    return f
print("bias (rad/s) with K free per recording, mean over the 3 recordings; entries = [bx by bz]")
for ctrl in CTRLS:
    for label, tau, delta in (("tau=0.25 shift 0", 0.25, 0), ("tau=0.5 shift 0", 0.5, 0), ("tau=1.0 shift 0", 1.0, 0), ("tau=0.5 shift -4ms", 0.5, -4), ("tau=0.5 shift +4ms", 0.5, +4), ("tau=0.5 shift +8ms", 0.5, +8)):
        go.load_imu = shifted(delta)
        B = []
        for n in recs:
            d = go.build(n, ctrl, tau, save=False); idx = np.arange(len(d["t0"])); B.append(go.solve(d, idx, with_K=True)[0])
        B = np.array(B); print(f"  {ctrl[:5]} {label:20s} mean {B.mean(0).round(4)}  range over recs x[{B[:,0].min():+.4f},{B[:,0].max():+.4f}] y[{B[:,1].min():+.4f},{B[:,1].max():+.4f}] z[{B[:,2].min():+.4f},{B[:,2].max():+.4f}]", flush=True)
