"""Step 7: headset IMU (imu0) gyro vs mocap headset rotation: bias magnitude, drift, headset motion level; and the effect on the
controller-bias estimate when ego-motion comes from imu0 (raw / oracle-bias-corrected) instead of mocap."""
import sys
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from common import *
from pipeline import *
name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
run = Run(name)
t0i, g0, a0 = run.imu0
# headset steps on the controller vision timestamp grid (dt<=50ms)
ts = run.ctrl["left_controller"]["vision"].ts
Rh = [run.R_wh(t) for t in ts]
ok = np.array([r is not None for r in Rh])
idx = np.flatnonzero(ok)
st = []
for a, b in zip(idx[:-1], idx[1:]):
    dt = (ts[b] - ts[a]) / 1e9
    if dt <= 0 or dt > 0.05: continue
    Rg = integrate_gyro_segment(t0i, g0, int(ts[a]), int(ts[b]))
    if Rg is None: continue
    dR = Rh[a].T @ Rh[b]
    e0 = Rotation.from_matrix(Rg.T @ dR).as_rotvec()
    if np.linalg.norm(e0) > np.radians(3): continue
    om = Rotation.from_matrix(dR).as_rotvec() / dt
    st.append(dict(t=ts[b], dt=dt, e0=e0, R_end=Rh[b], omega=om, omega_g=Rotation.from_matrix(Rg).as_rotvec()/dt, rate=np.linalg.norm(om)))
rates = np.array([s["rate"] for s in st])
print(f"[{name}] headset steps {len(st)}  median |omega| {np.median(rates):.3f} rad/s  p90 {np.percentile(rates,90):.3f}  ({np.degrees(np.median(rates)):.1f} deg/s)")
print(f"   per-step |e0| median {np.degrees(np.median([np.linalg.norm(s['e0']) for s in st])):.4f} deg")
tt = np.array([s["t"] for s in st]); t_first = tt[0]
for W in (10, 30, 60):
    cen = np.arange(tt[0] + int(W/2*1e9), tt[-1] - int(W/2*1e9), int(10e9))
    bs = np.array([b for b in (window_bias(st, c - int(W/2*1e9), c + int(W/2*1e9)) for c in cen) if b is not None])
    tsec = (cen[:len(bs)] - tt[0]) / 1e9
    print(f"   imu0 gyro bias (vs mocap), W={W:2d}s: mean {np.round(bs.mean(0),5)}  std over time {np.round(bs.std(0),5)}  trend {np.round([np.polyfit(tsec,bs[:,k],1)[0]*60 for k in range(3)],5)} rad/s/min")
b_all = window_bias(st, tt[0], tt[-1] + 1)
b_first5 = window_bias(st, tt[0], tt[0] + int(5e9))
raw_mean_first5 = g0[(t0i >= tt[0]) & (t0i < tt[0] + int(5e9))].mean(0)
print(f"   whole-recording bias (mocap regression) {np.round(b_all,5)}; first-5 s regression {None if b_first5 is None else np.round(b_first5,5)}; raw mean gyro first 5 s {np.round(raw_mean_first5,5)}")
# effect on controller-bias estimation
for c in CTRLS:
    d = run.ctrl[c]
    out = {}
    for lab, kw in (("ego=mocap", dict(ego="mocap")), ("ego=imu0 (raw, no bias corr.)", dict(ego="imu0")),
                    ("ego=imu0 (oracle bias removed)", dict(ego="imu0", imu0_bias=b_all)),
                    ("ego=none (rig frame, no ego removal)", dict(ego="none"))):
        prep = prepare(run, c, **kw); steps, n_out = make_steps(run, c, prep)
        t_lo, t_hi = steps[0]["t"], steps[-1]["t"] + 1
        b = window_bias(steps, t_lo, t_hi)
        e_med = np.degrees(np.median([np.linalg.norm(s["e0"]) for s in steps]))
        out[lab] = b
        print(f"   [{c}] {lab:38s}: whole-recording controller bias {np.round(b,5)} rad/s   per-step |e0| median {e_med:.3f} deg  ({n_out} gated)")
