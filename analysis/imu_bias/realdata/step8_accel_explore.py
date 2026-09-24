"""Step 8a: accelerometer data reality check for static_dark.
 (1) motion levels (mocap speed/rate) & rest intervals; (2) raw accel magnitude vs gravity; (3) literal per-frame position/velocity-residual
 SNR for an accel bias; (4) gravity-consistency measurement noise."""
import sys
import numpy as np
sys.path.insert(0, "/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/analysis/imu_bias/realdata")
from common import *
from pipeline import *
from src.imu_data import integrate_accel_to_position, integrate_accel_segment, MOCAP_ROOM_G_WORLD
name = sys.argv[1] if len(sys.argv) > 1 else "static_dark"
run = Run(name)
G = MOCAP_ROOM_G_WORLD
for c in CTRLS:
    d = run.ctrl[c]; v = d["vision"]; ts = v.ts
    print(f"\n[{name}/{c}] factory accel bias0 (T=0, applied) {np.round(d['calib'].accel.bias0,4)} m/s^2; lever arm {np.round(d['lever']*1000,1)} mm")
    # mocap world poses of controller IMU (mocap frame) -> use bridge-composed LED-frame orientation and position of the LED frame origin
    Tm = []
    for t in ts:
        T = world_pose(d["mocap"], int(t))
        Tm.append(None if T is None else T.compose(d["bridge"].inverse()))
    ok = np.array([T is not None for T in Tm]); idx = np.flatnonzero(ok)
    P = np.array([Tm[i].t for i in idx]); Rm = np.array([Tm[i].R for i in idx]); tm = ts[idx]
    # speed and angular rate (central differences over ~4 frames)
    sp, om = [], []
    for k in range(2, len(idx) - 2):
        dt = (tm[k+2] - tm[k-2]) / 1e9
        if dt <= 0 or dt > 0.12: sp.append(np.nan); om.append(np.nan); continue
        sp.append(np.linalg.norm(P[k+2] - P[k-2]) / dt); om.append(rot_deg(Rm[k-2].T @ Rm[k+2]) / dt)
    sp, om = np.array(sp), np.array(om)
    print(f"   controller mocap speed: median {np.nanmedian(sp):.2f} m/s p90 {np.nanpercentile(sp,90):.2f}; rate median {np.nanmedian(om):.0f} deg/s;  fraction 'at rest' (speed<0.03 m/s & rate<3 deg/s): {np.mean((sp<0.03)&(om<3)):.4f}")
    # raw accel magnitude
    an = np.linalg.norm(d["accel"], axis=1)
    print(f"   accel |a| (factory-corrected): median {np.median(an):.3f} p5 {np.percentile(an,5):.2f} p95 {np.percentile(an,95):.2f} m/s^2 (gravity 9.81)")
    # literal per-step position-residual SNR
    prep = prepare(run, c, ego="mocap"); okv = prep["ok"]; iv = np.flatnonzero(okv)
    ptrue = []
    # absolute positions: p_w = R_wh p_hc + p_wh (mocap headset)
    Pw = {}
    for i in iv:
        Th = run.T_wh(ts[i])
        if Th is not None: Pw[i] = Th.R @ v.p[i] + Th.t
    meas = []
    for a, b in zip(iv[:-2], iv[1:-1]):
        # need frames a-1 (for v0), a, b consecutive: use (i-1,i,i+1) triples with dt<=25ms each
        pass
    trip = [(iv[k-1], iv[k], iv[k+1]) for k in range(1, len(iv)-1) if iv[k-1] in Pw and iv[k] in Pw and iv[k+1] in Pw]
    m_list = []
    for (i0, i1, i2) in trip[::3]:
        dt01 = (ts[i1]-ts[i0])/1e9; dt12 = (ts[i2]-ts[i1])/1e9
        if dt01 <= 0 or dt01 > 0.03 or dt12 <= 0 or dt12 > 0.03: continue
        v0 = (Pw[i1] - Pw[i0]) / dt01                                   # causal backward-difference velocity
        R0 = prep["R_wc"][i1]; R1 = prep["R_wc"][i2]
        dp = integrate_accel_to_position(d["t"], d["accel"], int(ts[i1]), int(ts[i2]), R0, R1, v0, G, t_gyro=d["t"], gyro_body=d["gyro"], r=d["lever"])
        if dp is None: continue
        p_pred = Pw[i1] + dp
        r = Pw[i2] - p_pred                                             # position residual (world)
        m_list.append(dict(r=r, dt=dt12, R=R1, resid_norm=np.linalg.norm(r)))
    rn = np.array([m["resid_norm"] for m in m_list]); dts = np.array([m["dt"] for m in m_list])
    # accel-bias effect per step is 0.5 b dt^2 ; measurement m = -2 r/dt^2 (in world), rotate to body: R^T
    mb = np.array([-2 * (m["R"].T @ m["r"]) / m["dt"] ** 2 for m in m_list])
    print(f"   literal position-residual scheme: n={len(m_list)} steps; |pos residual| median {np.median(rn)*1000:.2f} mm (accel-bias 0.15 m/s^2 would add {0.5*0.15*np.median(dts)**2*1000:.4f} mm)")
    print(f"     per-frame b_a measurement -2R^T r/dt^2: mean {np.round(mb.mean(0),2)}  std {np.round(mb.std(0),1)} m/s^2  -> frames needed for 0.05 m/s^2 accuracy if iid: {int((mb.std(0).mean()/0.05)**2)}")
