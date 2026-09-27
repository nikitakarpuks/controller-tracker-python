import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trust_sweep as ts
from src.imu_data import integrate_accel_to_position, integrate_gyro_segment, MOCAP_ROOM_G_WORLD
from src.transformations import Transform
from src.mocap_data import load_mocap_bridge
rec, ctrl = "euroc_recording_20260826173103_static_dark", "right_controller"
V = ts.VARIANTS["CLED"]
df = pd.read_csv(Path(__file__).resolve().parent/"out"/"CLED"/"imu_trust_all.csv"); df = df[(df.recording==rec)&(df.ctrl_name==ctrl)]
rd = ts.RECORDINGS_ROOT/rec
cfg = ts.load_json_config(ts.CONFIG["controllers"][ctrl]["config_path"])
t_imu,gyro,accel = ts.load_and_calibrate_controller_imu(rd/"mav0"/ts._IMU_REL_PATH[ctrl], cfg, lag_ns=ts.controller_imu_lag_ns(ctrl))
lever = ts.accel_lever_arm_body(ts.create_imu_calib_from_config(cfg))
hm = ts.load_device_mocap(rd,"headset"); cc = ts.CONFIG["controllers"][ctrl]
cm = ts.load_device_mocap(rd, ctrl, ts.load_vision_offset_ns(cc), ts.load_vision_drift_params(cc))
binv = load_mocap_bridge(str(ts.REPO/cc["mocap_bridge_path"].lstrip("./"))).inverse()
def Wled(t):  # controller LED-frame pose in mocap world
    m = cm.pose_at(t); return Transform(*m).compose(cm.T_imu_marker.inverse()).compose(binv)
h = int(ts.DEFAULT_EGO_MOTION_WINDOW_S*1e9/2)
for dt in (0.1, 0.3, 1.0):
    s = df[np.isclose(df.dt_s, dt) & (df.peak_accel_mps2 < 5)]
    s = s.iloc[::max(1, len(s)//120)]
    e_sweep, e_trueR, e_trueR_g0, e_nolever = [], [], [], []
    for _, r in s.iterrows():
        t0, t1 = int(r.anchor_ts_ns), int(r.anchor_ts_ns) + int(round(dt*1e9))
        W0, W1 = Wled(t0), Wled(t1)
        v0 = (Wled(t0+h).t - Wled(t0-h).t)/ts.DEFAULT_EGO_MOTION_WINDOW_S
        tt = np.linspace(t0, t1, 41); 
        # (1) truth rotation slerp endpoints (as integrate_accel_to_position does) vs gyro-integrated R1
        R1g = W0.R @ integrate_gyro_segment(t_imu, gyro, t0, t1)
        for lab, R1, lv, g, acc in (("trueR", W1.R, lever, MOCAP_ROOM_G_WORLD, None), ("gyroR", R1g, lever, MOCAP_ROOM_G_WORLD, None), ("nolever", W1.R, None, MOCAP_ROOM_G_WORLD, None)):
            dp = integrate_accel_to_position(t_imu, accel, t0, t1, W0.R, R1, v0, g, t_gyro=t_imu, gyro_body=gyro, r=lv)
            err = np.linalg.norm(W0.t + dp - W1.t)*1000
            {"trueR": e_trueR, "gyroR": e_trueR_g0, "nolever": e_nolever}[lab].append(err)
    print(f"dt={dt}: calm-accel windows n={len(e_trueR)}  median pos err mm | endpoint-slerp true R1: {np.median(e_trueR):8.1f} | gyro-integrated R1: {np.median(e_trueR_g0):8.1f} | true R1 & no lever: {np.median(e_nolever):8.1f} | sweep row: {s.pos_err_mm.median():8.1f}")

# --- dense-truth-rotation check: same accel, lever, gravity, v0; only R(t) differs (mocap at every accel sample)
from src.imu_data import _lever_arm_correction
print("\nDENSE true-R(t) integration vs endpoint-slerp (same accel stream/lever/g/v0), calm-accel windows:")
for dt in (0.3, 1.0):
    s = df[np.isclose(df.dt_s, dt) & (df.peak_accel_mps2 < 5)]; s = s.iloc[::max(1, len(s)//100)]
    e_dense, e_slerp, span = [], [], []
    for _, r in s.iterrows():
        t0, t1 = int(r.anchor_ts_ns), int(r.anchor_ts_ns) + int(round(dt*1e9))
        m = (t_imu > t0) & (t_imu < t1); ts_ = np.concatenate(([t0], t_imu[m], [t1])).astype(np.int64)
        acc = np.stack([np.interp(ts_, t_imu, accel[:, i]) for i in range(3)], 1) - _lever_arm_correction(ts_, t_imu, gyro, lever)
        try:
            Rs = np.stack([Wled(int(t)).R for t in ts_])
        except TypeError:
            continue   # mocap coverage gap inside the window -> skip
        a_w = np.einsum("nij,nj->ni", Rs, acc) + MOCAP_ROOM_G_WORLD
        W0, W1 = Wled(t0), Wled(t1); v = (Wled(t0+h).t - Wled(t0-h).t)/ts.DEFAULT_EGO_MOTION_WINDOW_S
        p = W0.t.copy(); vv = v.copy()
        for i in range(len(ts_)-1):
            d = (ts_[i+1]-ts_[i])/1e9; v2 = vv + 0.5*(a_w[i]+a_w[i+1])*d; p = p + 0.5*(vv+v2)*d; vv = v2
        e_dense.append(np.linalg.norm(p - W1.t)*1000)
        e_slerp.append(r.pos_err_mm); span.append(ts.rotation_angle_deg(W0.R.T @ W1.R))
    if not e_dense: print(f"  dt={dt}: no gap-free windows"); continue
    print(f"  dt={dt}: dense-R(t) median {np.median(e_dense):8.1f} mm | endpoint-slerp (sweep) median {np.median(e_slerp):8.1f} mm | median end-to-end rotation {np.median(span):5.1f} deg")
