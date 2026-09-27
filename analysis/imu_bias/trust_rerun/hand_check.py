#!/usr/bin/env python3
"""Independent re-derivation of a few sweep rows (variant given on the command line) WITHOUT the headset-relative
wrapper: lift the mocap start state to the absolute frame by hand, propagate with integrate_gyro_segment (rotation)
and predict_world_pose (position) directly, compare with the absolute-frame mocap truth by hand, and check the
result equals the CSV row. Usage: python3 hand_check.py <variant> <recording_dir_name> <ctrl>"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trust_sweep as ts  # noqa: E402
from src.imu_data import integrate_gyro_segment, predict_world_pose  # noqa: E402
from src.mocap_data import DeviceMocap  # noqa: E402

variant_name, rec, ctrl = sys.argv[1:4]
V = ts.VARIANTS[variant_name]
df = pd.read_csv(Path(__file__).resolve().parent / "out" / variant_name / "imu_trust_all.csv")
df = df[(df.recording == rec) & (df.ctrl_name == ctrl)]

rec_dir = ts.RECORDINGS_ROOT / rec
cfg = ts.load_json_config(ts.CONFIG["controllers"][ctrl]["config_path"])
lag = ts._OLD_LAG_NS[ctrl] if V["lag"] == "old" else ts.controller_imu_lag_ns(ctrl)
t_imu, gyro, accel = ts.load_and_calibrate_controller_imu(rec_dir / "mav0" / ts._IMU_REL_PATH[ctrl], cfg, lag_ns=lag,
                                                          factory_corrected_input=V["factory_corrected"],
                                                          accel_scale=V["accel_scale"])
calib = ts.create_imu_calib_from_config(cfg)
lever = {"old": lambda: calib.accel.T_rt.compose(calib.gyro.T_rt.inverse()).t,
         "new": lambda: ts.accel_lever_arm_body(calib), "zero": lambda: np.zeros(3)}[V["lever"]]()
hm = ts.load_device_mocap(rec_dir, "headset")
cc = ts.CONFIG["controllers"][ctrl]
cm = (ts.load_device_mocap(rec_dir, ctrl, ts.load_vision_offset_ns(cc), ts.load_vision_drift_params(cc))
      if V["mocap_timing"] else ts.load_device_mocap(rec_dir, ctrl))
assert V["frame"] == "imu", "hand check implemented for the IMU-frame variants"


def T_world(dev, t):
    """absolute (mocap-world) IMU-frame pose from the raw trajectory, composed by hand: marker pose then T_imu_marker^-1."""
    m = dev.pose_at(t)
    from src.transformations import Transform
    return Transform(*m).compose(dev.T_imu_marker.inverse())


def check(anchor, dt):
    t0, t1 = int(anchor), int(anchor) + int(round(dt * 1e9))
    Wc0, Wc1 = T_world(cm, t0), T_world(cm, t1)          # controller IMU pose in mocap world at both ends
    R_rel = integrate_gyro_segment(t_imu, gyro, t0, t1)
    rot_hand = ts.rotation_angle_deg((Wc0.R @ R_rel).T @ Wc1.R)
    # position: start velocity by central difference of the ABSOLUTE trajectory (controller world velocity)
    h = int(ts.DEFAULT_EGO_MOTION_WINDOW_S * 1e9 / 2)
    v_w = (T_world(cm, t0 + h).t - T_world(cm, t0 - h).t) / ts.DEFAULT_EGO_MOTION_WINDOW_S
    pred = predict_world_pose(t_imu, gyro, t_imu, accel, ts.MOCAP_ROOM_G_WORLD, lever, t0, t1, Wc0.R, Wc0.t, v_w)
    pos_hand = float(np.linalg.norm(pred[1] - Wc1.t)) * 1000.0
    return rot_hand, pos_hand


for label, sel in (("calm (<60 deg/s)", df[df.peak_gyro_dps < 60]), ("fast (600-900 deg/s)", df[(df.peak_gyro_dps > 600) & (df.peak_gyro_dps < 900)])):
    for dt in (0.035, 0.3):
        s = sel[np.isclose(sel.dt_s, dt)]
        row = s.iloc[len(s) // 2]
        rot_h, pos_h = check(row.anchor_ts_ns, dt)
        print(f"{variant_name} {rec[-10:]} {ctrl} {label:22s} dt={dt:5.3f} anchor={int(row.anchor_ts_ns)} peak_gyro={row.peak_gyro_dps:6.1f}dps"
              f" | sweep rot {row.rot_err_deg:8.4f} deg  hand {rot_h:8.4f} deg | sweep pos {row.pos_err_mm:9.3f} mm  hand {pos_h:9.3f} mm"
              f" | dRot {abs(row.rot_err_deg - rot_h):.2e} dPos {abs(row.pos_err_mm - pos_h):.2e}")
