#!/usr/bin/env python3
"""Parametrised copy of the repo-root imu_trust_analysis.py (the sweep behind the thesis's coast-trust
figure/table) so the IMU pipeline variants can be compared like-for-like. The ORIGINAL scripts are untouched.

Method (unchanged from the original): real mocap as the causal start state at anchors every 0.1 s, blind
dead-reckoning with src.imu_data.predict_headset_relative_pose (headset ego-motion from mocap,
MOCAP_ROOM_G_WORLD), compared with mocap truth at t0+dt for 12 dt values 11 ms..1 s.

Variants (dict keys): factory_corrected (loader flag), accel_scale, lever ('old' = raw factory t as main.py
used, 'new' = -R^T t body frame, 'zero'), lag ('old' -5/-7 ms constants, 'new' = -mocap_vision_offset_ns),
mocap_timing (controller DeviceMocap gets vision_offset_ns + drift, headset stays 0 as in main.py),
frame ('imu' = truth in the mocap accelerometer-IMU frame, exactly like the original sweep; 'led' = truth
converted to the LED/vision frame with the shipped bridge, T_led = T_imu . bridge^-1, i.e. the frame the live
filter's state and lever arm actually live in).
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nikitakarpuks/PyCharmProjects/controller-tracker-python")
os.chdir(REPO)   # compare_vision_mocap reads config/config.yml relative to cwd at import
sys.path.insert(0, str(REPO))
from compare_vision_mocap import rotation_angle_deg  # noqa: E402
from src.imu_data import (load_and_calibrate_controller_imu, create_imu_calib_from_config,  # noqa: E402
                           predict_headset_relative_pose, peak_gyro_accel_over_window, MOCAP_ROOM_G_WORLD,
                           accel_lever_arm_body, ACCEL_DRIVER_SCALE)
from src.mocap_data import (load_mocap_csv, load_mocap_fine_offset_ns, load_T_imu_marker, DeviceMocap,  # noqa: E402
                             relative_pose, world_pose, headset_angular_velocity, headset_linear_velocity,
                             DRIFT_CHECK_VARIANT, DEFAULT_EGO_MOTION_WINDOW_S, controller_imu_lag_ns,
                             load_vision_offset_ns, load_vision_drift_params, load_mocap_bridge)
from src.load_config import load_yaml_config, load_json_config  # noqa: E402

RECORDINGS_ROOT = Path("/home/nikitakarpuks/Downloads/recordings-aug26")
CONFIG = load_yaml_config(str(REPO / "config" / "config.yml"))
_MOCAP_DISK_NAMES = {"headset": "headset", "left_controller": "ctrlleft", "right_controller": "ctrlright"}
_MOCAP_CALIB_FILES = {
    "headset": CONFIG["cameras"]["mocap_calib_path"],
    "left_controller": CONFIG["controllers"]["left_controller"]["mocap_calib_path"],
    "right_controller": CONFIG["controllers"]["right_controller"]["mocap_calib_path"],
}
_IMU_REL_PATH = {"left_controller": "imu1/data.csv", "right_controller": "imu2/data.csv"}
_OLD_LAG_NS = {"left_controller": -5_000_000, "right_controller": -7_000_000}

ANCHOR_STRIDE_S = 0.10
DT_VALUES_S = [0.011, 0.022, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]

_NEW = dict(factory_corrected=True, accel_scale=ACCEL_DRIVER_SCALE, lever="new")
VARIANTS = {
    "old":  dict(factory_corrected=False, accel_scale=1.0, lever="old", lag="old", mocap_timing=False, frame="imu"),
    "A":    dict(**_NEW, lag="old", mocap_timing=False, frame="imu"),
    "B":    dict(**_NEW, lag="new", mocap_timing=False, frame="imu"),
    "C":    dict(**_NEW, lag="new", mocap_timing=True, frame="imu"),
    # C with lever arm zero: the physically consistent choice when the propagated state IS the accelerometer point
    "C0":   dict(factory_corrected=True, accel_scale=ACCEL_DRIVER_SCALE, lever="zero", lag="new", mocap_timing=True, frame="imu"),
    # production-faithful: state and lever arm in the LED frame (truth via the shipped bridge)
    "CLED": dict(**_NEW, lag="new", mocap_timing=True, frame="led"),
    "ALED": dict(**_NEW, lag="old", mocap_timing=False, frame="led"),   # (a) loader+lever only, correct frame
    "BLED": dict(**_NEW, lag="new", mocap_timing=False, frame="led"),   # (b) + new lag, correct frame
    # same frame but with the OLD everything else except timing/frame fix, to separate frame effect on old constants
    "OLDLED": dict(factory_corrected=False, accel_scale=1.0, lever="old", lag="old", mocap_timing=False, frame="led"),
}


def load_device_mocap(rec_dir: Path, device_key: str, vision_offset_ns=0.0, drift=(0.0, 0.0)) -> DeviceMocap:
    device_dir = rec_dir / "mocap_filtered" / _MOCAP_DISK_NAMES[device_key]
    t, position, quat_xyzw = load_mocap_csv(device_dir / "data.csv")
    fine = load_mocap_fine_offset_ns(device_dir / "drift_check" / DRIFT_CHECK_VARIANT / "drift_check.json")
    T_imu_marker = load_T_imu_marker(_MOCAP_CALIB_FILES[device_key])
    return DeviceMocap(t, position, quat_xyzw, fine, T_imu_marker, vision_offset_ns=vision_offset_ns,
                       drift_offset_ns=drift[0], drift_rate_ns_per_ns=drift[1])


def analyze(rec_dir: Path, ctrl_name: str, variant: dict, anchor_stride_s: float = ANCHOR_STRIDE_S,
            dt_values=DT_VALUES_S):
    mav0 = rec_dir / "mav0"
    ctrl_json_cfg = load_json_config(CONFIG["controllers"][ctrl_name]["config_path"])
    lag_ns = _OLD_LAG_NS[ctrl_name] if variant["lag"] == "old" else controller_imu_lag_ns(ctrl_name)
    t_imu, gyro_body, accel_body = load_and_calibrate_controller_imu(
        mav0 / _IMU_REL_PATH[ctrl_name], ctrl_json_cfg, lag_ns=lag_ns,
        factory_corrected_input=variant["factory_corrected"], accel_scale=variant["accel_scale"])
    calib = create_imu_calib_from_config(ctrl_json_cfg)
    if variant["lever"] == "old":
        lever_arm = calib.accel.T_rt.compose(calib.gyro.T_rt.inverse()).t
    elif variant["lever"] == "new":
        lever_arm = accel_lever_arm_body(calib)
    else:
        lever_arm = np.zeros(3)

    headset_mocap = load_device_mocap(rec_dir, "headset")
    cc = CONFIG["controllers"][ctrl_name]
    if variant["mocap_timing"]:
        ctrl_mocap = load_device_mocap(rec_dir, ctrl_name, load_vision_offset_ns(cc), load_vision_drift_params(cc))
    else:
        ctrl_mocap = load_device_mocap(rec_dir, ctrl_name)
    bridge_inv = None
    if variant["frame"] == "led":
        bridge_inv = load_mocap_bridge(str(REPO / cc["mocap_bridge_path"].lstrip("./"))).inverse()

    def T_hc(t):
        T = relative_pose(headset_mocap, ctrl_mocap, t)
        if T is None or bridge_inv is None:
            return T
        return T.compose(bridge_inv)

    win_half_ns = int(DEFAULT_EGO_MOTION_WINDOW_S * 1e9 / 2)
    t0_lo = max(t_imu[0], headset_mocap.t_ns[0], ctrl_mocap.t_ns[0]) + win_half_ns + 50_000_000
    t0_hi = (min(t_imu[-1], headset_mocap.t_ns[-1], ctrl_mocap.t_ns[-1]) - win_half_ns
             - int(max(dt_values) * 1e9) - 50_000_000)
    if t0_hi <= t0_lo:
        return []

    rows = []
    stride_ns = int(anchor_stride_s * 1e9)
    t0 = int(t0_lo)
    while t0 < t0_hi:
        T_hc0 = T_hc(t0)
        if T_hc0 is None:
            t0 += stride_ns
            continue
        T_hc0_a, T_hc0_b = T_hc(t0 - win_half_ns), T_hc(t0 + win_half_ns)
        if T_hc0_a is None or T_hc0_b is None:
            t0 += stride_ns
            continue
        v_hc0 = (T_hc0_b.t - T_hc0_a.t) / DEFAULT_EGO_MOTION_WINDOW_S
        wh0 = world_pose(headset_mocap, t0)
        omega_h0 = headset_angular_velocity(headset_mocap, t0)
        v_wh0 = headset_linear_velocity(headset_mocap, t0)
        if wh0 is None or omega_h0 is None or v_wh0 is None:
            t0 += stride_ns
            continue
        for dt in dt_values:
            t1 = t0 + int(dt * 1e9)
            wh1 = world_pose(headset_mocap, t1)
            T_true1 = T_hc(t1)
            if wh1 is None or T_true1 is None:
                continue
            predicted = predict_headset_relative_pose(
                t_imu, gyro_body, t_imu, accel_body, MOCAP_ROOM_G_WORLD, lever_arm,
                t0, t1, T_hc0.R, T_hc0.t, v_hc0, wh0.R, wh0.t, omega_h0, v_wh0, wh1.R, wh1.t)
            if predicted is None:
                continue
            R_pred, p_pred = predicted
            rot_err_deg = rotation_angle_deg(R_pred.T @ T_true1.R)
            pos_err_mm = float(np.linalg.norm(p_pred - T_true1.t)) * 1000.0
            peak_g, peak_a = peak_gyro_accel_over_window((t_imu, gyro_body), (t_imu, accel_body), t0, t1)
            rows.append((t0, dt, peak_g, peak_a, float(np.linalg.norm(v_hc0)), rot_err_deg, pos_err_mm))
        t0 += stride_ns
    return rows


HEADER = ["recording", "ctrl_name", "anchor_ts_ns", "dt_s", "peak_gyro_dps", "peak_accel_mps2", "speed_mps",
          "rot_err_deg", "pos_err_mm"]


def main():
    variant_name, rec_name, ctrl_name, out_csv = sys.argv[1:5]
    rows = analyze(RECORDINGS_ROOT / rec_name, ctrl_name, VARIANTS[variant_name])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for r in rows:
            w.writerow([rec_name, ctrl_name, *r])
    print(f"[{variant_name}/{rec_name}/{ctrl_name}] {len(rows)} rows -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
