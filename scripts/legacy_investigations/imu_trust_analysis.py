#!/usr/bin/env python3
"""Big IMU/gyro trust-duration investigation, across all 8 recordings.

Question: for how long, after vision goes away, can we trust BLIND gyro-only
rotation dead-reckoning and gyro+accel position dead-reckoning, as a function
of how violent the real motion is (peak gyro dps / peak dynamic accel m/s^2
over the gap)?

Methodology: reuses the EXACT production dead-reckoning primitive live
tracking uses -- src.imu_data.predict_headset_relative_pose (the same
function HeuristicPoseFusionFilter.predict() calls) -- fed with real mocap
ground truth as the KNOWN starting state at many anchor points t0 across each
recording, dead-reckoned BLIND (no peeking) forward by many candidate dt
values, and compared against real mocap ground truth at t0+dt. This is the
same "causal, no-peeking, dead-reckon vs real recorded motion" standard this
project has used before (see memory: project_imu_axis_fix_and_reversal's own
13-real-gap test), just swept densely across every recording instead of a
handful of hand-picked gaps.

Frame: entirely in the HEADSET-RELATIVE frame (T_headsetImu_ctrlImu) that
predict_headset_relative_pose itself operates in -- headset ego-motion
(world_pose/headset_angular_velocity/headset_linear_velocity, all real mocap
data) is fed in exactly as the live pipeline's own headset-correction path
does, so a moving/rotating headset (the person walking) doesn't contaminate
the controller's own IMU-drift measurement. g_world uses MOCAP_ROOM_G_WORLD
(src/imu_data.py) -- the same empirically-validated constant that already
REPLACES the live g_world_estimator_abs, per main.py's own comment -- so this
sidesteps needing g_world convergence entirely (unlike the live pipeline,
which sometimes runs with g_world NOT converged, per earlier investigation
logs this session).

Usage: python3 imu_trust_analysis.py <recording_dir_name> <ctrl_name> <out_csv>
  e.g. python3 imu_trust_analysis.py euroc_recording_20260826173103_static_dark right_controller /tmp/out.csv
Run once per (recording, controller) -- see run_imu_trust_analysis_all.py for
the parallel driver across all 8x2 combinations.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_vision_mocap import rotation_angle_deg  # noqa: E402
from src.imu_data import accel_lever_arm_body
from src.imu_data import (load_and_calibrate_controller_imu, create_imu_calib_from_config,
                           predict_headset_relative_pose, peak_gyro_accel_over_window,
                           MOCAP_ROOM_G_WORLD)  # noqa: E402
from src.mocap_data import (load_mocap_csv, load_mocap_fine_offset_ns, load_T_imu_marker,
                             DeviceMocap, relative_pose, world_pose, headset_angular_velocity,
                             headset_linear_velocity, DRIFT_CHECK_VARIANT,
                             DEFAULT_EGO_MOTION_WINDOW_S)  # noqa: E402
from src.load_config import load_yaml_config, load_json_config  # noqa: E402

RECORDINGS_ROOT = Path("/home/nikitakarpuks/Downloads/recordings-aug26")
REPO_ROOT = Path(__file__).resolve().parent
CONFIG = load_yaml_config(str(REPO_ROOT / "config" / "config.yml"))

_MOCAP_DISK_NAMES = {"headset": "headset", "left_controller": "ctrlleft", "right_controller": "ctrlright"}
_MOCAP_CALIB_FILES = {
    "headset": CONFIG["cameras"]["mocap_calib_path"],
    "left_controller": CONFIG["controllers"]["left_controller"]["mocap_calib_path"],
    "right_controller": CONFIG["controllers"]["right_controller"]["mocap_calib_path"],
}
_IMU_REL_PATH = {"left_controller": "imu1/data.csv", "right_controller": "imu2/data.csv"}
from src.mocap_data import controller_imu_lag_ns
_IMU_LAG_NS = {k: controller_imu_lag_ns(k) for k in ("left_controller", "right_controller")}  # from config.yml, shared with main.py

# Anchors every ~100ms (dense enough for thousands of samples per recording,
# sparse enough to keep runtime bounded -- see module docstring's timing note).
ANCHOR_STRIDE_S = 0.10
# dt sweep: short-horizon (this project's existing ~11-35ms budgets) through
# long-horizon (~1s, well past anything currently trusted, to see where the
# curve really breaks down).
DT_VALUES_S = [0.011, 0.022, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]


def load_device_mocap(rec_dir: Path, device_key: str) -> DeviceMocap:
    device_dir = rec_dir / "mocap_filtered" / _MOCAP_DISK_NAMES[device_key]
    calib_path = _MOCAP_CALIB_FILES[device_key]
    t, position, quat_xyzw = load_mocap_csv(device_dir / "data.csv")
    fine_offset_ns = load_mocap_fine_offset_ns(device_dir / "drift_check" / DRIFT_CHECK_VARIANT / "drift_check.json")
    T_imu_marker = load_T_imu_marker(calib_path)
    return DeviceMocap(t, position, quat_xyzw, fine_offset_ns, T_imu_marker)


def analyze(rec_dir: Path, ctrl_name: str, anchor_stride_s: float = ANCHOR_STRIDE_S,
            dt_values=DT_VALUES_S):
    mav0 = rec_dir / "mav0"
    ctrl_json_cfg = load_json_config(CONFIG["controllers"][ctrl_name]["config_path"])
    t_imu, gyro_body, accel_body = load_and_calibrate_controller_imu(
        mav0 / _IMU_REL_PATH[ctrl_name], ctrl_json_cfg, lag_ns=_IMU_LAG_NS[ctrl_name])
    imu_calib = create_imu_calib_from_config(ctrl_json_cfg)
    lever_arm = accel_lever_arm_body(imu_calib)

    headset_mocap = load_device_mocap(rec_dir, "headset")
    ctrl_mocap = load_device_mocap(rec_dir, ctrl_name)

    win_half_ns = int(DEFAULT_EGO_MOTION_WINDOW_S * 1e9 / 2)
    t0_lo = max(t_imu[0], headset_mocap.t_ns[0], ctrl_mocap.t_ns[0]) + win_half_ns + 50_000_000
    t0_hi = min(t_imu[-1], headset_mocap.t_ns[-1], ctrl_mocap.t_ns[-1]) - win_half_ns - int(max(dt_values) * 1e9) - 50_000_000
    if t0_hi <= t0_lo:
        return []

    rows = []
    stride_ns = int(anchor_stride_s * 1e9)
    t0 = int(t0_lo)
    while t0 < t0_hi:
        T_hc0 = relative_pose(headset_mocap, ctrl_mocap, t0)
        if T_hc0 is None:
            t0 += stride_ns
            continue
        T_hc0_a = relative_pose(headset_mocap, ctrl_mocap, t0 - win_half_ns)
        T_hc0_b = relative_pose(headset_mocap, ctrl_mocap, t0 + win_half_ns)
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
        R_wh0, p_wh0 = wh0.R, wh0.t

        for dt in dt_values:
            t1 = t0 + int(dt * 1e9)
            wh1 = world_pose(headset_mocap, t1)
            T_true1 = relative_pose(headset_mocap, ctrl_mocap, t1)
            if wh1 is None or T_true1 is None:
                continue
            predicted = predict_headset_relative_pose(
                t_imu, gyro_body, t_imu, accel_body, MOCAP_ROOM_G_WORLD, lever_arm,
                t0, t1, T_hc0.R, T_hc0.t, v_hc0,
                R_wh0, p_wh0, omega_h0, v_wh0, wh1.R, wh1.t,
            )
            if predicted is None:
                continue
            R_pred, p_pred = predicted
            rot_err_deg = rotation_angle_deg(R_pred.T @ T_true1.R)
            pos_err_mm = float(np.linalg.norm(p_pred - T_true1.t)) * 1000.0
            peak_gyro_dps, peak_accel_mps2 = peak_gyro_accel_over_window(
                (t_imu, gyro_body), (t_imu, accel_body), t0, t1)
            speed_mps = float(np.linalg.norm(v_hc0))
            rows.append((t0, dt, peak_gyro_dps, peak_accel_mps2, speed_mps, rot_err_deg, pos_err_mm))
        t0 += stride_ns

    return rows


def main():
    rec_name = sys.argv[1]
    ctrl_name = sys.argv[2]
    out_csv = sys.argv[3]
    rec_dir = RECORDINGS_ROOT / rec_name
    rows = analyze(rec_dir, ctrl_name)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["anchor_ts_ns", "dt_s", "peak_gyro_dps", "peak_accel_mps2", "speed_mps", "rot_err_deg", "pos_err_mm"])
        for r in rows:
            w.writerow(r)
    print(f"[{rec_name}/{ctrl_name}] {len(rows)} rows -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
