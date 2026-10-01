#!/usr/bin/env python3
"""
orientation_gravity_correction_check.py -- Step 3 follow-up, use-case #1:
does correcting gyro-only orientation integration against the accelerometer's
own gravity-direction reading reduce orientation error over LONG (occlusion)
gaps, compared to pure gyro-only integration?

This tests a DIFFERENT job than position dead-reckoning (which repeatedly
failed this session, see accel_short_horizon_check.py / accel_occlusion_check.py
/ accel_jerk_check.py): gyro-only orientation integration drifts too (bias
integrates linearly into orientation error, and gyro alone has no absolute
reference for roll/pitch), but accel's long-term gravity direction IS a
genuine absolute reference for "which way is down" -- exactly what Monado's
own WMR driver uses accel for (3dof orientation-vs-gravity fusion, never
translation -- see this session's Monado source dig), and exactly what the
low-motion bootstrap (accel_short_horizon_check.low_motion_bootstrap_g_world)
already validated is recoverable in this data (|g| converges to ~9.5-10.1
m/s^2, not the whole-recording bootstrap's biased ~6.9-7.7).

Method (standard single-step complementary-filter tilt correction):
  R_gyro_only(t1) = R(t0) . DeltaR_gyro(t0,t1)          [Step 1's validated integrate_gyro_segment]
  up_implied_world = R_gyro_only(t1) @ normalize(accel_body(t1))
      -- should equal -g_world_hat if R_gyro_only were exact (stationary-
         accelerometer convention: a_world=0 => accel_body = -R_body_world @ g_world,
         i.e. a stationary reading points AWAY from gravity, "up")
  Q = minimal rotation taking up_implied_world to -g_world_hat
  R_corrected(t1) = Q @ R_gyro_only(t1)                  [full tilt snap -- an upper
                                                           bound test, not a tuned filter]
Both R_gyro_only(t1) and R_corrected(t1) are compared against MOCAP orientation
at t1 (via the bridge, converting from vision's LED-ref frame convention to
mocap's accel-IMU frame convention -- see compare_vision_mocap.py's module
docstring / Step 0's residual_stats for why the bridge is needed for this
comparison specifically, unlike the un-bridged convention used for rotating
raw accel/gyro into world elsewhere this session).

Caveat this test does NOT address: the accel reading at t1 is a single noisy
instantaneous sample, contaminated by whatever real dynamic acceleration the
controller has at that exact moment (worst-case: right as it reappears from
occlusion, often the highest-dynamics moment in the whole gap) -- a full
"trust accel 100% for tilt" snap is an upper-bound/lower-bound experiment
(could help a lot if the accel reading happens to be clean, or hurt if it's
badly contaminated by real motion), not a properly-tuned complementary
filter. That refinement (weight the correction by how close |accel| is to
9.81, i.e. by the dynamics gate from motion_dynamics_check.py) is a natural
next step if this crude version shows the mechanism has any value at all.

Usage: python orientation_gravity_correction_check.py [config.yml] [output_dir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from accel_short_horizon_check import _IMU_FILES, low_motion_bootstrap_g_world
from accel_sign_check import world_vision_poses, _MAX_PAIR_DT_S
from compare_vision_mocap import load_pose_csv, load_device_mocap, load_or_fit_bridge, rotation_angle_deg
from src.imu_data import integrate_gyro_segment, load_and_calibrate_controller_imu
from src.load_config import load_yaml_config, load_json_config
from src.mocap_data import world_pose

_GAP_BINS_S = [0.0, 0.05, 0.1, 0.2, 0.5, np.inf]
_BIN_LABELS = ["<50ms", "50-100ms", "100-200ms", "200-500ms", ">500ms"]


def tilt_correct(R_gyro_only: np.ndarray, accel_at_t1: np.ndarray, g_world_hat: np.ndarray) -> np.ndarray:
    up_implied_world = R_gyro_only @ (accel_at_t1 / np.linalg.norm(accel_at_t1))
    target = -g_world_hat
    axis = np.cross(up_implied_world, target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        return R_gyro_only
    angle = np.arccos(np.clip(np.dot(up_implied_world, target), -1.0, 1.0))
    Q = Rotation.from_rotvec(axis / axis_norm * angle).as_matrix()
    return Q @ R_gyro_only


def plot_binned(ctrl_name, gap_s, err_gyro_deg, err_corrected_deg, out_path: Path):
    bin_idx = np.digitize(gap_s, _GAP_BINS_S) - 1
    gyro_medians, corr_medians, ns = [], [], []
    for b in range(len(_BIN_LABELS)):
        mask = bin_idx == b
        if mask.sum() < 3:
            gyro_medians.append(np.nan); corr_medians.append(np.nan); ns.append(int(mask.sum()))
            continue
        gyro_medians.append(float(np.median(err_gyro_deg[mask])))
        corr_medians.append(float(np.median(err_corrected_deg[mask])))
        ns.append(int(mask.sum()))
    x = np.arange(len(_BIN_LABELS))
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    ax.bar(x - width / 2, gyro_medians, width, label="gyro-only", color="steelblue")
    ax.bar(x + width / 2, corr_medians, width, label="gravity-tilt-corrected", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\n(n={n})" for lbl, n in zip(_BIN_LABELS, ns)])
    ax.set_ylabel("median orientation error vs. mocap (deg)")
    ax.set_title(f"{ctrl_name}: gyro-only vs. gravity-tilt-corrected orientation, by gap size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yml"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("visualization/step3")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml_config(config_path)

    pose_csv_path = config.get("debug", {}).get("pose_csv")
    poses, errors = load_pose_csv(pose_csv_path)
    headset_mocap = load_device_mocap("headset")
    mav0_root = Path(config["data"]["root"])

    for ctrl_name in ("left_controller", "right_controller"):
        if ctrl_name not in poses or ctrl_name not in _IMU_FILES:
            continue
        imu_rel_path, lag_ns = _IMU_FILES[ctrl_name]
        imu_path = mav0_root / imu_rel_path
        if not imu_path.exists():
            print(f"[{ctrl_name}] IMU file not found -- skipping")
            continue

        ctrl_cfg = config["controllers"][ctrl_name]
        ctrl_json_cfg = load_json_config(ctrl_cfg["config_path"])
        t_gyro, gyro_body, accel_body = load_and_calibrate_controller_imu(imu_path, ctrl_json_cfg, lag_ns=lag_ns)
        ctrl_mocap = load_device_mocap(ctrl_name)
        bridge = load_or_fit_bridge(ctrl_name, ctrl_cfg, poses[ctrl_name], headset_mocap, ctrl_mocap,
                                     errors.get(ctrl_name))

        world_poses = world_vision_poses(poses[ctrl_name], headset_mocap)
        ts_sorted = sorted(world_poses.keys())

        g_world, n_used, n_total = low_motion_bootstrap_g_world(t_gyro, gyro_body, t_gyro, accel_body, world_poses)
        if g_world is None:
            print(f"[{ctrl_name}] not enough low-motion frames -- skipping")
            continue
        g_world_hat = g_world / np.linalg.norm(g_world)
        print(f"[{ctrl_name}] g_world bootstrap: {n_used}/{n_total} low-motion frames, "
              f"|g|={np.linalg.norm(g_world):.3f} m/s^2")

        gap_s, err_gyro, err_corrected = [], [], []
        n_skipped = 0
        for i in range(len(ts_sorted) - 1):
            t0, t1 = ts_sorted[i], ts_sorted[i + 1]
            dt = (t1 - t0) / 1e9
            if dt <= 0 or t0 < t_gyro[0] or t1 > t_gyro[-1]:
                n_skipped += 1
                continue

            R_mocap_t1 = world_pose(ctrl_mocap, t1)
            if R_mocap_t1 is None:
                n_skipped += 1
                continue

            R0 = world_poses[t0].R
            dR_gyro = integrate_gyro_segment(t_gyro, gyro_body, t0, t1)
            if dR_gyro is None:
                n_skipped += 1
                continue
            R_gyro_only = R0 @ dR_gyro

            accel_t1 = np.array([np.interp(t1, t_gyro, accel_body[:, ax]) for ax in range(3)])
            R_corrected = tilt_correct(R_gyro_only, accel_t1, g_world_hat)

            # compare in mocap's frame convention (bridge), see module docstring
            R_gyro_bridged = R_gyro_only @ bridge.R
            R_corrected_bridged = R_corrected @ bridge.R
            R_mocap = R_mocap_t1.R

            gap_s.append(dt)
            err_gyro.append(rotation_angle_deg(R_gyro_bridged.T @ R_mocap))
            err_corrected.append(rotation_angle_deg(R_corrected_bridged.T @ R_mocap))

        gap_s = np.array(gap_s)
        err_gyro = np.array(err_gyro)
        err_corrected = np.array(err_corrected)
        n = len(gap_s)
        if n == 0:
            print(f"[{ctrl_name}] no usable gaps -- skipping")
            continue

        print(f"[{ctrl_name}] {n} gaps ({n_skipped} skipped)")
        bin_idx = np.digitize(gap_s, _GAP_BINS_S) - 1
        for b, label in enumerate(_BIN_LABELS):
            mask = bin_idx == b
            nb = int(mask.sum())
            if nb == 0:
                continue
            med_gyro = np.median(err_gyro[mask])
            med_corr = np.median(err_corrected[mask])
            win_rate = float(np.mean(err_corrected[mask] < err_gyro[mask]))
            print(f"    gap {label:<10} n={nb:<5} gyro_only={med_gyro:6.2f}deg  "
                  f"gravity_corrected={med_corr:6.2f}deg  corrected_wins={win_rate * 100:5.1f}%")

        plot_binned(ctrl_name, gap_s, err_gyro, err_corrected, out_dir / f"orientation_gravity_{ctrl_name}_by_gap.png")
        print(f"[{ctrl_name}] saved plot to {out_dir}/orientation_gravity_{ctrl_name}_by_gap.png\n")


if __name__ == "__main__":
    main()
