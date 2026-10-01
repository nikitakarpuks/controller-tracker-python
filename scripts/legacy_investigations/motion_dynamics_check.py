#!/usr/bin/env python3
"""
motion_dynamics_check.py -- Step 3 follow-up, use-case #2: does an
accelerometer/gyro-derived "how much is the controller actually moving right
now" score correlate with vision solve quality (reproj_err_px, inlier_count,
true residual vs. mocap)? This is the same mechanism this codebase already
uses for a coarser purpose (visualize_imu.py's GRAVITY_MS2/
GRAVITY_LOW_DYNAMICS_TOL_MS2 "gravity band" check, and Neto et al. 2013's
ZUPT motion-stop detector, see session notes) -- here tested as a candidate
CONFIDENCE/GATING signal, complementing Step 2's PnP-certainty weighting
(reproj_err_px, inlier_count) with an independent, IMU-derived signal.

Dynamics score per accepted frame t: |a_world(t)| = |R_world_body(t) @
accel_body(t) + g_world| -- the world-frame DYNAMIC linear acceleration
magnitude after removing gravity (g_world from the validated low-motion
bootstrap) -- 0 when stationary, large during fast motion. Also reports
|gyro_body(t)| (angular velocity magnitude) as a secondary rotational-
dynamics indicator.

Usage: python motion_dynamics_check.py [config.yml] [output_dir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from accel_short_horizon_check import _IMU_FILES, low_motion_bootstrap_g_world
from accel_sign_check import world_vision_poses
from compare_vision_mocap import load_pose_csv, load_device_mocap, load_or_fit_bridge, residual_stats
from src.imu_data import load_and_calibrate_controller_imu
from src.load_config import load_yaml_config, load_json_config


def load_pose_csv_with_inliers(path):
    """Same shape as pnp_certainty_check.py's loader -- (poses, reproj_err_px,
    inlier_count), all {ctrl_name: {timestamp_ns: value}}."""
    import csv
    from scipy.spatial.transform import Rotation
    from src.transformations import Transform
    poses, errors, inliers = {}, {}, {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        has_inliers = "inlier_count" in header
        for row in reader:
            ts_ns = int(row[0])
            ctrl_name = row[1]
            qx, qy, qz, qw = (float(x) for x in row[2:6])
            px, py, pz = (float(x) for x in row[6:9])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([px, py, pz])
            poses.setdefault(ctrl_name, {})[ts_ns] = Transform(R, t)
            errors.setdefault(ctrl_name, {})[ts_ns] = float(row[9])
            if has_inliers:
                inliers.setdefault(ctrl_name, {})[ts_ns] = int(row[10])
    return poses, errors, inliers


def pearson(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def plot_dynamics_vs_quality(ctrl_name, dyn_accel, dyn_gyro, reproj_err, inlier_count,
                              trans_mm, rot_deg, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].scatter(dyn_accel, reproj_err, s=6, alpha=0.3, color="steelblue")
    axes[0, 0].set_xlabel("dynamic |a| (m/s^2)"); axes[0, 0].set_ylabel("reproj_err_px")
    axes[0, 0].set_title(f"r={pearson(dyn_accel, reproj_err):+.3f}")

    axes[0, 1].scatter(dyn_accel, inlier_count, s=6, alpha=0.3, color="steelblue")
    axes[0, 1].set_xlabel("dynamic |a| (m/s^2)"); axes[0, 1].set_ylabel("inlier_count")
    axes[0, 1].set_title(f"r={pearson(dyn_accel, inlier_count):+.3f}")

    axes[0, 2].scatter(dyn_accel, trans_mm, s=6, alpha=0.3, color="steelblue")
    axes[0, 2].set_xlabel("dynamic |a| (m/s^2)"); axes[0, 2].set_ylabel("true trans residual (mm)")
    axes[0, 2].set_title(f"r={pearson(dyn_accel, trans_mm):+.3f}")

    axes[1, 0].scatter(dyn_gyro, reproj_err, s=6, alpha=0.3, color="darkorange")
    axes[1, 0].set_xlabel("|gyro| (rad/s)"); axes[1, 0].set_ylabel("reproj_err_px")
    axes[1, 0].set_title(f"r={pearson(dyn_gyro, reproj_err):+.3f}")

    axes[1, 1].scatter(dyn_gyro, inlier_count, s=6, alpha=0.3, color="darkorange")
    axes[1, 1].set_xlabel("|gyro| (rad/s)"); axes[1, 1].set_ylabel("inlier_count")
    axes[1, 1].set_title(f"r={pearson(dyn_gyro, inlier_count):+.3f}")

    axes[1, 2].scatter(dyn_gyro, rot_deg, s=6, alpha=0.3, color="darkorange")
    axes[1, 2].set_xlabel("|gyro| (rad/s)"); axes[1, 2].set_ylabel("true rot residual (deg)")
    axes[1, 2].set_title(f"r={pearson(dyn_gyro, rot_deg):+.3f}")

    fig.suptitle(f"{ctrl_name}: does IMU-derived dynamics predict vision quality?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yml"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("visualization/step3")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml_config(config_path)

    pose_csv_path = config.get("debug", {}).get("pose_csv")
    poses, errors, inlier_counts = load_pose_csv_with_inliers(pose_csv_path)
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

        g_world, n_used, n_total = low_motion_bootstrap_g_world(t_gyro, gyro_body, t_gyro, accel_body, world_poses)
        if g_world is None:
            print(f"[{ctrl_name}] not enough low-motion frames -- skipping")
            continue
        print(f"[{ctrl_name}] g_world bootstrap: {n_used}/{n_total} low-motion frames, "
              f"|g|={np.linalg.norm(g_world):.3f} m/s^2")

        rot_deg, trans_mm, n_skipped, _residuals, kept_ts = residual_stats(
            poses[ctrl_name], headset_mocap, ctrl_mocap, bridge)

        dyn_accel, dyn_gyro, reproj_err, inlier_count = [], [], [], []
        kept_ts_final, rot_final, trans_final = [], [], []
        for ts, r, tr in zip(kept_ts, rot_deg, trans_mm):
            if ts not in world_poses or ts < t_gyro[0] or ts > t_gyro[-1]:
                continue
            R = world_poses[ts].R
            acc = np.array([np.interp(ts, t_gyro, accel_body[:, ax]) for ax in range(3)])
            gyr = np.array([np.interp(ts, t_gyro, gyro_body[:, ax]) for ax in range(3)])
            a_dyn = R @ acc + g_world

            dyn_accel.append(np.linalg.norm(a_dyn))
            dyn_gyro.append(np.linalg.norm(gyr))
            reproj_err.append(errors[ctrl_name][ts])
            inlier_count.append(inlier_counts[ctrl_name][ts])
            rot_final.append(r)
            trans_final.append(tr)

        dyn_accel = np.array(dyn_accel)
        dyn_gyro = np.array(dyn_gyro)
        reproj_err = np.array(reproj_err)
        inlier_count = np.array(inlier_count)
        rot_final = np.array(rot_final)
        trans_final = np.array(trans_final)
        n = len(dyn_accel)
        if n == 0:
            print(f"[{ctrl_name}] no usable frames -- skipping")
            continue

        print(f"[{ctrl_name}] {n} frames (dynamic |a| mean={dyn_accel.mean():.2f} std={dyn_accel.std():.2f} m/s^2, "
              f"|gyro| mean={dyn_gyro.mean():.2f} std={dyn_gyro.std():.2f} rad/s)")
        print(f"    dynamic|a|  vs reproj_err_px  r={pearson(dyn_accel, reproj_err):+.3f}")
        print(f"    dynamic|a|  vs inlier_count   r={pearson(dyn_accel, inlier_count):+.3f}")
        print(f"    dynamic|a|  vs true trans_mm  r={pearson(dyn_accel, trans_final):+.3f}")
        print(f"    |gyro|      vs reproj_err_px  r={pearson(dyn_gyro, reproj_err):+.3f}")
        print(f"    |gyro|      vs inlier_count   r={pearson(dyn_gyro, inlier_count):+.3f}")
        print(f"    |gyro|      vs true rot_deg   r={pearson(dyn_gyro, rot_final):+.3f}")

        plot_dynamics_vs_quality(ctrl_name, dyn_accel, dyn_gyro, reproj_err, inlier_count, trans_final, rot_final,
                                  out_dir / f"motion_dynamics_{ctrl_name}.png")
        print(f"[{ctrl_name}] saved plot to {out_dir}/motion_dynamics_{ctrl_name}.png\n")


if __name__ == "__main__":
    main()
