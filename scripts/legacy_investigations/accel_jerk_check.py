#!/usr/bin/env python3
"""
accel_jerk_check.py -- Step 3 follow-up: tests a constant-jerk (cubic
position) predictor as an alternative to raw double-integration of every
noisy raw accel sample (accel_short_horizon_check.py's approach).

Idea (from external research review, session notes): raw double integration
sums up EVERY noisy raw accelerometer sample across the gap -- for i.i.d.
sensor noise, integrating a noisy forcing signal produces a random walk whose
variance grows with elapsed time regardless of how finely it's sampled, so
"use every sample" buys nothing over "use two smoothed acceleration readings"
and pays for the noise of every single one. A constant-jerk model instead
fits a smooth CUBIC position curve using only p0, v0 (from vision, as
before), a0 (gravity-corrected accel AT t0), and a constant jerk j
extrapolated from the PAST (a0, a_prev) -- i.e. two smoothed acceleration
READINGS rather than dozens of raw ones:

    j = (a0 - a_prev) / (t0 - t_prev)                      [causal, from the past]
    p(t0+dt) = p0 + v0*dt + 0.5*a0*dt^2 + (1/6)*j*dt^3      [constant-jerk kinematics]

This is DELIBERATELY causal (unlike a naive Hermite fit using a1 measured AT
t1, which would need future data not yet available at prediction time) --
jerk is extrapolated forward from what's already known at t0, same as v0's
own backward-difference estimate, so this is a fair three-way comparison
against naive constant-velocity and accel_short_horizon_check.py's full
double-integration, not an unfair lookahead advantage for the new method.

Ground truth: vision's own solve at t1, same convention
accel_short_horizon_check.py used -- appropriate here since this runs on the
smooth static_dark recording where vision itself is reliable (see the
static_hard occlusion test's own finding: vision re-acquisition is reliable,
so this convention is not the methodological issue it would be on a harder
recording with real occlusion-adjacent reacquisition frames).

Usage: python accel_jerk_check.py [path/to/config.yml] [output_dir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from accel_short_horizon_check import _IMU_FILES, low_motion_bootstrap_g_world
from accel_sign_check import world_vision_poses, _MAX_PAIR_DT_S
from compare_vision_mocap import load_pose_csv, load_device_mocap
from src.imu_data import integrate_accel_to_position, load_and_calibrate_controller_imu
from src.load_config import load_yaml_config, load_json_config


def gravity_corrected_accel(t_accel, accel_body, ts, R, g_world):
    """Gravity-corrected WORLD-frame accel at a single instant ts (linear
    interp of raw samples, rotated by the known orientation R at ts)."""
    acc = np.array([np.interp(ts, t_accel, accel_body[:, i]) for i in range(3)])
    return R @ acc + g_world


def plot_three_way(ctrl_name, err_naive_mm, err_accel_mm, err_jerk_mm, out_path: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    hi = np.percentile(np.concatenate([err_naive_mm, err_accel_mm, err_jerk_mm]), 99)
    bins = np.linspace(0, hi, 60)
    ax.hist(err_naive_mm, bins=bins, alpha=0.5, label=f"naive const-velocity (median={np.median(err_naive_mm):.2f}mm)",
            color="steelblue")
    ax.hist(err_accel_mm, bins=bins, alpha=0.5, label=f"full double-integration (median={np.median(err_accel_mm):.2f}mm)",
            color="darkorange")
    ax.hist(err_jerk_mm, bins=bins, alpha=0.5, label=f"constant-jerk (median={np.median(err_jerk_mm):.2f}mm)",
            color="mediumseagreen")
    ax.set_xlabel("|prediction error| at t1 (mm)")
    ax.set_ylabel("count")
    ax.set_title(f"{ctrl_name}: short-horizon prediction methods (n={len(err_naive_mm)})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yml"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("visualization/step3")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml_config(config_path)

    pose_csv_path = config.get("debug", {}).get("pose_csv")
    if not pose_csv_path or not Path(pose_csv_path).exists():
        raise SystemExit(f"debug.pose_csv not set or missing ({pose_csv_path})")

    poses, _errors = load_pose_csv(pose_csv_path)
    headset_mocap = load_device_mocap("headset")
    mav0_root = Path(config["data"]["root"])

    for ctrl_name in ("left_controller", "right_controller"):
        if ctrl_name not in poses or ctrl_name not in _IMU_FILES:
            continue
        imu_rel_path, lag_ns = _IMU_FILES[ctrl_name]
        imu_path = mav0_root / imu_rel_path
        if not imu_path.exists():
            print(f"[{ctrl_name}] IMU file not found ({imu_path}) -- skipping")
            continue

        ctrl_cfg = config["controllers"][ctrl_name]
        ctrl_json_cfg = load_json_config(ctrl_cfg["config_path"])
        t_gyro, gyro_body, accel_body = load_and_calibrate_controller_imu(imu_path, ctrl_json_cfg, lag_ns=lag_ns)
        t_accel = t_gyro

        world_poses = world_vision_poses(poses[ctrl_name], headset_mocap)
        ts_sorted = sorted(world_poses.keys())

        g_world, n_used, n_total = low_motion_bootstrap_g_world(t_gyro, gyro_body, t_accel, accel_body, world_poses)
        if g_world is None:
            print(f"[{ctrl_name}] not enough low-motion frames -- skipping")
            continue
        print(f"[{ctrl_name}] g_world bootstrap: {n_used}/{n_total} low-motion frames, "
              f"|g|={np.linalg.norm(g_world):.3f} m/s^2")

        err_naive, err_accel, err_jerk = [], [], []
        n_skipped = 0
        for i in range(1, len(ts_sorted) - 1):
            t_prev, t0, t1 = ts_sorted[i - 1], ts_sorted[i], ts_sorted[i + 1]
            dt_prev = (t0 - t_prev) / 1e9
            dt = (t1 - t0) / 1e9
            if dt_prev <= 0 or dt <= 0 or dt_prev > _MAX_PAIR_DT_S or dt > _MAX_PAIR_DT_S:
                n_skipped += 1
                continue
            if t_prev < t_accel[0] or t1 > t_accel[-1]:
                n_skipped += 1
                continue

            p0, p_prev, p1_true = world_poses[t0].t, world_poses[t_prev].t, world_poses[t1].t
            v0 = (p0 - p_prev) / dt_prev  # causal, backward-difference (same as accel_short_horizon_check.py)

            dp_accel = integrate_accel_to_position(t_accel, accel_body, t0, t1,
                                                    world_poses[t0].R, world_poses[t1].R, v0, g_world)
            if dp_accel is None:
                n_skipped += 1
                continue
            p1_accel = p0 + dp_accel

            p1_naive = p0 + v0 * dt

            a_prev = gravity_corrected_accel(t_accel, accel_body, t_prev, world_poses[t_prev].R, g_world)
            a0 = gravity_corrected_accel(t_accel, accel_body, t0, world_poses[t0].R, g_world)
            j = (a0 - a_prev) / dt_prev  # causal jerk estimate, from the past only
            p1_jerk = p0 + v0 * dt + 0.5 * a0 * dt ** 2 + (1.0 / 6.0) * j * dt ** 3

            err_naive.append(np.linalg.norm(p1_naive - p1_true) * 1000.0)
            err_accel.append(np.linalg.norm(p1_accel - p1_true) * 1000.0)
            err_jerk.append(np.linalg.norm(p1_jerk - p1_true) * 1000.0)

        err_naive = np.array(err_naive)
        err_accel = np.array(err_accel)
        err_jerk = np.array(err_jerk)
        n = len(err_naive)
        if n == 0:
            print(f"[{ctrl_name}] no usable gaps -- skipping")
            continue

        print(f"[{ctrl_name}] {n} gaps ({n_skipped} skipped):")
        print(f"    naive const-velocity     (mm): mean={err_naive.mean():.3f} median={np.median(err_naive):.3f} "
              f"p90={np.percentile(err_naive, 90):.3f}")
        print(f"    full double-integration  (mm): mean={err_accel.mean():.3f} median={np.median(err_accel):.3f} "
              f"p90={np.percentile(err_accel, 90):.3f}")
        print(f"    constant-jerk            (mm): mean={err_jerk.mean():.3f} median={np.median(err_jerk):.3f} "
              f"p90={np.percentile(err_jerk, 90):.3f}")
        print(f"    jerk beats naive on {np.mean(err_jerk < err_naive) * 100:.1f}% of gaps, "
              f"jerk beats full-integration on {np.mean(err_jerk < err_accel) * 100:.1f}% of gaps")

        plot_three_way(ctrl_name, err_naive, err_accel, err_jerk,
                        out_dir / f"accel_jerk_{ctrl_name}_hist.png")
        print(f"[{ctrl_name}] saved plot to {out_dir}/accel_jerk_{ctrl_name}_hist.png\n")


if __name__ == "__main__":
    main()
