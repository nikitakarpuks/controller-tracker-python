#!/usr/bin/env python3
"""
accel_short_horizon_check.py -- Step 3 (revised sub-stage 2): tests the
ACTUAL intended use of accel data -- short-horizon forward integration
between vision frames, then fused/corrected by the next vision solve -- as
opposed to the original sub-stage 2 (accel_lever_arm_solve.py), which tried
to CALIBRATE r/b_a/g_world from scratch via a whole-recording least-squares
regression against vision-derived acceleration (DOUBLE-DIFFERENTIATED vision
position). That calibration approach was unstable (see accel_lever_arm_solve.py's
docstring/session notes) because differentiation AMPLIFIES noise by ~1/dt^2 --
mm-level vision position noise became ~10 m/s^2 of acceleration noise, larger
than the signal being solved for.

Short-horizon INTEGRATION does the opposite: a calibration error shrinks by
~dt^2 over a short window. Concretely, over a ~15-30ms single-frame gap: a
~2.8 m/s^2 gravity error integrated twice gives ~0.3mm; completely IGNORING
the lever arm (r=0) at a fast 10 rad/s rotation gives ~1mm -- both at or
below vision's own ~3-5mm noise floor. So this test deliberately uses a
CRUDE model -- r=0 (still, as in sub-stage 1), b_a=0 (factory-corrected
already), g_world from a LOW-ANGULAR-VELOCITY-frame-only bootstrap (fixing
sub-stage 1's flaw: its whole-recording bootstrap was itself biased by the
omitted lever arm whenever the controller was rotating fast; restricting to
near-stationary frames removes that bias) -- and checks whether that's
already good enough to be USEFUL, not perfectly calibrated.

Method: for each vision frame t0 (with a causal, backward-difference initial
velocity v0 from t0's own PRECEDING gap, so no future data leaks in) and the
next accepted frame t1, predict p(t1) two ways:
  1. accel-integrated:      p(t0) + integrate_accel_to_position(...)
  2. naive constant-velocity: p(t0) + v0 * (t1 - t0)   (no accel at all)
and compare each against vision's own solved p(t1) -- the real question is
whether accel prediction (1) BEATS the naive baseline (2), which is the
actual bar an accel term needs to clear to be worth including in Step 4.

Usage: python accel_short_horizon_check.py [path/to/config.yml] [output_dir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from accel_sign_check import world_vision_poses, _MAX_PAIR_DT_S
from compare_vision_mocap import load_pose_csv, load_device_mocap
from src.imu_data import integrate_accel_to_position, load_and_calibrate_controller_imu, LOW_OMEGA_THRESH_RAD_S
from src.load_config import load_yaml_config, load_json_config

from src.mocap_data import controller_imu_files
_IMU_FILES = controller_imu_files()  # lag_ns = -mocap_vision_offset_ns from config.yml (shared with main.py)
# "near-stationary" gate for the g_world bootstrap -- canonical value now lives in
# src.imu_data (shared with LiveGravityEstimator's online counterpart; this was
# previously its own independent copy of the same magic number, found in code review).
_LOW_OMEGA_THRESH_RAD_S = LOW_OMEGA_THRESH_RAD_S


def low_motion_bootstrap_g_world(t_gyro, gyro_body, t_accel, accel_body, world_poses: dict,
                                  omega_thresh=_LOW_OMEGA_THRESH_RAD_S):
    """Same idea as accel_sign_check.bootstrap_g_world but restricted to
    frames where |gyro| is small -- at low angular velocity the omitted
    lever-arm terms (~omega^2 * r) are small too, so this estimate isn't
    biased by sub-stage 1's flaw. Returns (g_world, n_used, n_total)."""
    rows, n_total = [], 0
    for ts_ns, T in world_poses.items():
        if ts_ns < t_accel[0] or ts_ns > t_accel[-1] or ts_ns < t_gyro[0] or ts_ns > t_gyro[-1]:
            continue
        n_total += 1
        omega = np.array([np.interp(ts_ns, t_gyro, gyro_body[:, i]) for i in range(3)])
        if np.linalg.norm(omega) > omega_thresh:
            continue
        acc = np.array([np.interp(ts_ns, t_accel, accel_body[:, i]) for i in range(3)])
        rows.append(T.R @ acc)
    if len(rows) < 20:
        return None, len(rows), n_total
    return -np.mean(np.stack(rows), axis=0), len(rows), n_total


def pearson(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def plot_comparison(ctrl_name, err_accel_mm, err_naive_mm, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, max(np.percentile(err_accel_mm, 99), np.percentile(err_naive_mm, 99)), 60)
    ax.hist(err_naive_mm, bins=bins, alpha=0.55, label=f"naive const-velocity (median={np.median(err_naive_mm):.2f}mm)",
            color="steelblue")
    ax.hist(err_accel_mm, bins=bins, alpha=0.55, label=f"accel-integrated (median={np.median(err_accel_mm):.2f}mm)",
            color="darkorange")
    ax.set_xlabel("|prediction error| at t1 (mm)")
    ax.set_ylabel("count")
    ax.set_title(f"{ctrl_name}: short-horizon position-prediction error (n={len(err_accel_mm)})")
    ax.legend(fontsize=9)
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
            print(f"[{ctrl_name}] not enough low-motion frames ({n_used}) -- skipping")
            continue
        print(f"[{ctrl_name}] low-motion g_world bootstrap: {n_used}/{n_total} frames "
              f"(|omega|<{_LOW_OMEGA_THRESH_RAD_S} rad/s), g_world={g_world}  |g|={np.linalg.norm(g_world):.3f} m/s^2 "
              f"(expect ~9.81)")

        err_accel, err_naive = [], []
        n_skipped = 0
        for i in range(1, len(ts_sorted) - 1):
            t_prev, t0, t1 = ts_sorted[i - 1], ts_sorted[i], ts_sorted[i + 1]
            dt_prev = (t0 - t_prev) / 1e9
            dt = (t1 - t0) / 1e9
            if dt_prev <= 0 or dt <= 0 or dt_prev > _MAX_PAIR_DT_S or dt > _MAX_PAIR_DT_S:
                n_skipped += 1
                continue

            v0 = (world_poses[t0].t - world_poses[t_prev].t) / dt_prev  # causal, backward-difference

            dp_accel = integrate_accel_to_position(t_accel, accel_body, t0, t1,
                                                    world_poses[t0].R, world_poses[t1].R, v0, g_world)
            if dp_accel is None:
                n_skipped += 1
                continue

            p1_true = world_poses[t1].t
            p1_accel = world_poses[t0].t + dp_accel
            p1_naive = world_poses[t0].t + v0 * dt

            err_accel.append(np.linalg.norm(p1_accel - p1_true) * 1000.0)
            err_naive.append(np.linalg.norm(p1_naive - p1_true) * 1000.0)

        err_accel = np.array(err_accel)
        err_naive = np.array(err_naive)
        n = len(err_accel)
        if n == 0:
            print(f"[{ctrl_name}] no gaps covered -- skipping")
            continue

        win_rate = float(np.mean(err_accel < err_naive))
        print(f"[{ctrl_name}] {n} gaps ({n_skipped} skipped):")
        print(f"    naive const-velocity error (mm): mean={err_naive.mean():.3f} median={np.median(err_naive):.3f} "
              f"p90={np.percentile(err_naive, 90):.3f}")
        print(f"    accel-integrated error     (mm): mean={err_accel.mean():.3f} median={np.median(err_accel):.3f} "
              f"p90={np.percentile(err_accel, 90):.3f}")
        print(f"    accel beats naive on {win_rate * 100:.1f}% of gaps")

        plot_comparison(ctrl_name, err_accel, err_naive, out_dir / f"accel_short_horizon_{ctrl_name}_hist.png")
        print(f"[{ctrl_name}] saved plot to {out_dir}/accel_short_horizon_{ctrl_name}_hist.png\n")


if __name__ == "__main__":
    main()
