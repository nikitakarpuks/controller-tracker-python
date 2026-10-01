#!/usr/bin/env python3
"""
accel_occlusion_check.py -- Step 3, occlusion-focused follow-up to
accel_short_horizon_check.py. That script found accel-integrated prediction
doesn't beat naive constant-velocity extrapolation on NORMAL (~15-30ms)
frame-to-frame gaps -- but per-plan, the actual reason to use accel at all
is bridging the gaps where vision has NOTHING (occlusion / TRACKING LOST),
not out-predicting vision when vision is already working. This script
targets exactly that regime, on euroc_recording_20260826174510_static_hard
(chosen because, unlike the smooth recording used so far, it has real
TRACKING LOST events -- 3036 in this run's log, ~1440/2958 frames accepted
per controller, gaps up to 1.0s).

Method: same short-horizon integrate-vs-naive comparison as
accel_short_horizon_check.py, but (a) uses THIS recording's own IMU/mocap
data (compare_vision_mocap._RECORDING_ROOT now derives from config.yml's
data.root, so no hardcoded-path juggling needed), (b) does NOT skip large
gaps -- bins results by gap size so the "does accel help through occlusion"
question can actually be read off, and (c) requires a CLEAN (short, <50ms)
gap immediately before t0 for the causal backward-difference v0 estimate,
so a prior occlusion doesn't contaminate the "last known good velocity"
input both methods share.

GROUND TRUTH: compared against MOCAP (world_pose(ctrl_mocap, t1)), NOT
vision's own re-acquired solve at t1 -- the first version of this script used
vision, which is exactly wrong for a "static_hard" recording: the frame right
after a long occlusion gap is precisely a cold-restart/re-acquisition solve,
the least reliable kind this pipeline produces, so comparing two predictions
against a possibly-also-wrong "truth" was not a trustworthy test. Also
reports how far vision's OWN re-acquired solve lands from mocap, so that
suspicion is checked directly rather than assumed. Position-only comparison
(translation, not orientation) sidesteps the LED-ref-vs-mocap-accel-IMU-frame
distinction that matters for rotation -- both predictions and mocap ground
truth are being compared as physical points, under the same lever-arm=0
approximation already used everywhere else in this test (mocap's own tracked
point is the accelerometer-IMU marker, ~85mm from vision's LED-ref point in
reality; treating them as the same point here is consistent with, not an
additional error beyond, the r=0 assumption already in play).

Usage: python accel_occlusion_check.py [path/to/config.yml] [output_dir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from accel_short_horizon_check import _IMU_FILES, low_motion_bootstrap_g_world
from accel_sign_check import world_vision_poses
from compare_vision_mocap import load_pose_csv, load_device_mocap
from src.imu_data import integrate_accel_to_position, load_and_calibrate_controller_imu
from src.load_config import load_yaml_config, load_json_config
from src.mocap_data import world_pose

_CLEAN_V0_GAP_S = 0.05    # prior gap must be this short for v0 to count as "clean"
_GAP_BINS_S = [0.0, 0.05, 0.1, 0.2, 0.5, np.inf]
_BIN_LABELS = ["<50ms", "50-100ms", "100-200ms", "200-500ms", ">500ms"]


def plot_binned_comparison(ctrl_name, gap_s, err_accel_mm, err_naive_mm, err_vision_mm, out_path: Path):
    bin_idx = np.digitize(gap_s, _GAP_BINS_S) - 1
    naive_medians, accel_medians, vision_medians, ns = [], [], [], []
    for b in range(len(_BIN_LABELS)):
        mask = bin_idx == b
        if mask.sum() < 3:
            naive_medians.append(np.nan)
            accel_medians.append(np.nan)
            vision_medians.append(np.nan)
            ns.append(int(mask.sum()))
            continue
        naive_medians.append(float(np.median(err_naive_mm[mask])))
        accel_medians.append(float(np.median(err_accel_mm[mask])))
        vision_medians.append(float(np.median(err_vision_mm[mask])))
        ns.append(int(mask.sum()))

    x = np.arange(len(_BIN_LABELS))
    fig, ax = plt.subplots(figsize=(9.5, 5))
    width = 0.27
    ax.bar(x - width, naive_medians, width, label="naive const-velocity", color="steelblue")
    ax.bar(x, accel_medians, width, label="accel-integrated", color="darkorange")
    ax.bar(x + width, vision_medians, width, label="vision's OWN re-acquired solve", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\n(n={n})" for lbl, n in zip(_BIN_LABELS, ns)])
    ax.set_ylabel("median |error vs. mocap| (mm)")
    ax.set_title(f"{ctrl_name}: error vs. MOCAP ground truth, by gap size (log scale)")
    ax.set_yscale("log")
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
        ctrl_mocap = load_device_mocap(ctrl_name)

        world_poses = world_vision_poses(poses[ctrl_name], headset_mocap)
        ts_sorted = sorted(world_poses.keys())

        g_world, n_used, n_total = low_motion_bootstrap_g_world(t_gyro, gyro_body, t_accel, accel_body, world_poses)
        if g_world is None:
            print(f"[{ctrl_name}] not enough low-motion frames -- skipping")
            continue
        print(f"[{ctrl_name}] g_world bootstrap: {n_used}/{n_total} low-motion frames, "
              f"|g|={np.linalg.norm(g_world):.3f} m/s^2")

        gap_s, err_accel, err_naive, err_vision = [], [], [], []
        n_skipped_no_clean_v0, n_skipped_no_accel_cov, n_skipped_no_mocap = 0, 0, 0
        for i in range(1, len(ts_sorted) - 1):
            t_prev, t0, t1 = ts_sorted[i - 1], ts_sorted[i], ts_sorted[i + 1]
            dt_prev = (t0 - t_prev) / 1e9
            dt = (t1 - t0) / 1e9
            if dt <= 0 or dt_prev <= 0 or dt_prev > _CLEAN_V0_GAP_S:
                n_skipped_no_clean_v0 += 1
                continue

            p1_mocap = world_pose(ctrl_mocap, t1)
            if p1_mocap is None:
                n_skipped_no_mocap += 1
                continue

            v0 = (world_poses[t0].t - world_poses[t_prev].t) / dt_prev

            dp_accel = integrate_accel_to_position(t_accel, accel_body, t0, t1,
                                                    world_poses[t0].R, world_poses[t1].R, v0, g_world)
            if dp_accel is None:
                n_skipped_no_accel_cov += 1
                continue

            p1_true = p1_mocap.t
            p1_accel = world_poses[t0].t + dp_accel
            p1_naive = world_poses[t0].t + v0 * dt
            p1_vision = world_poses[t1].t

            gap_s.append(dt)
            err_accel.append(np.linalg.norm(p1_accel - p1_true) * 1000.0)
            err_naive.append(np.linalg.norm(p1_naive - p1_true) * 1000.0)
            err_vision.append(np.linalg.norm(p1_vision - p1_true) * 1000.0)

        gap_s = np.array(gap_s)
        err_accel = np.array(err_accel)
        err_naive = np.array(err_naive)
        err_vision = np.array(err_vision)
        n = len(gap_s)
        if n == 0:
            print(f"[{ctrl_name}] no usable gaps -- skipping")
            continue

        print(f"[{ctrl_name}] {n} gaps with a clean pre-gap v0 and mocap coverage at t1 "
              f"({n_skipped_no_clean_v0} skipped: no clean v0, {n_skipped_no_mocap} skipped: no mocap at t1, "
              f"{n_skipped_no_accel_cov} skipped: no accel coverage)")

        bin_idx = np.digitize(gap_s, _GAP_BINS_S) - 1
        for b, label in enumerate(_BIN_LABELS):
            mask = bin_idx == b
            nb = int(mask.sum())
            if nb == 0:
                continue
            med_naive = np.median(err_naive[mask])
            med_accel = np.median(err_accel[mask])
            med_vision = np.median(err_vision[mask])
            win_rate = float(np.mean(err_accel[mask] < err_naive[mask]))
            print(f"    gap {label:<10} n={nb:<5} naive={med_naive:8.2f}mm  accel={med_accel:8.2f}mm  "
                  f"vision_reacq={med_vision:8.2f}mm  accel_wins={win_rate * 100:5.1f}%")

        plot_binned_comparison(ctrl_name, gap_s, err_accel, err_naive, err_vision,
                                out_dir / f"accel_occlusion_{ctrl_name}_by_gap.png")
        print(f"[{ctrl_name}] saved plot to {out_dir}/accel_occlusion_{ctrl_name}_by_gap.png\n")


if __name__ == "__main__":
    main()
