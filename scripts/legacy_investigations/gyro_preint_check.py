#!/usr/bin/env python3
"""
gyro_preint_check.py -- Step 1 of the IMU+vision fusion plan: validates
src/imu_data.gyro_preint_residual (the gyro preintegration factor) against
Step 0's world-frame vision orientations, used here as FIXED reference nodes
(no optimization yet -- that's Step 4, where bias also becomes a free state).

Per controller, for every pair of consecutive ACCEPTED vision frames (pose_csv
rows, sorted by timestamp_ns), computes
    r_gyro(k) = Log(DeltaR_gyro(t_k,t_k+1)^-1 . (q_k^-1 q_k+1))
where q_k/q_k+1 are WORLD-frame (mocap-world) vision orientations composed the
same way world_frame_vision.py's Step 0 does, but WITHOUT the mocap bridge:
gyro_body (src/imu_data.load_and_calibrate_controller_imu) lives in the
controller's own LED/body frame, the same frame pose_csv's T_ctrl_vision(t) is
already expressed in (see compare_vision_mocap.py's module docstring) -- the
bridge only matters when comparing against mocap, not against this
controller's own gyro. So q = world_pose(headset, t).compose(T_ctrl_vision(t))
needs no bridge here.

This is deliberately NOT the same computation as visualize_imu.py's
VisionPoseLog.gyro_prediction_error, which composes/compares raw (HEADSET-
RELATIVE) pose_csv orientations directly against gyro-integrated rotation.
That mixes frames -- gyro measures rotation in an inertial frame, but a
headset-relative vision delta also encodes however much the HEADSET itself
rotated during the gap (only cancels if unrelated: it doesn't). It's an
adequate approximation for that file's short-horizon (single-frame) live
prediction use, but not accurate enough to reuse as a fusion factor's own
correctness check, hence Step 0/this script's world-frame conversion.

No bias correction applied (bias0=0) -- Step 1 doesn't optimize bias yet.

Outputs (headless, matplotlib Agg): per-controller PNGs under
<output_dir>/gyro_preint_<ctrl>_hist.png (distribution of |r_gyro(k)|, deg)
and gyro_preint_<ctrl>_overlay.png (per-axis gyro-predicted vs vision-measured
rotvec components, deg, one panel per axis) -- same cross-correlation spirit
as check_headset_imu_axis.py, at frame cadence instead of raw-sample cadence.
Defaults to visualization/step1/ (run per-step outputs live under
visualization/step<N>/ so they're easy to browse -- see the fusion-plan
session), relative to wherever this script is invoked from.

Usage: python gyro_preint_check.py [path/to/config.yml] [output_dir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from compare_vision_mocap import load_pose_csv, load_device_mocap
from src.imu_data import gyro_preint_residual, integrate_gyro_segment, load_and_calibrate_controller_imu
from src.load_config import load_yaml_config, load_json_config
from src.mocap_data import world_pose

from src.mocap_data import controller_imu_files
_IMU_FILES = controller_imu_files()  # lag_ns = -mocap_vision_offset_ns from config.yml (shared with main.py)
_AXIS_NAMES = ("x", "y", "z")


def world_vision_orientations(ctrl_poses: dict, headset_mocap) -> dict:
    """{ts_ns: R (3,3)} world-frame (mocap-world), UN-bridged vision orientation
    per accepted frame -- world_pose(headset,t).compose(T_ctrl_vision(t)).R,
    same frame gyro_body lives in (controller LED/body frame). Frames with no
    headset mocap coverage at t are dropped (world_pose returns None)."""
    out = {}
    for ts_ns, T_ctrl_vision in ctrl_poses.items():
        T_world_headsetImu = world_pose(headset_mocap, ts_ns)
        if T_world_headsetImu is None:
            continue
        out[ts_ns] = T_world_headsetImu.R @ T_ctrl_vision.R
    return out


def gather_gaps(t_gyro, gyro_body, R_world: dict):
    """Per consecutive (sorted) reference-node pair: r_gyro(k) (via the
    gyro_preint_residual factor), plus the raw gyro-predicted and
    vision-measured relative-rotation rotvecs (deg) for the overlay plot.
    Returns dict of equal-length arrays: t_mid_ns, r_gyro_deg (N,3),
    r_gyro_norm_deg (N,), gyro_rotvec_deg (N,3), vision_rotvec_deg (N,3);
    plus n_skipped (gaps outside gyro coverage)."""
    ts_sorted = sorted(R_world.keys())
    t_mid, r_gyro_deg, r_norm_deg, gyro_rv_deg, vision_rv_deg = [], [], [], [], []
    n_skipped = 0
    for tk, tk1 in zip(ts_sorted[:-1], ts_sorted[1:]):
        R0, R1 = R_world[tk], R_world[tk1]
        r_gyro, dt, _ = gyro_preint_residual(t_gyro, gyro_body, tk, tk1, R0, R1)
        if r_gyro is None:
            n_skipped += 1
            continue
        R_gyro_rel = integrate_gyro_segment(t_gyro, gyro_body, tk, tk1)
        R_vision_rel = R0.T @ R1

        t_mid.append((tk + tk1) // 2)
        r_gyro_deg.append(np.degrees(r_gyro))
        r_norm_deg.append(np.degrees(np.linalg.norm(r_gyro)))
        gyro_rv_deg.append(np.degrees(Rotation.from_matrix(R_gyro_rel).as_rotvec()))
        vision_rv_deg.append(np.degrees(Rotation.from_matrix(R_vision_rel).as_rotvec()))

    return {
        "t_mid_ns": np.array(t_mid, dtype=np.int64),
        "r_gyro_deg": np.array(r_gyro_deg),
        "r_gyro_norm_deg": np.array(r_norm_deg),
        "gyro_rotvec_deg": np.array(gyro_rv_deg),
        "vision_rotvec_deg": np.array(vision_rv_deg),
    }, n_skipped


def plot_histogram(ctrl_name: str, r_norm_deg: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r_norm_deg, bins=60, color="steelblue", edgecolor="none")
    ax.axvline(r_norm_deg.mean(), color="crimson", linestyle="--",
               label=f"mean={r_norm_deg.mean():.3f} deg")
    ax.set_xlabel("|r_gyro(k)| (deg)")
    ax.set_ylabel("count")
    ax.set_title(f"{ctrl_name}: gyro preintegration residual distribution (n={len(r_norm_deg)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_overlay(ctrl_name: str, t_mid_ns: np.ndarray, gyro_rv_deg: np.ndarray,
                  vision_rv_deg: np.ndarray, out_path: Path):
    t_s = (t_mid_ns - t_mid_ns[0]) / 1e9
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t_s, gyro_rv_deg[:, i], label="gyro-predicted", color="darkorange", linewidth=0.8)
        ax.plot(t_s, vision_rv_deg[:, i], label="vision-measured (world frame)", color="steelblue",
                 linewidth=0.8, alpha=0.8)
        ax.set_ylabel(f"{_AXIS_NAMES[i]} (deg)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{ctrl_name}: gyro-predicted vs. vision-measured per-gap rotation (world frame)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yml"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("visualization/step1")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml_config(config_path)

    pose_csv_path = config.get("debug", {}).get("pose_csv")
    if not pose_csv_path or not Path(pose_csv_path).exists():
        raise SystemExit(f"debug.pose_csv not set or missing ({pose_csv_path}) -- run main.py first "
                          f"with debug.pose_csv set")

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
        t_gyro, gyro_body, _accel_body = load_and_calibrate_controller_imu(imu_path, ctrl_json_cfg, lag_ns=lag_ns)

        R_world = world_vision_orientations(poses[ctrl_name], headset_mocap)
        data, n_skipped = gather_gaps(t_gyro, gyro_body, R_world)
        n = len(data["r_gyro_norm_deg"])
        if n == 0:
            print(f"[{ctrl_name}] no gaps covered by gyro -- skipping")
            continue

        r = data["r_gyro_norm_deg"]
        print(f"[{ctrl_name}] {n} gaps ({len(R_world)} world-frame vision poses, {n_skipped} gaps skipped "
              f"-- outside gyro coverage): |r_gyro(k)| mean={r.mean():.4f} std={r.std():.4f} "
              f"median={np.median(r):.4f} max={r.max():.4f} deg")

        plot_histogram(ctrl_name, r, out_dir / f"gyro_preint_{ctrl_name}_hist.png")
        plot_overlay(ctrl_name, data["t_mid_ns"], data["gyro_rotvec_deg"], data["vision_rotvec_deg"],
                     out_dir / f"gyro_preint_{ctrl_name}_overlay.png")
        print(f"[{ctrl_name}] saved plots to {out_dir}/gyro_preint_{ctrl_name}_{{hist,overlay}}.png")


if __name__ == "__main__":
    main()
