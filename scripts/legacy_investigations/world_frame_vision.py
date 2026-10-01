#!/usr/bin/env python3
"""
world_frame_vision.py -- Step 0 of the IMU+vision fusion plan: builds

    T_world_ctrl_vision(t) = T_world_headsetImu_mocap(t) . T_headsetImu_ctrl_vision(t)

i.e. the vision-only controller pose converted into the shared mocap-world
frame, by composing headset mocap's own world-frame pose (world_pose in
src/mocap_data.py) with the vision measurement (already headset-relative, see
compare_vision_mocap.py's docstring) mapped through the LED-ref -> mocap
accel-IMU bridge.

This is NOT a new result -- controller mocap is used here only as a held-out
regression check. Algebraically, world_pose(headset,t).inverse().compose(
world_pose(ctrl,t)) == relative_pose(headset,ctrl,t) (the world frame both
poses share cancels out), so for a FIXED bridge this script's per-frame
world-frame residual should be numerically identical (to float precision) to
compare_vision_mocap.py's relative_pose()-based residual, via a completely
different composition path -- confirming the new world-frame code is bug-free,
not establishing new accuracy. (Not checked against the archived ~3.4mm/1.41deg
(left) / ~4.9mm/1.44deg (right) numbers in data/controllers/mocap_bridge_fit.txt
directly -- those were fit UNWEIGHTED, whereas load_or_fit_bridge here applies
reproj-error weighting whenever pose_csv has that column, so an unweighted-vs
-weighted bridge mismatch would show up as a false "MISMATCH" unrelated to this
script's own correctness. Comparing against a live relative_pose() call with
the SAME bridge isolates that.)

Usage: python world_frame_vision.py [path/to/config.yml]
Requires the same real (out-of-repo) mocap files as compare_vision_mocap.py.
"""
import sys
from pathlib import Path

import numpy as np

from compare_vision_mocap import load_pose_csv, load_device_mocap, load_or_fit_bridge, rotation_angle_deg
from src.load_config import load_yaml_config
from src.mocap_data import world_pose, relative_pose

_REGRESSION_TOL_MM_DEG = (1e-6, 1e-6)  # float-precision tolerance -- the two
                                        # composition paths should agree to
                                        # numerical noise, not just "close"


def world_frame_residuals(ctrl_poses: dict, headset_mocap, ctrl_mocap, bridge):
    """Per-frame residual of T_world_ctrl_vision(t) against world_pose(ctrl_mocap,
    t), skipping frames with no mocap coverage on either device. Also returns,
    per kept frame, the relative_pose()-based residual computed the existing
    (headset-relative) way with the same bridge, as a live regression check --
    see module docstring for why a live check is used instead of the archived
    baseline numbers. Returns (rot_deg array, trans_mm array, n_skipped,
    d_rot_deg array, d_trans_mm array) where the d_* arrays are this frame's
    |world-frame residual - relative_pose residual|, expected ~0."""
    rot_deg, trans_mm, d_rot_deg, d_trans_mm = [], [], [], []
    n_skipped = 0
    for ts_ns, T_ctrl_vision in ctrl_poses.items():
        T_world_headsetImu = world_pose(headset_mocap, ts_ns)
        T_world_ctrl_mocap = world_pose(ctrl_mocap, ts_ns)
        mocap_rel = relative_pose(headset_mocap, ctrl_mocap, ts_ns)
        if T_world_headsetImu is None or T_world_ctrl_mocap is None or mocap_rel is None:
            n_skipped += 1
            continue
        T_headsetImu_ctrl_vision = T_ctrl_vision.compose(bridge)
        T_world_ctrl_vision = T_world_headsetImu.compose(T_headsetImu_ctrl_vision)
        residual = T_world_ctrl_vision.inverse().compose(T_world_ctrl_mocap)
        r_rot = rotation_angle_deg(residual.R)
        r_trans = float(np.linalg.norm(residual.t)) * 1000.0
        rot_deg.append(r_rot)
        trans_mm.append(r_trans)

        ref_residual = T_ctrl_vision.compose(bridge).inverse().compose(mocap_rel)
        ref_rot = rotation_angle_deg(ref_residual.R)
        ref_trans = float(np.linalg.norm(ref_residual.t)) * 1000.0
        d_rot_deg.append(abs(r_rot - ref_rot))
        d_trans_mm.append(abs(r_trans - ref_trans))
    return np.array(rot_deg), np.array(trans_mm), n_skipped, np.array(d_rot_deg), np.array(d_trans_mm)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yml"
    config = load_yaml_config(config_path)

    pose_csv_path = config.get("debug", {}).get("pose_csv")
    if not pose_csv_path or not Path(pose_csv_path).exists():
        raise SystemExit(f"debug.pose_csv not set or missing ({pose_csv_path}) -- run main.py first "
                          f"with debug.pose_csv set")

    poses, errors = load_pose_csv(pose_csv_path)
    headset_mocap = load_device_mocap("headset")

    for ctrl_name in ("left_controller", "right_controller"):
        if ctrl_name not in poses:
            continue
        ctrl_cfg = config["controllers"][ctrl_name]
        ctrl_mocap = load_device_mocap(ctrl_name)
        bridge = load_or_fit_bridge(ctrl_name, ctrl_cfg, poses[ctrl_name], headset_mocap, ctrl_mocap,
                                     errors.get(ctrl_name))

        rot_deg, trans_mm, n_skipped, d_rot_deg, d_trans_mm = world_frame_residuals(
            poses[ctrl_name], headset_mocap, ctrl_mocap, bridge)
        print(f"[{ctrl_name}] world-frame residual ({len(rot_deg)} frames, {n_skipped} skipped): "
              f"rot {rot_deg.mean():.4f}+/-{rot_deg.std():.4f} deg (max {rot_deg.max():.4f}), "
              f"trans {trans_mm.mean():.4f}+/-{trans_mm.std():.4f} mm (max {trans_mm.max():.4f})")

        tol_mm, tol_deg = _REGRESSION_TOL_MM_DEG
        ok = d_trans_mm.max() < tol_mm and d_rot_deg.max() < tol_deg
        status = "OK" if ok else "MISMATCH"
        print(f"    vs. live relative_pose() with the same bridge (per-frame max diff): "
              f"d_trans={d_trans_mm.max():.8f}mm d_rot={d_rot_deg.max():.8f}deg -- {status}")
        print()


if __name__ == "__main__":
    main()
