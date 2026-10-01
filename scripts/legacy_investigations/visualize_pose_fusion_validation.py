#!/usr/bin/env python3
"""
visualize_pose_fusion_validation.py -- Phase 1 validation report for
PoseFusionFilter (src/pose_fusion.py): per-gap raw dead-reckoning error (position,
rotation) and the filter's own accept/reject verdict, for every real tracking-loss
gap in the currently-configured pose_csv, plus a well-separated identity-swap
simulation (see this session's plan doc for the full design rationale).

g_world is bootstrapped from data/pose_log_2765frames.csv.bak_before_led_rerun (the
full-recording backup) rather than the current (possibly frame_range-restricted,
too-short-for-its-own-bootstrap) pose_csv -- gravity is a property of the fixed
camera-calibration world frame, valid across any window of the same recording.

Usage: python visualize_pose_fusion_validation.py [path/to/config.yml]
Writes visualization/pose_fusion_phase1_validation.html.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from accel_short_horizon_check import low_motion_bootstrap_g_world
from accel_sign_check import _MAX_PAIR_DT_S, world_vision_poses
from compare_vision_mocap import load_pose_csv, load_device_mocap
from src.imu_data import accel_lever_arm_body
from src.imu_data import create_imu_calib_from_config, load_and_calibrate_controller_imu, predict_world_pose
from src.load_config import load_json_config, load_yaml_config
from src.pose_fusion import PoseFusionFilter
from src.transformations import Transform

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

OUTPUT_DIR = Path("visualization")
from src.mocap_data import controller_imu_files
_IMU_FILES = controller_imu_files()  # lag_ns = -mocap_vision_offset_ns from config.yml (shared with main.py)
_BACKUP_POSE_CSV = "data/pose_log_2765frames.csv.bak_before_led_rerun"
_COLOR_ACCEPT = "#1baf7a"  # dataviz skill palette slot 3 (aqua/green) -- status: good
_COLOR_REJECT = "#e34948"  # dataviz skill palette slot 8 (red) -- status: critical


class _FixedGWorld:
    def __init__(self, g):
        self.g_world = g


def main():
    if not PLOTLY_AVAILABLE:
        print("plotly not installed -- skipping (pip install plotly)")
        return
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yml"
    config = load_yaml_config(config_path)
    poses, _errors = load_pose_csv(config["debug"]["pose_csv"])
    backup_poses, _ = load_pose_csv(_BACKUP_POSE_CSV) if Path(_BACKUP_POSE_CSV).exists() else (poses, None)
    headset_mocap = load_device_mocap("headset")
    mav0_root = Path(config["data"]["root"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []  # (label, duration_ms, pos_err_m, rot_err_deg, accepted, note)
    per_ctrl = {}
    for ctrl_name in ("left_controller", "right_controller"):
        if ctrl_name not in poses:
            continue
        imu_rel_path, lag_ns = _IMU_FILES[ctrl_name]
        ctrl_json_cfg = load_json_config(config["controllers"][ctrl_name]["config_path"])
        imu_calib = create_imu_calib_from_config(ctrl_json_cfg)
        lever_arm = accel_lever_arm_body(imu_calib)
        t_gyro, gyro_body, accel_body = load_and_calibrate_controller_imu(
            mav0_root / imu_rel_path, ctrl_json_cfg, lag_ns=lag_ns)
        world_poses = world_vision_poses(poses[ctrl_name], headset_mocap)
        backup_wp = world_vision_poses(backup_poses[ctrl_name], headset_mocap)
        g_world, n_used, n_total = low_motion_bootstrap_g_world(t_gyro, gyro_body, t_gyro, accel_body, backup_wp)
        per_ctrl[ctrl_name] = dict(gyro_data=(t_gyro, gyro_body), accel_data=(t_gyro, accel_body),
                                    lever_arm=lever_arm, g_world=g_world, world_poses=world_poses)
        print(f"[{ctrl_name}] g_world={g_world} ({n_used}/{n_total} low-motion frames, from backup)")

        ts_sorted = sorted(world_poses.keys())
        filt = PoseFusionFilter((t_gyro, gyro_body), (t_gyro, accel_body), lever_arm,
                                 _FixedGWorld(g_world), {})
        gap_i = 0
        for i in range(1, len(ts_sorted) - 1):
            dt = (ts_sorted[i + 1] - ts_sorted[i]) / 1e9
            if dt <= _MAX_PAIR_DT_S:
                continue
            gap_i += 1
            t0, t1 = ts_sorted[i], ts_sorted[i + 1]
            R0, p0 = world_poses[t0].R, world_poses[t0].t
            R1, p1 = world_poses[t1].R, world_poses[t1].t
            t_prev = ts_sorted[i - 1]
            dt_prev = (t0 - t_prev) / 1e9
            v0 = (p0 - world_poses[t_prev].t) / dt_prev if dt_prev > 0 else np.zeros(3)
            pred = predict_world_pose(t_gyro, gyro_body, t_gyro, accel_body, g_world, lever_arm,
                                       t0, t1, R0, p0, v0)
            if pred is None:
                continue
            R1_pred, p1_pred = pred
            pos_err = float(np.linalg.norm(p1_pred - p1))
            rot_err = float(np.degrees(Rotation.from_matrix(R1_pred.T @ R1).magnitude()))

            filt.reset()
            filt.try_update({"T_world_ctrl": Transform(R0, p0), "error": 0.3, "confidence": 1.0}, t0)
            accepted = filt.try_update({"T_world_ctrl": Transform(R1, p1), "error": 0.5, "confidence": 1.0}, t1)
            rows.append((f"{ctrl_name[:4]} gap #{gap_i}", dt * 1000, pos_err, rot_err, accepted, ""))

    # Swap-scenario probe (well-separated timestamp, as validated interactively)
    if "left_controller" in per_ctrl and "right_controller" in per_ctrl:
        left_wp = per_ctrl["left_controller"]["world_poses"]
        right_wp = per_ctrl["right_controller"]["world_poses"]
        common_ts = sorted(set(left_wp.keys()) & set(right_wp.keys()))
        if len(common_ts) >= 2:
            seps = np.array([np.linalg.norm(left_wp[ts].t - right_wp[ts].t) for ts in common_ts])
            i_best = int(np.argmax(seps[:-1]))
            filt = PoseFusionFilter(per_ctrl["left_controller"]["gyro_data"],
                                     per_ctrl["left_controller"]["accel_data"],
                                     per_ctrl["left_controller"]["lever_arm"],
                                     _FixedGWorld(per_ctrl["left_controller"]["g_world"]), {})
            ts0, ts1 = common_ts[i_best], common_ts[i_best + 1]
            filt.try_update({"T_world_ctrl": Transform(left_wp[ts0].R, left_wp[ts0].t),
                              "error": 0.3, "confidence": 1.0}, ts0)
            swap_accepted = filt.try_update({"T_world_ctrl": Transform(right_wp[ts1].R, right_wp[ts1].t),
                                              "error": 0.3, "confidence": 1.0}, ts1)
            pos_err = float(np.linalg.norm(right_wp[ts1].t - left_wp[ts0].t))
            rows.append(("SWAP SIMULATION (left filter fed right's pose)",
                         (ts1 - ts0) / 1e6, pos_err, float("nan"), swap_accepted,
                         f"sep={seps[i_best]:.2f}m"))

    if not rows:
        print("No gaps found -- nothing to plot")
        return

    labels = [r[0] for r in rows]
    durations = [r[1] for r in rows]
    pos_errs = [r[2] for r in rows]
    rot_errs = [r[3] for r in rows]
    accepted = [r[4] for r in rows]
    notes = [r[5] for r in rows]
    colors = [_COLOR_ACCEPT if a else _COLOR_REJECT for a in accepted]
    hover = [f"duration={d:.0f}ms<br>rot_err={re:.1f}deg<br>{'ACCEPTED' if a else 'REJECTED'}"
             + (f"<br>{n}" if n else "")
             for d, re, a, n in zip(durations, rot_errs, accepted, notes)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=pos_errs, marker_color=colors, hovertext=hover, hoverinfo="text",
        text=[f"{'ACCEPT' if a else 'REJECT'}" for a in accepted], textposition="outside",
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray",
                  annotation_text="1m (rough plausible-controller-motion scale)")
    fig.update_yaxes(title_text="raw dead-reckoning position error (m, log scale)", type="log")
    fig.update_xaxes(tickangle=-30)
    fig.update_layout(
        title="PoseFusionFilter Phase 1 validation: real tracking-loss gaps + identity-swap simulation",
        template="plotly_white", height=550,
        legend=dict(orientation="h", y=1.08),
    )
    # Manual legend proxy (single-series bar chart, color carries accept/reject, not a data series)
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=_COLOR_ACCEPT, name="accepted", showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], marker_color=_COLOR_REJECT, name="rejected", showlegend=True))

    out_path = OUTPUT_DIR / "pose_fusion_phase1_validation.html"
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"wrote {out_path}")
    print(f"\nSummary: {sum(accepted)}/{len(accepted)} accepted, {len(accepted) - sum(accepted)} rejected")


if __name__ == "__main__":
    main()
