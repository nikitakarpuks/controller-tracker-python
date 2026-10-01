#!/usr/bin/env python3
"""
visualize_position_orientation.py -- interactive position/orientation check:
constellation (vision) tracking overlaid with IMU dead-reckoning, per
controller. Two traces per chart:
  - vision: this project's own LED-constellation tracking result
    (vision/pose_log's T_world_ctrl), wherever a frame was actually accepted.
  - imu: BLIND dead-reckoning (no peeking at vision's true endpoint) bridging
    every REAL tracking-loss gap (occlusion / out of camera view, > this
    project's own _MAX_PAIR_DT_S=0.15s cutoff) -- i.e. exactly what you'd see
    if you only had IMU during a dropout. Reuses the same
    integrate_gyro_segment/integrate_accel_to_position machinery (with the
    confirmed axis-convention fix + lever arm) validated in this session's
    occlusion-gap investigation (visualization/controller_calibration_for_
    basalt/README.md finding 11) -- not new, unvalidated integration math.
    Densely sampled (one point per raw IMU sample within the gap) so the
    bridging curve isn't just a straight line between two points.
Outside tracking-loss gaps, only the vision trace is shown (matching normal
operation, where vision is trusted and an IMU trace reset every ~20ms would
just be a visually-redundant copy of it).

Position: x/y/z (m). Orientation: roll/pitch/yaw (deg, scipy 'xyz' extrinsic
Euler -- can wrap near +-180 deg for very large rotations, a known display
quirk, not a bug).

Usage: python visualize_position_orientation.py [path/to/config.yml]
Writes visualization/{position,orientation}_check_<ctrl_name>.html (skipped
with a warning if plotly isn't installed -- pip install plotly).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from accel_short_horizon_check import low_motion_bootstrap_g_world
from accel_sign_check import _MAX_PAIR_DT_S, world_vision_poses
from compare_vision_mocap import load_pose_csv, load_device_mocap
from src.imu_data import accel_lever_arm_body
from src.imu_data import create_imu_calib_from_config, load_and_calibrate_controller_imu, dead_reckon_dense
from src.load_config import load_json_config, load_yaml_config

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

OUTPUT_DIR = Path("visualization")
from src.mocap_data import controller_imu_files
_IMU_FILES = controller_imu_files()  # lag_ns = -mocap_vision_offset_ns from config.yml (shared with main.py)
# dataviz skill's validated categorical palette, slots 1/2 (fixed order, not cycled).
_COLOR_VISION = "#2a78d6"  # blue
_COLOR_IMU = "#eb6834"     # orange
_COLOR_GAP_SHADE = "rgba(120,120,120,0.10)"


def build_traces(ctrl_name, config, poses, headset_mocap, mav0_root):
    imu_rel_path, lag_ns = _IMU_FILES[ctrl_name]
    imu_path = mav0_root / imu_rel_path
    ctrl_json_cfg = load_json_config(config["controllers"][ctrl_name]["config_path"])
    imu_calib = create_imu_calib_from_config(ctrl_json_cfg)
    lever_arm = accel_lever_arm_body(imu_calib)

    t_gyro, gyro_body, accel_body = load_and_calibrate_controller_imu(imu_path, ctrl_json_cfg, lag_ns=lag_ns)
    t_accel = t_gyro

    world_poses = world_vision_poses(poses[ctrl_name], headset_mocap)
    ts_sorted = sorted(world_poses.keys())
    g_world, n_used, n_total = low_motion_bootstrap_g_world(t_gyro, gyro_body, t_accel, accel_body, world_poses)
    print(f"[{ctrl_name}] {len(ts_sorted)} vision frames, |g_world|={np.linalg.norm(g_world):.3f} m/s^2 "
          f"({n_used}/{n_total} low-motion frames)")

    t0_rec = ts_sorted[0]
    vision_t_s = np.array([(t - t0_rec) / 1e9 for t in ts_sorted])
    vision_p = np.array([world_poses[t].t for t in ts_sorted])
    vision_euler = np.array([Rotation.from_matrix(world_poses[t].R).as_euler("xyz", degrees=True)
                              for t in ts_sorted])

    gap_spans_s = []
    imu_t_s, imu_p, imu_euler = [], [], []
    for i in range(1, len(ts_sorted) - 1):
        dt = (ts_sorted[i + 1] - ts_sorted[i]) / 1e9
        if dt <= _MAX_PAIR_DT_S:
            continue
        t_prev, t0, t1 = ts_sorted[i - 1], ts_sorted[i], ts_sorted[i + 1]
        dt_prev = (t0 - t_prev) / 1e9
        v0 = (world_poses[t0].t - world_poses[t_prev].t) / dt_prev if dt_prev > 0 else np.zeros(3)
        R0, p0 = world_poses[t0].R, world_poses[t0].t

        ts_g, p_g, R_g = dead_reckon_dense(t_gyro, gyro_body, t_accel, accel_body, g_world, lever_arm,
                                            t0, t1, R0, p0, v0)
        eul_g = np.array([Rotation.from_matrix(R_i).as_euler("xyz", degrees=True) for R_i in R_g]) \
            if len(R_g) else np.zeros((0, 3))
        gap_spans_s.append(((t0 - t0_rec) / 1e9, (t1 - t0_rec) / 1e9))
        imu_t_s.append((ts_g - t0_rec) / 1e9)
        imu_p.append(p_g)
        imu_euler.append(eul_g)
        # NaN separator so distinct gap-bridges don't get a straight line drawn between them
        imu_t_s.append(np.array([np.nan]))
        imu_p.append(np.full((1, 3), np.nan))
        imu_euler.append(np.full((1, 3), np.nan))

    imu_t_s = np.concatenate(imu_t_s) if imu_t_s else np.array([])
    imu_p = np.concatenate(imu_p) if imu_p else np.zeros((0, 3))
    imu_euler = np.concatenate(imu_euler) if imu_euler else np.zeros((0, 3))
    print(f"[{ctrl_name}] {len(gap_spans_s)} tracking-loss gaps bridged with IMU-only dead-reckoning")
    return vision_t_s, vision_p, vision_euler, imu_t_s, imu_p, imu_euler, gap_spans_s


def _make_figure(title, y_labels, vision_t, vision_y, imu_t, imu_y, gap_spans_s, y_suffix):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                         subplot_titles=y_labels)
    for row, label in enumerate(y_labels, start=1):
        for t0, t1 in gap_spans_s:
            fig.add_vrect(x0=t0, x1=t1, fillcolor=_COLOR_GAP_SHADE, line_width=0, row=row, col=1)
        fig.add_trace(go.Scatter(x=vision_t, y=vision_y[:, row - 1], mode="lines+markers",
                                  name="vision (constellation)", legendgroup="vision",
                                  showlegend=(row == 1), line=dict(color=_COLOR_VISION, width=2),
                                  marker=dict(size=4),
                                  hovertemplate=f"t=%{{x:.3f}}s<br>{label}=%{{y:.4f}}{y_suffix}<extra>vision</extra>"),
                      row=row, col=1)
        if len(imu_t):
            fig.add_trace(go.Scatter(x=imu_t, y=imu_y[:, row - 1], mode="lines",
                                      name="imu (tracking lost)", legendgroup="imu",
                                      showlegend=(row == 1), line=dict(color=_COLOR_IMU, width=2),
                                      hovertemplate=f"t=%{{x:.3f}}s<br>{label}=%{{y:.4f}}{y_suffix}<extra>imu</extra>"),
                          row=row, col=1)
        fig.update_yaxes(title_text=f"{label} ({y_suffix.strip()})", row=row, col=1)
    fig.update_xaxes(title_text="time (s)", row=3, col=1)
    fig.update_layout(title=title, template="plotly_white", height=850,
                       hovermode="x unified", legend=dict(orientation="h", y=1.06))
    return fig


def main():
    if not PLOTLY_AVAILABLE:
        print("plotly not installed -- skipping (pip install plotly)")
        return
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yml"
    config = load_yaml_config(config_path)
    poses, _errors = load_pose_csv(config["debug"]["pose_csv"])
    headset_mocap = load_device_mocap("headset")
    mav0_root = Path(config["data"]["root"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ctrl_name in ("left_controller", "right_controller"):
        if ctrl_name not in poses:
            continue
        (vision_t, vision_p, vision_euler,
         imu_t, imu_p, imu_euler, gap_spans_s) = build_traces(ctrl_name, config, poses, headset_mocap, mav0_root)

        pos_fig = _make_figure(f"{ctrl_name}: position (constellation vs. IMU dead-reckoning through tracking loss)",
                                ["x", "y", "z"], vision_t, vision_p, imu_t, imu_p, gap_spans_s, " m")
        pos_path = OUTPUT_DIR / f"position_check_{ctrl_name}.html"
        pos_fig.write_html(pos_path, include_plotlyjs="cdn")
        print(f"[{ctrl_name}] wrote {pos_path}")

        orient_fig = _make_figure(f"{ctrl_name}: orientation (constellation vs. IMU dead-reckoning through tracking loss)",
                                   ["roll", "pitch", "yaw"], vision_t, vision_euler, imu_t, imu_euler,
                                   gap_spans_s, " deg")
        orient_path = OUTPUT_DIR / f"orientation_check_{ctrl_name}.html"
        orient_fig.write_html(orient_path, include_plotlyjs="cdn")
        print(f"[{ctrl_name}] wrote {orient_path}")


if __name__ == "__main__":
    main()
