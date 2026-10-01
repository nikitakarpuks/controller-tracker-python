#!/usr/bin/env python3
"""
visualize_pose_fusion.py -- offline, zoomable Plotly HTML export of
vision/fused/imu position+orientation, replaying the REAL PoseFusionFilter
(src/pose_fusion.py) over debug.vision_pose_csv (this project's own raw,
pre-fusion vision solves -- unaffected by fusion.enabled, see config.yml's
own comment on that key: pose_csv becomes the FUSED pose once fusion is on,
vision_pose_csv never does).

Deliberately decoupled from the live main.py -> Rerun pose_fusion_debug tool:
no live-tracking-loop cost here at all, so IMU is sampled at FULL raw-IMU-
sample resolution (~4x denser than camera frames, found by the user) via
PoseFusionFilter.predict_dense's own dense samples -- not the coarser one-
point-per-camera-frame the live tool is limited to (and which, even pushed to
its own path_stride=1, still wasn't legible in the rerun viewer and cost ~2x
runtime on the LIVE run -- reverted there, moved here instead).

Three traces per chart:
  - vision: the raw per-frame candidate (points -- a discrete measurement).
  - fused:  PoseFusionFilter's reported pose on every ACCEPTED update
            (points -- bootstrap/fail_open/gated_accept only).
  - imu:    the dense, continuously re-integrated dead-reckoned curve
            (line) -- predict_dense's own samples, captured BEFORE each
            try_update() call, so it shows the genuine pre-correction drift
            all the way up to the instant of each correction, not an
            average of the two.

Replays in the SAME world frame the live filter actually operates in --
Camera.T_world_cam's own (not gravity-aligned, not mocap-bridged) extrinsic
frame, see LiveGravityEstimator's own docstring -- so no mocap is needed or
used here, unlike visualize_position_orientation.py's mocap-bridged
comparison (a different, independent check).

Usage: python visualize_pose_fusion.py [path/to/config.yml]
Writes visualization/pose_fusion_{position,orientation}_<ctrl_name>.html
(skipped with a warning if plotly isn't installed -- pip install plotly).
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.imu_data import accel_lever_arm_body
from src.imu_data import create_imu_calib_from_config, load_and_calibrate_controller_imu, dead_reckon_dense
from src.load_config import load_json_config, load_yaml_config
from src.pose_fusion import PoseFusionFilter
from src.transformations import Transform

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

OUTPUT_DIR = Path("visualization")
from src.mocap_data import controller_imu_files
_IMU_FILES = controller_imu_files()  # lag_ns = -mocap_vision_offset_ns from config.yml (shared with main.py)
# dataviz skill's validated categorical palette, slots 1/2/3 (fixed order, not cycled).
_COLOR_VISION = "#2a78d6"  # blue
_COLOR_FUSED  = "#eb6834"  # orange
_COLOR_IMU    = "#7c5cd6"  # violet
_MAX_PAIR_DT_S = 0.15      # same real-tracking-loss-gap cutoff as visualize_position_orientation.py


class _FixedGWorld:
    """PoseFusionFilter only ever reads .g_world off whatever object it's given
    (see LiveGravityEstimator) -- this stands in for the live incremental
    estimator with an offline-bootstrapped constant, exactly as this project's
    own prior validation sessions have done."""
    def __init__(self, g_world):
        self.g_world = g_world


def load_vision_pose_csv(path):
    """{ctrl_name: {ts_ns: (Transform, confidence, error_px)}} from
    debug.vision_pose_csv's own format (see config.yml's comment on that
    key) -- NOT compare_vision_mocap.load_pose_csv, which doesn't know about
    the confidence/error_px columns this needs to reconstruct the same
    accept/reject gating the live run actually had (see try_update's
    _meas_pos_sigma_m -- confidence directly scales the measurement-noise
    sigma, so assuming confidence=1.0 always makes the gate artificially
    tight, found empirically: replaying with a hardcoded 1.0 rejected 247/292
    frames where the live run accepted the large majority)."""
    out = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            ts_ns = int(row[0])
            ctrl_name = row[1]
            qx, qy, qz, qw = (float(x) for x in row[2:6])
            px, py, pz = (float(x) for x in row[6:9])
            confidence = float(row[9]) if len(row) > 9 else 1.0
            error_px = float(row[10]) if len(row) > 10 else 0.0
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            out.setdefault(ctrl_name, {})[ts_ns] = (Transform(R, np.array([px, py, pz])), confidence, error_px)
    return out


def _load_ctrl_imu(ctrl_name: str, config: dict):
    """(t_gyro, gyro_body, accel_body, lever_arm) for one controller -- shared
    by bootstrap_shared_g_world and replay_controller so both use identical
    calibration."""
    imu_rel_path, lag_ns = _IMU_FILES[ctrl_name]
    mav0_root = Path(config["data"]["root"])
    imu_path = mav0_root / imu_rel_path
    ctrl_json_cfg = load_json_config(config["controllers"][ctrl_name]["config_path"])
    imu_calib = create_imu_calib_from_config(ctrl_json_cfg)
    lever_arm = accel_lever_arm_body(imu_calib)
    t_gyro, gyro_body, accel_body = load_and_calibrate_controller_imu(imu_path, ctrl_json_cfg, lag_ns=lag_ns)
    return t_gyro, gyro_body, accel_body, lever_arm


def bootstrap_shared_g_world(config: dict, frames_per_ctrl: dict):
    """ONE g_world shared across every controller, from low-motion samples
    pooled across ALL of them -- matching main.py's live LiveGravityEstimator,
    which really is a single instance shared across controllers (gravity is
    one physical direction in the shared camera-calibration world frame, not
    a per-controller unknown -- see its own docstring). Bootstrapping a
    SEPARATE g_world per controller (an earlier version of this script did
    exactly that) is architecturally wrong and was the dominant cause of a
    much higher reject rate here than the live run actually had (found
    empirically: two different g_world values, one per controller, ~9.61 and
    ~9.68 m/s^2 -- close but not the same physical gravity). Returns
    (g_world, n_used, n_total) -- g_world is None if pooled low-motion
    samples across every controller are still under 20."""
    fusion_cfg = config.get("fusion", {})
    omega_thresh = float(fusion_cfg.get("g_world_low_omega_thresh_rad_s", 0.5))
    rows, n_total = [], 0
    for ctrl_name, frames in frames_per_ctrl.items():
        t_gyro, gyro_body, accel_body, _lever_arm = _load_ctrl_imu(ctrl_name, config)
        for ts_ns, (T, _confidence, _error_px) in frames.items():
            # t_accel is t_gyro itself in this project's IMU loading convention
            # (both calibrated onto one shared timeline) -- see replay_controller.
            if ts_ns < t_gyro[0] or ts_ns > t_gyro[-1]:
                continue
            n_total += 1
            omega = np.array([np.interp(ts_ns, t_gyro, gyro_body[:, i]) for i in range(3)])
            if np.linalg.norm(omega) > omega_thresh:
                continue
            acc = np.array([np.interp(ts_ns, t_gyro, accel_body[:, i]) for i in range(3)])
            rows.append(T.R @ acc)
    if len(rows) < 20:
        return None, len(rows), n_total
    return -np.mean(np.stack(rows), axis=0), len(rows), n_total


def replay_controller(ctrl_name: str, config: dict, vision_frames: dict, g_world: np.ndarray):
    """vision_frames: {ts_ns: (Transform, confidence, error_px)}, this
    controller's raw vision solves (already in the SAME camera-extrinsic
    world frame PoseFusionFilter itself operates in -- see module docstring)
    plus the SAME quality signals its own gate uses. g_world: the ONE value
    shared across every controller (see bootstrap_shared_g_world). Returns a
    dict of (t_s, pos (N,3), euler_deg (N,3)) arrays for vision/fused/imu
    plus gap_spans_s (real tracking-loss gaps, for shading)."""
    t_gyro, gyro_body, accel_body, lever_arm = _load_ctrl_imu(ctrl_name, config)
    t_accel = t_gyro

    fusion_cfg = config.get("fusion", {})

    pf_filter = PoseFusionFilter(
        gyro_data=(t_gyro, gyro_body), accel_data=(t_accel, accel_body),
        lever_arm=lever_arm, g_world_estimator=_FixedGWorld(g_world), cfg=fusion_cfg,
    )

    ts_sorted = sorted(vision_frames.keys())
    t0_rec = ts_sorted[0]

    vision_t, vision_p, vision_eul = [], [], []
    fused_t,  fused_p,  fused_eul  = [], [], []
    gap_spans_s = []
    n_accept = n_reject = 0

    # segments: one dead_reckon_dense call per (anchor -> next accept) stretch,
    # NOT one per frame. Calling PoseFusionFilter.predict_dense every frame
    # during a reject streak re-integrates the SAME growing prefix from
    # scratch each time -- fine live (predict_dense's own window per frame is
    # small), but summed over a whole offline replay it's cubic in the
    # streak length, not quadratic (measured: didn't finish in several
    # minutes on this dataset's real gaps). Snapshotting the anchor once and
    # covering the whole segment in a single dense call is the same total
    # integration work as the live filter would ever do, just done once
    # instead of once-per-frame-observed-so-far.
    segments = []          # [R0, p0, v0, t0, t1], t1 extended in place below
    seg_anchor_ts = None    # identifies which segment (by its anchor ts) is open

    for i, ts in enumerate(ts_sorted):
        T, confidence, error_px = vision_frames[ts]
        t_s = (ts - t0_rec) / 1e9
        vision_t.append(t_s)
        vision_p.append(T.t)
        vision_eul.append(Rotation.from_matrix(T.R).as_euler("xyz", degrees=True))

        if i > 0:
            dt = (ts - ts_sorted[i - 1]) / 1e9
            if dt > _MAX_PAIR_DT_S:
                gap_spans_s.append(((ts_sorted[i - 1] - t0_rec) / 1e9, t_s))

        anchor_ts = pf_filter.last_update_ts_ns
        if anchor_ts is not None:
            if anchor_ts != seg_anchor_ts:
                segments.append([pf_filter.R.copy(), pf_filter.p.copy(), pf_filter.v.copy(), anchor_ts, ts])
                seg_anchor_ts = anchor_ts
            else:
                segments[-1][4] = ts
        else:
            seg_anchor_ts = None

        solution = {"T_world_ctrl": T, "confidence": confidence, "error": error_px}
        accepted = pf_filter.try_update(solution, ts)
        if accepted:
            n_accept += 1
            fused_t.append(t_s)
            fused_p.append(pf_filter.reported_p.copy())
            fused_eul.append(Rotation.from_matrix(pf_filter.reported_R).as_euler("xyz", degrees=True))
        else:
            n_reject += 1

    imu_t, imu_p, imu_eul = [], [], []
    for R0, p0, v0, t0, t1 in segments:
        if t1 <= t0:
            continue
        d_ts, d_p, d_R = dead_reckon_dense(t_gyro, gyro_body, t_accel, accel_body, g_world,
                                            lever_arm, t0, t1, R0, p0, v0, sample_every_n=1)
        for j in range(len(d_ts)):
            imu_t.append((int(d_ts[j]) - t0_rec) / 1e9)
            imu_p.append(d_p[j])
            imu_eul.append(Rotation.from_matrix(d_R[j]).as_euler("xyz", degrees=True))
        # NaN separator: distinct segments don't get a straight line drawn between them.
        imu_t.append(np.nan)
        imu_p.append(np.full(3, np.nan))
        imu_eul.append(np.full(3, np.nan))

    print(f"[{ctrl_name}] replayed {len(ts_sorted)} vision frames: "
          f"{n_accept} accepted, {n_reject} rejected, {len(segments)} dead-reckoning segments, "
          f"{len(imu_t)} dense IMU samples, {len(gap_spans_s)} real tracking-loss gaps")

    def _arr(lst):
        return np.array(lst) if lst else np.zeros((0, 3))

    return {
        "vision": (np.array(vision_t), _arr(vision_p), _arr(vision_eul)),
        "fused":  (np.array(fused_t),  _arr(fused_p),  _arr(fused_eul)),
        "imu":    (np.array(imu_t),    _arr(imu_p),    _arr(imu_eul)),
        "gap_spans_s": gap_spans_s,
    }


def _make_figure(title, y_labels, traces, gap_spans_s, y_suffix, field_idx):
    """field_idx: 1 for position (traces[key][1]), 2 for euler angles
    (traces[key][2]) -- each traces[key] is (t, pos (N,3), euler_deg (N,3)).

    y-axis range is fixed explicitly from vision+fused only (NOT imu): a
    reject streak's dense dead-reckoning can diverge to tens/hundreds of
    meters with no bias correction (see project_imu_fusion_step3_accel.md --
    Finding 10, dead-reckoning fails without online bias estimation), and
    Plotly's default autorange scales every trace on the subplot to fit
    whatever is largest. With imu included in that autorange, a single
    divergent segment squashes vision/fused -- the two trustworthy,
    bounded-range signals -- into a flat line pinned near zero. Found
    empirically: left_controller's imu y-range hit -0.06..111m against
    vision's own -0.06..0.64m. imu itself is NOT clipped or dropped, it's
    still fully plotted and traceable via the interactive zoom/pan -- only
    the default view's range is chosen to keep the two signals a user
    actually reads by default legible."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                         subplot_titles=y_labels)
    for row, label in enumerate(y_labels, start=1):
        for t0, t1 in gap_spans_s:
            fig.add_vrect(x0=t0, x1=t1, fillcolor="rgba(120,120,120,0.10)", line_width=0, row=row, col=1)
        t_imu, y_imu = traces["imu"][0], traces["imu"][field_idx]
        if len(t_imu):
            fig.add_trace(go.Scatter(x=t_imu, y=y_imu[:, row - 1], mode="lines",
                                      name="imu (dead-reckoned)", legendgroup="imu",
                                      showlegend=(row == 1), line=dict(color=_COLOR_IMU, width=1.5),
                                      hovertemplate=f"t=%{{x:.4f}}s<br>{label}=%{{y:.4f}}{y_suffix}<extra>imu</extra>"),
                          row=row, col=1)
        t_vis, y_vis = traces["vision"][0], traces["vision"][field_idx]
        fig.add_trace(go.Scatter(x=t_vis, y=y_vis[:, row - 1], mode="markers",
                                  name="vision (raw)", legendgroup="vision",
                                  showlegend=(row == 1), marker=dict(color=_COLOR_VISION, size=4),
                                  hovertemplate=f"t=%{{x:.4f}}s<br>{label}=%{{y:.4f}}{y_suffix}<extra>vision</extra>"),
                      row=row, col=1)
        t_fus, y_fus = traces["fused"][0], traces["fused"][field_idx]
        fig.add_trace(go.Scatter(x=t_fus, y=y_fus[:, row - 1], mode="markers",
                                  name="fused (reported)", legendgroup="fused",
                                  showlegend=(row == 1), marker=dict(color=_COLOR_FUSED, size=4, symbol="diamond"),
                                  hovertemplate=f"t=%{{x:.4f}}s<br>{label}=%{{y:.4f}}{y_suffix}<extra>fused</extra>"),
                      row=row, col=1)
        fig.update_yaxes(title_text=f"{label} ({y_suffix.strip()})", row=row, col=1,
                          range=_stable_axis_range(y_vis[:, row - 1], y_fus[:, row - 1]))
    fig.update_xaxes(title_text="time (s)", row=3, col=1)
    fig.update_layout(title=title, template="plotly_white", height=850,
                       hovermode="closest", legend=dict(orientation="h", y=1.06))
    return fig


def _stable_axis_range(*columns):
    """Fixed [lo, hi] range spanning only the given (vision, fused) columns,
    padded 12% each side (a flat 0.05 unit floor when they're all ~equal, so
    a near-constant signal still gets visible headroom). Returns None (i.e.
    Plotly autorange) if every column is empty/all-NaN -- e.g. fused on a
    controller with zero accepted frames."""
    vals = np.concatenate([c[~np.isnan(c)] for c in columns if len(c)]) if any(len(c) for c in columns) else np.array([])
    if len(vals) == 0:
        return None
    lo, hi = float(np.min(vals)), float(np.max(vals))
    pad = max((hi - lo) * 0.12, 0.05)
    return [lo - pad, hi + pad]


def main():
    if not PLOTLY_AVAILABLE:
        print("plotly not installed -- skipping (pip install plotly)")
        return
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yml"
    config = load_yaml_config(config_path)
    vision_csv_path = config["debug"].get("vision_pose_csv")
    if not vision_csv_path or not Path(vision_csv_path).exists():
        print(f"debug.vision_pose_csv ({vision_csv_path}) not set or doesn't exist -- "
              f"run main.py once with fusion.enabled: true first")
        return
    frames = load_vision_pose_csv(vision_csv_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fusion_cfg = config.get("fusion", {})
    omega_thresh = float(fusion_cfg.get("g_world_low_omega_thresh_rad_s", 0.5))
    g_world, n_used, n_total = bootstrap_shared_g_world(config, frames)
    if g_world is None:
        print(f"g_world did NOT converge ({n_used}/{n_total} pooled low-motion samples, "
              f"need >=20 at omega_thresh={omega_thresh} rad/s) -- nothing to replay")
        return
    print(f"g_world=[{g_world[0]:.3f} {g_world[1]:.3f} {g_world[2]:.3f}] "
          f"|g|={np.linalg.norm(g_world):.3f} m/s^2 ({n_used}/{n_total} pooled low-motion frames, "
          f"shared across every controller)")

    for ctrl_name in ("left_controller", "right_controller"):
        if ctrl_name not in frames:
            continue
        result = replay_controller(ctrl_name, config, frames[ctrl_name], g_world)
        if result is None:
            continue

        pos_fig = _make_figure(f"{ctrl_name}: position -- vision vs. fused vs. dense IMU dead-reckoning",
                                ["x", "y", "z"], result, result["gap_spans_s"], " m", field_idx=1)
        pos_path = OUTPUT_DIR / f"pose_fusion_position_{ctrl_name}.html"
        pos_fig.write_html(pos_path, include_plotlyjs="cdn")
        print(f"[{ctrl_name}] wrote {pos_path}")

        orient_fig = _make_figure(f"{ctrl_name}: orientation -- vision vs. fused vs. dense IMU dead-reckoning",
                                   ["roll", "pitch", "yaw"], result, result["gap_spans_s"], " deg", field_idx=2)
        orient_path = OUTPUT_DIR / f"pose_fusion_orientation_{ctrl_name}.html"
        orient_fig.write_html(orient_path, include_plotlyjs="cdn")
        print(f"[{ctrl_name}] wrote {orient_path}")


if __name__ == "__main__":
    main()
