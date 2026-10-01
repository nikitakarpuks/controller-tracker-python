#!/usr/bin/env python3
"""
visualize_mocap_offset.py -- interactive sanity check for the per-device
IMU<->mocap FINE time offset src/mocap_data.py applies at runtime (see that
module's docstring on coarse-vs-fine). Modeled on basalt's own
scripts/mocap/mocap_drift_check.py interactive report, but reads only
already-computed artifacts (mocap_filtered/<device>/data.csv, mav0/imu*/data.csv,
drift_check/*/drift_check.json) -- this does NOT re-run the offset search
itself (that needs the compiled basalt_mocap_time_sync binary, which lives
outside this repo). The question this answers is narrower and purely visual:
given the offset main.py is actually about to use (drift_check's value, or a
config override), do the two independently-measured motion signals actually
line up once that offset is applied -- or are we screwing it up.

For each device with complete mocap data (mirrors main.py's own mocap-loading
gating):
  - mocap position (raw, native ~120Hz samples, mocap's own clock) -- did the
    device physically move at all (a flat/motionless device can't validate an
    offset regardless of correctness)
  - mocap |omega| (central-difference from consecutive raw mocap quats) vs raw
    IMU |omega| (gyro magnitude), with mocap's time axis shifted by the fine
    offset being used (device_time + fine_offset_ns ~= mocap_time, so
    device_time ~= mocap_time - fine_offset_ns) -- if the offset is right,
    matching motion features (peaks, pauses) in both curves should visually
    line up on this shared axis; a shift between them means the offset is
    wrong regardless of what any single number claims
  - per-chunk refined offset trend, read directly from a 10chunks
    drift_check.json if one exists alongside the 1chunk one actually used at
    runtime (no recomputation -- just visualizes whether a single constant
    offset looks adequate across the recording, or whether the two clocks are
    drifting)

Usage: python visualize_mocap_offset.py [path/to/config.yml]
Requires mocap.enabled: true and each checked device's mocap_calib_path (or
mocap_fine_offset_override_ns) set, same as main.py. Writes
visualization/mocap_offset_check_<device>.html (one per device; see
OUTPUT_DIR) -- skipped with a warning if plotly isn't installed
(pip install plotly).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

from src.imu_data import load_imu_csv
from src.load_config import load_yaml_config
from src.mocap_data import load_mocap_csv, load_mocap_fine_offset_ns, DRIFT_CHECK_VARIANT

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Same disk-layout knowledge as main.py's mocap-loading block.
_MOCAP_DISK_NAMES = {"headset": "headset", "left_controller": "ctrlleft", "right_controller": "ctrlright"}
_MAV0_IMU_FILES   = {"headset": "imu0/data.csv", "left_controller": "imu1/data.csv", "right_controller": "imu2/data.csv"}

# Reports are written here (one mocap_offset_check_<device>.html per device)
# rather than into each device's own mocap_filtered/<disk_name>/ dir -- keeps
# them alongside this project's other Rerun/viewer output instead of scattered
# across recording directories that may live outside the repo.
OUTPUT_DIR = Path(__file__).resolve().parent / "visualization"


def _mocap_angvel_series(t_ns: np.ndarray, quat_xyzw: np.ndarray, max_pair_dt_s: float = 1.0):
    """Central-difference angular velocity MAGNITUDE (deg/s) from consecutive
    mocap samples -- the geodesic angle between q[i-1] and q[i+1] divided by
    dt equals ||omega||, rotation-invariant, so no T_imu_marker rotation is
    needed just to plot this (unlike relative_pose's actual composition)."""
    ts_s, mags = [], []
    for i in range(1, len(t_ns) - 1):
        dt = (t_ns[i + 1] - t_ns[i - 1]) / 1e9
        if dt <= 0 or dt > max_pair_dt_s:
            continue
        dot = abs(float(np.dot(quat_xyzw[i - 1], quat_xyzw[i + 1])))
        angle_deg = math.degrees(2 * math.acos(min(1.0, max(-1.0, dot))))
        ts_s.append(t_ns[i] / 1e9)
        mags.append(angle_deg / dt)
    return np.array(ts_s), np.array(mags)


def _imu_angvel_series(t_ns: np.ndarray, gyro: np.ndarray):
    return t_ns / 1e9, np.degrees(np.linalg.norm(gyro, axis=1))


def build_report(device_key: str, mocap_csv_path: Path, imu_csv_path: Path,
                  fine_offset_ns: float, offset_source: str,
                  drift_10chunks_path: Path, out_path: Path):
    t_mocap, position, quat_xyzw = load_mocap_csv(mocap_csv_path)
    t_imu, gyro, _accel = load_imu_csv(imu_csv_path)
    t0_s = t_mocap[0] / 1e9

    mocap_ts, mocap_mag = _mocap_angvel_series(t_mocap, quat_xyzw)
    imu_ts, imu_mag = _imu_angvel_series(t_imu, gyro)
    # device_time + fine_offset_ns ~= mocap_time (see src/mocap_data.py) -->
    # shift mocap's own timestamps by -fine_offset_ns to plot it on the
    # device/IMU clock axis, directly overlaying against raw IMU |omega|.
    mocap_ts_aligned = mocap_ts - fine_offset_ns / 1e9

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=False,
        row_heights=[0.28, 0.42, 0.30],
        vertical_spacing=0.08,
        subplot_titles=(
            f"{device_key}: mocap position (native samples -- did it physically move?)",
            f"{device_key}: angular velocity magnitude, mocap shifted by fine offset "
            f"({fine_offset_ns / 1e6:+.1f} ms, from {offset_source}) -- features should line up if correct",
            f"{device_key}: per-chunk refined offset (10chunks drift_check, if available)",
        ),
    )

    t_mocap_s = t_mocap / 1e9 - t0_s
    fig.add_trace(go.Scattergl(x=t_mocap_s, y=position[:, 0], name="p_x", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scattergl(x=t_mocap_s, y=position[:, 1], name="p_y", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scattergl(x=t_mocap_s, y=position[:, 2], name="p_z", line=dict(width=1)), row=1, col=1)
    fig.update_yaxes(title_text="position (m)", row=1, col=1)

    fig.add_trace(go.Scattergl(x=mocap_ts_aligned - t0_s, y=mocap_mag, name="mocap |omega| (offset-aligned)",
                                line=dict(width=1, color="#4a3aa7")), row=2, col=1)
    fig.add_trace(go.Scattergl(x=imu_ts - t0_s, y=imu_mag, name="IMU |omega| (raw gyro)",
                                line=dict(width=0.8, color="#eb6834"), opacity=0.75), row=2, col=1)
    fig.update_yaxes(title_text="ang. vel. (deg/s)", row=2, col=1)
    fig.update_xaxes(title_text="time since recording start (s)", row=2, col=1)

    if drift_10chunks_path and drift_10chunks_path.exists():
        with open(drift_10chunks_path) as f:
            d = json.load(f)
        chunks = d.get("chunks", [])
        xs = [c["chunk_center_s"] - t0_s for c in chunks]
        ys = [c["refined_offset_ns"] / 1e6 for c in chunks]
        hover = [f"chunk {c['chunk_index']}<br>offset={c['refined_offset_ns'] / 1e6:.4f}ms"
                 f"<br>min_cost={c['min_cost_rad_s']:.4f} rad/s" for c in chunks]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="refined offset per chunk (ms)",
                                  line=dict(width=1, color="#2a78d6"), marker=dict(size=9),
                                  text=hover, hoverinfo="text"), row=3, col=1)
        fig.add_hline(y=fine_offset_ns / 1e6, line=dict(width=1, dash="dash", color="#999"),
                      annotation_text="offset actually used", row=3, col=1)
    else:
        fig.add_annotation(text="no 10chunks/drift_check.json found -- run drift_check with num_chunks>1 to "
                                 "see whether a single constant offset is adequate across the recording",
                            xref="x3 domain", yref="y3 domain", x=0.5, y=0.5, showarrow=False, row=3, col=1)
    fig.update_yaxes(title_text="offset (ms)", row=3, col=1)
    fig.update_xaxes(title_text="time since recording start (s)", row=3, col=1)

    fig.update_layout(
        height=1100, hovermode="x unified",
        title=f"{device_key}: mocap<->IMU fine-offset sanity check -- scroll/drag to zoom, double-click to reset",
    )
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
    return out_path


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yml"
    config = load_yaml_config(config_path)

    if not PLOTLY_AVAILABLE:
        raise SystemExit("plotly not installed -- pip install plotly")
    if not config.get("mocap", {}).get("enabled", False):
        raise SystemExit("mocap.enabled is false in this config -- nothing to check")

    recording_root = Path(config["data"]["root"]).parent
    mav0_root      = Path(config["data"]["root"])
    device_cfg = {"headset": config["cameras"],
                  "left_controller": config["controllers"]["left_controller"],
                  "right_controller": config["controllers"]["right_controller"]}
    enabled_devices = ["headset"] + [k for k in ("left_controller", "right_controller")
                                      if config["controllers"].get(k, {}).get("enabled", False)]

    for device_key in enabled_devices:
        dev_cfg    = device_cfg[device_key]
        device_dir = recording_root / "mocap_filtered" / _MOCAP_DISK_NAMES[device_key]
        mocap_csv_path = device_dir / "data.csv"
        imu_csv_path   = mav0_root / _MAV0_IMU_FILES[device_key]
        drift_1chunk_path   = device_dir / "drift_check" / DRIFT_CHECK_VARIANT / "drift_check.json"
        drift_10chunks_path = device_dir / "drift_check" / "10chunks" / "drift_check.json"

        if not mocap_csv_path.exists() or not imu_csv_path.exists():
            print(f"[{device_key}] missing mocap or IMU data, skipping ({mocap_csv_path} / {imu_csv_path})")
            continue

        offset_override_ns = dev_cfg.get("mocap_fine_offset_override_ns")
        if offset_override_ns is not None:
            fine_offset_ns, offset_source = float(offset_override_ns), "config override"
        elif drift_1chunk_path.exists():
            fine_offset_ns = load_mocap_fine_offset_ns(drift_1chunk_path)
            offset_source = f"{DRIFT_CHECK_VARIANT}/drift_check.json"
        else:
            print(f"[{device_key}] no offset override and no {drift_1chunk_path}, skipping")
            continue

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"mocap_offset_check_{device_key}.html"
        build_report(device_key, mocap_csv_path, imu_csv_path, fine_offset_ns, offset_source,
                     drift_10chunks_path, out_path)
        print(f"[{device_key}] wrote {out_path}")


if __name__ == "__main__":
    main()
