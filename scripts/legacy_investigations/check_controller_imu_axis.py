#!/usr/bin/env python3
"""check_controller_imu_axis.py -- re-derives the controller gyro sensor->body
axis transform (src/imu_data.py's _DIAG_FLIP) against real VISION-derived
rotation ground truth (a data/*vision_pose_log*.csv from this project's own
tracker), the same methodology documented in src/imu_data.py's module
docstring as the "decisive" full-recording re-check (2509/2610 gyro-vs-vision
rotation-error samples, 8 candidate transforms) that originally validated
diag(1,-1,-1) -- but that check's own script was never committed (lost
scratch script, per that docstring), so there was nothing to re-run when a
new recording needed checking. This is that tool.

IMPORTANT -- ground truth source, and why it's specifically vision, not mocap:
investigating a false IMPLAUSIBLE-vision-jump rejection 2026-09-09 (frame 178,
left_controller, euroc_recording_20260826175510_walk_medium), an EARLIER
version of this script used mocap ground truth instead (each controller's own
T_imu_marker calibration file + world_pose(), the same machinery
HeuristicPoseFusionFilter's mocap-corrected dead-reckoning path already uses
for the HEADSET). That version decisively favored `identity` over
`diag(1,-1,-1)` -- on walk_medium AND, suspiciously, also on static_dark
(contradicting that recording's own documented, vision-validated result).
Re-running the SAME candidate sweep against real vision ground truth instead
(this script) reversed that: diag(1,-1,-1) wins decisively on walk_medium too
(median 1.70deg vs 7+deg for every alternative, including identity), matching
static_dark's original vision-based validation. Conclusion: the controller's
T_imu_marker mocap calibration file most likely targets a DIFFERENT frame
convention than _DIAG_FLIP's live-pipeline "body/LED frame" (an unresolved,
separate question -- NOT the same thing this script is checking, and NOT
something this script can answer), which made the mocap-ground-truth version
of this check silently compare against the wrong reference frame. Vision
ground truth has no such dependency (T_world_ctrl comes straight from this
project's own P3P/PnP solves against the LED model, the same frame gyro_body
already needs to agree with for the implausibility gate to work correctly) --
use vision, not mocap T_imu_marker, for this specific question.

Method: for each pair of consecutive rows in the vision log (same controller,
dt small enough to trust as one continuous track -- see max_dt_s), integrate
calibrated (mix+bias corrected, NOT yet axis-flipped) gyro over that exact
window, apply each candidate sensor->body flip, and compare the resulting
relative rotation against R_gt = R_vision(t0).T @ R_vision(t1) (the same
body-frame-relative convention src.imu_data.integrate_gyro_segment composes
in).

Usage: python check_controller_imu_axis.py [path/to/config.yml] [left|right] \\
           [path/to/vision_pose_log.csv]

The vision log needs columns: timestamp_ns, ctrl_name, qx, qy, qz, qw (one row
per controller per frame) -- this project's own data/vision_pose_log.csv (or
similar) already has this shape. If you don't have one for a new recording,
generate it by running main.py with fusion disabled (raw vision solves only,
no IMU contamination) and capturing T_world_ctrl per frame.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.imu_data import load_imu_csv, create_imu_calib_from_config, integrate_gyro_segment
from src.load_config import load_yaml_config, load_json_config

_FLIP_CANDIDATES = {
    "identity": np.eye(3),
    "X-flip":   np.diag([-1.0,  1.0,  1.0]),
    "Y-flip":   np.diag([ 1.0, -1.0,  1.0]),
    "Z-flip":   np.diag([ 1.0,  1.0, -1.0]),
    "180-X":    np.diag([ 1.0, -1.0, -1.0]),  # == current production _DIAG_FLIP
    "180-Y":    np.diag([-1.0,  1.0, -1.0]),
    "180-Z":    np.diag([-1.0, -1.0,  1.0]),
    "full-flip":np.diag([-1.0, -1.0, -1.0]),
}

from src.mocap_data import controller_imu_files
_IMU_FILES = {side: controller_imu_files()[f"{side}_controller"] for side in ("left", "right")}  # lag_ns from config.yml, shared with main.py


def _load_vision_rows(csv_path: Path, ctrl_key: str):
    import csv
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if r["ctrl_name"] == ctrl_key]
    rows.sort(key=lambda r: int(r["timestamp_ns"]))
    return rows


def check_one(config_path: str, side: str, vision_csv: Path, max_dt_s: float = 0.2):
    config = load_yaml_config(config_path)
    mav0_root = Path(config["data"]["root"])
    ctrl_key = f"{side}_controller"
    ctrl_cfg = config["controllers"][ctrl_key]

    imu_rel, lag_ns = _IMU_FILES[side]
    controller_json = load_json_config(ctrl_cfg["config_path"])
    calib = create_imu_calib_from_config(controller_json)  # entry_index default=1, matches production

    t_imu_raw, gyro_raw, _accel_raw = load_imu_csv(mav0_root / imu_rel)
    t_imu = t_imu_raw + lag_ns
    # NOTE (2026-09-23): the recorded controller CSVs are ALREADY factory-corrected by the Monado driver
    # (see src/imu_data.py module docstring); this diagnostic applies mix+bias again (the legacy chain), so
    # it no longer matches main.py's loader (load_and_calibrate_controller_imu). Kept for the joint-solve
    # archaeology only -- do not compare its absolute bias numbers against the live pipeline.
    gyro_calibrated = calib.gyro.correct(gyro_raw.astype(np.float64))  # mix+bias only, no axis flip

    rows = _load_vision_rows(vision_csv, ctrl_key)
    if len(rows) < 2:
        print(f"  [{ctrl_key}] fewer than 2 vision rows in {vision_csv} -- skipped")
        return

    errors = {name: [] for name in _FLIP_CANDIDATES}
    n_ok = n_skipped = 0
    for i in range(len(rows) - 1):
        t0, t1 = int(rows[i]["timestamp_ns"]), int(rows[i + 1]["timestamp_ns"])
        dt_s = (t1 - t0) / 1e9
        if dt_s <= 0 or dt_s > max_dt_s or t0 < t_imu[0] or t1 > t_imu[-1]:
            n_skipped += 1
            continue
        q0 = [float(rows[i][k]) for k in ("qx", "qy", "qz", "qw")]
        q1 = [float(rows[i + 1][k]) for k in ("qx", "qy", "qz", "qw")]
        R0 = Rotation.from_quat(q0).as_matrix()
        R1 = Rotation.from_quat(q1).as_matrix()
        R_gt = R0.T @ R1
        n_ok += 1
        for name, F in _FLIP_CANDIDATES.items():
            gyro_body = (F @ gyro_calibrated.T).T
            R_rel = integrate_gyro_segment(t_imu, gyro_body, t0, t1)
            if R_rel is None:
                continue
            err_mat = R_gt.T @ R_rel
            cos_a = np.clip((np.trace(err_mat) - 1.0) / 2.0, -1.0, 1.0)
            errors[name].append(float(np.degrees(np.arccos(cos_a))))

    print(f"\n=== {ctrl_key}  (vision log: {vision_csv.name}, "
          f"{n_ok} usable windows, {n_skipped} skipped [gap/out-of-range]) ===")
    ranked = sorted(errors.items(), key=lambda kv: np.median(kv[1]) if kv[1] else float("inf"))
    for name, errs in ranked:
        errs = np.asarray(errs)
        if len(errs) == 0:
            print(f"  {name:12s}  NO DATA")
            continue
        marker = "  <-- current production _DIAG_FLIP" if name == "180-X" else ""
        print(f"  {name:12s}  n={len(errs):4d}  median={np.median(errs):6.2f}deg  "
              f"p90={np.percentile(errs, 90):6.2f}deg  max={errs.max():6.2f}deg{marker}")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yml"
    sides = [sys.argv[2]] if len(sys.argv) > 2 and sys.argv[2] in ("left", "right") else ["left", "right"]
    vision_csv = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data/vision_pose_log.csv")
    if not vision_csv.exists():
        print(f"Vision pose log not found: {vision_csv}\n"
              f"See this script's own module docstring for how to generate one.")
        sys.exit(1)
    for side in sides:
        check_one(config_path, side, vision_csv)


if __name__ == "__main__":
    main()
