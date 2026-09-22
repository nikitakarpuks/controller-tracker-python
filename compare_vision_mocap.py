#!/usr/bin/env python3
"""
compare_vision_mocap.py -- checks vision (pure PnP, no IMU help) against
mocap ground truth in the headset-relative frame the tracker actually
targets (T_headsetImu_ctrlImu), using a per-controller LED-reference-frame
-> mocap accelerometer-IMU-frame bridge that is either loaded from a stored
mocap_bridge_path (config.yml) or fit fresh from this run's data (and then
saved to that path for reuse, if one was given).

Why the bridge can't be derived from the controller's factory InertialSensors
Rt: tried directly (see git history / session notes) -- reconstructing the
bridge from Rt's accelerometer entry (its translation magnitude does match
mocap's T_imu_marker, ruling out gyro's Rt, which is exactly zero by
definition) got within ~9mm / ~5-6deg of the true value, not exact. Root
cause, best evidence available: a few degrees of real calibration tolerance
in the factory Rt itself, amplified into several mm of position error by the
~85mm accelerometer lever arm -- not a bug in this script's math. So the
bridge is fit directly against real vision + mocap data instead (see
fit_bridge below), which reaches sub-degree/sub-2mm and doesn't depend on
trusting Rt's precision at all.

Why headset-relative works with no headset-side bridge needed: config.yml's
cameras.extrinsics_convention is genuinely "T_imu_cam", and the active
calibration file (data/cameras/kalibr_generated.json) has a non-identity
cam0 T_imu_cam entry -- confirmed by inspecting the file directly. src/
camera.py's Camera.T_world_cam always returns T_imu_cam regardless of the
declared convention (that field is documentation only, never branched on in
the code), so vision's "world" frame IS ALREADY headset-imu0's own frame --
no headset-side correction needed to compare against src/mocap_data.
relative_pose(headset, ctrl, t). Separately verified (check_headset_imu_axis.py):
mocap's headset T_imu_marker rotation, applied as-is, correlates far better
against raw imu0 gyro than any axis-flip alternative -- so this assumption
is checked, not just asserted.

Usage: python compare_vision_mocap.py [path/to/config.yml]
Requires: debug.pose_csv populated by a prior main.py run (imu.enabled: false
recommended, so the comparison is pure-vision). Mocap files are read directly
from mocap_calibrations_for_each_device and mocap_filtered/{headset,ctrlleft,
ctrlright}/ (real paths for this recording, outside the repo) -- independent
of config.yml's mocap.enabled/mocap_calib_path, which aren't required here.
Each controller's config.yml entry may set mocap_bridge_path (e.g.
"./data/mocap_calib/controller_left_mocap_bridge.json"): if that file exists
it's loaded and used directly (no fitting); if the path is set but the file
doesn't exist yet, this run fits the bridge and writes it there for next time;
if unset, the bridge is fit fresh every run without being saved anywhere.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.load_config import load_yaml_config
from src.mocap_data import load_mocap_csv, load_mocap_fine_offset_ns, load_T_imu_marker, load_mocap_bridge, \
                            load_vision_offset_ns, load_vision_drift_params, DeviceMocap, relative_pose, \
                            DRIFT_CHECK_VARIANT
from src.transformations import Transform

# Real mocap files for whichever recording config.yml's data.root currently
# points at -- outside the repo, same paths used throughout the imu/mocap
# organization work this script builds on. Derived from config.yml (rather
# than hardcoded to one specific recording) so every script importing
# load_device_mocap automatically follows data.root when it's switched to a
# different recording, instead of silently comparing against stale mocap
# data for the wrong recording.
_CONFIG = load_yaml_config("config/config.yml")
_RECORDING_ROOT = Path(_CONFIG["data"]["root"]).parent
_MOCAP_CALIB_DIR = _RECORDING_ROOT.parent / "mocap_calibrations_for_each_device"

_MOCAP_DISK_NAMES = {"headset": "headset", "left_controller": "ctrlleft", "right_controller": "ctrlright"}
_MOCAP_CALIB_FILES = {"headset": "mocap_calibration_headset.json",
                       "left_controller": "controller_left_calib.json",
                       "right_controller": "controller_right_calib.json"}

# Starting point ONLY for fit_bridge's search -- not claimed to be exact (the
# real fit lands close to but not exactly here, see fit_bridge's docstring).
# A proper 180-degree rotation about X: two independently-designed coordinate
# systems (this WMR-based controller format vs. mocap/Motive's own Y-up
# convention) disagreeing on "up" always shows up as a two-axis flip like
# this (a single-axis flip would reverse handedness, which two real rigid
# frames can never do) -- picked as the fit's starting rotation for exactly
# that reason, not because it's already known to be the final answer.
_ROTATION_SEED = np.diag([1.0, -1.0, -1.0])


def load_pose_csv(path):
    """(poses, reproj_err_px) -- both {ctrl_name: {timestamp_ns: value}} -- from
    debug.pose_csv's own format (timestamp_ns, ctrl_name, qx,qy,qz,qw, px,py,pz,
    reproj_err_px), one entry per accepted frame. reproj_err_px is that frame's
    solved pose's own mean PnP reprojection error (sol['error'] in main.py) --
    used to weight frames in fit_bridge's fit_constant_correction (a 0.3px frame
    is a much better-conditioned PnP solve than a 1.5px frame, so it should pull
    the fit harder). Missing/older CSVs without that column read as NaN, which
    fit_bridge treats as "unweighted" (falls back to an unweighted mean)."""
    poses, errors = {}, {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        has_err = "reproj_err_px" in header
        for row in reader:
            ts_ns = int(row[0])
            ctrl_name = row[1]
            qx, qy, qz, qw = (float(x) for x in row[2:6])
            px, py, pz = (float(x) for x in row[6:9])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([px, py, pz])
            poses.setdefault(ctrl_name, {})[ts_ns] = Transform(R, t)
            errors.setdefault(ctrl_name, {})[ts_ns] = float(row[9]) if has_err else float("nan")
    return poses, errors


def load_device_mocap(device_key: str) -> DeviceMocap:
    device_dir = _RECORDING_ROOT / "mocap_filtered" / _MOCAP_DISK_NAMES[device_key]
    calib_path = _MOCAP_CALIB_DIR / _MOCAP_CALIB_FILES[device_key]
    t, position, quat_xyzw = load_mocap_csv(device_dir / "data.csv")
    fine_offset_ns = load_mocap_fine_offset_ns(device_dir / "drift_check" / DRIFT_CHECK_VARIANT / "drift_check.json")
    T_imu_marker = load_T_imu_marker(calib_path)
    # Headset has no mocap_vision_offset_ns (its entry lives under cameras:, and is deliberately
    # left at 0 -- weakly constrained, see src/mocap_data.py); controllers read their own.
    dev_cfg = _CONFIG["controllers"].get(device_key) if device_key != "headset" else None
    drift_offset_ns, drift_rate_ns_per_ns = load_vision_drift_params(dev_cfg)
    return DeviceMocap(t, position, quat_xyzw, fine_offset_ns, T_imu_marker,
                       vision_offset_ns=load_vision_offset_ns(dev_cfg),
                       drift_offset_ns=drift_offset_ns, drift_rate_ns_per_ns=drift_rate_ns_per_ns)


def rotation_angle_deg(R: np.ndarray) -> float:
    cos_angle = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


_MIN_REPROJ_ERR_PX = 0.05  # floor for the 1/err^2 weighting below -- guards against a
                            # near-zero reprojection error (LED blob detection has no
                            # real sub-this precision anyway) blowing up its weight


def fit_constant_correction(residuals: list, weights: np.ndarray = None) -> Transform:
    """Best-fit constant Transform explaining a list of per-frame residual
    Transforms. Rotation: (weighted) chordal-L2 mean via scipy's Rotation.mean()
    (appropriate since the per-frame spread is small -- no outlier robustness
    needed here). Translation: (weighted) mean. weights=None = unweighted, as
    before."""
    Rs = Rotation.from_matrix(np.stack([r.R for r in residuals]))
    R_mean = Rs.mean(weights=weights).as_matrix()
    t_mean = np.average(np.stack([r.t for r in residuals]), axis=0, weights=weights)
    return Transform(R_mean, t_mean)


def residual_stats(ctrl_poses: dict, headset_mocap: DeviceMocap, ctrl_mocap: DeviceMocap, bridge: Transform):
    """Per-frame residual of T_ctrl_vision(t).compose(bridge) against
    relative_pose(headset_mocap, ctrl_mocap, t), skipping frames with no
    mocap coverage (see src/mocap_data.py's max_interp_gap_ns). Returns
    (rot_deg array, trans_mm array, n_skipped, list of residual Transforms,
    list of the kept frames' timestamp_ns -- same order as the other lists,
    lets callers align a weight array against them)."""
    rot_deg, trans_mm, residuals, kept_ts = [], [], [], []
    n_skipped = 0
    for ts_ns, T_ctrl_vision in ctrl_poses.items():
        mocap_rel = relative_pose(headset_mocap, ctrl_mocap, ts_ns)
        if mocap_rel is None:
            n_skipped += 1
            continue
        residual = T_ctrl_vision.compose(bridge).inverse().compose(mocap_rel)
        rot_deg.append(rotation_angle_deg(residual.R))
        trans_mm.append(float(np.linalg.norm(residual.t)) * 1000.0)
        residuals.append(residual)
        kept_ts.append(ts_ns)
    return np.array(rot_deg), np.array(trans_mm), n_skipped, residuals, kept_ts


def _reproj_weights(kept_ts: list, errors: dict) -> np.ndarray:
    """1/err^2 per kept frame (inverse-variance weighting: a frame's reprojection
    error is a proxy for that frame's pose noise, so this is the standard way to
    let well-conditioned PnP solves outweigh noisy ones in a mean/chordal-mean
    fit). errors=None or an all-NaN column (older pose_csv) falls back to
    unweighted (uniform weights)."""
    if errors is None:
        return None
    err = np.array([errors.get(ts, float("nan")) for ts in kept_ts])
    if np.all(np.isnan(err)):
        return None
    err = np.nan_to_num(err, nan=np.nanmean(err))
    return 1.0 / np.maximum(err, _MIN_REPROJ_ERR_PX) ** 2


def fit_bridge(ctrl_poses: dict, headset_mocap: DeviceMocap, ctrl_mocap: DeviceMocap, errors: dict = None):
    """Fits the LED-reference-frame -> mocap accelerometer-IMU-frame bridge
    directly from this run's vision + mocap data -- see module docstring for
    why this replaces trying to derive it from factory Rt. Two-step (seed +
    correction, composed) rather than a single averaging pass only because
    the seed needs to be a real rotation for fit_constant_correction's
    chordal mean to behave sensibly starting from a reasonable neighborhood;
    the seed itself does not appear in the final result's accuracy, only in
    how the search is set up. errors (optional {timestamp_ns: reproj_err_px})
    weights the fit by 1/err^2 -- see _reproj_weights. Returns (bridge
    Transform, rot_deg array, trans_mm array, n_skipped)."""
    seed = Transform(_ROTATION_SEED, np.zeros(3))
    _, _, _, seed_residuals, kept_ts = residual_stats(ctrl_poses, headset_mocap, ctrl_mocap, seed)
    weights = _reproj_weights(kept_ts, errors)
    correction = fit_constant_correction(seed_residuals, weights)
    bridge = seed.compose(correction)
    rot_deg, trans_mm, n_skipped, _, _ = residual_stats(ctrl_poses, headset_mocap, ctrl_mocap, bridge)
    return bridge, rot_deg, trans_mm, n_skipped


def save_bridge(path: Path, bridge: Transform, rot_deg: np.ndarray, trans_mm: np.ndarray,
                 n_frames: int, recording_name: str):
    qx, qy, qz, qw = Rotation.from_matrix(bridge.R).as_quat()
    doc = {
        "value0": {
            "comment": (
                "Empirically-fit LED-reference-frame -> mocap accelerometer-IMU-frame bridge "
                "(T_ledRef_mocapAccelImu), fit directly against real vision + mocap data via "
                "compare_vision_mocap.py -- NOT derived from this controller's factory InertialSensors "
                "Rt (that path was tried and found to be off by several mm/degrees, see module docstring). "
                f"residual after fit: rot {rot_deg.mean():.3f}+/-{rot_deg.std():.3f} deg, "
                f"trans {trans_mm.mean():.3f}+/-{trans_mm.std():.3f} mm."
            ),
            "T_ledRef_mocapAccelImu": {
                "qx": float(qx), "qy": float(qy), "qz": float(qz), "qw": float(qw),
                "px": float(bridge.t[0]), "py": float(bridge.t[1]), "pz": float(bridge.t[2]),
            },
            "fitted_from_recording": recording_name,
            "n_frames": n_frames,
            "residual_rotation_deg_mean": float(rot_deg.mean()),
            "residual_rotation_deg_std": float(rot_deg.std()),
            "residual_translation_mm_mean": float(trans_mm.mean()),
            "residual_translation_mm_std": float(trans_mm.std()),
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)


def load_or_fit_bridge(ctrl_name: str, ctrl_cfg: dict, ctrl_poses: dict,
                        headset_mocap: DeviceMocap, ctrl_mocap: DeviceMocap, ctrl_errors: dict = None) -> Transform:
    bridge_path = ctrl_cfg.get("mocap_bridge_path")
    if bridge_path and Path(bridge_path).exists():
        print(f"[{ctrl_name}] loaded mocap bridge from {bridge_path} (no fitting)")
        return load_mocap_bridge(bridge_path)

    bridge, rot_deg, trans_mm, n_skipped = fit_bridge(ctrl_poses, headset_mocap, ctrl_mocap, ctrl_errors)
    print(f"[{ctrl_name}] fit mocap bridge from {len(rot_deg)} frames ({n_skipped} skipped, no mocap coverage): "
          f"rot {rot_deg.mean():.3f}±{rot_deg.std():.3f} deg, trans {trans_mm.mean():.3f}±{trans_mm.std():.3f} mm")
    if bridge_path:
        save_bridge(Path(bridge_path), bridge, rot_deg, trans_mm, len(rot_deg), _RECORDING_ROOT.name)
        print(f"[{ctrl_name}] saved to {bridge_path} for reuse next run")
    return bridge


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

        rot_deg, trans_mm, n_skipped, _, _ = residual_stats(poses[ctrl_name], headset_mocap, ctrl_mocap, bridge)
        print(f"[{ctrl_name}] this run's residual using that bridge ({len(rot_deg)} frames, {n_skipped} skipped): "
              f"rot {rot_deg.mean():.3f}±{rot_deg.std():.3f} deg (max {rot_deg.max():.3f}), "
              f"trans {trans_mm.mean():.3f}±{trans_mm.std():.3f} mm (max {trans_mm.max():.3f})\n")


if __name__ == "__main__":
    main()
