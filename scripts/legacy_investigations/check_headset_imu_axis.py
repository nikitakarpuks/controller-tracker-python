#!/usr/bin/env python3
"""
check_headset_imu_axis.py -- tests whether mocap's headset T_imu_marker
rotation actually targets the same axes raw imu0 (mav0/imu0/data.csv)
reports in, i.e. whether the "does Kalibr's T_imu_cam-implied imu frame
match mocap's T_imu_marker-implied imu frame" assumption underlying
compare_vision_mocap.py's headset-side "no bridge needed" choice is
justified. This was flagged as unverified (ambiguity #3) in the conversation
this script comes from -- rather than continue assuming it, test it the
same way the controller's own gyro axis convention was originally validated
in src/imu_data.py's module docstring: cross-correlate against an
INDEPENDENT angular-velocity measurement of the same physical rotation.

Method: central-difference the headset's raw mocap trajectory
(mocap_filtered/headset/data.csv) into a BODY-FRAME angular velocity VECTOR
(not just magnitude, unlike visualize_mocap_offset.py's coarser diagnostic --
a magnitude-only check can't catch an axis-convention mismatch, only a full
vector comparison can), rotate it through candidate T_imu_marker-based
transforms, shift onto imu0's own clock via the fine offset (same convention
as src/mocap_data.py), and correlate per-axis against raw imu0 gyro. If
"T_imu_marker.R applied as-is (no extra correction)" wins clearly over the
alternatives, the headset-side "no bridge needed" assumption is supported.

Usage: python check_headset_imu_axis.py [path/to/config.yml]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.imu_data import load_imu_csv
from src.load_config import load_yaml_config
from src.mocap_data import load_mocap_csv, load_mocap_fine_offset_ns, load_T_imu_marker, DRIFT_CHECK_VARIANT

_RECORDING_ROOT = Path("/home/nikitakarpuks/Downloads/recordings-aug26/euroc_recording_20260826173103_static_dark")
_MOCAP_CALIB_DIR = Path("/home/nikitakarpuks/Downloads/recordings-aug26/mocap_calibrations_for_each_device")


def mocap_body_angvel(t_ns: np.ndarray, quat_xyzw: np.ndarray, max_pair_dt_s: float = 1.0):
    """Central-difference BODY-FRAME angular velocity VECTOR (rad/s) from
    consecutive marker poses -- omega_body ~= rotvec(R[i-1].T @ R[i+1]) / dt,
    the same convention src/controller.py's integrate_gyro_segment already
    uses for composing gyro-integrated rotations (R_new = R_old @ R_rel),
    just read in reverse here. Returns (ts_ns, omega (N,3))."""
    ts, omegas = [], []
    for i in range(1, len(t_ns) - 1):
        dt = (t_ns[i + 1] - t_ns[i - 1]) / 1e9
        if dt <= 0 or dt > max_pair_dt_s:
            continue
        R_prev = Rotation.from_quat(quat_xyzw[i - 1]).as_matrix()
        R_next = Rotation.from_quat(quat_xyzw[i + 1]).as_matrix()
        rotvec = Rotation.from_matrix(R_prev.T @ R_next).as_rotvec()
        ts.append(t_ns[i])
        omegas.append(rotvec / dt)
    return np.array(ts), np.array(omegas)


def resample_vector(t_query: np.ndarray, t_src: np.ndarray, v_src: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t_query, t_src, v_src[:, i]) for i in range(3)], axis=1)


def score_candidate(name: str, R_candidate: np.ndarray, omega_mocap_marker: np.ndarray,
                     mocap_ts_device: np.ndarray, t_imu0: np.ndarray, gyro0: np.ndarray):
    omega_candidate = (R_candidate @ omega_mocap_marker.T).T   # marker-frame vector -> candidate imu-frame
    in_range = (mocap_ts_device >= t_imu0[0]) & (mocap_ts_device <= t_imu0[-1])
    ts = mocap_ts_device[in_range]
    omega_candidate = omega_candidate[in_range]
    gyro0_resampled = resample_vector(ts, t_imu0, gyro0)

    per_axis_corr = [np.corrcoef(omega_candidate[:, i], gyro0_resampled[:, i])[0, 1] for i in range(3)]
    # Overall vector alignment: mean cosine similarity per sample (rotation-
    # invariant sanity companion to the per-axis correlations above -- a real
    # axis-convention match should score well on BOTH).
    norms = np.linalg.norm(omega_candidate, axis=1) * np.linalg.norm(gyro0_resampled, axis=1)
    valid = norms > 1e-6
    cos_sim = np.sum(omega_candidate[valid] * gyro0_resampled[valid], axis=1) / norms[valid]
    print(f"  {name:<28} per-axis corr = [{per_axis_corr[0]:+.3f}, {per_axis_corr[1]:+.3f}, {per_axis_corr[2]:+.3f}]"
          f"   mean cos-sim = {cos_sim.mean():+.3f}   (n={len(ts)})")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yml"
    config = load_yaml_config(config_path)

    mav0_root = Path(config["data"]["root"])
    t_imu0, gyro0, _accel0 = load_imu_csv(mav0_root / "imu0/data.csv")

    device_dir = _RECORDING_ROOT / "mocap_filtered" / "headset"
    t_mocap, _position, quat_xyzw = load_mocap_csv(device_dir / "data.csv")
    fine_offset_ns = load_mocap_fine_offset_ns(device_dir / "drift_check" / DRIFT_CHECK_VARIANT / "drift_check.json")
    T_imu_marker = load_T_imu_marker(_MOCAP_CALIB_DIR / "mocap_calibration_headset.json")

    print(f"headset fine offset: {fine_offset_ns / 1e6:.2f} ms")
    ts_mocap_marker, omega_marker = mocap_body_angvel(t_mocap, quat_xyzw)
    # device_time + fine_offset_ns ~= mocap_time (src/mocap_data.py convention)
    # -> device_time = mocap_time - fine_offset_ns
    ts_device = ts_mocap_marker - fine_offset_ns

    print(f"{len(ts_device)} mocap angular-velocity samples, imu0 range "
          f"[{t_imu0[0]}, {t_imu0[-1]}] ns\n")

    R_flip_options = {
        "identity": np.eye(3),
        "X-flip":   np.diag([-1, 1, 1]),
        "Y-flip":   np.diag([1, -1, 1]),
        "Z-flip":   np.diag([1, 1, -1]),
        "180-X":    np.diag([1, -1, -1]),
        "180-Y":    np.diag([-1, 1, -1]),
        "180-Z":    np.diag([-1, -1, 1]),
    }
    for flip_name, R_flip in R_flip_options.items():
        for transpose_name, R_tim in (("T_imu_marker.R", T_imu_marker.R), ("T_imu_marker.R.T", T_imu_marker.R.T)):
            R_candidate = R_flip @ R_tim
            score_candidate(f"{flip_name} @ {transpose_name}", R_candidate,
                             omega_marker, ts_device, t_imu0, gyro0)


if __name__ == "__main__":
    main()
