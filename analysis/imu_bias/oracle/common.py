"""Shared helpers for the IMU-bias oracle study (investigator O).

Everything here reuses the project's own conventions:
  - src.imu_data.load_and_calibrate_controller_imu  (mix+factory-bias, _DIAG_FLIP sensor->body, lag_ns)
  - src.mocap_data.DeviceMocap/world_pose           (vision offset + drift correction driven by config.yml)
Frames:
  gyro_body / accel_body : controller body (= LED reference) frame, factory T=0 calibration applied.
  sensor frame           : body = _DIAG_FLIP @ sensor  (so sensor = _DIAG_FLIP @ body, involution).
  mocap IMU frame        : world_pose(dev, t).R = R_world_imu ; expected ~ sensor frame (bridge R ~ diag(1,-1,-1)).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO = Path("/home/nikitakarpuks/PyCharmProjects/controller-tracker-python")
sys.path.insert(0, str(REPO))
from src.load_config import load_yaml_config, load_json_config                          # noqa: E402
from src.imu_data import (load_and_calibrate_controller_imu, load_imu_csv, _DIAG_FLIP,   # noqa: E402
                          create_imu_calib_from_config, integrate_gyro_segment)
from src.mocap_data import (DeviceMocap, load_mocap_csv, load_mocap_fine_offset_ns, load_T_imu_marker,  # noqa: E402
                            load_vision_offset_ns, load_vision_drift_params, controller_imu_lag_ns,
                            load_mocap_bridge, DRIFT_CHECK_VARIANT)

REC_ROOT = Path("/home/nikitakarpuks/Downloads/recordings-aug26")
EVAL = REPO / "visualization" / "evaluate_2026-09-22"
CONFIG = load_yaml_config(str(REPO / "config" / "config.yml"))
CTRLS = ("left_controller", "right_controller")
_DISK = {"headset": "headset", "left_controller": "ctrlleft", "right_controller": "ctrlright"}
_IMU = {"left_controller": "imu1", "right_controller": "imu2", "headset": "imu0"}
REC_NAMES = ["static_dark", "walk_dark", "static_easy", "static_medium", "static_hard", "walk_easy", "walk_medium", "walk_hard"]


def rec_dir(name):
    return next(p for p in sorted(REC_ROOT.iterdir()) if p.is_dir() and p.name.endswith("_" + name))


def vision_csv(name):
    if name in ("static_hard", "walk_hard"):
        return EVAL / f"{name}_vision_pose.csv"
    return EVAL / name / "vision_pose.csv"


def load_mocap_device(rdir: Path, key: str) -> DeviceMocap:
    d = rdir / "mocap_filtered" / _DISK[key]
    t, pos, quat = load_mocap_csv(d / "data.csv")
    fine = load_mocap_fine_offset_ns(d / "drift_check" / DRIFT_CHECK_VARIANT / "drift_check.json")
    if key == "headset":
        calib = CONFIG["cameras"]["mocap_calib_path"]
        cfg = None
    else:
        calib = CONFIG["controllers"][key]["mocap_calib_path"]
        cfg = CONFIG["controllers"][key]
    voff = load_vision_offset_ns(cfg)
    doff, drate = load_vision_drift_params(cfg)
    return DeviceMocap(t, pos, quat, fine, load_T_imu_marker(calib),
                       max_interp_gap_ns=float(CONFIG["mocap"].get("max_interp_gap_ms", 30.0)) * 1e6,
                       vision_offset_ns=voff, drift_offset_ns=doff, drift_rate_ns_per_ns=drate)


def load_imu(rdir: Path, ctrl: str):
    """(t_ns_lagged int64, gyro_body, accel_body) using the shipped lag."""
    p = rdir / "mav0" / _IMU[ctrl] / "data.csv"
    cfg = load_json_config(str(REPO / CONFIG["controllers"][ctrl]["config_path"]))
    return load_and_calibrate_controller_imu(p, cfg, lag_ns=controller_imu_lag_ns(ctrl, CONFIG))


class MocapOrientation:
    """R_world_imu(t) for one device via SLERP over its whole track, evaluated at CAMERA-clock times
    (same domain as the lagged IMU stamps): the DeviceMocap lookup shift (vision offset + drift + fine)
    is applied to the query, exactly like DeviceMocap.pose_at, but vectorised."""

    def __init__(self, dev: DeviceMocap):
        self.dev = dev
        self.slerp = Slerp(dev.t_ns.astype(np.float64), Rotation.from_quat(dev.quat_xyzw.astype(np.float64)))
        self.R_im = dev.T_imu_marker.R
        self.t_mocap = dev.t_ns

    def lookup_times(self, q_ns):
        q = np.asarray(q_ns, dtype=np.float64)
        d = self.dev
        return q + d.vision_offset_ns + d.drift_offset_ns + d.drift_rate_ns_per_ns * (q - d._drift_pivot_ns) + d.fine_offset_ns

    def valid(self, q_ns, max_gap_ns=30e6):
        tl = self.lookup_times(q_ns)
        ok = (tl >= self.t_mocap[0]) & (tl <= self.t_mocap[-1])
        idx = np.clip(np.searchsorted(self.t_mocap, tl, side="right") - 1, 0, len(self.t_mocap) - 2)
        ok &= (self.t_mocap[idx + 1] - self.t_mocap[idx]) <= max_gap_ns
        return ok

    def R_world_imu(self, q_ns):
        tl = self.lookup_times(q_ns)
        tl = np.clip(tl, self.t_mocap[0], self.t_mocap[-1])
        R_marker = self.slerp(tl)
        return (R_marker * Rotation.from_matrix(self.R_im.T))   # R_world_marker @ R_imu_marker^T


def rotvec_body(R0: np.ndarray, R1: np.ndarray):
    """Relative rotation vector R0^T R1 (body frame at t0 -> t1)."""
    return Rotation.from_matrix(R0.T @ R1).as_rotvec()
DIAG_FLIP = _DIAG_FLIP
