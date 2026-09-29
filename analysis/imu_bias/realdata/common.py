"""common.py -- shared loaders/helpers for the IMU-bias real-data study (investigator R).

Everything here reuses the project's own conventions (src/imu_data.py, src/mocap_data.py); nothing is
re-derived except where noted. All timestamps are on the CAMERA clock (controller IMU stamps are shifted
by lag_ns = -mocap_vision_offset_ns; mocap is looked up through DeviceMocap.pose_at which applies
vision_offset + drift + fine offset), i.e. exactly what main.py does.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path("/home/nikitakarpuks/PyCharmProjects/controller-tracker-python")
sys.path.insert(0, str(REPO))
import os
os.chdir(str(REPO))   # config.yml uses repo-relative paths (./data/...)
from src.imu_data import (create_imu_calib_from_config, integrate_gyro_segment, load_and_calibrate_controller_imu,
                          load_imu_csv, _DIAG_FLIP)
from src.load_config import load_json_config, load_yaml_config
from src.mocap_data import (DRIFT_CHECK_VARIANT, DeviceMocap, controller_imu_lag_ns, load_mocap_bridge,
                            load_mocap_csv, load_mocap_fine_offset_ns, load_T_imu_marker, load_vision_drift_params,
                            load_vision_offset_ns, world_pose)
from src.transformations import Transform

LEVER_MODE = os.environ.get("LEVER", "factory")      # "factory" (what main.py uses today) | "bridge" (accelerometer position from the mocap bridge)
LEVER_TAG = "_leverBridge" if LEVER_MODE == "bridge" else ""
REC_ROOT = Path("/home/nikitakarpuks/Downloads/recordings-aug26")
EVAL_DIR = REPO / "visualization" / "evaluate_2026-09-27_full"  # was evaluate_2026-09-22, no longer on disk;
# repointed 2026-09-27 at the final run (same flat config.yml/vision_pose.csv symlinks as compute_per_frame_errors.py
# expects) -- only vision_pose.csv (raw, unaffected by fusion) and config.yml are read from here, so any batch works
OUT_DIR = REPO / "analysis" / "imu_bias" / "realdata"
CTRLS = ("left_controller", "right_controller")
_DISK = {"headset": "headset", "left_controller": "ctrlleft", "right_controller": "ctrlright"}
_IMU_FILE = {"left_controller": "imu1", "right_controller": "imu2"}


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def rec_dir(name):
    return next(p for p in REC_ROOT.iterdir() if p.name.endswith("_" + name))


def load_device_mocap(recdir, device_key, config):
    d = recdir / "mocap_filtered" / _DISK[device_key]
    cfg = config["cameras"] if device_key == "headset" else config["controllers"][device_key]
    t, pos, q = load_mocap_csv(d / "data.csv")
    fine = load_mocap_fine_offset_ns(d / "drift_check" / DRIFT_CHECK_VARIANT / "drift_check.json")
    T_im = load_T_imu_marker(cfg["mocap_calib_path"])
    gap = float(config.get("mocap", {}).get("max_interp_gap_ms", 30.0)) * 1e6
    off = load_vision_offset_ns(cfg) if device_key != "headset" else 0.0
    d_off, d_rate = load_vision_drift_params(cfg) if device_key != "headset" else (0.0, 0.0)
    return DeviceMocap(t, pos, q, fine, T_im, max_interp_gap_ns=gap, vision_offset_ns=off,
                       drift_offset_ns=d_off, drift_rate_ns_per_ns=d_rate)


class Vision:
    """Raw vision solutions for one controller (rig-frame T_world_ctrl, LED-reference frame)."""

    def __init__(self, csv_path, ctrl):
        ts, R, p, conf, err, nin = [], [], [], [], [], []
        with open(csv_path, newline="") as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                if row[1] != ctrl:
                    continue
                ts.append(int(row[0]))
                R.append(Rotation.from_quat([float(x) for x in row[2:6]]).as_matrix())
                p.append([float(x) for x in row[6:9]])
                conf.append(float(row[9])); err.append(float(row[10])); nin.append(float(row[11]))
        o = np.argsort(ts)
        self.ts = np.array(ts, dtype=np.int64)[o]
        self.R = np.array(R)[o]
        self.p = np.array(p)[o]
        self.conf = np.array(conf)[o]
        self.err = np.array(err)[o]
        self.nin = np.array(nin)[o]

    def strong_mask(self, min_inl=8, max_err=0.5):
        return (self.nin >= min_inl) & (self.err <= max_err) & np.isfinite(self.err)


class Run:
    """One recording: vision, mocap, controller IMU (camera-clock), headset imu0 (raw)."""

    def __init__(self, name, eval_name=None):
        self.name = name
        self.eval = EVAL_DIR / name
        self.config = load_yaml_config(str(self.eval / "config.yml"))
        self.recdir = rec_dir(name)
        self.mav0 = self.recdir / "mav0"
        self.headset_mocap = load_device_mocap(self.recdir, "headset", self.config)
        self.ctrl = {}
        for c in CTRLS:
            cfg_json = load_json_config(self.config["controllers"][c]["config_path"])
            lag = controller_imu_lag_ns(c, self.config)
            t, g, a = load_and_calibrate_controller_imu(self.mav0 / _IMU_FILE[c] / "data.csv", cfg_json, lag_ns=lag)
            calib = create_imu_calib_from_config(cfg_json)
            lever = calib.accel.T_rt.compose(calib.gyro.T_rt.inverse()).t
            bridge_ = load_mocap_bridge(self.config["controllers"][c]["mocap_bridge_path"])
            lever_bridge = -bridge_.R.T @ bridge_.t                      # accelerometer position in the LED/body frame (from the mocap bridge)
            lever_factory = lever
            if LEVER_MODE == "bridge":
                lever = lever_bridge
            t_raw, g_raw, a_raw = load_imu_csv(self.mav0 / _IMU_FILE[c] / "data.csv")
            self.ctrl[c] = dict(
                vision=Vision(self.eval / "vision_pose.csv", c),
                mocap=load_device_mocap(self.recdir, c, self.config),
                bridge=load_mocap_bridge(self.config["controllers"][c]["mocap_bridge_path"]),
                t=t, gyro=g, accel=a, lag=lag, lever=lever, lever_factory=lever_factory, lever_bridge=lever_bridge, calib=calib,
                raw=(t_raw, g_raw.astype(np.float64), a_raw.astype(np.float64)),
            )
        t0, g0, a0 = load_imu_csv(self.mav0 / "imu0" / "data.csv")
        self.imu0 = (t0, g0.astype(np.float64), a0.astype(np.float64))

    # -- mocap helpers (all in the ABSOLUTE mocap-world frame) -------------------------------------
    def R_wh(self, ts):
        T = world_pose(self.headset_mocap, int(ts))
        return None if T is None else T.R

    def T_wh(self, ts):
        return world_pose(self.headset_mocap, int(ts))

    def R_w_ctrl_mocap_led(self, ctrl, ts):
        """Controller LED-frame orientation in mocap world from CONTROLLER mocap (bridge-composed:
        T_w_ledRef = T_w_mocapIMU . bridge^-1)."""
        c = self.ctrl[ctrl]
        T = world_pose(c["mocap"], int(ts))
        if T is None:
            return None
        return (T.compose(c["bridge"].inverse())).R


def step_pairs(ts, ok, max_dt_s=0.040, gap_frames=1):
    """Index pairs (i, j) with j = i+gap_frames in the (already filtered) sequence where ok[i] & ok[j] and the
    time gap is bounded and the frames are consecutive in the ORIGINAL sequence (no dropped frames between)."""
    out = []
    for i in range(len(ts) - gap_frames):
        j = i + gap_frames
        if ok[i] and ok[j] and (ts[j] - ts[i]) / 1e9 <= max_dt_s * gap_frames and ts[j] > ts[i]:
            out.append((i, j))
    return out
