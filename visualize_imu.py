#!/usr/bin/env python3
"""
visualize_imu.py — Standalone controller IMU reader, calibrator, and viewer.

PURPOSE
-------
Read a recording's controller IMU (raw + calibrated), and visualize it —
without running the full tracking pipeline (no blob detection, no pose
search — just CSV loading + Rerun logging, seconds not minutes). Reads
data.root straight from config.yml, same as main.py.

Organized as four classes, one responsibility each:
  ControllerImuData  — read + fix (load + calibrate) one controller's IMU.
  VisionPoseLog      — one controller's accepted-pose trajectory from a
                        previous main.py run's pose_csv, and the checks
                        derived from it (angular velocity, position jumps,
                        gyro-prediction error).
  ImuRerunLogger      — owns the Rerun session; knows how to log either of
                        the above.
  ImuStudy            — orchestrator: config in, loaded data + a populated
                        Rerun viewer out.

The calibration chain itself (mix+bias correction, the empirically-resolved
flip_y @ T_rt.R.T axis transform, clock-offset correction) lives in
src/imu_data.py, not here — that module is shared with the live tracking
pipeline (main.py, src/controller.py) and deliberately has no rerun
dependency; this file is purely the read-fix-visualize workflow built on
top of it.

WORKFLOW
--------
python visualize_imu.py [path/to/config.yml] [max_vision_frames]
  # max_vision_frames: cut the vision-overlay comparison (angular velocity,
  # position-jump, gyro-prediction-error plots) to the first N frames of the
  # recording -- e.g. 500 to stay inside the known-good region and exclude
  # the false-jump frames found after ~frame 500. Does not affect the raw/
  # calibrated IMU plots themselves, only the vision-derived comparisons.
"""
import csv
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation

from src.load_config import load_yaml_config, load_json_config
from src.imu_data import load_imu_csv, load_and_calibrate_controller_imu, integrate_gyro_segment

# Confirmed mapping + measured clock-offset (Stage 1 cross-correlation, this
# recording only — see src/imu_data.py's module docstring for how these were
# resolved; re-measure if this ever runs against different data).
CTRL_IMU_FILES = {
    "left_controller":  ("imu1/data.csv", -5_000_000),
    "right_controller": ("imu2/data.csv", -7_000_000),
}
AXIS_NAMES = ("x", "y", "z")


class ControllerImuData:
    """One controller's IMU stream for a recording: raw as read, and
    calibrated (mix+bias corrected, axis-transformed into body frame,
    clock-offset corrected — see src/imu_data.load_and_calibrate_controller_imu).
    """

    def __init__(self, ctrl_key: str,
                 t_raw: np.ndarray, gyro_raw: np.ndarray, accel_raw: np.ndarray,
                 t_ns: np.ndarray, gyro_body: np.ndarray, accel_body: np.ndarray):
        self.ctrl_key = ctrl_key
        self.t_raw = t_raw
        self.gyro_raw = gyro_raw
        self.accel_raw = accel_raw
        self.t_ns = t_ns
        self.gyro_body = gyro_body
        self.accel_body = accel_body

    @classmethod
    def load(cls, ctrl_key: str, imu_path: Path, controller_cfg: dict, lag_ns: int) -> "ControllerImuData":
        t_raw, gyro_raw, accel_raw = load_imu_csv(imu_path)
        t_ns, gyro_body, accel_body = load_and_calibrate_controller_imu(
            imu_path, controller_cfg, lag_ns=lag_ns,
        )
        return cls(ctrl_key, t_raw, gyro_raw.astype(np.float64), accel_raw.astype(np.float64),
                   t_ns, gyro_body, accel_body)

    @property
    def accel_magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.accel_body, axis=1)


class VisionPoseLog:
    """One controller's accepted-pose trajectory (timestamp, orientation,
    position) from a previous main.py run's pose_csv, plus the checks we've
    been deriving from it throughout this project."""

    def __init__(self, ctrl_key: str, t_ns: np.ndarray, quat_xyzw: np.ndarray, pos: np.ndarray):
        self.ctrl_key = ctrl_key
        self.t_ns = t_ns
        self.quat_xyzw = quat_xyzw
        self.pos = pos

    @classmethod
    def load_all(cls, path: Path, max_frames: Optional[int] = None) -> Dict[str, "VisionPoseLog"]:
        """Same shape as the Stage 1/2 scratch scripts' loader, one instance per ctrl_name.

        max_frames: if given, drops every row timestamped after the max_frames-th
        smallest *unique* timestamp across all controllers combined — i.e. "the
        first N frames of the recording", not "the first N accepted poses of
        each controller" (those two differ whenever a controller drops frames,
        e.g. left_controller's 900/1000 vs right_controller's 918/1000 in the
        Stage 3 run). Use this to cut a known-bad region (like the false-jump
        frames after ~500) out of every derived comparison at once.
        """
        rows = list(csv.DictReader(open(path)))
        cutoff_ns = None
        if max_frames is not None:
            all_ts = np.array(sorted({int(row["timestamp_ns"]) for row in rows}), dtype=np.int64)
            if len(all_ts) > max_frames:
                cutoff_ns = int(all_ts[max_frames - 1])

        out = {}
        for name in {r["ctrl_name"] for r in rows}:
            r = [row for row in rows if row["ctrl_name"] == name]
            t = np.array([int(row["timestamp_ns"]) for row in r], dtype=np.int64)
            order = np.argsort(t)
            t = t[order]
            quat = np.array([[float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])]
                              for row in r])[order]
            pos = np.array([[float(row["px"]), float(row["py"]), float(row["pz"])] for row in r])[order]
            if cutoff_ns is not None:
                keep = t <= cutoff_ns
                t, quat, pos = t[keep], quat[keep], pos[keep]
            out[name] = cls(name, t, quat, pos)
        return out

    def angular_velocity(self):
        """Body-frame angular velocity between consecutive accepted poses
        (Stage 1's imu_vision_sync_check.py technique). Returns (t_mid, omega_mag)."""
        rot = Rotation.from_quat(self.quat_xyzw)
        dt = np.diff(self.t_ns) / 1e9
        rel = rot[:-1].inv() * rot[1:]
        omega = rel.as_rotvec() / dt[:, None]
        t_mid = (self.t_ns[:-1] + self.t_ns[1:]) // 2
        return t_mid, np.linalg.norm(omega, axis=1)

    def position_jumps(self, max_speed_ms: float = 2.0):
        """Frame-to-frame implied speed (Stage 2's outlier-discovery check).
        Returns (t_mid, speed_ms) for every consecutive pair."""
        dt = np.diff(self.t_ns) / 1e9
        speed = np.linalg.norm(np.diff(self.pos, axis=0), axis=1) / dt
        t_mid = (self.t_ns[:-1] + self.t_ns[1:]) // 2
        return t_mid, speed

    def gyro_prediction_error(self, imu: ControllerImuData):
        """For each consecutive pair of accepted poses (t0 -> t1), integrate
        gyro over that real gap (src/imu_data.integrate_gyro_segment — the same
        function Stage 3 calls live for rotation prediction), compose it onto
        the pose at t0 (R_pred = R_0 @ R_rel_gyro, _predict_pose's convention),
        and measure the angle to what vision actually solved at t1. Skips any
        gap the gyro stream doesn't fully cover.

        Reading this plot: large isolated spikes are usually a bad VISION
        frame (the gyro had no way to predict vision's own error), not a bad
        gyro prediction — see Stage 3's false-jump frames for exactly this
        pattern. The steady few-degree baseline between spikes is the real
        gyro-prediction-quality signal.

        Returns (t1_ns[int64], error_deg[float]), one point per covered gap.
        """
        rot = Rotation.from_quat(self.quat_xyzw)
        R_all = rot.as_matrix()
        t_out, err_out = [], []
        for i in range(len(self.t_ns) - 1):
            t0, t1 = int(self.t_ns[i]), int(self.t_ns[i + 1])
            R_rel = integrate_gyro_segment(imu.t_ns, imu.gyro_body, t0, t1)
            if R_rel is None:
                continue
            R_pred = R_all[i] @ R_rel
            cos_angle = np.clip((np.trace(R_pred.T @ R_all[i + 1]) - 1) / 2, -1.0, 1.0)
            err_out.append(np.degrees(np.arccos(cos_angle)))
            t_out.append(t1)
        return np.array(t_out, dtype=np.int64), np.array(err_out, dtype=np.float64)


class ImuRerunLogger:
    """Owns the Rerun session and knows how to log a ControllerImuData /
    VisionPoseLog under a per-controller entity path.

    Entity paths are organised axis-first (e.g. "{ctrl}/gyro/x/raw" and
    "{ctrl}/gyro/x/calibrated" share the parent "{ctrl}/gyro/x"), not
    variant-first — that's what makes build_blueprint() below able to put
    raw and calibrated on the *same* chart per axis: a TimeSeriesView's
    origin pulls in everything under it, so grouping by axis is what makes
    "the same plot" possible instead of a layout guess.
    """

    GRAVITY_MS2 = 9.81
    GRAVITY_LOW_DYNAMICS_TOL_MS2 = 2.0  # matches src/controller.py's _GRAVITY_LOW_DYNAMICS_TOL_MS2

    def __init__(self, app_id: str = "imu_visualizer", spawn: bool = True):
        rr.init(app_id, spawn=spawn)

    @staticmethod
    def _time_col(t_ns: np.ndarray) -> "rr.TimeColumn":
        return rr.TimeColumn("device_time", duration=t_ns.astype("timedelta64[ns]"))

    def log_scalar(self, entity_path: str, t_ns: np.ndarray, values: np.ndarray):
        rr.send_columns(entity_path, indexes=[self._time_col(t_ns)],
                         columns=rr.Scalars.columns(scalars=values))

    def log_raw_vs_calibrated(self, entity_path: str,
                               t_raw: np.ndarray, raw: np.ndarray,
                               t_cal: np.ndarray, calibrated: np.ndarray,
                               names=AXIS_NAMES) -> None:
        """raw/calibrated: (N,3). Logs one {entity_path}/{axis}/raw and
        {entity_path}/{axis}/calibrated pair per axis, so build_blueprint()
        can put each pair on one chart."""
        for i, name in enumerate(names):
            self.log_scalar(f"{entity_path}/{name}/raw", t_raw, raw[:, i])
            self.log_scalar(f"{entity_path}/{name}/calibrated", t_cal, calibrated[:, i])

    def log_imu(self, imu: ControllerImuData) -> None:
        base = imu.ctrl_key
        self.log_raw_vs_calibrated(f"{base}/gyro", imu.t_raw, imu.gyro_raw, imu.t_ns, imu.gyro_body)
        self.log_raw_vs_calibrated(f"{base}/accel", imu.t_raw, imu.accel_raw, imu.t_ns, imu.accel_body)

        accel_mag = imu.accel_magnitude
        self.log_scalar(f"{base}/accel_magnitude/calibrated", imu.t_ns, accel_mag)
        # Flat reference lines marking the gravity-check's low-dynamics gate band —
        # grouped under the same accel_magnitude parent so they land on that chart too.
        self.log_scalar(f"{base}/accel_magnitude/gravity_band_lo", imu.t_ns,
                         np.full_like(accel_mag, self.GRAVITY_MS2 - self.GRAVITY_LOW_DYNAMICS_TOL_MS2))
        self.log_scalar(f"{base}/accel_magnitude/gravity_band_hi", imu.t_ns,
                         np.full_like(accel_mag, self.GRAVITY_MS2 + self.GRAVITY_LOW_DYNAMICS_TOL_MS2))

        print(f"[{base}] logged {len(imu.t_ns)} IMU samples")

    def log_vision_overlay(self, pose_log: VisionPoseLog, imu: Optional[ControllerImuData]) -> None:
        base = pose_log.ctrl_key
        if len(pose_log.t_ns) < 2:
            return

        t_mid, omega_mag = pose_log.angular_velocity()
        self.log_scalar(f"{base}/vision_omega_magnitude", t_mid, omega_mag)

        t_jump, speed = pose_log.position_jumps()
        self.log_scalar(f"{base}/vision_position_speed", t_jump, speed)
        n_flagged = int((speed > 2.0).sum())
        print(f"[{base}] vision overlay: {len(pose_log.t_ns)} poses, "
              f"{n_flagged} frame-to-frame jumps > 2 m/s")

        if imu is not None:
            t_err, err_deg = pose_log.gyro_prediction_error(imu)
            if len(t_err):
                self.log_scalar(f"{base}/gyro_prediction_error_deg", t_err, err_deg)
                print(f"[{base}] gyro prediction error: {len(t_err)} gaps covered, "
                      f"mean={err_deg.mean():.2f}deg  max={err_deg.max():.2f}deg")

    @staticmethod
    def build_blueprint(ctrl_keys, with_vision_overlay) -> "rrb.Blueprint":
        """One TimeSeriesView per (controller, signal) — origin sized so raw
        and calibrated for the same axis land in the same chart (gyro/accel),
        and magnitude + its gravity-band reference lines land in the same
        chart too. Tabs let you flip between controllers instead of scrolling
        a huge grid.
        """
        tabs = []
        for ctrl in ctrl_keys:
            views = []
            for signal in ("gyro", "accel"):
                for axis in AXIS_NAMES:
                    views.append(rrb.TimeSeriesView(
                        origin=f"{ctrl}/{signal}/{axis}", name=f"{signal} {axis}"))
            views.append(rrb.TimeSeriesView(origin=f"{ctrl}/accel_magnitude", name="accel magnitude"))
            if with_vision_overlay:
                views.append(rrb.TimeSeriesView(origin=f"{ctrl}/vision_omega_magnitude", name="vision omega"))
                views.append(rrb.TimeSeriesView(origin=f"{ctrl}/vision_position_speed", name="vision speed"))
                views.append(rrb.TimeSeriesView(origin=f"{ctrl}/gyro_prediction_error_deg", name="gyro pred error"))
            tabs.append(rrb.Grid(*views, name=ctrl, grid_columns=3))
        return rrb.Blueprint(rrb.Tabs(*tabs))


class ImuStudy:
    """Orchestrator: config in, loaded ControllerImuData/VisionPoseLog + a
    populated Rerun viewer out."""

    def __init__(self, config_path: str = "config/config.yml", max_vision_frames: Optional[int] = None):
        self.config = load_yaml_config(config_path)
        self.mav0_root = Path(self.config["data"]["root"])
        self.max_vision_frames = max_vision_frames
        self.imu_by_ctrl: Dict[str, ControllerImuData] = {}
        self.pose_logs: Dict[str, VisionPoseLog] = {}

    def load(self) -> None:
        for ctrl_key, (imu_rel_path, lag_ns) in CTRL_IMU_FILES.items():
            ctrl_cfg = self.config["controllers"].get(ctrl_key, {})
            if not ctrl_cfg.get("enabled", False):
                continue
            imu_path = self.mav0_root / imu_rel_path
            if not imu_path.exists():
                print(f"[{ctrl_key}] IMU file not found ({imu_path}), skipping")
                continue
            self.imu_by_ctrl[ctrl_key] = ControllerImuData.load(
                ctrl_key, imu_path, load_json_config(ctrl_cfg["config_path"]), lag_ns,
            )

        pose_csv_path = self.config.get("debug", {}).get("pose_csv")
        if pose_csv_path and Path(pose_csv_path).exists():
            self.pose_logs = VisionPoseLog.load_all(Path(pose_csv_path), max_frames=self.max_vision_frames)
        else:
            print(f"No pose_csv found at debug.pose_csv ({pose_csv_path!r}) — "
                  f"skipping vision overlay (run main.py first to generate one)")

    def visualize(self, logger: ImuRerunLogger) -> None:
        for imu in self.imu_by_ctrl.values():
            logger.log_imu(imu)
        for pose_log in self.pose_logs.values():
            logger.log_vision_overlay(pose_log, self.imu_by_ctrl.get(pose_log.ctrl_key))

    def run(self) -> None:
        self.load()
        logger = ImuRerunLogger()
        self.visualize(logger)
        rr.send_blueprint(logger.build_blueprint(
            list(self.imu_by_ctrl.keys()), with_vision_overlay=bool(self.pose_logs)))


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yml"
    max_vision_frames = int(sys.argv[2]) if len(sys.argv) > 2 else None
    ImuStudy(config_path, max_vision_frames=max_vision_frames).run()
