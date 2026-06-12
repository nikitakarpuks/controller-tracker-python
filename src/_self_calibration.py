"""
Online self-calibration of inter-camera extrinsics from tracker observations.

Accumulates frames where the primary camera has a high-quality PnP solution,
then runs Levenberg-Marquardt to minimize aux-camera reprojection error with
respect to T_{camN}_{cam_primary} (the transform that maps primary-frame 3D
points into the aux camera frame).

Usage (inside TrackingSystem.update):
    calibrator.add_frame(rvec, tvec, error, inliers, {cam_idx: (pts_prim, blobs)})
    if calibrator.should_run():
        results = calibrator.run()          # optimise + save
        calibrator.apply_to_cameras()       # update Camera objects in-place
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from src.transformations import Transform


class SelfCalibrator:
    def __init__(self, primary_camera, aux_cameras: List, cfg: dict):
        """
        primary_camera : Camera — the anchor camera (extrinsics held fixed)
        aux_cameras    : list[Camera] — cameras to calibrate
        cfg            : self_calibration section from config
        """
        self.primary_camera = primary_camera
        self.aux_cameras: Dict[int, object] = {c.camera_idx: c for c in aux_cameras}

        self.min_primary_error   = float(cfg.get("min_primary_error_px",  0.5))
        self.min_primary_inliers = int(  cfg.get("min_primary_inliers",   6))
        self.min_frames          = int(  cfg.get("min_frames",            50))
        self.max_frames          = int(  cfg.get("max_frames",           300))
        self.run_every_n         = int(  cfg.get("run_every_n_frames",    50))
        self.output_path         = Path( cfg.get("output_path",
                                         "./data/cameras/self_calibrated_extrinsics.json"))
        self.source_calibration  = str(  cfg.get("source_calibration",   ""))

        # Circular observation buffer per aux camera:
        # each entry is (pts_cam_primary: (N,3), blobs_aux: (N,2))
        self._obs: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {
            cid: [] for cid in self.aux_cameras
        }

        self._n_accepted  = 0
        self._last_run_at = 0

        # Latest optimised result per aux camera
        self.results: Dict[int, dict] = {}

    # ------------------------------------------------------------------
    def add_frame(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        primary_error: float,
        primary_inliers: int,
        aux_observations: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ) -> bool:
        """
        Add one frame if it passes quality gates.

        aux_observations : {aux_cam_idx: (pts_cam_primary, blobs_aux)}
            pts_cam_primary : (N, 3) LED positions in primary camera frame
            blobs_aux       : (N, 2) matched 2D detections in aux camera
        """
        if primary_error > self.min_primary_error:
            return False
        if primary_inliers < self.min_primary_inliers:
            return False
        if not aux_observations:
            return False

        accepted = False
        for cid, (pts_prim, blobs_aux) in aux_observations.items():
            if cid not in self._obs or len(pts_prim) < 3:
                continue
            buf = self._obs[cid]
            if len(buf) >= self.max_frames:
                buf.pop(0)
            buf.append((pts_prim.astype(np.float32), blobs_aux.astype(np.float32)))
            accepted = True

        if accepted:
            self._n_accepted += 1
        return accepted

    # ------------------------------------------------------------------
    def should_run(self) -> bool:
        min_obs = min((len(v) for v in self._obs.values()), default=0)
        return (min_obs >= self.min_frames and
                (self._n_accepted - self._last_run_at) >= self.run_every_n)

    # ------------------------------------------------------------------
    def run(self) -> Dict[int, dict]:
        """Optimise T_{camN}_{cam_primary} for each aux camera, save, return results."""
        self._last_run_at = self._n_accepted
        new_results = {}

        for cid, obs in self._obs.items():
            if len(obs) < self.min_frames:
                continue
            result = self._optimise_one(cid, obs)
            if result is not None:
                new_results[cid] = result
                self.results[cid] = result

        if new_results:
            self._save(new_results)

        return new_results

    # ------------------------------------------------------------------
    def _initial_params(self, aux_cam_idx: int) -> np.ndarray:
        """
        T_{camN}_{cam_primary} from current camera calibration.
        Maps primary-frame 3D points into aux-cam frame.
        """
        aux_cam = self.aux_cameras[aux_cam_idx]
        # T_{camN}_{cam_prim} = T_{cam_imu_N} ∘ T_{imu_cam_prim}
        T_camN_camprim = aux_cam.T_cam_imu.compose(self.primary_camera.T_imu_cam)
        rvec, _ = cv2.Rodrigues(T_camN_camprim.R.astype(np.float32))
        return np.concatenate([rvec.ravel(), T_camN_camprim.t]).astype(np.float64)

    # ------------------------------------------------------------------
    def _optimise_one(self, cid: int, obs: list) -> Optional[dict]:
        aux_cam = self.aux_cameras[cid]
        K  = aux_cam.camera_matrix.astype(np.float32)
        dc = aux_cam.dist_coeffs.astype(np.float32)

        pts_list   = [o[0] for o in obs]
        blobs_list = [o[1] for o in obs]

        x0 = self._initial_params(cid)

        def _residuals(x):
            rvec = x[:3].reshape(3, 1).astype(np.float32)
            tvec = x[3:].astype(np.float32)
            parts = []
            for pts, blobs in zip(pts_list, blobs_list):
                proj, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec, tvec, K, dc)
                parts.append((proj.reshape(-1, 2) - blobs).ravel())
            return np.concatenate(parts)

        init_rms = float(np.sqrt(np.mean(_residuals(x0) ** 2)))

        try:
            result = least_squares(
                _residuals, x0, method="lm",
                ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=2000,
            )
        except Exception as exc:
            logger.warning(f"[SelfCal] cam{cid}: optimisation failed: {exc}")
            return None

        opt_rms = float(np.sqrt(np.mean(_residuals(result.x) ** 2)))
        logger.info(
            f"[SelfCal] cam{cid}: {init_rms:.3f}px → {opt_rms:.3f}px  "
            f"frames={len(obs)}  converged={result.success}"
        )

        return {
            "aux_cam_idx": cid,
            "rvec":        result.x[:3].astype(np.float32),
            "tvec":        result.x[3:].astype(np.float32),
            "mean_error":  opt_rms,
            "init_error":  init_rms,
            "converged":   bool(result.success),
            "num_frames":  len(obs),
        }

    # ------------------------------------------------------------------
    def apply_to_cameras(self, results: Optional[Dict[int, dict]] = None):
        """Update aux Camera objects in-place with the optimised transform."""
        if results is None:
            results = self.results
        for cid, res in results.items():
            if cid not in self.aux_cameras:
                continue
            aux_cam = self.aux_cameras[cid]
            R_camN_prim, _ = cv2.Rodrigues(res["rvec"].reshape(3, 1))
            T_camN_prim = Transform(
                R_camN_prim.astype(np.float64),
                res["tvec"].astype(np.float64),
            )
            # T_{imu}_{camN} = T_{imu}_{cam_prim} ∘ T_{cam_prim}_{camN}
            T_camprim_camN = T_camN_prim.inverse()
            T_imu_camN_new = self.primary_camera.T_imu_cam.compose(T_camprim_camN)
            aux_cam.T_imu_cam = Transform(
                T_imu_camN_new.R.astype(np.float64),
                T_imu_camN_new.t.astype(np.float64),
            )
            aux_cam.T_cam_imu = aux_cam.T_imu_cam.inverse()
            logger.info(f"[SelfCal] Applied updated extrinsics to cam{cid}")

    # ------------------------------------------------------------------
    def _save(self, new_results: Dict[int, dict]):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        out: dict = {}
        if self.output_path.exists():
            try:
                with open(self.output_path) as f:
                    out = json.load(f)
            except Exception:
                pass

        out["primary_camera"]    = self.primary_camera.camera_idx
        out["calibrated_at"]     = datetime.utcnow().isoformat()
        out["source_calibration"] = self.source_calibration
        out.setdefault("cameras", {})

        for cid, res in new_results.items():
            R_camN_prim, _ = cv2.Rodrigues(res["rvec"].reshape(3, 1))
            T_camN_prim    = Transform(R_camN_prim.astype(np.float64), res["tvec"].astype(np.float64))
            T_camprim_camN = T_camN_prim.inverse()
            T_imu_camN_new = self.primary_camera.T_imu_cam.compose(T_camprim_camN)

            q_camN_prim = Rotation.from_matrix(R_camN_prim).as_quat()
            q_imu_camN  = Rotation.from_matrix(T_imu_camN_new.R).as_quat()

            out["cameras"][str(cid)] = {
                "T_camN_cam_primary": {
                    "px": float(res["tvec"][0]),
                    "py": float(res["tvec"][1]),
                    "pz": float(res["tvec"][2]),
                    "qx": float(q_camN_prim[0]),
                    "qy": float(q_camN_prim[1]),
                    "qz": float(q_camN_prim[2]),
                    "qw": float(q_camN_prim[3]),
                },
                "T_imu_cam": {
                    "px": float(T_imu_camN_new.t[0]),
                    "py": float(T_imu_camN_new.t[1]),
                    "pz": float(T_imu_camN_new.t[2]),
                    "qx": float(q_imu_camN[0]),
                    "qy": float(q_imu_camN[1]),
                    "qz": float(q_imu_camN[2]),
                    "qw": float(q_imu_camN[3]),
                },
                "mean_reprojection_error_px": res["mean_error"],
                "initial_error_px":           res["init_error"],
                "num_frames":                 res["num_frames"],
                "converged":                  res["converged"],
            }

        with open(self.output_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"[SelfCal] Saved → {self.output_path}")

        if self.source_calibration:
            merged = self.output_path.with_name(self.output_path.stem + "_merged.json")
            self._merge_into_source(Path(self.source_calibration), out, merged)

    # ------------------------------------------------------------------
    def _merge_into_source(self, source_path: Path, self_cal: dict, merged_path: Path):
        """Patch source calibration file with refined T_imu_cam values and save."""
        if not source_path.exists():
            logger.warning(f"[SelfCal] Source calibration not found: {source_path} — skipping merge")
            return
        try:
            with open(source_path) as f:
                src = json.load(f)
        except Exception as exc:
            logger.warning(f"[SelfCal] Cannot read source calibration: {exc}")
            return

        timu_arr = src.get("value0", {}).get("T_imu_cam")
        if not isinstance(timu_arr, list):
            logger.warning("[SelfCal] Source calibration has no value0.T_imu_cam array — skipping merge")
            return

        for cam_str, cam_data in self_cal.get("cameras", {}).items():
            cid = int(cam_str)
            if cid >= len(timu_arr):
                continue
            t = cam_data["T_imu_cam"]
            timu_arr[cid] = {k: t[k] for k in ("px", "py", "pz", "qx", "qy", "qz", "qw")}

        merged_path.parent.mkdir(parents=True, exist_ok=True)
        with open(merged_path, "w") as f:
            json.dump(src, f, indent=4)
        logger.info(f"[SelfCal] Merged calibration → {merged_path}")

    # ------------------------------------------------------------------
    @staticmethod
    def load_and_apply(output_path: Path, cameras: Dict[int, "Camera"],
                       primary_cam_idx: int):
        """Load a previously saved calibration file and apply it to cameras."""
        if not output_path.exists():
            return
        try:
            with open(output_path) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"[SelfCal] Failed to load {output_path}: {exc}")
            return

        if data.get("primary_camera") != primary_cam_idx:
            logger.warning(
                f"[SelfCal] Saved primary cam {data.get('primary_camera')} "
                f"!= configured primary {primary_cam_idx} — skipping load"
            )
            return

        for cam_str, cam_data in data.get("cameras", {}).items():
            cid = int(cam_str)
            if cid not in cameras:
                continue
            timu = cam_data["T_imu_cam"]
            R_new = Rotation.from_quat(
                [timu["qx"], timu["qy"], timu["qz"], timu["qw"]]
            ).as_matrix()
            t_new = np.array([timu["px"], timu["py"], timu["pz"]])
            cameras[cid].T_imu_cam = Transform(R_new, t_new)
            cameras[cid].T_cam_imu = cameras[cid].T_imu_cam.inverse()
            logger.info(
                f"[SelfCal] Loaded extrinsics for cam{cid} "
                f"(err={cam_data.get('mean_reprojection_error_px', '?'):.3f}px  "
                f"frames={cam_data.get('num_frames', '?')})"
            )
