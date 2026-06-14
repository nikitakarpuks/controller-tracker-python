"""
Online self-calibration of per-camera intrinsics from tracker observations.

Uses primary-camera PnP solutions as the 3D reference. Factory extrinsics
(T_imu_cam) are held fixed — cameras are physically bolted to the headset.
Only a selected subset of intrinsic parameters is optimised per aux camera.

Which params to optimise is controlled by cfg["optimize_params"], e.g.:
    optimize_params: [fx, fy, cx, cy]          # pinhole only
    optimize_params: [fx, fy, cx, cy, k1, k2]  # + radial distortion

Accumulates all valid frames throughout the session; call run() once at the
end to optimise and print/plot results.

Usage:
    calibrator.add_frame(rvec, tvec, error, inliers, {cam_idx: (pts_prim, blobs)})
    # ... repeat for every frame ...
    calibrator.run()   # once, at end of session
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from scipy.optimize import least_squares


_ALL_PARAMS = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]


class SelfCalibrator:
    def __init__(self, primary_camera, aux_cameras: List, cfg: dict):
        """
        primary_camera : Camera — anchor camera (T_imu_cam held fixed as ground truth)
        aux_cameras    : list[Camera] — cameras whose intrinsics to calibrate
        cfg            : self_calibration section from config
        """
        self.primary_camera = primary_camera
        self.aux_cameras: Dict[int, object] = {c.camera_idx: c for c in aux_cameras}

        self.optimize_params: List[str] = list(cfg.get("optimize_params", ["fx", "fy", "cx", "cy"]))
        unknown = set(self.optimize_params) - set(_ALL_PARAMS)
        if unknown:
            raise ValueError(f"[SelfCal] Unknown optimize_params: {unknown}. Valid: {_ALL_PARAMS}")

        self.min_primary_error   = float(cfg.get("min_primary_error_px",  0.5))
        self.min_primary_inliers = int(  cfg.get("min_primary_inliers",   6))
        self.plots_dir           = Path( cfg.get("plots_dir", "./data/cameras/"))

        # Unbounded accumulation — cleared only on run()
        self._obs: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {
            cid: [] for cid in self.aux_cameras
        }
        self._n_accepted = 0
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
            self._obs[cid].append(
                (pts_prim.astype(np.float32), blobs_aux.astype(np.float32))
            )
            accepted = True

        if accepted:
            self._n_accepted += 1
        return accepted

    # ------------------------------------------------------------------
    def run(self) -> Dict[int, dict]:
        """Optimise intrinsics on all accumulated observations, print and plot results."""
        new_results = {}
        plot_data: Dict[int, dict] = {}

        for cid, obs in self._obs.items():
            if not obs:
                logger.warning(f"[SelfCal] cam{cid}: no observations — skipping")
                continue
            result, pdata = self._optimise_one(cid, obs)
            if result is not None:
                new_results[cid] = result
                self.results[cid] = result
                plot_data[cid] = pdata

        if new_results:
            self._print_results(new_results)
            self._plot_results(new_results, plot_data)

        return new_results

    # ------------------------------------------------------------------
    def _cam_intrinsics_dict(self, cam) -> dict:
        return {
            "fx": float(cam.fx), "fy": float(cam.fy),
            "cx": float(cam.cx), "cy": float(cam.cy),
            "k1": float(cam.k1), "k2": float(cam.k2),
            "p1": float(cam.p1), "p2": float(cam.p2),
            "k3": float(cam.k3), "k4": float(cam.k4),
            "k5": float(cam.k5), "k6": float(cam.k6),
        }

    def _params_to_K_dc(self, x: np.ndarray, base: dict) -> Tuple:
        updated = dict(base)
        for i, name in enumerate(self.optimize_params):
            updated[name] = float(x[i])
        K = np.array([[updated["fx"], 0,            updated["cx"]],
                      [0,            updated["fy"], updated["cy"]],
                      [0,            0,             1           ]], dtype=np.float32)
        dc = np.array([updated[k] for k in ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]],
                      dtype=np.float32)
        return K, dc, updated

    # ------------------------------------------------------------------
    def _optimise_one(self, cid: int, obs: list) -> Tuple[Optional[dict], dict]:
        aux_cam = self.aux_cameras[cid]

        # Factory extrinsics are fixed: primary-cam frame → aux-cam frame
        T_camN_camprim = aux_cam.T_cam_imu.compose(self.primary_camera.T_imu_cam)
        rvec_fixed, _  = cv2.Rodrigues(T_camN_camprim.R.astype(np.float32))
        tvec_fixed     = T_camN_camprim.t.astype(np.float32)

        base = self._cam_intrinsics_dict(aux_cam)
        x0   = np.array([base[p] for p in self.optimize_params], dtype=np.float64)

        pts_list   = [o[0] for o in obs]
        blobs_list = [o[1] for o in obs]

        def _residuals(x):
            K, dc, _ = self._params_to_K_dc(x, base)
            parts = []
            for pts, blobs in zip(pts_list, blobs_list):
                proj, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec_fixed, tvec_fixed, K, dc)
                parts.append((proj.reshape(-1, 2) - blobs).ravel())
            return np.concatenate(parts)

        res_before = _residuals(x0)
        init_rms   = float(np.sqrt(np.mean(res_before ** 2)))

        try:
            result = least_squares(
                _residuals, x0, method="lm",
                ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=2000,
            )
        except Exception as exc:
            logger.warning(f"[SelfCal] cam{cid}: optimisation failed: {exc}")
            return None, {}

        res_after = _residuals(result.x)
        opt_rms   = float(np.sqrt(np.mean(res_after ** 2)))
        _, _, updated = self._params_to_K_dc(result.x, base)

        logger.info(
            f"[SelfCal] cam{cid}: {init_rms:.3f}px → {opt_rms:.3f}px  "
            f"frames={len(obs)}  converged={result.success}"
        )

        res_dict = {
            "aux_cam_idx":          cid,
            "optimized_intrinsics": updated,
            "mean_error":           opt_rms,
            "init_error":           init_rms,
            "converged":            bool(result.success),
            "num_frames":           len(obs),
        }

        # Collect projected positions for plotting
        K0, dc0, _ = self._params_to_K_dc(x0,       base)
        K1, dc1, _ = self._params_to_K_dc(result.x,  base)
        proj_before, proj_after, observed = [], [], []
        for pts, blobs in zip(pts_list, blobs_list):
            pb, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec_fixed, tvec_fixed, K0, dc0)
            pa, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec_fixed, tvec_fixed, K1, dc1)
            proj_before.append(pb.reshape(-1, 2))
            proj_after.append(pa.reshape(-1, 2))
            observed.append(blobs)

        plot_data = {
            "proj_before": np.concatenate(proj_before),
            "proj_after":  np.concatenate(proj_after),
            "observed":    np.concatenate(observed),
            "res_before":  res_before,
            "res_after":   res_after,
        }

        return res_dict, plot_data

    # ------------------------------------------------------------------
    def apply_to_cameras(self, results: Optional[Dict[int, dict]] = None):
        """Update aux Camera objects in-place with optimised intrinsics."""
        if results is None:
            results = self.results
        for cid, res in results.items():
            if cid not in self.aux_cameras:
                continue
            cam  = self.aux_cameras[cid]
            intr = res["optimized_intrinsics"]
            cam.fx = intr["fx"]; cam.fy = intr["fy"]
            cam.cx = intr["cx"]; cam.cy = intr["cy"]
            cam.k1 = intr["k1"]; cam.k2 = intr["k2"]
            cam.p1 = intr["p1"]; cam.p2 = intr["p2"]
            cam.k3 = intr["k3"]; cam.k4 = intr["k4"]
            cam.k5 = intr["k5"]; cam.k6 = intr["k6"]
            cam.camera_matrix = np.array(
                [[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]], dtype=np.float32
            )
            cam.dist_coeffs = np.array(
                [cam.k1, cam.k2, cam.p1, cam.p2, cam.k3, cam.k4, cam.k5, cam.k6],
                dtype=np.float32,
            )
            logger.info(f"[SelfCal] Applied updated intrinsics to cam{cid}")

    # ------------------------------------------------------------------
    def _print_results(self, new_results: Dict[int, dict]):
        for cid, res in new_results.items():
            cam_idx = self.aux_cameras[cid].camera_idx
            intr    = res["optimized_intrinsics"]
            block   = {k: intr[k] for k in _ALL_PARAMS}
            block["rpmax"] = float(getattr(self.aux_cameras[cid], "rpmax", 0.0))
            print(
                f"\n[SelfCal] cam{cam_idx} — paste into calibration file "
                f"at value0.intrinsics[{cam_idx}].intrinsics:\n"
                + json.dumps(block, indent=4)
            )

    # ------------------------------------------------------------------
    def _plot_results(self, new_results: Dict[int, dict], plot_data: Dict[int, dict]):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("[SelfCal] matplotlib not available — skipping plots")
            return

        for cid, res in new_results.items():
            if cid not in plot_data:
                continue
            cam_idx = self.aux_cameras[cid].camera_idx
            pd      = plot_data[cid]
            obs     = pd["observed"]
            pb      = pd["proj_before"]
            pa      = pd["proj_after"]
            rb      = pd["res_before"].reshape(-1, 2)
            ra      = pd["res_after"].reshape(-1, 2)

            fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(
                f"cam{cam_idx}  {res['init_error']:.3f}px → {res['mean_error']:.3f}px"
                f"  ({res['num_frames']} frames)",
                fontsize=13,
            )

            # --- Image-plane scatter ---
            ax_scatter.scatter(obs[:, 0], obs[:, 1], s=12, c="tab:blue",  label="observed",       zorder=3)
            ax_scatter.scatter(pb[:, 0],  pb[:, 1],  s=12, c="tab:red",   label="projected before", zorder=2, alpha=0.6)
            ax_scatter.scatter(pa[:, 0],  pa[:, 1],  s=12, c="tab:green", label="projected after",  zorder=2, alpha=0.6)
            ax_scatter.set_aspect("equal")
            ax_scatter.invert_yaxis()
            ax_scatter.set_xlabel("x (px)"); ax_scatter.set_ylabel("y (px)")
            ax_scatter.set_title("Image-plane positions")
            ax_scatter.legend(markerscale=1.5)

            # --- Residual histogram ---
            err_before = np.linalg.norm(rb, axis=1)
            err_after  = np.linalg.norm(ra, axis=1)
            bins = np.linspace(0, max(err_before.max(), err_after.max()) * 1.05, 50)
            ax_hist.hist(err_before, bins=bins, alpha=0.5, color="tab:red",   label=f"before  rms={res['init_error']:.3f}px")
            ax_hist.hist(err_after,  bins=bins, alpha=0.5, color="tab:green", label=f"after   rms={res['mean_error']:.3f}px")
            ax_hist.set_xlabel("reprojection error (px)")
            ax_hist.set_ylabel("count")
            ax_hist.set_title("Error distribution")
            ax_hist.legend()

            plt.tight_layout()
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.plots_dir / f"self_cal_cam{cam_idx}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            logger.info(f"[SelfCal] Plot saved → {out_path}")
