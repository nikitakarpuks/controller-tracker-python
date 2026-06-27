import copy
import cv2
import numpy as np
from collections import deque
from loguru import logger
from typing import List, Tuple, Optional, Dict, Set

from src._pnp import _project_points
from src._visibility import _visible_mask
from src.camera import Camera
from src.debug_config import is_deep
from src.transformations import Transform
from src._self_calibration import SelfCalibrator


# =========================================================
# 1. DATA STRUCTURES
# =========================================================

class ControllerLED:
    def __init__(self, position: np.ndarray, normal: np.ndarray):
        self.position = np.asarray(position, dtype=np.float32).reshape(3)
        self.normal = np.asarray(normal, dtype=np.float32).reshape(3)


class ControllerModel:
    def __init__(self, leds: List[ControllerLED], name: str):
        self.name = name
        self.leds = leds

        # Precompute for speed
        self.positions = np.stack([l.position for l in leds])
        self.normals = np.stack([l.normal for l in leds])


def create_leds_from_config(cfg) -> List[ControllerLED]:
    leds_cfg = cfg["CalibrationInformation"]["ControllerLeds"]

    return [
        ControllerLED(
            position=np.array(led["Position"], dtype=np.float32),
            normal=np.array(led["Normal"], dtype=np.float32),
        )
        for led in leds_cfg
    ]


# =========================================================
# 2. GEOMETRY — fit frustum + bake handle primitives
# =========================================================

def mirror_primitives(prim_cfg: dict) -> dict:
    """Return a YZ-plane (X-reflected) copy of a handle_primitives config block."""
    p = copy.deepcopy(prim_cfg)
    for b in p.get("boxes", []):
        b["center"][0] = -b["center"][0]
        if "axes" in b:
            for row in b["axes"]:
                row[0] = -row[0]
    for cy in p.get("cylinders", []):
        cy["center"][0] = -cy["center"][0]
        cy["angle"] = -cy.get("angle", 0.0)
    return p


# =========================================================
# 3. TRACKER (per camera + controller)
# =========================================================

class CameraTracker:
    def __init__(self, camera: Camera, model: ControllerModel, matching_cfg: dict = None, geometry_cfg: dict = None):
        self.camera = camera
        self.model = model

        # T_world_cam: camera frame → world/IMU frame (used to express solutions in world frame)
        self.T_world_cam: Transform = camera.T_world_cam

        # Tracking state — current frame
        self.prev_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.prev_prev_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None  # for velocity extrapolation
        self.prev_assignment = None

        # Last frame where tracking was confirmed good.
        # Retained across loss events so brute re-acquisition can be
        # validated for plausibility (pose-jump guard).
        self.last_good_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.last_good_assignment = None

        # Consecutive frames without a valid solution.
        self.consecutive_failures: int = 0

        self._matching_cfg = matching_cfg or {}

        _window = int(self._matching_cfg.get('pose_history_window', 5))
        self.pose_history: deque = deque(maxlen=_window)
        self.vel_ema: Optional[np.ndarray] = None  # shape (3,) float32, m/frame

        # Lazy cache (e.g. KD-tree later)
        self.kd_tree_cache = None

        from src.pose_search import PoseSearcher
        self._pose_searcher = PoseSearcher(camera, model, geometry_cfg, matching_cfg)
        self._geometry = self._pose_searcher._geometry

        self.proximity_match         = self._pose_searcher.proximity_search
        self.brute_match             = self._pose_searcher.brute_search
        self.prior_constrained_match = self._pose_searcher.constrained_search

    # # -----------------------------------------------------
    # # Custom projection
    # # -----------------------------------------------------
    # def project_leds_to_image(
    #         self,
    #         T_cam_ctrl: Transform  # controller → camera
    # ) -> List[Tuple[int, Optional[np.ndarray], bool]]:
    #
    #     projected = []
    #
    #     for idx, (pos, normal) in enumerate(zip(self.model.positions, self.model.normals)):
    #
    #         # --- transform to camera ---
    #         led_cam = T_cam_ctrl.apply(pos[None])[0]
    #         z = float(led_cam[2])
    #
    #         if z <= 1e-6:
    #             projected.append((idx, None, False))
    #             continue
    #
    #         # --- normalized coordinates ---
    #         x = led_cam[0] / z
    #         y = led_cam[1] / z
    #
    #         # --- distortion ---
    #         r2 = x * x + y * y
    #         r4 = r2 * r2
    #         r6 = r2 * r4
    #
    #         cam = self.camera
    #
    #         radial = (
    #                 (1 + cam.k1 * r2 + cam.k2 * r4 + cam.k3 * r6) /
    #                 (1 + cam.k4 * r2 + cam.k5 * r4 + cam.k6 * r6)
    #         )
    #
    #         dx = 2 * cam.p1 * x * y + cam.p2 * (r2 + 2 * x * x)
    #         dy = cam.p1 * (r2 + 2 * y * y) + 2 * cam.p2 * x * y
    #
    #         x_dist = x * radial + dx
    #         y_dist = y * radial + dy
    #
    #         # --- pixel ---
    #         u = cam.fx * x_dist + cam.cx
    #         v = cam.fy * y_dist + cam.cy
    #
    #         # --- visibility ---
    #         normal_cam = T_cam_ctrl.R @ normal
    #         view_dir = -led_cam / np.linalg.norm(led_cam)
    #
    #         is_visible = normal_cam @ view_dir > 0.2
    #
    #         projected.append((idx, np.array([u, v], dtype=np.float32), is_visible))
    #
    #     return projected

    # -----------------------------------------------------
    # Pose-jump guard
    # -----------------------------------------------------
    @staticmethod
    def _pose_jump_too_large(
        rvec_new, tvec_new,
        rvec_ref, tvec_ref,
        max_dist_m: float = 0.15,
        max_angle_deg: float = 25.0,
        pos_thresh_xyz_m: tuple = None,
        rot_thresh_xyz_deg: tuple = None,
    ) -> bool:
        """
        Return True if the new pose is implausibly far from the reference.

        Scalar mode (default): Euclidean translation distance and total rotation angle.
        Per-axis mode: if pos_thresh_xyz_m or rot_thresh_xyz_deg are given, each
          axis is checked independently (any axis over threshold → reject).
          For rotation the axis errors come from the Rodrigues log of the relative
          rotation, i.e. the rotation-vector components in radians.
          Per-axis mode overrides the corresponding scalar check when provided.
        """
        tvec_new = np.asarray(tvec_new, dtype=np.float64).reshape(3)
        tvec_ref = np.asarray(tvec_ref, dtype=np.float64).reshape(3)
        pos_diff = tvec_new - tvec_ref

        if pos_thresh_xyz_m is not None:
            tx, ty, tz = pos_thresh_xyz_m
            if abs(pos_diff[0]) > tx or abs(pos_diff[1]) > ty or abs(pos_diff[2]) > tz:
                return True
        elif np.linalg.norm(pos_diff) > max_dist_m:
            return True

        R_new, _ = cv2.Rodrigues(np.asarray(rvec_new, dtype=np.float32).reshape(3, 1))
        R_ref, _ = cv2.Rodrigues(np.asarray(rvec_ref, dtype=np.float32).reshape(3, 1))

        if rot_thresh_xyz_deg is not None:
            # Rodrigues log of relative rotation gives axis-angle as a 3-vector (radians).
            R_rel = R_new @ R_ref.T
            rvec_rel, _ = cv2.Rodrigues(R_rel.astype(np.float32))
            rot_deg = np.degrees(np.abs(rvec_rel.reshape(3)))
            rx, ry, rz = rot_thresh_xyz_deg
            if rot_deg[0] > rx or rot_deg[1] > ry or rot_deg[2] > rz:
                return True
        else:
            cos_a = np.clip((np.trace(R_new @ R_ref.T) - 1.0) / 2.0, -1.0, 1.0)
            if float(np.degrees(np.arccos(cos_a))) > max_angle_deg:
                return True

        return False

    @staticmethod
    def _predict_pose(
        pose_history,
        weight_decay: float = 0.7,
        vel_ema: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Predict pose at frame n+1 from pose history.

        pose_history[0] = most recent (rvec, tvec); index increases toward older frames.

        vel_ema: if provided, overrides translation prediction with pose_history[0].tvec + vel_ema.
                 Rotation prediction is always derived from pose history.

        n=0 → None (no information)
        n=1 → constant position (same pose)
        n=2 → constant-velocity extrapolation (matrix-based rotation)
        n>=3 → weighted degree-1 (linear) fit; exponential weights (weight_decay^i)
               Computes a weighted mean velocity across the window — averaging out
               the alternating big/small step oscillation from fast hand motion.
        """
        n = len(pose_history)

        if n == 0:
            return None

        if n == 1:
            return (
                np.asarray(pose_history[0][0], np.float32).reshape(3, 1),
                np.asarray(pose_history[0][1], np.float32).reshape(3),
            )

        if vel_ema is not None:
            tvec_pred = (np.asarray(pose_history[0][1], np.float32).reshape(3)
                         + vel_ema.reshape(3)).astype(np.float32)
        else:
            tvec_pred = None  # filled in by the branch below

        if n == 2:
            rvec_n   = np.asarray(pose_history[0][0], np.float32).reshape(3, 1)
            tvec_n   = np.asarray(pose_history[0][1], np.float64).reshape(3)
            rvec_nm1 = np.asarray(pose_history[1][0], np.float32).reshape(3, 1)
            tvec_nm1 = np.asarray(pose_history[1][1], np.float64).reshape(3)

            if tvec_pred is None:
                tvec_pred = (2.0 * tvec_n - tvec_nm1).astype(np.float32)

            R_n,   _ = cv2.Rodrigues(rvec_n)
            R_nm1, _ = cv2.Rodrigues(rvec_nm1)
            R_pred    = (R_n @ R_nm1.T) @ R_n
            rvec_pred, _ = cv2.Rodrigues(R_pred.astype(np.float32))

            return rvec_pred.reshape(3, 1).astype(np.float32), tvec_pred.reshape(3)

        # n >= 3: weighted degree-1 (linear) fit — estimates a single average velocity
        # across the window. Degree-2 would add an acceleration term that amplifies
        # the alternating big/small step oscillation typical of fast hand motion.
        # time axis: most recent pose = t=0, one frame older = t=-1, …; predict at t=+1
        t_pts   = -np.arange(n, dtype=np.float64)
        weights = weight_decay ** np.arange(n, dtype=np.float64)

        # Translation: linear fit per axis (skipped when vel_ema already set tvec_pred)
        if tvec_pred is None:
            tvecs = np.stack([np.asarray(p[1], np.float64).reshape(3) for p in pose_history])
            tvec_pred = np.empty(3, dtype=np.float32)
            for ax in range(3):
                tvec_pred[ax] = np.polyval(np.polyfit(t_pts, tvecs[:, ax], deg=1, w=weights), 1.0)

        # Rotation: linear fit in the tangent space of R_0 (most recent rotation).
        # rel_rvecs[i] = log(R_0^T @ R_i) — rotation from current pose back to the i-th
        # historical pose, expressed as a Rodrigues vector. These are always small-angle
        # deltas and avoid the ±π discontinuity of fitting absolute Rodrigues components.
        R_0, _ = cv2.Rodrigues(np.asarray(pose_history[0][0], np.float32).reshape(3, 1))
        rel_rvecs = np.zeros((n, 3), dtype=np.float64)  # rel_rvecs[0] = [0,0,0] by definition
        for i in range(1, n):
            R_i, _ = cv2.Rodrigues(np.asarray(pose_history[i][0], np.float32).reshape(3, 1))
            rv, _ = cv2.Rodrigues((R_0.T @ R_i).astype(np.float32))
            rel_rvecs[i] = rv.reshape(3)

        rvec_rel_pred = np.empty(3, dtype=np.float32)
        for ax in range(3):
            rvec_rel_pred[ax] = np.polyval(np.polyfit(t_pts, rel_rvecs[:, ax], deg=1, w=weights), 1.0)

        R_rel_pred, _ = cv2.Rodrigues(rvec_rel_pred.reshape(3, 1).astype(np.float32))
        rvec_pred, _ = cv2.Rodrigues((R_0 @ R_rel_pred).astype(np.float32))

        return rvec_pred.reshape(3, 1).astype(np.float32), tvec_pred.reshape(3)

    # -----------------------------------------------------
    # Tracking
    # -----------------------------------------------------
    def search(self, blobs: np.ndarray, blob_radii: Optional[np.ndarray] = None,
               blob_brightnesses: Optional[np.ndarray] = None,
               other_cameras_blobs: Optional[List] = None,
               blob_mask: Optional[np.ndarray] = None,
               occluders_per_cam: Optional[Dict] = None) -> Optional[Dict]:
        """Pure pose solve — reads self state, does not commit results.

        Returns a validated solution dict (error below threshold, T_world_ctrl populated)
        or None.  Call apply() with the returned value to commit state changes.
        """
        blobs   = np.asarray(blobs, dtype=np.float32).reshape(-1, 2)
        n_blobs = len(blobs)

        # When blob_mask provided, blobs is the full camera observation array.
        # Proximity and prior_constrained use the filtered subset; brute uses the full array.
        if blob_mask is not None:
            _avail_idx  = np.where(blob_mask)[0].astype(np.int32)
            blobs_prox  = blobs[_avail_idx]
            radii_prox  = blob_radii[_avail_idx]  if blob_radii        is not None else None
            brts_prox   = blob_brightnesses[_avail_idx] if blob_brightnesses is not None else None
            n_available = len(blobs_prox)
        else:
            blobs_prox  = blobs
            radii_prox  = blob_radii
            brts_prox   = blob_brightnesses
            n_available = n_blobs

        cam_idx = self.camera.camera_idx
        ctrl_name = self.model.name.replace("_controller", "")

        # Per-axis jump thresholds from config (fall back to scalar defaults if absent).
        _cfg = self._matching_cfg
        _use_proximity = bool(_cfg.get('use_proximity_match', True))
        _accept_err_px = float(_cfg.get('accept_error_px', 3.0))
        _pos_ax = _cfg.get('pose_jump_pos_thresh_m')
        _rot_ax = _cfg.get('pose_jump_rot_thresh_deg')
        _jump_kw = {}
        if _pos_ax is not None:
            _jump_kw['pos_thresh_xyz_m']   = tuple(_pos_ax)
        if _rot_ax is not None:
            _jump_kw['rot_thresh_xyz_deg'] = tuple(_rot_ax)

        # Normalise prev_pose shapes (idempotent — canonicalises (3,1) rvec and (3,) tvec)
        if self.prev_pose is not None:
            rvec, tvec = self.prev_pose
            self.prev_pose = (
                np.asarray(rvec, dtype=np.float32).reshape(3, 1),
                np.asarray(tvec, dtype=np.float32).reshape(3),
            )
        if self.prev_prev_pose is not None:
            rvec, tvec = self.prev_prev_pose
            self.prev_prev_pose = (
                np.asarray(rvec, dtype=np.float32).reshape(3, 1),
                np.asarray(tvec, dtype=np.float32).reshape(3),
            )

        predicted_pose = self._predict_pose(
            self.pose_history,
            weight_decay=float(self._matching_cfg.get("pose_prediction_weight_decay", 0.7)),
            vel_ema=self.vel_ema,
        )

        # Velocity-scaled search gates: expand proximity radius proportionally to speed.
        # v_px = ‖predicted.t − prev.t‖ × fx / depth  (pixels/frame, camera-agnostic)
        _v_px = 0.0
        if predicted_pose is not None and self.prev_pose is not None:
            _v_vec = predicted_pose[1].reshape(3) - np.asarray(self.prev_pose[1], np.float64).reshape(3)
            _depth = max(float(np.asarray(self.prev_pose[1]).reshape(3)[2]), 0.1)
            _v_px  = float(np.linalg.norm(_v_vec)) * self.camera.fx / _depth
        _base_expansion = float(_cfg.get('proximity_expansion_px', 8.0))
        _prox_vel_k     = float(_cfg.get('proximity_expansion_velocity_k', 0.0))
        _eff_expansion  = _base_expansion + _prox_vel_k * _v_px
        # Uncertainty term: larger neighbourhood when prediction history is short.
        # Decays as 1/n — full boost at n=1 (constant-position), half at n=2, etc.
        _uncertainty_k = float(_cfg.get('proximity_expansion_uncertainty_k', 0.0))
        if _uncertainty_k > 0.0:
            _eff_expansion += _uncertainty_k / max(len(self.pose_history), 1)
        # Depth term: closer controller → larger pixel-space uncertainty → bigger neighbourhood.
        _depth_k = float(_cfg.get('proximity_expansion_depth_k', 0.0))
        if _depth_k > 0.0 and predicted_pose is not None:
            _ctrl_depth = max(float(predicted_pose[1].reshape(3)[2]), 0.01)
            _eff_expansion += _depth_k / _ctrl_depth

        # ------------------------------------------------------------------
        # Candidate search
        # ------------------------------------------------------------------
        solution = None

        if self.prev_pose is not None:
            # --- Primary: proximity (fast, assignment-locked) ---
            if _use_proximity and n_available >= 3:
                solution = self.proximity_match(
                    blobs_prox, predicted_pose,
                    blob_brightnesses=brts_prox,
                    other_cameras_blobs=other_cameras_blobs,
                    occluders_per_cam=occluders_per_cam,
                    expansion_px=_eff_expansion,
                )
                # Remap proximity result indices from filtered → full array space.
                if solution is not None and blob_mask is not None:
                    solution['assignment'] = [(_avail_idx[b], lid) for b, lid in solution['assignment']]
                    solution['_orig_idx']  = True

            # --- Fallback: brute only when proximity found no solution ---
            if solution is None and n_available >= 4:
                logger.debug(f"[{ctrl_name} | cam {cam_idx} | track] proximity → None, running brute fallback")
                brute = self.brute_match(blobs, pose_prior=predicted_pose,
                                         other_cameras_blobs=other_cameras_blobs,
                                         blob_radii=blob_radii,
                                         blob_mask=blob_mask,
                                         occluders_per_cam=occluders_per_cam)
                if brute is not None:
                    solution = brute

        else:
            # --- No prior pose: brute-force re-acquisition ---
            if n_available >= 4:
                logger.debug(f"[{ctrl_name} | cam {cam_idx} | track] no prev_pose → cold-start brute")
                solution = self.brute_match(blobs, other_cameras_blobs=other_cameras_blobs,
                                            blob_radii=blob_radii,
                                            blob_mask=blob_mask,
                                            occluders_per_cam=occluders_per_cam)

                # In deep-debug mode frames are non-consecutive, so the last_good_pose
                # plausibility check is skipped — the controller can be anywhere.
                if solution is not None and self.last_good_pose is not None and not is_deep():
                    rvec_lg, tvec_lg = self.last_good_pose
                    if self._pose_jump_too_large(
                        solution["rvec"], solution["tvec"],
                        rvec_lg, tvec_lg,
                        max_dist_m=0.5,
                        max_angle_deg=60.0,
                        **_jump_kw,
                    ):
                        logger.debug(f"[{ctrl_name} | cam {cam_idx} | track] brute re-acquisition rejected: too far from last known good pose")
                        solution = None

        # ------------------------------------------------------------------
        # Low-blob-count fallback: prior-constrained translation solve
        # P2P (3 blobs): fix R, solve t from 2 pairs, validate with 3rd.
        # P1P (2 blobs): fix R + depth, solve (tx,ty) from 1 pair, validate with 2nd.
        # Only reachable when prev_pose exists (prior quality requirement).
        # ------------------------------------------------------------------
        if (solution is None
                and self.prev_pose is not None
                and self.prev_assignment is not None
                and 2 <= n_available <= 3):
            logger.debug(f"[{ctrl_name} | cam {cam_idx} | track] n_blobs={n_available} + prior → prior_constrained_match")
            solution = self.prior_constrained_match(
                blobs_prox, predicted_pose,
                prior_assignment=self.prev_assignment,
                blob_radii=radii_prox,
                other_cameras_blobs=other_cameras_blobs,
            )
            if solution is not None and blob_mask is not None:
                solution['assignment'] = [(_avail_idx[b], lid) for b, lid in solution['assignment']]
                solution['_orig_idx']  = True

        # ------------------------------------------------------------------
        # Pose-jump guard against prev_pose (tight, per-frame)
        # ------------------------------------------------------------------
        if solution is not None and self.prev_pose is not None:
            rvec_p, tvec_p = self.prev_pose
            if self._pose_jump_too_large(
                solution["rvec"], solution["tvec"],
                rvec_p, tvec_p,
                **_jump_kw,
            ):
                print(f"[{ctrl_name} | cam {cam_idx} | tracking] Pose jump detected "
                      f"(method={solution.get('method','?')}, "
                      f"err={solution['error']:.2f} px) — "
                      f"attempting brute recovery.")
                solution = None
                if n_available >= 4:
                    _jump_prior = predicted_pose if predicted_pose is not None else self.prev_pose
                    brute = self.brute_match(blobs, pose_prior=_jump_prior,
                                             other_cameras_blobs=other_cameras_blobs,
                                             blob_radii=blob_radii,
                                             blob_mask=blob_mask)
                    if brute is not None:
                        _near_prev = not self._pose_jump_too_large(
                            brute["rvec"], brute["tvec"], rvec_p, tvec_p, **_jump_kw
                        )
                        _near_pred = (predicted_pose is not None and not self._pose_jump_too_large(
                            brute["rvec"], brute["tvec"],
                            predicted_pose[0], predicted_pose[1], **_jump_kw
                        ))
                        if _near_prev or _near_pred:
                            solution = brute

        # ------------------------------------------------------------------
        # Accept / reject
        # ------------------------------------------------------------------

        # Proximity found a solution but error is too high — try brute before giving up
        if solution is not None and solution["error"] >= _accept_err_px and n_available >= 4:
            logger.debug(f"[{ctrl_name} | cam {cam_idx} | track] proximity err={solution['error']:.2f}px exceeds threshold, attempting brute recovery")
            brute = self.brute_match(blobs, pose_prior=self.prev_pose,
                                     other_cameras_blobs=other_cameras_blobs,
                                     blob_radii=blob_radii,
                                     blob_mask=blob_mask)
            if brute is not None:
                logger.debug(f"[{ctrl_name} | cam {cam_idx} | track] brute recovery err={brute['error']:.2f}px")
                solution = brute

        if solution is not None and solution["error"] < _accept_err_px:
            logger.debug(
                f"[{ctrl_name} | cam {cam_idx} | track] accepted — method={solution.get('method','?')}  "
                f"inliers={len(solution['assignment'])}  err={solution['error']:.2f}px"
            )
            # World-frame controller pose: T_world_ctrl = T_world_cam ∘ T_cam_ctrl
            R_ctrl, _ = cv2.Rodrigues(solution["rvec"])
            T_cam_ctrl = Transform(R_ctrl, solution["tvec"].reshape(3))
            solution["T_world_ctrl"] = self.T_world_cam.compose(T_cam_ctrl)
            return solution

        if solution is not None:
            logger.debug(f"[{ctrl_name} | cam {cam_idx} | track] rejected — err={solution['error']:.2f}px exceeds {_accept_err_px:.1f}px threshold")
        else:
            logger.debug(f"[{ctrl_name} | cam {cam_idx} | track] rejected — no solution found")
        return None

    def apply(self, result: Optional[Dict]) -> None:
        """Commit a search() result to tracker state.

        On success: update prev_pose, pose_history, vel_ema, assignment caches.
        On failure: clear transient state so the next frame starts a cold brute search.
        last_good_pose is preserved across failures for re-acquisition plausibility checks.
        """
        if result is not None:
            if self.prev_pose is not None:
                _step = (np.asarray(result["tvec"], np.float64).reshape(3)
                         - np.asarray(self.prev_pose[1], np.float64).reshape(3)).astype(np.float32)
                _beta = float(self._matching_cfg.get("pose_prediction_vel_ema_beta", 0.3))
                self.vel_ema = (_beta * _step + (1.0 - _beta) * self.vel_ema
                               if self.vel_ema is not None else _step)
            self.prev_prev_pose  = self.prev_pose
            self.prev_pose       = (result["rvec"], result["tvec"])
            self.pose_history.appendleft((
                np.asarray(result["rvec"], np.float32).reshape(3, 1),
                np.asarray(result["tvec"], np.float32).reshape(3),
            ))
            self.prev_assignment      = result["assignment"]
            self.last_good_pose       = self.prev_pose
            self.last_good_assignment = self.prev_assignment
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.prev_pose       = None
            self.prev_prev_pose  = None
            self.prev_assignment = None
            self.vel_ema         = None
            self.pose_history.clear()

    def track(self, blobs: np.ndarray, blob_radii: Optional[np.ndarray] = None,
              blob_brightnesses: Optional[np.ndarray] = None,
              other_cameras_blobs: Optional[List] = None,
              blob_mask: Optional[np.ndarray] = None,
              occluders_per_cam: Optional[Dict] = None) -> Optional[Dict]:
        """Thin wrapper: search() then apply(). Preserves backward compatibility."""
        result = self.search(blobs, blob_radii, blob_brightnesses,
                             other_cameras_blobs, blob_mask, occluders_per_cam)
        self.apply(result)
        return result


# =========================================================
# 2.5 PER-CONTROLLER TRACKER
# =========================================================

class ControllerTracker:
    """Manages all camera-trackers for a single controller.

    Owns primary-camera selection, camera fallback, handoff heuristics, state
    propagation to non-primary CameraTrackers, and claimed-blob registration.
    Cross-controller concerns (Voronoi reservation, ordering) remain in TrackingSystem.
    """

    def __init__(self, ctrl_name: str, cameras: Dict[int, "Camera"],
                 trackers: Dict[int, CameraTracker],
                 matching_cfg: Optional[dict] = None):
        self.ctrl_name        = ctrl_name
        self.cameras          = cameras
        self.trackers         = trackers           # {cam_id: CameraTracker}
        self._matching_cfg    = matching_cfg or {}
        self._designated_primary: Optional[int]  = None
        self._prev_primary:       Optional[int]  = None
        self._handoff_counter:    Dict[int, int] = {}

    def update(
        self,
        avail:         Dict[int, tuple],        # cam_id → (blobs, radii, brts, orig_idx)
        obs_src:       Dict[int, np.ndarray],   # full observations per camera
        rad_src:       Dict,
        brt_src:       Dict,
        claimed_blobs: Dict[int, Set[int]],     # mutated in place
        fixed_primary_cam: Optional[int] = None,
        occluders_per_cam: Optional[Dict] = None,
        self_cal=None,
    ) -> Optional[Dict]:
        if not avail:
            return None

        _cfg = self._matching_cfg

        # ── Primary camera selection ───────────────────────────────────────────
        if fixed_primary_cam is not None and fixed_primary_cam in avail:
            primary_cam_id = fixed_primary_cam
        else:
            _designated  = self._designated_primary
            _des_tracker = self.trackers.get(_designated)
            _min_inliers = int(_cfg.get('min_inliers', 4))
            _des_ok = (
                _des_tracker is not None
                and _designated in avail
                and len(avail[_designated][0]) >= _min_inliers
            )
            if _des_ok:
                primary_cam_id = _designated
            else:
                _with_prior = [
                    cid for cid in avail
                    if self.trackers[cid].prev_pose is not None
                ]
                if _with_prior:
                    primary_cam_id = max(_with_prior, key=lambda cid: len(avail[cid][0]))
                else:
                    _cold_eligible = [
                        cid for cid in avail if len(avail[cid][0]) >= _min_inliers
                    ]
                    if _cold_eligible:
                        primary_cam_id = min(_cold_eligible, key=lambda cid: len(avail[cid][0]))
                    else:
                        primary_cam_id = max(avail, key=lambda cid: len(avail[cid][0]))

        primary_tracker = self.trackers[primary_cam_id]
        _, _, _, primary_orig = avail[primary_cam_id]

        other_cameras_blobs = [
            (self.cameras[cid], av[0], av[1])
            for cid, av in avail.items()
            if cid != primary_cam_id
        ]
        aux_orig: Dict[int, List[int]] = {
            cid: av[3] for cid, av in avail.items() if cid != primary_cam_id
        }

        _prim_obs_full = obs_src[primary_cam_id]
        _prim_rad_full = rad_src.get(primary_cam_id) if rad_src else None
        _prim_brt_full = brt_src.get(primary_cam_id) if brt_src else None
        _prim_mask = np.zeros(len(_prim_obs_full), dtype=bool)
        _prim_mask[primary_orig] = True

        # ── Primary camera track ───────────────────────────────────────────────
        solution = primary_tracker.track(
            _prim_obs_full,
            blob_radii=_prim_rad_full,
            blob_brightnesses=_prim_brt_full,
            other_cameras_blobs=other_cameras_blobs,
            blob_mask=_prim_mask,
            occluders_per_cam=occluders_per_cam,
        )

        # ── Fallback cameras ───────────────────────────────────────────────────
        _cold_start = primary_tracker.prev_pose is None
        if solution is None and len(avail) > 1:
            _fallback_order = sorted(
                [cid for cid in avail if cid != primary_cam_id],
                key=lambda cid: (
                    0 if self.trackers[cid].prev_pose is not None else 1,
                    len(avail[cid][0]) if _cold_start else -len(avail[cid][0]),
                ),
            )
            for _fb_cid in _fallback_order:
                _fb_blobs, _fb_radii, _fb_brts, _fb_orig = avail[_fb_cid]
                if len(_fb_blobs) < 3:
                    continue
                _fb_tracker = self.trackers[_fb_cid]
                _fb_other = [
                    (self.cameras[cid], av[0], av[1])
                    for cid, av in avail.items() if cid != _fb_cid
                ]
                _fb_obs_full = obs_src[_fb_cid]
                _fb_rad_full = rad_src.get(_fb_cid) if rad_src else None
                _fb_brt_full = brt_src.get(_fb_cid) if brt_src else None
                _fb_mask = np.zeros(len(_fb_obs_full), dtype=bool)
                _fb_mask[_fb_orig] = True
                solution = _fb_tracker.track(
                    _fb_obs_full,
                    blob_radii=_fb_rad_full,
                    blob_brightnesses=_fb_brt_full,
                    other_cameras_blobs=_fb_other,
                    blob_mask=_fb_mask,
                    occluders_per_cam=occluders_per_cam,
                )
                if solution is not None:
                    primary_cam_id  = _fb_cid
                    primary_tracker = _fb_tracker
                    primary_orig    = _fb_orig
                    aux_orig = {
                        cid: av[3] for cid, av in avail.items() if cid != _fb_cid
                    }
                    break

        if solution is None:
            return None

        solution["primary_cam"] = primary_cam_id

        # ── Primary switch logging ─────────────────────────────────────────────
        if self._prev_primary is not None and primary_cam_id != self._prev_primary:
            logger.info(
                f"[{self.ctrl_name}] Primary camera switched "
                f"cam{self._prev_primary} → cam{primary_cam_id}"
            )
        self._prev_primary = primary_cam_id
        if fixed_primary_cam is None:
            self._designated_primary = primary_cam_id

        # ── Index remapping ────────────────────────────────────────────────────
        if not solution.get('_orig_idx', False):
            solution["assignment"] = [
                (primary_orig[b], lid) for b, lid in solution["assignment"]
            ]
        _raw_aux = solution.get("aux_assignments") or {}
        if _raw_aux:
            solution["aux_assignments"] = {
                cid: [(aux_orig[cid][b], lid) for b, lid in pairs]
                for cid, pairs in _raw_aux.items()
                if cid in aux_orig
            }

        # ── Register claimed blobs ─────────────────────────────────────────────
        for b, _ in solution["assignment"]:
            claimed_blobs.setdefault(primary_cam_id, set()).add(b)
        for cid, pairs in (solution.get("aux_assignments") or {}).items():
            for b, _ in pairs:
                claimed_blobs.setdefault(cid, set()).add(b)

        # ── Self-calibration feed ──────────────────────────────────────────────
        if (self_cal is not None
                and primary_cam_id == self_cal.primary_camera.camera_idx):
            _R_prim, _ = cv2.Rodrigues(solution["rvec"].reshape(3, 1).astype(np.float32))
            _t_prim = solution["tvec"].reshape(3).astype(np.float32)
            _sc_aux_obs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            for _aux_cid, _aux_pairs in (solution.get("aux_assignments") or {}).items():
                if len(_aux_pairs) < 3:
                    continue
                _led_ids   = np.array([lid for _, lid in _aux_pairs], dtype=np.int32)
                _blob_idxs = np.array([b   for b, _  in _aux_pairs], dtype=np.int32)
                _led_pos   = primary_tracker.model.positions[_led_ids]
                _pts_prim  = (_R_prim @ _led_pos.T).T + _t_prim
                _blobs_aux = obs_src[_aux_cid][_blob_idxs]
                _sc_aux_obs[_aux_cid] = (_pts_prim, _blobs_aux)
            if _sc_aux_obs:
                self_cal.add_frame(
                    solution["rvec"], solution["tvec"],
                    primary_error=solution["error"],
                    primary_inliers=len(solution["assignment"]),
                    aux_observations=_sc_aux_obs,
                )

        # ── State propagation to non-primary cameras ───────────────────────────
        T_world_ctrl = solution["T_world_ctrl"]
        aux_asgns    = solution.get("aux_assignments") or {}
        for _cid, _tracker in self.trackers.items():
            if _cid == primary_cam_id:
                continue
            _T_cam_ctrl = self.cameras[_cid].T_world_cam.inverse().compose(T_world_ctrl)
            _rv_np, _ = cv2.Rodrigues(_T_cam_ctrl.R.astype(np.float32))
            _tv_np = _T_cam_ctrl.t.astype(np.float32)
            if _tracker.prev_pose is not None:
                _step = (_tv_np.reshape(3).astype(np.float64)
                         - np.asarray(_tracker.prev_pose[1], np.float64).reshape(3)).astype(np.float32)
                _beta = float(self._matching_cfg.get("pose_prediction_vel_ema_beta", 0.3))
                _tracker.vel_ema = (_beta * _step + (1.0 - _beta) * _tracker.vel_ema
                                   if _tracker.vel_ema is not None else _step)
            _tracker.prev_prev_pose = _tracker.prev_pose
            _tracker.prev_pose      = (_rv_np.reshape(3, 1), _tv_np)
            _tracker.last_good_pose = _tracker.prev_pose
            _tracker.pose_history.appendleft((_rv_np.reshape(3, 1), _tv_np))
            _aux_asgn = aux_asgns.get(_cid)
            if _aux_asgn is not None:
                _tracker.prev_assignment      = _aux_asgn
                _tracker.last_good_assignment = _aux_asgn
            elif _tracker.prev_assignment is None:
                _tracker.last_good_assignment = None
            _tracker.consecutive_failures = 0

        # ── Camera handoff check ───────────────────────────────────────────────
        if bool(_cfg.get('camera_handoff', True)) and fixed_primary_cam is None:
            _ratio_thr  = float(_cfg.get('handoff_coverage_ratio',   1.5))
            _min_adv    = int(  _cfg.get('handoff_min_advantage',     3))
            _hysteresis = int(  _cfg.get('handoff_hysteresis_frames', 3))
            _n_primary  = len(solution["assignment"])
            for _aux_cid, _aux_pairs in aux_asgns.items():
                _aux_n = len(_aux_pairs)
                if (_aux_n >= _ratio_thr * _n_primary
                        and _aux_n - _n_primary >= _min_adv):
                    self._handoff_counter[_aux_cid] = self._handoff_counter.get(_aux_cid, 0) + 1
                    if self._handoff_counter[_aux_cid] >= _hysteresis:
                        logger.info(
                            f"[{self.ctrl_name}] Camera handoff: "
                            f"cam{primary_cam_id}({_n_primary} LEDs)"
                            f" → cam{_aux_cid}({_aux_n} LEDs)"
                            f" — warm start from propagated pose"
                        )
                        self._designated_primary = _aux_cid
                        self._handoff_counter.clear()
                else:
                    self._handoff_counter[_aux_cid] = 0

        return solution


# =========================================================
# 3. SYSTEM (multi-controller, multi-camera)
# =========================================================

class TrackingSystem:
    def __init__(self, controllers: List[ControllerModel], cameras: List[Camera],
                 matching_cfg: dict = None, geometry_cfg: dict = None,
                 geometry_cfg_per_ctrl: dict = None,
                 self_calibration_cfg: dict = None):

        self.cameras: Dict[int, Camera] = {cam.camera_idx: cam for cam in cameras}

        # Self-calibration: optionally apply saved extrinsics before tracker creation
        # so every tracker's T_world_cam starts with the correct (calibrated) value.
        self._self_cal: Optional[SelfCalibrator] = None
        self._self_cal = None
        sc_cfg = self_calibration_cfg or {}
        _sc_primary_idx: Optional[int] = None

        if sc_cfg.get("enabled", False):
            _sc_primary_idx = int(sc_cfg.get("primary_camera", 0))
            _aux_cam_idxs = sc_cfg.get("aux_cameras") or [
                cid for cid in self.cameras if cid != _sc_primary_idx
            ]
            _primary_cam = self.cameras.get(_sc_primary_idx)
            _aux_cams = [self.cameras[cid] for cid in _aux_cam_idxs if cid in self.cameras]

            if _primary_cam and _aux_cams:
                self._self_cal = SelfCalibrator(_primary_cam, _aux_cams, sc_cfg)

        self._matching_cfg: dict = matching_cfg or {}

        # Resolve fixed primary camera: matching cfg wins, then self-cal lock_primary.
        _fpc = self._matching_cfg.get("fixed_primary_camera")
        if _fpc is None and _sc_primary_idx is not None and sc_cfg.get("lock_primary", True):
            _fpc = _sc_primary_idx
        self._fixed_primary_cam: Optional[int] = int(_fpc) if _fpc is not None else None

        self.trackers: Dict[Tuple[str, int], CameraTracker] = {}
        self.ctrl_trackers: Dict[str, ControllerTracker]    = {}

        for ctrl in controllers:
            ctrl_cam_trackers: Dict[int, CameraTracker] = {}
            for cam in cameras:
                key = (ctrl.name, cam.camera_idx)
                geo = (geometry_cfg_per_ctrl or {}).get(ctrl.name) or geometry_cfg
                cam_tracker = CameraTracker(cam, ctrl, matching_cfg=matching_cfg, geometry_cfg=geo)
                self.trackers[key]           = cam_tracker
                ctrl_cam_trackers[cam.camera_idx] = cam_tracker
            self.ctrl_trackers[ctrl.name] = ControllerTracker(
                ctrl.name, self.cameras, ctrl_cam_trackers, matching_cfg=matching_cfg,
            )

        if self._fixed_primary_cam is not None:
            for ctrl in controllers:
                self.ctrl_trackers[ctrl.name]._designated_primary = self._fixed_primary_cam



    def get_designated_primary_cameras(self) -> Dict[str, Optional[int]]:
        """Return {ctrl_name: primary_cam_id} as of the last successful tracking frame.

        Used at detection time to choose per-camera search radii before tracking runs.
        None means no designation exists yet (cold start); callers should treat all
        cameras as equally primary in that case.
        """
        return {name: ct._designated_primary for name, ct in self.ctrl_trackers.items()}

    def get_predicted_led_projections_per_camera(
        self,
    ) -> Tuple[Dict[int, Dict[str, Optional[np.ndarray]]], Dict[int, Dict[str, float]]]:
        """Return (proj_hints, vel_hints).

        proj_hints: {cam_id: {ctrl_name: Nx5 array or None}}
          Each row: [proj_x, proj_y, depth_m, facing_cos, led_id]
          for each LED visible from the predicted pose.
          None when no prior pose exists for this (ctrl, cam) pair.

        vel_hints: {cam_id: {ctrl_name: v_px}}
          Estimated controller speed in pixels/frame for this camera.
          0.0 when no prediction is available.
        """
        ctrl_names   = sorted({ctrl for ctrl, _ in self.trackers})
        _facing_deg  = float(self._matching_cfg.get('led_facing_angle_deg', 86.0))
        result:       Dict[int, Dict[str, Optional[np.ndarray]]] = {}
        vel_hints:    Dict[int, Dict[str, float]]                = {}
        radius_hints: Dict[int, Dict[str, float]]                = {}

        # Pre-read all four expansion terms once — same values used in track()
        _base_r = float(self._matching_cfg.get('proximity_expansion_px', 8.0))
        _vel_k  = float(self._matching_cfg.get('proximity_expansion_velocity_k', 0.0))
        _unc_k  = float(self._matching_cfg.get('proximity_expansion_uncertainty_k', 0.0))
        _dpt_k  = float(self._matching_cfg.get('proximity_expansion_depth_k', 0.0))

        for cam_id, camera in self.cameras.items():
            proj_per_ctrl:   Dict[str, Optional[np.ndarray]] = {}
            vel_per_ctrl:    Dict[str, float]                = {}
            radius_per_ctrl: Dict[str, float]               = {}
            for ctrl_name in ctrl_names:
                tracker = self.trackers.get((ctrl_name, cam_id))
                pred = CameraTracker._predict_pose(tracker.pose_history) if tracker else None
                if pred is None:
                    proj_per_ctrl[ctrl_name]   = None
                    vel_per_ctrl[ctrl_name]    = 0.0
                    radius_per_ctrl[ctrl_name] = _base_r
                    continue
                rvec_pred, tvec_pred = pred
                _ph = tracker.pose_history
                if tracker.vel_ema is not None:
                    vel_3d = tracker.vel_ema.astype(np.float64)
                elif len(_ph) >= 2:
                    vel_3d = (np.asarray(_ph[0][1], np.float64).reshape(3)
                              - np.asarray(_ph[1][1], np.float64).reshape(3))
                else:
                    vel_3d = np.zeros(3, np.float64)

                R_pred, _ = cv2.Rodrigues(rvec_pred.reshape(3, 1))
                R_pred    = R_pred.astype(np.float32)
                tvec_pred = tvec_pred.reshape(3).astype(np.float32)

                positions = tracker.model.positions
                normals   = tracker.model.normals

                vis_mask = _visible_mask(
                    R_pred, tvec_pred, positions, normals, tracker._geometry,
                    cam_K=camera.camera_matrix, cam_dc=camera.dist_coeffs,
                    cam_w=camera.width, cam_h=camera.height, cam_rpmax=camera.rpmax,
                    cam_is_fisheye=camera.is_fisheye,
                    facing_threshold_deg=_facing_deg,
                )
                vis_ids = np.where(vis_mask)[0]
                if len(vis_ids) == 0:
                    proj_per_ctrl[ctrl_name]   = None
                    vel_per_ctrl[ctrl_name]    = 0.0
                    radius_per_ctrl[ctrl_name] = _base_r
                    continue

                proj_pts = _project_points(
                    rvec_pred, tvec_pred, positions[vis_ids],
                    camera.camera_matrix, camera.dist_coeffs,
                    is_fisheye=camera.is_fisheye,
                )  # (M, 2)

                led_cam     = (R_pred @ positions[vis_ids].T).T + tvec_pred  # (M, 3)
                depths      = led_cam[:, 2]                                   # (M,)
                view_dir    = led_cam / (np.linalg.norm(led_cam, axis=1, keepdims=True) + 1e-8)
                normals_cam = (R_pred @ normals[vis_ids].T).T                 # (M, 3)
                facing_cos  = -(normals_cam * view_dir).sum(axis=1)           # positive = faces cam

                proj_per_ctrl[ctrl_name] = np.column_stack([
                    proj_pts.astype(np.float32),
                    depths.astype(np.float32).reshape(-1, 1),
                    facing_cos.astype(np.float32).reshape(-1, 1),
                    vis_ids.astype(np.float32).reshape(-1, 1),
                ])  # (M, 5): proj_x, proj_y, depth_m, facing_cos, led_id

                _depth_pred = max(float(tvec_pred[2]), 0.1)
                _v_px       = float(np.linalg.norm(vel_3d)) * camera.fx / _depth_pred
                vel_per_ctrl[ctrl_name] = _v_px

                # Full effective blob-detection search radius — mirrors track() expansion logic
                _r = _base_r + _vel_k * _v_px
                if _unc_k > 0.0:
                    _r += _unc_k / max(len(tracker.pose_history), 1)
                if _dpt_k > 0.0:
                    _r += _dpt_k / _depth_pred
                radius_per_ctrl[ctrl_name] = _r

            result[cam_id]        = proj_per_ctrl
            vel_hints[cam_id]     = vel_per_ctrl
            radius_hints[cam_id]  = radius_per_ctrl
        return result, vel_hints, radius_hints

    def get_ctrl_processing_order(self) -> List[str]:
        """Return controller names in the same priority order used by update()."""
        ctrl_names = sorted({ctrl for ctrl, _ in self.trackers})
        _first = self._matching_cfg.get('first_controller', None)

        def _has_prior(name: str) -> bool:
            return any(
                t.prev_pose is not None
                for (cname, _), t in self.trackers.items()
                if cname == name
            )

        def _order_key(name: str):
            has_prior_rank = 0 if _has_prior(name) else 1
            preferred_rank = (0 if name == f"{_first}_controller" else 1) if _first else 0
            return (has_prior_rank, preferred_rank, name)

        return sorted(ctrl_names, key=_order_key)

    def update(
        self,
        observations_per_camera: Dict[int, np.ndarray],
        radii_per_camera: Optional[Dict[int, np.ndarray]] = None,
        brightnesses_per_camera: Optional[Dict[int, np.ndarray]] = None,
        per_ctrl_observations: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
        per_ctrl_radii: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
        per_ctrl_brightnesses: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
        ctrl_name_filter: Optional[str] = None,
    ) -> Dict[str, Optional[Dict]]:
        """
        Run tracking for every controller using a single primary camera.

        Blob ownership model: observations_per_camera is never mutated. Instead,
        claimed_blobs tracks which original blob indices have been consumed by
        earlier controllers. Each controller receives a pre-filtered view of
        unclaimed blobs; assignments are remapped to original indices by ControllerTracker.

        Controllers with a prior pose run first so their high-confidence claims
        are registered before cold-starting controllers search.

        Returns {ctrl_name: solution_or_None}. The solution dict gains a
        "primary_cam" key with the index of the camera that produced the pose.
        """
        results: Dict[str, Optional[Dict]] = {}

        ordered = self.get_ctrl_processing_order()
        if ctrl_name_filter is not None:
            ordered = [n for n in ordered if n == ctrl_name_filter]

        # Blob ownership: original observation indices claimed by earlier controllers.
        claimed_blobs: Dict[int, Set[int]] = {}

        def _filter_cam(cam_id: int, obs_map: dict, rad_map: dict, brt_map: dict,
                        extra_excluded: Optional[Set[int]] = None):
            """Return (filtered_blobs, filtered_radii, filtered_brts, orig_indices)
            excluding claimed and reserved blobs. Returns (None,None,None,[]) if empty."""
            obs = obs_map.get(cam_id)
            if obs is None or len(obs) == 0:
                return None, None, None, []
            claimed  = claimed_blobs.get(cam_id, set())
            excluded = claimed if not extra_excluded else claimed | extra_excluded
            keep = [i for i in range(len(obs)) if i not in excluded]
            if not keep:
                return None, None, None, []
            k = np.array(keep, dtype=np.int32)
            f_blobs = obs[k]
            _r = rad_map.get(cam_id) if rad_map else None
            f_radii = _r[k] if _r is not None else None
            _b = brt_map.get(cam_id) if brt_map else None
            f_brts  = _b[k] if _b is not None else None
            return f_blobs, f_radii, f_brts, keep

        _cfg = self._matching_cfg

        # Reservation sets: blobs that later controllers must not use (cross-controller Voronoi).
        _reservations: Dict[str, Dict[int, Set[int]]] = {_cn: {} for _cn in ordered}

        # ── Phase 3 / 4: Proximity mutual exclusion ───────────────────────────
        if per_ctrl_observations is None and len(ordered) >= 2:
            _reserve_px = float(_cfg.get(
                'proximity_mutual_exclusion_px', _cfg.get('aux_snap_px', 15.0),
            ))
            _voronoi_px = float(_cfg.get('voronoi_max_dist_px', 100.0))

            _ctrl_proj: Dict[str, Dict[int, Optional[np.ndarray]]] = {}
            for _cn in ordered:
                _ctrl_proj[_cn] = {}
                for _cid, _cam in self.cameras.items():
                    _trk = self.trackers.get((_cn, _cid))
                    if _trk is None or _trk.prev_pose is None:
                        _ctrl_proj[_cn][_cid] = None
                        continue
                    _rv, _tv = _trk.prev_pose
                    _R, _ = cv2.Rodrigues(np.asarray(_rv, dtype=np.float32).reshape(3))
                    _tv_np = np.asarray(_tv, dtype=np.float32).reshape(3)
                    _pts_cam = (_R @ _trk.model.positions.T).T + _tv_np
                    _vis = _pts_cam[:, 2] > 0.01
                    if not _vis.any():
                        _ctrl_proj[_cn][_cid] = None
                        continue
                    _ctrl_proj[_cn][_cid] = _project_points(
                        _rv, _tv, _trk.model.positions[_vis],
                        _cam.camera_matrix, _cam.dist_coeffs,
                        is_fisheye=_cam.is_fisheye,
                    )

            for _i, _cn_i in enumerate(ordered[:-1]):
                _proj_i = _ctrl_proj.get(_cn_i, {})
                for _j in range(_i + 1, len(ordered)):
                    _cn_j = ordered[_j]
                    _proj_j = _ctrl_proj.get(_cn_j, {})
                    for _cid, _obs in observations_per_camera.items():
                        if _obs is None or len(_obs) == 0:
                            continue
                        _pj = _proj_j.get(_cid)
                        if _pj is None or len(_pj) == 0:
                            continue
                        _d_j = np.linalg.norm(
                            _obs[:, None, :] - _pj[None, :, :], axis=2
                        ).min(axis=1)
                        _pi = _proj_i.get(_cid)
                        if _pi is not None and len(_pi) > 0:
                            _d_i = np.linalg.norm(
                                _obs[:, None, :] - _pi[None, :, :], axis=2
                            ).min(axis=1)
                            _near_either = (_d_j < _voronoi_px) | (_d_i < _voronoi_px)
                            if not _near_either.any():
                                continue
                            _reserve_mask = _near_either & (_d_j < _d_i)
                        else:
                            _near_j = _d_j < _reserve_px
                            if not _near_j.any():
                                continue
                            _reserve_mask = _near_j
                        if not _reserve_mask.any():
                            continue
                        _res_set = _reservations[_cn_i].setdefault(_cid, set())
                        _res_set.update(int(k) for k in np.where(_reserve_mask)[0])

        # ── Per-controller tracking ────────────────────────────────────────────
        for ctrl_name in ordered:
            if per_ctrl_observations is not None and ctrl_name in per_ctrl_observations:
                _obs_src = per_ctrl_observations[ctrl_name]
                _rad_src = (per_ctrl_radii or {}).get(ctrl_name) or {}
                _brt_src = (per_ctrl_brightnesses or {}).get(ctrl_name) or {}
            else:
                _obs_src = observations_per_camera
                _rad_src = radii_per_camera or {}
                _brt_src = brightnesses_per_camera or {}

            _ctrl_reservations = _reservations.get(ctrl_name, {})
            avail: Dict[int, tuple] = {}
            for cam_id in self.ctrl_trackers[ctrl_name].trackers:
                _extra = _ctrl_reservations.get(cam_id) or None
                fb, fr, fbt, keep = _filter_cam(cam_id, _obs_src, _rad_src, _brt_src,
                                                 extra_excluded=_extra)
                if fb is not None:
                    avail[cam_id] = (fb, fr, fbt, keep)

            # Build cross-controller occluder dict from controllers matched this frame.
            _occluders_per_cam = None
            if bool(_cfg.get('cross_controller_occlusion', False)) and len(ordered) > 1:
                for _occ_ctrl in ordered:
                    if _occ_ctrl == ctrl_name:
                        break
                    _occ_result = results.get(_occ_ctrl)
                    if not (_occ_result and _occ_result.get('T_world_ctrl') is not None):
                        continue
                    _occ_T_world = _occ_result['T_world_ctrl']
                    _occ_tracker = next(
                        (t for (cn, _), t in self.trackers.items() if cn == _occ_ctrl), None,
                    )
                    if _occ_tracker is None:
                        continue
                    _occluders_per_cam = {}
                    for _cam_id, _cam in self.cameras.items():
                        if _cam.T_world_cam is None:
                            continue
                        _T_cam_occ = _cam.T_world_cam.inverse().compose(_occ_T_world)
                        _occluders_per_cam[_cam_id] = (
                            _T_cam_occ.R.astype(np.float32),
                            _T_cam_occ.t.astype(np.float32),
                            _occ_tracker._geometry,
                        )
                    break

            results[ctrl_name] = self.ctrl_trackers[ctrl_name].update(
                avail, _obs_src, _rad_src, _brt_src,
                claimed_blobs,
                fixed_primary_cam=self._fixed_primary_cam,
                occluders_per_cam=_occluders_per_cam,
                self_cal=self._self_cal,
            )

        return results
