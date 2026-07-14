import copy
import time
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

def cheap_search_core(
    pose_searcher,
    prior: dict,
    matching_cfg: dict,
    blobs: np.ndarray,
    frame_ts_ns: int,
    blob_radii: Optional[np.ndarray] = None,
    blob_brightnesses: Optional[np.ndarray] = None,
    other_cameras_blobs: Optional[List] = None,
    blob_mask: Optional[np.ndarray] = None,
    occluders_per_cam: Optional[Dict] = None,
) -> Tuple[Optional[Dict], Optional[Tuple[np.ndarray, np.ndarray]], Optional[Tuple], Optional[Tuple]]:
    """
    Pure proximity + prior_constrained search — no brute-force, no CameraTracker
    instance needed, just a PoseSearcher and an explicit prior-state bundle. This is
    what lets a worker process run the cheap search using only its own resident
    PoseSearcher (see src/parallel_search.py) — a live CameraTracker's mutable
    tracking state can't be shared with a worker (fork only gives it a stale
    snapshot), so the caller must pass that state in explicitly instead.

    prior: {'prev_pose', 'prev_prev_pose', 'pose_history', 'vel_ema', 'prev_assignment'}
    — the same fields CameraTracker.search_cheap() reads from self. 'pose_history'
    entries are (rvec, tvec, ts_ns); 'vel_ema' is a position-per-second rate, not a
    raw step — see CameraTracker._predict_pose's docstring.

    frame_ts_ns: the current frame's real capture timestamp (nanoseconds, parsed
    from the frame's filename in main.py) — _predict_pose extrapolates against
    this exact elapsed time rather than assuming uniform frame spacing.

    Returns (solution_or_None, predicted_pose, normalized_prev_pose,
    normalized_prev_prev_pose) — the normalized values mirror the idempotent
    (3,1)/(3,) canonicalisation CameraTracker.search_cheap() applies to
    self.prev_pose/self.prev_prev_pose as a side effect; a caller with live state
    (CameraTracker.search_cheap) writes these back, a stateless worker just returns
    them for the orchestrator to write back into the real tracker.
    """
    blobs   = np.asarray(blobs, dtype=np.float32).reshape(-1, 2)
    n_blobs = len(blobs)

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

    _cfg = matching_cfg
    _use_proximity = bool(_cfg.get('use_proximity_match', True))

    prev_pose       = prior.get('prev_pose')
    prev_prev_pose  = prior.get('prev_prev_pose')
    pose_history    = prior.get('pose_history')
    vel_ema         = prior.get('vel_ema')
    prev_assignment = prior.get('prev_assignment')

    # Normalise prev_pose shapes (idempotent — canonicalises (3,1) rvec and (3,) tvec)
    if prev_pose is not None:
        rvec, tvec = prev_pose
        prev_pose = (
            np.asarray(rvec, dtype=np.float32).reshape(3, 1),
            np.asarray(tvec, dtype=np.float32).reshape(3),
        )
    if prev_prev_pose is not None:
        rvec, tvec = prev_prev_pose
        prev_prev_pose = (
            np.asarray(rvec, dtype=np.float32).reshape(3, 1),
            np.asarray(tvec, dtype=np.float32).reshape(3),
        )

    predicted_pose = CameraTracker._predict_pose(
        pose_history,
        frame_ts_ns,
        weight_decay=float(_cfg.get("pose_prediction_weight_decay", 0.7)),
        vel_ema_rate=vel_ema,
    )

    # Velocity-scaled search gates: expand proximity radius proportionally to speed.
    # v_px = ‖predicted.t − prev.t‖ × fx / depth  (pixels/frame, camera-agnostic)
    _v_px = 0.0
    if predicted_pose is not None and prev_pose is not None:
        _v_vec = predicted_pose[1].reshape(3) - np.asarray(prev_pose[1], np.float64).reshape(3)
        _depth = max(float(np.asarray(prev_pose[1]).reshape(3)[2]), 0.1)
        _v_px  = float(np.linalg.norm(_v_vec)) * pose_searcher.camera.fx / _depth
    _base_expansion = float(_cfg.get('proximity_expansion_px', 8.0))
    _prox_vel_k     = float(_cfg.get('proximity_expansion_velocity_k', 0.0))
    _eff_expansion  = _base_expansion + _prox_vel_k * _v_px
    # Uncertainty term: larger neighbourhood when prediction history is short.
    # Decays as 1/n — full boost at n=1 (constant-position), half at n=2, etc.
    _uncertainty_k = float(_cfg.get('proximity_expansion_uncertainty_k', 0.0))
    if _uncertainty_k > 0.0:
        _eff_expansion += _uncertainty_k / max(len(pose_history), 1)
    # Depth term: closer controller → larger pixel-space uncertainty → bigger neighbourhood.
    _depth_k = float(_cfg.get('proximity_expansion_depth_k', 0.0))
    if _depth_k > 0.0 and predicted_pose is not None:
        _ctrl_depth = max(float(predicted_pose[1].reshape(3)[2]), 0.01)
        _eff_expansion += _depth_k / _ctrl_depth

    solution = None

    if prev_pose is not None:
        # --- Primary: proximity (fast, assignment-locked) ---
        if _use_proximity and n_available >= 3:
            solution = pose_searcher.proximity_search(
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

        # ------------------------------------------------------------------
        # Low-blob-count fallback: prior-constrained translation solve
        # P2P (3 blobs): fix R, solve t from 2 pairs, validate with 3rd.
        # P1P (2 blobs): fix R + depth, solve (tx,ty) from 1 pair, validate with 2nd.
        # ------------------------------------------------------------------
        if (solution is None and prev_assignment is not None
                and 2 <= n_available <= 3):
            logger.debug(f"[{pose_searcher._ctrl} | cam {pose_searcher._cam} | track] n_blobs={n_available} + prior → prior_constrained_match")
            solution = pose_searcher.constrained_search(
                blobs_prox, predicted_pose,
                prior_assignment=prev_assignment,
                blob_radii=radii_prox,
                other_cameras_blobs=other_cameras_blobs,
            )
            if solution is not None and blob_mask is not None:
                solution['assignment'] = [(_avail_idx[b], lid) for b, lid in solution['assignment']]
                solution['_orig_idx']  = True

    return solution, predicted_pose, prev_pose, prev_prev_pose


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

        # Set (on every camera of a controller) when that controller's fused
        # update() came up empty; cleared only once a solution is next accepted.
        # prev_pose/pose_history/vel_ema are deliberately NOT cleared on a lost
        # frame (a controller recovered by brute-force still warm-starts next
        # frame from whatever history survives), so prev_pose stays non-None
        # straight through a loss — this flag is what tells finalize_search() to
        # use the loose re-acquisition check instead of the tight per-frame
        # pose-jump guard for the frame that actually recovers.
        self.tracking_lost_last_frame: bool = False

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
        target_ts_ns: int,
        weight_decay: float = 0.7,
        vel_ema_rate: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Predict pose at `target_ts_ns` from pose history.

        pose_history[0] = most recent (rvec, tvec, ts_ns); index increases toward
        older frames. Real capture intervals are NOT uniform (consecutive frame
        gaps can alternate between substantially different durations) — every
        branch below extrapolates against actual elapsed time, never "one
        history slot ahead".

        vel_ema_rate: if provided, a position-per-second rate (EMA-smoothed)
                 that overrides translation prediction:
                 pose_history[0].tvec + vel_ema_rate * dt_target.
                 Rotation prediction is always derived from pose history.

        n=0 → None (no information)
        n=1 → constant position (same pose) — vel_ema_rate is provably always
              None here in practice (nothing has been tracked long enough yet
              to populate it), kept for defensive symmetry only.
        n=2 → constant-velocity extrapolation scaled by the ratio of the target
              gap to the one known historical gap (matrix-based rotation,
              scaled via a fractional Rodrigues vector — see below).
        n>=3 → weighted degree-1 (linear) fit over real elapsed time; exponential
               weights (weight_decay^i, still per history slot, not per elapsed
               time — see module notes on why that's a deliberate simplification).
               Computes a weighted mean velocity across the window — averaging out
               the alternating big/small step oscillation from fast hand motion.

        Degenerate timestamps (dt_hist <= 0 from duplicate/out-of-order frames,
        or dt_target <= 0 from deep-debug mode's non-consecutive replay) fall
        back to a one-unit-step assumption rather than dividing by zero or
        extrapolating backwards.
        """
        n = len(pose_history)

        if n == 0:
            return None

        if n == 1:
            return (
                np.asarray(pose_history[0][0], np.float32).reshape(3, 1),
                np.asarray(pose_history[0][1], np.float32).reshape(3),
            )

        ts0 = int(pose_history[0][2])
        dt_target = (target_ts_ns - ts0) / 1e9  # seconds; may be <=0, guarded per branch

        if vel_ema_rate is not None:
            tvec_pred = (np.asarray(pose_history[0][1], np.float32).reshape(3)
                         + vel_ema_rate.reshape(3) * dt_target).astype(np.float32)
        else:
            tvec_pred = None  # filled in by the branch below

        if n == 2:
            rvec_n   = np.asarray(pose_history[0][0], np.float32).reshape(3, 1)
            tvec_n   = np.asarray(pose_history[0][1], np.float64).reshape(3)
            rvec_nm1 = np.asarray(pose_history[1][0], np.float32).reshape(3, 1)
            tvec_nm1 = np.asarray(pose_history[1][1], np.float64).reshape(3)
            ts_nm1   = int(pose_history[1][2])

            dt_hist = (ts0 - ts_nm1) / 1e9
            # frac=1.0 reproduces the old "assume one uniform step" behavior
            # exactly, for degenerate timestamps only.
            frac = (dt_target / dt_hist) if (dt_hist > 0 and dt_target > 0) else 1.0

            if tvec_pred is None:
                tvec_pred = (tvec_n + (tvec_n - tvec_nm1) * frac).astype(np.float32)

            R_n,   _ = cv2.Rodrigues(rvec_n)
            R_nm1, _ = cv2.Rodrigues(rvec_nm1)
            # Fractional rotation: scale the relative-rotation Rodrigues vector by
            # frac (small-angle-safe "fraction of a rotation") instead of always
            # applying the full historical step once more regardless of gap size.
            rvec_rel, _ = cv2.Rodrigues((R_n @ R_nm1.T).astype(np.float32))
            rvec_rel_scaled = (rvec_rel.reshape(3) * frac).astype(np.float32)
            R_rel_scaled, _ = cv2.Rodrigues(rvec_rel_scaled.reshape(3, 1))
            R_pred = R_rel_scaled @ R_n
            rvec_pred, _ = cv2.Rodrigues(R_pred.astype(np.float32))

            return rvec_pred.reshape(3, 1).astype(np.float32), tvec_pred.reshape(3)

        # n >= 3: weighted degree-1 (linear) fit — estimates a single average velocity
        # across the window. Degree-2 would add an acceleration term that amplifies
        # the alternating big/small step oscillation typical of fast hand motion.
        # time axis: real elapsed seconds from the most recent pose (t=0, negative
        # for older frames); predict at t=dt_target — the actual elapsed time to the
        # frame being predicted for, not always +1.
        t_pts   = np.array([(int(p[2]) - ts0) / 1e9 for p in pose_history], dtype=np.float64)
        weights = weight_decay ** np.arange(n, dtype=np.float64)
        _t_eval = dt_target if dt_target > 0 else 1.0

        # Translation: linear fit per axis (skipped when vel_ema already set tvec_pred)
        if tvec_pred is None:
            tvecs = np.stack([np.asarray(p[1], np.float64).reshape(3) for p in pose_history])
            tvec_pred = np.empty(3, dtype=np.float32)
            for ax in range(3):
                tvec_pred[ax] = np.polyval(np.polyfit(t_pts, tvecs[:, ax], deg=1, w=weights), _t_eval)

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
            rvec_rel_pred[ax] = np.polyval(np.polyfit(t_pts, rel_rvecs[:, ax], deg=1, w=weights), _t_eval)

        R_rel_pred, _ = cv2.Rodrigues(rvec_rel_pred.reshape(3, 1).astype(np.float32))
        rvec_pred, _ = cv2.Rodrigues((R_0 @ R_rel_pred).astype(np.float32))

        return rvec_pred.reshape(3, 1).astype(np.float32), tvec_pred.reshape(3)

    # -----------------------------------------------------
    # Tracking
    # -----------------------------------------------------
    def search_cheap(self, blobs: np.ndarray, frame_ts_ns: int,
                      blob_radii: Optional[np.ndarray] = None,
                      blob_brightnesses: Optional[np.ndarray] = None,
                      other_cameras_blobs: Optional[List] = None,
                      blob_mask: Optional[np.ndarray] = None,
                      occluders_per_cam: Optional[Dict] = None,
                      ) -> Tuple[Optional[Dict], Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Proximity + prior_constrained only — no brute-force. Reads self state (does
        not commit). Returns (solution_or_None, predicted_pose).

        A None solution here means this camera needs brute-force recovery to get any
        candidate at all this frame — whether because it has no prior (cold-start) or
        because proximity/prior_constrained came up empty despite having one. Callers
        doing cross-camera recovery should treat both cases identically: proceed to
        brute-force with pose_prior=predicted_pose (None for cold-start, matching
        today's unprimed cold-start brute call).
        """
        prior = {
            'prev_pose':       self.prev_pose,
            'prev_prev_pose':  self.prev_prev_pose,
            'pose_history':    self.pose_history,
            'vel_ema':         self.vel_ema,
            'prev_assignment': self.prev_assignment,
        }
        solution, predicted_pose, norm_prev, norm_prev_prev = cheap_search_core(
            self._pose_searcher, prior, self._matching_cfg,
            blobs, frame_ts_ns, blob_radii, blob_brightnesses,
            other_cameras_blobs, blob_mask, occluders_per_cam,
        )
        self.prev_pose      = norm_prev
        self.prev_prev_pose = norm_prev_prev
        return solution, predicted_pose

    def finalize_search(self, solution: Optional[Dict],
                         predicted_pose: Optional[Tuple[np.ndarray, np.ndarray]],
                         blobs: np.ndarray, blob_radii: Optional[np.ndarray] = None,
                         other_cameras_blobs: Optional[List] = None,
                         blob_mask: Optional[np.ndarray] = None,
                         occluders_per_cam: Optional[Dict] = None,
                         allow_expensive_fallback: bool = True) -> Optional[Dict]:
        """Validate and accept/reject an already-obtained candidate `solution` — from
        search_cheap(), or from a brute-force recovery attempt (cold-start or
        proximity-failed). Reads self state, does not commit results — call apply()
        with the returned value to commit state changes.

        allow_expensive_fallback: gates the brute-force retries that fire when this
        camera HAS a prior but the candidate is a pose-jump or its error is too high —
        brute-force can cost hundreds of ms for no result. Callers with other cameras
        available this frame should pass False on a first pass and only retry with
        True if the whole controller came up empty — a struggling camera can otherwise
        be warm-started next frame from a fused pose at zero cost.
        """
        blobs   = np.asarray(blobs, dtype=np.float32).reshape(-1, 2)
        n_blobs = len(blobs)
        n_available = int(blob_mask.sum()) if blob_mask is not None else n_blobs

        cam_idx = self.camera.camera_idx
        ctrl_name = self.model.name.replace("_controller", "")

        _cfg = self._matching_cfg
        _accept_err_px = float(_cfg.get('accept_error_px', 3.0))
        _pos_ax = _cfg.get('pose_jump_pos_thresh_m')
        _rot_ax = _cfg.get('pose_jump_rot_thresh_deg')
        _jump_kw = {}
        if _pos_ax is not None:
            _jump_kw['pos_thresh_xyz_m']   = tuple(_pos_ax)
        if _rot_ax is not None:
            _jump_kw['rot_thresh_xyz_deg'] = tuple(_rot_ax)

        # Re-acquisition: either a genuine cold start (prev_pose is None — not
        # reachable in practice today, kept for completeness) or the first
        # solution accepted after tracking_lost_last_frame was set. prev_pose/
        # pose_history are deliberately left untouched by a lost frame (so a
        # recovered controller can still warm-start from whatever history
        # survives), so prev_pose being non-None is NOT by itself evidence of
        # continuous per-frame tracking — tracking_lost_last_frame is.
        _reacquiring = self.prev_pose is None or self.tracking_lost_last_frame

        # Cold-start / re-acquisition plausibility check: reject a candidate
        # that's too far from the last known good pose. In deep-debug mode
        # frames are non-consecutive, so this check is skipped — the controller can
        # be anywhere.
        if solution is not None and _reacquiring and self.last_good_pose is not None and not is_deep():
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
        # Pose-jump guard against prev_pose (tight, per-frame — continuous
        # tracking only; skipped while re-acquiring, see above)
        # ------------------------------------------------------------------
        if solution is not None and self.prev_pose is not None and not _reacquiring:
            rvec_p, tvec_p = self.prev_pose
            if self._pose_jump_too_large(
                solution["rvec"], solution["tvec"],
                rvec_p, tvec_p,
                **_jump_kw,
            ):
                logger.warning(f"[{ctrl_name} | cam {cam_idx} | tracking] Pose jump detected "
                               f"(method={solution.get('method','?')}, "
                               f"err={solution['error']:.2f} px) — "
                               f"attempting brute recovery.")
                solution = None
                if allow_expensive_fallback and n_available >= 4:
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
        if (solution is not None and solution["error"] >= _accept_err_px
                and allow_expensive_fallback and n_available >= 4):
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

    def search(self, blobs: np.ndarray, blob_radii: Optional[np.ndarray] = None,
               blob_brightnesses: Optional[np.ndarray] = None,
               other_cameras_blobs: Optional[List] = None,
               blob_mask: Optional[np.ndarray] = None,
               occluders_per_cam: Optional[Dict] = None,
               allow_expensive_fallback: bool = True) -> Optional[Dict]:
        """Pure pose solve — reads self state, does not commit results.

        Returns a validated solution dict (error below threshold, T_world_ctrl populated)
        or None.  Call apply() with the returned value to commit state changes.

        Thin wrapper: search_cheap() for a candidate; if none and this camera can use
        brute-force (cold-start is unconditional; otherwise gated on
        allow_expensive_fallback — same reasoning as finalize_search), run it
        monolithically; then finalize_search() to validate/accept.
        """
        solution, predicted_pose = self.search_cheap(
            blobs, blob_radii, blob_brightnesses, other_cameras_blobs,
            blob_mask, occluders_per_cam,
        )

        if solution is None:
            n_available = int(blob_mask.sum()) if blob_mask is not None else len(np.asarray(blobs).reshape(-1, 2))
            cam_idx = self.camera.camera_idx
            ctrl_name = self.model.name.replace("_controller", "")
            if self.prev_pose is not None:
                if allow_expensive_fallback and n_available >= 4:
                    logger.debug(f"[{ctrl_name} | cam {cam_idx} | track] proximity → None, running brute fallback")
                    solution = self.brute_match(blobs, pose_prior=predicted_pose,
                                                other_cameras_blobs=other_cameras_blobs,
                                                blob_radii=blob_radii,
                                                blob_mask=blob_mask,
                                                occluders_per_cam=occluders_per_cam)
            else:
                if n_available >= 4:
                    logger.debug(f"[{ctrl_name} | cam {cam_idx} | track] no prev_pose → cold-start brute")
                    solution = self.brute_match(blobs, other_cameras_blobs=other_cameras_blobs,
                                                blob_radii=blob_radii,
                                                blob_mask=blob_mask,
                                                occluders_per_cam=occluders_per_cam)

        return self.finalize_search(
            solution, predicted_pose, blobs, blob_radii, other_cameras_blobs,
            blob_mask, occluders_per_cam, allow_expensive_fallback,
        )

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
            elif self.consecutive_failures > 0 and self.last_good_pose is not None:
                _warmstart_max = int(self._matching_cfg.get('pose_gap_velocity_warmstart_frames', 3))
                if self.consecutive_failures <= _warmstart_max:
                    _lg_tvec  = np.asarray(self.last_good_pose[1], np.float64).reshape(3)
                    _new_tvec = np.asarray(result["tvec"],          np.float64).reshape(3)
                    self.vel_ema = ((_new_tvec - _lg_tvec) / self.consecutive_failures).astype(np.float32)
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


# =========================================================
# 2.5 PER-CONTROLLER TRACKER
# =========================================================

class ControllerTracker:
    """Manages all camera-trackers for a single controller.

    Every available camera searches independently (its own blobs only, no
    aux-camera folding), then all resolved per-camera solutions are fused into
    one pose via fuse_camera_poses(). Owns state propagation to every
    CameraTracker from that fused pose and claimed-blob registration.
    Cross-controller concerns (Voronoi reservation, ordering) remain in TrackingSystem.
    """

    def __init__(self, ctrl_name: str, cameras: Dict[int, "Camera"],
                 trackers: Dict[int, CameraTracker],
                 matching_cfg: Optional[dict] = None):
        self.ctrl_name        = ctrl_name
        self.cameras          = cameras
        self.trackers         = trackers           # {cam_id: CameraTracker}
        self._matching_cfg    = matching_cfg or {}
        # Informational only (reporting/self-cal anchor from the last successful frame) —
        # every camera searches independently every frame, so this no longer gates work.
        self._designated_primary: Optional[int]  = None

    def update(
        self,
        avail:         Dict[int, tuple],        # cam_id → (blobs, radii, brts, orig_idx)
        obs_src:       Dict[int, np.ndarray],   # full observations per camera
        rad_src:       Dict,
        brt_src:       Dict,
        claimed_blobs: Dict[int, Set[int]],     # mutated in place
        frame_ts_ns:   int,
        fixed_primary_cam: Optional[int] = None,
        occluders_per_cam: Optional[Dict] = None,
        self_cal=None,
        pool=None,
        allow_brute: bool = True,
        force_brute: bool = False,
    ) -> Optional[Dict]:
        """allow_brute=False stops after the cheap (proximity/prior_constrained)
        pass and returns None instead of running tier-round brute-force recovery
        — lets a caller detect "every camera's cheap search failed" and react
        (e.g. re-detect blobs cold) before paying for brute-force on blobs that
        were detected under a now-untrustworthy prediction.

        force_brute=True skips the cheap pass entirely and goes straight to
        brute-force, with no pose_prior (same as a cold-start first frame) —
        for a caller that already re-detected blobs cold after cheap search
        failed on the (now known bad) extrapolated pose: retrying cheap search
        against that same untrustworthy prediction, just with a fuller blob
        set, would fail for the same reason it failed the first time."""
        if not avail:
            for _t in self.trackers.values():
                _t.tracking_lost_last_frame = True
            return None

        # ── Every available camera searches independently ──────────────────────
        # Own blobs only — no aux-camera folding at the per-camera search level;
        # cross-camera fusion happens once, below, via fuse_camera_poses().
        #
        # Pass 1: cheap methods (proximity + prior_constrained) for every camera — a
        # camera with a prior whose cheap methods come up short can simply contribute
        # nothing this frame and get warm-started next frame from the fused pose, at
        # zero cost. Only if NO camera produced anything this frame (the controller is
        # about to lose track entirely) do we pay for brute-force recovery — run in
        # tier-rounds across every camera needing it (see below), rather than one
        # camera exhausting its whole tier ladder before the next is even tried.
        #
        # When `pool` is given, both stages dispatch to it: pass 1 submits one cheap-
        # search task per camera; the tier-round loop submits one brute-tier task per
        # camera still needing that round, waiting for the whole round before
        # deciding to widen or stop (no mid-tier cancellation — see plan). Without a
        # pool, both stages fall back to the equivalent sequential loop.
        cam_solutions: List[dict] = []
        predicted_pose_by_cid: Dict[int, Optional[Tuple[np.ndarray, np.ndarray]]] = {}

        t_cheap = 0.0
        if not force_brute:
            _t_cheap_start = time.perf_counter()
            _cheap_specs = []   # (cid, tracker, obs_full, rad_full, brt_full, mask, av_orig)
            for cid, (av_blobs, av_radii, av_brts, av_orig) in avail.items():
                if len(av_blobs) == 0:
                    continue
                tracker  = self.trackers[cid]
                obs_full = obs_src[cid]
                rad_full = rad_src.get(cid) if rad_src else None
                brt_full = brt_src.get(cid) if brt_src else None
                mask = np.zeros(len(obs_full), dtype=bool)
                mask[av_orig] = True
                _cheap_specs.append((cid, tracker, obs_full, rad_full, brt_full, mask, av_orig))

            _cheap_raw: Dict[int, tuple] = {}   # cid -> (solution, predicted_pose)
            if pool is not None and _cheap_specs:
                from src.parallel_search import run_cheap_search
                _futures = {}
                for cid, tracker, obs_full, rad_full, brt_full, mask, av_orig in _cheap_specs:
                    prior = {
                        'prev_pose':       tracker.prev_pose,
                        'prev_prev_pose':  tracker.prev_prev_pose,
                        'pose_history':    tracker.pose_history,
                        'vel_ema':         tracker.vel_ema,
                        'prev_assignment': tracker.prev_assignment,
                    }
                    _futures[cid] = pool.submit(
                        run_cheap_search, (self.ctrl_name, cid), self._matching_cfg, prior,
                        obs_full, frame_ts_ns, rad_full, brt_full, mask, occluders_per_cam,
                    )
                for cid, fut in _futures.items():
                    solution, predicted_pose, norm_prev, norm_prev_prev = fut.result()
                    self.trackers[cid].prev_pose      = norm_prev
                    self.trackers[cid].prev_prev_pose = norm_prev_prev
                    _cheap_raw[cid] = (solution, predicted_pose)
            else:
                for cid, tracker, obs_full, rad_full, brt_full, mask, av_orig in _cheap_specs:
                    _cheap_raw[cid] = tracker.search_cheap(
                        obs_full, frame_ts_ns, blob_radii=rad_full, blob_brightnesses=brt_full,
                        other_cameras_blobs=None, blob_mask=mask,
                        occluders_per_cam=occluders_per_cam,
                    )

            for cid, tracker, obs_full, rad_full, brt_full, mask, av_orig in _cheap_specs:
                solution, predicted_pose = _cheap_raw[cid]
                predicted_pose_by_cid[cid] = predicted_pose
                solution = tracker.finalize_search(
                    solution, predicted_pose, obs_full,
                    blob_radii=rad_full, other_cameras_blobs=None,
                    blob_mask=mask, occluders_per_cam=occluders_per_cam,
                    allow_expensive_fallback=False,
                )
                if solution is None:
                    continue
                if not solution.get('_orig_idx', False):
                    solution["assignment"] = [(av_orig[b], lid) for b, lid in solution["assignment"]]
                cam_solutions.append({"cam_id": cid, "tracker": tracker, "solution": solution})

            t_cheap = time.perf_counter() - _t_cheap_start

        t_new_state = t_tier_rounds = t_finalize = 0.0
        n_rounds = 0
        _pool_rt_ms: Dict[int, float] = {}   # cid -> worst brute-tier pool round-trip, if pool was used

        if force_brute or (not cam_solutions and allow_brute):
            # ── Tier-round brute-force recovery ─────────────────────────────────
            # tier_0 for every camera with a chance (>=4 available blobs); wait for
            # the whole round to finish before deciding anything (no mid-tier
            # cancellation); stop widening the instant any camera hits strong_found
            # this round.
            states: Dict[int, object] = {}
            _t_new_state_start = time.perf_counter()
            for cid, (av_blobs, av_radii, av_brts, av_orig) in avail.items():
                if len(av_blobs) == 0:
                    continue
                tracker  = self.trackers[cid]
                obs_full = obs_src[cid]
                rad_full = rad_src.get(cid) if rad_src else None
                mask = np.zeros(len(obs_full), dtype=bool)
                mask[av_orig] = True
                states[cid] = tracker._pose_searcher.new_brute_state(
                    obs_full, pose_prior=predicted_pose_by_cid.get(cid),
                    other_cameras_blobs=None, blob_radii=rad_full,
                    blob_mask=mask, occluders_per_cam=occluders_per_cam,
                )
            t_new_state = time.perf_counter() - _t_new_state_start

            _any_ps = next(
                (self.trackers[cid]._pose_searcher for cid, st in states.items() if st is not None),
                None,
            )
            max_tiers = len(_any_ps._c_brute_depth_tiers) if _any_ps is not None else 0

            _t_tier_start = time.perf_counter()
            if pool is not None:
                from src.parallel_search import run_brute_tier
                for tier_idx in range(max_tiers):
                    remaining = [cid for cid, st in states.items() if st is not None and not st.strong_found]
                    if not remaining:
                        break
                    n_rounds += 1
                    _submit_t0 = time.perf_counter()
                    _futures = {
                        cid: pool.submit(run_brute_tier, (self.ctrl_name, cid), states[cid], tier_idx)
                        for cid in remaining
                    }
                    for cid, fut in _futures.items():
                        states[cid] = fut.result()
                        _rt = (time.perf_counter() - _submit_t0) * 1000
                        _pool_rt_ms[cid] = max(_pool_rt_ms.get(cid, 0.0), _rt)
                    if any(states[cid].strong_found for cid in remaining):
                        break
            else:
                for tier_idx in range(max_tiers):
                    remaining = [cid for cid, st in states.items() if st is not None and not st.strong_found]
                    if not remaining:
                        break
                    n_rounds += 1
                    for cid in remaining:
                        self.trackers[cid]._pose_searcher.brute_search_tier(states[cid], tier_idx)
                    if any(states[cid].strong_found for cid in remaining):
                        break
            t_tier_rounds = time.perf_counter() - _t_tier_start

            _t_finalize_start = time.perf_counter()
            for cid, st in states.items():
                if st is None:
                    continue
                tracker = self.trackers[cid]
                sol = tracker._pose_searcher.finalize_brute_state(st)
                if sol is None:
                    continue
                _, _, _, av_orig = avail[cid]
                if not sol.get('_orig_idx', False):
                    sol["assignment"] = [(av_orig[b], lid) for b, lid in sol["assignment"]]
                obs_full = obs_src[cid]
                rad_full = rad_src.get(cid) if rad_src else None
                mask = np.zeros(len(obs_full), dtype=bool)
                mask[av_orig] = True
                sol = tracker.finalize_search(
                    sol, predicted_pose_by_cid.get(cid), obs_full,
                    blob_radii=rad_full, other_cameras_blobs=None,
                    blob_mask=mask, occluders_per_cam=occluders_per_cam,
                    allow_expensive_fallback=True,
                )
                if sol is not None:
                    cam_solutions.append({"cam_id": cid, "tracker": tracker, "solution": sol})
            t_finalize = time.perf_counter() - _t_finalize_start

        _pool_rt_str = (
            "  pool_rt=[" + " ".join(f"cam{c}={ms:.1f}ms" for c, ms in sorted(_pool_rt_ms.items())) + "]"
            if _pool_rt_ms else ""
        )
        logger.debug(
            f"[{self.ctrl_name}] update timing: cheap={t_cheap*1000:.1f}ms  "
            f"brute(new_state={t_new_state*1000:.1f}ms "
            f"tier_rounds={t_tier_rounds*1000:.1f}ms[{n_rounds} rounds] "
            f"finalize={t_finalize*1000:.1f}ms){_pool_rt_str}"
        )

        if not cam_solutions:
            for _t in self.trackers.values():
                _t.tracking_lost_last_frame = True
            return None

        return self._fuse_and_finalize(
            cam_solutions, obs_src, claimed_blobs, frame_ts_ns,
            fixed_primary_cam=fixed_primary_cam, self_cal=self_cal,
        )

    def _fuse_and_finalize(
        self,
        cam_solutions: List[dict],
        obs_src: Dict[int, np.ndarray],
        claimed_blobs: Optional[Dict[int, Set[int]]],
        frame_ts_ns: int,
        fixed_primary_cam: Optional[int] = None,
        self_cal=None,
    ) -> Dict:
        """Fuse every camera's independent solution into one controller pose,
        register claimed blobs, feed self-calibration, and propagate the fused
        pose into every camera tracker's state. Shared tail for both the
        tier-round recovery path (update(), above) and the batched warm-warm
        path (TrackingSystem.update_warm_batch()) — extracted so the ~140
        lines of fusion/propagation logic aren't duplicated between them.

        claimed_blobs=None skips claim registration entirely — used by the
        warm-warm batched path, which deliberately does not track
        cross-controller blob claims (two controllers assigning the same
        blob is accepted there; each controller's own RANSAC/inlier scoring
        absorbs it). cam_solutions must be non-empty.
        """
        from src.pose_search import fuse_camera_poses

        # ── Fuse every camera's independent solution into one pose ─────────────
        _fuse_in = [
            {
                "camera":       self.cameras[cs["cam_id"]],
                "blobs":        obs_src[cs["cam_id"]],
                "pairs":        cs["solution"]["assignment"],
                "T_world_ctrl": cs["solution"]["T_world_ctrl"],
                "error":        cs["solution"]["error"],
                "confidence":   cs["solution"].get("confidence", 1.0),
            }
            for cs in cam_solutions
        ]
        model_positions = next(iter(self.trackers.values())).model.positions
        _t_fuse_start = time.perf_counter()
        T_world_ctrl, fused_err = fuse_camera_poses(
            _fuse_in, model_positions, matching_cfg=self._matching_cfg,
        )
        _t_fuse = time.perf_counter() - _t_fuse_start

        # ── Anchor camera for reporting/self-cal: fixed_primary_cam if it solved
        # this frame, else whichever camera had the most inlier pairs ──────────
        _solved_ids = {cs["cam_id"] for cs in cam_solutions}
        if fixed_primary_cam is not None and fixed_primary_cam in _solved_ids:
            primary_cam_id = fixed_primary_cam
        else:
            primary_cam_id = max(
                cam_solutions, key=lambda cs: len(cs["solution"]["assignment"])
            )["cam_id"]
        anchor_solution = next(
            cs["solution"] for cs in cam_solutions if cs["cam_id"] == primary_cam_id
        )

        # A camera whose own independent search failed this frame is left out of
        # the fusion entirely — no Hungarian snap-recovery. Its confidence would
        # be zero anyway (no RANSAC/hypothesis validation behind a raw nearest-
        # neighbour snap), so it can't influence the fused pose either way; but an
        # unvalidated snap's pairs still get registered in claimed_blobs/
        # aux_assignments below, which can falsely block a *different* controller
        # from matching a blob it actually owns in this camera this frame. Zero
        # upside, real downside — so a failed camera is simply absent this frame.

        # Per-camera importance: each contributor's share of the fusion weight —
        # (1/sqrt(n_pairs)) * confidence, normalised to sum to 1 — i.e. how much
        # this frame's fused pose actually leaned on each camera. Replaces the old
        # "anchor camera" concept for reporting purposes.
        _raw_w = {
            e["camera"].camera_idx: float(e.get("confidence", 1.0)) / max(len(e["pairs"]), 1) ** 0.5
            for e in _fuse_in
        }
        _w_total = sum(_raw_w.values()) or 1.0
        _importance_str = "  ".join(
            f"cam{cid}={w / _w_total:.2f}" for cid, w in sorted(_raw_w.items())
        )

        logger.debug(
            f"[{self.ctrl_name}] Joint fusion: {len(cam_solutions)} solved  "
            f"err={fused_err:.2f}px  lm={_t_fuse*1000:.1f}ms  "
            f"importance=[{_importance_str}]"
        )

        solution = dict(anchor_solution)
        solution["primary_cam"]  = primary_cam_id
        solution["T_world_ctrl"] = T_world_ctrl
        solution["error"]        = fused_err
        _other_assignments = {
            cs["cam_id"]: cs["solution"]["assignment"]
            for cs in cam_solutions if cs["cam_id"] != primary_cam_id
        }
        solution["aux_assignments"] = _other_assignments
        solution["aux_cameras"] = [(cid, len(pairs)) for cid, pairs in _other_assignments.items()]

        # Reporting/self-cal anchor tracking (per-camera importance above is now the
        # reported signal; kept only for get_designated_primary_cameras()).
        self._designated_primary = primary_cam_id

        # ── Register claimed blobs (every camera that actually solved) ──────────
        if claimed_blobs is not None:
            for cs in cam_solutions:
                for b, _ in cs["solution"]["assignment"]:
                    claimed_blobs.setdefault(cs["cam_id"], set()).add(b)

        # ── Self-calibration feed ────────────────────────────────────────────────
        if (self_cal is not None
                and primary_cam_id == self_cal.primary_camera.camera_idx):
            _T_cam_ctrl_anchor = self.cameras[primary_cam_id].T_world_cam.inverse().compose(T_world_ctrl)
            _R_prim = _T_cam_ctrl_anchor.R.astype(np.float32)
            _rv_prim, _ = cv2.Rodrigues(_R_prim)
            _t_prim = _T_cam_ctrl_anchor.t.astype(np.float32)
            _sc_aux_obs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            for _aux_cid, _aux_pairs in _other_assignments.items():
                if len(_aux_pairs) < 3:
                    continue
                _led_ids   = np.array([lid for _, lid in _aux_pairs], dtype=np.int32)
                _blob_idxs = np.array([b   for b, _  in _aux_pairs], dtype=np.int32)
                _led_pos   = self.trackers[primary_cam_id].model.positions[_led_ids]
                _pts_prim  = (_R_prim @ _led_pos.T).T + _t_prim
                _blobs_aux = obs_src[_aux_cid][_blob_idxs]
                _sc_aux_obs[_aux_cid] = (_pts_prim, _blobs_aux)
            if _sc_aux_obs:
                self_cal.add_frame(
                    _rv_prim, _t_prim,
                    primary_error=fused_err,
                    primary_inliers=len(anchor_solution["assignment"]),
                    aux_observations=_sc_aux_obs,
                )

        # ── State propagation: every camera's own history is warm-started from
        # the one fused pose, using that camera's own assignment if it solved ──
        _beta = float(self._matching_cfg.get("pose_prediction_vel_ema_beta", 0.3))
        for _cid, _tracker in self.trackers.items():
            _T_cam_ctrl = self.cameras[_cid].T_world_cam.inverse().compose(T_world_ctrl)
            _rv_np, _ = cv2.Rodrigues(_T_cam_ctrl.R.astype(np.float32))
            _tv_np = _T_cam_ctrl.t.astype(np.float32)
            if _tracker.prev_pose is not None:
                _step = (_tv_np.reshape(3).astype(np.float64)
                         - np.asarray(_tracker.prev_pose[1], np.float64).reshape(3)).astype(np.float32)
                # vel_ema is a position-per-second RATE, not a raw per-call step —
                # pose_history is non-empty whenever prev_pose is set (both are
                # always written together, right here), so [0][2] is this
                # tracker's previous frame's real timestamp.
                _prev_ts_ns = int(_tracker.pose_history[0][2])
                _dt_s = (frame_ts_ns - _prev_ts_ns) / 1e9
                if _dt_s > 0:
                    _step_rate = _step / _dt_s
                    _tracker.vel_ema = (_beta * _step_rate + (1.0 - _beta) * _tracker.vel_ema
                                       if _tracker.vel_ema is not None else _step_rate)
                # else: degenerate/duplicate timestamp — leave vel_ema at its
                # previous value rather than dividing by a non-positive dt.
            _tracker.prev_prev_pose = _tracker.prev_pose
            _tracker.prev_pose      = (_rv_np.reshape(3, 1), _tv_np)
            _tracker.last_good_pose = _tracker.prev_pose
            _tracker.pose_history.appendleft((_rv_np.reshape(3, 1), _tv_np, frame_ts_ns))
            _own_asgn = (
                anchor_solution["assignment"] if _cid == primary_cam_id
                else _other_assignments.get(_cid)
            )
            if _own_asgn is not None:
                _tracker.prev_assignment      = _own_asgn
                _tracker.last_good_assignment = _own_asgn
            elif _tracker.prev_assignment is None:
                _tracker.last_good_assignment = None
            _tracker.consecutive_failures = 0
            _tracker.tracking_lost_last_frame = False

        return solution


# =========================================================
# 3. SYSTEM (multi-controller, multi-camera)
# =========================================================

class TrackingSystem:
    def __init__(self, controllers: List[ControllerModel], cameras: List[Camera],
                 matching_cfg: dict = None, geometry_cfg: dict = None,
                 geometry_cfg_per_ctrl: dict = None,
                 self_calibration_cfg: dict = None,
                 blob_detection_cfg: dict = None):

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

        _parallel_enabled = bool(self._matching_cfg.get('parallel_search_enabled', True))
        if _parallel_enabled:
            from src.parallel_search import register_pose_searcher_spec

        for ctrl in controllers:
            ctrl_cam_trackers: Dict[int, CameraTracker] = {}
            for cam in cameras:
                key = (ctrl.name, cam.camera_idx)
                geo = (geometry_cfg_per_ctrl or {}).get(ctrl.name) or geometry_cfg
                cam_tracker = CameraTracker(cam, ctrl, matching_cfg=matching_cfg, geometry_cfg=geo)
                self.trackers[key]           = cam_tracker
                ctrl_cam_trackers[cam.camera_idx] = cam_tracker
                if _parallel_enabled:
                    # Register the picklable construction spec (not the live
                    # PoseSearcher) — the pool uses 'spawn', so each worker builds
                    # its own PoseSearcher from these at startup. See
                    # src/parallel_search.py's module docstring for why not 'fork'.
                    register_pose_searcher_spec(key, cam, ctrl, geo, matching_cfg)
            self.ctrl_trackers[ctrl.name] = ControllerTracker(
                ctrl.name, self.cameras, ctrl_cam_trackers, matching_cfg=matching_cfg,
            )

        # Independent of parallel_search_enabled — lets blob-detection parallelism be
        # turned off on its own (e.g. if IPC/pickling the debug canvases turns out to
        # dominate the warm path's actual per-LED-ROI compute) without giving up
        # pose-search parallelism, which still needs the pool regardless.
        self._blob_parallel_enabled = (
            _parallel_enabled
            and bool(self._matching_cfg.get('parallel_blob_detection_enabled', True))
            and blob_detection_cfg is not None
        )
        if self._blob_parallel_enabled:
            from src.parallel_search import register_blob_detector_spec
            for cam in cameras:
                register_blob_detector_spec(cam.camera_idx, blob_detection_cfg)

        # ── Persistent process pool for parallel cheap-search / brute-tier dispatch,
        # and (when enabled) parallel blob detection ──────────────────────────────
        self._pool = None
        if _parallel_enabled:
            from src.parallel_search import create_pool, warmup_pool
            from src import debug_config
            _workers_cfg = self._matching_cfg.get('parallel_search_workers')
            _n_workers = int(_workers_cfg) if _workers_cfg is not None else max(1, len(self.cameras))
            self._pool = create_pool(_n_workers, debug_cfg=debug_config.get_config())
            import atexit
            atexit.register(self._pool.shutdown)
            # Force every worker to spawn now (interpreter boot + heavy imports +
            # PoseSearcher/BlobDetector construction) instead of lazily on the first
            # tracked frame — spawn only starts a worker when it has a task to run.
            warmup_pool(self._pool, _n_workers)

    def get_pool(self):
        """The persistent process pool (or None if parallel search is disabled).
        Callers submitting blob-detection tasks must also check
        blob_parallel_enabled — the pool can exist for pose search alone while blob
        parallelism is independently turned off, in which case workers never built a
        _BLOB_DETECTORS registry."""
        return self._pool

    @property
    def blob_parallel_enabled(self) -> bool:
        return self._blob_parallel_enabled

    def shutdown(self) -> None:
        """Cleanly shut down the process pool, if one was created. Safe to call more
        than once; also registered via atexit as a safety net."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def get_designated_primary_cameras(self) -> Dict[str, Optional[int]]:
        """Return {ctrl_name: anchor_cam_id} as of the last successful tracking frame.

        Informational only — every camera searches independently every frame regardless
        of this value. None means no successful frame has been tracked yet.
        """
        return {name: ct._designated_primary for name, ct in self.ctrl_trackers.items()}

    def get_predicted_led_projections_per_camera(
        self,
        frame_ts_ns: int,
    ) -> Tuple[Dict[int, Dict[str, Optional[np.ndarray]]], Dict[int, Dict[str, float]]]:
        """Return (proj_hints, vel_hints).

        frame_ts_ns: the current frame's real capture timestamp (nanoseconds,
        parsed from the frame's filename) — predictions are extrapolated to this
        exact elapsed time, not an assumed uniform frame step.

        proj_hints: {cam_id: {ctrl_name: Nx5 array or None}}
          Each row: [proj_x, proj_y, depth_m, facing_cos, led_id]
          for each LED visible from the predicted pose.
          None when no prior pose exists for this (ctrl, cam) pair.

        vel_hints: {cam_id: {ctrl_name: v_px}}
          Estimated pixel displacement expected over THIS frame's actual gap
          since the last known pose (not a fixed "per-frame" rate — see below).
          0.0 when no prediction is available.
        """
        ctrl_names   = sorted({ctrl for ctrl, _ in self.trackers})
        _facing_deg  = float(self._matching_cfg.get('led_facing_angle_deg', 86.0))
        # A camera predicting fewer visible LEDs than min_inliers cannot produce
        # a valid pose solve regardless of detection quality (matching/proximity
        # and brute-force both require >= this many inlier pairs — see
        # PoseSearcher._c_prox_min_inliers / _c_brute_min_inliers in
        # pose_search.py, same matching_cfg key) — so treat it the same as zero
        # visible LEDs below, rather than handing it a prediction that's
        # geometrically guaranteed not to help.
        _min_vis_leds = int(self._matching_cfg.get(
            'min_visible_leds_for_search', self._matching_cfg.get('min_inliers', 4)))
        # Theoretical visibility (>= _min_vis_leds) doesn't mean "worth searching" —
        # a controller can end up warm-tracked in 3-4 cameras when 2 would do, and a
        # grazing-angle 3rd/4th camera is often barely useful. Capped below, after
        # every camera's visibility is computed, by ranking on total facing_cos.
        _max_warm_cams = int(self._matching_cfg.get('max_warm_search_cameras', 0)) or None
        result:       Dict[int, Dict[str, Optional[np.ndarray]]] = {}
        vel_hints:    Dict[int, Dict[str, float]]                = {}
        radius_hints: Dict[int, Dict[str, float]]                = {}

        # Pre-read all four expansion terms once — same values used in track()
        _base_r = float(self._matching_cfg.get('proximity_expansion_px', 8.0))
        _vel_k  = float(self._matching_cfg.get('proximity_expansion_velocity_k', 0.0))
        _unc_k  = float(self._matching_cfg.get('proximity_expansion_uncertainty_k', 0.0))
        _dpt_k  = float(self._matching_cfg.get('proximity_expansion_depth_k', 0.0))
        # Same weight_decay cheap_search_core passes to _predict_pose — keep this
        # projection hint aligned with what proximity_search actually matches
        # against, or the search ROI drawn here silently disagrees with cheap_search_core's
        # own (vel_ema-corrected) prediction under fast/oscillating motion.
        _weight_decay = float(self._matching_cfg.get('pose_prediction_weight_decay', 0.7))

        for cam_id, camera in self.cameras.items():
            proj_per_ctrl:   Dict[str, Optional[np.ndarray]] = {}
            vel_per_ctrl:    Dict[str, float]                = {}
            radius_per_ctrl: Dict[str, float]               = {}
            for ctrl_name in ctrl_names:
                tracker = self.trackers.get((ctrl_name, cam_id))
                pred = (CameraTracker._predict_pose(
                            tracker.pose_history,
                            frame_ts_ns,
                            weight_decay=_weight_decay,
                            vel_ema_rate=tracker.vel_ema,
                        ) if tracker else None)
                if pred is None:
                    proj_per_ctrl[ctrl_name]   = None
                    vel_per_ctrl[ctrl_name]    = 0.0
                    radius_per_ctrl[ctrl_name] = _base_r
                    continue
                rvec_pred, tvec_pred = pred
                _ph = tracker.pose_history
                # dt from the last known pose to THIS frame — converts a rate
                # (vel_ema, position/second) or a historical per-step delta into
                # the expected displacement over the actual upcoming gap, not a
                # fixed "per assumed-uniform-frame" amount.
                _dt_target = max((frame_ts_ns - int(_ph[0][2])) / 1e9, 0.0) if _ph else 0.0
                if tracker.vel_ema is not None:
                    vel_disp_3d = tracker.vel_ema.astype(np.float64) * _dt_target
                elif len(_ph) >= 2:
                    _dt_hist = (int(_ph[0][2]) - int(_ph[1][2])) / 1e9
                    _step    = (np.asarray(_ph[0][1], np.float64).reshape(3)
                                - np.asarray(_ph[1][1], np.float64).reshape(3))
                    vel_disp_3d = (_step / _dt_hist * _dt_target) if _dt_hist > 0 else np.zeros(3, np.float64)
                else:
                    vel_disp_3d = np.zeros(3, np.float64)

                R_pred, _ = cv2.Rodrigues(rvec_pred.reshape(3, 1))
                R_pred    = R_pred.astype(np.float32)
                tvec_pred = tvec_pred.reshape(3).astype(np.float32)

                positions = tracker.model.positions
                normals   = tracker.model.normals

                vis_ids = np.where(_visible_mask(
                    R_pred, tvec_pred, positions, normals, tracker._geometry,
                    cam_K=camera.camera_matrix, cam_dc=camera.dist_coeffs,
                    cam_w=camera.width, cam_h=camera.height, cam_rpmax=camera.rpmax,
                    cam_is_fisheye=camera.is_fisheye,
                    facing_threshold_deg=_facing_deg,
                ) >= 1.0)[0]
                if len(vis_ids) < _min_vis_leds:
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
                _v_px       = float(np.linalg.norm(vel_disp_3d)) * camera.fx / _depth_pred
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

        # ── Cap warm-search cameras per controller ──────────────────────────────
        # Rank every camera that's theoretically visible for this controller by
        # sum(facing_cos) over its visible LEDs — a single score that jointly
        # captures "how many LEDs" and "how well-oriented" (column 3 of the Nx5
        # array already computed above, zero new work), keep only the top
        # max_warm_search_cameras. Demoted cameras are reset exactly like the
        # "too few visible LEDs" branch above, so every downstream consumer
        # (blob-detection skip, _filter_cam, update_warm_batch's non-empty check)
        # treats them identically to "not visible" — no other code needs to know
        # about this cap. A cold controller has no prediction on any camera, so
        # this is a no-op for it (brute-force reacquisition isn't gated by
        # proj_hints at all).
        if _max_warm_cams is not None:
            for ctrl_name in ctrl_names:
                _scored = [
                    (float(result[cid][ctrl_name][:, 3].sum()), cid)
                    for cid in result
                    if result[cid][ctrl_name] is not None
                ]
                if len(_scored) <= _max_warm_cams:
                    continue
                _scored.sort(reverse=True)
                for _, cid in _scored[_max_warm_cams:]:
                    result[cid][ctrl_name]       = None
                    vel_hints[cid][ctrl_name]    = 0.0
                    radius_hints[cid][ctrl_name] = _base_r

        return result, vel_hints, radius_hints

    def solved_led_geometry(
        self,
        ctrl_name: str,
        cam_id: int,
        T_world_ctrl: Transform,
        led_ids,
    ) -> Dict[int, Tuple[float, float]]:
        """Depth (m) and facing_cos for specific LEDs, computed from an
        already-solved T_world_ctrl rather than the pre-match extrapolated
        prediction (proj_hints from get_predicted_led_projections_per_camera)
        — that prediction is the prior fed *into* pose search, and drifts from
        the truth exactly when the controller is rotating/accelerating fast,
        which is the regime calibration data most needs to be accurate for.
        """
        tracker = self.trackers.get((ctrl_name, cam_id))
        if tracker is None:
            return {}
        camera     = self.cameras[cam_id]
        T_cam_ctrl = camera.T_world_cam.inverse().compose(T_world_ctrl)
        R_c = T_cam_ctrl.R.astype(np.float32)
        t_c = T_cam_ctrl.t.reshape(3).astype(np.float32)

        ids       = np.asarray(list(led_ids), dtype=int)
        positions = tracker.model.positions[ids]
        normals   = tracker.model.normals[ids]

        led_cam     = (R_c @ positions.T).T + t_c
        depths      = led_cam[:, 2]
        view_dir    = led_cam / (np.linalg.norm(led_cam, axis=1, keepdims=True) + 1e-8)
        normals_cam = (R_c @ normals.T).T
        facing_cos  = -(normals_cam * view_dir).sum(axis=1)

        return {int(lid): (float(d), float(fc)) for lid, d, fc in zip(ids, depths, facing_cos)}

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
        frame_ts_ns: int,
        radii_per_camera: Optional[Dict[int, np.ndarray]] = None,
        brightnesses_per_camera: Optional[Dict[int, np.ndarray]] = None,
        per_ctrl_observations: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
        per_ctrl_radii: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
        per_ctrl_brightnesses: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
        ctrl_name_filter: Optional[str] = None,
        allow_brute: bool = True,
        force_brute: bool = False,
    ) -> Dict[str, Optional[Dict]]:
        """
        Run tracking for every controller using a single primary camera.

        frame_ts_ns: the current frame's real capture timestamp (nanoseconds,
        parsed from the frame's filename) — required; used to extrapolate pose
        history against actual elapsed time rather than an assumed uniform step.

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
                frame_ts_ns,
                fixed_primary_cam=self._fixed_primary_cam,
                occluders_per_cam=_occluders_per_cam,
                self_cal=self._self_cal,
                pool=self._pool,
                allow_brute=allow_brute,
                force_brute=force_brute,
            )

        return results

    def _build_extrapolated_occluders(
        self, ctrl_names: List[str], frame_ts_ns: int,
    ) -> Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray, object]]]:
        """Return {ctrl_name: {cam_id: (R_occ_in_cam, t_occ_in_cam, geometry)}}
        — one controller's occlusion-relevant pose per camera, built from its
        own *extrapolated* (predicted) pose rather than a same-frame
        confirmed solution. This is what makes cross-controller occlusion
        usable from update_warm_batch: unlike ControllerTracker.update()'s
        sequential occluder-building (controller.py, in update()), which
        needs an earlier-processed controller's already-solved T_world_ctrl
        for THIS frame, extrapolation needs nothing from this frame at all —
        every controller's dict can be built independently, before any
        cheap-search task even runs, with no ordering dependency between
        controllers. Cost is a handful of cheap host-side numpy/Rodrigues
        ops (one _predict_pose + up to len(self.cameras) Transform
        compositions per controller) — no image or blob data, no pool
        round-trip.

        Gated on the existing cross_controller_occlusion config flag (same
        one the sequential path uses). Returns {} when disabled, when fewer
        than 2 controllers are given, or when a controller has no pose
        history yet to extrapolate from (its own entry is simply absent —
        callers already treat occluders_per_cam.get(...) returning None as
        "no occluder this frame", the same as the sequential path does for a
        controller that hasn't solved yet).
        """
        if not bool(self._matching_cfg.get('cross_controller_occlusion', False)) or len(ctrl_names) < 2:
            return {}

        _weight_decay = float(self._matching_cfg.get('pose_prediction_weight_decay', 0.7))
        occluders_by_ctrl: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray, object]]] = {}
        for ctrl_name in ctrl_names:
            _any_tracker = next(iter(self.ctrl_trackers[ctrl_name].trackers.values()), None)
            if _any_tracker is None:
                continue
            _pred = CameraTracker._predict_pose(
                _any_tracker.pose_history, frame_ts_ns,
                weight_decay=_weight_decay, vel_ema_rate=_any_tracker.vel_ema,
            )
            if _pred is None:
                continue
            _rv_pred, _tv_pred = _pred
            _R_pred, _ = cv2.Rodrigues(np.asarray(_rv_pred, np.float32).reshape(3, 1))
            _T_cam_ctrl_pred = Transform(
                _R_pred.astype(np.float32), np.asarray(_tv_pred, np.float32).reshape(3)
            )
            _T_world_ctrl_pred = _any_tracker.camera.T_world_cam.compose(_T_cam_ctrl_pred)
            _geom = _any_tracker._geometry

            _occ_per_cam: Dict[int, Tuple[np.ndarray, np.ndarray, object]] = {}
            for cid, cam in self.cameras.items():
                if cam.T_world_cam is None:
                    continue
                _T_cam_occ = cam.T_world_cam.inverse().compose(_T_world_ctrl_pred)
                _occ_per_cam[cid] = (
                    _T_cam_occ.R.astype(np.float32),
                    _T_cam_occ.t.astype(np.float32),
                    _geom,
                )
            occluders_by_ctrl[ctrl_name] = _occ_per_cam

        if occluders_by_ctrl:
            logger.debug(
                f"[warm-batch] extrapolated occluders built for: "
                f"{list(occluders_by_ctrl.keys())}"
            )
        return occluders_by_ctrl

    def update_warm_batch(
        self,
        ctrl_names: List[str],
        per_ctrl_observations: Dict[str, Dict[int, np.ndarray]],
        per_ctrl_radii: Optional[Dict[str, Dict[int, np.ndarray]]],
        per_ctrl_brightnesses: Optional[Dict[str, Dict[int, np.ndarray]]],
        frame_ts_ns: int,
    ) -> Dict[str, Optional[Dict]]:
        """Batched cheap-search-only update for controllers that are ALL warm
        this frame (every enabled controller has a prior pose). Submits every
        (ctrl_name, cam_id) cheap-search task across every controller in ONE
        pool round instead of the one-round-per-controller sequence update()
        runs when called once per controller with ctrl_name_filter — this is
        what actually lets two warm controllers' per-camera work overlap.
        Mirrors ControllerTracker.update()'s cheap pass exactly, except for
        the batching and the deliberate absence of blob_mask/claimed_blobs/
        reservations: two controllers matching the same blob is accepted
        here (each controller's own RANSAC/inlier scoring in its independent
        fit absorbs it) — see the warm-warm design notes. No brute-force
        fallback happens inside this path (out of scope for the warm-warm
        case). Cross-controller occlusion IS applied here, but — unlike
        ControllerTracker.update()'s sequential occluder-building, which
        requires another controller's already-confirmed same-frame result
        (structurally incompatible with this batched dispatch, since no
        controller has a same-frame result before the round completes) —
        each controller's occluder pose is its own *extrapolated* prediction
        (see _predict_pose), not a same-frame solve. See
        _build_extrapolated_occluders.

        Returns {ctrl_name: solution_or_None}. None means this controller's
        cheap pass didn't produce a fused solution this frame (e.g. every
        camera came up empty) — the caller must fall back to the existing
        sequential per-controller path (cold re-detect + brute) for that
        controller only, exactly as it would on any other cheap-search
        failure.
        """
        from src.parallel_search import run_cheap_search

        specs = []   # (ctrl_name, cid, tracker, obs, rad, brt)
        for ctrl_name in ctrl_names:
            obs_src = per_ctrl_observations.get(ctrl_name) or {}
            rad_src = (per_ctrl_radii or {}).get(ctrl_name) or {}
            brt_src = (per_ctrl_brightnesses or {}).get(ctrl_name) or {}
            for cid, tracker in self.ctrl_trackers[ctrl_name].trackers.items():
                obs = obs_src.get(cid)
                if obs is None or len(obs) == 0:
                    continue
                specs.append((ctrl_name, cid, tracker, obs, rad_src.get(cid), brt_src.get(cid)))

        if not specs:
            return {c: None for c in ctrl_names}

        occluders_by_ctrl = self._build_extrapolated_occluders(ctrl_names, frame_ts_ns)

        def _other_ctrl(name: str) -> Optional[str]:
            return next((c for c in ctrl_names if c != name), None)

        if self._pool is not None:
            futures = {}
            for ctrl_name, cid, tracker, obs, rad, brt in specs:
                prior = {
                    'prev_pose':       tracker.prev_pose,
                    'prev_prev_pose':  tracker.prev_prev_pose,
                    'pose_history':    tracker.pose_history,
                    'vel_ema':         tracker.vel_ema,
                    'prev_assignment': tracker.prev_assignment,
                }
                futures[(ctrl_name, cid)] = self._pool.submit(
                    run_cheap_search, (ctrl_name, cid), self._matching_cfg, prior,
                    obs, frame_ts_ns, rad, brt, None,
                    occluders_by_ctrl.get(_other_ctrl(ctrl_name)),
                )
            raw: Dict[Tuple[str, int], Tuple] = {}
            for key, fut in futures.items():
                solution, predicted_pose, norm_prev, norm_prev_prev = fut.result()
                _tracker = self.ctrl_trackers[key[0]].trackers[key[1]]
                _tracker.prev_pose      = norm_prev
                _tracker.prev_prev_pose = norm_prev_prev
                raw[key] = (solution, predicted_pose)
        else:
            raw = {
                (ctrl_name, cid): tracker.search_cheap(
                    obs, frame_ts_ns, blob_radii=rad, blob_brightnesses=brt,
                    occluders_per_cam=occluders_by_ctrl.get(_other_ctrl(ctrl_name)),
                )
                for ctrl_name, cid, tracker, obs, rad, brt in specs
            }

        cam_solutions_per_ctrl: Dict[str, list] = {c: [] for c in ctrl_names}
        for ctrl_name, cid, tracker, obs, rad, brt in specs:
            solution, predicted_pose = raw[(ctrl_name, cid)]
            solution = tracker.finalize_search(
                solution, predicted_pose, obs, blob_radii=rad,
                allow_expensive_fallback=False,
            )
            if solution is not None:
                cam_solutions_per_ctrl[ctrl_name].append(
                    {"cam_id": cid, "tracker": tracker, "solution": solution}
                )

        results: Dict[str, Optional[Dict]] = {}
        for ctrl_name in ctrl_names:
            cam_solutions = cam_solutions_per_ctrl[ctrl_name]
            if not cam_solutions:
                results[ctrl_name] = None
                continue
            obs_src = per_ctrl_observations.get(ctrl_name) or {}
            results[ctrl_name] = self.ctrl_trackers[ctrl_name]._fuse_and_finalize(
                cam_solutions, obs_src, claimed_blobs=None, frame_ts_ns=frame_ts_ns,
                fixed_primary_cam=self._fixed_primary_cam, self_cal=self._self_cal,
            )
        return results
