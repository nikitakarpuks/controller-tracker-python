import copy
import time
import cv2
import numpy as np
from collections import deque
from loguru import logger
from typing import List, Tuple, Optional, Dict, Set

from src._pnp import _project_points
from src._visibility import _visible_mask, _cross_occluded_mask
from src.camera import Camera
from src.debug_config import is_continuous_sequence
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
        brts_prox   = blob_brightnesses[_avail_idx] if blob_brightnesses is not None else None
        n_available = len(blobs_prox)
    else:
        blobs_prox  = blobs
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
            logger.bind(cat="matching_decisions").debug(f"[{pose_searcher._ctrl} | cam {pose_searcher._cam} | track] n_blobs={n_available} + prior → prior_constrained_match")
            solution = pose_searcher.constrained_search(
                blobs_prox, predicted_pose,
                prior_assignment=prev_assignment,
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

        # Consecutive frames (cold-start only) this camera has shown >= min_inliers
        # available blobs — see the cold brute-force confirm-streak gate in update().
        self._consecutive_good_blob_frames: int = 0

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
        # that's too far from the last known good pose. Only valid when frames
        # are a real temporal sequence — against a curated/isolated frame set
        # (assume_continuous_frames: false) there's no real "last frame" to be
        # near, so this check is skipped and the controller can be anywhere.
        if (solution is not None and _reacquiring and self.last_good_pose is not None
                and is_continuous_sequence()):
            rvec_lg, tvec_lg = self.last_good_pose
            if self._pose_jump_too_large(
                solution["rvec"], solution["tvec"],
                rvec_lg, tvec_lg,
                max_dist_m=0.5,
                max_angle_deg=60.0,
                **_jump_kw,
            ):
                logger.bind(cat="matching_decisions").debug(f"[{ctrl_name} | cam {cam_idx} | track] brute re-acquisition rejected: too far from last known good pose")
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
            logger.bind(cat="matching_decisions").debug(f"[{ctrl_name} | cam {cam_idx} | track] proximity err={solution['error']:.2f}px exceeds threshold, attempting brute recovery")
            brute = self.brute_match(blobs, pose_prior=self.prev_pose,
                                     other_cameras_blobs=other_cameras_blobs,
                                     blob_mask=blob_mask)
            if brute is not None:
                logger.bind(cat="matching_decisions").debug(f"[{ctrl_name} | cam {cam_idx} | track] brute recovery err={brute['error']:.2f}px")
                solution = brute

        if solution is not None and solution["error"] < _accept_err_px:
            logger.bind(cat="matching_decisions").debug(
                f"[{ctrl_name} | cam {cam_idx} | track] accepted — method={solution.get('method','?')}  "
                f"inliers={len(solution['assignment'])}  err={solution['error']:.2f}px"
            )
            # World-frame controller pose: T_world_ctrl = T_world_cam ∘ T_cam_ctrl
            R_ctrl, _ = cv2.Rodrigues(solution["rvec"])
            T_cam_ctrl = Transform(R_ctrl, solution["tvec"].reshape(3))
            solution["T_world_ctrl"] = self.T_world_cam.compose(T_cam_ctrl)
            return solution

        if solution is not None:
            logger.bind(cat="matching_decisions").debug(f"[{ctrl_name} | cam {cam_idx} | track] rejected — err={solution['error']:.2f}px exceeds {_accept_err_px:.1f}px threshold")
        else:
            logger.bind(cat="matching_decisions").debug(f"[{ctrl_name} | cam {cam_idx} | track] rejected — no solution found")
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
                    logger.bind(cat="matching_decisions").debug(f"[{ctrl_name} | cam {cam_idx} | track] proximity → None, running brute fallback")
                    solution = self.brute_match(blobs, pose_prior=predicted_pose,
                                                other_cameras_blobs=other_cameras_blobs,
                                                blob_mask=blob_mask,
                                                occluders_per_cam=occluders_per_cam)
            else:
                if n_available >= 4:
                    logger.bind(cat="matching_decisions").debug(f"[{ctrl_name} | cam {cam_idx} | track] no prev_pose → cold-start brute")
                    solution = self.brute_match(blobs, other_cameras_blobs=other_cameras_blobs,
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

    def _mark_all_lost(self) -> None:
        """Record a failed frame on every camera-tracker and, once the
        configured grace period is exhausted, clear prev_pose/pose_history so
        the controller stops warm-starting off stale history. A single retry
        (the default grace of 1) still gets to warm-start next frame per the
        tracking_lost_last_frame contract in CameraTracker.finalize_search;
        beyond that, surviving history was letting a controller with no real
        track (or one truly gone cold) look "warm" forever — see
        get_predicted_led_projections_per_camera / ctrl_has_prior in main.py,
        which key off pose_history alone and don't otherwise expire it.
        Clearing here also starves _build_extrapolated_occluders (no
        extrapolated pose survives), so a genuinely cold controller no longer
        casts a phantom cross-controller occlusion mask.
        """
        _grace = int(self._matching_cfg.get('tracking_lost_grace_frames', 1))
        for _t in self.trackers.values():
            _t.tracking_lost_last_frame = True
            _t.consecutive_failures += 1
            if _t.consecutive_failures > _grace:
                _t.prev_pose      = None
                _t.prev_prev_pose = None
                _t.vel_ema        = None
                _t.pose_history.clear()

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
            self._mark_all_lost()
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
            _confirm_frames = int(self._matching_cfg.get('cold_brute_force_confirm_frames', 3))
            for cid, (av_blobs, av_radii, av_brts, av_orig) in avail.items():
                tracker = self.trackers[cid]
                # force_brute=True covers TWO distinct cases (main.py:585 and
                # main.py:633): a truly cold controller (no prior anywhere), and
                # a controller that was warm THIS frame whose cheap search just
                # failed, falling back to an immediate cold re-detect + brute
                # attempt. Only the former should pay the persistence-streak
                # cost below — a just-lost controller's prior is still trusted
                # (that's the whole point of tracking_lost_grace_frames) and
                # deserves an immediate recovery attempt, not an added
                # multi-frame confirmation delay. _mark_all_lost only clears
                # prev_pose once the grace period is exhausted, so
                # prev_pose is None is exactly "truly cold" here; prev_pose is
                # still set means "recently lost, within grace" — skip the gate.
                if force_brute and tracker.prev_pose is None:
                    # Persistence gate: require _confirm_frames CONSECUTIVE
                    # frames of >= min_inliers blobs before paying for a full
                    # P3P/gate enumeration, since real reacquired LEDs persist
                    # frame-to-frame while noise blobs don't (e.g. blob counts
                    # 4,5,1 never trigger; 4,5,6 triggers on the 3rd frame).
                    # Reset on success lives in _fuse_and_finalize alongside
                    # consecutive_failures.
                    if len(av_blobs) >= tracker._pose_searcher._c_brute_min_inliers:
                        tracker._consecutive_good_blob_frames += 1
                    else:
                        tracker._consecutive_good_blob_frames = 0
                    if tracker._consecutive_good_blob_frames < _confirm_frames:
                        logger.bind(cat="matching_decisions").debug(
                            f"[{self.ctrl_name} | cam {cid}] cold brute-force gated: "
                            f"streak={tracker._consecutive_good_blob_frames}/{_confirm_frames} "
                            f"(n_blobs={len(av_blobs)})"
                        )
                        continue
                if len(av_blobs) == 0:
                    continue
                obs_full = obs_src[cid]
                mask = np.zeros(len(obs_full), dtype=bool)
                mask[av_orig] = True
                states[cid] = tracker._pose_searcher.new_brute_state(
                    obs_full, pose_prior=predicted_pose_by_cid.get(cid),
                    other_cameras_blobs=None,
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
        logger.bind(cat="timings").debug(
            f"[{self.ctrl_name}] update timing: cheap={t_cheap*1000:.1f}ms  "
            f"brute(new_state={t_new_state*1000:.1f}ms "
            f"tier_rounds={t_tier_rounds*1000:.1f}ms[{n_rounds} rounds] "
            f"finalize={t_finalize*1000:.1f}ms){_pool_rt_str}"
        )

        if not cam_solutions:
            self._mark_all_lost()
            return None

        return self._fuse_and_finalize(
            cam_solutions, obs_src, claimed_blobs, frame_ts_ns,
            fixed_primary_cam=fixed_primary_cam, self_cal=self_cal,
        )

    def _compute_fused_solution(
        self,
        cam_solutions: List[dict],
        obs_src: Dict[int, np.ndarray],
        fixed_primary_cam: Optional[int] = None,
    ) -> Dict:
        """Fuse every camera's independent solution into one controller pose
        and build the resulting solution dict — pure, no tracker-state
        mutation, no claimed_blobs registration, no self-cal feed. Split out
        of _fuse_and_finalize so a caller with several simultaneously-solved
        candidates for DIFFERENT controllers (TrackingSystem.update_cold_batch)
        can compare them — by solution["error"] and matched-pair counts — and
        resolve cross-controller conflicts before committing any one of them
        via _commit_fused_solution. cam_solutions must be non-empty.
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

        logger.bind(cat="pose_fusion").debug(
            f"[{self.ctrl_name}] Joint fusion: {len(cam_solutions)} solved  "
            f"err={fused_err:.2f}px  lm={_t_fuse*1000:.1f}ms  "
            f"importance=[{_importance_str}]"
        )

        solution = dict(anchor_solution)
        solution["primary_cam"]  = primary_cam_id
        # Preserved for the self-cal feed below: the primary camera's own
        # independent PnP solve, untainted by any other (possibly still
        # uncalibrated) camera's contribution to the fused pose — see
        # _commit_fused_solution's self-cal block.
        solution["primary_T_world_ctrl"] = anchor_solution["T_world_ctrl"]
        solution["primary_error"]        = anchor_solution["error"]
        solution["T_world_ctrl"] = T_world_ctrl
        solution["error"]        = fused_err
        # Normalised per-camera fusion weight (sums to 1 across every solved
        # camera) — the same values behind the "importance=[...]" debug line
        # above, exposed here for consumers (e.g. visualization) that want to
        # reflect actual per-camera contribution rather than a primary/aux label.
        solution["camera_importance"] = {cid: w / _w_total for cid, w in _raw_w.items()}
        _other_assignments = {
            cs["cam_id"]: cs["solution"]["assignment"]
            for cs in cam_solutions if cs["cam_id"] != primary_cam_id
        }
        solution["aux_assignments"] = _other_assignments
        solution["aux_cameras"] = [(cid, len(pairs)) for cid, pairs in _other_assignments.items()]

        return solution

    def _commit_fused_solution(
        self,
        solution: Dict,
        cam_solutions: List[dict],
        obs_src: Dict[int, np.ndarray],
        claimed_blobs: Optional[Dict[int, Set[int]]],
        frame_ts_ns: int,
        self_cal=None,
    ) -> None:
        """Side effects for a solution already chosen as the accepted result
        for this controller this frame: designated-primary bookkeeping,
        claimed-blob registration, self-cal feed, and propagation of the
        fused pose into every camera tracker's state. Call only once a
        solution has actually been accepted — for the plain single-candidate
        case (ControllerTracker.update(), TrackingSystem.update_warm_batch)
        that's unconditional (see _fuse_and_finalize below); for
        TrackingSystem.update_cold_batch's multi-candidate case, only after
        cross-controller conflict resolution has picked a winner.

        claimed_blobs=None skips claim registration entirely — used by the
        warm-warm and cold-cold batched paths, neither of which tracks
        cross-controller blob claims this way (warm-warm: each controller's
        own RANSAC/inlier scoring absorbs a shared blob; cold-cold: conflicts
        are resolved explicitly by TrackingSystem._resolve_cold_conflicts
        before this is ever called).
        """
        T_world_ctrl       = solution["T_world_ctrl"]
        primary_cam_id     = solution["primary_cam"]
        anchor_assignment  = solution["assignment"]
        _other_assignments = solution["aux_assignments"]

        # Reporting/self-cal anchor tracking (per-camera importance above is now the
        # reported signal; kept only for get_designated_primary_cameras()).
        self._designated_primary = primary_cam_id

        # ── Register claimed blobs (every camera that actually solved) ──────────
        if claimed_blobs is not None:
            for cs in cam_solutions:
                for b, _ in cs["solution"]["assignment"]:
                    claimed_blobs.setdefault(cs["cam_id"], set()).add(b)

        # ── Self-calibration feed ────────────────────────────────────────────────
        # Uses the primary camera's own independent PnP solve (primary_T_world_ctrl/
        # primary_error, set in _compute_fused_solution) as the 3D reference, NOT
        # the fused T_world_ctrl above — the fused pose is a confidence-weighted
        # blend across every solved camera, including the very aux cameras being
        # calibrated here. Feeding that blend back in as "ground truth" would be
        # circular: an aux camera's still-wrong intrinsics would leak into the
        # reference used to correct it. SelfCalibrator's own docstring assumes an
        # aux-camera-untainted primary solve; this is what actually provides one.
        if (self_cal is not None
                and primary_cam_id == self_cal.primary_camera.camera_idx):
            _primary_T_world_ctrl = solution["primary_T_world_ctrl"]
            _primary_err          = solution["primary_error"]
            _T_cam_ctrl_anchor = self.cameras[primary_cam_id].T_world_cam.inverse().compose(_primary_T_world_ctrl)
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
                    primary_error=_primary_err,
                    primary_inliers=len(anchor_assignment),
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
                anchor_assignment if _cid == primary_cam_id
                else _other_assignments.get(_cid)
            )
            if _own_asgn is not None:
                _tracker.prev_assignment      = _own_asgn
                _tracker.last_good_assignment = _own_asgn
            elif _tracker.prev_assignment is None:
                _tracker.last_good_assignment = None
            _tracker.consecutive_failures = 0
            _tracker._consecutive_good_blob_frames = 0
            _tracker.tracking_lost_last_frame = False

    def _fuse_and_finalize(
        self,
        cam_solutions: List[dict],
        obs_src: Dict[int, np.ndarray],
        claimed_blobs: Optional[Dict[int, Set[int]]],
        frame_ts_ns: int,
        fixed_primary_cam: Optional[int] = None,
        self_cal=None,
    ) -> Dict:
        """Fuse then immediately commit — thin wrapper preserving the
        original combined behavior for the tier-round recovery path
        (update(), above) and the batched warm-warm path
        (TrackingSystem.update_warm_batch()). See _compute_fused_solution /
        _commit_fused_solution for the split and why it exists."""
        solution = self._compute_fused_solution(
            cam_solutions, obs_src, fixed_primary_cam=fixed_primary_cam,
        )
        self._commit_fused_solution(
            solution, cam_solutions, obs_src, claimed_blobs, frame_ts_ns, self_cal=self_cal,
        )
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

        # Fixed primary camera exists only to serve self-calibration (it pins the
        # 3D reference every frame feeds SelfCalibrator.add_frame from — see
        # ControllerTracker._commit_fused_solution's self-cal block); with no
        # self-calibration running, there's no reason to override the normal
        # per-frame auto-select (most inlier pairs wins).
        self._fixed_primary_cam: Optional[int] = _sc_primary_idx

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
                register_blob_detector_spec(cam.camera_idx, blob_detection_cfg,
                                             cam.width, cam.height)

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
    ) -> Tuple[Dict[int, Dict[str, Optional[np.ndarray]]], Dict[int, Dict[str, float]],
               Dict[int, Dict[str, float]], Dict[int, Dict[str, bool]]]:
        """Return (proj_hints, vel_hints, radius_hints, search_eligible).

        frame_ts_ns: the current frame's real capture timestamp (nanoseconds,
        parsed from the frame's filename) — predictions are extrapolated to this
        exact elapsed time, not an assumed uniform frame step.

        proj_hints: {cam_id: {ctrl_name: Nx5 array or None}}
          Each row: [proj_x, proj_y, depth_m, facing_cos, led_id]
          for each LED visible from the predicted pose.
          None when no prior pose exists for this (ctrl, cam) pair, i.e. zero
          geometrically visible LEDs. A camera that's visible but lost the
          warm-cam-cap ranking below still has a real (non-None) entry here —
          see search_eligible.

        vel_hints, radius_hints: {cam_id: {ctrl_name: v_px / search_radius_px}}
          Estimated pixel displacement / effective search radius, for whatever
          camera main.py actually searches. 0.0 / base radius when no
          prediction is available. Retained (not reset) for cap-losing
          cameras too, since _build_blackout_images uses radius_hints to size
          the region it blacks out for them — see search_eligible.

        search_eligible: {cam_id: {ctrl_name: bool}}
          Whether main.py should actually run blob detection / pose search on
          this (cam, ctrl) pair this frame. False whenever proj_hints is None
          (nothing to search) OR the warm-cam-cap ranking demoted this camera
          in favor of better candidates — in the latter case proj_hints/
          vel_hints/radius_hints are still the real, non-reset values, kept
          specifically so a cold-starting OTHER controller's
          _build_blackout_images can still mask this controller's expected
          LED neighborhoods there, even though this controller itself won't
          search that camera this frame (warm-cold cross-controller
          contamination guard; irrelevant in the warm-warm case since
          _build_blackout_images is only invoked for a controller with no
          prior at all — see its call site in main.py).
        """
        ctrl_names   = sorted({ctrl for ctrl, _ in self.trackers})
        _facing_deg  = float(self._matching_cfg.get('led_facing_angle_deg', 86.0))
        # Any geometric visibility at all (>= 1 LED) is candidacy, not a search
        # guarantee — a controller can end up warm-tracked in 3-4 cameras when 2
        # would do. Rather than pre-filtering candidacy on a per-camera-independent
        # LED-count estimate (which has been observed to wrongly exclude a camera
        # whose real image would actually be the best one to search — see the cap
        # comment below), every camera with >=1 visible LED is a ranking candidate:
        # cameras viewing the controller too edge-on are excluded outright
        # (warm_cam_cap_min_facing_deg), survivors ranked by mean LED distance,
        # top max_warm_search_cameras kept.
        _max_warm_cams = int(self._matching_cfg.get('max_warm_search_cameras', 0)) or None
        _min_facing_cos_gate = float(np.cos(np.radians(
            float(self._matching_cfg.get('warm_cam_cap_min_facing_deg', 80.0)))))
        result:          Dict[int, Dict[str, Optional[np.ndarray]]] = {}
        vel_hints:       Dict[int, Dict[str, float]]                = {}
        radius_hints:    Dict[int, Dict[str, float]]                = {}
        search_eligible: Dict[int, Dict[str, bool]]                 = {}
        # (ctrl_name, cam_id) -> mean perpendicular distance (m) from the
        # controller's visible LEDs to that camera's optical axis (NOT Euclidean
        # range to the camera center — see the lateral_dist comment below) — used
        # below (together with _center_facing) to rank cameras for the warm-search
        # cap.
        _center_dist:   Dict[Tuple[str, int], float]              = {}
        # (ctrl_name, cam_id) -> mean facing_cos over the controller's visible
        # LEDs (how squarely they face this camera, averaged — NOT summed, so
        # LED count doesn't leak into what's meant to be a pure angle score) —
        # used below purely as a gate (grazing cameras excluded outright), not
        # as the ranking key: a camera can score well here on a couple of
        # borderline LEDs while still lacking enough of them to ever actually
        # match — see the cap comment below.
        _center_facing: Dict[Tuple[str, int], float]              = {}

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
                if len(vis_ids) == 0:
                    # Nothing to project or rank. The LED-count floor that used to
                    # live here has moved entirely into the warm-cam-cap ranking
                    # pass below, which now treats every camera with >=1 visible
                    # LED as a candidate (see comment above).
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
                led_ranges  = np.linalg.norm(led_cam, axis=1)                 # (M,) true 3D distance to camera center
                view_dir    = led_cam / (led_ranges.reshape(-1, 1) + 1e-8)
                normals_cam = (R_pred @ normals[vis_ids].T).T                 # (M, 3)
                facing_cos  = -(normals_cam * view_dir).sum(axis=1)           # positive = faces cam
                # Perpendicular distance from each LED to this camera's optical axis
                # (the line through its center along local +z): for a point (x,y,z)
                # in camera frame that's sqrt(x^2+y^2), i.e. range with the
                # along-axis (depth) component removed. depths > 1e-6 is already
                # guaranteed per-LED by _visible_mask, so "behind the projection
                # surface" is already excluded before we get here — no separate
                # reject step needed.
                lateral_dist = np.linalg.norm(led_cam[:, :2], axis=1)         # (M,)

                proj_per_ctrl[ctrl_name] = np.column_stack([
                    proj_pts.astype(np.float32),
                    depths.astype(np.float32).reshape(-1, 1),
                    facing_cos.astype(np.float32).reshape(-1, 1),
                    vis_ids.astype(np.float32).reshape(-1, 1),
                ])  # (M, 5): proj_x, proj_y, depth_m, facing_cos, led_id

                # Mean perpendicular (off-axis) distance from the visible LEDs to
                # this camera's optical axis — used as the ranking key below,
                # replacing mean Euclidean range. A camera viewing the controller
                # near edge-on has almost all its range in the lateral component
                # (depth ~0), so this collapses toward the full range for such a
                # camera, while a camera viewing it near head-on has most of its
                # range in depth, so this collapses toward ~0 — a much starker
                # separation than raw Euclidean distance gave between e.g. cam0/1
                # (near head-on) and cam2/3 (near edge-on, ~180°-rotated) in
                # practice. Mean facing_cos is kept as the gate below, not used
                # for ranking. Both are means, not sums, so LED count doesn't leak
                # into either score.
                _center_dist[(ctrl_name, cam_id)]   = float(np.mean(lateral_dist))
                _center_facing[(ctrl_name, cam_id)] = float(np.mean(facing_cos))
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

            result[cam_id]          = proj_per_ctrl
            vel_hints[cam_id]       = vel_per_ctrl
            radius_hints[cam_id]    = radius_per_ctrl
            search_eligible[cam_id] = {
                ctrl_name: proj_per_ctrl[ctrl_name] is not None for ctrl_name in ctrl_names
            }

        # ── Cap warm-search cameras per controller ──────────────────────────────
        # Two-stage selection among cameras theoretically visible for this
        # controller: (1) GATE — drop any camera whose mean facing_cos falls
        # below warm_cam_cap_min_facing_deg's cosine outright, regardless of
        # distance (a grazing view gives worse blobs — dimmer, more elongated,
        # noisier centroid — no matter how close the camera is); (2) RANK the
        # survivors by mean perpendicular (off-axis) distance from the visible
        # LEDs to each camera's optical axis, closest wins, keep the top
        # max_warm_search_cameras (see lateral_dist comment above). Three
        # single-metric schemes were tried and rejected first: an original
        # visible-LED-count / facing_cos-sum score let LED count leak into the
        # ranking; pure mean-facing_cos ranking let a camera with a great angle
        # on just a couple of borderline LEDs outrank one with plenty of LEDs
        # at a merely-good angle; mean Euclidean range to the camera center
        # worked much better but still occasionally lost on razor-thin margins
        # (e.g. 0.280m vs 0.286m) to a camera that structurally never produces
        # a match — a ~180°-rotated camera's near-edge-on view keeps its
        # Euclidean range deceptively similar to a well-facing camera's, since
        # range doesn't distinguish "close and off to the side" from "close and
        # dead ahead". Off-axis distance separates these sharply (observed
        # 0.048m/0.188m for two good cameras vs 0.278m for the bad one, on the
        # exact frame the Euclidean version mis-ranked) because it collapses
        # toward the full range for an edge-on view and toward ~0 for a
        # head-on one. Gating on facing_cos rather than folding it into a
        # blended score keeps each metric doing the job it's actually good at.
        # If every candidate fails the gate (all views are marginal), the
        # gate is skipped rather than demoting all of them, so far-fetched
        # geometry never leaves a controller with zero search cameras.
        # Demoted cameras are marked search_eligible=False — NOT reset to None —
        # so main.py's Phase 1 skips real blob detection/search there (via
        # search_eligible, not proj_hints), exactly like the "too few visible
        # LEDs" branch above, EXCEPT proj_hints/vel_hints/radius_hints are left
        # as the real, computed values rather than nulled out. That's
        # deliberate, not an oversight: this controller's own tracker won't
        # search this camera this frame, but if some OTHER controller has no
        # prior at all this frame and is about to run a full-image cold
        # brute-force search on this same camera, _build_blackout_images (in
        # main.py) needs this controller's real predicted LED positions/radii
        # here to black them out first — otherwise a demoted-but-still-real
        # bright blob sits fully exposed for the cold controller's search to
        # accidentally match as one of its own LEDs. That blackout is scoped
        # tightly (just the predicted-LED neighborhoods, not the whole frame)
        # and only matters for one frame at 30fps, so the cold controller
        # still has the rest of the image plus every subsequent frame to
        # reacquire — an acceptable trade against a false cross-controller
        # match, which is not recoverable the same way. _build_blackout_images
        # itself needs no changes for this: it already keys purely off
        # proj_hints being non-None, and only runs at all for a controller
        # with no prior anywhere this frame (the warm-cold case specifically —
        # irrelevant, and never invoked, in the warm-warm case). A cold
        # controller has no prediction on any camera, so none of this affects
        # its own brute-force reacquisition (never gated by proj_hints or
        # search_eligible at all).
        #
        # Candidacy here (membership in _all_cids, i.e. result[cid][ctrl_name] is
        # not None) is gated ONLY on >=1 geometrically visible LED — a camera
        # predicting too few LEDs to ever reach min_inliers downstream is no
        # longer pre-excluded before this ranking runs. If such a camera still
        # wins the cap (closest, passes the facing gate), it's kept as a real
        # search candidate: main.py runs real blob detection and a real
        # proximity/brute-force search on it, which is mathematically guaranteed
        # to return None via pose_search.py's own _c_prox_min_inliers /
        # _c_brute_min_inliers checks if the LED count is genuinely too low — so
        # it can never inject a bad pose into fusion, and it contributes zero
        # cam_solutions weight there, exactly like the "no candidate" case. What
        # it gains over being pre-excluded: real detection, a real (if losing)
        # search, and real visualization output for a camera whose actual image
        # often turns out better than this geometric estimate predicts.
        if _max_warm_cams is not None:
            for ctrl_name in ctrl_names:
                _all_cids = [cid for cid in result if result[cid][ctrl_name] is not None]
                if _all_cids:
                    logger.bind(cat="matching_decisions").debug(
                        f"[{ctrl_name}] warm-cam-cap candidates: " + "  ".join(
                            f"cam{cid}(n={len(result[cid][ctrl_name])},"
                            f"dist={_center_dist[(ctrl_name, cid)]:.3f}m,"
                            f"facing={_center_facing[(ctrl_name, cid)]:.3f})"
                            for cid in sorted(_all_cids, key=lambda c: _center_dist[(ctrl_name, c)])
                        )
                    )
                if len(_all_cids) <= _max_warm_cams:
                    continue
                _gated_cids = [
                    cid for cid in _all_cids
                    if _center_facing[(ctrl_name, cid)] >= _min_facing_cos_gate
                ] or _all_cids  # safety net: don't starve the controller if every view is marginal
                _kept_cids = {
                    cid for _, cid in sorted((_center_dist[(ctrl_name, cid)], cid)
                                              for cid in _gated_cids)[:_max_warm_cams]
                }
                for cid in _all_cids:
                    if cid in _kept_cids:
                        continue
                    logger.bind(cat="matching_decisions").debug(f"[{ctrl_name}] warm-cam-cap demoting cam{cid}")
                    search_eligible[cid][ctrl_name] = False

        return result, vel_hints, radius_hints, search_eligible

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

        def _has_prior(name: str) -> bool:
            return any(
                t.prev_pose is not None
                for (cname, _), t in self.trackers.items()
                if cname == name
            )

        def _order_key(name: str):
            return (0 if _has_prior(name) else 1, name)

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

        def _filter_cam(cam_id: int, obs_map: dict, rad_map: dict, brt_map: dict):
            """Return (filtered_blobs, filtered_radii, filtered_brts, orig_indices)
            excluding claimed blobs. Returns (None,None,None,[]) if empty."""
            obs = obs_map.get(cam_id)
            if obs is None or len(obs) == 0:
                return None, None, None, []
            excluded = claimed_blobs.get(cam_id, set())
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

            avail: Dict[int, tuple] = {}
            for cam_id in self.ctrl_trackers[ctrl_name].trackers:
                fb, fr, fbt, keep = _filter_cam(cam_id, _obs_src, _rad_src, _brt_src)
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
            logger.bind(cat="occlusion").debug(
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

    def update_cold_batch(
        self,
        ctrl_names: List[str],
        per_ctrl_observations: Dict[str, Dict[int, np.ndarray]],
        per_ctrl_radii: Optional[Dict[str, Dict[int, np.ndarray]]],
        per_ctrl_brightnesses: Optional[Dict[str, Dict[int, np.ndarray]]],
        frame_ts_ns: int,
        committed_solutions: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Optional[Dict]]:
        """Batched brute-force-only update for controllers that are ALL truly
        cold this frame (every camera tracker's prev_pose is None — callers
        must only pass controllers satisfying that; main.py guarantees it by
        routing only ctrl_has_prior[name] is False controllers here). Submits
        every (ctrl_name, cam_id) tier-round brute-force task across every
        controller in ONE shared set of pool rounds, instead of the
        one-controller-at-a-time sequence main.py's per-controller
        _update_ctrl(force_brute=True) loop runs today — this is what lets
        two simultaneously-cold controllers' brute recovery overlap. Mirrors
        ControllerTracker.update()'s tier-round brute path (see there), with
        two structural differences:

          - The tier-widening stop condition ("any camera hit strong_found
            this round -> stop widening") is evaluated per (ctrl_name, cid)
            key via `not st.strong_found`, not per-controller-with-an-early-
            break -- so one controller reaching strong_found on an early tier
            does not truncate another controller's later tiers, and vice
            versa. Both controllers' cameras simply share the same round of
            pool.submit calls.

          - Since no controller here has any prior pose or pose_history (by
            definition -- see above), there is no claimed_blobs exclusion, no
            reservation biasing, and no occluders_per_cam during the search
            itself (same reasoning as _build_extrapolated_occluders: occlusion
            needs a pose to project from, and none exists yet for any of
            these controllers). Every candidate is computed independently and
            only reconciled with the others AFTER all of them finish solving
            -- see _resolve_cold_conflicts. Two cold controllers matching the
            same blob, or landing in a mutually impossible (occluding)
            configuration, are real possibilities here in a way they aren't
            for update_warm_batch's warm priors, precisely because nothing
            here excludes the other controller's candidate blobs/pose during
            the search.

        committed_solutions: {ctrl_name: solution} for controllers already
        committed to tracker state THIS frame before update_cold_batch was
        called -- i.e. every warm-processed controller (get_ctrl_processing_
        order guarantees these always precede the cold suffix, so "already
        committed" and "warm this frame" coincide in practice). These are
        never returned or re-solved here; they're folded into
        _resolve_cold_conflicts purely as FIXED candidates a cold candidate
        can lose to but never beat -- closing the gap where a cold
        controller's brute search could otherwise land on a pose that
        shares a blob with, or geometrically occludes, an already-tracked
        warm controller with nothing to catch it (unlike warm-warm, which
        gets occlusion-awareness from _build_extrapolated_occluders before
        the search even runs, and unlike cold-cold, which this method
        already reconciles post-hoc). Omit or pass {} for no such check.

        Returns {ctrl_name: solution_or_None}. None means either this
        controller's brute search didn't produce a solution this frame (no
        camera cleared the K-frame persistence gate, or none solved), or it
        did solve but lost a same-frame conflict -- against another cold
        controller's better-fit candidate, or against an already-committed
        warm controller it can never outrank -- both cases are routed
        through _mark_all_lost, identically to any other failed brute
        search, so _consecutive_good_blob_frames is left untouched either
        way (see the conflict-loser branch below for why that matters).
        """
        from src.parallel_search import run_brute_tier

        _confirm_frames = int(self._matching_cfg.get('cold_brute_force_confirm_frames', 3))
        states: Dict[Tuple[str, int], object] = {}
        for ctrl_name in ctrl_names:
            obs_src = per_ctrl_observations.get(ctrl_name) or {}
            for cid, tracker in self.ctrl_trackers[ctrl_name].trackers.items():
                obs_full = obs_src.get(cid)
                if obs_full is None or len(obs_full) == 0:
                    continue
                # Persistence gate, verbatim logic from
                # ControllerTracker.update() (controller.py:930-948, this
                # file) -- unconditional here (not "force_brute and
                # prev_pose is None") because every controller passed to
                # update_cold_batch is already known cold by the caller's
                # contract, so the gate always applies.
                if len(obs_full) >= tracker._pose_searcher._c_brute_min_inliers:
                    tracker._consecutive_good_blob_frames += 1
                else:
                    tracker._consecutive_good_blob_frames = 0
                if tracker._consecutive_good_blob_frames < _confirm_frames:
                    logger.bind(cat="matching_decisions").debug(
                        f"[{ctrl_name} | cam {cid}] cold brute-force gated (batch): "
                        f"streak={tracker._consecutive_good_blob_frames}/{_confirm_frames} "
                        f"(n_blobs={len(obs_full)})"
                    )
                    continue
                mask = np.ones(len(obs_full), dtype=bool)
                states[(ctrl_name, cid)] = tracker._pose_searcher.new_brute_state(
                    obs_full, pose_prior=None, other_cameras_blobs=None,
                    blob_mask=mask, occluders_per_cam=None,
                )

        if not states:
            return {c: None for c in ctrl_names}

        _any_ps = next(
            (self.ctrl_trackers[k[0]].trackers[k[1]]._pose_searcher
             for k, st in states.items() if st is not None),
            None,
        )
        max_tiers = len(_any_ps._c_brute_depth_tiers) if _any_ps is not None else 0

        if self._pool is not None:
            for tier_idx in range(max_tiers):
                remaining = [k for k, st in states.items() if st is not None and not st.strong_found]
                if not remaining:
                    break
                logger.bind(cat="batch_orchestration").debug(f"[cold-batch] tier {tier_idx}: submitting {sorted(remaining)}")
                futures = {k: self._pool.submit(run_brute_tier, k, states[k], tier_idx) for k in remaining}
                for k, fut in futures.items():
                    states[k] = fut.result()
        else:
            for tier_idx in range(max_tiers):
                remaining = [k for k, st in states.items() if st is not None and not st.strong_found]
                if not remaining:
                    break
                for ctrl_name, cid in remaining:
                    self.ctrl_trackers[ctrl_name].trackers[cid]._pose_searcher.brute_search_tier(
                        states[(ctrl_name, cid)], tier_idx)

        cam_solutions_per_ctrl: Dict[str, list] = {c: [] for c in ctrl_names}
        for (ctrl_name, cid), st in states.items():
            if st is None:
                continue
            tracker = self.ctrl_trackers[ctrl_name].trackers[cid]
            sol = tracker._pose_searcher.finalize_brute_state(st)
            if sol is None:
                continue
            obs_full = (per_ctrl_observations.get(ctrl_name) or {})[cid]
            rad_full = (per_ctrl_radii or {}).get(ctrl_name, {}).get(cid)
            mask = np.ones(len(obs_full), dtype=bool)
            sol = tracker.finalize_search(
                sol, None, obs_full, blob_radii=rad_full, other_cameras_blobs=None,
                blob_mask=mask, occluders_per_cam=None, allow_expensive_fallback=True,
            )
            if sol is not None:
                cam_solutions_per_ctrl[ctrl_name].append(
                    {"cam_id": cid, "tracker": tracker, "solution": sol}
                )

        candidates: Dict[str, Dict] = {}
        cam_solutions_of: Dict[str, list] = {}
        for ctrl_name in ctrl_names:
            cam_solutions = cam_solutions_per_ctrl[ctrl_name]
            if not cam_solutions:
                self.ctrl_trackers[ctrl_name]._mark_all_lost()
                continue
            obs_src = per_ctrl_observations.get(ctrl_name) or {}
            candidates[ctrl_name] = self.ctrl_trackers[ctrl_name]._compute_fused_solution(
                cam_solutions, obs_src, fixed_primary_cam=self._fixed_primary_cam,
            )
            cam_solutions_of[ctrl_name] = cam_solutions

        if not candidates:
            return {c: None for c in ctrl_names}

        _committed = committed_solutions or {}
        _all_for_conflict_check = {**candidates, **_committed}
        losers = self._resolve_cold_conflicts(
            _all_for_conflict_check, fixed_names=set(_committed.keys()),
        )

        results: Dict[str, Optional[Dict]] = {}
        for ctrl_name, solution in candidates.items():
            if ctrl_name in losers:
                # Same bookkeeping as any other failed brute search
                # (_mark_all_lost, controller.py:767-789) -- deliberately NOT
                # a fresh reset of _consecutive_good_blob_frames. This
                # controller already cleared the K-frame gate above, before
                # any brute enumeration even ran, so it does not need to
                # re-accumulate _confirm_frames consecutive frames before
                # retrying next frame -- only _mark_all_lost's own fields
                # (consecutive_failures, tracking_lost_last_frame) change.
                _n_pairs = len(solution.get('assignment') or []) + sum(
                    len(v) for v in (solution.get('aux_assignments') or {}).values()
                )
                logger.bind(cat="occlusion").info(
                    f"[cold-batch] conflict: dropping {ctrl_name} "
                    f"(err={solution['error']:.2f}px, n={_n_pairs}, "
                    f"lost to a better inlier-discounted candidate this frame)"
                )
                self.ctrl_trackers[ctrl_name]._mark_all_lost()
                results[ctrl_name] = None
                continue
            obs_src = per_ctrl_observations.get(ctrl_name) or {}
            self.ctrl_trackers[ctrl_name]._commit_fused_solution(
                solution, cam_solutions_of[ctrl_name], obs_src,
                claimed_blobs=None, frame_ts_ns=frame_ts_ns, self_cal=self._self_cal,
            )
            results[ctrl_name] = solution

        for ctrl_name in ctrl_names:
            results.setdefault(ctrl_name, None)
        return results

    def _resolve_cold_conflicts(
        self, candidates: Dict[str, Dict], fixed_names: Optional[Set[str]] = None,
    ) -> Set[str]:
        """Detect and greedily resolve conflicts among simultaneously-solved
        cold-start candidates (update_cold_batch, above) BEFORE any of them
        is committed to tracker state.

        fixed_names: candidates (typically already-committed warm controllers
        passed in via update_cold_batch's committed_solutions) that must
        never appear in the returned loser set -- their tracker state is
        already committed and un-committing isn't supported, so they always
        win any conflict they're part of. Processed before the normal greedy
        pass below (see the loop that drains fixed_names first). Omit for the
        pure cold-cold case -- behavior is then identical to before this
        parameter existed.

        Two candidates conflict if EITHER:
          - shared blob: the same blob index, in the same camera, appears in
            both candidates' matched pairs (primary `assignment` or
            `aux_assignments`).
          - cross-occlusion: one candidate's SOLVED pose (both here -- unlike
            _build_extrapolated_occluders, which necessarily uses a
            *predicted* pose since neither warm-warm controller has solved
            yet when it runs) should, per _cross_occluded_mask, have blocked
            one of the other candidate's matched LEDs from view in a camera
            they both used. Checked in both directions.

        Both conflict types are treated identically -- no special-casing, no
        cheap-repair exception: any conflict of either kind drops the loser
        outright. Winner selection is greedy: repeatedly take the remaining
        candidate with the lowest inlier-discounted error (see _score below;
        ties broken by more total matched pairs across assignment +
        aux_assignments), keep it, and drop every candidate directly in
        conflict with it; repeat among what's left. This correctly handles
        conflict components of 3+ controllers, not just pairs -- a controller
        connected to the graph only through an already-dropped neighbor is
        free to win in a later round.

        Returns the set of ctrl_names to drop (losers). candidates values
        must be solution dicts as produced by
        ControllerTracker._compute_fused_solution (primary_cam, T_world_ctrl,
        error, assignment, aux_assignments).
        """
        names = list(candidates.keys())
        if len(names) < 2:
            return set()

        def _matched_pairs(sol: Dict, cid: int) -> List[Tuple[int, int]]:
            if sol.get('primary_cam') == cid:
                return sol.get('assignment') or []
            return (sol.get('aux_assignments') or {}).get(cid) or []

        def _matched_cams(sol: Dict) -> Set[int]:
            cams = set((sol.get('aux_assignments') or {}).keys())
            if sol.get('primary_cam') is not None:
                cams.add(sol['primary_cam'])
            return cams

        _occlusion_on = bool(self._matching_cfg.get('cross_controller_occlusion', False))
        _br          = float(self._matching_cfg.get('cross_occlusion_bounding_radius_m', 0.18))
        _gate_margin = float(self._matching_cfg.get('cross_occlusion_gate_margin_px', 20.0))

        conflicts: Dict[str, Set[str]] = {n: set() for n in names}
        for i, a in enumerate(names):
            sol_a = candidates[a]
            for b in names[i + 1:]:
                sol_b = candidates[b]
                conflict = False

                for cid in _matched_cams(sol_a) & _matched_cams(sol_b):
                    idx_a = {blob for blob, _ in _matched_pairs(sol_a, cid)}
                    idx_b = {blob for blob, _ in _matched_pairs(sol_b, cid)}
                    if idx_a & idx_b:
                        conflict = True
                        break

                if not conflict and _occlusion_on:
                    for occluder, victim, sol_occ, sol_vic in ((a, b, sol_a, sol_b), (b, a, sol_b, sol_a)):
                        tracker_vic = next(iter(self.ctrl_trackers[victim].trackers.values()))
                        geom_occ    = next(iter(self.ctrl_trackers[occluder].trackers.values()))._geometry
                        positions_vic = tracker_vic.model.positions
                        for cid in _matched_cams(sol_vic):
                            cam = self.cameras.get(cid)
                            if cam is None or cam.T_world_cam is None:
                                continue
                            led_ids = [lid for _, lid in _matched_pairs(sol_vic, cid)]
                            if not led_ids:
                                continue
                            T_cam_vic = cam.T_world_cam.inverse().compose(sol_vic['T_world_ctrl'])
                            T_cam_occ = cam.T_world_cam.inverse().compose(sol_occ['T_world_ctrl'])
                            focal_px = float(max(cam.camera_matrix[0, 0], cam.camera_matrix[1, 1]))
                            occluded = _cross_occluded_mask(
                                T_cam_vic.R.astype(np.float32), T_cam_vic.t.astype(np.float32),
                                positions_vic,
                                T_cam_occ.R.astype(np.float32), T_cam_occ.t.astype(np.float32),
                                geom_occ, _br, _br, focal_px, _gate_margin,
                                log_tag=f"[cold-conflict {occluder}->{victim} cam{cid}]",
                            )
                            if any(occluded[lid] for lid in led_ids):
                                conflict = True
                                break
                        if conflict:
                            break

                if conflict:
                    conflicts[a].add(b)
                    conflicts[b].add(a)

        if not any(conflicts.values()):
            return set()

        # A fit sitting right at the minimal-inlier floor has almost no spare
        # degrees of freedom, so a near-zero residual there is not evidence of
        # a correct match -- it's just what an under-constrained fit looks
        # like (compare _redundancy_factor in pose_search.py's proximity
        # path, same reasoning). Comparing raw error across candidates with
        # very different inlier counts lets exactly that kind of fit
        # outrank a genuinely well-constrained, higher-coverage, multi-camera
        # one. Discount error linearly by how many multiples of the floor a
        # candidate has: right at the floor -> no discount; 5x the floor ->
        # error divided by 5.
        _min_inliers = float(self._matching_cfg.get('min_inliers', 4))

        def _score(name: str) -> Tuple[float, int]:
            sol = candidates[name]
            total_pairs = len(sol.get('assignment') or []) + sum(
                len(v) for v in (sol.get('aux_assignments') or {}).values()
            )
            effective_error = sol['error'] * (_min_inliers / max(total_pairs, 1))
            return (effective_error, -total_pairs)

        remaining = set(names)
        losers: Set[str] = set()

        # Fixed candidates always win — drain them first, unconditionally,
        # before the score-based greedy pass below ever runs. A candidate
        # conflicting with a fixed one is dropped here regardless of its own
        # error, and a fixed candidate is never itself added to losers —
        # including in the (out of scope for this method to resolve, since
        # neither side can be un-committed) case where two fixed candidates
        # conflict with each other: that conflict is simply left unresolved
        # rather than corrupting the loser set.
        _fixed = set(fixed_names or ())
        for fixed in _fixed & remaining:
            if fixed not in remaining:
                continue
            remaining.discard(fixed)
            for loser in conflicts[fixed] & remaining:
                if loser in _fixed:
                    continue
                losers.add(loser)
                remaining.discard(loser)

        while remaining:
            winner = min(remaining, key=_score)
            remaining.discard(winner)
            for loser in conflicts[winner] & remaining:
                losers.add(loser)
                remaining.discard(loser)
        return losers
