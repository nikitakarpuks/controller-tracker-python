import copy
import cv2
import numpy as np
from loguru import logger
from typing import List, Tuple, Optional, Dict

from src.camera import Camera
from src.debug_config import is_deep
from src.geometry import Box3D, Cylinder3D, ControllerGeometry
from src.transformations import Transform


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


def _primitives_from_cfg(prim_cfg: dict):
    """Parse Box3D / Cylinder3D lists from a handle_primitives config dict."""
    boxes = []
    for b in prim_cfg.get("boxes", []):
        axes = np.asarray(b["axes"], float) if "axes" in b else None
        boxes.append(Box3D(
            name=b["name"],
            center=np.array(b["center"], float),
            half_dims=np.array(b["half_dims"], float),
            axes=axes,
            color=b.get("color", [220, 180, 80]),
        ))
    cylinders = []
    for cy in prim_cfg.get("cylinders", []):
        cylinders.append(Cylinder3D(
            name=cy["name"],
            center=np.array(cy["center"], float),
            axis=np.array(cy["axis"], float),
            radius=float(cy["radius"]),
            radius_v=float(cy["radius_v"]) if cy.get("radius_v") is not None else None,
            half_length=float(cy["half_length"]),
            angle=float(cy.get("angle", 0.0)),
            color=cy.get("color", [100, 180, 255]),
        ))
    return boxes, cylinders


def _compute_geometry(positions: np.ndarray, normals: np.ndarray, geometry_cfg: dict = None) -> ControllerGeometry:
    """
    Fit the LED-ring frustum and bundle it with the hardcoded handle body
    primitives (finalized in handle_vis.py) into one ControllerGeometry object.

    This replaces _compute_frustum_geometry which is now retired from _matching.
    """
    # ── Frustum: ring axis, inner/outer classification, cone fit ─────────────
    centroid  = np.array([0.0, 0.0, 0.0])
    ring_axis = np.array([0.0, 0.0, -1.0])

    rel        = positions - centroid
    rel_proj   = rel - np.outer(rel @ ring_axis, ring_axis)
    radial_out = rel_proj / (np.linalg.norm(rel_proj, axis=1, keepdims=True) + 1e-8)

    is_inner   = (normals * radial_out).sum(axis=1) < 0
    outer_mask = ~is_inner

    # orient axis toward the wider (large-radius) base → frustum_slope > 0
    big_ring_axis = -ring_axis if float((normals[outer_mask] @ ring_axis).mean()) > 0 else ring_axis

    axial_projs    = positions @ big_ring_axis
    ring_center_ax = float(axial_projs.mean())
    z_rel          = axial_projs - ring_center_ax

    outer_idx    = np.where(outer_mask)[0]
    outer_radial = np.linalg.norm(rel_proj[outer_idx], axis=1)
    outer_z_rel  = z_rel[outer_idx]

    A = np.column_stack([np.ones(len(outer_idx)), outer_z_rel])
    coeffs, _, _, _ = np.linalg.lstsq(A, outer_radial, rcond=None)
    R_fc          = float(coeffs[0])
    frustum_slope = float(coeffs[1])

    cfg = geometry_cfg or {}
    z_frustum_top = float(outer_z_rel.max()) + float(cfg.get("z_frustum_top_padding", 0.0045))
    z_frustum_bot = float(outer_z_rel.min()) - float(cfg.get("z_frustum_bot_padding", 0.0055))

    ## ── Inner cone radius (wall thickness from inner LED h_corpus) ────────────
    # r_led = np.linalg.norm(rel_proj, axis=1)
    # if is_inner.any():
    #     h_corpus       = np.maximum(0.0, R_fc + frustum_slope * z_rel[is_inner] - r_led[is_inner])
    #     wall_thickness = float(h_corpus.mean())
    # else:
    #     wall_thickness = 0.007
    wall_thickness = float(cfg.get("wall_thickness", 0.007))
    R_fc_inner = R_fc - wall_thickness

    # ── Handle body primitives ────────────────────────────────────────────────
    # Use config values if provided (right/left selectable), else fall back to
    # hardcoded right-controller defaults (keeps existing callers working).
    prim_cfg = cfg.get("handle_primitives")
    if prim_cfg is not None:
        boxes, cylinders = _primitives_from_cfg(prim_cfg)
    else:
        boxes = [
            Box3D(
                name="box_vertical",
                center=np.array([-0.009, -0.012, -0.015]),
                half_dims=np.array([0.021, 0.0028, 0.013]),
                color=[220, 130, 60],
            ),
            Box3D(
                name="box_horizontal",
                center=np.array([-0.009, -0.029, -0.0048]),
                half_dims=np.array([0.021, 0.0032, 0.015]),
                axes=np.column_stack([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).astype(float),
                color=[130, 60, 220],
            ),
        ]
        cylinders = [
            Cylinder3D(
                name="control_panel",
                center=np.array([-0.008, 0.0, -0.05]),
                axis=np.array([0.0, 1.0, 0.0]),
                radius=0.031,
                radius_v=0.033,
                half_length=0.017,
                angle=np.pi / 10,
                color=[100, 180, 255],
            ),
            Cylinder3D(
                name="handle",
                center=np.array([0.001, 0.022, -0.098]),
                axis=np.array([0.0, 1.0, -1.5]),
                radius=0.021,
                half_length=0.037,
                color=[255, 55, 55],
            ),
        ]

    return ControllerGeometry(
        ring_axis=big_ring_axis,
        is_inner=is_inner,
        radial_out=radial_out,
        ring_centroid=centroid,
        R_fc=R_fc,
        R_fc_inner=R_fc_inner,
        frustum_slope=frustum_slope,
        z_frustum_top=z_frustum_top,
        z_frustum_bot=z_frustum_bot,
        z_rel=z_rel,
        ring_center_ax=ring_center_ax,
        boxes=boxes,
        cylinders=cylinders,
    )


# =========================================================
# 3. TRACKER (per camera + controller)
# =========================================================

class SingleViewTracker:
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

        # Blob ID persistence: each blob carries the LED ID it was matched to
        # last frame, carried forward via spatial nearest-neighbour matching.
        self.prev_blob_positions: Optional[np.ndarray] = None  # (N, 2)
        self.prev_blob_led_ids:   Optional[np.ndarray] = None  # (N,) int, -1 = unknown

        # Consecutive frames without a valid solution.
        self.consecutive_failures: int = 0

        self._matching_cfg = matching_cfg or {}

        # Lazy cache (e.g. KD-tree later)
        self.kd_tree_cache = None

        # Pre-compute body geometry and LED quad cache once from the fixed model.
        from src._led_graph import (
            _build_led_neighbor_lists,
            _build_led_neighbor_lists_edge,
            _precompute_led_quads,
        )
        positions = model.positions.astype("float32")
        normals   = model.normals.astype("float32")

        self._geometry: ControllerGeometry = _compute_geometry(positions, normals, geometry_cfg)

        self._led_nbr = _build_led_neighbor_lists(positions, normals)
        self._led_triple_idx, self._led_triple_depth, self._led_triple_gates = _precompute_led_quads(
            positions, self._led_nbr,
        )
        self._led_nbr_edge = _build_led_neighbor_lists_edge(
            positions, normals, self._geometry.is_inner, self._geometry.z_rel
        )
        self._led_triple_idx_edge, self._led_triple_depth_edge, self._led_triple_gates_edge = (
            _precompute_led_quads(positions, self._led_nbr_edge)
        )

        # Bind strategies
        from src._matching import (
            proximity_match, brute_match, prior_constrained_match, _carry_led_ids,
        )

        self.proximity_match          = proximity_match.__get__(self)
        self.brute_match              = brute_match.__get__(self)
        self.prior_constrained_match  = prior_constrained_match.__get__(self)
        self._carry_led_ids           = _carry_led_ids

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
    def _extrapolate_pose(rvec_n, tvec_n, rvec_nm1, tvec_nm1):
        """
        Constant-velocity extrapolation: predict pose at frame n+1.

        Translation: tvec_{n+1} = 2*tvec_n - tvec_{n-1}
        Rotation: R_delta = R_n @ R_{n-1}.T; R_{n+1} = R_delta @ R_n
        """
        tvec_pred = 2.0 * np.asarray(tvec_n, np.float64) - np.asarray(tvec_nm1, np.float64)

        R_n,   _ = cv2.Rodrigues(np.asarray(rvec_n,   np.float32).reshape(3, 1))
        R_nm1, _ = cv2.Rodrigues(np.asarray(rvec_nm1, np.float32).reshape(3, 1))
        R_pred    = (R_n @ R_nm1.T) @ R_n
        rvec_pred, _ = cv2.Rodrigues(R_pred.astype(np.float32))

        return rvec_pred.reshape(3, 1).astype(np.float32), tvec_pred.reshape(3).astype(np.float32)

    # -----------------------------------------------------
    # Tracking
    # -----------------------------------------------------
    def track(self, blobs: np.ndarray, blob_radii: Optional[np.ndarray] = None,
              blob_brightnesses: Optional[np.ndarray] = None,
              other_cameras_blobs: Optional[List] = None) -> Optional[Dict]:
        """
        State machine:

        Have prev_pose?
          Yes → proximity_match (fast locked path); brute fallback only if proximity returns None
          Prediction via constant-velocity extrapolation (prev_pose + prev_prev_pose).
          No  → brute_match (cold start or re-acquisition)

        After any candidate solution:
          1. Pose-jump check against prev_pose (tight: 15 cm / 25°).
             If jump is too large → try brute_match; reject if still jumping.
          2. Error threshold (< 5 px accepted).

        On success: update prev_pose AND last_good_pose.
        On failure: clear prev_pose (trigger brute next frame)
                    but KEEP last_good_pose for re-acquisition plausibility check.
        """
        blobs   = np.asarray(blobs, dtype=np.float32).reshape(-1, 2)
        n_blobs = len(blobs)
        cam_idx = self.camera.camera_idx
        ctrl_name = self.model.name.replace("_controller", "")

        # Per-axis jump thresholds from config (fall back to scalar defaults if absent).
        _cfg = self._matching_cfg
        _use_proximity = bool(_cfg.get('use_proximity_match', True))
        _blob_snap_px  = float(_cfg.get('blob_tracking_snap_px', 25.0))
        _accept_err_px = float(_cfg.get('accept_error_px', 3.0))
        _pos_ax = _cfg.get('pose_jump_pos_thresh_m')
        _rot_ax = _cfg.get('pose_jump_rot_thresh_deg')
        _jump_kw = {}
        if _pos_ax is not None:
            _jump_kw['pos_thresh_xyz_m']   = tuple(_pos_ax)
        if _rot_ax is not None:
            _jump_kw['rot_thresh_xyz_deg'] = tuple(_rot_ax)

        # Normalise prev_pose shapes
        if self.prev_pose is not None:
            rvec, tvec = self.prev_pose
            self.prev_pose = (
                np.asarray(rvec, dtype=np.float32).reshape(3, 1),
                np.asarray(tvec, dtype=np.float32).reshape(3),
            )

        # Pose prediction: constant-velocity extrapolation from the last two frames.
        if self.prev_pose is not None and self.prev_prev_pose is not None:
            predicted_pose = self._extrapolate_pose(
                self.prev_pose[0],      self.prev_pose[1],
                self.prev_prev_pose[0], self.prev_prev_pose[1],
            )
        else:
            predicted_pose = self.prev_pose

        # ------------------------------------------------------------------
        # Blob LED-ID carry: inherit LED IDs from previous frame's blobs
        # ------------------------------------------------------------------
        blob_led_ids = None
        if self.prev_blob_positions is not None and self.prev_blob_led_ids is not None:
            blob_led_ids = self._carry_led_ids(
                blobs,
                self.prev_blob_positions,
                self.prev_blob_led_ids,
                snap_px=_blob_snap_px,
            )

        # ------------------------------------------------------------------
        # Candidate search
        # ------------------------------------------------------------------
        solution = None

        if self.prev_pose is not None:
            # --- Primary: proximity (fast, assignment-locked) ---
            if _use_proximity and n_blobs >= 3:
                solution = self.proximity_match(
                    blobs, predicted_pose,
                    prior_assignment=self.prev_assignment,
                    blob_led_ids=blob_led_ids,
                    blob_radii=blob_radii,
                    blob_brightnesses=blob_brightnesses,
                    other_cameras_blobs=other_cameras_blobs,
                )

            # --- Fallback: brute only when proximity found no solution ---
            if solution is None and n_blobs >= 4:
                logger.debug(f"[{ctrl_name} | cam {cam_idx} | track] proximity → None, running brute fallback")
                brute = self.brute_match(blobs, pose_prior=predicted_pose,
                                         other_cameras_blobs=other_cameras_blobs,
                                         blob_radii=blob_radii)
                if brute is not None:
                    solution = brute

        else:
            # --- No prior pose: brute-force re-acquisition ---
            if n_blobs >= 4:
                logger.debug(f"[{ctrl_name} | cam {cam_idx} | track] no prev_pose → cold-start brute")
                solution = self.brute_match(blobs, other_cameras_blobs=other_cameras_blobs,
                                            blob_radii=blob_radii)

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
                and 2 <= n_blobs <= 3):
            logger.debug(f"[{ctrl_name} | cam {cam_idx} | track] n_blobs={n_blobs} + prior → prior_constrained_match")
            solution = self.prior_constrained_match(
                blobs, predicted_pose,
                prior_assignment=self.prev_assignment,
                blob_radii=blob_radii,
                other_cameras_blobs=other_cameras_blobs,
            )

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
                if n_blobs >= 4:
                    brute = self.brute_match(blobs, pose_prior=self.prev_pose,
                                             other_cameras_blobs=other_cameras_blobs,
                                             blob_radii=blob_radii)
                    if brute is not None and not self._pose_jump_too_large(
                        brute["rvec"], brute["tvec"], rvec_p, tvec_p,
                        **_jump_kw,
                    ):
                        solution = brute

        # ------------------------------------------------------------------
        # Accept / reject
        # ------------------------------------------------------------------

        # Proximity found a solution but error is too high — try brute before giving up
        if solution is not None and solution["error"] >= _accept_err_px and n_blobs >= 4:
            logger.debug(f"[{ctrl_name} | cam {cam_idx} | track] proximity err={solution['error']:.2f}px exceeds threshold, attempting brute recovery")
            brute = self.brute_match(blobs, pose_prior=self.prev_pose,
                                     other_cameras_blobs=other_cameras_blobs,
                                     blob_radii=blob_radii)
            if brute is not None:
                logger.debug(f"[{ctrl_name} | cam {cam_idx} | track] brute recovery err={brute['error']:.2f}px")
                solution = brute

        if solution is not None and solution["error"] < _accept_err_px:
            logger.debug(
                f"[{ctrl_name} | cam {cam_idx} | track] accepted — method={solution.get('method','?')}  "
                f"inliers={len(solution['assignment'])}  err={solution['error']:.2f}px"
            )
            self.prev_prev_pose  = self.prev_pose  # shift window for next-frame extrapolation
            self.prev_pose       = (solution["rvec"], solution["tvec"])
            self.prev_assignment = solution["assignment"]
            self.last_good_pose       = self.prev_pose
            self.last_good_assignment = self.prev_assignment
            self.consecutive_failures = 0

            # World-frame controller pose: T_world_ctrl = T_world_cam ∘ T_cam_ctrl
            R_ctrl, _ = cv2.Rodrigues(solution["rvec"])
            T_cam_ctrl = Transform(R_ctrl, solution["tvec"].reshape(3))
            solution["T_world_ctrl"] = self.T_world_cam.compose(T_cam_ctrl)

            # Update blob ID state for next frame.
            blob_led_ids_out = np.full(n_blobs, -1, dtype=np.int32)
            for b_idx, l_id in solution["assignment"]:
                blob_led_ids_out[b_idx] = l_id
            self.prev_blob_positions = blobs.copy()
            self.prev_blob_led_ids   = blob_led_ids_out

            return solution

        # Tracking lost — clear velocity history and blob ID state so re-acquisition starts fresh
        if solution is not None:
            logger.debug(f"[{ctrl_name} | cam {cam_idx} | track] rejected — err={solution['error']:.2f}px exceeds {_accept_err_px:.1f}px threshold")
        else:
            logger.debug(f"[{ctrl_name} | cam {cam_idx} | track] rejected — no solution found")
        self.consecutive_failures += 1
        self.prev_pose           = None
        self.prev_prev_pose      = None
        self.prev_assignment     = None
        self.prev_blob_positions = None
        self.prev_blob_led_ids   = None
        return None


# =========================================================
# 3. SYSTEM (multi-controller, multi-camera)
# =========================================================

class TrackingSystem:
    def __init__(self, controllers: List[ControllerModel], cameras: List[Camera],
                 matching_cfg: dict = None, geometry_cfg: dict = None,
                 geometry_cfg_per_ctrl: dict = None):

        self.cameras: Dict[int, Camera] = {cam.camera_idx: cam for cam in cameras}
        self.trackers: Dict[Tuple[str, int], SingleViewTracker] = {}

        for ctrl in controllers:
            for cam in cameras:
                key = (ctrl.name, cam.camera_idx)
                geo = (geometry_cfg_per_ctrl or {}).get(ctrl.name) or geometry_cfg
                self.trackers[key] = SingleViewTracker(cam, ctrl, matching_cfg=matching_cfg, geometry_cfg=geo)

        self._matching_cfg:       dict                       = matching_cfg or {}
        self._designated_primary: Dict[str, Optional[int]]  = {}
        self._handoff_counter:    Dict[str, Dict[int, int]] = {}

    @staticmethod
    def _strip_matched(
        cam_pool: Dict[int, np.ndarray],
        radii_pool: Dict[int, Optional[np.ndarray]],
        pool_orig_idx: Dict[int, List[int]],
        solution: Dict,
        primary_cam_id: int,
    ) -> None:
        """Remove blobs consumed by this controller from the shared pools.
        Also updates pool_orig_idx so subsequent controllers can remap their
        pool-relative blob IDs back to the original observation indices."""
        def _remove(cam_id: int, indices: set) -> None:
            if not indices or cam_id not in cam_pool or cam_pool[cam_id] is None:
                return
            keep = [i for i in range(len(cam_pool[cam_id])) if i not in indices]
            cam_pool[cam_id] = cam_pool[cam_id][keep]
            if radii_pool.get(cam_id) is not None:
                radii_pool[cam_id] = radii_pool[cam_id][keep]
            if cam_id in pool_orig_idx:
                pool_orig_idx[cam_id] = [pool_orig_idx[cam_id][k] for k in keep]

        _remove(primary_cam_id, {b for b, _ in solution.get("assignment", [])})
        for cam_id, pairs in (solution.get("aux_assignments") or {}).items():
            _remove(cam_id, {b for b, _ in pairs})

    def update(
        self,
        observations_per_camera: Dict[int, np.ndarray],
        radii_per_camera: Optional[Dict[int, np.ndarray]] = None,
        brightnesses_per_camera: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[str, Optional[Dict]]:
        """
        Run tracking for every controller using a single primary camera.

        For each controller:
          - Tracking mode  (a tracker already has prev_pose):
              use that tracker's camera — it runs proximity or brute on its own blobs.
          - Cold start / re-acquisition:
              pick the camera with the most blobs as the primary camera,
              run brute_match once, and pass every other camera's blobs as
              other_cameras_blobs so the existing aux validation scores them.

        Controllers that are already tracking run first; their matched blobs are
        removed from the shared pool before cold-starting controllers search,
        reducing false-positive P3P candidates during re-acquisition.

        Returns {ctrl_name: solution_or_None}.  The solution dict gains a
        "primary_cam" key with the index of the camera that produced the pose.
        """
        results: Dict[str, Optional[Dict]] = {}

        ctrl_names: set = {ctrl for ctrl, _ in self.trackers}

        # Tracking controllers first → their blobs are stripped before cold-start search
        def _has_prior(name: str) -> bool:
            return any(
                t.prev_pose is not None
                for (cname, _), t in self.trackers.items()
                if cname == name
            )
        ordered = sorted(ctrl_names, key=lambda n: 0 if _has_prior(n) else 1)

        # Mutable per-camera pools; tracking controllers consume their blobs first
        cam_pool: Dict[int, np.ndarray] = {
            cid: blobs.copy() if blobs is not None else None
            for cid, blobs in observations_per_camera.items()
        }
        radii_pool: Dict[int, Optional[np.ndarray]] = {
            cid: (r.copy() if r is not None else None)
            for cid, r in (radii_per_camera or {}).items()
        }
        brightnesses_pool: Dict[int, Optional[np.ndarray]] = {
            cid: (b.copy() if b is not None else None)
            for cid, b in (brightnesses_per_camera or {}).items()
        }
        # Maps current pool index → original observation index per camera.
        # Stays in sync with cam_pool so later controllers can remap their
        # pool-relative blob IDs to original indices for visualization.
        pool_orig_idx: Dict[int, List[int]] = {
            cid: list(range(len(blobs)))
            for cid, blobs in observations_per_camera.items()
            if blobs is not None
        }

        for ctrl_name in ordered:
            ctrl_pairs: List[Tuple[int, "SingleViewTracker"]] = [
                (cam_id, tracker)
                for (cname, cam_id), tracker in self.trackers.items()
                if cname == ctrl_name
            ]

            # Cameras that delivered blob observations this frame (draw from mutable pool)
            available: Dict[int, np.ndarray] = {
                cam_id: cam_pool[cam_id]
                for cam_id, _ in ctrl_pairs
                if cam_id in cam_pool
                   and cam_pool[cam_id] is not None
            }

            if not available:
                results[ctrl_name] = None
                continue

            tracker_map = {cam_id: t for cam_id, t in ctrl_pairs}
            _cfg = self._matching_cfg

            # --- Select primary camera ---
            # Prefer the designated primary (handoff winner) if it has a prior pose.
            _designated = self._designated_primary.get(ctrl_name)
            if (_designated is not None
                    and _designated in available
                    and tracker_map[_designated].prev_pose is not None):
                primary_cam_id = _designated
            else:
                tracking_cam = next(
                    (cam_id for cam_id, tracker in ctrl_pairs
                     if tracker.prev_pose is not None and cam_id in available),
                    None,
                )
                primary_cam_id = tracking_cam if tracking_cam is not None else max(
                    available, key=lambda cid: len(available[cid])
                )

            primary_tracker      = tracker_map[primary_cam_id]
            primary_blobs        = available[primary_cam_id]
            primary_radii        = radii_pool.get(primary_cam_id)
            primary_brightnesses = brightnesses_pool.get(primary_cam_id)

            other_cameras_blobs = [
                (self.cameras[cid], available[cid], radii_pool.get(cid))
                for cid in available
                if cid != primary_cam_id
            ]

            solution = primary_tracker.track(
                primary_blobs,
                blob_radii=primary_radii,
                blob_brightnesses=primary_brightnesses,
                other_cameras_blobs=other_cameras_blobs,
            )

            if solution is not None:
                solution["primary_cam"] = primary_cam_id

                # Save pool-relative blob IDs for stripping (must stay pool-relative).
                _pool_rel = {
                    "assignment":      list(solution["assignment"]),
                    "aux_assignments": dict(solution.get("aux_assignments") or {}),
                }

                # Remap pool-relative blob IDs → original observation indices so
                # visualization can index directly into the unstripped blob arrays.
                _orig_p = pool_orig_idx.get(primary_cam_id, [])
                if _orig_p:
                    solution["assignment"] = [
                        (_orig_p[b], lid) for b, lid in solution["assignment"]
                    ]
                _aux_asgns_raw = solution.get("aux_assignments") or {}
                if _aux_asgns_raw:
                    solution["aux_assignments"] = {
                        cam_id: [(_orig_a[b], lid) for b, lid in pairs]
                        for cam_id, pairs in _aux_asgns_raw.items()
                        if (_orig_a := pool_orig_idx.get(cam_id, []))
                    }

                # Strip using pool-relative IDs, then update pool_orig_idx.
                self._strip_matched(cam_pool, radii_pool, pool_orig_idx, _pool_rel, primary_cam_id)

                # State propagation: push T_world_ctrl into every non-primary tracker
                # so any camera can be promoted to primary without cold-start brute search.
                T_world_ctrl = solution["T_world_ctrl"]
                aux_asgns = solution.get("aux_assignments") or {}
                for _cid, _tracker in tracker_map.items():
                    if _cid == primary_cam_id:
                        continue
                    _T_cam_ctrl = self.cameras[_cid].T_world_cam.inverse().compose(T_world_ctrl)
                    _rv_np, _ = cv2.Rodrigues(_T_cam_ctrl.R.astype(np.float32))
                    _tv_np = _T_cam_ctrl.t.astype(np.float32)
                    _tracker.prev_prev_pose = _tracker.prev_pose
                    _tracker.prev_pose      = (_rv_np.reshape(3, 1), _tv_np)
                    _tracker.last_good_pose = _tracker.prev_pose
                    _aux_asgn = aux_asgns.get(_cid)
                    if _aux_asgn is not None:
                        _tracker.prev_assignment      = _aux_asgn
                        _tracker.last_good_assignment = _aux_asgn
                    elif _tracker.prev_assignment is None:
                        _tracker.last_good_assignment = None
                    _tracker.consecutive_failures = 0

                # Handoff check: promote an aux camera that consistently dominates.
                # Uses full aux_assignments (snap + expansion) rather than aux_cameras
                # (which only counts LEDs in the primary's RANSAC inlier set).
                if bool(_cfg.get('camera_handoff', True)):
                    _ratio_thr  = float(_cfg.get('handoff_coverage_ratio',   1.5))
                    _min_adv    = int(  _cfg.get('handoff_min_advantage',     3))
                    _hysteresis = int(  _cfg.get('handoff_hysteresis_frames', 3))
                    _n_primary  = len(solution["assignment"])
                    _hctr = self._handoff_counter.setdefault(ctrl_name, {})
                    for _aux_cid, _aux_pairs in (aux_asgns.items()):
                        _aux_n = len(_aux_pairs)
                        if (_aux_n >= _ratio_thr * _n_primary
                                and _aux_n - _n_primary >= _min_adv):
                            _hctr[_aux_cid] = _hctr.get(_aux_cid, 0) + 1
                            if _hctr[_aux_cid] >= _hysteresis:
                                logger.info(
                                    f"[{ctrl_name}] Camera handoff: "
                                    f"cam{primary_cam_id}({_n_primary} LEDs)"
                                    f" → cam{_aux_cid}({_aux_n} LEDs)"
                                )
                                self._designated_primary[ctrl_name] = _aux_cid
                                _hctr.clear()
                        else:
                            _hctr[_aux_cid] = 0

            results[ctrl_name] = solution

        return results


# class MultiViewFusion:
#     def fuse(self, poses_from_cameras):
#         # average / optimize / triangulate
#         return fused_pose
