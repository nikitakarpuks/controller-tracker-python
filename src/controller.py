import copy
import cv2
import numpy as np
from loguru import logger
from typing import List, Tuple, Optional, Dict, Set

from src._pnp import _project_points
from src.camera import Camera
from src.debug_config import is_deep
from src.geometry import Box3D, Cylinder3D, ControllerGeometry
from src.transformations import Transform
# from src._self_calibration import SelfCalibrator


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
              other_cameras_blobs: Optional[List] = None,
              blob_mask: Optional[np.ndarray] = None) -> Optional[Dict]:
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
                blobs_prox,
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
            if _use_proximity and n_available >= 3:
                solution = self.proximity_match(
                    blobs_prox, predicted_pose,
                    prior_assignment=self.prev_assignment,
                    blob_led_ids=blob_led_ids,
                    blob_radii=radii_prox,
                    blob_brightnesses=brts_prox,
                    other_cameras_blobs=other_cameras_blobs,
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
                                         blob_mask=blob_mask)
                if brute is not None:
                    solution = brute

        else:
            # --- No prior pose: brute-force re-acquisition ---
            if n_available >= 4:
                logger.debug(f"[{ctrl_name} | cam {cam_idx} | track] no prev_pose → cold-start brute")
                solution = self.brute_match(blobs, other_cameras_blobs=other_cameras_blobs,
                                            blob_radii=blob_radii,
                                            blob_mask=blob_mask)

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
                    brute = self.brute_match(blobs, pose_prior=self.prev_pose,
                                             other_cameras_blobs=other_cameras_blobs,
                                             blob_radii=blob_radii,
                                             blob_mask=blob_mask)
                    if brute is not None and not self._pose_jump_too_large(
                        brute["rvec"], brute["tvec"], rvec_p, tvec_p,
                        **_jump_kw,
                    ):
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
            # State is always stored against blobs_prox so _carry_led_ids remains consistent.
            n_prox = len(blobs_prox)
            blob_led_ids_out = np.full(n_prox, -1, dtype=np.int32)
            if blob_mask is not None:
                # assignment indices are in full-array space; convert to prox space.
                _full_to_prox = -np.ones(n_blobs, dtype=np.int32)
                _full_to_prox[_avail_idx] = np.arange(n_prox, dtype=np.int32)
                for b_full, l_id in solution["assignment"]:
                    b_prox = int(_full_to_prox[b_full])
                    if b_prox >= 0:
                        blob_led_ids_out[b_prox] = l_id
            else:
                for b_idx, l_id in solution["assignment"]:
                    blob_led_ids_out[b_idx] = l_id
            self.prev_blob_positions = blobs_prox.copy()
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
                 geometry_cfg_per_ctrl: dict = None,
                 self_calibration_cfg: dict = None):

        self.cameras: Dict[int, Camera] = {cam.camera_idx: cam for cam in cameras}

        # Self-calibration: optionally apply saved extrinsics before tracker creation
        # so every tracker's T_world_cam starts with the correct (calibrated) value.
        # self._self_cal: Optional[SelfCalibrator] = None
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
                if sc_cfg.get("apply_on_load", True):
                    from pathlib import Path
                    _out = Path(sc_cfg.get("output_path",
                                           "./data/cameras/self_calibrated_extrinsics.json"))
                    SelfCalibrator.load_and_apply(_out, self.cameras, _sc_primary_idx)

                self._self_cal = SelfCalibrator(_primary_cam, _aux_cams, sc_cfg)

        self.trackers: Dict[Tuple[str, int], SingleViewTracker] = {}

        for ctrl in controllers:
            for cam in cameras:
                key = (ctrl.name, cam.camera_idx)
                geo = (geometry_cfg_per_ctrl or {}).get(ctrl.name) or geometry_cfg
                self.trackers[key] = SingleViewTracker(cam, ctrl, matching_cfg=matching_cfg, geometry_cfg=geo)

        self._matching_cfg:       dict                       = matching_cfg or {}
        self._designated_primary: Dict[str, Optional[int]]  = {}
        self._handoff_counter:    Dict[str, Dict[int, int]] = {}

        # Resolve fixed primary camera: matching cfg wins, then self-cal lock_primary.
        _fpc = self._matching_cfg.get("fixed_primary_camera")
        if _fpc is None and _sc_primary_idx is not None and sc_cfg.get("lock_primary", True):
            _fpc = _sc_primary_idx
        self._fixed_primary_cam: Optional[int] = int(_fpc) if _fpc is not None else None

        if self._fixed_primary_cam is not None:
            for ctrl in controllers:
                self._designated_primary[ctrl.name] = self._fixed_primary_cam


    def update(
        self,
        observations_per_camera: Dict[int, np.ndarray],
        radii_per_camera: Optional[Dict[int, np.ndarray]] = None,
        brightnesses_per_camera: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[str, Optional[Dict]]:
        """
        Run tracking for every controller using a single primary camera.

        Blob ownership model: observations_per_camera is never mutated. Instead,
        claimed_blobs tracks which original blob indices have been consumed by
        earlier controllers. Each controller receives a pre-filtered view of
        unclaimed blobs; assignments returned are remapped to original indices.

        Controllers with a prior pose run first so their high-confidence claims
        are registered before cold-starting controllers search.

        Returns {ctrl_name: solution_or_None}. The solution dict gains a
        "primary_cam" key with the index of the camera that produced the pose.
        """
        results: Dict[str, Optional[Dict]] = {}

        ctrl_names: List[str] = sorted({ctrl for ctrl, _ in self.trackers})

        # Controllers with a prior pose run first — their claims reduce the search
        # space for cold-starting controllers.
        # Sort by (has_prior_rank, name) for full determinism when priorities are equal.
        def _has_prior(name: str) -> bool:
            return any(
                t.prev_pose is not None
                for (cname, _), t in self.trackers.items()
                if cname == name
            )
        ordered = sorted(ctrl_names, key=lambda n: (0 if _has_prior(n) else 1, n))

        # Blob ownership: original observation indices claimed by controllers so far.
        claimed_blobs: Dict[int, Set[int]] = {}

        def _filter_cam(cam_id: int, extra_excluded: Optional[Set[int]] = None):
            """Return (filtered_blobs, filtered_radii, filtered_brts, orig_indices)
            for camera cam_id, excluding already-claimed blobs and any extra_excluded
            original indices (Phase 3 reservations for later controllers).
            Returns (None, None, None, []) when no unclaimed blobs remain."""
            obs = observations_per_camera.get(cam_id)
            if obs is None or len(obs) == 0:
                return None, None, None, []
            claimed = claimed_blobs.get(cam_id, set())
            excluded = claimed if not extra_excluded else claimed | extra_excluded
            keep = [i for i in range(len(obs)) if i not in excluded]
            if not keep:
                return None, None, None, []
            k = np.array(keep, dtype=np.int32)
            f_blobs = obs[k]
            _r = (radii_per_camera or {}).get(cam_id)
            f_radii = _r[k] if _r is not None else None
            _b = (brightnesses_per_camera or {}).get(cam_id)
            f_brts = _b[k] if _b is not None else None
            return f_blobs, f_radii, f_brts, keep

        # ── Phase 3: Proximity mutual exclusion ───────────────────────────────
        # When multiple controllers have prior poses, pre-reserve blobs that are
        # clearly "owned" by a later-ordered controller so the earlier controller
        # cannot steal them via aux-camera joint optimisation.
        #
        # A blob is reserved for controller J (index j > i in ordered) if:
        #   • it is within reserve_px of J's projected LEDs, AND
        #   • it is closer to J's projection than to I's projection (Voronoi).
        # When I has no projection for that camera, all blobs near J are reserved.
        _cfg = self._matching_cfg
        # Phase 3 radius: used when only one controller has a projection on a camera.
        _reserve_px  = float(_cfg.get(
            'proximity_mutual_exclusion_px',
            _cfg.get('aux_snap_px', 15.0),
        ))
        # Phase 4 radius: when both controllers have projections, extend the Voronoi
        # partition to cover the full controller body region (ring + handle area).
        # Blobs within this distance of EITHER controller's LEDs are partitioned;
        # blobs beyond it are unpartitioned (available to both, helps brute recovery).
        _voronoi_px  = float(_cfg.get('voronoi_max_dist_px', 100.0))

        # Cache per-controller, per-camera projected LED positions.
        # {ctrl_name → {cam_id → (N,2) float32 or None}}
        _ctrl_proj: Dict[str, Dict[int, Optional[np.ndarray]]] = {}
        if len(ordered) >= 2:
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
                    _vis = _pts_cam[:, 2] > 0.01  # LED must be in front of camera
                    if not _vis.any():
                        _ctrl_proj[_cn][_cid] = None
                        continue
                    _ctrl_proj[_cn][_cid] = _project_points(
                        _rv, _tv, _trk.model.positions[_vis],
                        _cam.camera_matrix, _cam.dist_coeffs,
                    )

        # Build reservation sets: _reservations[ctrl_i][cam_id] = set of original
        # blob indices that ctrl_i must NOT use (reserved for later controllers).
        _reservations: Dict[str, Dict[int, Set[int]]] = {_cn: {} for _cn in ordered}
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
                        continue  # j has no projection on this camera — nothing to reserve
                    # Min distance from each blob to j's visible LED projections.
                    _d_j = np.linalg.norm(
                        _obs[:, None, :] - _pj[None, :, :], axis=2
                    ).min(axis=1)
                    _pi = _proj_i.get(_cid)
                    if _pi is not None and len(_pi) > 0:
                        # Phase 4 — Full Voronoi: both controllers have projections.
                        # Partition every blob within voronoi_px of EITHER controller.
                        # Blobs closer to j than i are reserved for j.
                        _d_i = np.linalg.norm(
                            _obs[:, None, :] - _pi[None, :, :], axis=2
                        ).min(axis=1)
                        _near_either = (_d_j < _voronoi_px) | (_d_i < _voronoi_px)
                        if not _near_either.any():
                            continue
                        _reserve_mask = _near_either & (_d_j < _d_i)
                    else:
                        # Phase 3 — only j has projection: radius-bounded reservation.
                        _near_j = _d_j < _reserve_px
                        if not _near_j.any():
                            continue
                        _reserve_mask = _near_j
                    if not _reserve_mask.any():
                        continue
                    _res_set = _reservations[_cn_i].setdefault(_cid, set())
                    _res_set.update(int(k) for k in np.where(_reserve_mask)[0])

        for ctrl_name in ordered:
            ctrl_pairs: List[Tuple[int, "SingleViewTracker"]] = [
                (cam_id, tracker)
                for (cname, cam_id), tracker in self.trackers.items()
                if cname == ctrl_name
            ]

            # Build per-camera filtered views for this controller.
            # Exclude both claimed blobs (Phase 1) and blobs reserved for later
            # controllers whose projected LEDs are closer (Phase 3).
            # Key: cam_id → (filtered_blobs, filtered_radii, filtered_brts, orig_idx_list)
            _ctrl_reservations = _reservations.get(ctrl_name, {})
            avail: Dict[int, tuple] = {}
            for cam_id, _ in ctrl_pairs:
                _extra = _ctrl_reservations.get(cam_id) or None
                fb, fr, fbt, keep = _filter_cam(cam_id, extra_excluded=_extra)
                if fb is not None:
                    avail[cam_id] = (fb, fr, fbt, keep)

            if not avail:
                results[ctrl_name] = None
                continue

            tracker_map = {cam_id: t for cam_id, t in ctrl_pairs}

            # --- Select primary camera ---
            # Fixed primary always wins when it has unclaimed blobs.
            if (self._fixed_primary_cam is not None
                    and self._fixed_primary_cam in avail):
                primary_cam_id = self._fixed_primary_cam
            else:
                _designated = self._designated_primary.get(ctrl_name)
                _des_tracker = tracker_map.get(_designated)
                # Require at least 3 unclaimed blobs: minimum for proximity match.
                # When the competing controller claimed most blobs from this camera,
                # fall through to the best available camera instead.
                _des_ok = (
                    _des_tracker is not None
                    and _designated in avail
                    and _des_tracker.prev_pose is not None
                    and len(avail[_designated][0]) >= 3
                )
                if _des_ok:
                    primary_cam_id = _designated
                else:
                    # Pick camera with prev_pose that has the most unclaimed blobs.
                    # Cold-start (no prev_pose anywhere): prefer the camera with the
                    # fewest blobs — it is less likely to be contaminated by another
                    # controller's LEDs, giving a more reliable brute-match anchor.
                    # Require ≥ min_inliers blobs so brute_match can actually run.
                    _min_inliers = int(_cfg.get('min_inliers', 4))
                    _with_prior = [
                        cid for cid in avail
                        if tracker_map[cid].prev_pose is not None
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

            primary_tracker = tracker_map[primary_cam_id]
            primary_blobs, primary_radii, primary_brightnesses, primary_orig = avail[primary_cam_id]

            other_cameras_blobs = [
                (self.cameras[cid], av[0], av[1])
                for cid, av in avail.items()
                if cid != primary_cam_id
            ]
            # orig-index maps for aux cameras — needed to remap solution indices.
            aux_orig: Dict[int, List[int]] = {
                cid: av[3] for cid, av in avail.items() if cid != primary_cam_id
            }

            # Phase 5: pass the full camera observation array + availability mask so
            # brute_match can search across all blobs (not just the pre-filtered subset).
            _prim_obs_full = observations_per_camera[primary_cam_id]
            _prim_rad_full = (radii_per_camera or {}).get(primary_cam_id)
            _prim_brt_full = (brightnesses_per_camera or {}).get(primary_cam_id)
            _prim_mask = np.zeros(len(_prim_obs_full), dtype=bool)
            _prim_mask[primary_orig] = True

            solution = primary_tracker.track(
                _prim_obs_full,
                blob_radii=_prim_rad_full,
                blob_brightnesses=_prim_brt_full,
                other_cameras_blobs=other_cameras_blobs,
                blob_mask=_prim_mask,
            )

            # Primary failed — try other cameras before giving up.
            # Warm: prev_pose cameras first, then most blobs (best proximity coverage).
            # Cold: prev_pose cameras first, then fewest blobs (least contamination).
            _cold_start = primary_tracker.prev_pose is None
            if solution is None and len(avail) > 1:
                _fallback_order = sorted(
                    [cid for cid in avail if cid != primary_cam_id],
                    key=lambda cid: (
                        0 if tracker_map[cid].prev_pose is not None else 1,
                        len(avail[cid][0]) if _cold_start else -len(avail[cid][0]),
                    ),
                )
                for _fb_cid in _fallback_order:
                    _fb_blobs, _fb_radii, _fb_brts, _fb_orig = avail[_fb_cid]
                    if len(_fb_blobs) < 3:
                        continue
                    _fb_tracker = tracker_map[_fb_cid]
                    _fb_other = [
                        (self.cameras[cid], av[0], av[1])
                        for cid, av in avail.items() if cid != _fb_cid
                    ]
                    _fb_obs_full = observations_per_camera[_fb_cid]
                    _fb_rad_full = (radii_per_camera or {}).get(_fb_cid)
                    _fb_brt_full = (brightnesses_per_camera or {}).get(_fb_cid)
                    _fb_mask = np.zeros(len(_fb_obs_full), dtype=bool)
                    _fb_mask[_fb_orig] = True
                    solution = _fb_tracker.track(
                        _fb_obs_full,
                        blob_radii=_fb_rad_full,
                        blob_brightnesses=_fb_brt_full,
                        other_cameras_blobs=_fb_other,
                        blob_mask=_fb_mask,
                    )
                    if solution is not None:
                        primary_cam_id  = _fb_cid
                        primary_tracker = _fb_tracker
                        primary_orig    = _fb_orig
                        aux_orig = {
                            cid: av[3] for cid, av in avail.items() if cid != _fb_cid
                        }
                        break

            if solution is not None:
                solution["primary_cam"] = primary_cam_id

                # Persist winning camera so next frame's selection prefers it.
                if self._fixed_primary_cam is None:
                    self._designated_primary[ctrl_name] = primary_cam_id

                # Remap filtered indices → original observation indices.
                # When _orig_idx is set, track() already mapped indices to original space.
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

                # Register claimed blobs (original indices) so subsequent controllers
                # receive pre-filtered views that exclude these.
                for b, _ in solution["assignment"]:
                    claimed_blobs.setdefault(primary_cam_id, set()).add(b)
                for cid, pairs in (solution.get("aux_assignments") or {}).items():
                    for b, _ in pairs:
                        claimed_blobs.setdefault(cid, set()).add(b)

                # Self-calibration: feed observations when primary cam matches anchor.
                if (self._self_cal is not None
                        and primary_cam_id == self._self_cal.primary_camera.camera_idx):
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
                        _blobs_aux = observations_per_camera[_aux_cid][_blob_idxs]
                        _sc_aux_obs[_aux_cid] = (_pts_prim, _blobs_aux)
                    if _sc_aux_obs:
                        self._self_cal.add_frame(
                            solution["rvec"], solution["tvec"],
                            primary_error=solution["error"],
                            primary_inliers=len(solution["assignment"]),
                            aux_observations=_sc_aux_obs,
                        )
                        if self._self_cal.should_run():
                            _cal = self._self_cal.run()
                            self._self_cal.apply_to_cameras(_cal)
                            for (_, _cam_id), _trk in self.trackers.items():
                                _trk.T_world_cam = self.cameras[_cam_id].T_world_cam

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
                if bool(_cfg.get('camera_handoff', True)) and self._fixed_primary_cam is None:
                    _ratio_thr  = float(_cfg.get('handoff_coverage_ratio',   1.5))
                    _min_adv    = int(  _cfg.get('handoff_min_advantage',     3))
                    _hysteresis = int(  _cfg.get('handoff_hysteresis_frames', 3))
                    _n_primary  = len(solution["assignment"])
                    _hctr = self._handoff_counter.setdefault(ctrl_name, {})
                    for _aux_cid, _aux_pairs in aux_asgns.items():
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
