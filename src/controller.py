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

def _compute_geometry(positions: np.ndarray, normals: np.ndarray) -> ControllerGeometry:
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

    z_frustum_top = float(outer_z_rel.max()) + 0.0015
    z_frustum_bot = float(outer_z_rel.min()) - 0.0055

    # ── Inner cone radius (wall thickness from inner LED h_corpus) ────────────
    r_led = np.linalg.norm(rel_proj, axis=1)
    # if is_inner.any():
    #     h_corpus       = np.maximum(0.0, R_fc + frustum_slope * z_rel[is_inner] - r_led[is_inner])
    #     wall_thickness = float(h_corpus.mean())
    # else:
    #     wall_thickness = 0.007
    wall_thickness = 0.007  # hardcode so inner leds would not be inside frustum and hence always occluded
    R_fc_inner = R_fc - wall_thickness

    # ── Handle body primitives (values finalized via handle_vis.py) ───────────
    boxes = [
        Box3D(
            name="box_vertical",
            center=np.array([-0.009, -0.012, -0.015]),
            half_dims=np.array([0.021, 0.0028, 0.013]),
        ),
        Box3D(
            name="box_horizontal",
            center=np.array([-0.009, -0.029, -0.0048]),
            half_dims=np.array([0.021, 0.0032, 0.015]),
            axes=np.column_stack([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).astype(float),
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
        ),
        Cylinder3D(
            name="handle",
            center=np.array([0.001, 0.022, -0.098]),
            axis=np.array([0.0, 1.0, -1.5]),
            radius=0.021,
            half_length=0.037,
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
    def __init__(self, camera: Camera, model: ControllerModel):
        self.camera = camera
        self.model = model

        # Cached transform (Camera -> VRH IMU)
        self.T_imu_cam: Transform = camera.T_imu_cam

        # Tracking state — current frame
        self.prev_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.prev_assignment = None

        # Last frame where tracking was confirmed good.
        # Retained across loss events so brute re-acquisition can be
        # validated for plausibility (pose-jump guard).
        self.last_good_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.last_good_assignment = None

        # Consecutive frames without a valid solution.
        self.consecutive_failures: int = 0

        # Lazy cache (e.g. KD-tree later)
        self.kd_tree_cache = None

        # Pre-compute body geometry and LED quad cache once from the fixed model.
        from src._matching import (
            _build_led_neighbor_lists,
            _build_led_neighbor_lists_edge,
            _precompute_led_quads,
        )
        positions = model.positions.astype("float32")
        normals   = model.normals.astype("float32")

        self._geometry: ControllerGeometry = _compute_geometry(positions, normals)

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
        from src._matching import proximity_match, brute_match
        from src._pnp_solver import p2p_solver, p1p_solver

        self.proximity_match = proximity_match.__get__(self)
        self.brute_match = brute_match.__get__(self)
        self.p2p_solver = p2p_solver.__get__(self)
        self.p1p_solver = p1p_solver.__get__(self)

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
    ) -> bool:
        """
        Return True if the new pose is implausibly far from the reference.

        Position:  Euclidean distance between translation vectors.
        Rotation:  angle of the relative rotation R_new @ R_ref^T,
                   computed as arccos((trace - 1) / 2).
        """
        tvec_new = np.asarray(tvec_new, dtype=np.float64).reshape(3)
        tvec_ref = np.asarray(tvec_ref, dtype=np.float64).reshape(3)
        if np.linalg.norm(tvec_new - tvec_ref) > max_dist_m:
            return True

        R_new, _ = cv2.Rodrigues(np.asarray(rvec_new, dtype=np.float32).reshape(3, 1))
        R_ref, _ = cv2.Rodrigues(np.asarray(rvec_ref, dtype=np.float32).reshape(3, 1))
        cos_a = np.clip((np.trace(R_new @ R_ref.T) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_a))) > max_angle_deg

    # -----------------------------------------------------
    # Tracking
    # -----------------------------------------------------
    def track(self, blobs: np.ndarray) -> Optional[Dict]:
        """
        State machine:

        Have prev_pose?
          Yes → proximity_match (fast locked path)
                  if error > 2.5 px or no result → brute_match fallback
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

        # Normalise prev_pose shapes
        if self.prev_pose is not None:
            rvec, tvec = self.prev_pose
            self.prev_pose = (
                np.asarray(rvec, dtype=np.float32).reshape(3, 1),
                np.asarray(tvec, dtype=np.float32).reshape(3),
            )

        # ------------------------------------------------------------------
        # Candidate search
        # ------------------------------------------------------------------
        solution = None

        # if self.prev_pose is not None:
        #     # --- Primary: proximity (fast, assignment-locked) ---
        #     if n_blobs >= 3:
        #         solution = self.proximity_match(
        #             blobs, self.prev_pose,
        #             prior_assignment=self.prev_assignment,
        #         )
        #
        #     # --- Fallback: brute when proximity is absent or degraded ---
        #     # Threshold 2.5 px: correct tracking gives < 0.5–1 px; values
        #     # above this indicate drifting or a wrong assignment lock.
        #     proximity_poor = solution is None or solution["error"] > 2.5
        #     if proximity_poor and n_blobs >= 4:
        #         brute = self.brute_match(blobs, pose_prior=self.prev_pose)
        #         if brute is not None:
        #             prox_n   = len(solution["assignment"]) if solution else 0
        #             brute_n  = brute.get("inliers", len(brute["assignment"]))
        #             brute_better = (
        #                 solution is None or
        #                 brute_n > prox_n or
        #                 (brute_n == prox_n and brute["error"] < solution["error"])
        #             )
        #             if brute_better:
        #                 solution = brute
        #
        #     elif n_blobs >= 1 and solution is None:
        #         solution = self.p1p_solver(blobs, self.prev_pose)

        # else:
        # --- No prior pose: brute-force re-acquisition ---
        if n_blobs >= 4:
            solution = self.brute_match(blobs)

            # In deep-debug mode frames are non-consecutive, so the last_good_pose
            # plausibility check is skipped — the controller can be anywhere.
            if solution is not None and self.last_good_pose is not None and not is_deep():
                rvec_lg, tvec_lg = self.last_good_pose
                if self._pose_jump_too_large(
                    solution["rvec"], solution["tvec"],
                    rvec_lg, tvec_lg,
                    max_dist_m=0.5,
                    max_angle_deg=60.0,
                ):
                    logger.debug("Brute re-acquisition rejected: too far from last known good pose.")
                    solution = None

        # # ------------------------------------------------------------------
        # # Pose-jump guard against prev_pose (tight, per-frame)
        # # ------------------------------------------------------------------
        # if solution is not None and self.prev_pose is not None:
        #     rvec_p, tvec_p = self.prev_pose
        #     if self._pose_jump_too_large(
        #         solution["rvec"], solution["tvec"],
        #         rvec_p, tvec_p,
        #     ):
        #         print(f"[tracking] Pose jump detected "
        #               f"(method={solution.get('method','?')}, "
        #               f"err={solution['error']:.2f} px) — "
        #               f"attempting brute recovery.")
        #         solution = None
        #         if n_blobs >= 4:
        #             brute = self.brute_match(blobs, pose_prior=self.prev_pose)
        #             if brute is not None and not self._pose_jump_too_large(
        #                 brute["rvec"], brute["tvec"], rvec_p, tvec_p,
        #             ):
        #                 solution = brute

        # ------------------------------------------------------------------
        # Accept / reject
        # ------------------------------------------------------------------
        if solution is not None and solution["error"] < 5.0:
            self.prev_pose       = (solution["rvec"], solution["tvec"])
            self.prev_assignment = solution["assignment"]
            self.last_good_pose       = self.prev_pose
            self.last_good_assignment = self.prev_assignment
            self.consecutive_failures = 0
            return solution

        # Tracking lost — keep last_good_pose for re-acquisition plausibility
        self.consecutive_failures += 1
        self.prev_pose       = None
        self.prev_assignment = None
        return None


# =========================================================
# 3. SYSTEM (multi-controller, multi-camera)
# =========================================================

class TrackingSystem:
    def __init__(self, controllers: List[ControllerModel], cameras: List[Camera]):

        self.trackers: Dict[Tuple[str, int], SingleViewTracker] = {}

        for ctrl in controllers:
            for cam in cameras:
                key = (ctrl.name, cam.camera_idx)
                self.trackers[key] = SingleViewTracker(cam, ctrl)

    def update(self, observations_per_camera: Dict[int, np.ndarray]):

        results = {}

        for (ctrl_name, cam_id), tracker in self.trackers.items():

            blobs = observations_per_camera.get(cam_id)

            if blobs is None:
                results[(ctrl_name, cam_id)] = None
                continue

            results[(ctrl_name, cam_id)] = tracker.track(blobs)

        return results


# class MultiViewFusion:
#     def fuse(self, poses_from_cameras):
#         # average / optimize / triangulate
#         return fused_pose
