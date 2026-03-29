import numpy as np
from typing import List, Tuple, Optional, Dict

from src.camera import Camera
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
# 2. TRACKER (per camera + controller)
# =========================================================

class SingleViewTracker:
    def __init__(self, camera: Camera, model: ControllerModel):
        self.camera = camera
        self.model = model

        # Cached transform (Camera -> VRH IMU)
        self.T_imu_cam: Transform = camera.T_imu_cam

        # Tracking state
        self.prev_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.prev_assignment = None

        # Lazy cache (e.g. KD-tree later)
        self.kd_tree_cache = None

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
    # Tracking
    # -----------------------------------------------------
    def track(self, blobs: np.ndarray) -> Optional[Dict]:

        # blobs are in camera coordinates (pixels), shape (N, 2)
        blobs = np.asarray(blobs, dtype=np.float32).reshape(-1, 2)
        n_blobs = len(blobs)

        # Normalize previous pose format todo
        if self.prev_pose is not None:
            rvec, tvec = self.prev_pose
            self.prev_pose = (
                np.asarray(rvec, dtype=np.float32).reshape(3, 1),
                np.asarray(tvec, dtype=np.float32).reshape(3),
            )

        # -----------------------------
        # State machine
        # -----------------------------
        if self.prev_pose is None:
            if n_blobs >= 4:
                solution = self.brute_match(blobs)
            else:
                return None
        else:
            solution = None

            if n_blobs >= 3:
                solution = self.proximity_match(blobs, self.prev_pose)

                if not solution or solution["error"] > 12.0:
                    alt = self.p2p_solver(
                        blobs,
                        self.prev_pose,
                        proximity_result=solution
                    )
                    if alt and alt["error"] < 10.0:
                        solution = alt

            elif n_blobs >= 1:
                solution = self.p1p_solver(blobs, self.prev_pose)

        # -----------------------------
        # State update
        # -----------------------------
        if solution and solution["error"] < 15.0:
            self.prev_pose = (solution["rvec"], solution["tvec"])
            self.prev_assignment = solution["assignment"]
            return solution

        # Lost tracking
        self.prev_pose = None
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
