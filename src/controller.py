import numpy as np
from typing import List, Tuple, Optional, Dict
from src.camera import CameraCalibration


def create_leds_from_config(cfg):
    """Create list of ControllerLED instances from config data"""
    calibration_information = cfg["CalibrationInformation"]["ControllerLeds"]
    leds = []
    for led in calibration_information:
        # position = np.array([led["Position"]]).T
        # normal = np.array([led["Normal"]]).T

        position = np.array(led["Position"], dtype=np.float32).reshape(3, )
        normal = np.array(led["Normal"], dtype=np.float32).reshape(3, )

        leds.append(ControllerLED(position, normal))
    return leds


class ControllerLED:
    def __init__(self, position: np.ndarray, normal: np.ndarray):
        """LED on the controller with position and normal"""

        self.position = position # 3D position relative to controller center
        self.normal = normal # LED orientation vector


    # def __le__(self, other):
    #     """Less than or equal comparison"""
    #     if not isinstance(other, ControllerLED):
    #         return NotImplemented
    #     return tuple(self.position) <= tuple(other.position)


class ControllerTracker:
    def __init__(self, camera_calib: CameraCalibration,
                 leds_3d: List[ControllerLED]):
        """
        Initialize tracker

        Args:
            camera_calib: Camera calibration parameters
            leds_3d: List of LED positions and normals in controller space
            camera_pose: (rotation, translation) of camera in world space
        """

        self.cam = camera_calib
        self.leds_3d = leds_3d

        self.cam_R, self.cam_t = camera_calib.camera_pose  # headset imu to camera
        self.cam_R = np.asarray(self.cam_R, dtype=np.float32).reshape(3, 3)
        self.cam_t = np.asarray(self.cam_t, dtype=np.float32).reshape(3, )

        # For temporal tracking
        self.prev_pose = None  # Previous controller pose (rvec, tvec)
        self.prev_assignment = None  # Previous LED-to-blob mapping
        self.kd_tree_cache = None  # For proximity matching

        from src._matching import proximity_match, brute_match
        from src._pnp_solver import p2p_solver, p1p_solver
        self.proximity_match = proximity_match.__get__(self)
        self.brute_match = brute_match.__get__(self)
        self.p2p_solver = p2p_solver.__get__(self)
        self.p1p_solver = p1p_solver.__get__(self)


    def project_leds_to_image(self, controller_R: np.ndarray,
                              controller_t: np.ndarray) -> List[Tuple[int, np.ndarray, bool]]:
        """
        Project LEDs from controller frame into camera image.

        Args:
            controller_R: Rotation from controller -> IMU(world)
            controller_t: Translation from controller -> IMU(world)

        Returns:
            List of (led_index, projected_point (2D), is_visible)
        """
        projected = []

        controller_t = np.asarray(controller_t, dtype=np.float32).reshape(3, )
        controller_R = np.asarray(controller_R, dtype=np.float32).reshape(3, 3)

        assert controller_t.shape == (3,)
        assert self.cam_t.shape == (3,)

        for idx, led in enumerate(self.leds_3d):
            # --- Controller -> IMU (world) ---
            led_imu = controller_R @ led.position + controller_t

            # --- IMU -> Camera ---
            led_cam = self.cam_R @ led_imu + self.cam_t

            z = float(led_cam[2])

            # Reject points behind or too close to camera
            if z <= 1e-6:
                projected.append((idx, None, False))
                continue

            # Project to image plane (pinhole model)
            x = led_cam[0] / led_cam[2]
            y = led_cam[1] / led_cam[2]

            # Apply distortion (radtan8 model)
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r2 * r4

            # Radial distortion
            radial = (1 + self.cam.k1 * r2 + self.cam.k2 * r4 + self.cam.k3 * r6) / \
                     (1 + self.cam.k4 * r2 + self.cam.k5 * r4 + self.cam.k6 * r6)

            # Tangential distortion
            dx = 2 * self.cam.p1 * x * y + self.cam.p2 * (r2 + 2 * x * x)
            dy = self.cam.p1 * (r2 + 2 * y * y) + 2 * self.cam.p2 * x * y

            x_dist = x * radial + dx
            y_dist = y * radial + dy

            # Convert to pixel coordinates
            u = self.cam.fx * x_dist + self.cam.cx
            v = self.cam.fy * y_dist + self.cam.cy

            # --- Visibility check (LED normal) ---
            # Controller -> IMU -> Camera
            normal_imu = controller_R @ led.normal
            normal_cam = self.cam_R @ normal_imu

            view_dir = -led_cam / np.linalg.norm(led_cam)

            # Slightly relaxed threshold is safer in practice
            is_visible = normal_cam.T @ view_dir > 0.2

            projected.append((idx, np.array([u, v]), is_visible))

        return projected


    def track(self, blobs: np.ndarray) -> Optional[Dict]:
        """
        Main tracking function with state machine for selecting appropriate solver

        Args:
            blobs: Detected blob centers (N x 2)

        Returns:
            Pose solution or None if tracking lost
        """

        blobs = np.asarray(blobs, dtype=np.float32).reshape(-1, 2)

        n_blobs = len(blobs)

        if self.prev_pose is not None:
            rvec, tvec = self.prev_pose
            self.prev_pose = (
                np.asarray(rvec, dtype=np.float32).reshape(3, 1),  # OpenCV expects (3,1)
                np.asarray(tvec, dtype=np.float32).reshape(3, )
            )

        # State machine based on number of blobs and prior information
        if self.prev_pose is None:
            # No prior pose - use brute force matching
            if n_blobs >= 4:
                solution = self.brute_match(blobs)
            else:
                return None
        else:
            # Have prior pose - try progressive solvers
            solution = None

            if n_blobs >= 3:
                # Try proximity matching first
                solution = self.proximity_match(blobs, self.prev_pose)

                # If that fails, try P2P
                if not solution or solution["error"] > 12.0:
                    p2p_solution = self.p2p_solver(blobs, self.prev_pose, proximity_result=solution)
                    if p2p_solution and p2p_solution["error"] < 10.0:
                        solution = p2p_solution

            # elif n_blobs >= 2:
            #     # Try P2P
            #     solution = self.p2p_solver(blobs, self.prev_pose)

            elif n_blobs >= 1:
                # Try P1P as last resort
                solution = self.p1p_solver(blobs, self.prev_pose)

        # Update tracking state
        if solution and solution["error"] < 15.0:
            self.prev_pose = (solution["rvec"], solution["tvec"])
            self.prev_assignment = solution["assignment"]
            return solution
        else:
            # Tracking lost - reset
            self.prev_pose = None
            self.prev_assignment = None
            return None
