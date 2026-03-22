import numpy as np
import cv2

class CameraCalibration:
    def __init__(self, cfg, camera_idx: int = 0):
        """Camera calibration parameters"""
        intrinsics = cfg["value0"]["intrinsics"][camera_idx]["intrinsics"]
        self.fx = intrinsics["fx"]
        self.fy = intrinsics["fy"]
        self.cx = intrinsics["cx"]
        self.cy = intrinsics["cy"]
        self.k1 = intrinsics["k1"]
        self.k2 = intrinsics["k2"]
        self.p1 = intrinsics["p1"]
        self.p2 = intrinsics["p2"]
        self.k3 = intrinsics["k3"]
        self.k4 = intrinsics["k4"]
        self.k5 = intrinsics["k5"]
        self.k6 = intrinsics["k6"]
        self.rpmax = intrinsics["rpmax"]

        extrinsics = cfg["value0"]["T_imu_cam"][camera_idx]
        self.px = extrinsics["px"]
        self.py = extrinsics["py"]
        self.pz = extrinsics["pz"]
        self.qx = extrinsics["qx"]
        self.qy = extrinsics["qy"]
        self.qz = extrinsics["qz"]
        # self.qw = extrinsics["qw"]

        self.camera_matrix = self.camera_matrix()
        self.dist_coeffs = self.dist_coeffs()
        self.camera_pose = self.camera_pose()

    def camera_matrix(self):
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float32)

    def dist_coeffs(self):
        # radtan8 model: [k1, k2, p1, p2, k3, k4, k5, k6]
        return np.array([self.k1, self.k2, self.p1, self.p2,
                         self.k3, self.k4, self.k5, self.k6], dtype=np.float32)

    def camera_pose(self):
        # camera relative to IMU
        cam_R = cv2.Rodrigues(np.array([self.qx, self.qy, self.qz]))[0] # performs a change of basis from world to camera coordinate system
        cam_t = np.array([self.px, self.py, self.pz])
        return cam_R, cam_t
