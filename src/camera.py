import numpy as np
from scipy.spatial.transform import Rotation as R

from src.transformations import Transform


class Camera:
    def __init__(self, cfg, camera_idx: int = 0):
        """Camera calibration parameters"""

        self.camera_idx = camera_idx

        resolution = cfg["value0"]["resolution"][camera_idx]
        self.width = resolution[0]
        self.height = resolution[1]

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
        self.qw = extrinsics["qw"]

        self.camera_matrix = np.array([[self.fx, 0, self.cx],
                                       [0, self.fy, self.cy],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.array([self.k1, self.k2, self.p1, self.p2,
                                     self.k3, self.k4, self.k5, self.k6],
                                    dtype=np.float32)

        # T_imu_cam: camera frame → IMU/world frame  (convention: T_target_source)
        R_imu_cam = R.from_quat([self.qx, self.qy, self.qz, self.qw]).as_matrix()
        t_imu_cam = np.array([self.px, self.py, self.pz])
        self.T_imu_cam: Transform = Transform(R_imu_cam, t_imu_cam)
        self.T_cam_imu: Transform = self.T_imu_cam.inverse()

    @property
    def T_world_cam(self) -> Transform:
        """World frame = IMU frame. Transforms camera-frame points into world frame."""
        return self.T_imu_cam