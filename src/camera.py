import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.transformations import Transform


def _cam_rvec(R_or_rvec: np.ndarray) -> np.ndarray:
    if R_or_rvec.shape == (3, 3):
        return cv2.Rodrigues(R_or_rvec)[0]
    return R_or_rvec.astype(np.float32).reshape(3, 1)


class Camera:
    def __init__(self, cfg, camera_idx: int = 0, extrinsics_convention: str = "T_imu_cam"):
        """Camera calibration parameters.

        extrinsics_convention: "T_imu_cam"  — each entry is camera→IMU transform
                               "T_cam0_camN" — each entry is camera→cam0 transform (cam0 = identity)
        """
        self.camera_idx = camera_idx
        self.extrinsics_convention = extrinsics_convention

        resolution = cfg["value0"]["resolution"][camera_idx]
        self.width = resolution[0]
        self.height = resolution[1]

        cam_entry = cfg["value0"]["intrinsics"][camera_idx]
        self.camera_type = cam_entry.get("camera_type", "pinhole-radtan8")
        self.is_fisheye  = (self.camera_type == "kb4")

        intrinsics = cam_entry["intrinsics"]
        self.fx = intrinsics["fx"]
        self.fy = intrinsics["fy"]
        self.cx = intrinsics["cx"]
        self.cy = intrinsics["cy"]
        self.k1 = intrinsics["k1"]
        self.k2 = intrinsics["k2"]
        self.k3 = intrinsics["k3"]
        self.k4 = intrinsics["k4"]

        if self.is_fisheye:
            self.p1   = 0.0
            self.p2   = 0.0
            self.k5   = 0.0
            self.k6   = 0.0
            self.rpmax = 0.0
            # cv2.fisheye expects D shaped (4, 1) float64
            self.dist_coeffs = np.array([[self.k1], [self.k2], [self.k3], [self.k4]], dtype=np.float64)
        else:  # pinhole-radtan8
            self.p1   = intrinsics["p1"]
            self.p2   = intrinsics["p2"]
            self.k5   = intrinsics["k5"]
            self.k6   = intrinsics["k6"]
            self.rpmax = intrinsics["rpmax"]
            self.dist_coeffs = np.array([self.k1, self.k2, self.p1, self.p2,
                                         self.k3, self.k4, self.k5, self.k6],
                                        dtype=np.float32)

        self.camera_matrix = np.array([[self.fx, 0, self.cx],
                                       [0, self.fy, self.cy],
                                       [0, 0, 1]], dtype=np.float32)

        extrinsics = cfg["value0"]["T_imu_cam"][camera_idx]
        self.px = extrinsics["px"]
        self.py = extrinsics["py"]
        self.pz = extrinsics["pz"]
        self.qx = extrinsics["qx"]
        self.qy = extrinsics["qy"]
        self.qz = extrinsics["qz"]
        self.qw = extrinsics["qw"]

        # T_imu_cam: camera frame → IMU/world frame  (convention: T_target_source)
        R_imu_cam = R.from_quat([self.qx, self.qy, self.qz, self.qw]).as_matrix()
        t_imu_cam = np.array([self.px, self.py, self.pz])
        self.T_imu_cam: Transform = Transform(R_imu_cam, t_imu_cam)
        self.T_cam_imu: Transform = self.T_imu_cam.inverse()

    @property
    def T_world_cam(self) -> Transform:
        """World frame = IMU frame (T_imu_cam) or cam0 frame (T_cam0_camN). Transforms camera-frame points into world frame."""
        return self.T_imu_cam

    def project_points(self, pts3d: np.ndarray, rvec, tvec):
        """(N,3) → ((N,2) projected, jacobian). Dispatches to the correct OpenCV model."""
        r = _cam_rvec(rvec)
        t = np.asarray(tvec, dtype=np.float32).reshape(3, 1)
        if self.is_fisheye:
            pts, jac = cv2.fisheye.projectPoints(
                pts3d.astype(np.float64).reshape(-1, 1, 3),
                r.astype(np.float64), t.astype(np.float64),
                self.camera_matrix.astype(np.float64), self.dist_coeffs,
            )
        else:
            pts, jac = cv2.projectPoints(
                pts3d.astype(np.float32), r, t, self.camera_matrix, self.dist_coeffs,
            )
        return pts.reshape(-1, 2), jac

    def undistort_points(self, pts2d: np.ndarray, P=None):
        """(N,2) distorted → (N,2) normalised/undistorted. Dispatches to the correct OpenCV model."""
        inp = pts2d.astype(np.float32).reshape(-1, 1, 2)
        if self.is_fisheye:
            P64 = P.astype(np.float64) if P is not None else None
            return cv2.fisheye.undistortPoints(
                inp.astype(np.float64),
                self.camera_matrix.astype(np.float64), self.dist_coeffs,
                P=P64,
            ).reshape(-1, 2)
        else:
            return cv2.undistortPoints(inp, self.camera_matrix, self.dist_coeffs, P=P).reshape(-1, 2)
