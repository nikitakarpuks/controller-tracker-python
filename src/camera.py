import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.transformations import Transform


def _cam_rvec(R_or_rvec: np.ndarray) -> np.ndarray:
    if R_or_rvec.shape == (3, 3):
        return cv2.Rodrigues(R_or_rvec)[0]
    return R_or_rvec.astype(np.float32).reshape(3, 1)


def kb4_theta_max(k1: float, k2: float, k3: float, k4: float) -> float:
    """Largest ray half-angle (radians) where the kb4 forward mapping
    rho(theta) = theta*(1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
    is still monotonically increasing (first zero of d(rho)/d(theta)).
    Beyond this angle the inverse mapping (pixel -> ray) is ambiguous.
    """
    def _drho(t):
        return 1 + 3*k1*t**2 + 5*k2*t**4 + 7*k3*t**6 + 9*k4*t**8

    if _drho(np.pi / 2) >= 0:
        return np.pi / 2  # monotonic all the way to 90°

    lo, hi = 0.0, np.pi / 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if _drho(mid) > 0: lo = mid
        else:              hi = mid
    return lo


def kb4_rpmax(k1: float, k2: float, k3: float, k4: float,
              fx: float, fy: float, margin: float = 0.99) -> tuple:
    """Returns (rpmax_tan, rpmax_px) for a kb4 camera.

    rpmax_tan = tan(theta_max)                     -- angular limit, dimensionless,
                matches the pinhole-radtan8 rpmax convention (used by
                src/_visibility.py's tan-scale in-frame check).
    rpmax_px  = rho(theta_max) * (fx+fy)/2 * margin -- TRUE pixel-space turning-point
                radius. rho(theta_max), not theta_max itself, is what maps to pixel
                radius: pixel_radius = f * rho(theta), so the max valid pixel radius
                is f * rho(theta_max), not f * theta_max.
    """
    theta_max = kb4_theta_max(k1, k2, k3, k4)
    rho_max   = theta_max * (1 + k1*theta_max**2 + k2*theta_max**4
                              + k3*theta_max**6 + k4*theta_max**8)
    rpmax_tan = float(np.tan(theta_max) * margin)
    rpmax_px  = float(rho_max * (fx + fy) / 2 * margin)
    return rpmax_tan, rpmax_px


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

            # cv2.fisheye expects D shaped (4, 1) float64
            self.dist_coeffs = np.array([[self.k1], [self.k2], [self.k3], [self.k4]], dtype=np.float64)

            # self.rpmax: angular limit tan(theta_max), dimensionless -- matches the
            # pinhole-radtan8 convention below, consumed by src/_visibility.py's
            # tan-scale in-frame check.
            # self.rpmax_px: TRUE pixel-space turning-point radius (where the kb4
            # polynomial rho(theta) stops being monotonic) -- consumed by
            # undistort_points' pixel-space clamp and visualization's boundary sampling.
            self.rpmax, self.rpmax_px = kb4_rpmax(self.k1, self.k2, self.k3, self.k4,
                                                   self.fx, self.fy)
        else:  # pinhole-radtan8
            self.p1   = intrinsics["p1"]
            self.p2   = intrinsics["p2"]
            self.k5   = intrinsics["k5"]
            self.k6   = intrinsics["k6"]
            self.rpmax = intrinsics["rpmax"]
            self.rpmax_px = 0.0  # not meaningful for radtan8 -- no such turning point
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

    def _kb4_undistort_points(self, flat: np.ndarray, P=None) -> np.ndarray:
        """Vectorised bisection inverse of the kb4 forward model rho(theta) = theta*(1 +
        k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8). Bisection is used instead of
        cv2.fisheye.undistortPoints because that solver's Newton-Raphson seed (theta_0 =
        r/f, ignoring the distortion factor) can overshoot into the non-monotonic region
        and diverge for pixels whose TRUE angle is still well inside theta_max. Bisection
        over [0, theta_max] is provably convergent since rho(theta) is monotonic there
        by construction.

        flat: (N,2) float64 pixel coords (already clamped to rpmax_px by the caller).
        Returns (N,2) normalised (or P-projected) coords.
        """
        # Normalise x/y by fx/fy separately (not a shared average focal length) --
        # matches cv2.fisheye.undistortPoints' own K^-1 step and keeps this exact
        # for cameras where fx != fy, however slightly.
        nx = (flat[:, 0] - self.cx) / self.fx
        ny = (flat[:, 1] - self.cy) / self.fy
        phi = np.arctan2(ny, nx)
        target_rho = np.hypot(nx, ny)

        theta_max = kb4_theta_max(self.k1, self.k2, self.k3, self.k4)
        lo = np.zeros_like(target_rho)
        hi = np.full_like(target_rho, theta_max)
        for _ in range(40):  # ~1e-12 rad precision on [0, pi/2]
            mid = (lo + hi) / 2
            rho_mid = mid * (1 + self.k1*mid**2 + self.k2*mid**4
                                + self.k3*mid**6 + self.k4*mid**8)
            go_hi = rho_mid < target_rho
            lo = np.where(go_hi, mid, lo)
            hi = np.where(go_hi, hi, mid)
        theta = (lo + hi) / 2

        tan_theta = np.tan(theta)
        xn = tan_theta * np.cos(phi)
        yn = tan_theta * np.sin(phi)

        if P is not None:
            pts = np.stack([xn, yn, np.ones_like(xn)], axis=1)
            return (P @ pts.T).T[:, :2]
        return np.stack([xn, yn], axis=1)

    def undistort_points(self, pts2d: np.ndarray, P=None):
        """(N,2) distorted → (N,2) normalised/undistorted. Dispatches to the correct OpenCV model.

        For fisheye (kb4): pixels beyond rpmax_px (the true turning point of the kb4
        polynomial, where its inverse mapping becomes ambiguous) are clamped onto the
        rpmax_px circle (preserving direction from the optical centre) before inverting,
        so callers always receive finite values. Inside rpmax_px, inversion uses a
        bisection solve (see _kb4_undistort_points) rather than cv2.fisheye.undistortPoints.
        Use cam.rpmax_px > 0 and a pixel-radius check if you need to know which points
        were out-of-range.
        """
        flat = pts2d.reshape(-1, 2).astype(np.float64)
        if self.is_fisheye and self.rpmax_px > 0:
            dx = flat[:, 0] - self.cx
            dy = flat[:, 1] - self.cy
            r  = np.sqrt(dx**2 + dy**2)
            beyond = r > self.rpmax_px
            if beyond.any():
                flat = flat.copy()
                scale = self.rpmax_px / r[beyond]
                flat[beyond, 0] = self.cx + dx[beyond] * scale
                flat[beyond, 1] = self.cy + dy[beyond] * scale

        if self.is_fisheye:
            P64 = P.astype(np.float64) if P is not None else None
            return self._kb4_undistort_points(flat, P64)
        else:
            inp = flat.astype(np.float32).reshape(-1, 1, 2)
            return cv2.undistortPoints(inp, self.camera_matrix, self.dist_coeffs, P=P).reshape(-1, 2)
