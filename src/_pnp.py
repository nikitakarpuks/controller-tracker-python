import cv2
import numpy as np
from typing import Optional, Tuple

# Pre-allocated identity matrices reused by _ransac_pnp.
_K_IDENTITY = np.eye(3, dtype=np.float64)
_DC_ZERO    = np.zeros(4, dtype=np.float64)


def _ransac_pnp(
    obj_pts: np.ndarray,
    pts_norm: np.ndarray,
    fx: float,
    rvec_init=None,
    tvec_init=None,
    reprojection_px: float = 2.0,
    iterations: int = 100,
    confidence: float = 0.99,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    RANSAC PnP over already-undistorted, normalised image coordinates.

    Ported from OpenHMD / Monado (ransac_pnp.cpp):
      • Callers undistort blob points with the real calibration (via
        Camera.undistort_points) before calling this function, so the RANSAC
        loop never touches distortion maths — faster and more numerically
        stable, and keeps a single, shared, correct undistortion path (see
        Camera.undistort_points' kb4 bisection inverse) instead of this
        function re-undistorting the same points via its own, separate call.
      • An identity K + zero distortion are passed to the solver to match
        the undistorted point space.
      • The reprojection threshold is converted from pixels to normalised
        units as reprojection_px / fx.
      • SOLVEPNP_SQPNP is used as the minimal solver (non-iterative,
        closed-form — more robust than ITERATIVE LM inside RANSAC).

    Parameters
    ----------
    obj_pts  : (N, 3) world points
    pts_norm : (N, 2) already-undistorted normalised image coordinates
               (Camera.undistort_points(pts2d), P=None)
    fx       : camera's fx, used only to convert reprojection_px to normalised units

    Returns
    -------
    ok          : bool
    rvec        : (3, 1) float64 or None
    tvec        : (3, 1) float64 or None
    inlier_idx  : 1-D int array indexing obj_pts/pts_norm, or None
    """
    if len(obj_pts) < 4:
        return False, None, None, None

    use_guess = rvec_init is not None and tvec_init is not None
    r0 = np.asarray(rvec_init, dtype=np.float64).reshape(3, 1) if use_guess else None
    t0 = np.asarray(tvec_init, dtype=np.float64).reshape(3, 1) if use_guess else None

    try:
        ret, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts.astype(np.float64),
            pts_norm.astype(np.float64),
            _K_IDENTITY, _DC_ZERO,
            r0, t0,
            useExtrinsicGuess=use_guess,
            iterationsCount=iterations,
            reprojectionError=reprojection_px / fx,
            confidence=confidence,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:
        return False, None, None, None

    if not ret or inliers is None or len(inliers) < 4:
        return False, None, None, None

    return True, rvec, tvec, inliers.flatten()


def _to_rvec(R_or_rvec: np.ndarray) -> np.ndarray:
    if R_or_rvec.shape == (3, 3):
        return cv2.Rodrigues(R_or_rvec)[0]
    return R_or_rvec.astype(np.float32).reshape(3, 1)


def _project_points(rvec, tvec, points: np.ndarray, K, dc, is_fisheye: bool = False) -> np.ndarray:
    """Project (N,3) world points → (N,2) image points."""
    r = _to_rvec(rvec)
    t = np.asarray(tvec, dtype=np.float32).reshape(3, 1)
    if is_fisheye:
        pts, _ = cv2.fisheye.projectPoints(
            points.astype(np.float64).reshape(-1, 1, 3),
            r.astype(np.float64), t.astype(np.float64),
            K.astype(np.float64), dc,
        )
    else:
        pts, _ = cv2.projectPoints(points.astype(np.float32), r, t, K, dc)
    return pts.reshape(-1, 2)


def _check_z_range(tvec_h: np.ndarray, z_min: float = 0.05, z_max: float = 15.0) -> bool:
    """Return True if the hypothesis depth is within plausible range (OpenHMD: 0.05–15 m)."""
    return z_min < tvec_h[2] < z_max
