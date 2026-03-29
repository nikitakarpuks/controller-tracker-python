import cv2
import numpy as np
from scipy.spatial import KDTree
from typing import Tuple, Dict, Optional
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from src.transformations import Transform


def proximity_match(
    self,
    blobs: np.ndarray,
    predicted_pose: Tuple[np.ndarray, np.ndarray],
    max_distance_px: float = 30.0
) -> Optional[Dict]:

    rvec_pred, tvec_pred = predicted_pose

    rvec_pred = np.asarray(rvec_pred, dtype=np.float32).reshape(3, 1)
    tvec_pred = np.asarray(tvec_pred, dtype=np.float32).reshape(3)

    led_positions = self.model.positions

    # =====================================================
    # Option A: OpenCV projection (USED)
    # =====================================================
    projected, _ = cv2.projectPoints(
        led_positions,
        rvec_pred,
        tvec_pred,
        self.camera.camera_matrix,
        self.camera.dist_coeffs
    )

    projected = projected.reshape(-1, 2)

    if not np.all(np.isfinite(projected)):
        return None

    # =====================================================
    # Option B: Custom projection (DISABLED)
    # =====================================================
    # R_pred, _ = cv2.Rodrigues(rvec_pred)
    # T_cam_ctrl = Transform(R_pred, tvec_pred)
    # projected_full = self.project_leds_to_image(T_cam_ctrl)
    # projected = np.array([p for _, p, vis in projected_full if p is not None])

    # =====================================================
    # Optional: visibility filtering (recommended)
    # =====================================================
    R_pred, _ = cv2.Rodrigues(rvec_pred)

    normals_cam = (R_pred @ self.model.normals.T).T
    led_cam = (R_pred @ led_positions.T).T + tvec_pred

    view_dir = -led_cam / np.linalg.norm(led_cam, axis=1, keepdims=True)
    visible_mask = (normals_cam * view_dir).sum(axis=1) > 0.2

    # filter visible LEDs
    led_ids = np.where(visible_mask)[0]
    led_proj = projected[visible_mask]

    if len(led_proj) < 3:
        return None

    # =====================================================
    # KDTree matching
    # =====================================================
    tree = KDTree(led_proj)
    distances, matches = tree.query(blobs)

    valid = [
        (b_idx, led_ids[l_idx])
        for b_idx, (d, l_idx) in enumerate(zip(distances, matches))
        if d < max_distance_px
    ]

    if len(valid) < 3:
        return None

    # =====================================================
    # Hungarian refinement
    # =====================================================
    cost = cdist(blobs, led_proj)
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    for b_idx, l_idx in zip(row_ind, col_ind):
        if cost[b_idx, l_idx] < max_distance_px:
            matches.append((b_idx, led_ids[l_idx]))

    if len(matches) < 3:
        return None

    # =====================================================
    # Build PnP inputs
    # =====================================================
    object_points = self.model.positions[[lid for _, lid in matches]]
    image_points = blobs[[bid for bid, _ in matches]]

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        self.camera.camera_matrix,
        self.camera.dist_coeffs,
        rvec_pred,
        tvec_pred,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    # =====================================================
    # Error computation
    # =====================================================
    proj, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        self.camera.camera_matrix,
        self.camera.dist_coeffs
    )

    proj = proj.reshape(-1, 2)

    error = np.mean(np.linalg.norm(proj - image_points, axis=1))

    if error > max_distance_px:
        return None

    return {
        "rvec": rvec,
        "tvec": tvec,
        "error": float(error),
        "assignment": matches,
        "method": "proximity"
    }

def brute_match(
    self,
    blobs: np.ndarray,
    max_iterations: int = 2000,
    reprojection_threshold: float = 5.0,
    min_inliers: int = 5
) -> Optional[Dict]:
    """
    Brute-force RANSAC PnP matching without prior pose.
    This method randomly samples 4 blobs and 4 LEDs to estimate an initial pose,
    then counts inliers based on reprojection error, and finally refines the pose using all inliers.
    Args:
    - blobs: Detected blob positions in the image (shape: Nx2).
    - max_iterations: Number of RANSAC iterations.
    - reprojection_threshold: Maximum pixel distance for a blob to be considered an inlier.
    - min_inliers: Minimum number of inliers required to accept a solution.
    Returns:
    - A dictionary containing the best pose estimate (in camera coordinates), inlier count, error, and assignment
    """

    n_blobs = len(blobs)
    n_leds = len(self.model.positions)

    if n_blobs < 4:
        return None

    led_positions = self.model.positions

    best_solution = None
    best_inliers = 0
    best_error = np.inf

    for _ in range(max_iterations):

        # choosing blobs and leds randomly for initial PnP estimation
        blob_idx = np.random.choice(n_blobs, 4, replace=False)
        led_idx = np.random.choice(n_leds, 4, replace=False)

        obj = led_positions[led_idx]
        img = blobs[blob_idx]

        success, rvec, tvec = cv2.solvePnP(
            obj,
            img,
            self.camera.camera_matrix,
            self.camera.dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP
        )

        if not success:
            continue

        # project to camera
        proj, _ = cv2.projectPoints(
            led_positions, rvec, tvec,
            self.camera.camera_matrix,
            self.camera.dist_coeffs
        )

        proj = proj.reshape(-1, 2)

        if not np.all(np.isfinite(proj)):
            continue

        # nearest neighbours
        tree = KDTree(proj)
        distances, indices = tree.query(blobs)

        used = set()
        inlier_b = []
        inlier_l = []

        # extract inliers
        for b_idx, (d, l_idx) in enumerate(zip(distances, indices)):
            if d < reprojection_threshold and l_idx not in used:
                used.add(l_idx)
                inlier_b.append(b_idx)
                inlier_l.append(l_idx)

        if len(inlier_b) < min_inliers:
            continue

        inlier_b = np.array(inlier_b)
        inlier_l = np.array(inlier_l)

        proj_in = proj[inlier_l]

        # calculate error between projected inliers and blobs
        err = np.linalg.norm(proj_in - blobs[inlier_b], axis=1).mean()

        is_better = (
            len(inlier_b) > best_inliers or
            (len(inlier_b) == best_inliers and err < best_error)
        )

        if not is_better:
            continue

        # --- refine ---
        success, rvec_ref, tvec_ref = cv2.solvePnP(
            led_positions[inlier_l],
            blobs[inlier_b],
            self.camera.camera_matrix,
            self.camera.dist_coeffs,
            rvec, tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            continue

        proj_ref, _ = cv2.projectPoints(
            led_positions[inlier_l],
            rvec_ref, tvec_ref,
            self.camera.camera_matrix,
            self.camera.dist_coeffs
        )

        proj_ref = proj_ref.reshape(-1, 2)
        final_err = np.linalg.norm(
            proj_ref - blobs[inlier_b], axis=1
        ).mean()


        best_solution = {
            "rvec": rvec_ref,
            "tvec": tvec_ref,
            "inliers": len(inlier_b),
            "error": float(final_err),
            "assignment": list(zip(inlier_b.tolist(), inlier_l.tolist())),
            "method": "ransac_brute"
        }

        best_inliers = len(inlier_b)
        best_error = final_err

    return best_solution
