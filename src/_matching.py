import cv2
import numpy as np
# import KDTree
# from sklearn.neighbors import KDTree
from scipy.spatial import KDTree
from typing import Tuple, Dict, Optional
import itertools

def proximity_match(self, blobs: np.ndarray,
                    predicted_pose: Tuple[np.ndarray, np.ndarray],
                    max_distance_px: float = 30.0) -> Optional[Dict]:
    """
    Match blobs to LEDs using prior pose information (proximity matching)

    Args:
        blobs: Detected blob centers in image coordinates (N x 2)
        predicted_pose: (rvec, tvec) predicted controller pose
        max_distance_px: Maximum allowed reprojection error in pixels

    Returns:
        Matching solution or None if insufficient matches
    """
    rvec_pred, tvec_pred = predicted_pose
    R_pred = cv2.Rodrigues(rvec_pred)[0]

    R_cam_ctrl = R_pred
    t_cam_ctrl = tvec_pred

    # Invert camera extrinsics
    R_imu_cam = self.cam_R.T
    t_imu_cam = -R_imu_cam @ self.cam_t

    # Compose
    R_imu_ctrl = R_imu_cam @ R_cam_ctrl
    t_imu_ctrl = R_imu_cam @ t_cam_ctrl + t_imu_cam

    # Project LEDs using predicted pose
    projected = self.project_leds_to_image(R_imu_ctrl, t_imu_ctrl)

    # Filter visible LEDs
    visible = [(idx, proj) for idx, proj, visible in projected if visible and proj is not None]

    if len(visible) < 3:
        return None

    # Build KD-tree for fast nearest neighbor matching
    led_positions = np.array([proj[:, 0] for _, proj in visible])
    led_indices = [idx for idx, _ in visible]
    tree = KDTree(led_positions)

    # Find nearest LED for each blob
    distances, matches = tree.query(blobs)

    # Only keep matches within max_distance_px
    valid_matches = [(blob_idx, led_indices[led_idx])
                     for blob_idx, (dist, led_idx) in enumerate(zip(distances, matches))
                     if dist < max_distance_px]

    if len(valid_matches) < 3:
        return None

    # Check for uniqueness (one LED per blob)
    used_leds = set()
    unique_matches = []
    for blob_idx, led_idx in valid_matches:
        if led_idx not in used_leds:
            used_leds.add(led_idx)
            unique_matches.append((blob_idx, led_idx))

    if len(unique_matches) < 3:
        return None

    # Prepare for PnP
    object_points = np.array([self.leds_3d[led_idx].position for _, led_idx in unique_matches])
    image_points = np.array([blobs[blob_idx] for blob_idx, _ in unique_matches])

    # Solve pose with prior as initial guess
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        self.cam.camera_matrix,
        self.cam.dist_coeffs,
        rvec_pred,
        tvec_pred,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    # Compute reprojection error
    projected_pts, _ = cv2.projectPoints(object_points, rvec, tvec,
                                         self.cam.camera_matrix, self.cam.dist_coeffs)
    projected_pts = projected_pts.reshape(-1, 2)
    error = np.mean(np.linalg.norm(projected_pts - image_points, axis=1))

    if error > max_distance_px:
        return None

    return {
        "rvec": rvec,
        "tvec": tvec,
        "error": error,
        "assignment": unique_matches,
        "method": "proximity"
    }

def brute_match(self,
                blobs: np.ndarray,
                max_iterations: int = 2000,
                reprojection_threshold: float = 5.0,
                min_inliers: int = 5) -> Optional[Dict]:
    """
    RANSAC-style brute matching (no prior pose)

    - Uses small hypothesis (4 points)
    - Validates using all blobs
    - Selects best based on inliers + reprojection error
    """

    n_blobs = len(blobs)
    n_leds = len(self.leds_3d)

    if n_blobs < 4:
        return None

    led_positions = np.array([led.position for led in self.leds_3d])

    best_solution = None
    best_inliers = 0
    best_error = np.inf

    for iteration in range(max_iterations):

        # --- 1. Sample hypothesis (4-point) ---
        blob_indices = np.random.choice(n_blobs, 4, replace=False)
        led_indices = np.random.choice(n_leds, 4, replace=False)

        object_points = led_positions[led_indices]
        image_points = blobs[blob_indices]

        # --- 2. Solve PnP (stable method) ---
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.cam.camera_matrix,
            self.cam.dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP
        )

        if not success:
            continue

        # --- 3. Project all LEDs ---
        projected, _ = cv2.projectPoints(
            led_positions, rvec, tvec,
            self.cam.camera_matrix,
            self.cam.dist_coeffs
        )
        projected = projected.reshape(-1, 2)

        # --- NaN / Inf safety ---
        if not np.all(np.isfinite(projected)):
            continue

        # --- 4. Match blobs to nearest projected LEDs ---
        tree = KDTree(projected)
        distances, indices = tree.query(blobs)

        # --- 5. Inlier selection with uniqueness ---
        used_leds = set()
        inlier_blob_idx = []
        inlier_led_idx = []

        for blob_idx, (dist, led_idx) in enumerate(zip(distances, indices)):
            if dist < reprojection_threshold and led_idx not in used_leds:
                used_leds.add(led_idx)
                inlier_blob_idx.append(blob_idx)
                inlier_led_idx.append(led_idx)

        num_inliers = len(inlier_blob_idx)

        if num_inliers < min_inliers:
            continue

        inlier_blob_idx = np.array(inlier_blob_idx)
        inlier_led_idx = np.array(inlier_led_idx)

        # --- 6. Compute reprojection error on inliers ---
        proj_inliers, _ = cv2.projectPoints(
            led_positions[inlier_led_idx],
            rvec, tvec,
            self.cam.camera_matrix,
            self.cam.dist_coeffs
        )
        proj_inliers = proj_inliers.reshape(-1, 2)

        errors = np.linalg.norm(proj_inliers - blobs[inlier_blob_idx], axis=1)
        mean_error = np.mean(errors)

        # --- 7. Check if this is best solution ---
        is_better = (
            (num_inliers > best_inliers) or
            (num_inliers == best_inliers and mean_error < best_error)
        )

        if not is_better:
            continue

        # --- 8. Refine pose using all inliers ---
        success, rvec_refined, tvec_refined = cv2.solvePnP(
            led_positions[inlier_led_idx],
            blobs[inlier_blob_idx],
            self.cam.camera_matrix,
            self.cam.dist_coeffs,
            rvec, tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            continue

        # --- 9. Final error after refinement ---
        proj_refined, _ = cv2.projectPoints(
            led_positions[inlier_led_idx],
            rvec_refined, tvec_refined,
            self.cam.camera_matrix,
            self.cam.dist_coeffs
        )
        proj_refined = proj_refined.reshape(-1, 2)

        final_errors = np.linalg.norm(
            proj_refined - blobs[inlier_blob_idx], axis=1
        )
        final_mean_error = np.mean(final_errors)

        # --- 10. Update best solution ---
        best_inliers = num_inliers
        best_error = final_mean_error

        best_solution = {
            "rvec": rvec_refined,
            "tvec": tvec_refined,
            "inliers": num_inliers,
            "error": final_mean_error,
            "assignment": list(zip(inlier_blob_idx, inlier_led_idx)),
            "method": "ransac_brute"
        }

    return best_solution
