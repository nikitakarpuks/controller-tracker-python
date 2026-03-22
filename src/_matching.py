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

    # Project LEDs using predicted pose
    projected = self.project_leds_to_image(R_pred, tvec_pred)

    # Filter visible LEDs
    visible = [(idx, proj) for idx, proj, visible in projected if visible and proj is not None]

    if len(visible) < 3:
        return None

    # Build KD-tree for fast nearest neighbor matching
    led_positions = np.array([proj for _, proj in visible])
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

def brute_match(self, blobs: np.ndarray,
                min_matches: int = 4,
                reprojection_threshold: float = 5.0) -> Optional[Dict]:
    """
    Exhaustive matching when no prior pose exists

    Args:
        blobs: Detected blob centers (N x 2)
        min_matches: Minimum number of matches required
        reprojection_threshold: Maximum allowed reprojection error

    Returns:
        Best matching solution
    """
    n_blobs = len(blobs)
    n_leds = len(self.leds_3d)

    if n_blobs < min_matches:
        return None

    best_solution = None
    best_error = np.inf

    # Try all combinations of LEDs
    for led_indices in itertools.combinations(range(n_leds), n_blobs):
        selected_leds = [self.leds_3d[idx].position for idx in led_indices]

        # Try all permutations of assignments
        for perm in itertools.permutations(range(n_blobs)):
            object_points = np.array([selected_leds[idx] for idx in perm])
            image_points = blobs

            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.cam.camera_matrix,
                self.cam.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                continue

            # Reprojection error
            projected, _ = cv2.projectPoints(object_points, rvec, tvec,
                                             self.cam.camera_matrix, self.cam.dist_coeffs)
            projected = projected.reshape(-1, 2)
            error = np.mean(np.linalg.norm(projected - image_points, axis=1))

            if error < best_error and error < reprojection_threshold:
                best_error = error
                best_solution = {
                    "rvec": rvec,
                    "tvec": tvec,
                    "error": error,
                    "assignment": list(zip(range(n_blobs), np.array(led_indices)[list(perm)])),
                    "method": "brute"
                }

    return best_solution
