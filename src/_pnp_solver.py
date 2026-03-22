import cv2
import numpy as np
from typing import Tuple, Dict, Optional


def p2p_solver(self, blobs: np.ndarray,
               prior_pose: Tuple[np.ndarray, np.ndarray],
               prior_error: float = None) -> Optional[Dict]:
    """
    P2P solver - uses 2 matches with prior orientation information
    Suitable when only 2-3 LEDs are visible

    Args:
        blobs: Detected blob centers (N x 2)
        prior_pose: (rvec, tvec) from previous frame
        prior_error: Reprojection error of prior pose

    Returns:
        Pose solution with at least 3 matches total
    """
    if len(blobs) < 2:
        return None

    rvec_prior, tvec_prior = prior_pose
    best_solution = None
    best_error = np.inf

    # Use proximity matching to find good matches
    proximity_result = self.proximity_match(blobs, prior_pose)
    if not proximity_result or len(proximity_result["assignment"]) < 2:
        return None

    # Get the 2 best matches from proximity matching
    matches = proximity_result["assignment"]
    matches_sorted = sorted(matches, key=lambda x: abs(x[0]))  # Sort by blob index

    # Try all pairs of matches as the hypothesis-generating set
    for i in range(len(matches_sorted) - 1):
        for j in range(i + 1, len(matches_sorted)):
            pair = [matches_sorted[i], matches_sorted[j]]

            # Use these 2 matches as hard constraints
            object_points = np.array([self.leds_3d[led_idx].position for _, led_idx in pair])
            image_points = np.array([blobs[blob_idx] for blob_idx, _ in pair])

            # Solve with prior orientation
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.cam.camera_matrix,
                self.cam.dist_coeffs,
                rvec_prior,
                tvec_prior,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                continue

            # Validate with remaining matches
            remaining = [m for m in matches_sorted if m not in pair]
            if remaining:
                remaining_obj = np.array([self.leds_3d[led_idx].position for _, led_idx in remaining])
                remaining_img = np.array([blobs[blob_idx] for blob_idx, _ in remaining])

                projected, _ = cv2.projectPoints(remaining_obj, rvec, tvec,
                                                 self.cam.camera_matrix, self.cam.dist_coeffs)
                projected = projected.reshape(-1, 2)
                errors = np.linalg.norm(projected - remaining_img, axis=1)
                mean_error = np.mean(errors)

                if mean_error < best_error and mean_error < 10.0:
                    best_error = mean_error
                    best_solution = {
                        "rvec": rvec,
                        "tvec": tvec,
                        "error": mean_error,
                        "assignment": pair + remaining,
                        "method": "p2p"
                    }

    return best_solution


def p1p_solver(self, blobs: np.ndarray,
               prior_pose: Tuple[np.ndarray, np.ndarray]) -> Optional[Dict]:
    """
    P1P solver - uses single match with prior pose for translation-only optimization
    Suitable when only 1-2 LEDs are visible

    Args:
        blobs: Detected blob centers (N x 2)
        prior_pose: (rvec, tvec) from previous frame

    Returns:
        Pose solution optimized for translation
    """
    if len(blobs) < 1:
        return None

    rvec_prior, tvec_prior = prior_pose
    R_prior = cv2.Rodrigues(rvec_prior)[0]

    # Use proximity matching to find best single match
    proximity_result = self.proximity_match(blobs, prior_pose)
    if not proximity_result:
        return None

    # Use the best match
    best_match = proximity_result["assignment"][0]
    blob_idx, led_idx = best_match

    led_pos = self.leds_3d[led_idx].position
    blob_pos = blobs[blob_idx]

    # Optimize only translation using the single correspondence
    # Keep orientation fixed from prior

    # Project LED using prior pose to get initial guess
    led_world = R_prior @ led_pos + tvec_prior
    led_cam = self.cam_R @ led_world + self.cam_t

    if led_cam[2] <= 0:
        return None

    # Solve for translation along the ray
    # This is a simplified stereo pose optimization as mentioned in the article

    # For now, use the prior pose and validate
    # In practice, you'd implement a proper translation-only optimization
    object_points = np.array([led_pos])
    image_points = np.array([blob_pos])

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        self.cam.camera_matrix,
        self.cam.dist_coeffs,
        rvec_prior,
        tvec_prior,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    # Validate with reprojection
    projected, _ = cv2.projectPoints(object_points, rvec, tvec,
                                     self.cam.camera_matrix, self.cam.dist_coeffs)
    error = np.linalg.norm(projected.reshape(-1, 2) - image_points)

    if error < 15.0:  # Higher threshold for P1P
        return {
            "rvec": rvec,
            "tvec": tvec,
            "error": error,
            "assignment": [best_match],
            "method": "p1p"
        }

    return None
