import numpy as np
import itertools
import cv2


def solve_pose_and_error(object_points, image_points, K, dist):
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None, np.inf

    # Reproject
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    projected = projected.reshape(-1, 2)

    error = np.mean(np.linalg.norm(projected - image_points, axis=1))

    return rvec, tvec, error


def match_blobs_to_leds(blobs, leds_3d, K, dist):
    blobs = np.array(blobs, dtype=np.float32)
    leds_3d = np.array(leds_3d, dtype=np.float32)

    n_blobs = len(blobs)
    n_leds = len(leds_3d)

    best_error = np.inf
    best_solution = None

    # Choose subset of LEDs (if more LEDs than blobs)
    for led_indices in itertools.combinations(range(n_leds), n_blobs):
        selected_leds = leds_3d[list(led_indices)]

        # Try all permutations (assignments)
        for perm in itertools.permutations(range(n_blobs)):
            object_points = selected_leds[list(perm)]
            image_points = blobs

            rvec, tvec, error = solve_pose_and_error(
                object_points,
                image_points,
                K,
                dist
            )

            if error < best_error:
                best_error = error
                best_solution = {
                    "rvec": rvec,
                    "tvec": tvec,
                    "error": error,
                    "assignment": list(zip(range(n_blobs), np.array(led_indices)[list(perm)]))
                }

    return best_solution
