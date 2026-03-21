import cv2
import numpy as np


def get_centroids(image, cfg):
    # --- 1. Threshold for detection (binary mask) ---
    _, mask = cv2.threshold(image, cfg["min_threshold"], 255, cv2.THRESH_BINARY)

    # --- 2. Find blobs ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filter small noise
        if area < cfg["min_area"] or area > cfg["max_area"]:
            continue

        # --- 3. Create mask for this blob ---
        blob_mask = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, -1)

        # --- 4. Weighted centroid (use original intensities!) ---
        intensities = image.astype(np.float32)
        weights = intensities * (blob_mask > 0)

        total_weight = np.sum(weights)
        if total_weight == 0:
            continue

        ys, xs = np.nonzero(blob_mask)

        cx = np.sum(xs * weights[ys, xs]) / total_weight
        cy = np.sum(ys * weights[ys, xs]) / total_weight

        print(f"Area: {area:.2f}, Center: ({cx:.2f}, {cy:.2f})")

        centroids.append((cx, cy))

    return centroids
