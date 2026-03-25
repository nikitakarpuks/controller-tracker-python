import cv2
import numpy as np
from pathlib import Path


def get_centroids(image, cfg, visualize=False, img_path=None):
    # --- 1. Threshold ---
    _, mask = cv2.threshold(image, cfg["min_threshold"], 255, cv2.THRESH_BINARY)

    # --- 2. Find contours ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    intensities = image.astype(np.float32)

    # Visualization canvas
    if visualize:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    idx = 0  # correct indexing after filtering

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < cfg["min_area"] or area > cfg["max_area"]:
            continue

        # --- Blob mask ---
        blob_mask = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, -1)

        weights = intensities * (blob_mask > 0)
        total_weight = np.sum(weights)

        if total_weight == 0:
            continue

        ys, xs = np.nonzero(blob_mask)

        cx = np.sum(xs * weights[ys, xs]) / total_weight
        cy = np.sum(ys * weights[ys, xs]) / total_weight

        centroids.append((cx, cy))

        if visualize:
            pt = (int(cx), int(cy))

            # --- Draw centroid (visible on any background) ---
            cv2.circle(vis, pt, 2, (255, 0, 0), -1)      # red fill
            cv2.circle(vis, pt, 3, (255, 255, 255), 1)   # white outline

            # --- Draw index ---
            cv2.putText(
                vis,
                str(idx),
                (pt[0] + 20, pt[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        idx += 1

    centroids = np.array(centroids)

    # --- Save full image ---
    if visualize and img_path is not None:
        out_dir = Path("./visualization/blobs")
        out_dir.mkdir(parents=True, exist_ok=True)

        img_out_path = out_dir / f"{Path(img_path).stem}_blobs.png"
        cv2.imwrite(str(img_out_path), vis)

    return centroids
