import cv2
import numpy as np
from pathlib import Path


def get_centroids(image, cfg, visualize=False, img_path=None):
    """
    Detect LED blobs and return their intensity-weighted centroids.

    Detection strategy (inspired by blobwatch.c / Monado):

    Two-threshold approach
    ─────────────────────
    pixel_threshold (min_threshold):   minimum pixel value to be included in a
                                       blob at all — a low value allows collecting
                                       faint halos around bright LEDs.
    required_threshold (min_bright):   at least one pixel in the blob must reach
                                       this level; blobs that never exceed it are
                                       likely background noise and are dropped.
                                       Defaults to 2×min_threshold when omitted.

    Size filtering
    ──────────────
    Blobs are rejected if their bounding-box width or height exceeds max_wh
    (defaults to 35 px, matching blobwatch.c), or if their filled area falls
    outside [min_area, max_area].  Single-pixel blobs are always dropped.

    Centroid
    ────────
    Intensity-weighted centroid over all pixels inside the blob contour.
    Uses 1-based coordinates internally (matching blobwatch.c greysum) to
    prevent the first row/column from never contributing — the result is then
    shifted back to 0-based.
    """
    pixel_threshold   = int(cfg["min_threshold"])
    required_threshold = int(cfg.get("required_threshold", pixel_threshold * 2))
    min_area          = float(cfg["min_area"])
    max_area          = float(cfg["max_area"])
    max_wh            = int(cfg.get("max_wh", 35))   # blobwatch default

    # ── 1. Threshold at pixel_threshold ──────────────────────────────────────
    _, mask = cv2.threshold(image, pixel_threshold, 255, cv2.THRESH_BINARY)

    # ── 2. Find contours ─────────────────────────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    centroids = []
    blob_contours = []
    intensities = image.astype(np.float32)

    if visualize:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    idx = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # ── Reject by area ────────────────────────────────────────────────────
        if area < min_area or area > max_area:
            continue

        # ── Reject 1×1 blobs (pure noise) ────────────────────────────────────
        x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
        if w_b == 1 and h_b == 1:
            continue

        # ── Reject blobs that are too large to be LEDs ───────────────────────
        if w_b > max_wh or h_b > max_wh:
            continue

        # ── Build pixel mask for this blob ───────────────────────────────────
        blob_mask = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, -1)

        # ── Required-threshold gate: need at least one bright pixel ───────────
        # Equivalent to blobwatch.c blob_required_threshold check:
        # collects faint blobs but discards purely-dim background regions.
        roi_pixels = image[blob_mask > 0]
        if roi_pixels.size == 0 or int(roi_pixels.max()) < required_threshold:
            continue

        # ── Intensity-weighted centroid (greysum, 1-based coords) ─────────────
        weights = intensities * (blob_mask > 0)
        total_weight = np.sum(weights)
        if total_weight == 0:
            continue

        ys, xs = np.nonzero(blob_mask)

        # 1-based coordinates prevent the first row/column from never
        # contributing to the weighted sum (matches blobwatch.c greysum logic).
        cx = np.sum((xs + 1) * weights[ys, xs]) / total_weight - 1.0
        cy = np.sum((ys + 1) * weights[ys, xs]) / total_weight - 1.0

        centroids.append((cx, cy))
        blob_contours.append(cnt.reshape(-1, 2).astype(np.float32))

        if visualize:
            pt = (int(round(cx)), int(round(cy)))
            cv2.circle(vis, pt, 2, (255, 0, 0), -1)
            cv2.circle(vis, pt, 3, (255, 255, 255), 1)
            cv2.putText(
                vis,
                str(idx),
                (pt[0] + 10, pt[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        idx += 1

    centroids = np.array(centroids) if centroids else np.empty((0, 2), dtype=np.float32)
    # blob_contours: list of (M_i, 2) float32 pixel coordinate arrays, one per blob

    if visualize and img_path is not None:
        out_dir = Path("./visualization/blobs")
        out_dir.mkdir(parents=True, exist_ok=True)
        img_out_path = out_dir / f"{Path(img_path).stem}_blobs.png"
        cv2.imwrite(str(img_out_path), vis)

    return centroids, blob_contours
