import cv2
import numpy as np
from pathlib import Path


def _mean_knn_distances(centroids, k):
    """For each point return mean Euclidean distance to its k nearest neighbors."""
    n = len(centroids)
    diff = centroids[:, None, :] - centroids[None, :, :]   # (n, n, 2)
    dists = np.sqrt((diff ** 2).sum(axis=-1))               # (n, n)
    mean_nn = np.empty(n, dtype=np.float32)
    for i in range(n):
        row = dists[i].copy()
        row[i] = np.inf                     # exclude self
        mean_nn[i] = np.sort(row)[:k].mean()
    return mean_nn


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

    Distance-based outlier filter
    ─────────────────────────────
    After detection, each blob's mean distance to its neighbor_k nearest
    neighbors is computed.  Blobs whose mean exceeds outlier_factor × median
    of all such means are dropped as spatially isolated noise.
    Skipped when fewer than neighbor_k + 1 blobs are detected.
    """
    pixel_threshold    = int(cfg["min_threshold"])
    required_threshold = int(cfg.get("required_threshold", pixel_threshold * 2))
    min_area           = float(cfg["min_area"])
    max_area           = float(cfg["max_area"])
    max_wh             = int(cfg.get("max_wh", 35))
    neighbor_k         = int(cfg.get("neighbor_k", 3))
    outlier_factor     = float(cfg.get("outlier_factor", 3.0))

    # ── 1. Threshold at pixel_threshold ──────────────────────────────────────
    _, mask = cv2.threshold(image, pixel_threshold, 255, cv2.THRESH_BINARY)

    # ── 2. Find contours ─────────────────────────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    centroids = []
    blob_contours = []
    intensities = image.astype(np.float32)

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

    centroids_arr = np.array(centroids, dtype=np.float32) if centroids else np.empty((0, 2), dtype=np.float32)

    # ── 3. Distance-based outlier filter ─────────────────────────────────────
    keep_mask = None
    if len(centroids_arr) > neighbor_k:
        mean_nn = _mean_knn_distances(centroids_arr, neighbor_k)
        keep_mask = mean_nn <= outlier_factor * np.median(mean_nn)

    kept_indices     = np.where(keep_mask)[0] if keep_mask is not None else np.arange(len(centroids_arr))
    rejected_indices = np.where(~keep_mask)[0] if keep_mask is not None else np.array([], dtype=int)
    filtered_centroids  = centroids_arr[kept_indices]
    filtered_contours   = [blob_contours[i] for i in kept_indices]
    rejected_centroids  = centroids_arr[rejected_indices]
    rejected_contours   = [blob_contours[i] for i in rejected_indices]

    # ── 4. Visualization ──────────────────────────────────────────────────────
    if visualize:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Rejected blobs in red
        if keep_mask is not None:
            for i in np.where(~keep_mask)[0]:
                pt = (int(round(centroids_arr[i, 0])), int(round(centroids_arr[i, 1])))
                cv2.circle(vis, pt, 4, (0, 0, 255), 1)

        # Kept blobs
        for idx, pt_f in enumerate(filtered_centroids):
            pt = (int(round(pt_f[0])), int(round(pt_f[1])))
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

        if img_path is not None:
            out_dir = Path("./visualization/blobs")
            out_dir.mkdir(parents=True, exist_ok=True)
            img_out_path = out_dir / f"{Path(img_path).stem}_blobs.png"
            cv2.imwrite(str(img_out_path), vis)

    return filtered_centroids, filtered_contours, rejected_centroids, rejected_contours
