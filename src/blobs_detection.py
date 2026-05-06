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


def _find_split_maxima(image, ys, xs, peak_threshold, min_split_dist):
    """
    Find local maxima within a blob (pixel coords given by ys, xs).

    A pixel is a local maximum if its value is >= all 8-connected neighbors
    AND >= peak_threshold.  NMS then suppresses weaker maxima within
    min_split_dist pixels of a stronger one.  Returns a list of (y, x).
    """
    vals = image[ys, xs].astype(np.int32)
    is_max = vals >= peak_threshold

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            ny = np.clip(ys + dy, 0, image.shape[0] - 1)
            nx = np.clip(xs + dx, 0, image.shape[1] - 1)
            is_max &= vals >= image[ny, nx].astype(np.int32)

    if not np.any(is_max):
        return []

    max_ys  = ys[is_max]
    max_xs  = xs[is_max]
    max_vals = vals[is_max]

    # NMS: process strongest first, suppress within min_split_dist
    order = np.argsort(max_vals)[::-1]
    kept = []
    for i in order:
        y, x = int(max_ys[i]), int(max_xs[i])
        if all(np.hypot(y - ky, x - kx) >= min_split_dist for ky, kx in kept):
            kept.append((y, x))

    return kept


def _saddle_min(image, seed_a, seed_b):
    """Return the minimum pixel value along the straight line between two seed points."""
    y0, x0 = seed_a
    y1, x1 = seed_b
    n = max(abs(y1 - y0), abs(x1 - x0)) + 1
    ys = np.round(np.linspace(y0, y1, n)).astype(int)
    xs = np.round(np.linspace(x0, x1, n)).astype(int)
    ys = np.clip(ys, 0, image.shape[0] - 1)
    xs = np.clip(xs, 0, image.shape[1] - 1)
    return int(image[ys, xs].min())


def _split_blob_at_seeds(image, intensities, ys, xs, seed_a, seed_b):
    """
    Split blob pixels by Voronoi partition around seed_a and seed_b (y, x tuples).
    Returns two ((cx, cy), contour) pairs, or None if either partition is empty.
    """
    dist_a = (ys - seed_a[0]) ** 2 + (xs - seed_a[1]) ** 2
    dist_b = (ys - seed_b[0]) ** 2 + (xs - seed_b[1]) ** 2
    part_masks = [dist_a <= dist_b, dist_a > dist_b]

    results = []
    for part_bool in part_masks:
        p_ys = ys[part_bool]
        p_xs = xs[part_bool]
        if len(p_ys) == 0:
            return None

        w = intensities[p_ys, p_xs]
        total_w = w.sum()
        if total_w > 0:
            pcx = float(np.sum((p_xs + 1) * w) / total_w) - 1.0
            pcy = float(np.sum((p_ys + 1) * w) / total_w) - 1.0
        else:
            pcx = float(p_xs.mean())
            pcy = float(p_ys.mean())

        part_img = np.zeros(image.shape[:2], dtype=np.uint8)
        part_img[p_ys, p_xs] = 255
        cnts, _ = cv2.findContours(part_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        else:
            cnt = np.column_stack([p_xs, p_ys]).astype(np.float32)

        results.append(((pcx, pcy), cnt))

    return results


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

    Merged-blob splitting (optional)
    ────────────────────────────────
    Enabled with split_merged=true.  For each blob, local maxima are found
    among its pixels (pixels >= all 8 neighbors AND >= required_threshold).
    NMS suppresses weaker maxima within min_split_dist pixels of a stronger
    one.  If exactly 2 maxima survive, the blob is split: pixels are assigned
    to the nearest seed by Voronoi partition, producing two centroids and two
    contours.  Blobs with 1 maximum are kept as-is.  Blobs with 3+ maxima are
    also kept as-is (treated as noise rather than 3 merged LEDs).
    Split blobs are drawn in cyan with their partition contours visible.
    """
    pixel_threshold    = int(cfg["min_threshold"])
    required_threshold = int(cfg.get("required_threshold", pixel_threshold * 2))
    min_area           = float(cfg["min_area"])
    max_area           = float(cfg["max_area"])
    max_wh             = int(cfg.get("max_wh", 35))
    neighbor_k         = int(cfg.get("neighbor_k", 3))
    outlier_factor     = float(cfg.get("outlier_factor", 3.0))
    min_split_dist     = float(cfg.get("min_split_dist", 4.0))
    split_valley_ratio = float(cfg.get("split_valley_ratio", 0.6))

    # ── 1. Threshold at pixel_threshold ──────────────────────────────────────
    _, mask = cv2.threshold(image, pixel_threshold, 255, cv2.THRESH_BINARY)

    # ── 2. Find contours ─────────────────────────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    centroids = []
    blob_contours = []
    blob_pixels_list = []
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
        blob_pixels_list.append((ys, xs))

    # ── 2b. Merged-blob splitting ─────────────────────────────────────────────
    new_centroids     = []
    new_blob_contours = []
    blob_is_split = []

    for i in range(len(centroids)):
        ys_b, xs_b = blob_pixels_list[i]
        maxima = _find_split_maxima(image, ys_b, xs_b, required_threshold, min_split_dist)

        if len(maxima) == 2:
            peak_lower = min(int(image[maxima[0][0], maxima[0][1]]),
                             int(image[maxima[1][0], maxima[1][1]]))
            saddle = _saddle_min(image, maxima[0], maxima[1])
            valley_ok = peak_lower > 0 and saddle / peak_lower < split_valley_ratio
            if not valley_ok:
                new_centroids.append(centroids[i])
                new_blob_contours.append(blob_contours[i])
                blob_is_split.append(False)
                continue
            split_result = _split_blob_at_seeds(image, intensities, ys_b, xs_b, maxima[0], maxima[1])
            if split_result is not None:
                for (pcx, pcy), part_cnt in split_result:
                    new_centroids.append((pcx, pcy))
                    new_blob_contours.append(part_cnt)
                    blob_is_split.append(True)
                continue

        new_centroids.append(centroids[i])
        new_blob_contours.append(blob_contours[i])
        blob_is_split.append(False)

    centroids     = new_centroids
    blob_contours = new_blob_contours

    centroids_arr     = np.array(centroids, dtype=np.float32) if centroids else np.empty((0, 2), dtype=np.float32)
    blob_is_split_arr = np.array(blob_is_split, dtype=bool)

    # ── 3. Distance-based outlier filter ─────────────────────────────────────
    keep_mask = None
    if len(centroids_arr) > neighbor_k:
        mean_nn = _mean_knn_distances(centroids_arr, neighbor_k)
        keep_mask = mean_nn <= outlier_factor * np.median(mean_nn)

    kept_indices     = np.where(keep_mask)[0] if keep_mask is not None else np.arange(len(centroids_arr))
    rejected_indices = np.where(~keep_mask)[0] if keep_mask is not None else np.array([], dtype=int)
    filtered_centroids  = centroids_arr[kept_indices]
    filtered_contours   = [blob_contours[i] for i in kept_indices]
    filtered_is_split   = blob_is_split_arr[kept_indices] if len(blob_is_split_arr) else np.zeros(0, dtype=bool)
    rejected_centroids  = centroids_arr[rejected_indices]
    rejected_contours   = [blob_contours[i] for i in rejected_indices]

    # Equivalent-circle radius (px) for each kept blob: radius = sqrt(area / π).
    # Used downstream for depth-scaled size filtering in matching.
    filtered_radii = np.array(
        [np.sqrt(max(cv2.contourArea(cnt.reshape(-1, 1, 2)), 1.0) / np.pi)
         for cnt in filtered_contours],
        dtype=np.float32,
    ) if len(filtered_contours) > 0 else np.empty(0, dtype=np.float32)

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
            is_split = bool(filtered_is_split[idx]) if idx < len(filtered_is_split) else False

            if is_split:
                # Cyan: draw the partition contour so the split boundary is visible
                cnt_draw = filtered_contours[idx].astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(vis, [cnt_draw], -1, (255, 255, 0), 1)
                cv2.circle(vis, pt, 2, (255, 255, 0), -1)
                cv2.circle(vis, pt, 3, (255, 255, 0), 1)
            else:
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

    return filtered_centroids, filtered_contours, filtered_radii, rejected_centroids, rejected_contours