import cv2
import numpy as np
from pathlib import Path
from loguru import logger


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

    Outlier filters (applied after splitting)
    ─────────────────────────────────────────
    Two sequential stages, both depth-invariant:

    1. Area filter — rejects blobs whose area exceeds
       median(area) + area_outlier_k × MAD(area).  Catches large reflections
       or unsplit merged groups without needing a pose estimate.  Skipped when
       fewer than 3 blobs survive (MAD undefined / uninformative).

    2. DBSCAN 1-NN spatial filter — for each blob (from the area-kept set),
       finds its nearest neighbour distance.  Any blob whose 1-NN distance
       exceeds outlier_factor × median(1-NN distances) is rejected as
       spatially isolated noise.  Epsilon is derived from the surviving blob
       distances, so it scales naturally with depth.  Skipped when fewer than
       2 blobs survive.

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
    outlier_factor     = float(cfg.get("outlier_factor", 3.0))
    area_outlier_k     = float(cfg.get("area_outlier_k", 6.0))
    min_split_dist     = float(cfg.get("min_split_dist", 4.0))
    split_valley_ratio = float(cfg.get("split_valley_ratio", 0.6))

    _dbbox = cfg.get("debug_bbox")  # [x1, y1, x2, y2] or null

    def _in_bbox(cx, cy):
        if _dbbox is None:
            return False
        x1, y1, x2, y2 = _dbbox
        return x1 <= cx <= x2 and y1 <= cy <= y2

    # ── 1. Threshold at pixel_threshold ──────────────────────────────────────
    _, mask = cv2.threshold(image, pixel_threshold, 255, cv2.THRESH_BINARY)

    # ── 2. Find contours ─────────────────────────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    centroids = []
    blob_contours = []
    blob_pixels_list = []
    # Per-reason rejection lists — populated only when visualize=True
    det_rej_area_small = []   # area < min_area (or 1×1)
    det_rej_area_large = []   # area > max_area (hard limit)
    det_rej_wh         = []   # bounding-box dimension > max_wh
    det_rej_threshold  = []   # max pixel < required_threshold (or zero weight)
    intensities = image.astype(np.float32)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
        bx, by = x_b + w_b / 2.0, y_b + h_b / 2.0  # position proxy before centroid is known

        # ── Reject by area ────────────────────────────────────────────────────
        if area < min_area:
            if _in_bbox(bx, by):
                logger.debug(f"[blob_debug] ({bx:.0f},{by:.0f}) dropped: area {area:.1f} < min_area {min_area}")
            if visualize: det_rej_area_small.append(cnt)
            continue
        if area > max_area:
            if _in_bbox(bx, by):
                logger.debug(f"[blob_debug] ({bx:.0f},{by:.0f}) dropped: area {area:.1f} > max_area {max_area}")
            if visualize: det_rej_area_large.append(cnt)
            continue

        # ── Reject 1×1 blobs (pure noise) ────────────────────────────────────
        if w_b == 1 and h_b == 1:
            if _in_bbox(bx, by):
                logger.debug(f"[blob_debug] ({bx:.0f},{by:.0f}) dropped: 1×1 blob")
            if visualize: det_rej_area_small.append(cnt)
            continue

        # ── Reject blobs that are too large to be LEDs ───────────────────────
        if w_b > max_wh or h_b > max_wh:
            if _in_bbox(bx, by):
                logger.debug(f"[blob_debug] ({bx:.0f},{by:.0f}) dropped: size {w_b}×{h_b} > max_wh {max_wh}")
            if visualize: det_rej_wh.append(cnt)
            continue

        # ── Build pixel mask for this blob ───────────────────────────────────
        blob_mask = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, -1)

        # ── Required-threshold gate: need at least one bright pixel ───────────
        # Equivalent to blobwatch.c blob_required_threshold check:
        # collects faint blobs but discards purely-dim background regions.
        roi_pixels = image[blob_mask > 0]
        if roi_pixels.size == 0 or int(roi_pixels.max()) < required_threshold:
            if _in_bbox(bx, by):
                logger.debug(f"[blob_debug] ({bx:.0f},{by:.0f}) dropped: max pixel "
                             f"{int(roi_pixels.max()) if roi_pixels.size else 0} < required_threshold {required_threshold}")
            if visualize: det_rej_threshold.append(cnt)
            continue

        # ── Intensity-weighted centroid (greysum, 1-based coords) ─────────────
        weights = intensities * (blob_mask > 0)
        total_weight = np.sum(weights)
        if total_weight == 0:
            if _in_bbox(bx, by):
                logger.debug(f"[blob_debug] ({bx:.0f},{by:.0f}) dropped: zero total weight")
            if visualize: det_rej_threshold.append(cnt)
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

    # ── 3. Area outlier filter + DBSCAN 1-NN spatial outlier filter ──────────
    areas_arr = np.array(
        [cv2.contourArea(cnt.reshape(-1, 1, 2)) for cnt in blob_contours],
        dtype=np.float32,
    ) if blob_contours else np.empty(0, dtype=np.float32)

    keep_mask      = np.ones(len(centroids_arr), dtype=bool)
    area_keep_mask = np.ones(len(centroids_arr), dtype=bool)
    upper_limit    = np.inf
    median_area    = mad = 0.0
    epsilon        = np.inf

    # ── 3a. Area filter: reject blobs with area > median + area_outlier_k * MAD
    if len(areas_arr) >= 3:
        median_area = float(np.median(areas_arr))
        mad         = float(np.median(np.abs(areas_arr - median_area)))
        if mad > 0:
            upper_limit    = median_area + area_outlier_k * mad
            area_keep_mask = areas_arr <= upper_limit
            keep_mask     &= area_keep_mask

    # ── 3b. DBSCAN 1-NN filter: reject blobs isolated from the surviving set.
    #   Epsilon is set from the surviving blob distances so it is depth-invariant.
    candidates   = centroids_arr[keep_mask]
    min_nn_full  = np.full(len(centroids_arr), np.inf)
    if len(candidates) > 1:
        diffs    = candidates[:, None, :] - candidates[None, :, :]
        sq_dists = (diffs ** 2).sum(axis=-1)
        np.fill_diagonal(sq_dists, np.inf)
        min_nn      = np.sqrt(sq_dists.min(axis=1))
        epsilon     = outlier_factor * float(np.median(min_nn))
        cand_idx    = np.where(keep_mask)[0]
        min_nn_full[cand_idx] = min_nn
        keep_mask[cand_idx[min_nn > epsilon]] = False

    kept_indices     = np.where(keep_mask)[0]
    rejected_indices = np.where(~keep_mask)[0]

    if _dbbox is not None:
        for i in rejected_indices:
            cx_r, cy_r = float(centroids_arr[i, 0]), float(centroids_arr[i, 1])
            if _in_bbox(cx_r, cy_r):
                if not area_keep_mask[i]:
                    logger.debug(
                        f"[blob_debug] ({cx_r:.1f},{cy_r:.1f}) dropped: area outlier "
                        f"area={areas_arr[i]:.1f} > {median_area:.1f}+{area_outlier_k}×{mad:.1f}"
                    )
                else:
                    logger.debug(
                        f"[blob_debug] ({cx_r:.1f},{cy_r:.1f}) dropped: spatial outlier "
                        f"1-NN={min_nn_full[i]:.1f}px > epsilon={epsilon:.1f}px"
                    )
        for i in kept_indices:
            cx_k, cy_k = float(centroids_arr[i, 0]), float(centroids_arr[i, 1])
            if _in_bbox(cx_k, cy_k):
                logger.debug(f"[blob_debug] ({cx_k:.1f},{cy_k:.1f}) survived all stages → kept")

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

        # BGR colours per category
        C_AREA_SM  = (130, 130, 130)  # grey        — area < min_area or 1×1
        C_AREA_LG  = (0,   80,  200)  # dark orange — area > max_area (hard limit)
        C_WH       = (200,  80,    0)  # dark blue   — bounding-box too wide/tall
        C_THRESH   = (200,   0,  200)  # magenta     — below required brightness
        C_AREA_OUT = (0,  140,  255)  # orange      — area outlier filter
        C_SPAT     = (0,    0,  210)  # red         — spatial (DBSCAN) outlier
        C_KEPT     = (0,  200,    0)  # green       — kept
        C_SPLT     = (255, 200,    0)  # cyan        — kept (split)

        def _draw_cnt(cnt, color):
            cv2.drawContours(vis, [cnt.reshape(-1, 1, 2).astype(np.int32)], -1, color, 1)

        for cnt in det_rej_area_small: _draw_cnt(cnt, C_AREA_SM)
        for cnt in det_rej_area_large: _draw_cnt(cnt, C_AREA_LG)
        for cnt in det_rej_wh:         _draw_cnt(cnt, C_WH)
        for cnt in det_rej_threshold:  _draw_cnt(cnt, C_THRESH)

        for i in rejected_indices:
            _draw_cnt(blob_contours[i], C_AREA_OUT if not area_keep_mask[i] else C_SPAT)

        for i, (pt_f, cnt) in enumerate(zip(filtered_centroids, filtered_contours)):
            is_split = bool(filtered_is_split[i]) if i < len(filtered_is_split) else False
            color = C_SPLT if is_split else C_KEPT
            _draw_cnt(cnt, color)
            pt = (int(round(pt_f[0])), int(round(pt_f[1])))
            cv2.circle(vis, pt, 2, color, -1)
            cv2.putText(vis, str(i), (pt[0] + 5, pt[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # Legend — horizontal strip below image
        strip_entries = [
            (C_KEPT,     "kept"),
            (C_SPLT,     "split"),
            (C_AREA_OUT, "area outlier"),
            (C_SPAT,     "spatial outlier"),
            (C_THRESH,   "too dim"),
            (C_WH,       f"wh > {max_wh}px"),
            (C_AREA_LG,  f"area > {int(max_area)}px"),
            (C_AREA_SM,  f"area < {int(min_area)}px"),
        ]
        strip_h = 22
        strip = np.zeros((strip_h, vis.shape[1], 3), dtype=np.uint8)
        x = 6
        for color, label in strip_entries:
            cv2.rectangle(strip, (x, 5), (x + 10, 15), color, -1)
            x += 13
            cv2.putText(strip, label, (x, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            x += tw + 10
        vis = np.vstack([vis, strip])

        if img_path is not None:
            out_dir = Path("./visualization/blobs")
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{Path(img_path).stem}_blobs.png"), vis)

    return filtered_centroids, filtered_contours, filtered_radii, rejected_centroids, rejected_contours