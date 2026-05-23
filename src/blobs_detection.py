import cv2
import numpy as np
from pathlib import Path
from loguru import logger

_pass2_memory: dict = {}  # {cam_key: {...}}  — persists per-camera pass-2 state across frames


_NEIGHBOR_DY = np.array([-1, -1, -1,  0,  0,  1,  1,  1], dtype=np.int32)
_NEIGHBOR_DX = np.array([-1,  0,  1, -1,  1, -1,  0,  1], dtype=np.int32)


def _find_split_maxima(image, ys, xs, peak_threshold, min_split_dist):
    """
    Find local maxima within a blob (pixel coords given by ys, xs).

    A pixel is a local maximum if its value is >= all 8-connected neighbors
    AND >= peak_threshold.  NMS then suppresses weaker maxima within
    min_split_dist pixels of a stronger one.  Returns a list of (y, x).
    """
    vals = image[ys, xs].astype(np.int32)
    is_max = vals >= peak_threshold
    if not is_max.any():
        return []

    h, w = image.shape[:2]
    # Batch all 8 directions: 2 clips instead of 16.
    ny = np.clip(ys[None, :] + _NEIGHBOR_DY[:, None], 0, h - 1)  # (8, N)
    nx = np.clip(xs[None, :] + _NEIGHBOR_DX[:, None], 0, w - 1)  # (8, N)
    is_max &= (vals >= image[ny, nx].astype(np.int32)).all(axis=0)

    if not is_max.any():
        return []

    max_ys   = ys[is_max]
    max_xs   = xs[is_max]
    max_vals = vals[is_max]

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


def _detect_blobs(image, pixel_threshold, required_threshold, cfg,
                   visualize=False, img_path=None, vis_suffix="",
                   max_area_override=None, min_area_override=None,
                   interior_exclude_blobs=None):
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
    Three sequential stages, all depth-invariant:

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

    3. Gradient smoothness filter (H2, very last) — for each surviving blob,
       casts gradient_num_rays rays outward from the NMS peak (same logic as
       split detection, tested on large blobs).  Each ray must be
       monotonically non-increasing in the original unthresholded image
       (within gradient_step_tolerance) until either the neighbourhood radius
       is reached or a valley is detected (val < gradient_valley_ratio * peak,
       indicating the boundary with a neighbouring blob — same valley logic
       as merge splitting).  Blobs where more than gradient_max_bad_ray_fraction
       of rays are non-monotonic are rejected as pseudo-LEDs (e.g. fragments
       inside a bright window with no clear bright maximum).

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

    Pass-2 interior exclusion (H1)
    ───────────────────────────────
    When interior_exclude_blobs is provided (large blobs rejected in pass 1),
    any candidate whose centroid lies deep inside one of those blobs is dropped.
    "Deep" means distance-to-nearest-edge > interior_edge_margin_px (static px
    value, independent of blob size).  Candidates near the edge (real LEDs that
    were merged into the large blob at the lower threshold) are kept.
    """
    min_area           = float(min_area_override if min_area_override is not None else cfg["min_area"])
    max_area           = float(max_area_override if max_area_override is not None else cfg["max_area"])
    max_wh             = int(cfg.get("max_wh", 35))
    outlier_factor     = float(cfg.get("outlier_factor", 3.0))
    area_outlier_k     = float(cfg.get("area_outlier_k", 6.0))
    min_split_dist     = float(cfg.get("min_split_dist", 4.0))
    split_valley_ratio = float(cfg.get("split_valley_ratio", 0.6))
    min_circularity    = float(cfg.get("min_circularity", 0.5))

    # H1: interior-of-large-blob exclusion
    interior_edge_margin_px = float(cfg.get("interior_edge_margin_px", 10.0))

    # H2: smooth-gradient filter
    gradient_radius_factor    = float(cfg.get("gradient_radius_factor", 2.5))
    gradient_num_rays         = int(cfg.get("gradient_num_rays", 16))
    gradient_valley_ratio     = float(cfg.get("gradient_valley_ratio", split_valley_ratio))
    gradient_step_tolerance   = float(cfg.get("gradient_step_tolerance", 0.15))
    gradient_max_bad_ray_frac = float(cfg.get("gradient_max_bad_ray_fraction", 0.4))
    gradient_min_radius       = float(cfg.get("gradient_min_radius", 5.0))

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

    centroids        = []
    blob_contours    = []
    blob_pixels_list = []
    blob_max_pixels  = []
    # Large blobs rejected by area/wh — always tracked; passed to pass-2 as H1 exclusion zones.
    large_rejected_contours = []
    # Per-reason rejection lists — populated only when visualize=True.
    det_rej_area_small = []
    det_rej_area_large = []
    det_rej_wh         = []
    det_rej_threshold  = []
    det_rej_interior   = []   # H1: centroid deep inside a large blob
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
            large_rejected_contours.append(cnt)
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
            large_rejected_contours.append(cnt)
            if visualize: det_rej_wh.append(cnt)
            continue

        # ── Build pixel mask for this blob (ROI-sized, not full-image) ──────
        roi_mask = np.zeros((h_b, w_b), dtype=np.uint8)
        cnt_roi = cnt.reshape(-1, 2).copy()
        cnt_roi[:, 0] -= x_b
        cnt_roi[:, 1] -= y_b
        cv2.drawContours(roi_mask, [cnt_roi.reshape(-1, 1, 2).astype(np.int32)], -1, 255, -1)
        ys_roi, xs_roi = np.nonzero(roi_mask)
        ys = ys_roi + y_b
        xs = xs_roi + x_b

        # ── Required-threshold gate: need at least one bright pixel ───────────
        if ys.size == 0:
            if visualize: det_rej_threshold.append(cnt)
            continue
        w_pix = intensities[ys, xs]
        max_pix = int(w_pix.max())
        if max_pix < required_threshold:
            if _in_bbox(bx, by):
                logger.debug(f"[blob_debug] ({bx:.0f},{by:.0f}) dropped: max pixel "
                             f"{max_pix} < required_threshold {required_threshold}")
            if visualize: det_rej_threshold.append(cnt)
            continue

        # ── Intensity-weighted centroid (greysum, 1-based coords) ─────────────
        total_weight = float(w_pix.sum())
        if total_weight == 0:
            if _in_bbox(bx, by):
                logger.debug(f"[blob_debug] ({bx:.0f},{by:.0f}) dropped: zero total weight")
            if visualize: det_rej_threshold.append(cnt)
            continue

        # 1-based coordinates prevent the first row/column from never
        # contributing to the weighted sum (matches blobwatch.c greysum logic).
        cx = float(np.sum((xs + 1) * w_pix)) / total_weight - 1.0
        cy = float(np.sum((ys + 1) * w_pix)) / total_weight - 1.0

        # ── H1: reject if centroid is deep inside a large pass-1 blob ────────
        if interior_exclude_blobs:
            deep_inside = False
            for large_cnt in interior_exclude_blobs:
                dist = cv2.pointPolygonTest(large_cnt, (float(cx), float(cy)), True)
                if dist > interior_edge_margin_px:
                    deep_inside = True
                    if _in_bbox(cx, cy):
                        logger.debug(
                            f"[blob_debug] ({cx:.0f},{cy:.0f}) dropped: deep inside large blob "
                            f"dist_to_edge={dist:.1f} > margin={interior_edge_margin_px:.1f}"
                        )
                    break
            if deep_inside:
                if visualize: det_rej_interior.append(cnt)
                continue

        centroids.append((cx, cy))
        blob_contours.append(cnt.reshape(-1, 2).astype(np.float32))
        blob_pixels_list.append((ys, xs))
        blob_max_pixels.append(max_pix)

    # ── 2b. Merged-blob splitting ─────────────────────────────────────────────
    new_centroids     = []
    new_blob_contours = []
    new_blob_max_pixels = []
    new_blob_peaks    = []  # cached (peak_y, peak_x) per post-split blob for H2 reuse
    blob_is_split     = []

    for i in range(len(centroids)):
        ys_b, xs_b = blob_pixels_list[i]
        maxima = _find_split_maxima(image, ys_b, xs_b, required_threshold, min_split_dist)
        peak = maxima[0] if maxima else None

        if len(maxima) == 2:
            peak_lower = min(int(image[maxima[0][0], maxima[0][1]]),
                             int(image[maxima[1][0], maxima[1][1]]))
            saddle = _saddle_min(image, maxima[0], maxima[1])
            valley_ok = peak_lower > 0 and saddle / peak_lower < split_valley_ratio
            if not valley_ok:
                new_centroids.append(centroids[i])
                new_blob_contours.append(blob_contours[i])
                new_blob_max_pixels.append(blob_max_pixels[i])
                new_blob_peaks.append(peak)
                blob_is_split.append(False)
                continue
            split_result = _split_blob_at_seeds(image, intensities, ys_b, xs_b, maxima[0], maxima[1])
            if split_result is not None:
                dist_a  = (ys_b - maxima[0][0]) ** 2 + (xs_b - maxima[0][1]) ** 2
                dist_b_ = (ys_b - maxima[1][0]) ** 2 + (xs_b - maxima[1][1]) ** 2
                for part_idx, ((pcx, pcy), part_cnt) in enumerate(split_result):
                    new_centroids.append((pcx, pcy))
                    new_blob_contours.append(part_cnt)
                    blob_is_split.append(True)
                    part_mask = (dist_a <= dist_b_) if part_idx == 0 else (dist_a > dist_b_)
                    part_pix  = image[ys_b[part_mask], xs_b[part_mask]] if part_mask.any() else np.array([0])
                    new_blob_max_pixels.append(int(part_pix.max()))
                    new_blob_peaks.append(maxima[part_idx])
                continue

        new_centroids.append(centroids[i])
        new_blob_contours.append(blob_contours[i])
        new_blob_max_pixels.append(blob_max_pixels[i])
        new_blob_peaks.append(peak)
        blob_is_split.append(False)

    centroids       = new_centroids
    blob_contours   = new_blob_contours
    blob_max_pixels = new_blob_max_pixels
    blob_peaks      = new_blob_peaks

    centroids_arr       = np.array(centroids, dtype=np.float32) if centroids else np.empty((0, 2), dtype=np.float32)
    blob_max_pixels_arr = np.array(blob_max_pixels, dtype=np.float32) if blob_max_pixels else np.empty(0, dtype=np.float32)
    blob_is_split_arr   = np.array(blob_is_split, dtype=bool)

    # ── 3. Post-split outlier filters ────────────────────────────────────────
    areas_arr = np.array(
        [cv2.contourArea(cnt.reshape(-1, 1, 2)) for cnt in blob_contours],
        dtype=np.float32,
    ) if blob_contours else np.empty(0, dtype=np.float32)

    keep_mask          = np.ones(len(centroids_arr), dtype=bool)
    area_keep_mask     = np.ones(len(centroids_arr), dtype=bool)
    circ_keep_mask     = np.ones(len(centroids_arr), dtype=bool)
    gradient_keep_mask = np.ones(len(centroids_arr), dtype=bool)
    upper_limit        = np.inf
    median_area        = mad = 0.0
    epsilon            = np.inf

    # ── 3a. Circularity filter: reject non-circular blobs (4π·area/perimeter²)
    for i, cnt in enumerate(blob_contours):
        perimeter = cv2.arcLength(cnt.reshape(-1, 1, 2), closed=True)
        if perimeter > 0:
            circularity = 4 * np.pi * areas_arr[i] / (perimeter ** 2)
            if circularity < min_circularity:
                circ_keep_mask[i] = False
        else:
            circ_keep_mask[i] = False
    keep_mask &= circ_keep_mask

    # ── 3b. Area filter: reject blobs with area > median + area_outlier_k * MAD
    if len(areas_arr) >= 3:
        median_area = float(np.median(areas_arr))
        mad         = float(np.median(np.abs(areas_arr - median_area)))
        if mad > 0:
            upper_limit    = median_area + area_outlier_k * mad
            area_keep_mask = areas_arr <= upper_limit
            keep_mask     &= area_keep_mask

    # ── 3c. DBSCAN 1-NN filter: reject blobs isolated from the surviving set.
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

    # ── 3d. Gradient smoothness filter (vectorized ray casting)
    h_img, w_img = image.shape[:2]
    _ray_angles = (2.0 * np.pi / gradient_num_rays) * np.arange(gradient_num_rays)
    _ray_dy = np.sin(_ray_angles)   # (R,) — precomputed, same for every blob
    _ray_dx = np.cos(_ray_angles)   # (R,)
    for i in np.where(keep_mask)[0]:
        cnt_i       = blob_contours[i]
        blob_radius = np.sqrt(max(float(areas_arr[i]), 1.0) / np.pi)
        nr_int      = max(int(gradient_radius_factor * blob_radius), int(gradient_min_radius))

        # ROI-sized mask — avoids allocating / scanning a full (H×W) image.
        x_b, y_b, w_b, h_b = cv2.boundingRect(cnt_i.reshape(-1, 1, 2).astype(np.int32))
        roi_mask = np.zeros((h_b, w_b), dtype=np.uint8)
        cnt_roi  = cnt_i.reshape(-1, 2).copy()
        cnt_roi[:, 0] -= x_b
        cnt_roi[:, 1] -= y_b
        cv2.drawContours(roi_mask, [cnt_roi.reshape(-1, 1, 2).astype(np.int32)], -1, 255, -1)
        ys_roi, xs_roi = np.nonzero(roi_mask)
        if ys_roi.size == 0:
            continue
        ys_b = ys_roi + y_b
        xs_b = xs_roi + x_b

        cached_peak = blob_peaks[i]
        if cached_peak is None:
            continue
        peak_y, peak_x = cached_peak
        peak_val = int(image[peak_y, peak_x])
        if peak_val == 0:
            continue

        valley_floor = gradient_valley_ratio * peak_val

        # Vectorized: compute all (R × S) pixel coords at once and read image in one shot.
        steps  = np.arange(1, nr_int + 1, dtype=np.float32)             # (S,)
        py_all = np.round(peak_y + _ray_dy[:, None] * steps).astype(np.int32)  # (R, S)
        px_all = np.round(peak_x + _ray_dx[:, None] * steps).astype(np.int32)  # (R, S)

        in_bounds = ((py_all >= 0) & (py_all < h_img) &
                     (px_all >= 0) & (px_all < w_img))                   # (R, S)
        py_clip = np.clip(py_all, 0, h_img - 1)
        px_clip = np.clip(px_all, 0, w_img - 1)

        vals = image[py_clip, px_clip].astype(np.float32)                # (R, S)

        # Active range: each ray stops at the first valley or first out-of-bounds step.
        valley_hit  = in_bounds & (vals < valley_floor)
        oob_hit     = ~in_bounds
        first_valley = np.where(valley_hit.any(1), np.argmax(valley_hit, axis=1), nr_int)
        first_oob    = np.where(oob_hit.any(1),    np.argmax(oob_hit,    axis=1), nr_int)
        active_end   = np.minimum(first_valley, first_oob)               # (R,)

        # Monotonicity: detect any step where intensity rises beyond tolerance.
        prev_vals = np.empty_like(vals)
        prev_vals[:, 0] = float(peak_val)
        if nr_int > 1:
            prev_vals[:, 1:] = vals[:, :-1]
        step_idx  = np.arange(nr_int)[None, :]                           # (1, S)
        in_active = step_idx < active_end[:, None]                       # (R, S)
        bad_rays  = int(
            (in_active & in_bounds & (vals > prev_vals * (1.0 + gradient_step_tolerance)))
            .any(axis=1).sum()
        )

        if bad_rays / gradient_num_rays > gradient_max_bad_ray_frac:
            gradient_keep_mask[i] = False
            if _in_bbox(float(centroids_arr[i, 0]), float(centroids_arr[i, 1])):
                logger.debug(
                    f"[blob_debug] ({centroids_arr[i, 0]:.1f},{centroids_arr[i, 1]:.1f}) "
                    f"dropped: gradient not smooth, bad_rays={bad_rays}/{gradient_num_rays}"
                )

    keep_mask &= gradient_keep_mask

    kept_indices     = np.where(keep_mask)[0]
    rejected_indices = np.where(~keep_mask)[0]

    if _dbbox is not None:
        for i in rejected_indices:
            cx_r, cy_r = float(centroids_arr[i, 0]), float(centroids_arr[i, 1])
            if _in_bbox(cx_r, cy_r):
                if not circ_keep_mask[i]:
                    logger.debug(f"[blob_debug] ({cx_r:.1f},{cy_r:.1f}) dropped: not circular")
                elif not area_keep_mask[i]:
                    logger.debug(
                        f"[blob_debug] ({cx_r:.1f},{cy_r:.1f}) dropped: area outlier "
                        f"area={areas_arr[i]:.1f} > {median_area:.1f}+{area_outlier_k}×{mad:.1f}"
                    )
                elif not gradient_keep_mask[i]:
                    pass  # already logged in the gradient loop above
                else:
                    logger.debug(
                        f"[blob_debug] ({cx_r:.1f},{cy_r:.1f}) dropped: spatial outlier "
                        f"1-NN={min_nn_full[i]:.1f}px > epsilon={epsilon:.1f}px"
                    )
        for i in kept_indices:
            cx_k, cy_k = float(centroids_arr[i, 0]), float(centroids_arr[i, 1])
            if _in_bbox(cx_k, cy_k):
                logger.debug(f"[blob_debug] ({cx_k:.1f},{cy_k:.1f}) survived all stages → kept")

    filtered_centroids   = centroids_arr[kept_indices]
    filtered_contours    = [blob_contours[i] for i in kept_indices]
    filtered_is_split    = blob_is_split_arr[kept_indices] if len(blob_is_split_arr) else np.zeros(0, dtype=bool)
    filtered_brightnesses = blob_max_pixels_arr[kept_indices] if len(kept_indices) else np.empty(0, dtype=np.float32)
    rejected_centroids   = centroids_arr[rejected_indices]
    rejected_contours    = [blob_contours[i] for i in rejected_indices]

    filtered_radii = np.array(
        [np.sqrt(max(cv2.contourArea(cnt.reshape(-1, 1, 2)), 1.0) / np.pi)
         for cnt in filtered_contours],
        dtype=np.float32,
    ) if len(filtered_contours) > 0 else np.empty(0, dtype=np.float32)

    # ── 4. Visualization ──────────────────────────────────────────────────────
    if visualize:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # BGR colours per category
        # Highly distinct debug colors (BGR)

        C_AREA_SM = (255, 0, 255)  # pink         — area < min_area or 1×1
        C_AREA_LG = (0, 140, 255)  # dark orange  — area > max_area (hard limit)
        C_WH = (255, 255, 0)  # dark blue    — bounding-box too wide/tall
        C_THRESH = (180, 0, 0)  # magenta      — below required brightness
        C_CIRC = (0, 255, 255)  # yellow       — circularity filter
        C_AREA_OUT = (0, 0, 255)  # orange       — area outlier filter
        C_SPAT = (128, 128, 128)  # red          — spatial (DBSCAN) outlier
        C_INTERIOR = (128, 0, 255)  # purple       — H1: deep inside large blob
        C_GRAD = (0, 255, 0)  # teal         — H2: no smooth gradient
        C_KEPT = (255, 255, 255)  # green        — kept
        C_SPLT = (0, 255, 128)  # cyan         — kept (split)

        def _draw_cnt(cnt, color):
            cv2.drawContours(vis, [cnt.reshape(-1, 1, 2).astype(np.int32)], -1, color, 1)

        for cnt in det_rej_area_small: _draw_cnt(cnt, C_AREA_SM)
        for cnt in det_rej_area_large: _draw_cnt(cnt, C_AREA_LG)
        for cnt in det_rej_wh:         _draw_cnt(cnt, C_WH)
        for cnt in det_rej_threshold:  _draw_cnt(cnt, C_THRESH)
        for cnt in det_rej_interior:   _draw_cnt(cnt, C_INTERIOR)

        for i in rejected_indices:
            if not circ_keep_mask[i]:
                _draw_cnt(blob_contours[i], C_CIRC)
            elif not area_keep_mask[i]:
                _draw_cnt(blob_contours[i], C_AREA_OUT)
            elif not gradient_keep_mask[i]:
                _draw_cnt(blob_contours[i], C_GRAD)
            else:
                _draw_cnt(blob_contours[i], C_SPAT)

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
            (C_CIRC,     f"not circular (< {min_circularity})"),
            (C_SPAT,     "spatial outlier"),
            (C_GRAD,     "no gradient"),
            (C_INTERIOR, "deep in large blob"),
            (C_THRESH,   f"too dim (< {required_threshold})"),
            (C_WH,       f"wh > {max_wh}px"),
            (C_AREA_LG,  f"area > {int(max_area)}px"),
            (C_AREA_SM,  f"area < {int(min_area)}px"),
        ]
        row_h = 20
        half = len(strip_entries) // 2 + len(strip_entries) % 2
        rows = [strip_entries[:half], strip_entries[half:]]
        strip = np.zeros((row_h * 3, vis.shape[1], 3), dtype=np.uint8)
        for row_idx, row in enumerate(rows):
            x = 6
            y_sq_top = row_idx * row_h + 5
            y_sq_bot = y_sq_top + 10
            y_text   = y_sq_bot - 1
            for color, label in row:
                cv2.rectangle(strip, (x, y_sq_top), (x + 10, y_sq_bot), color, -1)
                x += 13
                cv2.putText(strip, label, (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
                (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                x += tw + 10
        thr_label = f"pixel_thr={pixel_threshold}  required_thr={required_threshold}"
        cv2.putText(strip, thr_label, (6, row_h * 2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        vis = np.vstack([vis, strip])

        if img_path is not None:
            out_dir = Path("./visualization/blobs")
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{Path(img_path).stem}_blobs{vis_suffix}.png"), vis)

    return (filtered_centroids, filtered_contours, filtered_radii,
            filtered_brightnesses,
            rejected_centroids, rejected_contours, large_rejected_contours)


def get_centroids(image, cfg, visualize=False, img_path=None, cam_idx=None):
    pixel_threshold             = int(cfg["min_threshold"])
    required_threshold          = int(cfg.get("required_threshold", pixel_threshold * 2))
    pass2_factor                = float(cfg.get("pass2_threshold_factor", 0.0))
    pass2_required_factor       = float(cfg.get("pass2_required_factor", 0.7))
    pass2_brightness_percentile = float(cfg.get("pass2_brightness_percentile", 25.0))
    ema_alpha                   = float(cfg.get("pass2_threshold_ema_alpha", 1.0))
    count_gate_max_factor       = float(cfg.get("pass2_count_gate_max_factor", 1.7))

    # Per-camera EMA memory — keyed by cam_idx so cameras don't bleed into each other.
    _cam_key = cam_idx if cam_idx is not None else 0
    _mem = _pass2_memory.setdefault(_cam_key, {})

    cam_str      = f"_cam{cam_idx}" if cam_idx is not None else ""
    vis_suffix_1 = f"{cam_str}_pass1" if pass2_factor > 0 else cam_str
    result = _detect_blobs(image, pixel_threshold, required_threshold, cfg,
                            visualize, img_path, vis_suffix_1)
    large_blobs_pass1 = result[6]

    min_area_pass2 = cfg.get("pass2_min_area")
    min_area_pass2 = float(min_area_pass2) if min_area_pass2 is not None else None

    if pass2_factor > 0 and len(result[0]) >= 2:
        # Normal path: derive pass-2 params from pass-1 blob stats.
        mean_vals = []
        area_vals = []
        for cnt in result[1]:
            cnt_int = cnt.reshape(-1, 1, 2).astype(np.int32)
            xb, yb, wb, hb = cv2.boundingRect(cnt_int)
            roi_m = np.zeros((hb, wb), dtype=np.uint8)
            cnt_r = cnt.reshape(-1, 2).copy()
            cnt_r[:, 0] -= xb
            cnt_r[:, 1] -= yb
            cv2.drawContours(roi_m, [cnt_r.reshape(-1, 1, 2).astype(np.int32)], -1, 255, -1)
            px = image[yb:yb+hb, xb:xb+wb][roi_m > 0]
            if px.size > 0:
                mean_vals.append(float(np.mean(px)))
            area_vals.append(float(cv2.contourArea(cnt_int)))
        if mean_vals:
            ref_brightness   = float(np.percentile(mean_vals, pass2_brightness_percentile))
            raw_pixel_thr    = int(ref_brightness * pass2_factor)
            raw_required_thr = int(ref_brightness * pass2_required_factor)
            if raw_pixel_thr > pixel_threshold:
                cur_count  = len(mean_vals)
                prev_count = _mem.get("blob_count", cur_count)
                count_stable = (max(cur_count, prev_count) / max(min(cur_count, prev_count), 1)
                                <= count_gate_max_factor)

                if count_stable or not _mem:
                    if _mem:
                        pixel_threshold_2    = int(ema_alpha * raw_pixel_thr
                                                   + (1 - ema_alpha) * _mem.get("pixel_threshold", raw_pixel_thr))
                        required_threshold_2 = int(ema_alpha * raw_required_thr
                                                   + (1 - ema_alpha) * _mem.get("required_threshold", raw_required_thr))
                    else:
                        pixel_threshold_2    = raw_pixel_thr
                        required_threshold_2 = raw_required_thr
                    update_memory = True
                else:
                    # Blob count jumped — hold previous stable thresholds, don't update memory.
                    pixel_threshold_2    = _mem["pixel_threshold"]
                    required_threshold_2 = _mem.get("required_threshold", required_threshold)
                    update_memory = False

                max_area_pass2 = float(max(area_vals)) if area_vals else None
                result2 = _detect_blobs(image, pixel_threshold_2, required_threshold_2, cfg,
                                         visualize, img_path, f"{cam_str}_pass2",
                                         max_area_override=max_area_pass2,
                                         min_area_override=min_area_pass2,
                                         interior_exclude_blobs=large_blobs_pass1)
                if len(result2[0]) >= 1:
                    if update_memory:
                        _mem["pixel_threshold"]    = pixel_threshold_2
                        _mem["required_threshold"] = required_threshold_2
                        _mem["max_area"]           = max_area_pass2
                        _mem["large_blobs"]        = large_blobs_pass1
                        _mem["blob_count"]         = cur_count
                    return result2

    elif pass2_factor > 0 and _mem:
        # Fallback: pass 1 didn't yield enough blobs — reuse params from last
        # successful pass 2.  Use large blobs from the current frame (not stored
        # ones) since they represent the actual exclusion zones this frame.
        result2 = _detect_blobs(image, _mem["pixel_threshold"],
                                 _mem.get("required_threshold", required_threshold), cfg,
                                 visualize, img_path, f"{cam_str}_pass2",
                                 max_area_override=_mem["max_area"],
                                 min_area_override=min_area_pass2,
                                 interior_exclude_blobs=large_blobs_pass1)
        if len(result2[0]) >= 1:
            return result2
        _mem.clear()

    return result
