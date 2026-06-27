import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


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
                   interior_exclude_blobs=None,
                   warm_mode=False,
                   vis_patch_out=None):
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
    outlier_factor     = float(cfg.get("outlier_factor", 3.0))
    min_split_dist     = float(cfg.get("min_split_dist", 4.0))
    split_valley_ratio = float(cfg.get("split_valley_ratio", 0.6))
    min_circularity    = float(cfg.get("min_circularity", 0.5))

    # H1: interior-of-large-blob exclusion (no-prior / 2-pass path only)
    interior_edge_margin_px = float(cfg.get("interior_edge_margin_px", 10.0))

    # ── 1. Threshold at pixel_threshold ──────────────────────────────────────
    _, mask = cv2.threshold(image, pixel_threshold, 255, cv2.THRESH_BINARY)

    # ── 2. Find contours ─────────────────────────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    centroids        = []
    blob_contours    = []
    blob_pixels_list = []
    blob_max_pixels  = []
    # Large blobs rejected by area — always tracked; passed to pass-2 as H1 exclusion zones.
    large_rejected_contours = []
    # Per-reason rejection lists — populated only when visualize=True.
    det_rej_area_small = []
    det_rej_area_large = []
    det_rej_threshold  = []
    det_rej_interior   = []   # H1: centroid deep inside a large blob (no-prior path only)
    intensities = image.astype(np.float32)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)

        # ── Reject by max area and bounding-box size (cheap, before mask) ─────
        if area > max_area:
            large_rejected_contours.append(cnt)
            if visualize: det_rej_area_large.append(cnt)
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

        # ── Reject by pixel count (cv2.contourArea underestimates for small
        #    irregular blobs — a 3-px L-shape has geometric area ≈ 0.5) ────────
        if len(ys_roi) < min_area:
            if visualize: det_rej_area_small.append(cnt)
            continue

        # ── Required-threshold gate: need at least one bright pixel ───────────
        if ys.size == 0:
            if visualize: det_rej_threshold.append(cnt)
            continue
        w_pix = intensities[ys, xs]
        max_pix = int(w_pix.max())
        if max_pix < required_threshold:
            if visualize: det_rej_threshold.append(cnt)
            continue

        # ── Intensity-weighted centroid (greysum, 1-based coords) ─────────────
        total_weight = float(w_pix.sum())
        if total_weight == 0:
            if visualize: det_rej_threshold.append(cnt)
            continue

        # 1-based coordinates prevent the first row/column from never
        # contributing to the weighted sum (matches blobwatch.c greysum logic).
        cx = float(np.sum((xs + 1) * w_pix)) / total_weight - 1.0
        cy = float(np.sum((ys + 1) * w_pix)) / total_weight - 1.0

        # ── H1: reject if centroid is deep inside a large pass-1 blob ────────
        if not warm_mode and interior_exclude_blobs:
            deep_inside = False
            for large_cnt in interior_exclude_blobs:
                dist = cv2.pointPolygonTest(large_cnt, (float(cx), float(cy)), True)
                if dist > interior_edge_margin_px:
                    deep_inside = True
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

    keep_mask      = np.ones(len(centroids_arr), dtype=bool)
    circ_keep_mask = np.ones(len(centroids_arr), dtype=bool)
    epsilon        = np.inf

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

    # ── 3b. DBSCAN 1-NN filter: reject blobs isolated from the surviving set.
    # Skipped in local-search mode: LED projections are spatially spread across
    # the image so outer-ring LEDs would be wrongly rejected as isolated.
    candidates   = centroids_arr[keep_mask]
    min_nn_full  = np.full(len(centroids_arr), np.inf)
    if not warm_mode and len(candidates) > 1:
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
    vis_out = None
    if visualize:
        # Warm-mode uses a black background so only colored annotation pixels are
        # non-zero; the compositor then blends only those pixels onto the canvas.
        vis = (np.zeros((*image.shape[:2], 3), dtype=np.uint8)
               if warm_mode else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))

        C_AREA_SM  = (255, 0, 255)    # pink        — pixels < min_area or 1×1
        C_AREA_LG  = (0, 140, 255)   # dark orange — area > max_area (hard limit)
        C_THRESH   = (180, 0, 0)     # dark red    — below required brightness
        C_CIRC     = (0, 255, 255)   # yellow      — circularity filter
        C_SPAT     = (128, 128, 128) # grey        — spatial (DBSCAN) outlier (cold path only)
        C_INTERIOR = (100, 180, 255) # light blue  — H1: deep inside large blob (cold path only)
        C_KEPT     = (255, 255, 255) # white       — kept
        C_SPLT     = (0, 255, 128)   # cyan        — kept (split)

        def _draw_cnt(cnt, color):
            cv2.drawContours(vis, [cnt.reshape(-1, 1, 2).astype(np.int32)], -1, color, 1)

        for cnt in det_rej_area_small: _draw_cnt(cnt, C_AREA_SM)
        for cnt in det_rej_area_large: _draw_cnt(cnt, C_AREA_LG)
        for cnt in det_rej_threshold:  _draw_cnt(cnt, C_THRESH)
        if not warm_mode:
            for cnt in det_rej_interior: _draw_cnt(cnt, C_INTERIOR)

        for i in rejected_indices:
            if not circ_keep_mask[i]:
                _draw_cnt(blob_contours[i], C_CIRC)
            elif not warm_mode:
                _draw_cnt(blob_contours[i], C_SPAT)

        for i, (pt_f, cnt) in enumerate(zip(filtered_centroids, filtered_contours)):
            is_split = bool(filtered_is_split[i]) if i < len(filtered_is_split) else False
            color = C_SPLT if is_split else C_KEPT
            _draw_cnt(cnt, color)
            pt = (int(round(pt_f[0])), int(round(pt_f[1])))
            cv2.circle(vis, pt, 2, color, -1)

        if warm_mode:
            # Return the annotated crop patch; the caller composites and saves.
            if vis_patch_out is not None:
                vis_patch_out.append(vis)
        else:
            # Cold path: add legend strip and save.
            strip_entries = [
                (C_KEPT,     "kept"),
                (C_SPLT,     "split"),
                (C_CIRC,     f"not circular (< {min_circularity})"),
                (C_SPAT,     "spatial outlier"),
                (C_INTERIOR, "deep in large blob (no-prior)"),
                (C_THRESH,   f"too dim (< {required_threshold})"),
                (C_AREA_LG,  f"area > {int(max_area)}px"),
                (C_AREA_SM,  f"pixels < {int(min_area)}"),
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
                suffix_stripped = vis_suffix.lstrip("_")
                pass_type = "main"
                for _pt in ("pass1", "pass2", "pose", "local"):
                    if suffix_stripped.endswith("_" + _pt):
                        pass_type = _pt
                        suffix_stripped = suffix_stripped[: -len("_" + _pt)]
                        break
                out_dir = Path("./visualization/blobs") / (suffix_stripped or "default") / pass_type
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / f"{Path(img_path).stem}.png"), vis)
            vis_out = vis

    return (filtered_centroids, filtered_contours, filtered_radii,
            filtered_brightnesses,
            rejected_centroids, rejected_contours, large_rejected_contours,
            vis_out)


def _correct_radii_for_threshold(radii: np.ndarray, brightnesses: np.ndarray,
                                 thr1: int, thr2: int) -> np.ndarray:
    """Scale pass-2 radii to approximate the physical apparent LED size.

    Pass 2 uses a higher pixel threshold (thr2 > thr1), so its contours only
    cover the bright core — smaller than the true LED disc.  For a Gaussian
    intensity profile with peak P the blob radius at threshold T satisfies
    r(T) ∝ sqrt(ln(P/T)), giving the correction:

        r_corrected = r_pass2 * sqrt( ln(P/thr1) / ln(P/thr2) )

    Scale is capped at 4 to guard against near-threshold blobs where the
    denominator approaches zero.
    """
    if thr2 <= thr1 or len(radii) == 0:
        return radii
    r = radii.copy()
    P = np.asarray(brightnesses, dtype=np.float64)
    safe_P = np.maximum(P, thr2 + 1)              # ensure ln > 0 in denominator
    ln_t2  = np.log(safe_P / thr2)
    ln_t1  = np.log(np.maximum(safe_P / thr1, 1.0 + 1e-9))
    scale  = np.where(ln_t2 > 1e-9, np.sqrt(ln_t1 / ln_t2), 1.0)
    scale  = np.minimum(scale, 4.0)
    r *= scale.astype(np.float32)
    return r


def _eval_model(block: dict, depth, facing_cos=1.0):
    """Evaluate threshold model: a * facing_cos / depth_m² + b.
    depth and facing_cos may be scalars or numpy arrays.
    """
    return block["a"] * np.asarray(facing_cos) / np.asarray(depth) ** 2 + block["b"]


def _nms_blobs(all_blobs, min_split_dist):
    """
    Deduplicate blobs from per-LED detection runs.

    Two blobs are considered the same physical detection when their centroids are
    within min_split_dist pixels of each other.  In that case the blob detected
    under the higher pixel_threshold is kept (more discriminating detection).
    Blobs with genuinely different centroids always both survive — even if one is
    brighter, they may belong to different LEDs from different controllers.
    """
    # Higher pixel_threshold = more discriminating = preferred in a tie.
    all_blobs = sorted(all_blobs, key=lambda b: -b["pixel_threshold"])
    kept = []
    for blob in all_blobs:
        cx, cy = blob["cx"], blob["cy"]
        is_dup = any(
            np.hypot(cx - k["cx"], cy - k["cy"]) < min_split_dist
            for k in kept
        )
        if not is_dup:
            kept.append(blob)
    return kept


def _detect_blobs_local(image, led_projections, cfg,
                        visualize=False, img_path=None, vis_suffix="",
                        search_radius_px=None, threshold_scale: float = 1.0):
    """
    Pose-guided per-LED local blob detection.

    led_projections : (N, 5) float32 — [proj_x, proj_y, depth_m, facing_cos, led_id]

    For each LED a small ROI crop is extracted and _detect_blobs() is run with
    per-LED thresholds computed from: a * facing_cos / depth_m² + b.
    Centroids and contours are remapped to full-image coordinates.
    Near-duplicate blobs (centroid distance < min_split_dist) from overlapping
    ROIs are deduplicated by NMS, preferring the detection with the higher
    pixel_threshold.

    Returns the same 7-tuple as _detect_blobs().
    """
    pg             = cfg.get("pose_guided_thresholds")
    base_px        = float(search_radius_px if search_radius_px is not None else 8.0)
    depth_k        = float(cfg.get("local_search_depth_k", 0.0))
    base_pixel_thr = int(cfg["min_threshold"])
    req_factor     = float(cfg.get("required_threshold_factor", 1.5))
    min_split_dist = float(cfg.get("min_split_dist", 4.0))

    h_img, w_img = image.shape[:2]
    n_leds       = len(led_projections)

    # ── Vectorised per-LED threshold computation ──────────────────────────────
    depths    = led_projections[:, 2].astype(np.float64)
    cos_vals  = led_projections[:, 3].astype(np.float64)
    safe_d    = np.maximum(depths, 0.01)

    base_min_area = float(cfg["min_area"])
    if pg is not None:
        pixel_thrs = np.maximum(
            _eval_model(pg["pixel_threshold"], safe_d, cos_vals).astype(int),
            base_pixel_thr,
        )
        max_areas  = _eval_model(pg["max_area"], safe_d, cos_vals)
        if "min_area" in pg:
            min_areas = np.maximum(_eval_model(pg["min_area"], safe_d, cos_vals), base_min_area)
        else:
            min_areas = np.full(n_leds, base_min_area)
    else:
        pixel_thrs = np.full(n_leds, base_pixel_thr, dtype=int)
        max_areas  = np.full(n_leds, None,            dtype=object)
        min_areas  = np.full(n_leds, base_min_area)
    if threshold_scale != 1.0:
        pixel_thrs = np.maximum((pixel_thrs * threshold_scale).astype(int), 1)
    req_thrs = np.clip((pixel_thrs * req_factor).astype(int), 0, 255)

    # Per-LED search radii
    search_rs = np.round(base_px + depth_k / safe_d).astype(int)

    # ── Per-LED ROI detection ─────────────────────────────────────────────────
    all_blobs           = []   # list of dicts for NMS
    all_large_rejected  = []   # union of large blobs for the return tuple
    vis_patches         = []   # (patch_bgr, x1, y1, led_idx) — warm vis only

    for i in range(n_leds):
        proj_x, proj_y, depth_m, facing_cos, led_id = led_projections[i]
        led_id = int(led_id)
        sr = int(search_rs[i])

        # Bounding box of the circular ROI, clamped to image bounds
        cx_i = int(round(proj_x))
        cy_i = int(round(proj_y))
        x1 = max(0, cx_i - sr)
        y1 = max(0, cy_i - sr)
        x2 = min(w_img, cx_i + sr + 1)
        y2 = min(h_img, cy_i + sr + 1)
        if x2 <= x1 or y2 <= y1:
            continue

        # Mask pixels outside the circle within the crop
        crop = image[y1:y2, x1:x2].copy()
        ch, cw = crop.shape[:2]
        gy, gx = np.ogrid[:ch, :cw]
        circle_mask = ((gx - (cx_i - x1)) ** 2 + (gy - (cy_i - y1)) ** 2) <= sr ** 2
        crop[~circle_mask] = 0

        pixel_thr_i = int(pixel_thrs[i])
        req_thr_i   = int(req_thrs[i])
        max_area_i  = float(max_areas[i]) if max_areas[i] is not None else None
        min_area_i  = float(min_areas[i])

        patch_out = [] if visualize else None
        result = _detect_blobs(
            crop, pixel_thr_i, req_thr_i, cfg,
            visualize=visualize, img_path=None, vis_suffix="",
            max_area_override=max_area_i,
            min_area_override=min_area_i,
            warm_mode=True,
            vis_patch_out=patch_out,
        )

        if visualize and patch_out:
            vis_patches.append((patch_out[0], x1, y1, i, pixel_thr_i, req_thr_i))

        centroids_crop, contours_crop, radii, brightnesses, \
            rej_centroids_crop, rej_contours_crop, large_rejected, _ = result

        # Remap centroids and contours to full-image coordinates
        if len(centroids_crop) > 0:
            centroids_full = centroids_crop + np.array([[x1, y1]], dtype=np.float32)
        else:
            centroids_full = centroids_crop

        contours_full     = [cnt + np.array([[x1, y1]], dtype=np.float32) for cnt in contours_crop]
        rej_centroids_full = (rej_centroids_crop + np.array([[x1, y1]], dtype=np.float32)
                              if len(rej_centroids_crop) > 0 else rej_centroids_crop)
        rej_contours_full = [cnt + np.array([[x1, y1]], dtype=np.float32) for cnt in rej_contours_crop]

        all_large_rejected.extend(
            cnt + np.array([[x1, y1]], dtype=np.float32) for cnt in large_rejected
        )

        for j in range(len(centroids_full)):
            all_blobs.append({
                "cx":              float(centroids_full[j, 0]),
                "cy":              float(centroids_full[j, 1]),
                "contour":         contours_full[j],
                "radius":          float(radii[j]),
                "brightness":      float(brightnesses[j]),
                "pixel_threshold": pixel_thr_i,
                "source_led_idx":  i,
            })

    # ── NMS deduplication ─────────────────────────────────────────────────────
    kept_blobs = _nms_blobs(all_blobs, min_split_dist)

    if kept_blobs:
        filtered_centroids   = np.array([[b["cx"], b["cy"]] for b in kept_blobs], dtype=np.float32)
        filtered_contours    = [b["contour"]  for b in kept_blobs]
        filtered_radii       = np.array([b["radius"]     for b in kept_blobs], dtype=np.float32)
        filtered_brightnesses = np.array([b["brightness"] for b in kept_blobs], dtype=np.float32)
    else:
        filtered_centroids    = np.empty((0, 2), dtype=np.float32)
        filtered_contours     = []
        filtered_radii        = np.empty(0, dtype=np.float32)
        filtered_brightnesses = np.empty(0, dtype=np.float32)

    # ── Composite visualization ───────────────────────────────────────────────
    canvas_out = None
    if visualize:
        C_ROI  = (0, 180, 0)    # green — search circle
        C_KEPT = (255, 255, 255)
        C_AREA_SM = (255, 0, 255)
        C_AREA_LG = (0, 140, 255)
        C_THRESH  = (180, 0, 0)
        C_CIRC    = (0, 255, 255)
        min_circularity = float(cfg.get("min_circularity", 0.5))

        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Search circles + LED ID labels first (so blob pixels are drawn on top)
        for i in range(n_leds):
            proj_x, proj_y = float(led_projections[i, 0]), float(led_projections[i, 1])
            led_id_vis     = int(led_projections[i, 4])
            sr = int(search_rs[i])
            cx_draw = int(round(proj_x))
            cy_draw = int(round(proj_y))
            cv2.circle(canvas, (cx_draw, cy_draw), sr, C_ROI, 1)
            label_pos = (cx_draw + sr + 2, cy_draw - sr // 2 if cy_draw - sr // 2 > 10 else cy_draw + sr + 10)
            cv2.putText(canvas, str(led_id_vis), label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_ROI, 1, cv2.LINE_AA)

        # Paint only colored annotation pixels from each patch (background is black)
        for patch, x1, y1, led_idx, pix_thr, req_thr in vis_patches:
            ph, pw = patch.shape[:2]
            dst  = canvas[y1:y1 + ph, x1:x1 + pw]
            mask = patch.any(axis=2)
            dst[mask] = patch[mask]

        # Warm-mode legend strip
        strip_entries = [
            (C_KEPT,     "kept"),
            ((0, 255, 128), "split"),
            (C_CIRC,     f"not circular (< {min_circularity})"),
            (C_THRESH,   f"too dim (< req_thr)"),
            (C_AREA_LG,  f"area > max_area"),
            (C_AREA_SM,  "pixels < min_area"),
        ]
        row_h = 20
        half  = len(strip_entries) // 2 + len(strip_entries) % 2
        rows  = [strip_entries[:half], strip_entries[half:]]
        strip = np.zeros((row_h * 3, canvas.shape[1], 3), dtype=np.uint8)
        for row_idx, row in enumerate(rows):
            x = 6
            y_sq_top = row_idx * row_h + 5
            y_sq_bot = y_sq_top + 10
            y_text   = y_sq_bot - 1
            for color, label in row:
                cv2.rectangle(strip, (x, y_sq_top), (x + 10, y_sq_bot), color, -1)
                x += 13
                cv2.putText(strip, label, (x, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
                (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                x += tw + 10
        cv2.putText(strip, f"warm: {n_leds} LEDs  base_r={base_px:.1f}px  NMS min_dist={min_split_dist:.1f}px",
                    (6, row_h * 2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        canvas = np.vstack([canvas, strip])

        if img_path is not None:
            suffix_stripped = vis_suffix.lstrip("_")
            out_dir = Path("./visualization/blobs") / (suffix_stripped or "default") / "local"
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{Path(img_path).stem}.png"), canvas)
        canvas_out = canvas

    return (
        filtered_centroids,
        filtered_contours,
        filtered_radii,
        filtered_brightnesses,
        np.empty((0, 2), dtype=np.float32),  # rejected centroids (pre-NMS rejects are in patches)
        [],                                   # rejected contours
        all_large_rejected,
        canvas_out,
    )


@dataclass
class BlobResult:
    """Detection output for one (camera, controller, frame) triplet."""
    centroids:    np.ndarray   # (N, 2) float32
    radii:        np.ndarray   # (N,)   float32
    brightnesses: np.ndarray   # (N,)   float32
    contours:     List         # N × (M, 2) float32 contour arrays

    def __len__(self) -> int:
        return len(self.centroids)

    def filter(self, mask: np.ndarray) -> "BlobResult":
        """Return a new BlobResult containing only blobs where mask is True."""
        idx = np.where(mask)[0]
        return BlobResult(
            centroids=self.centroids[idx],
            radii=self.radii[idx],
            brightnesses=self.brightnesses[idx],
            contours=[self.contours[i] for i in idx],
        )

    @staticmethod
    def empty() -> "BlobResult":
        return BlobResult(
            centroids=np.empty((0, 2), dtype=np.float32),
            radii=np.empty(0, dtype=np.float32),
            brightnesses=np.empty(0, dtype=np.float32),
            contours=[],
        )


class BlobDetector:
    """
    Per-camera blob detector. Holds per-camera EMA pass-2 state as instance
    state, replacing the module-level _pass2_memory global in blobs_detection.py.
    One instance per camera — instances are independent and safe to use from
    separate threads (different cameras).
    """

    def __init__(self, camera_idx: int, cfg: dict):
        self.camera_idx = camera_idx
        self._cfg = cfg
        self._memory: dict = {}

    @property
    def pass2_params(self) -> Optional[dict]:
        """Return the pass-2 thresholds last used, or None if pass-2 has not run."""
        if not self._memory or "pixel_threshold" not in self._memory:
            return None
        return {
            "pixel_threshold":    self._memory["pixel_threshold"],
            "required_threshold": self._memory.get("required_threshold"),
            "max_area":           self._memory.get("max_area"),
        }

    def detect(
        self,
        image: np.ndarray,
        ctrl_label: str = "",
        predicted_leds: Optional[np.ndarray] = None,
        local_search_radius_px: float = 0.0,
        threshold_scale: float = 1.0,
        visualize: bool = False,
        img_path=None,
    ) -> BlobResult:
        cfg     = self._cfg
        cam_idx = self.camera_idx

        pixel_threshold    = max(int(cfg["min_threshold"] * threshold_scale), 1)
        _req_factor        = float(cfg.get("required_threshold_factor", 1.5))
        required_threshold = min(int(pixel_threshold * _req_factor), 255)

        cam_str  = f"_cam{cam_idx}"
        ctrl_str = f"_{ctrl_label}" if ctrl_label else ""

        # ── Warm path: per-LED local search (no EMA state access) ────────────
        if predicted_leds is not None and len(predicted_leds) > 0:
            raw = _detect_blobs_local(
                image, predicted_leds, cfg,
                visualize, img_path, f"{cam_str}{ctrl_str}",
                search_radius_px=local_search_radius_px,
                threshold_scale=threshold_scale,
            )
            canvases = {"local": raw[7]} if raw[7] is not None else {}
            return (BlobResult(centroids=raw[0], contours=raw[1],
                               radii=raw[2], brightnesses=raw[3]), canvases)

        # ── Cold path: global search with optional pass-2 EMA ────────────────
        pass2_factor                = float(cfg.get("pass2_threshold_factor", 0.0))
        pass2_required_factor       = float(cfg.get("pass2_required_factor", 0.7))
        pass2_brightness_percentile = float(cfg.get("pass2_brightness_percentile", 25.0))
        ema_alpha                   = float(cfg.get("pass2_threshold_ema_alpha", 1.0))
        count_gate_max_factor       = float(cfg.get("pass2_count_gate_max_factor", 1.7))
        _mem = self._memory

        vis_suffix_1 = f"{cam_str}{ctrl_str}_pass1" if pass2_factor > 0 else f"{cam_str}{ctrl_str}"
        result = _detect_blobs(image, pixel_threshold, required_threshold, cfg,
                               visualize, img_path, vis_suffix_1)
        large_blobs_pass1 = result[6]
        canvases = {}
        if result[7] is not None:
            canvases["pass1"] = result[7]

        min_area_pass2 = cfg.get("pass2_min_area")
        min_area_pass2 = float(min_area_pass2) if min_area_pass2 is not None else None

        if pass2_factor > 0 and len(result[0]) >= 2:
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
                px = image[yb:yb + hb, xb:xb + wb][roi_m > 0]
                if px.size > 0:
                    mean_vals.append(float(np.mean(px)))
                area_vals.append(float(cv2.contourArea(cnt_int)))
            if mean_vals:
                ref_brightness   = float(np.percentile(mean_vals, pass2_brightness_percentile))
                raw_pixel_thr    = int(ref_brightness * pass2_factor)
                raw_required_thr = int(ref_brightness * pass2_required_factor)
                if raw_pixel_thr > pixel_threshold:
                    cur_count    = len(mean_vals)
                    prev_count   = _mem.get("blob_count", cur_count)
                    count_stable = (max(cur_count, prev_count) /
                                    max(min(cur_count, prev_count), 1) <= count_gate_max_factor)
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
                        pixel_threshold_2    = _mem["pixel_threshold"]
                        required_threshold_2 = _mem.get("required_threshold", required_threshold)
                        update_memory        = False
                    max_area_pass2 = float(max(area_vals)) if area_vals else None
                    result2 = _detect_blobs(
                        image, pixel_threshold_2, required_threshold_2, cfg,
                        visualize, img_path, f"{cam_str}{ctrl_str}_pass2",
                        max_area_override=max_area_pass2,
                        min_area_override=min_area_pass2,
                        interior_exclude_blobs=large_blobs_pass1,
                    )
                    if result2[7] is not None:
                        canvases["pass2"] = result2[7]
                    if len(result2[0]) >= 1:
                        if update_memory:
                            _mem["pixel_threshold"]    = pixel_threshold_2
                            _mem["required_threshold"] = required_threshold_2
                            _mem["max_area"]           = max_area_pass2
                            _mem["large_blobs"]        = large_blobs_pass1
                            _mem["blob_count"]         = cur_count
                        corrected_radii = _correct_radii_for_threshold(
                            result2[2], result2[3], pixel_threshold, pixel_threshold_2)
                        return (BlobResult(centroids=result2[0], contours=result2[1],
                                           radii=corrected_radii, brightnesses=result2[3]),
                                canvases)

        elif pass2_factor > 0 and _mem:
            result2 = _detect_blobs(
                image, _mem["pixel_threshold"],
                _mem.get("required_threshold", required_threshold), cfg,
                visualize, img_path, f"{cam_str}{ctrl_str}_pass2",
                max_area_override=_mem["max_area"],
                min_area_override=min_area_pass2,
                interior_exclude_blobs=large_blobs_pass1,
            )
            if result2[7] is not None:
                canvases["pass2"] = result2[7]
            if len(result2[0]) >= 1:
                corrected_radii = _correct_radii_for_threshold(
                    result2[2], result2[3], pixel_threshold, _mem["pixel_threshold"])
                return (BlobResult(centroids=result2[0], contours=result2[1],
                                   radii=corrected_radii, brightnesses=result2[3]),
                        canvases)
            _mem.clear()

        return (BlobResult(centroids=result[0], contours=result[1],
                           radii=result[2], brightnesses=result[3]),
                canvases)
