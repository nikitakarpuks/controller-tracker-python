import math

import cv2
import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from src.debug_config import is_deep, get_debug_triple, is_verbose_all, log_best
from src._pnp import _ransac_pnp, _project_points, _check_z_range
from src._visibility import _visible_mask
from src._led_graph import _build_blob_neighbor_lists


# ---------------------------------------------------------------------------
# brute_match gate helpers
# ---------------------------------------------------------------------------

def _gate_any_point(
    R_h: np.ndarray, tvec_h: np.ndarray,
    gate_obj: np.ndarray,
    gate_img: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    thresh_sq: float,
) -> Tuple[bool, float]:
    """
    Return True if ANY gate LED projects within sqrt(thresh_sq) pixels of ANY gate blob.
    If either pool is empty, returns True (no gate to fail).
    Returns (passed, min_dist_px); min_dist is only tracked in deep-debug mode.
    """
    if len(gate_obj) == 0 or len(gate_img) == 0:
        return True, 0.0

    track_dist = is_deep()
    min_dist   = np.inf if track_dist else 0.0
    for obj in gate_obj:
        p = R_h @ obj + tvec_h
        if p[2] <= 0:
            continue
        iz = 1.0 / p[2]
        px = fx * p[0] * iz + cx
        py = fy * p[1] * iz + cy
        for img in gate_img:
            dx = px - img[0]
            dy = py - img[1]
            if track_dist:
                min_dist = min(min_dist, math.sqrt(dx * dx + dy * dy))
            if dx * dx + dy * dy <= thresh_sq:
                return True, min_dist
    return False, min_dist


def _gate_fourth_point(
    R_h: np.ndarray, tvec_h: np.ndarray,
    obj4: np.ndarray, img4: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    p4_thresh_sq: float,
) -> bool:
    """
    Fast single-point reprojection gate (no distortion, inline perspective divide).
    Return True if the 4th LED projects within sqrt(p4_thresh_sq) pixels of img4.
    """
    p4_cam = R_h @ obj4 + tvec_h
    if p4_cam[2] <= 0:
        return False
    iz = 1.0 / p4_cam[2]
    dx = fx * p4_cam[0] * iz + cx - img4[0]
    dy = fy * p4_cam[1] * iz + cy - img4[1]
    return dx * dx + dy * dy <= p4_thresh_sq


def _tier_label(t):
    nbr = t[2] if len(t) > 2 else 'standard'
    return f"led≤{t[0]}, blob≤{t[1]}, nbr={nbr}"


# ---------------------------------------------------------------------------
# blob LED-ID carry helper
# ---------------------------------------------------------------------------

def _carry_led_ids(
    current_blobs: np.ndarray,   # (N, 2)
    prev_positions: np.ndarray,  # (M, 2)
    prev_led_ids: np.ndarray,    # (M,) int, -1 = unmatched
    snap_px: float,
) -> np.ndarray:                 # (N,) int, -1 = unmatched
    """
    Nearest-neighbour blob tracking across frames.
    Each current blob inherits the led_id of its nearest previous blob if within snap_px.
    No 1-to-1 enforcement: conflicts (two blobs claiming the same id) are resolved
    downstream by proximity_match dropping to argmin when len(candidates) != 1.
    """
    led_ids = np.full(len(current_blobs), -1, dtype=np.int32)
    if len(prev_positions) == 0:
        return led_ids
    for i, blob in enumerate(current_blobs):
        dists = np.linalg.norm(prev_positions - blob, axis=1)
        j = int(np.argmin(dists))
        if dists[j] < snap_px and prev_led_ids[j] != -1:
            led_ids[i] = int(prev_led_ids[j])
    return led_ids


# ---------------------------------------------------------------------------
# proximity_match
# ---------------------------------------------------------------------------

def proximity_match(
    self,
    blobs: np.ndarray,
    predicted_pose: Tuple[np.ndarray, np.ndarray],
    prior_assignment: Optional[List] = None,
    blob_led_ids: Optional[np.ndarray] = None,
    blob_radii: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    """
    Refine a predicted pose using the assignment-locked nearest-neighbour path.

    For each LED from the previous frame's assignment, project it with the
    predicted pose and snap to its nearest blob. The snap radius is
    depth-scaled: snap_px = focal * (led_radius_mm/1000) / depth * snap_factor.
    RANSAC + visibility recheck then filter the locked pairs.
    Returns None when too few pairs survive; caller falls back to brute_match.
    """
    rvec_pred, tvec_pred = predicted_pose
    rvec_pred = np.asarray(rvec_pred, dtype=np.float32).reshape(3, 1)
    tvec_pred = np.asarray(tvec_pred, dtype=np.float32).reshape(3)

    K  = self.camera.camera_matrix
    dc = self.camera.dist_coeffs

    geom = self._geometry
    _cfg = getattr(self, '_matching_cfg', None) or {}
    facing_threshold_deg   = float(_cfg.get('led_facing_angle_deg',     86.0))
    reprojection_threshold = float(_cfg.get('reprojection_threshold',    2.0))
    min_inliers            = int(  _cfg.get('min_inliers',               4))
    led_radius_mm          = float(_cfg.get('led_radius_mm',             2.5))
    snap_factor            = float(_cfg.get('proximity_snap_factor',     4.0))
    blob_size_max_factor   = float(_cfg.get('blob_size_max_factor',      4.0))
    blob_size_min_factor   = float(_cfg.get('blob_size_min_factor',      0.2))
    blob_size_score_weight = float(_cfg.get('blob_size_score_weight',    0.5))

    if prior_assignment is None or len(prior_assignment) < min_inliers:
        return None

    prior_lids = [lid for _, lid in prior_assignment]
    prior_obj  = self.model.positions[prior_lids].astype(np.float32)
    proj_prior = _project_points(rvec_pred, tvec_pred, prior_obj, K, dc)

    # Pre-rotate all prior LED positions into camera frame to get per-LED depth.
    R_pred_arr, _ = cv2.Rodrigues(rvec_pred)
    led_cam = (R_pred_arr @ prior_obj.T).T + tvec_pred  # (N, 3)
    focal_px = float(max(K[0, 0], K[1, 1]))

    # Compute model visibility with the predicted pose up front — needed both for
    # the vis-drop expansion trigger and for the post-snap expansion block.
    expansion_threshold   = float(_cfg.get('proximity_expansion_threshold', 0.6))
    expansion_px          = float(_cfg.get('proximity_expansion_px',        5.0))
    vis_drop_threshold    = int(  _cfg.get('proximity_vis_drop_threshold',  1))

    vis_mask_pred = _visible_mask(
        R_pred_arr, tvec_pred,
        self.model.positions, self.model.normals,
        geom,
        cam_K=K, cam_dc=dc, cam_w=self.camera.width, cam_h=self.camera.height,
        cam_rpmax=self.camera.rpmax,
        facing_threshold_deg=facing_threshold_deg,
    )
    n_model_visible = int(vis_mask_pred.sum())

    # Snap: assign blobs to prior LEDs in two passes.
    # Direction is blob→LED: each blob claims its nearest unclaimed prior LED.
    # This prevents an occluded LED's projection from stealing a blob that belongs
    # to a still-visible neighbour — occluded LEDs simply receive no blob and are
    # absent from locked_pairs (natural occlusion inference, no vis_mask needed).
    #
    # Pass 1 (ID fast path): blobs that carry a LED ID from the previous frame are
    #   matched to that LED directly, subject to size + distance checks.
    # Pass 2 (blob→LED greedy): each remaining blob finds its nearest unclaimed
    #   prior LED within the LED's depth-scaled snap radius and argmin_max_px cap.
    #   Score = dist + blob_size_score_weight * size_err to prefer size-correct matches.
    argmin_max_px = float(_cfg.get('proximity_argmin_max_dist_px',
                                    float(_cfg.get('proximity_expansion_px', 8.0))))

    # Precompute per-LED depth-scaled snap parameters.
    expected_pxs = np.zeros(len(prior_lids))
    snap_pxs     = np.zeros(len(prior_lids))
    for i in range(len(prior_lids)):
        depth            = float(max(led_cam[i, 2], 0.01))
        expected_pxs[i]  = focal_px * (led_radius_mm / 1000.0) / depth
        snap_pxs[i]      = expected_pxs[i] * snap_factor

    # Index for O(1) LED-ID → prior index lookup used in the ID fast path.
    prior_lid_to_idx = {lid: i for i, lid in enumerate(prior_lids)}

    locked_pairs, locked_obj, locked_img = [], [], []
    used_blobs: set = set()
    used_led_idxs: set = set()
    n_id_locked = 0

    # Pass 1: ID fast path.
    if blob_led_ids is not None:
        for j in range(len(blobs)):
            lid = int(blob_led_ids[j])
            if lid == -1:
                continue
            i = prior_lid_to_idx.get(lid, -1)
            if i == -1 or i in used_led_idxs:
                continue
            dist = float(np.linalg.norm(blobs[j] - proj_prior[i]))
            if dist >= snap_pxs[i]:
                continue
            if blob_radii is not None:
                if not (expected_pxs[i] * blob_size_min_factor
                        <= blob_radii[j]
                        <= expected_pxs[i] * blob_size_max_factor):
                    continue
            locked_pairs.append((j, lid))
            locked_obj.append(self.model.positions[lid])
            locked_img.append(blobs[j])
            used_blobs.add(j)
            used_led_idxs.add(i)
            n_id_locked += 1
            if is_deep():
                logger.debug(f"  LED {lid}: ID-path → blob {j}")

    # Pass 2: blob→LED greedy for unmatched blobs.
    for j in range(len(blobs)):
        if j in used_blobs:
            continue
        blob_r     = blob_radii[j] if blob_radii is not None else None
        best_i     = -1
        best_score = float('inf')
        best_dist  = float('inf')
        for i, lid in enumerate(prior_lids):
            if i in used_led_idxs:
                continue
            dist = float(np.linalg.norm(blobs[j] - proj_prior[i]))
            if dist >= snap_pxs[i] or dist >= argmin_max_px:
                continue
            if blob_r is not None:
                if not (expected_pxs[i] * blob_size_min_factor
                        <= blob_r
                        <= expected_pxs[i] * blob_size_max_factor):
                    continue
            size_err = float(abs(blob_r - expected_pxs[i])) if blob_r is not None else 0.0
            score    = dist + blob_size_score_weight * size_err
            if score < best_score:
                best_score = score
                best_i     = i
                best_dist  = dist
        if best_i >= 0:
            lid = prior_lids[best_i]
            locked_pairs.append((j, lid))
            locked_obj.append(self.model.positions[lid])
            locked_img.append(blobs[j])
            used_blobs.add(j)
            used_led_idxs.add(best_i)
            if is_deep():
                logger.debug(f"  LED {lid}: blob {j} → nearest ({best_dist:.1f}px)")

    logger.debug(
        f"Proximity snap: {len(locked_pairs)}/{len(blobs)} blobs locked "
        f"({n_id_locked} ID-path, {len(locked_pairs) - n_id_locked} nearest) "
        f"of {len(prior_lids)} prior LEDs"
        + ("  [size filter active]" if blob_radii is not None else "")
    )
    if len(locked_pairs) < min_inliers:
        logger.debug(f"Proximity: too few pairs ({len(locked_pairs)} < {min_inliers}) → None")
        return None

    # Hungarian expansion: triggered when proximity locked fewer than expansion_threshold
    # of model-visible LEDs, OR when visible LED count dropped significantly from the
    # prior (vis_drop_threshold) and there are still unassigned blobs to place.
    # Runs with the predicted pose; RANSAC downstream filters bad pairs.
    locked_blob_set = {b for b, _ in locked_pairs}
    locked_led_set  = {l for _, l in locked_pairs}

    free_blob_idx = [i for i in range(len(blobs)) if i not in locked_blob_set]
    free_led_idx  = [int(lid) for lid in np.where(vis_mask_pred)[0]
                     if int(lid) not in locked_led_set]

    n_prior_gone = sum(1 for lid in prior_lids if not vis_mask_pred[lid])
    vis_dropped  = n_prior_gone > vis_drop_threshold
    did_expand  = False

    if n_model_visible > 0 and (
        len(locked_pairs) / n_model_visible < expansion_threshold
        or (vis_dropped and len(free_blob_idx) > 0)
    ):
        if free_blob_idx and free_led_idx:
            free_led_obj = self.model.positions[free_led_idx].astype(np.float32)
            proj_free    = _project_points(rvec_pred, tvec_pred, free_led_obj, K, dc)
            free_blobs   = blobs[free_blob_idx]

            cost = cdist(proj_free, free_blobs)  # (n_free_leds, n_free_blobs)

            if blob_radii is not None:
                free_blob_radii = blob_radii[free_blob_idx]
                free_led_cam    = (R_pred_arr @ free_led_obj.T).T + tvec_pred
                for k in range(len(free_led_idx)):
                    depth       = float(max(free_led_cam[k, 2], 0.01))
                    expected_px = focal_px * (led_radius_mm / 1000.0) / depth
                    ineligible  = (
                        (free_blob_radii < expected_px * blob_size_min_factor) |
                        (free_blob_radii > expected_px * blob_size_max_factor)
                    )
                    cost[k, ineligible] = 1e9

            row_ind, col_ind = linear_sum_assignment(cost)

            n_expanded = 0
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < expansion_px:
                    blob_j = free_blob_idx[c]
                    led_id = free_led_idx[r]
                    locked_pairs.append((blob_j, led_id))
                    locked_obj.append(self.model.positions[led_id])
                    locked_img.append(blobs[blob_j])
                    n_expanded += 1

            if n_expanded:
                did_expand = True
                logger.debug(
                    f"Proximity expansion: +{n_expanded} via Hungarian "
                    f"(coverage {len(locked_pairs) - n_expanded}/{n_model_visible} "
                    f"→ {len(locked_pairs)}/{n_model_visible})"
                )

    lo = np.array(locked_obj, dtype=np.float32)
    li = np.array(locked_img, dtype=np.float32)

    ok, rvec, tvec, ransac_idx = _ransac_pnp(
        lo, li, K, dc, rvec_pred, tvec_pred,
        reprojection_px=reprojection_threshold,
    )
    if not ok:
        logger.debug("Proximity: RANSAC failed → None")
        return None

    # Keep all RANSAC inliers — the visibility check is an approximation and
    # the blob's physical presence is stronger evidence than the model geometry.
    final_pairs = [locked_pairs[k] for k in ransac_idx]

    if len(final_pairs) < min_inliers:
        logger.debug(f"Proximity: too few inliers ({len(final_pairs)} < {min_inliers}) → None")
        return None

    lo_f  = self.model.positions[[l for _, l in final_pairs]].astype(np.float32)
    li_f  = blobs[[b for b, _ in final_pairs]].astype(np.float32)
    proj  = _project_points(rvec, tvec, lo_f, K, dc)
    pair_errors = np.linalg.norm(proj - li_f, axis=1)
    error     = float(pair_errors.mean())
    max_error = float(pair_errors.max())

    logger.debug(f"Proximity: OK  inliers={len(final_pairs)}  err={error:.2f}px  max={max_error:.2f}px")
    return {
        "rvec":       rvec,
        "tvec":       tvec,
        "error":      error,
        "max_error":  max_error,
        "assignment": final_pairs,
        "method":     "proximity_expanded" if did_expand else "proximity_locked",
    }


# ---------------------------------------------------------------------------
# prior_constrained_match
# ---------------------------------------------------------------------------

def prior_constrained_match(
    self,
    blobs: np.ndarray,
    predicted_pose: Tuple[np.ndarray, np.ndarray],
    prior_assignment: Optional[List] = None,
    blob_radii: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    """
    Prior-constrained translation solver for 2–3 blob frames.

    When only 2–3 blobs are visible, P3P/RANSAC cannot run.  This solver
    fixes the rotation from the predicted pose as a hard constraint and
    collapses the 6-DOF problem to translation-only (3-DOF).

    P2P mode  (n_blobs == 3):
        Project prior LEDs → snap 3 pairs.
        Build a 4×3 overdetermined system from 2 pairs (undistorted normalised
        coordinates) and solve for (tx, ty, tz) via least-squares.
        Validate with the 3rd pair.

    P1P mode  (n_blobs == 2):
        Project prior LEDs → snap 2 pairs.
        Additionally fix tz = tvec_prior[2] (single-camera depth cannot be
        recovered from one correspondence alone; the prior depth is the best
        available estimate).
        Solve (tx, ty) analytically from 1 pair, validate with the 2nd.

    All pairs must reproject within reprojection_threshold after solving —
    there are too few matches to average away a wrong correspondence.

    Note: P1P depth-fixation approximates the stereo triangulation used in
    the original multi-camera implementation (Meta blog, 2021).  It works
    well for slow radial motion but drifts when depth changes quickly.
    """
    blobs   = np.asarray(blobs, dtype=np.float32)
    n_blobs = len(blobs)

    if n_blobs < 2:
        return None
    if prior_assignment is None or len(prior_assignment) < 2:
        return None

    rvec_pred, tvec_pred = predicted_pose
    rvec_pred  = np.asarray(rvec_pred, dtype=np.float32).reshape(3, 1)
    tvec_pred  = np.asarray(tvec_pred, dtype=np.float32).reshape(3)
    R_prior, _ = cv2.Rodrigues(rvec_pred)

    K  = self.camera.camera_matrix
    dc = self.camera.dist_coeffs

    _cfg = getattr(self, '_matching_cfg', None) or {}
    reprojection_threshold = float(_cfg.get('reprojection_threshold', 2.0))
    led_radius_mm          = float(_cfg.get('led_radius_mm',          2.5))
    snap_factor            = float(_cfg.get('proximity_snap_factor',  4.0))
    blob_size_max_factor   = float(_cfg.get('blob_size_max_factor',   4.0))
    blob_size_min_factor   = float(_cfg.get('blob_size_min_factor',   0.2))
    blob_size_score_weight = float(_cfg.get('blob_size_score_weight', 0.5))

    mode       = 'p2p' if n_blobs >= 3 else 'p1p'
    n_required = 3     if mode == 'p2p' else 2

    # ── Snap assignment: project prior LEDs, snap to nearest unique blobs ────
    prior_lids = [lid for _, lid in prior_assignment]
    prior_obj  = self.model.positions[prior_lids].astype(np.float32)
    proj_prior = _project_points(rvec_pred, tvec_pred, prior_obj, K, dc)
    led_cam    = (R_prior @ prior_obj.T).T + tvec_pred
    focal_px   = float(max(K[0, 0], K[1, 1]))

    locked_pairs: List[Tuple[int, int]] = []
    used_blobs: set = set()

    for i, lid in enumerate(prior_lids):
        if len(locked_pairs) >= n_required:
            break

        depth       = float(max(led_cam[i, 2], 0.01))
        expected_px = focal_px * (led_radius_mm / 1000.0) / depth
        snap_px     = expected_px * snap_factor

        if blob_radii is not None:
            eligible = (
                (blob_radii >= expected_px * blob_size_min_factor) &
                (blob_radii <= expected_px * blob_size_max_factor)
            )
        else:
            eligible = None

        dists = np.linalg.norm(blobs - proj_prior[i], axis=1)

        if eligible is not None and eligible.any():
            cand_idx   = np.where(eligible)[0]
            dists_cand = dists[cand_idx]
            size_err   = np.abs(blob_radii[cand_idx] - expected_px)
            scores     = dists_cand + blob_size_score_weight * size_err
            j          = int(cand_idx[np.argmin(scores)])
        else:
            j = int(np.argmin(dists))

        if dists[j] < snap_px and j not in used_blobs:
            locked_pairs.append((j, lid))
            used_blobs.add(j)

    if len(locked_pairs) < n_required:
        logger.debug(
            f"prior_constrained ({mode}): snapped {len(locked_pairs)}/{n_required} pairs → None"
        )
        return None

    pairs_hyp = locked_pairs[:n_required - 1]   # hypothesis pairs (2 for P2P, 1 for P1P)
    pair_val  = locked_pairs[n_required - 1]     # validating pair

    # ── Undistort hypothesis blob positions → normalised image coordinates ───
    hyp_blobs  = np.array([blobs[b] for b, _ in pairs_hyp], dtype=np.float32)
    pts_undist = cv2.undistortPoints(
        hyp_blobs.reshape(-1, 1, 2), K, dc,
    ).reshape(-1, 2)    # K-removed, distortion-removed

    # ── Translation solve ─────────────────────────────────────────────────────
    # Pinhole projection with fixed R:
    #   x_n = (R[0]@P + tx) / (R[2]@P + tz)
    #   x_n*(r2 + tz) = r0 + tx
    #   tx - x_n*tz  = x_n*r2 - r0
    #   → [1, 0, -x_n] @ [tx, ty, tz] = x_n*R[2]@P - R[0]@P
    # P2P: 2 pairs → 4×3 overdetermined → least-squares
    # P1P: 1 pair + fixed tz → solve (tx, ty) directly
    if mode == 'p2p':
        A_rows: List = []
        b_rows: List = []
        for k, (_, lk) in enumerate(pairs_hyp):
            Pi  = self.model.positions[lk].astype(np.float64)
            xn  = float(pts_undist[k, 0])
            yn  = float(pts_undist[k, 1])
            r0  = float(R_prior[0] @ Pi)
            r1  = float(R_prior[1] @ Pi)
            r2  = float(R_prior[2] @ Pi)
            A_rows += [[1.0, 0.0, -xn], [0.0, 1.0, -yn]]
            b_rows += [xn * r2 - r0,    yn * r2 - r1]
        t_solved, _, _, _ = np.linalg.lstsq(
            np.array(A_rows, dtype=np.float64),
            np.array(b_rows, dtype=np.float64),
            rcond=None,
        )

    else:   # p1p: additionally fix depth from prior
        # tx = x_n*(r2 + tz) - r0,  ty = y_n*(r2 + tz) - r1
        tz  = float(tvec_pred[2])
        Pi  = self.model.positions[pairs_hyp[0][1]].astype(np.float64)
        xn  = float(pts_undist[0, 0])
        yn  = float(pts_undist[0, 1])
        r0  = float(R_prior[0] @ Pi)
        r1  = float(R_prior[1] @ Pi)
        r2  = float(R_prior[2] @ Pi)
        tx  = xn * (r2 + tz) - r0
        ty  = yn * (r2 + tz) - r1
        t_solved = np.array([tx, ty, tz])

    t_solved = np.asarray(t_solved, dtype=np.float64)

    # Depth sanity check
    if not _check_z_range(t_solved.astype(np.float32)):
        logger.debug(
            f"prior_constrained ({mode}): solved depth {t_solved[2]:.3f} m out of range → None"
        )
        return None

    # ── Validate all pairs (no averaging — too few to hide a bad match) ───────
    all_pairs = pairs_hyp + [pair_val]
    all_obj   = self.model.positions[[l for _, l in all_pairs]].astype(np.float32)
    all_img   = blobs[[b for b, _ in all_pairs]].astype(np.float32)
    proj_all  = _project_points(rvec_pred, t_solved.astype(np.float32), all_obj, K, dc)
    errors    = np.linalg.norm(proj_all - all_img, axis=1)

    if np.any(errors > reprojection_threshold):
        logger.debug(
            f"prior_constrained ({mode}): validation failed "
            f"errors={errors.round(2)} thresh={reprojection_threshold} → None"
        )
        return None

    error = float(np.mean(errors))
    logger.debug(f"prior_constrained ({mode}): OK  pairs={len(all_pairs)}  err={error:.2f}px")

    return {
        "rvec":       rvec_pred.astype(np.float64).reshape(3, 1),
        "tvec":       t_solved.reshape(3, 1),
        "error":      error,
        "assignment": all_pairs,
        "method":     f"prior_constrained_{mode}",
    }


# ---------------------------------------------------------------------------
# brute_match
# ---------------------------------------------------------------------------

def brute_match(
    self,
    blobs: np.ndarray,
    depth_tiers: Tuple[Tuple, ...] = ((2, 3), (2, 4), (2, 4, 'edge'), (3, 5), (3, 5, 'edge'), (4, 6)),  # (led_max, blob_max[, 'standard'|'edge'])
    p4_threshold_px: float = 2.0,
    hungarian_threshold_px: float = 5.0,  # pre-filter on the raw P3P hypothesis pose. Loose because P3P poses can be noisy; RANSAC does the real filtering after this
    reprojection_threshold: float = 1.5,  # passed to RANSAC, controls which blobs make it into the final assignment. This is now what the visualization reflects: all shown errors will be ≤ this
    min_inliers: int = 4,
    min_inlier_fraction: Optional[float] = None,
    strong_match_inliers: int = 7,
    strong_match_error_px: float = 1.5,
    min_vis_coverage: float = 0.75,
    pose_prior: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    rng_seed: Optional[int] = 42,
) -> Optional[Dict]:
    """
    Exhaustive pose search via P3P over LED/blob triple correspondences.
    Follows OpenHMD's correspondence_search structure:
      - Outer loop: LED triple (anchor + 2 neighbours, precomputed; no duplicates)
      - Middle loop: every blob as a potential blob anchor
      - Inner loop: C(k,2) blob pairs × 2 orderings (covers all LED–blob assignments)

    Each unique (anchor, l1, l2) ↔ (b_anchor, b1, b2) bijection is evaluated exactly once.
    Gate check: any remaining gate LED projecting near any remaining gate blob.

    Progressive 2D deepening via depth_tiers — each (led_max, blob_max) pair defines a tier.
    Each tier evaluates only the (LED triple, blob pair) combinations not covered by prior tiers:
      - LED triples newly eligible because depth ≤ led_max but depth > prev led_max
      - Blob pairs newly eligible because i2 ≥ prev blob_max for already-eligible triples
    Exits on strong match at the end of any tier's LED triple.
    """
    _cfg = getattr(self, '_matching_cfg', None) or {}
    if _cfg:
        depth_tiers            = tuple(tuple(t) for t in _cfg.get('depth_tiers', depth_tiers))
        p4_threshold_px        = float(_cfg.get('p4_threshold_px',        p4_threshold_px))
        hungarian_threshold_px = float(_cfg.get('hungarian_threshold_px', hungarian_threshold_px))
        reprojection_threshold = float(_cfg.get('reprojection_threshold', reprojection_threshold))
        min_inliers            = int(  _cfg.get('min_inliers',            min_inliers))
        min_inlier_fraction    = _cfg.get('min_inlier_fraction', min_inlier_fraction) or None
        strong_match_inliers   = int(  _cfg.get('strong_match_inliers',   strong_match_inliers))
        strong_match_error_px  = float(_cfg.get('strong_match_error_px',  strong_match_error_px))
        min_vis_coverage       = float(_cfg.get('min_vis_coverage',       min_vis_coverage))
        rng_seed               = _cfg.get('rng_seed', rng_seed)
    facing_threshold_deg = float(_cfg.get('led_facing_angle_deg', 86.0))

    blobs   = np.asarray(blobs, dtype=np.float32)
    n_blobs = len(blobs)
    if n_blobs < 4:
        return None

    if min_inlier_fraction is not None:
        fraction_floor     = int(np.ceil(min_inlier_fraction * n_blobs))
        min_inliers_eff    = max(min_inliers, fraction_floor)
        strong_inliers_eff = min(strong_match_inliers, fraction_floor)
    else:
        min_inliers_eff    = min_inliers
        strong_inliers_eff = strong_match_inliers

    positions = self.model.positions.astype(np.float32)
    normals   = self.model.normals.astype(np.float32)
    K         = self.camera.camera_matrix
    dc        = self.camera.dist_coeffs

    led_triple_idx   = self._led_triple_idx    # (N_LT, 3) int32
    led_triple_depth = self._led_triple_depth  # (N_LT,) int32
    led_triple_gates = self._led_triple_gates  # List[np.ndarray] gate LED indices per triple

    led_triple_idx_edge   = self._led_triple_idx_edge
    led_triple_depth_edge = self._led_triple_depth_edge
    led_triple_gates_edge = self._led_triple_gates_edge

    geom = self._geometry

    max_blob_depth = max(t[1] for t in depth_tiers)
    blob_nbr = _build_blob_neighbor_lists(blobs, k=max_blob_depth)

    # Undistort blobs once (mirrors OpenHMD's correspondence_search_set_blobs).
    # Gate check uses pinhole projection which is valid in undistorted space.
    blobs_undist = cv2.undistortPoints(
        blobs.reshape(-1, 1, 2), K, dc, P=K
    ).reshape(-1, 2).astype(np.float32)

    p4_thresh_sq = p4_threshold_px ** 2
    fx, fy       = float(K[0, 0]), float(K[1, 1])
    cx, cy       = float(K[0, 2]), float(K[1, 2])

    R_prior    = None
    tvec_prior = None
    if pose_prior is not None:
        rvec_pr, tvec_pr = pose_prior
        R_prior, _ = cv2.Rodrigues(np.asarray(rvec_pr, dtype=np.float32).reshape(3, 1))
        tvec_prior = np.asarray(tvec_pr, dtype=np.float32).reshape(3)

    best_solution   = None
    best_inliers    = 0
    best_error      = np.inf
    best_orient_err = np.inf
    best_tvec_err   = np.inf
    strong_found    = False
    solution_tier   = None

    # Per-triple: how far blob depth has been explored (the blob_max of the last tier
    # that processed this triple). Enables delta coverage across tiers.
    # Maintained separately for each neighbourhood type so delta tracking is consistent.
    prev_blob_max_per_triple      = np.zeros(len(led_triple_idx),      dtype=np.int32)
    prev_blob_max_per_triple_edge = np.zeros(len(led_triple_idx_edge), dtype=np.int32)

    rng = np.random.default_rng(rng_seed)

    tier_p3p_calls = [0] * len(depth_tiers)
    tier_lq_tried  = [0] * len(depth_tiers)
    tier_lq_total  = [0] * len(depth_tiers)

    _dbg_leds, _dbg_blobs = get_debug_triple()
    debug_active      = _dbg_leds is not None or _dbg_blobs is not None
    debug_led_anchor  = int(_dbg_leds[0])     if _dbg_leds  is not None else None
    debug_led_set     = frozenset(_dbg_leds)  if _dbg_leds  is not None else None
    debug_blob_anchor = int(_dbg_blobs[0])    if _dbg_blobs is not None else None
    debug_blob_set    = frozenset(_dbg_blobs) if _dbg_blobs is not None else None

    bijection_counts: Dict[frozenset, int] = {} if is_deep() else None

    for tier_idx, tier_spec in enumerate(depth_tiers):
        if strong_found:
            break

        led_max, blob_max = tier_spec[0], tier_spec[1]
        nbr_type = tier_spec[2] if len(tier_spec) > 2 else 'standard'

        if nbr_type == 'edge':
            cur_triple_idx   = led_triple_idx_edge
            cur_triple_depth = led_triple_depth_edge
            cur_triple_gates = led_triple_gates_edge
            cur_prev_blob    = prev_blob_max_per_triple_edge
        else:
            cur_triple_idx   = led_triple_idx
            cur_triple_depth = led_triple_depth
            cur_triple_gates = led_triple_gates
            cur_prev_blob    = prev_blob_max_per_triple

        eligible_mask = cur_triple_depth <= led_max
        eligible_triple_idx = np.where(eligible_mask)[0]

        # Keep only triples that have new blob pairs to explore at this tier.
        has_new_blob_pairs = cur_prev_blob[eligible_triple_idx] < blob_max
        active_triple_idx  = eligible_triple_idx[has_new_blob_pairs]
        tier_lq_total[tier_idx] = len(active_triple_idx)

        if len(active_triple_idx) == 0:
            continue

        active_triple_idx = active_triple_idx[rng.permutation(len(active_triple_idx))]

        for triple_i in active_triple_idx:
            led_ids            = cur_triple_idx[triple_i]    # [anchor, l1, l2]
            p3p_world_pts      = positions[led_ids]           # (3, 3) world points for P3P
            gate_led           = cur_triple_gates[triple_i]
            gate_led_world_pts = positions[gate_led].astype(np.float32) if len(gate_led) > 0 else np.zeros((0, 3), dtype=np.float32)

            # Start blob-pair enumeration from where the previous tier left off for
            # this triple; avoids re-evaluating combinations already covered earlier.
            min_blob_i2 = int(cur_prev_blob[triple_i])
            did_p3p = False

            for b_anchor in range(n_blobs):
                if strong_found:
                    break
                blob_neighbors   = blob_nbr[b_anchor]
                n_blob_neighbors = min(len(blob_neighbors), blob_max)
                if n_blob_neighbors < 2:
                    continue

                for i1, i2 in combinations(range(n_blob_neighbors), 2):
                    if strong_found:
                        break
                    if i2 < min_blob_i2:
                        continue
                    b1 = int(blob_neighbors[i1])
                    b2 = int(blob_neighbors[i2])

                    gate_blob_idx    = [int(blob_neighbors[j]) for j in range(n_blob_neighbors) if j != i1 and j != i2]
                    # If gate_blob_idx is empty (n_blob_neighbors == 2), _gate_any_point
                    # returns False — a 4th blob neighbour is required for gate validation.
                    gate_blob_img_pts = blobs_undist[gate_blob_idx] if gate_blob_idx else np.zeros((0, 2), dtype=np.float32)

                    for b1_ord, b2_ord in ((b1, b2), (b2, b1)):
                        if strong_found:
                            break

                        # ── Debug trigger ─────────────────────────────────────
                        # Anchors are matched positionally (first element of each
                        # debug triple); l1/l2 and b1/b2 are matched as sets so
                        # their internal ordering doesn't matter.  This gives at
                        # most 2 prints per frame: one per b1↔b2 swap.
                        dbg = is_verbose_all() or (
                            debug_active and
                            (debug_led_set  is None or (
                                int(led_ids[0]) == debug_led_anchor and
                                frozenset(led_ids) == debug_led_set)) and
                            (debug_blob_set is None or (
                                b_anchor == debug_blob_anchor and
                                frozenset([b_anchor, b1_ord, b2_ord]) == debug_blob_set))
                        )
                        if dbg:
                            logger.debug(
                                f"Target triple reached — "
                                f"LEDs {list(led_ids)}  blobs [{b_anchor},{b1_ord},{b2_ord}]  "
                                f"tier={tier_idx} ({_tier_label(tier_spec)})"
                            )

                        p3p_img_pts = blobs[[b_anchor, b1_ord, b2_ord]]

                        bij = frozenset(((int(led_ids[0]), b_anchor),
                                         (int(led_ids[1]), b1_ord),
                                         (int(led_ids[2]), b2_ord)))

                        if bijection_counts is not None:
                            bijection_counts[bij] = bijection_counts.get(bij, 0) + 1

                        # ── 1. P3P → up to 4 pose hypotheses ─────────────────
                        tier_p3p_calls[tier_idx] += 1
                        did_p3p = True
                        n_sols, rvecs, tvecs = cv2.solveP3P(
                            p3p_world_pts.reshape(3, 1, 3),
                            p3p_img_pts.reshape(3, 1, 2),
                            K, dc,
                            flags=cv2.SOLVEPNP_P3P,
                        )
                        if dbg:
                            logger.debug(f"  P3P returned {n_sols} solutions")
                        if not n_sols or rvecs is None:
                            continue

                        # Sort hypotheses by rotation distance to prior so the
                        # closest-to-prior solution is tried first — increases the
                        # chance of strong_found triggering early.
                        if R_prior is not None and n_sols > 1:
                            def _rot_score(rv):
                                R_i, _ = cv2.Rodrigues(rv.reshape(3, 1).astype(np.float32))
                                return float(np.trace(R_i @ R_prior.T))
                            order  = sorted(range(n_sols), key=lambda k: -_rot_score(rvecs[k]))
                            rvecs  = [rvecs[k] for k in order]
                            tvecs  = [tvecs[k] for k in order]

                        for sol_i, (rvec_h, tvec_h) in enumerate(zip(rvecs, tvecs)):
                            if strong_found:
                                break
                            rvec_h = rvec_h.reshape(3, 1).astype(np.float32)
                            tvec_h = tvec_h.reshape(3).astype(np.float32)

                            # ── 2. Depth range check (OpenHMD: 0.05 m – 15 m) ─
                            z_ok = _check_z_range(tvec_h)
                            if dbg:
                                logger.debug(f"  sol {sol_i}: z={tvec_h[2]:.3f} m  depth_ok={z_ok}")
                            if not z_ok:
                                continue

                            R_h, _ = cv2.Rodrigues(rvec_h)

                            # ── 3. Gate check (any gate LED near any gate blob) ─
                            gate_ok, gate_dist = _gate_any_point(R_h, tvec_h, gate_led_world_pts, gate_blob_img_pts, fx, fy, cx, cy, p4_thresh_sq)
                            if dbg:
                                logger.debug(f"  sol {sol_i}: gate_ok={gate_ok}, dist={gate_dist:.2f}px")
                            if not gate_ok:
                                continue

                            # ── 4. Full inlier count on all visible LEDs ───────
                            vis_mask_h = _visible_mask(
                                R_h, tvec_h, positions, normals,
                                geom,
                                cam_K=K, cam_dc=dc, cam_w=self.camera.width, cam_h=self.camera.height,
                                cam_rpmax=self.camera.rpmax,
                                facing_threshold_deg=facing_threshold_deg,
                            )
                            vis_ids = np.where(vis_mask_h)[0]
                            if dbg:
                                logger.debug(f"  sol {sol_i}: {len(vis_ids)} visible LEDs")
                            if len(vis_ids) < min_inliers:
                                continue

                            proj_all = _project_points(rvec_h, tvec_h, positions[vis_ids], K, dc)
                            cost     = cdist(blobs, proj_all)
                            hungarian_blob_rows, hungarian_led_cols = linear_sum_assignment(cost)

                            inlier_mask    = cost[hungarian_blob_rows, hungarian_led_cols] < hungarian_threshold_px
                            inlier_blobs   = hungarian_blob_rows[inlier_mask]
                            inlier_leds    = vis_ids[hungarian_led_cols[inlier_mask]]

                            if dbg:
                                outlier_mask     = cost[hungarian_blob_rows, hungarian_led_cols] >= hungarian_threshold_px
                                outlier_blob_rows = hungarian_blob_rows[outlier_mask]
                                outlier_led_cols  = vis_ids[hungarian_led_cols[outlier_mask]]
                                logger.debug(f"  sol {sol_i}: {len(inlier_blobs)} inliers after Hungarian "
                                             f"(need {min_inliers_eff})")
                            if len(inlier_blobs) < min_inliers_eff:
                                continue

                            # ── 5. RANSAC PnP refinement on inliers ───────────
                            ok_r, rvec_r, tvec_r, ransac_inliers = _ransac_pnp(
                                positions[inlier_leds], blobs[inlier_blobs], K, dc,
                                rvec_h, tvec_h.reshape(3, 1),
                                reprojection_px=reprojection_threshold,
                            )
                            if dbg:
                                logger.debug(f"  sol {sol_i}: RANSAC ok={ok_r}, "
                                             f"inliers={len(ransac_inliers) if ok_r else 0}")
                            if not ok_r:
                                continue

                            inlier_leds  = inlier_leds[ransac_inliers]
                            inlier_blobs = inlier_blobs[ransac_inliers]

                            if len(inlier_blobs) < min_inliers_eff:
                                continue

                            # ── 6. Visibility recheck with the refined pose ────
                            # Recompute on ALL LEDs so that:
                            #   (a) the denominator for coverage is accurate,
                            #   (b) inliers occluded under the refined pose are dropped.
                            R_r, _ = cv2.Rodrigues(rvec_r.reshape(3, 1).astype(np.float32))
                            tvec_r_flat = tvec_r.reshape(3)
                            vis_mask_r = _visible_mask(
                                R_r, tvec_r_flat, positions, normals,
                                geom,
                                cam_K=K, cam_dc=dc, cam_w=self.camera.width, cam_h=self.camera.height,
                                cam_rpmax=self.camera.rpmax,
                                facing_threshold_deg=facing_threshold_deg,
                            )
                            # Drop inliers that became occluded under the refined pose.
                            inlier_still_visible = vis_mask_r[inlier_leds]
                            inlier_leds  = inlier_leds[inlier_still_visible]
                            inlier_blobs = inlier_blobs[inlier_still_visible]
                            if dbg:
                                logger.debug(f"  sol {sol_i}: {len(inlier_blobs)} inliers after vis recheck")
                            if len(inlier_blobs) < min_inliers_eff:
                                continue

                            vis_ids_r = np.where(vis_mask_r)[0]

                            # ── 6.5. Post-RANSAC blob recovery ────────────────
                            # Blobs outside hungarian_threshold_px on the coarse
                            # P3P pose may land within reprojection_threshold
                            # under the refined pose.  One greedy nearest-
                            # neighbour pass recovers them (one cdist, no PnP).
                            matched_blob_set  = set(inlier_blobs.tolist())
                            matched_led_set   = set(inlier_leds.tolist())
                            unmatched_blobs   = np.array([b for b in range(n_blobs) if b not in matched_blob_set], dtype=np.int32)
                            unmatched_col_idx = np.array([j for j, lid in enumerate(vis_ids_r) if int(lid) not in matched_led_set], dtype=np.int32)

                            if len(unmatched_blobs) > 0 and len(unmatched_col_idx) > 0:
                                proj_vis_r = _project_points(rvec_r, tvec_r, positions[vis_ids_r], K, dc)
                                cost_r     = cdist(blobs, proj_vis_r)
                                sub_min    = cost_r[np.ix_(unmatched_blobs, unmatched_col_idx)].min(axis=0)
                                extra_blobs: List[int] = []
                                extra_leds:  List[int] = []
                                for order_j in np.argsort(sub_min):
                                    if sub_min[order_j] >= reprojection_threshold:
                                        break
                                    col    = int(unmatched_col_idx[order_j])
                                    led_id = int(vis_ids_r[col])
                                    for row_i in np.argsort(cost_r[unmatched_blobs, col]):
                                        b = int(unmatched_blobs[row_i])
                                        if b in matched_blob_set:
                                            continue
                                        if cost_r[b, col] >= reprojection_threshold:
                                            break
                                        matched_blob_set.add(b)
                                        extra_blobs.append(b)
                                        extra_leds.append(led_id)
                                        break
                                if extra_blobs:
                                    inlier_blobs = np.concatenate([inlier_blobs, np.array(extra_blobs, dtype=inlier_blobs.dtype)])
                                    inlier_leds  = np.concatenate([inlier_leds,  np.array(extra_leds,  dtype=inlier_leds.dtype)])
                                    if dbg:
                                        logger.debug(f"  sol {sol_i}: +{len(extra_blobs)} blob(s) recovered post-RANSAC")

                            # ── 7. Visibility coverage check ──────────────────
                            # Weight each visible LED by cos(θ) — the dot product
                            # of its normal with the view direction.  LEDs at
                            # grazing angles (θ → 90°) may be theoretically
                            # visible but are rarely detected reliably, so they
                            # contribute less to both numerator and denominator.
                            # This prevents those LEDs from unfairly failing the
                            # coverage gate when they simply aren't bright enough.
                            n_visible_leds = len(vis_ids_r)
                            n_inlier_blobs = len(inlier_blobs)

                            led_cam_pts    = (R_r @ positions[vis_ids_r].T).T + tvec_r_flat
                            led_cam_normals = (R_r @ normals[vis_ids_r].T).T
                            led_view_dirs  = -led_cam_pts / (np.linalg.norm(led_cam_pts, axis=1, keepdims=True) + 1e-9)
                            led_vis_weights = np.clip((led_cam_normals * led_view_dirs).sum(axis=1), 0.0, 1.0)

                            # inlier_leds ⊂ vis_ids_r is guaranteed by step 6; searchsorted
                            # maps each inlier LED index to its position in vis_ids_r so we
                            # can index led_vis_weights without a full boolean mask.
                            inlier_idx_in_vis    = np.searchsorted(vis_ids_r, inlier_leds)
                            weighted_visible_count = float(led_vis_weights.sum())
                            weighted_inlier_count  = float(led_vis_weights[inlier_idx_in_vis].sum())

                            if weighted_visible_count > 0 and weighted_inlier_count < min_vis_coverage * weighted_visible_count:
                                if dbg:
                                    logger.debug(
                                        f"  sol {sol_i}: weighted vis coverage "
                                        f"{weighted_inlier_count:.2f}/{weighted_visible_count:.2f}"
                                        f"={weighted_inlier_count/weighted_visible_count:.2f} < {min_vis_coverage:.2f}"
                                        f"  (raw {n_inlier_blobs}/{n_visible_leds})"
                                    )
                                continue

                            proj_r = _project_points(rvec_r, tvec_r, positions[inlier_leds], K, dc)
                            err    = float(np.mean(np.linalg.norm(proj_r - blobs[inlier_blobs], axis=1)))

                            orient_err = np.inf
                            if R_prior is not None:
                                cos_orient_angle = np.clip((np.trace(R_r @ R_prior.T) - 1.0) / 2.0, -1.0, 1.0)
                                orient_err = float(np.arccos(cos_orient_angle))

                            tvec_err = np.inf
                            if tvec_prior is not None:
                                tvec_err = float(np.linalg.norm(tvec_r.reshape(3) - tvec_prior))

                            error_per_inlier      = err  / max(n_inlier_blobs, 1)
                            best_error_per_inlier = best_error / max(best_inliers, 1)

                            # Prefer more inliers, break ties by absolute error, then
                            # orientation distance to prior, then translation distance.
                            is_better = (
                                (n_inlier_blobs > best_inliers and error_per_inlier < best_error_per_inlier) or
                                (n_inlier_blobs >= best_inliers + 2 and error_per_inlier < best_error_per_inlier * 1.1) or
                                (n_inlier_blobs == best_inliers and err < best_error) or
                                (n_inlier_blobs == best_inliers and
                                 abs(err - best_error) < 0.5 and
                                 orient_err < best_orient_err) or
                                (n_inlier_blobs == best_inliers and
                                 abs(err - best_error) < 0.5 and
                                 orient_err == best_orient_err and
                                 tvec_err < best_tvec_err)
                            )

                            if dbg:
                                logger.debug(f"  sol {sol_i}: err={err:.3f} px  inliers={n_inlier_blobs}  "
                                             f"is_better={is_better}")

                            if is_better:
                                best_solution = {
                                    "rvec":       rvec_r,
                                    "tvec":       tvec_r,
                                    "inliers":    n_inlier_blobs,
                                    "error":      err,
                                    "assignment": list(zip(inlier_blobs.tolist(), inlier_leds.tolist())),
                                    "method":     "p3p_systematic",
                                }
                                best_inliers    = n_inlier_blobs
                                best_error      = err
                                best_orient_err = orient_err
                                best_tvec_err   = tvec_err
                                solution_tier   = tier_idx

                                if log_best():
                                    logger.debug(
                                        f"  ★ new best — tier={tier_idx} "
                                        f"LEDs{list(led_ids)} blobs[{b_anchor},{b1_ord},{b2_ord}] "
                                        f"sol={sol_i}  inliers={n_inlier_blobs}  err={err:.3f}px  "
                                        f"vis={weighted_inlier_count:.1f}/{weighted_visible_count:.1f} (raw {n_inlier_blobs}/{n_visible_leds})"
                                    )

                                if (best_inliers >= strong_inliers_eff
                                        and best_error <= strong_match_error_px
                                        and (weighted_visible_count == 0 or weighted_inlier_count >= min_vis_coverage * weighted_visible_count)):
                                    strong_found = True

            cur_prev_blob[triple_i] = blob_max
            if did_p3p:
                tier_lq_tried[tier_idx] += 1
            if strong_found:
                break

    total_p3p_tried = sum(tier_p3p_calls)

    result_str = (
        f"found in tier_{solution_tier} ({_tier_label(depth_tiers[solution_tier])})  "
        f"({best_inliers} inliers, {best_error:.2f} px)"
        if best_solution is not None else "not found"
    )
    dup_line = ""
    if bijection_counts is not None:
        n_unique = len(bijection_counts)
        n_dup    = total_p3p_tried - n_unique
        max_dup  = max(bijection_counts.values(), default=0)
        dup_line = (
            f"\n  bijections — {n_unique} unique / {total_p3p_tried} calls  "
            f"({n_dup} duplicate calls, max {max_dup}× same bijection)"
        )
    tier_lines = "\n".join(
        f"  tier_{i} ({_tier_label(depth_tiers[i])}) — "
        f"{tier_p3p_calls[i]:>7} P3P calls  "
        f"({tier_lq_tried[i]}/{tier_lq_total[i]} LED triples reached inner loop)"
        for i in range(len(depth_tiers))
    )
    logger.debug(
        f"Brute-force: {result_str}\n"
        f"{tier_lines}\n"
        f"  total — {total_p3p_tried:>7} P3P calls"
        f"{dup_line}"
    )

    # if best_solution is not None:
    #     R_best, _ = cv2.Rodrigues(best_solution["rvec"])
    #     tvec_best = best_solution["tvec"].reshape(3)
    #     cam_dbg = -(R_best.T @ tvec_best)
    #     print(f"[vis check] cam_world = {cam_dbg.round(4)}")
    #     print("[frustum debug] inner LED occlusion check for best solution:")
    #     _visible_mask(R_best, tvec_best, positions, normals, geom, debug=True)

    return best_solution
