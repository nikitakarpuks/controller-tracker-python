import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from itertools import combinations
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Internal helpers (module-level, not methods)
# ---------------------------------------------------------------------------

def _to_rvec(R_or_rvec: np.ndarray) -> np.ndarray:
    if R_or_rvec.shape == (3, 3):
        return cv2.Rodrigues(R_or_rvec)[0]
    return R_or_rvec.astype(np.float32).reshape(3, 1)


def _project_points(rvec, tvec, points: np.ndarray, K, dc) -> np.ndarray:
    """Project (N,3) world points → (N,2) image points."""
    pts, _ = cv2.projectPoints(
        points.astype(np.float32),
        _to_rvec(rvec),
        np.asarray(tvec, dtype=np.float32).reshape(3, 1),
        K, dc,
    )
    return pts.reshape(-1, 2)


def _visible_mask(R: np.ndarray, tvec: np.ndarray,
                  positions: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """
    Boolean mask of LEDs that are (approximately) camera-facing.

    Convention — matching OpenHMD:
        led_cam  = R @ pos + t
        view_dir = normalize(led_cam)    ← ray from camera origin toward LED
        normal_cam = R @ normal

    dot(normal_cam, view_dir) < 0  →  normal faces camera  →  LED visible
    dot(normal_cam, view_dir) > 0  →  normal faces away    →  LED culled

    Threshold calibrated from hardware data:
        32 LEDs, minimum normal spacing 22°.
        dot < -0.5  (60° half-angle) → mean 9.8 LEDs pass, matches 10 observed blobs.
        dot <  0.0  (90°)            → ~16 LEDs pass, adds 6 false candidates to Hungarian.
    Using -0.5 keeps the visible set tight and consistent with the emission cone.
    """
    led_cam     = (R @ positions.T).T + tvec              # (N, 3)
    z_ok        = led_cam[:, 2] > 0.01
    view_dir    = led_cam / (np.linalg.norm(led_cam, axis=1, keepdims=True) + 1e-8)
    normals_cam = (R @ normals.T).T
    dot         = (normals_cam * view_dir).sum(axis=1)
    return z_ok & (dot < -0.5)


def _build_led_normal_neighbor_lists(normals: np.ndarray, k: int) -> List[np.ndarray]:
    """
    For each LED: indices of k LEDs with the most similar normal directions,
    sorted by ascending angular distance, excluding the LED itself.

    Uses a KD-tree on unit normals. Because normals are unit vectors,
    Euclidean distance is a monotone function of angle:
        |n1 - n2|^2 = 2 - 2*dot(n1, n2)
    so nearest neighbours in L2 == smallest angular separation.

    Grouping by normal direction (rather than 3-D position) makes sense for a
    torus-shaped controller: LEDs on opposite sides of the ring are physically
    far apart but may face the same direction, and those are exactly the
    candidates that can all be visible together within a narrow emission cone.
    """
    unit = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    tree = KDTree(unit)
    _, idx = tree.query(unit, k=k + 1)   # +1 = includes self at idx[:,0]
    return [row[1:] for row in idx]


def _build_blob_neighbor_lists(blobs: np.ndarray, k: int) -> List[np.ndarray]:
    """
    For each blob: indices of up to k nearest blob neighbours, excluding self.
    """
    n = len(blobs)
    if n <= 1:
        return [np.array([], dtype=int) for _ in range(n)]
    tree  = KDTree(blobs)
    k_act = min(k, n - 1)
    _, idx = tree.query(blobs, k=k_act + 1)
    return [row[1:] for row in idx]


# ---------------------------------------------------------------------------
# proximity_match
# ---------------------------------------------------------------------------

def proximity_match(
    self,
    blobs: np.ndarray,
    predicted_pose: Tuple[np.ndarray, np.ndarray],
    max_distance_px: float = 30.0,
    prior_assignment: Optional[List] = None,
) -> Optional[Dict]:
    """
    Refine a predicted pose by matching projected LEDs to detected blobs.

    Two paths:

    Path 1 — assignment-locked (used when prior_assignment is provided):
        For each LED from the previous frame's assignment, project it with
        the predicted pose and snap to its nearest blob within max_distance_px.
        This is O(N_matches) instead of O(N_blobs × N_leds) and cannot
        accidentally swap LED–blob pairs — the main source of oscillation at 30fps.

    Path 2 — full Hungarian (fallback when locked path fails or no prior):
        Project all visible LEDs, run Hungarian assignment, solvePnP.
    """
    rvec_pred, tvec_pred = predicted_pose
    rvec_pred = np.asarray(rvec_pred, dtype=np.float32).reshape(3, 1)
    tvec_pred = np.asarray(tvec_pred, dtype=np.float32).reshape(3)

    K  = self.camera.camera_matrix
    dc = self.camera.dist_coeffs

    # ------------------------------------------------------------------
    # Path 1: assignment-locked nearest-neighbour
    # ------------------------------------------------------------------
    if prior_assignment is not None and len(prior_assignment) >= 3:
        prior_lids   = [lid for _, lid in prior_assignment]
        prior_obj    = self.model.positions[prior_lids].astype(np.float32)
        proj_prior   = _project_points(rvec_pred, tvec_pred, prior_obj, K, dc)

        # Nearest-neighbour snap: for each prior LED find which blob it moved to.
        # Store (blob_idx, led_idx) so the same list becomes prev_assignment next frame.
        # DO NOT run a full Hungarian expansion here — expansion with a loose threshold
        # introduces spurious LED-blob pairs that corrupt the prior for the next frame.
        locked_pairs, locked_obj, locked_img = [], [], []
        for i, lid in enumerate(prior_lids):
            dists = np.linalg.norm(blobs - proj_prior[i], axis=1)
            j     = int(np.argmin(dists))
            if dists[j] < max_distance_px:
                locked_pairs.append((j, lid))
                locked_obj.append(self.model.positions[lid])
                locked_img.append(blobs[j])

        if len(locked_pairs) >= 3:
            lo = np.array(locked_obj, dtype=np.float32)
            li = np.array(locked_img, dtype=np.float32)

            ok, rvec, tvec = cv2.solvePnP(
                lo, li, K, dc,
                rvec_pred, tvec_pred,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                proj  = _project_points(rvec, tvec, lo, K, dc)
                error = float(np.mean(np.linalg.norm(proj - li, axis=1)))
                if error < max_distance_px:
                    return {
                        "rvec":       rvec,
                        "tvec":       tvec,
                        "error":      error,
                        "assignment": locked_pairs,
                        "method":     "proximity_locked",
                    }

    # ------------------------------------------------------------------
    # Path 2: full Hungarian (no prior, or locked path failed)
    # ------------------------------------------------------------------
    R_pred, _ = cv2.Rodrigues(rvec_pred)
    vis_mask  = _visible_mask(R_pred, tvec_pred, self.model.positions, self.model.normals)
    vis_ids   = np.where(vis_mask)[0]
    if len(vis_ids) < 3:
        return None

    led_proj = _project_points(rvec_pred, tvec_pred,
                               self.model.positions[vis_ids], K, dc)

    cost    = cdist(blobs, led_proj)
    ri, ci  = linear_sum_assignment(cost)
    matches = [
        (int(b), int(vis_ids[l]))
        for b, l in zip(ri, ci)
        if cost[b, l] < max_distance_px
    ]
    if len(matches) < 3:
        return None

    obj_pts = self.model.positions[[lid for _, lid in matches]].astype(np.float32)
    img_pts = blobs[[bid for bid, _ in matches]].astype(np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, dc,
        rvec_pred, tvec_pred,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    proj  = _project_points(rvec, tvec, obj_pts, K, dc)
    error = float(np.mean(np.linalg.norm(proj - img_pts, axis=1)))
    if error > max_distance_px:
        return None

    return {
        "rvec":       rvec,
        "tvec":       tvec,
        "error":      error,
        "assignment": matches,
        "method":     "proximity",
    }


# ---------------------------------------------------------------------------
# brute_match
# ---------------------------------------------------------------------------

def brute_match(
    self,
    blobs: np.ndarray,
    led_neighbor_depth: int = 8,        # k nearest LED neighbours to enumerate
    blob_neighbor_depth: int = 6,       # k nearest blob neighbours to enumerate
    p4_threshold_px: float = 8.0,       # 4th-point pre-filter gate (pixels)
    reprojection_threshold: float = 5.0,
    min_inliers: int = 4,
) -> Optional[Dict]:
    """
    Systematic correspondence search, ported from OpenHMD's approach.

    Structure
    ---------
    Outer loop  : every LED as anchor
                  → C(k, 3) combos from its k nearest LED neighbours
                  → LED quadruple  [A, B, C, D]

    Inner loop  : every blob as anchor
                  → C(k, 3) combos from its k nearest blob neighbours
                  → blob quadruple [a, b, c, d]

    Two blob permutations per combo: [a,b,c,d] and [a,c,b,d]
    (OpenHMD's swap of positions 1 and 2 — doubles P3P coverage cheaply)

    Per hypothesis
    --------------
    1. P3P on (A→a, B→b, C→c)  →  up to 4 algebraic pose candidates
    2. Pre-filter A: anchor LEDs in front of camera and not back-facing
    3. Pre-filter B: 4th point D projects within p4_threshold_px of blob d
       (cheap pixel-distance gate — avoids expensive full inlier check)
    4. Full inlier count via Hungarian on all visible LEDs
    5. solvePnP refinement on inliers
    """
    blobs   = np.asarray(blobs, dtype=np.float32)
    n_blobs = len(blobs)
    if n_blobs < 4:
        return None

    positions = self.model.positions.astype(np.float32)   # (N_leds, 3)
    normals   = self.model.normals.astype(np.float32)     # (N_leds, 3)
    n_leds    = len(positions)
    K         = self.camera.camera_matrix
    dc        = self.camera.dist_coeffs

    # Build / reuse LED neighbour lists grouped by normal direction (cheap after first call).
    # Neighbours share similar emission directions, so they are all visible together
    # when the camera falls within their 5° cone — much better P3P candidates than
    # spatially-close LEDs which can face completely different directions on a torus.
    if not hasattr(self, "_led_nbr") or len(self._led_nbr) != n_leds:
        self._led_nbr = _build_led_normal_neighbor_lists(normals, k=led_neighbor_depth)
    led_nbr  = self._led_nbr
    blob_nbr = _build_blob_neighbor_lists(blobs, k=blob_neighbor_depth)

    best_solution = None
    best_inliers  = 0
    best_error    = np.inf

    # -----------------------------------------------------------------------
    # Outer loop: every LED as anchor → C(k,3) from its k nearest neighbours
    # -----------------------------------------------------------------------
    for led_anchor in range(n_leds):
        nbrs_l = led_nbr[led_anchor][:led_neighbor_depth]
        if len(nbrs_l) < 3:
            continue

        for l1, l2, l3 in combinations(nbrs_l, 3):
            # OpenHMD tries two orderings of the middle LED neighbours
            for led_quad in (
                (led_anchor, int(l1), int(l2), int(l3)),
                (led_anchor, int(l2), int(l1), int(l3)),
            ):
                obj3     = positions[list(led_quad[:3])]  # (3,3) — P3P points
                obj4     = positions[led_quad[3]]          # (3,)  — 4th-point gate
                obj3_nrm = normals[list(led_quad[:3])]    # (3,3) — normals for pre-filter A

                # -----------------------------------------------------------
                # Inner loop: every blob as anchor → C(k,3) from neighbours
                # -----------------------------------------------------------
                for blob_anchor in range(n_blobs):
                    nbrs_b = blob_nbr[blob_anchor][:blob_neighbor_depth]
                    if len(nbrs_b) < 3:
                        continue

                    for b1, b2, b3 in combinations(nbrs_b, 3):
                        # Two blob permutations (OpenHMD swap)
                        for blob_quad in (
                            (blob_anchor, int(b1), int(b2), int(b3)),
                            (blob_anchor, int(b2), int(b1), int(b3)),
                        ):
                            img3 = blobs[list(blob_quad[:3])]  # (3,2)
                            img4 = blobs[blob_quad[3]]          # (2,)

                            # -----------------------------------------------
                            # 1. P3P → up to 4 pose hypotheses
                            # -----------------------------------------------
                            n_sols, rvecs, tvecs = cv2.solveP3P(
                                obj3.reshape(3, 1, 3),
                                img3.reshape(3, 1, 2),
                                K, dc,
                                flags=cv2.SOLVEPNP_P3P,
                            )
                            if not n_sols or rvecs is None:
                                continue

                            for rvec_h, tvec_h in zip(rvecs, tvecs):
                                rvec_h = rvec_h.reshape(3, 1).astype(np.float32)
                                tvec_h = tvec_h.reshape(3).astype(np.float32)
                                R_h, _ = cv2.Rodrigues(rvec_h)

                                # -------------------------------------------
                                # 2. Pre-filter A: anchor LEDs face the camera
                                #
                                # All 3 P3P LEDs share similar normals (built from
                                # normal-based neighbour lists), so they are either
                                # all visible together or all not.  A correct pose
                                # always passes this; a wrong pose fails ~98% of
                                # the time → cheap rejection of most bad hypotheses.
                                # Threshold -0.5 (60°) matches _visible_mask.
                                # -------------------------------------------
                                led_cam3 = (R_h @ obj3.T).T + tvec_h  # (3,3)
                                if np.any(led_cam3[:, 2] <= 0):
                                    continue  # any anchor LED behind camera

                                view3    = led_cam3 / (np.linalg.norm(led_cam3, axis=1, keepdims=True) + 1e-8)
                                nrm3_cam = (R_h @ obj3_nrm.T).T
                                dot3     = (nrm3_cam * view3).sum(axis=1)
                                if np.any(dot3 > -0.5):
                                    continue  # at least one anchor LED outside 60° emission cone

                                # -------------------------------------------
                                # 3. Pre-filter B: 4th point pixel gate
                                # -------------------------------------------
                                p4_proj = _project_points(
                                    rvec_h, tvec_h, obj4.reshape(1, 3), K, dc
                                )
                                if not np.all(np.isfinite(p4_proj)):
                                    continue
                                if np.linalg.norm(p4_proj[0] - img4) > p4_threshold_px:
                                    continue

                                # -------------------------------------------
                                # 4. Full inlier count on all visible LEDs
                                # -------------------------------------------
                                vis_mask = _visible_mask(R_h, tvec_h, positions, normals)
                                vis_ids  = np.where(vis_mask)[0]
                                if len(vis_ids) < min_inliers:
                                    continue

                                # Standard pinhole projection — must match how blobs were detected.
                                # Angular filter already applied by _visible_mask (dot < -0.5).
                                proj_all = _project_points(
                                    rvec_h, tvec_h, positions[vis_ids], K, dc
                                )

                                cost   = cdist(blobs, proj_all)
                                ri, ci = linear_sum_assignment(cost)

                                inlier_b, inlier_l = [], []
                                for i in range(len(ri)):
                                    if cost[ri[i], ci[i]] < reprojection_threshold:
                                        inlier_b.append(int(ri[i]))
                                        inlier_l.append(int(vis_ids[ci[i]]))

                                if len(inlier_b) < min_inliers:
                                    continue

                                # -------------------------------------------
                                # 5. Refinement PnP on inliers
                                # -------------------------------------------
                                ib = np.array(inlier_b)
                                il = np.array(inlier_l)

                                ok_r, rvec_r, tvec_r = cv2.solvePnP(
                                    positions[il], blobs[ib], K, dc,
                                    rvec_h, tvec_h.reshape(3, 1),
                                    useExtrinsicGuess=True,
                                    flags=cv2.SOLVEPNP_ITERATIVE,
                                )
                                if not ok_r:
                                    continue

                                proj_r = _project_points(rvec_r, tvec_r, positions[il], K, dc)
                                err    = float(np.mean(np.linalg.norm(proj_r - blobs[ib], axis=1)))

                                if (len(ib) > best_inliers or
                                        (len(ib) == best_inliers and err < best_error)):
                                    best_solution = {
                                        "rvec":       rvec_r,
                                        "tvec":       tvec_r,
                                        "inliers":    len(ib),
                                        "error":      err,
                                        "assignment": list(zip(ib.tolist(), il.tolist())),
                                        "method":     "p3p_systematic",
                                    }
                                    best_inliers = len(ib)
                                    best_error   = err

    return best_solution