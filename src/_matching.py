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
                  positions: np.ndarray, normals: np.ndarray,
                  is_inner: Optional[np.ndarray] = None,
                  radial_out: Optional[np.ndarray] = None,
                  ring_axis: Optional[np.ndarray] = None,
                  h_corpus: Optional[np.ndarray] = None,
                  h_ax: float = 0.0,
                  ring_center_ax: float = 0.0,
                  R_ring: float = 0.0,
                  angular_threshold: float = -1.0) -> np.ndarray:
    """
    Boolean mask of LEDs that are (approximately) camera-facing and unoccluded.

    Standard check (all LEDs)
    ─────────────────────────
    dot(R @ normal, normalize(R @ pos + t)) < 0  →  LED faces camera  →  visible
    Threshold 0.0 = 90° half-angle emission cone (matches OpenHMD facing_dot > 0.0).

    Extra check for inner LEDs (two-part occlusion test)
    ────────────────────────────────────────────────────
    An inner LED passes the normal check but the outer torus wall may still
    block the line of sight.  Two separate mechanisms:

    1. Own-rim check (corpus height):
       Parametrize the ray from camera C to inner LED P as C + t*(P-C).
       The outer rim at the LED's angular position is Q = P + h_corpus * r.
       Q is on the ray at  t_Q = 1 + h_corpus / dot(P-C, r).
       The rim blocks the view if t_Q ∈ (0,1)  AND  the ray's axial coordinate
       at that crossing lies within the corpus axial extent (|z_tQ| ≤ h_ax).
       This correctly passes cameras above the ring (large axial component → z_tQ
       well outside h_ax) while blocking cameras in the ring plane.

    2. Far-rim check (cross-ring blocking):
       When the camera is nearly at the diametrically opposite angular position
       from the inner LED, the view ray crosses the far outer wall on the other
       side of the ring.  This happens only when
           dot(radial_out_LED, cam_dir_ring_plane) < angular_threshold
       where angular_threshold ≈ -sqrt(1-(r_tube/R_ring)²) ≈ -0.998 for a
       thin ring.  This nearly-never fires except for degenerate geometry.
    """
    led_cam     = (R @ positions.T).T + tvec              # (N, 3)
    z_ok        = led_cam[:, 2] > 0.01
    view_dir    = led_cam / (np.linalg.norm(led_cam, axis=1, keepdims=True) + 1e-8)
    normals_cam = (R @ normals.T).T
    dot         = (normals_cam * view_dir).sum(axis=1)
    mask        = z_ok & (dot < 0.0)                      # 90° half-angle emission cone

    if (is_inner is None or radial_out is None or ring_axis is None
            or h_corpus is None or not np.any(is_inner)):
        return mask

    cam_world = -(R.T @ tvec)                             # camera in world space

    # ── Part 2: far-rim (angular) check ─────────────────────────────────────
    cam_proj     = cam_world - (cam_world @ ring_axis) * ring_axis
    cam_proj_len = float(np.linalg.norm(cam_proj))
    if cam_proj_len > R_ring * 0.1:                       # camera not on ring axis
        cam_dir      = cam_proj / cam_proj_len
        far_ok       = (radial_out @ cam_dir) > angular_threshold  # (N,) bool
        mask         = mask & (~is_inner | far_ok)

    # ── Part 1: own-rim (corpus height) check ───────────────────────────────
    inner_active = np.where(mask & is_inner)[0]
    if len(inner_active) == 0:
        return mask

    P   = positions[inner_active]                         # (M, 3)
    r   = radial_out[inner_active]                        # (M, 3)
    h   = h_corpus[inner_active]                          # (M,)

    d        = P - cam_world                              # from camera toward LED (M, 3)
    d_rad    = (d * r).sum(axis=1)                        # radial component (M,)
    safe     = d_rad != 0
    d_rad_s  = np.where(safe, d_rad, 1.0)
    t_Q      = np.where(safe, 1.0 + h / d_rad_s, 2.0)   # 2.0 = outside (0,1) by default

    between  = (t_Q > 0.0) & (t_Q < 1.0)
    if not np.any(between):
        return mask

    cam_ax_val  = float(cam_world @ ring_axis)
    P_ax        = (P * ring_axis).sum(axis=1)             # (M,)
    z_tQ        = cam_ax_val + t_Q * (P_ax - cam_ax_val)  # axial pos at crossing (M,)
    blocked     = between & (np.abs(z_tQ - ring_center_ax) <= h_ax)
    mask[inner_active[blocked]] = False

    return mask


def _compute_torus_geometry(positions: np.ndarray, normals: np.ndarray):
    """
    Fit a ring plane to the LED positions and derive all geometry needed for
    inner-LED visibility testing.

    Returns
    -------
    ring_axis         : (3,) unit vector  – normal to ring plane (arbitrary sign)
    is_inner          : (N,) bool         – True for inner-surface LEDs
    radial_out        : (N, 3)            – unit outward-radial direction per LED
    h_corpus          : (N,) float        – for each inner LED: how far the outer
                                            rim extends above it along radial_out
                                            (0 for outer LEDs)
    h_ax              : float             – half-height of the full corpus along
                                            ring_axis (used in blocking test)
    ring_center_ax    : float             – ring centroid projected onto ring_axis
    R_ring            : float             – mean LED radial distance from ring axis
    angular_threshold : float             – dot(radial_out, cam_dir) threshold below
                                            which a cross-ring far-wall block occurs;
                                            derived from tube radius / ring radius
    """
    centroid  = positions.mean(axis=0)
    _, _, Vt  = np.linalg.svd(positions - centroid)
    ring_axis = Vt[-1]                                      # smallest singular value

    rel        = positions - centroid
    rel_proj   = rel - np.outer(rel @ ring_axis, ring_axis) # project to ring plane
    radial_out = rel_proj / (np.linalg.norm(rel_proj, axis=1, keepdims=True) + 1e-8)

    # Inner LED: normal points toward ring centre (opposite to radial_out)
    is_inner = (normals * radial_out).sum(axis=1) < 0

    # --- Corpus height per inner LED (how far outer rim is above it along r) ---
    n = len(positions)
    h_corpus = np.zeros(n, dtype=np.float64)
    for i in np.where(is_inner)[0]:
        r      = radial_out[i]
        projs  = positions @ r
        h_corpus[i] = float(projs.max() - projs[i])

    # --- Axial half-height of full corpus ---
    axial_projs    = positions @ ring_axis
    ring_center_ax = float(axial_projs.mean())
    h_ax           = float((axial_projs.max() - axial_projs.min()) / 2)

    # --- Ring radius and tube-radius estimate ---
    R_ring = float(np.linalg.norm(rel_proj, axis=1).mean())
    inner_idx = np.where(is_inner)[0]
    r_tube = float(h_corpus[inner_idx].mean() / 2) if len(inner_idx) else 0.005

    # --- Angular threshold for cross-ring blocking (camera nearly opposite inner LED)
    # A view ray crosses the far outer wall only when camera is within asin(r_tube/R_ring)
    # of the diametrically opposite angular position.
    angular_threshold = -float(np.sqrt(max(0.0, 1.0 - (r_tube / R_ring) ** 2))) if R_ring > 0 else -1.0

    return ring_axis, is_inner, radial_out, h_corpus, h_ax, ring_center_ax, R_ring, angular_threshold


def _build_led_neighbor_lists(positions: np.ndarray, k: int) -> List[np.ndarray]:
    """
    For each LED: k nearest spatial neighbours (inner OR outer, no type restriction).

    Why pure spatial, no type restriction?
    ──────────────────────────────────────
    The 140 "cross-type" (inner+outer) pairs produced by spatial k-NN are ALL
    same-face pairs: inner and outer LEDs at the same angular position are only
    ~tube-thickness apart (~6 mm).  These LEDs CAN be simultaneously visible when
    the camera is at a non-axial angle to the ring, and they are good P3P candidates.

    Contrast:
    • Normal-based (previous approach): 112 cross-type pairs that include cross-ring
      pairs — inner LED at one angular position + outer LED on the opposite side with
      coincidentally the same normal direction.  Those pairs are NEVER simultaneously
      visible and pollute the P3P search.
    • Same-type spatial: 0 cross-type pairs, but misses valid same-face cross-type
      groups when both inner and outer LEDs are visible (non-axial camera angle).
    • Pure spatial (this): 140 cross-type pairs, ALL same-face and valid.
      Identical to the OpenHMD approach.

    Any bad hypotheses from outer-wall-occluded inner LEDs are caught by the
    corpus-height occlusion check in _visible_mask.
    """
    k_act = min(k, len(positions) - 1)
    _, idx = KDTree(positions).query(positions, k=k_act + 1)  # +1 includes self
    return [row[1:] for row in idx]


def _build_normal_based_neighbors(positions: np.ndarray, normals: np.ndarray, k: int) -> List[np.ndarray]:
    """
    For each LED: k nearest LEDs by normal-direction similarity (highest dot product).

    LEDs with similar normals face the same direction and can be simultaneously
    visible from a camera in that direction.  For a cone this naturally creates
    valid cross-type (inner+outer) pairs for adjacent LEDs that share a facing
    direction, while avoiding cross-ring pairs (which have nearly opposite normals).

    Use this when both inner AND outer LEDs are expected to be visible (side-view).
    """
    n = len(normals)
    k_act = min(k, n - 1)
    sim = normals @ normals.T          # (N, N) pairwise cosine similarity
    np.fill_diagonal(sim, -2.0)        # exclude self
    result = []
    for i in range(n):
        idx = np.argsort(-sim[i])[:k_act]
        result.append(idx)
    return result


def _build_separate_group_neighbors(positions: np.ndarray, is_inner: np.ndarray, k: int) -> List[np.ndarray]:
    """
    For each LED: k nearest spatial neighbours within the SAME type (inner-only, outer-only).

    Produces zero cross-type pairs.  Correct and efficient when the camera only
    sees one LED type (e.g. looking straight at the cone tip → outer LEDs only,
    or from below → inner LEDs only).  Misses valid same-face cross-type groups
    at side-view angles — use _build_normal_based_neighbors for that scenario.
    """
    n = len(positions)
    k_act = min(k, n - 1)
    result: List[np.ndarray] = [np.array([], dtype=int)] * n

    for group_mask in (is_inner, ~is_inner):
        group_idx = np.where(group_mask)[0]
        if len(group_idx) < 2:
            continue
        k_g = min(k_act, len(group_idx) - 1)
        _, local_nbrs = KDTree(positions[group_idx]).query(positions[group_idx], k=k_g + 1)
        for local_i, global_i in enumerate(group_idx):
            result[global_i] = group_idx[local_nbrs[local_i, 1:]]   # skip self (index 0)

    return result


def _select_neighbor_strategy(
    positions: np.ndarray,
    normals: np.ndarray,
    is_inner: np.ndarray,
    R: np.ndarray,
    tvec: np.ndarray,
    nbr_normal: List[np.ndarray],
    nbr_separate: List[np.ndarray],
    radial_out: Optional[np.ndarray] = None,
    ring_axis: Optional[np.ndarray] = None,
    h_corpus: Optional[np.ndarray] = None,
    h_ax: float = 0.0,
    ring_center_ax: float = 0.0,
    R_ring: float = 0.0,
    angular_threshold: float = -1.0,
    min_visible_for_mixed: int = 2,
) -> List[np.ndarray]:
    """
    Pick the better neighbor strategy based on which LED types are visible from pose (R, tvec).

    Decision rule
    ─────────────
    If ≥ min_visible_for_mixed inner LEDs AND ≥ min_visible_for_mixed outer LEDs are
    visible → return normal-based neighbors (Option A).  Both types are simultaneously
    present in the image; cross-type pairs are valid and needed to form good hypotheses.

    Otherwise → return separate-group neighbors (Option B).  Only one type dominates;
    restricting neighbors to same-type reduces invalid hypotheses and speeds search.

    Fallback
    ────────
    If neither type reaches the threshold (≤ 1 of each), returns normal-based lists as
    the more permissive option so at least some hypotheses can be formed.
    """
    vis = _visible_mask(
        R, tvec, positions, normals,
        is_inner, radial_out, ring_axis,
        h_corpus, h_ax, ring_center_ax, R_ring, angular_threshold,
    )
    n_vis_inner = int(np.sum(vis & is_inner))
    n_vis_outer = int(np.sum(vis & ~is_inner))

    if n_vis_inner >= min_visible_for_mixed and n_vis_outer >= min_visible_for_mixed:
        return nbr_normal    # mixed visibility → allow cross-type pairs
    return nbr_separate      # one type dominant → keep groups separate


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
    is_inner_pm   = getattr(self, "_is_inner",   None)
    radial_out_pm = getattr(self, "_radial_out", None)
    ring_axis_pm  = getattr(self, "_ring_axis",  None)
    vis_mask  = _visible_mask(R_pred, tvec_pred,
                              self.model.positions, self.model.normals,
                              is_inner_pm, radial_out_pm, ring_axis_pm)
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
    strong_match_inliers: int = 7,      # stop immediately when this many inliers …
    strong_match_error_px: float = 1.5, # … and mean reprojection error is below this
    predicted_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # (rvec, tvec) hint
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

    # Build / reuse geometry and all three neighbour list variants (cheap after first call).
    if not hasattr(self, "_led_nbr_spatial") or len(self._led_nbr_spatial) != n_leds:
        (self._ring_axis, self._is_inner, self._radial_out,
         self._h_corpus, self._h_ax, self._ring_center_ax,
         self._R_ring, self._angular_threshold) = _compute_torus_geometry(positions, normals)
        self._led_nbr_spatial  = _build_led_neighbor_lists(positions, k=led_neighbor_depth)
        self._led_nbr_normal   = _build_normal_based_neighbors(positions, normals, k=led_neighbor_depth)
        self._led_nbr_separate = _build_separate_group_neighbors(positions, self._is_inner, k=led_neighbor_depth)

    # Adaptive strategy: if a predicted pose is available, pick the neighbor list that
    # best matches expected LED visibility; otherwise fall back to pure spatial.
    if predicted_pose is not None:
        _R_pred, _ = cv2.Rodrigues(np.asarray(predicted_pose[0], dtype=np.float32).reshape(3, 1))
        _tvec_pred = np.asarray(predicted_pose[1], dtype=np.float32).reshape(3)
        led_nbr = _select_neighbor_strategy(
            positions, normals, self._is_inner, _R_pred, _tvec_pred,
            self._led_nbr_normal, self._led_nbr_separate,
            self._radial_out, self._ring_axis, self._h_corpus,
            self._h_ax, self._ring_center_ax, self._R_ring, self._angular_threshold,
        )
    else:
        led_nbr = self._led_nbr_spatial

    is_inner          = self._is_inner
    radial_out        = self._radial_out
    ring_axis         = self._ring_axis
    h_corpus          = self._h_corpus
    h_ax              = self._h_ax
    ring_center_ax    = self._ring_center_ax
    R_ring            = self._R_ring
    angular_threshold = self._angular_threshold
    blob_nbr          = _build_blob_neighbor_lists(blobs, k=blob_neighbor_depth)

    best_solution = None
    best_inliers  = 0
    best_error    = np.inf
    strong_found  = False

    # -----------------------------------------------------------------------
    # Outer loop: every LED as anchor → C(k,3) from its k nearest neighbours
    # -----------------------------------------------------------------------
    for led_anchor in range(n_leds):
        if strong_found:
            break
        nbrs_l = led_nbr[led_anchor][:led_neighbor_depth]
        if len(nbrs_l) < 3:
            continue

        for l1, l2, l3 in combinations(nbrs_l, 3):
            if strong_found:
                break
            # OpenHMD tries two orderings of the middle LED neighbours
            for led_quad in (
                (led_anchor, int(l1), int(l2), int(l3)),
                (led_anchor, int(l2), int(l1), int(l3)),
            ):
                quad_ids  = list(led_quad[:3])
                obj3      = positions[quad_ids]            # (3,3) — P3P points
                obj4      = positions[led_quad[3]]         # (3,)  — 4th-point gate
                obj3_nrm  = normals[quad_ids]             # (3,3) — normals for pre-filter A
                obj3_rad  = radial_out[quad_ids]          # (3,3) — radial-out for inner check
                quad_inner = is_inner[quad_ids]           # (3,)  — True if inner LED

                # -----------------------------------------------------------
                # Inner loop: every blob as anchor → C(k,3) from neighbours
                # -----------------------------------------------------------
                for blob_anchor in range(n_blobs):
                    if strong_found:
                        break
                    nbrs_b = blob_nbr[blob_anchor][:blob_neighbor_depth]
                    if len(nbrs_b) < 3:
                        continue

                    for b1, b2, b3 in combinations(nbrs_b, 3):
                        if strong_found:
                            break
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
                                # Z-range sanity check (OpenHMD: 0.05 m – 15 m)
                                # -------------------------------------------
                                if tvec_h[2] < 0.05 or tvec_h[2] > 15.0:
                                    continue

                                # -------------------------------------------
                                # 2. Pre-filter A: anchor LEDs face the camera
                                #
                                # All 3 P3P LEDs share similar normals (built from
                                # normal-based neighbour lists), so they are either
                                # all visible together or all not.  A correct pose
                                # always passes this; a wrong pose fails ~98% of
                                # the time → cheap rejection of most bad hypotheses.
                                # Threshold 0.0 matches the 90° emission cone and
                                # OpenHMD's `facing_dot > 0.0` rejection.
                                # -------------------------------------------
                                led_cam3 = (R_h @ obj3.T).T + tvec_h  # (3,3)
                                if np.any(led_cam3[:, 2] <= 0):
                                    continue  # any anchor LED behind camera

                                view3    = led_cam3 / (np.linalg.norm(led_cam3, axis=1, keepdims=True) + 1e-8)
                                nrm3_cam = (R_h @ obj3_nrm.T).T
                                dot3     = (nrm3_cam * view3).sum(axis=1)
                                if np.any(dot3 > 0.0):
                                    continue  # at least one anchor LED outside 90° emission cone

                                # Inner-LED two-part occlusion check (mirrors _visible_mask)
                                if np.any(quad_inner):
                                    cam_world_h = -(R_h.T @ tvec_h)

                                    # Part 2: far-rim angular check
                                    cam_proj_h   = cam_world_h - (cam_world_h @ ring_axis) * ring_axis
                                    cam_proj_len = float(np.linalg.norm(cam_proj_h))
                                    if cam_proj_len > R_ring * 0.1:
                                        cam_dir_h = cam_proj_h / cam_proj_len
                                        far_fail  = quad_inner & ((obj3_rad @ cam_dir_h) <= angular_threshold)
                                        if np.any(far_fail):
                                            continue

                                    # Part 1: own-rim corpus-height check
                                    cam_ax_h      = float(cam_world_h @ ring_axis)
                                    _inner_blocked = False
                                    for qi in range(3):
                                        if not quad_inner[qi]:
                                            continue
                                        d_q   = obj3[qi] - cam_world_h
                                        d_rad = float(d_q @ obj3_rad[qi])
                                        if d_rad == 0.0:
                                            continue
                                        t_q = 1.0 + float(h_corpus[quad_ids[qi]]) / d_rad
                                        if not (0.0 < t_q < 1.0):
                                            continue
                                        z_q = cam_ax_h + t_q * (float(obj3[qi] @ ring_axis) - cam_ax_h)
                                        if abs(z_q - ring_center_ax) <= h_ax:
                                            _inner_blocked = True
                                            break
                                    if _inner_blocked:
                                        continue

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
                                vis_mask = _visible_mask(
                                    R_h, tvec_h, positions, normals,
                                    is_inner, radial_out, ring_axis,
                                    h_corpus, h_ax, ring_center_ax, R_ring, angular_threshold,
                                )
                                vis_ids  = np.where(vis_mask)[0]
                                if len(vis_ids) < min_inliers:
                                    continue

                                # Standard pinhole projection — must match how blobs were detected.
                                # Angular filter already applied by _visible_mask (dot < 0.0).
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

                                    # Strong match: stop immediately (OpenHMD POSE_MATCH_STRONG)
                                    if (best_inliers >= strong_match_inliers and
                                            best_error <= strong_match_error_px):
                                        strong_found = True

    return best_solution