import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from itertools import combinations
from typing import Dict, List, Optional, Tuple
from time import time


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


def _build_led_neighbor_lists(positions: np.ndarray, normals: np.ndarray, k: int) -> List[np.ndarray]:
    """
    For each LED (anchor): among LEDs whose normal is within 90° of the anchor's normal
    (dot product >= 0), return up to k nearest by Euclidean distance.

    Normal filter first — only LEDs facing roughly the same direction are candidates,
    ensuring they can be simultaneously visible.  Spatial sort second — closest
    normal-compatible LEDs form the tightest, most discriminative hypotheses.
    """
    n = len(positions)
    k_act = min(k, n - 1)
    dists = cdist(positions, positions)   # (N, N)
    dots  = normals @ normals.T           # (N, N) pairwise normal cosines

    result = []
    for i in range(n):
        valid = dots[i] >= 0.0
        valid[i] = False
        candidates = np.where(valid)[0]
        if len(candidates) == 0:
            result.append(np.array([], dtype=int))
            continue
        order = np.argsort(dists[i, candidates])
        result.append(candidates[order[:k_act]])
    return result


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


def _triangle_angles(p: np.ndarray) -> Tuple[float, float]:
    """
    Angles (radians) at vertices 0 and 1 of triangle *p* (shape 3×2 or 3×3).
    Returns (0.0, 0.0) for degenerate triangles.
    """
    v01 = p[1] - p[0]
    v02 = p[2] - p[0]
    v12 = p[2] - p[1]
    n01 = float(np.linalg.norm(v01))
    n02 = float(np.linalg.norm(v02))
    n12 = float(np.linalg.norm(v12))
    if n01 < 1e-8 or n02 < 1e-8 or n12 < 1e-8:
        return 0.0, 0.0
    cos0 = np.clip(np.dot(v01, v02) / (n01 * n02), -1.0, 1.0)
    cos1 = np.clip(np.dot(-v01, v12) / (n01 * n12), -1.0, 1.0)
    return float(np.arccos(cos0)), float(np.arccos(cos1))


def _precompute_led_quads(
    positions: np.ndarray, led_nbr: List[np.ndarray], k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enumerate all LED quadruples and their triangle angles.  Called once and
    cached on the TrackingSystem across frames.

    Returns
    -------
    indices : (N, 4) int32
        Each row is [anchor, l1, l2, l3].  anchor/l1/l2 → P3P triple;
        l3 → 4th-point gate LED.
    angles  : (N, 2) float32
        Triangle angles (radians) at *anchor* and *l1* for the P3P triple.
        Used to filter blob-quad candidates before calling P3P.
    """
    idx_rows: List[Tuple] = []
    ang_rows: List[Tuple] = []
    n = len(positions)
    for anchor in range(n):
        nbrs = led_nbr[anchor][:k]
        if len(nbrs) < 3:
            continue
        for l1, l2, l3 in combinations(nbrs, 3):
            for perm in (
                (anchor, int(l1), int(l2), int(l3)),
                (anchor, int(l2), int(l1), int(l3)),
            ):
                ang = _triangle_angles(positions[list(perm[:3])])
                if ang == (0.0, 0.0):
                    continue
                idx_rows.append(perm)
                ang_rows.append(ang)
    if not idx_rows:
        return np.zeros((0, 4), dtype=np.int32), np.zeros((0, 2), dtype=np.float32)
    return (
        np.array(idx_rows, dtype=np.int32),
        np.array(ang_rows, dtype=np.float32),
    )


def _build_blob_quads(
    blobs: np.ndarray, blob_nbr: List[np.ndarray], k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enumerate all blob quadruples for the current frame.

    Returns
    -------
    indices : (M, 4) int32   — [anchor, b1, b2, b3]
    angles  : (M, 2) float32 — triangle angles (radians) at anchor and b1
    """
    idx_rows: List[Tuple] = []
    ang_rows: List[Tuple] = []
    n = len(blobs)
    for anchor in range(n):
        nbrs = blob_nbr[anchor][:k]
        if len(nbrs) < 3:
            continue
        for b1, b2, b3 in combinations(nbrs, 3):
            for perm in (
                (anchor, int(b1), int(b2), int(b3)),
                (anchor, int(b2), int(b1), int(b3)),
            ):
                ang = _triangle_angles(blobs[list(perm[:3])])
                if ang == (0.0, 0.0):
                    continue
                idx_rows.append(perm)
                ang_rows.append(ang)
    if not idx_rows:
        return np.zeros((0, 4), dtype=np.int32), np.zeros((0, 2), dtype=np.float32)
    return (
        np.array(idx_rows, dtype=np.int32),
        np.array(ang_rows, dtype=np.float32),
    )


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
    led_neighbor_depth: int = 6,        # k nearest LED neighbours to enumerate
    blob_neighbor_depth: int = 5,       # k nearest blob neighbours to enumerate
    p4_threshold_px: float = 8.0,       # 4th-point pre-filter gate (pixels)
    reprojection_threshold: float = 5.0,
    min_inliers: int = 4,
    strong_match_inliers: int = 7,      # stop immediately when this many inliers …
    strong_match_error_px: float = 1.5, # … and mean reprojection error is below this
    angle_tolerance_deg: float = 20.0,  # triangle-angle pre-filter tolerance (degrees)
) -> Optional[Dict]:
    """
    Systematic correspondence search, ported from OpenHMD's approach.

    Speed strategy
    --------------
    All LED quadruples are precomputed once and cached on *self*.  All blob
    quadruples are built once per call.  For each LED quad the matching blob
    quads are selected via a vectorised triangle-angle comparison
    (±angle_tolerance_deg per vertex).  Under perspective, triangle angles are
    nearly preserved for co-planar nearby LEDs (depth variation <10% → <8°
    angle error), so the filter rejects ~98% of blob quads before P3P,
    reducing solver calls ~30–50× compared to exhaustive enumeration.

    Structure
    ---------
    Outer loop : every cached LED quad [anchor, l1, l2, l3]
                   → numpy angle filter selects compatible blob quads

    Inner loop : compatible blob quads [a, b1, b2, b3]
                   → P3P on (anchor→a, l1→b1, l2→b2)
                   → 4th-point gate (l3 → b3)
                   → full inlier count + PnP refinement
    """
    blobs   = np.asarray(blobs, dtype=np.float32)
    n_blobs = len(blobs)
    if n_blobs < 4:
        return None

    positions = self.model.positions.astype(np.float32)
    normals   = self.model.normals.astype(np.float32)
    n_leds    = len(positions)
    K         = self.camera.camera_matrix
    dc        = self.camera.dist_coeffs

    # ── Build / reuse geometry and LED quad cache ─────────────────────────────
    if not hasattr(self, "_led_nbr") or len(self._led_nbr) != n_leds:
        (self._ring_axis, self._is_inner, self._radial_out,
         self._h_corpus, self._h_ax, self._ring_center_ax,
         self._R_ring, self._angular_threshold) = _compute_torus_geometry(positions, normals)
        self._led_nbr = _build_led_neighbor_lists(positions, normals, k=led_neighbor_depth)
        self._led_quad_idx, self._led_quad_ang = _precompute_led_quads(
            positions, self._led_nbr, led_neighbor_depth
        )

    led_quad_idx = self._led_quad_idx   # (N_LQ, 4) int32
    led_quad_ang = self._led_quad_ang   # (N_LQ, 2) float32

    is_inner          = self._is_inner
    radial_out        = self._radial_out
    ring_axis         = self._ring_axis
    h_corpus          = self._h_corpus
    h_ax              = self._h_ax
    ring_center_ax    = self._ring_center_ax
    R_ring            = self._R_ring
    angular_threshold = self._angular_threshold

    # ── Per-frame blob quad precomputation ────────────────────────────────────
    blob_nbr = _build_blob_neighbor_lists(blobs, k=blob_neighbor_depth)
    blob_quad_idx, blob_quad_ang = _build_blob_quads(blobs, blob_nbr, blob_neighbor_depth)
    if len(blob_quad_idx) == 0:
        return None

    ANGLE_TOL      = np.deg2rad(angle_tolerance_deg)
    p4_thresh_sq   = p4_threshold_px ** 2
    fx, fy         = float(K[0, 0]), float(K[1, 1])
    cx, cy         = float(K[0, 2]), float(K[1, 2])

    best_solution = None
    best_inliers  = 0
    best_error    = np.inf
    strong_found  = False
    counter       = 0

    # ── Main search ───────────────────────────────────────────────────────────
    for lq_i in range(len(led_quad_idx)):
        if strong_found:
            break

        l_ids = led_quad_idx[lq_i]   # [anchor, l1, l2, l3]

        # Vectorised angle filter: keep only blob quads whose triangle angles
        # are within ANGLE_TOL of this LED quad's triangle angles.
        ang_diff   = np.abs(blob_quad_ang - led_quad_ang[lq_i])   # (M, 2)
        match_idxs = np.where(np.all(ang_diff <= ANGLE_TOL, axis=1))[0]
        if len(match_idxs) == 0:
            continue

        obj3 = positions[l_ids[:3]]   # (3,3)  P3P world points
        obj4 = positions[l_ids[3]]    # (3,)   4th-point gate

        for mi in match_idxs:
            if strong_found:
                break

            b_ids = blob_quad_idx[mi]   # [anchor, b1, b2, b3]
            img3  = blobs[b_ids[:3]]     # (3,2)
            img4  = blobs[b_ids[3]]      # (2,)

            # ── 1. P3P → up to 4 pose hypotheses ─────────────────────────────
            counter += 1
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

                # Z-range sanity check (OpenHMD: 0.05 m – 15 m)
                if tvec_h[2] < 0.05 or tvec_h[2] > 15.0:
                    continue

                R_h, _ = cv2.Rodrigues(rvec_h)

                # ── 2. Pre-filter A: all P3P LEDs in front of camera ─────────
                # _build_led_neighbor_lists already constrains normals to ±90°,
                # so all 3 LEDs face the same direction — if one is behind the
                # camera the hypothesis is wrong.
                led_cam3 = (R_h @ obj3.T).T + tvec_h   # (3,3)
                if np.any(led_cam3[:, 2] <= 0):
                    continue

                # ── 3. Pre-filter B: 4th-point pixel gate ────────────────────
                # Inline perspective divide (no distortion) — fast single-point
                # check; full projection only needed when this passes.
                p4_cam = R_h @ obj4 + tvec_h
                if p4_cam[2] <= 0:
                    continue
                iz = 1.0 / p4_cam[2]
                dx = fx * p4_cam[0] * iz + cx - img4[0]
                dy = fy * p4_cam[1] * iz + cy - img4[1]
                if dx * dx + dy * dy > p4_thresh_sq:
                    continue

                # ── 4. Full inlier count on all visible LEDs ──────────────────
                vis_mask = _visible_mask(
                    R_h, tvec_h, positions, normals,
                    is_inner, radial_out, ring_axis,
                    h_corpus, h_ax, ring_center_ax, R_ring, angular_threshold,
                )
                vis_ids = np.where(vis_mask)[0]
                if len(vis_ids) < min_inliers:
                    continue

                proj_all = _project_points(rvec_h, tvec_h, positions[vis_ids], K, dc)
                cost     = cdist(blobs, proj_all)
                ri, ci   = linear_sum_assignment(cost)

                inlier_mask = cost[ri, ci] < reprojection_threshold
                ib = ri[inlier_mask]
                il = vis_ids[ci[inlier_mask]]

                if len(ib) < min_inliers:
                    continue

                # ── 5. Refinement PnP on inliers ──────────────────────────────
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

                if len(ib) > best_inliers or (len(ib) == best_inliers and err < best_error):
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

                    if best_inliers >= strong_match_inliers and best_error <= strong_match_error_px:
                        strong_found = True

    print(f"Brute-force matching evaluated {counter} P3P hypotheses")
    return best_solution