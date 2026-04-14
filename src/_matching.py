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

# Pre-allocated identity matrices reused by _ransac_pnp.
_K_IDENTITY = np.eye(3, dtype=np.float64)
_DC_ZERO    = np.zeros(4, dtype=np.float64)


def _ransac_pnp(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    K, dc,
    rvec_init=None,
    tvec_init=None,
    reprojection_px: float = 3.0,
    iterations: int = 100,
    confidence: float = 0.99,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    RANSAC PnP using undistorted normalised image coordinates.

    Ported from OpenHMD / Monado (ransac_pnp.cpp):
      • Blob points are undistorted with the real calibration before being
        handed to the solver, so the RANSAC loop never touches distortion
        maths — faster and more numerically stable.
      • An identity K + zero distortion are passed to the solver to match
        the undistorted point space.
      • The reprojection threshold is converted from pixels to normalised
        units as reprojection_px / fx.
      • SOLVEPNP_SQPNP is used as the minimal solver (non-iterative,
        closed-form — more robust than ITERATIVE LM inside RANSAC).

    Returns
    -------
    ok          : bool
    rvec        : (3, 1) float64 or None
    tvec        : (3, 1) float64 or None
    inlier_idx  : 1-D int array indexing obj_pts/img_pts, or None
    """
    if len(obj_pts) < 4:
        return False, None, None, None

    fx = float(K[0, 0])

    # Undistort image points → normalised camera coordinates (K removed, distortion removed).
    pts_norm = cv2.undistortPoints(
        img_pts.astype(np.float32).reshape(-1, 1, 2), K, dc,
    ).reshape(-1, 2)

    use_guess = rvec_init is not None and tvec_init is not None
    r0 = np.asarray(rvec_init, dtype=np.float64).reshape(3, 1) if use_guess else None
    t0 = np.asarray(tvec_init, dtype=np.float64).reshape(3, 1) if use_guess else None

    try:
        ret, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts.astype(np.float64),
            pts_norm.astype(np.float64),
            _K_IDENTITY, _DC_ZERO,
            r0, t0,
            useExtrinsicGuess=use_guess,
            iterationsCount=iterations,
            reprojectionError=reprojection_px / fx,
            confidence=confidence,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:
        return False, None, None, None

    if not ret or inliers is None or len(inliers) < 4:
        return False, None, None, None

    return True, rvec, tvec, inliers.flatten()


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
                  angular_threshold: float = -1.0,
                  ring_centroid: Optional[np.ndarray] = None,
                  R_frustum_center: Optional[float] = None,
                  frustum_slope: Optional[float] = None,
                  z_frustum_top: Optional[float] = None,
                  z_frustum_bot: Optional[float] = None) -> np.ndarray:
    """
    Boolean mask of LEDs that are (approximately) camera-facing and unoccluded.

    Standard check (all LEDs)
    ─────────────────────────
    dot(R @ normal, normalize(R @ pos + t)) < 0  →  LED faces camera  →  visible
    Threshold 0.0 = 90° half-angle emission cone (matches OpenHMD facing_dot > 0.0).

    Extra check for inner LEDs (frustum cone occlusion test)
    ─────────────────────────────────────────────────────────
    An inner LED passes the normal check but the outer frustum (truncated cone) wall
    may still block the line of sight.  We solve for the intersection of the
    camera→LED ray with the frustum's lateral conical surface and block the LED if
    an intersection falls strictly between the camera and LED within the frustum's
    axial extent.

    The frustum is parametrised as:
        R_outer(z_rel) = R_frustum_center + frustum_slope * z_rel
    where z_rel = (pos · ring_axis) − ring_center_ax, and ring_axis is oriented
    toward the larger-radius base (frustum_slope ≥ 0).

    The frustum axial extent [z_frustum_bot, z_frustum_top] is intentionally
    asymmetric around the LED centroid because the larger-radius base is typically
    much closer to the centroid than the smaller base.  Using these asymmetric bounds
    avoids incorrectly blocking inner LEDs seen from cameras above the large base.

    A secondary far-rim angular check suppresses inner LEDs that are diametrically
    opposite the camera in the ring plane — these would require the ray to pass
    through the far outer wall, handled cheaply by a dot-product threshold.
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

    # ── Far-rim angular check (cross-ring blocking) ──────────────────────────
    cam_proj     = cam_world - (cam_world @ ring_axis) * ring_axis
    cam_proj_len = float(np.linalg.norm(cam_proj))
    if cam_proj_len > R_ring * 0.1:                       # camera not on ring axis
        cam_dir  = cam_proj / cam_proj_len
        far_ok   = (radial_out @ cam_dir) > angular_threshold  # (N,) bool
        mask     = mask & (~is_inner | far_ok)

    inner_active = np.where(mask & is_inner)[0]
    if len(inner_active) == 0:
        return mask

    if (ring_centroid is not None
            and R_frustum_center is not None
            and frustum_slope is not None):
        # ── Frustum cone intersection ────────────────────────────────────────
        # Ray: Q(t) = cam_world + t*(P - cam_world),  t ∈ (0, 1)
        # Frustum lateral surface: |Q_rad(t)| = R_fc + slope * Q_ax_rel(t)
        # where Q_ax_rel = Q · ring_axis − ring_center_ax,
        #       Q_rad = Q − (Q · ring_axis)*ring_axis − ring_centroid_rad
        P = positions[inner_active]            # (M, 3)

        # Decompose camera and LEDs relative to ring centroid
        C_rel = cam_world - ring_centroid      # (3,)
        P_rel = P - ring_centroid              # (M, 3)

        C_ax  = float(C_rel @ ring_axis)       # scalar axial coord of camera
        P_ax  = (P_rel * ring_axis).sum(axis=1)  # (M,) axial coords of LEDs

        C_rad = C_rel - C_ax * ring_axis       # (3,)  radial vec of camera
        P_rad = P_rel - np.outer(P_ax, ring_axis)  # (M,3) radial vecs of LEDs

        d_ax  = P_ax - C_ax                   # (M,) axial ray component
        d_rad = P_rad - C_rad                 # (M,3) radial ray component

        # Outer radius varies linearly along the ray:
        #   R(t) = A0 + Ad * t
        A0 = float(R_frustum_center) + float(frustum_slope) * C_ax  # scalar
        Ad = float(frustum_slope) * d_ax                              # (M,)

        # |C_rad + t*d_rad|² = (A0 + Ad*t)²
        # (|d_rad|² − Ad²)·t² + 2·(C_rad·d_rad − A0·Ad)·t + (|C_rad|² − A0²) = 0
        a     = (d_rad ** 2).sum(axis=1) - Ad ** 2                    # (M,)
        b     = 2.0 * ((C_rad * d_rad).sum(axis=1) - A0 * Ad)        # (M,)
        c_val = float(np.dot(C_rad, C_rad)) - A0 ** 2                 # scalar

        disc    = b ** 2 - 4.0 * a * c_val                            # (M,)
        blocked = np.zeros(len(inner_active), dtype=bool)

        # Asymmetric axial bounds: frustum does NOT extend symmetrically around
        # the LED centroid.  Use [z_frustum_bot, z_frustum_top] when available,
        # otherwise fall back to the symmetric ±h_ax estimate.
        use_asym = (z_frustum_top is not None and z_frustum_bot is not None)

        def _in_axial(z: np.ndarray) -> np.ndarray:
            if use_asym:
                return (z >= z_frustum_bot) & (z <= z_frustum_top)
            return np.abs(z) <= h_ax

        # Quadratic roots
        has_roots = disc >= 0.0
        if np.any(has_roots):
            sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
            safe_2a   = np.where(np.abs(a) > 1e-12, 2.0 * a, 1.0)
            for sign in (-1.0, 1.0):
                t = np.where(np.abs(a) > 1e-12,
                             (-b + sign * sqrt_disc) / safe_2a,
                             2.0)             # 2.0 is outside (0,1) → not blocking
                in_range = has_roots & (t > 1e-4) & (t < 1.0 - 1e-4)
                if not np.any(in_range):
                    continue
                z_rel_t = C_ax + t * d_ax    # axial pos at intersection (M,)
                blocked |= in_range & _in_axial(z_rel_t)

        # Linear fallback: ray nearly tangent to cone surface (a ≈ 0)
        lin_case = np.abs(a) <= 1e-12
        if np.any(lin_case):
            t_lin = np.where(np.abs(b) > 1e-12, -c_val / b, 2.0)
            in_range_lin = lin_case & (t_lin > 1e-4) & (t_lin < 1.0 - 1e-4)
            if np.any(in_range_lin):
                blocked |= in_range_lin & _in_axial(C_ax + t_lin * d_ax)

    else:
        # ── Fallback: cylinder approximation (no frustum geometry available) ─
        P   = positions[inner_active]
        r   = radial_out[inner_active]
        h   = h_corpus[inner_active]

        d       = P - cam_world
        d_rad   = (d * r).sum(axis=1)
        safe    = d_rad != 0
        d_rad_s = np.where(safe, d_rad, 1.0)
        t_Q     = np.where(safe, 1.0 + h / d_rad_s, 2.0)

        between = (t_Q > 0.0) & (t_Q < 1.0)
        blocked = np.zeros(len(inner_active), dtype=bool)
        if np.any(between):
            cam_ax_val = float(cam_world @ ring_axis)
            P_ax       = (P * ring_axis).sum(axis=1)
            z_tQ       = cam_ax_val + t_Q * (P_ax - cam_ax_val)
            blocked    = between & (np.abs(z_tQ - ring_center_ax) <= h_ax)

    mask[inner_active[blocked]] = False
    return mask


def _compute_torus_geometry(positions: np.ndarray, normals: np.ndarray):
    """
    Fit a frustum (truncated cone) to the LED ring and derive all geometry
    needed for inner-LED visibility testing.

    Returns
    -------
    ring_axis         : (3,) unit vector  – normal to ring plane, oriented so that
                                            it points toward the larger-radius base
                                            (outer LED normals have negative dot with
                                            ring_axis because they tilt toward the
                                            narrow end)
    is_inner          : (N,) bool         – True for inner-surface LEDs
    radial_out        : (N, 3)            – unit outward-radial direction per LED
    h_corpus          : (N,) float        – for each inner LED: radial gap from the
                                            LED to the outer frustum wall at the
                                            same axial position (0 for outer LEDs)
    h_ax              : float             – frustum axial half-height; estimated from
                                            the outermost outer LED's radial position
                                            and the fitted cone slope
    ring_center_ax    : float             – ring centroid projected onto ring_axis
    R_ring            : float             – mean LED radial distance from ring axis
    angular_threshold : float             – dot(radial_out, cam_dir) threshold for
                                            cross-ring far-wall blocking
    ring_centroid     : (3,) float        – 3-D centroid of all LED positions
    R_frustum_center  : float             – outer frustum radius at ring_center_ax
    frustum_slope     : float             – dR_outer/dz_rel; positive because
                                            ring_axis points toward the larger base
    """
    centroid  = positions.mean(axis=0)
    _, _, Vt  = np.linalg.svd(positions - centroid)
    ring_axis = Vt[-1]                                      # smallest singular value

    rel        = positions - centroid
    rel_proj   = rel - np.outer(rel @ ring_axis, ring_axis) # project to ring plane
    radial_out = rel_proj / (np.linalg.norm(rel_proj, axis=1, keepdims=True) + 1e-8)

    # Inner LED: normal points toward ring centre (opposite to radial_out)
    is_inner = (normals * radial_out).sum(axis=1) < 0

    # --- Orient ring_axis toward the larger-radius base ---
    # The outer frustum surface normal tilts toward the narrow end.  When
    # ring_axis points toward the larger base, outer normals have a *negative*
    # axial component.  Flip if the mean is positive.
    outer_mask = ~is_inner
    if outer_mask.any():
        if float((normals[outer_mask] @ ring_axis).mean()) > 0:
            ring_axis = -ring_axis

    # --- Axial positions relative to centroid ---
    axial_projs    = positions @ ring_axis        # (N,)
    ring_center_ax = float(axial_projs.mean())
    z_rel          = axial_projs - ring_center_ax  # (N,) relative axial positions

    # --- Fit outer frustum: R_outer(z_rel) = R_fc + slope * z_rel ---
    # Larger radius is at +z_rel end (ring_axis points toward larger base → slope > 0).
    outer_idx    = np.where(outer_mask)[0]
    outer_radial = np.linalg.norm(rel_proj[outer_idx], axis=1)
    outer_z_rel  = z_rel[outer_idx]

    if len(outer_idx) >= 2:
        A = np.column_stack([np.ones(len(outer_idx)), outer_z_rel])
        coeffs, _, _, _ = np.linalg.lstsq(A, outer_radial, rcond=None)
        R_fc          = float(coeffs[0])
        frustum_slope = float(coeffs[1])
    else:
        R_fc          = float(np.linalg.norm(rel_proj, axis=1).mean())
        frustum_slope = 0.0

    # --- Asymmetric frustum axial bounds ---
    # The frustum is typically NOT centred at the LED centroid: the larger-radius
    # base can be much closer to the centroid than the smaller base.
    # Use the outer LED z_rel extremes as the bounds — they are data-driven and
    # require no hardcoded physical dimensions.
    #
    # z_frustum_top : axial position (along ring_axis) of the outermost outer LED
    #                 ≈ the large-radius base (ring_axis points toward larger base,
    #                   so this is positive)
    # z_frustum_bot : axial position of the innermost outer LED
    #                 ≈ the small-radius base (negative)
    #
    # Note: inner LEDs face inward AND toward the large base (+z), so they are
    # never visible from below the small base — any false-negative from z_frustum_bot
    # not reaching the true small base is suppressed by the normal-facing check.
    led_axial_half = float((axial_projs.max() - axial_projs.min()) / 2)
    if len(outer_idx) >= 2:
        z_frustum_top = float(outer_z_rel.max())
        z_frustum_bot = float(outer_z_rel.min())
    else:
        z_frustum_top =  led_axial_half
        z_frustum_bot = -led_axial_half

    # h_ax kept for backward-compatible cylinder fallback
    h_ax = max(abs(z_frustum_top), abs(z_frustum_bot))

    # --- h_corpus: radial gap from each inner LED to the outer frustum wall ---
    n = len(positions)
    h_corpus = np.zeros(n, dtype=np.float64)
    for i in np.where(is_inner)[0]:
        R_out_i    = R_fc + frustum_slope * z_rel[i]  # outer wall radius at this z
        r_led_i    = float(np.linalg.norm(rel_proj[i]))
        h_corpus[i] = max(0.0, R_out_i - r_led_i)

    # --- Ring radius and angular threshold ---
    R_ring    = float(np.linalg.norm(rel_proj, axis=1).mean())
    inner_idx = np.where(is_inner)[0]
    r_tube    = float(h_corpus[inner_idx].mean() / 2) if len(inner_idx) else 0.005
    angular_threshold = -float(np.sqrt(max(0.0, 1.0 - (r_tube / R_ring) ** 2))) if R_ring > 0 else -1.0

    return (ring_axis, is_inner, radial_out, h_corpus, h_ax, ring_center_ax, R_ring,
            angular_threshold, centroid, R_fc, frustum_slope,
            z_frustum_top, z_frustum_bot)


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

    # Ensure torus geometry is available for visibility checks below.
    # (Normally computed lazily by brute_match on first call, but
    # proximity_match may be called first on warm-start frames.)
    if not hasattr(self, "_is_inner"):
        (self._ring_axis, self._is_inner, self._radial_out,
         self._h_corpus, self._h_ax, self._ring_center_ax,
         self._R_ring, self._angular_threshold,
         self._ring_centroid, self._R_frustum_center, self._frustum_slope,
         self._z_frustum_top, self._z_frustum_bot,
         ) = _compute_torus_geometry(
            self.model.positions.astype(np.float32),
            self.model.normals.astype(np.float32),
        )

    is_inner_pm          = self._is_inner
    radial_out_pm        = self._radial_out
    ring_axis_pm         = self._ring_axis
    h_corpus_pm          = self._h_corpus
    h_ax_pm              = self._h_ax
    ring_center_ax_pm    = self._ring_center_ax
    R_ring_pm            = self._R_ring
    angular_thresh_pm    = self._angular_threshold
    ring_centroid_pm     = self._ring_centroid
    R_frustum_center_pm  = self._R_frustum_center
    frustum_slope_pm     = self._frustum_slope
    z_frustum_top_pm     = self._z_frustum_top
    z_frustum_bot_pm     = self._z_frustum_bot

    # ------------------------------------------------------------------
    # Path 1: assignment-locked nearest-neighbour
    # ------------------------------------------------------------------
    if prior_assignment is not None and len(prior_assignment) >= 3:
        prior_lids = [lid for _, lid in prior_assignment]
        prior_obj  = self.model.positions[prior_lids].astype(np.float32)
        proj_prior = _project_points(rvec_pred, tvec_pred, prior_obj, K, dc)

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

            # RANSAC PnP on locked pairs: rejects any pairs where the blob
            # drifted to the wrong LED (outliers from sudden motion or occlusion).
            ok, rvec, tvec, ransac_idx = _ransac_pnp(
                lo, li, K, dc, rvec_pred, tvec_pred,
            )
            if ok:
                R_new = cv2.Rodrigues(rvec)[0]
                tvec_new = tvec.reshape(3)

                # Re-check visibility with the *refined* pose, not the predicted one.
                # This is the fix for inner-ring LEDs that were visible last frame
                # but are now occluded by the controller body: they were in the
                # prior assignment and pass the nearest-neighbour snap, but they
                # must be dropped before the result is returned.
                vis_mask_new = _visible_mask(
                    R_new, tvec_new,
                    self.model.positions, self.model.normals,
                    is_inner_pm, radial_out_pm, ring_axis_pm,
                    h_corpus_pm, h_ax_pm, ring_center_ax_pm,
                    R_ring_pm, angular_thresh_pm,
                    ring_centroid_pm, R_frustum_center_pm, frustum_slope_pm,
                    z_frustum_top_pm, z_frustum_bot_pm,
                )

                # Final pairs: must be a RANSAC inlier AND visible with new pose.
                final_pairs = [
                    locked_pairs[k]
                    for k in ransac_idx
                    if vis_mask_new[locked_pairs[k][1]]
                ]

                if len(final_pairs) >= 3:
                    lo_f = self.model.positions[[l for _, l in final_pairs]].astype(np.float32)
                    li_f = blobs[[b for b, _ in final_pairs]].astype(np.float32)
                    proj  = _project_points(rvec, tvec, lo_f, K, dc)
                    error = float(np.mean(np.linalg.norm(proj - li_f, axis=1)))
                    if error < max_distance_px:
                        return {
                            "rvec":       rvec,
                            "tvec":       tvec,
                            "error":      error,
                            "assignment": final_pairs,
                            "method":     "proximity_locked",
                        }

    # ------------------------------------------------------------------
    # Path 2: full Hungarian (no prior, or locked path failed)
    # ------------------------------------------------------------------
    R_pred, _ = cv2.Rodrigues(rvec_pred)
    vis_mask  = _visible_mask(R_pred, tvec_pred,
                              self.model.positions, self.model.normals,
                              is_inner_pm, radial_out_pm, ring_axis_pm,
                              h_corpus_pm, h_ax_pm, ring_center_ax_pm,
                              R_ring_pm, angular_thresh_pm,
                              ring_centroid_pm, R_frustum_center_pm, frustum_slope_pm,
                              z_frustum_top_pm, z_frustum_bot_pm)
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

    ok, rvec, tvec, ransac_idx = _ransac_pnp(
        obj_pts, img_pts, K, dc, rvec_pred, tvec_pred,
    )
    if not ok:
        return None

    # Take only the RANSAC inliers as the final assignment.
    final_matches = [matches[k] for k in ransac_idx]
    lo_f = self.model.positions[[l for _, l in final_matches]].astype(np.float32)
    li_f = blobs[[b for b, _ in final_matches]].astype(np.float32)
    proj  = _project_points(rvec, tvec, lo_f, K, dc)
    error = float(np.mean(np.linalg.norm(proj - li_f, axis=1)))
    if error > max_distance_px:
        return None

    return {
        "rvec":       rvec,
        "tvec":       tvec,
        "error":      error,
        "assignment": final_matches,
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
    min_inlier_fraction: float = 0.6,   # min fraction of visible blobs that must be inliers
    strong_match_inliers: int = 7,      # stop immediately when this many inliers …
    strong_match_error_px: float = 1.5, # … and mean reprojection error is below this
    angle_tolerance_deg: float = 20.0,  # triangle-angle pre-filter tolerance (degrees)
    pose_prior: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # (rvec, tvec) of previous pose
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

    # Require at least min_inlier_fraction of the visible blobs to be matched,
    # but never less than the absolute floor (min_inliers).
    min_inliers_eff = max(min_inliers, int(np.ceil(min_inlier_fraction * n_blobs)))

    positions = self.model.positions.astype(np.float32)
    normals   = self.model.normals.astype(np.float32)
    n_leds    = len(positions)
    K         = self.camera.camera_matrix
    dc        = self.camera.dist_coeffs

    # ── Build / reuse geometry and LED quad cache ─────────────────────────────
    if not hasattr(self, "_led_nbr") or len(self._led_nbr) != n_leds:
        (self._ring_axis, self._is_inner, self._radial_out,
         self._h_corpus, self._h_ax, self._ring_center_ax,
         self._R_ring, self._angular_threshold,
         self._ring_centroid, self._R_frustum_center, self._frustum_slope,
         self._z_frustum_top, self._z_frustum_bot,
         ) = _compute_torus_geometry(positions, normals)
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
    ring_centroid     = self._ring_centroid
    R_frustum_center  = self._R_frustum_center
    frustum_slope     = self._frustum_slope
    z_frustum_top     = self._z_frustum_top
    z_frustum_bot     = self._z_frustum_bot

    # ── Per-frame blob quad precomputation ────────────────────────────────────
    blob_nbr = _build_blob_neighbor_lists(blobs, k=blob_neighbor_depth)
    blob_quad_idx, blob_quad_ang = _build_blob_quads(blobs, blob_nbr, blob_neighbor_depth)
    if len(blob_quad_idx) == 0:
        return None

    ANGLE_TOL      = np.deg2rad(angle_tolerance_deg)
    p4_thresh_sq   = p4_threshold_px ** 2
    fx, fy         = float(K[0, 0]), float(K[1, 1])
    cx, cy         = float(K[0, 2]), float(K[1, 2])

    # Pre-compute prior rotation matrix for orientation tiebreaking.
    R_prior = None
    if pose_prior is not None:
        rvec_pr, _ = pose_prior
        R_prior, _ = cv2.Rodrigues(np.asarray(rvec_pr, dtype=np.float32).reshape(3, 1))

    best_solution    = None
    best_inliers     = 0
    best_error       = np.inf
    best_orient_err  = np.inf   # angular distance to prior (for tiebreaking)
    strong_found     = False
    counter          = 0

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
                    ring_centroid, R_frustum_center, frustum_slope,
                    z_frustum_top, z_frustum_bot,
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

                if len(ib) < min_inliers_eff:
                    continue

                # ── 5. RANSAC PnP refinement on inliers ───────────────────────
                # _ransac_pnp undistorts first (Monado approach), uses SQPNP as
                # the minimal solver, and may tighten the inlier set further.
                ok_r, rvec_r, tvec_r, ransac_inliers = _ransac_pnp(
                    positions[il], blobs[ib], K, dc,
                    rvec_h, tvec_h.reshape(3, 1),
                )
                if not ok_r:
                    continue

                # Narrow inliers to what RANSAC confirmed.
                il = il[ransac_inliers]
                ib = ib[ransac_inliers]
                if len(ib) < min_inliers_eff:
                    continue

                # Re-check visibility with the *refined* pose.
                # The hypothesis pose (rvec_h, tvec_h) may differ enough from
                # (rvec_r, tvec_r) that some LEDs are no longer camera-facing or
                # are now occluded — they must be removed from the assignment.
                R_r, _ = cv2.Rodrigues(rvec_r.reshape(3, 1).astype(np.float32))
                vis_mask_r = _visible_mask(
                    R_r, tvec_r.reshape(3), positions, normals,
                    is_inner, radial_out, ring_axis,
                    h_corpus, h_ax, ring_center_ax, R_ring, angular_threshold,
                    ring_centroid, R_frustum_center, frustum_slope,
                    z_frustum_top, z_frustum_bot,
                )
                keep = vis_mask_r[il]
                il = il[keep]
                ib = ib[keep]
                if len(ib) < min_inliers_eff:
                    continue

                proj_r = _project_points(rvec_r, tvec_r, positions[il], K, dc)
                err    = float(np.mean(np.linalg.norm(proj_r - blobs[ib], axis=1)))
                n_ib   = len(ib)

                # ── Orientation distance to prior (for tiebreaking) ───────────
                orient_err = np.inf
                if R_prior is not None:
                    cos_a = np.clip((np.trace(R_r @ R_prior.T) - 1.0) / 2.0, -1.0, 1.0)
                    orient_err = float(np.arccos(cos_a))

                # ── Is this hypothesis better than the current best? ──────────
                # Rules (ported from Monado's pose_metrics_score_is_better_pose):
                #   1. Strictly more inliers AND tighter error → better.
                #   2. At least 2 more inliers with at most 10 % worse error → better.
                #   3. Same inlier count AND lower error → better.
                #   4. Same inlier count AND similar error (< 0.5 px diff) AND
                #      closer orientation to prior → better.
                err_per  = err  / max(n_ib, 1)
                best_per = best_error / max(best_inliers, 1)

                is_better = (
                    (n_ib > best_inliers and err_per < best_per) or
                    (n_ib >= best_inliers + 2 and err_per < best_per * 1.1) or
                    (n_ib == best_inliers and err < best_error) or
                    (n_ib == best_inliers and
                     abs(err - best_error) < 0.5 and
                     orient_err < best_orient_err)
                )

                if is_better:
                    best_solution = {
                        "rvec":       rvec_r,
                        "tvec":       tvec_r,
                        "inliers":    n_ib,
                        "error":      err,
                        "assignment": list(zip(ib.tolist(), il.tolist())),
                        "method":     "p3p_systematic",
                    }
                    best_inliers    = n_ib
                    best_error      = err
                    best_orient_err = orient_err

                    if best_inliers >= strong_match_inliers and best_error <= strong_match_error_px:
                        strong_found = True

    print(f"Brute-force matching evaluated {counter} P3P hypotheses")
    return best_solution