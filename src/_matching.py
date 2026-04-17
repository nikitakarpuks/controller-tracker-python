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
                  is_inner: np.ndarray,
                  radial_out: np.ndarray,
                  ring_axis: np.ndarray,
                  ring_centroid: np.ndarray,
                  R_frustum_center: float,
                  frustum_slope: float,
                  z_frustum_top: float,
                  z_frustum_bot: float) -> np.ndarray:
    """
    Boolean mask: True for each LED that is camera-facing and not occluded.
    Outer LEDs: facing-dot check (normal vs. view direction, 90° emission cone).
    Inner LEDs: additionally tested against the outer frustum (truncated cone) wall.
    """

    # ── Check 1: LED must be in front of the camera (positive depth) ─────────────
    # Transform LED positions to camera space: led_cam = R @ pos + t.
    # Anything with z ≤ 0 is behind the image plane — physically impossible to see.
    led_cam = (R @ positions.T).T + tvec              # LED positions in camera coords (N, 3)
    z_ok    = led_cam[:, 2] > 0.01

    # ── Check 2: LED must face the camera (emission-cone test) ───────────────────
    # Each LED emits into a 90° half-angle cone in the direction of its surface normal.
    # We check whether the camera lies inside that cone.
    #
    # Geometry: view_dir points FROM camera TO the LED (same direction as led_cam).
    #           normals_cam is the LED surface normal in camera space — it points AWAY
    #           from the LED surface. For the camera to be inside the emission cone,
    #           normal and view_dir must oppose each other → dot product < 0.
    #           Threshold 0.0 = exactly 90° half-angle, matching OpenHMD.
    view_dir    = led_cam / (np.linalg.norm(led_cam, axis=1, keepdims=True) + 1e-8)  # unit vector TO led
    normals_cam = (R @ normals.T).T                                                    # normals in camera space
    dot         = (normals_cam * view_dir).sum(axis=1)   # < 0 → normal opposes view → LED faces camera
    mask        = z_ok & (dot < 0.0)

    # Outer LEDs have no occlusion geometry to test — return early.
    if is_inner is None or radial_out is None or ring_axis is None or not np.any(is_inner):
        return mask

    # ── Inner LEDs: frustum wall occlusion test ───────────────────────────────────
    # Inner LEDs sit inside the controller body. Even if they face the camera, the
    # outer conical wall (the frustum) can block the line of sight from outside.
    # We model the wall as a truncated cone: R_outer(z) = R_fc + slope * z_rel,
    # where z_rel is the axial distance from the ring centroid along ring_axis.
    #
    # Test: cast a ray from camera to each inner LED. If the ray intersects the
    # cone surface at a point that is (a) strictly between camera and LED (t ∈ (0,1))
    # AND (b) within the frustum's axial extent [z_frustum_bot, z_frustum_top],
    # the LED is blocked.

    cam_world    = -(R.T @ tvec)          # camera position in world space
    inner_active = np.where(is_inner)[0]  # indices of inner LEDs within the input array
    P            = positions[inner_active]  # (M, 3) positions of inner LEDs only

    # ── Step A: decompose into axial + radial components (ring-centroid frame) ──
    # ring_axis defines the symmetry axis of the frustum (pointing toward larger base).
    # Splitting each point into its axial projection and radial remainder lets us
    # work with the cone equation in a clean cylindrical-like coordinate system.
    C_rel = cam_world - ring_centroid       # camera offset from ring centroid
    P_rel = P - ring_centroid               # LED offsets from ring centroid (M, 3)

    C_ax  = float(C_rel @ ring_axis)          # camera's axial coordinate (scalar)
    P_ax  = (P_rel * ring_axis).sum(axis=1)   # each LED's axial coordinate (M,)

    C_rad = C_rel - C_ax * ring_axis          # camera radial vector (perpendicular to axis)
    P_rad = P_rel - np.outer(P_ax, ring_axis) # LED radial vectors (M, 3)

    # ── Step B: parametrize the camera→LED ray ────────────────────────────────────
    # Q(t) = camera + t * (LED - camera),  t ∈ (0, 1) is strictly between them.
    # Split the ray direction into axial and radial parts as well.
    d_ax  = P_ax - C_ax    # rate of change of axial coord along the ray (M,)
    d_rad = P_rad - C_rad  # rate of change of radial vector along the ray (M, 3)

    # ── Step C: express the outer-wall radius limit as a linear function of t ─────
    # At parameter t the ray is at axial position C_ax + t*d_ax, so the outer wall
    # radius at that point is: R(t) = (R_fc + slope*C_ax) + slope*d_ax*t = A0 + Ad*t.
    A0 = float(R_frustum_center) + float(frustum_slope) * C_ax  # wall radius at t=0 (camera pos)
    Ad = float(frustum_slope) * d_ax                             # how wall radius changes along ray (M,)

    # ── Step D: find t where the ray touches the cone wall → solve quadratic ──────
    # Intersection condition: |Q_rad(t)| = A0 + Ad*t
    # Squaring both sides and rearranging:
    #   (|d_rad|² - Ad²) t²  +  2(C_rad·d_rad - A0·Ad) t  +  (|C_rad|² - A0²) = 0
    #       a                         b/2                             c
    a     = (d_rad ** 2).sum(axis=1) - Ad ** 2                  # (M,)
    b     = 2.0 * ((C_rad * d_rad).sum(axis=1) - A0 * Ad)      # (M,)
    c_val = float(np.dot(C_rad, C_rad)) - A0 ** 2               # scalar (same camera for all LEDs)

    disc    = b ** 2 - 4.0 * a * c_val   # discriminant (M,); disc < 0 → ray misses cone entirely
    blocked = np.zeros(len(inner_active), dtype=bool)

    def _in_axial(z: np.ndarray) -> np.ndarray:
        # True if the intersection axial position falls within the physical frustum extent.
        return (z >= z_frustum_bot) & (z <= z_frustum_top)

    # ── Step E: evaluate both roots and check whether either blocks the ray ───────
    # For each root t: blocking requires t ∈ (0, 1) AND axial pos inside frustum.
    # t < 0 → intersection behind camera; t > 1 → beyond LED; both are non-blocking.
    has_roots = disc >= 0.0
    if np.any(has_roots):
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        safe_2a   = np.where(np.abs(a) > 1e-12, 2.0 * a, 1.0)
        for sign in (-1.0, 1.0):
            t = np.where(np.abs(a) > 1e-12,
                         (-b + sign * sqrt_disc) / safe_2a,
                         2.0)             # a≈0 handled below; 2.0 keeps t outside (0,1)
            in_range = has_roots & (t > 1e-4) & (t < 1.0 - 1e-4)
            if not np.any(in_range):
                continue
            z_rel_t = C_ax + t * d_ax    # axial position of intersection point (M,)
            blocked |= in_range & _in_axial(z_rel_t)

    # ── Step F: linear fallback when a ≈ 0 (ray nearly tangent to cone) ──────────
    # When a ≈ 0 the quadratic degenerates to b*t + c = 0 → single root t = -c/b.
    lin_case = np.abs(a) <= 1e-12
    if np.any(lin_case):
        t_lin = np.where(np.abs(b) > 1e-12, -c_val / b, 2.0)
        in_range_lin = lin_case & (t_lin > 1e-4) & (t_lin < 1.0 - 1e-4)
        if np.any(in_range_lin):
            blocked |= in_range_lin & _in_axial(C_ax + t_lin * d_ax)

    mask[inner_active[blocked]] = False
    return mask


def _compute_frustum_geometry(positions: np.ndarray, normals: np.ndarray):
    """
    Fit a truncated cone to the LED ring; return all geometry needed for
    inner-LED occlusion testing in _visible_mask. Called once at tracker init.
    """
    # ── Step 1: find ring_axis via SVD ────────────────────────────────────────────
    # The LEDs lie approximately on a ring (a circle tilted in 3D space).
    # SVD of the centered positions gives a set of principal axes; the axis with
    # the *smallest* singular value is the one with the least variance — that is
    # the normal to the ring plane, i.e. the symmetry axis of the frustum.
    centroid  = positions.mean(axis=0)
    _, _, Vt  = np.linalg.svd(positions - centroid)
    ring_axis = Vt[-1]    # last row of Vt → direction of minimum spread

    # ── Step 2: compute per-LED radial direction (outward from ring center) ───────
    # Project each LED position onto the ring plane (remove the axial component),
    # then normalise. This gives the unit vector pointing outward from the ring axis
    # to the LED — used to classify inner vs. outer and for the frustum geometry.
    rel        = positions - centroid
    rel_proj   = rel - np.outer(rel @ ring_axis, ring_axis)  # radial component only (axial removed)
    radial_out = rel_proj / (np.linalg.norm(rel_proj, axis=1, keepdims=True) + 1e-8)

    # ── Step 3: classify inner vs. outer LEDs ─────────────────────────────────────
    # Outer LEDs face outward (normal aligned with radial_out → dot > 0).
    # Inner LEDs face inward toward the ring center (normal opposes radial_out → dot < 0).
    is_inner = (normals * radial_out).sum(axis=1) < 0

    # ── Step 4: orient ring_axis toward the larger-radius base ───────────────────
    # The frustum has two bases: a wide end and a narrow end. We want ring_axis to
    # point toward the wide (large-radius) end so that frustum_slope > 0 by convention.
    # Intuition: outer LED normals tilt toward the narrow end to face outward on a cone.
    # When ring_axis points toward the large base, the axial component of outer normals
    # is negative (normals tilt away from ring_axis). If it's positive, flip.
    outer_mask = ~is_inner
    if float((normals[outer_mask] @ ring_axis).mean()) > 0:
        big_ring_axis = -ring_axis   # flip so outer normals have negative axial component
    else:
        big_ring_axis = ring_axis

    # ── Step 5: compute per-LED axial positions (z_rel) ──────────────────────────
    # z_rel is each LED's signed distance along big_ring_axis from the ring centroid.
    # Large-base LEDs have z_rel > 0; narrow-base LEDs have z_rel < 0.
    axial_projs    = positions @ big_ring_axis        # (N,) raw axial projections
    ring_center_ax = float(axial_projs.mean())
    z_rel          = axial_projs - ring_center_ax     # centred: 0 at ring centroid (N,)

    # ── Step 6: fit the outer frustum as a linear cone ───────────────────────────
    # Model: R_outer(z_rel) = R_fc + slope * z_rel
    # We fit this line through the outer LEDs using least squares, where:
    #   - outer_radial is the measured distance from each outer LED to the ring axis
    #   - outer_z_rel  is its axial position
    # slope > 0 because ring_axis points toward the wider base (radius increases with z).
    outer_idx    = np.where(outer_mask)[0]
    outer_radial = np.linalg.norm(rel_proj[outer_idx], axis=1)  # radial distance per outer LED
    outer_z_rel  = z_rel[outer_idx]

    A = np.column_stack([np.ones(len(outer_idx)), outer_z_rel])
    coeffs, _, _, _ = np.linalg.lstsq(A, outer_radial, rcond=None)
    R_fc          = float(coeffs[0])  # outer wall radius at ring centroid (z_rel = 0)
    frustum_slope = float(coeffs[1])  # dR/dz_rel; > 0 since ring_axis → large base

    # ── Step 7: asymmetric axial bounds for the frustum ──────────────────────────
    # The frustum is NOT symmetric around the centroid — the wide base is typically
    # much closer to the centroid than the narrow base. Using data-driven bounds
    # (outer LED z_rel extremes ± small margin) avoids hardcoding any dimensions.
    # A 5 mm margin on each side accounts for the physical LED-to-edge gap.
    z_frustum_top = float(outer_z_rel.max()) + 0.004   # just beyond the wide-base LEDs
    z_frustum_bot = float(outer_z_rel.min()) - 0.005   # just beyond the narrow-base LEDs

    # # --- h_corpus: radial gap from each inner LED to the outer frustum wall ---
    # n = len(positions)
    # h_corpus = np.zeros(n, dtype=np.float64)
    # for i in np.where(is_inner)[0]:
    #     R_out_i    = R_fc + frustum_slope * z_rel[i]  # outer wall radius at this z
    #     r_led_i    = float(np.linalg.norm(rel_proj[i]))
    #     h_corpus[i] = max(0.0, R_out_i - r_led_i)

    return (big_ring_axis, is_inner, radial_out, centroid,
            R_fc, frustum_slope, z_frustum_top, z_frustum_bot)


def _build_led_neighbor_lists(positions: np.ndarray, normals: np.ndarray, k: int = 8) -> List[np.ndarray]:
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



def _precompute_led_quads(
    positions: np.ndarray, led_nbr: List[np.ndarray], k: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enumerate all LED quadruples. Called once and cached.
    No deduplication across anchors — every LED is tried as anchor,
    matching OpenHMD's structure where each LED can be the P3P anchor.

    Returns
    -------
    indices : (N, 4) int32  — [anchor, l1, l2, l3]; anchor/l1/l2 → P3P triple, l3 → 4th-point gate
    depths  : (N,) int32   — max neighbor rank used (1-based); used to split shallow vs deep passes
    """
    idx_rows:   List[Tuple] = []
    depth_rows: List[int]   = []
    n = len(positions)
    for anchor in range(n):
        nbrs = led_nbr[anchor][:k]
        if len(nbrs) < 3:
            continue
        for i1, i2, i3 in combinations(range(len(nbrs)), 3):
            l1, l2, l3 = int(nbrs[i1]), int(nbrs[i2]), int(nbrs[i3])
            depth = i3 + 1  # i3 is always the largest index (combinations are sorted)
            for perm in (
                (anchor, l1, l2, l3),
                (anchor, l2, l1, l3),
            ):
                idx_rows.append(perm)
                depth_rows.append(depth)
    if not idx_rows:
        return np.zeros((0, 4), dtype=np.int32), np.zeros(0, dtype=np.int32)
    return (
        np.array(idx_rows,   dtype=np.int32),
        np.array(depth_rows, dtype=np.int32),
    )



# ---------------------------------------------------------------------------
# brute_match helpers
# ---------------------------------------------------------------------------

def _check_z_range(tvec_h: np.ndarray, z_min: float = 0.05, z_max: float = 15.0) -> bool:
    """Return True if the hypothesis depth is within plausible range (OpenHMD: 0.05–15 m)."""
    return z_min < tvec_h[2] < z_max


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

    # Frustum geometry is precomputed once in SingleViewTracker.__init__.
    is_inner_pm          = self._is_inner
    radial_out_pm        = self._radial_out
    ring_axis_pm         = self._ring_axis
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
                    ring_centroid_pm,
                    R_frustum_center_pm, frustum_slope_pm,
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
                              ring_centroid_pm, R_frustum_center_pm,
                              frustum_slope_pm, z_frustum_top_pm, z_frustum_bot_pm)
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
    blob_neighbor_depth: int = 6,       # k nearest blob neighbours to enumerate (full depth)
    p4_threshold_px: float = 2.0,       # 4th-point pre-filter gate (pixels)
    reprojection_threshold: float = 5.0,
    min_inliers: int = 4,
    min_inlier_fraction: float = 0.8,   # min fraction of visible blobs that must be inliers
    strong_match_inliers: int = 7,      # stop immediately when this many inliers …
    strong_match_error_px: float = 1.5, # … and mean reprojection error is below this
    shallow_led_depth: int = 4,         # pass 1: LED quads using only the first N neighbors
    shallow_blob_depth: int = 4,        # pass 1: blob combinations up to this neighbor rank
    pose_prior: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    rng_seed: Optional[int] = 42,
    debug_led_ids: Optional[List[int]]=None,   # exact ordered [anchor, l1, l2, l3] to watch
    debug_blob_ids: Optional[List[int]]=None,  # exact ordered [b_anchor, b1, b2, b3] to watch
) -> Optional[Dict]:
    """
    Exhaustive pose search via P3P over LED/blob quadruple correspondences.
    Follows OpenHMD's correspondence_search structure exactly:
      - Outer loop: LED quad (precomputed; anchor fixed, 2 permutations of the P3P triple)
      - Middle loop: every blob as a potential blob anchor
      - Inner loop: combinations of 3 from the blob anchor's neighbor list

    Two passes (OpenHMD shallow/deep split):
      Pass 1 — shallow: LED quads whose deepest neighbor rank ≤ shallow_led_depth,
               blob combos up to shallow_blob_depth.  Exits on strong match.
      Pass 2 — deep: LED quads whose deepest neighbor rank > shallow_led_depth,
               blob combos up to full blob_neighbor_depth.

    Per hypothesis (P3P output):
      1. Depth range check (0.05–15 m).
      2. Fast 4th-point pixel gate — rejects most wrong hypotheses cheaply.
      3. Full visible-LED inlier count + RANSAC PnP refinement.
      4. Visibility recheck on inliers with the refined pose.

    Debug: set debug_led_ids and debug_blob_ids (both ordered [anchor, p1, p2, gate]) to
    trace exactly what happens when that specific LED/blob quad is evaluated.
    """
    blobs   = np.asarray(blobs, dtype=np.float32)
    n_blobs = len(blobs)
    if n_blobs < 4:
        return None

    min_inliers_eff    = max(min_inliers, int(np.ceil(min_inlier_fraction * n_blobs)))
    strong_inliers_eff = min(strong_match_inliers, int(np.ceil(min_inlier_fraction * n_blobs)))

    positions = self.model.positions.astype(np.float32)
    normals   = self.model.normals.astype(np.float32)
    K         = self.camera.camera_matrix
    dc        = self.camera.dist_coeffs

    led_quad_idx   = self._led_quad_idx    # (N_LQ, 4) int32
    led_quad_depth = self._led_quad_depth  # (N_LQ,) int32

    is_inner         = self._is_inner
    radial_out       = self._radial_out
    ring_axis        = self._ring_axis
    ring_centroid    = self._ring_centroid
    R_frustum_center = self._R_frustum_center
    frustum_slope    = self._frustum_slope
    z_frustum_top    = self._z_frustum_top
    z_frustum_bot    = self._z_frustum_bot

    # ── Per-frame blob neighbor lists ─────────────────────────────────────────
    blob_nbr = _build_blob_neighbor_lists(blobs, k=blob_neighbor_depth)

    p4_thresh_sq = p4_threshold_px ** 2
    fx, fy       = float(K[0, 0]), float(K[1, 1])
    cx, cy       = float(K[0, 2]), float(K[1, 2])

    R_prior = None
    if pose_prior is not None:
        rvec_pr, _ = pose_prior
        R_prior, _ = cv2.Rodrigues(np.asarray(rvec_pr, dtype=np.float32).reshape(3, 1))

    best_solution   = None
    best_inliers    = 0
    best_error      = np.inf
    best_orient_err = np.inf
    strong_found    = False
    solution_pass   = None  # "shallow" or "deep"

    # Shallow LED quads: max neighbor rank ≤ shallow_led_depth.
    # Deep LED quads: max neighbor rank > shallow_led_depth (no overlap, no redundancy).
    shallow_mask = led_quad_depth <= shallow_led_depth
    deep_mask    = led_quad_depth >  shallow_led_depth
    n_shallow_lq = int(shallow_mask.sum())
    n_deep_lq    = int(deep_mask.sum())

    # Exact total P3P calls possible per pass.
    def _c3(n: int) -> int:
        return n * (n - 1) * (n - 2) // 6 if n >= 3 else 0

    def _total_blob_combos(max_blob_d: int) -> int:
        return sum(_c3(min(len(blob_nbr[b]), max_blob_d)) for b in range(n_blobs))

    total_blob_combos_shallow = _total_blob_combos(shallow_blob_depth)
    total_blob_combos_deep    = _total_blob_combos(blob_neighbor_depth)
    total_p3p_shallow = n_shallow_lq * total_blob_combos_shallow
    total_p3p_deep    = n_deep_lq    * total_blob_combos_deep

    rng = np.random.default_rng(rng_seed)

    # Per-pass counters.
    pass_p3p_calls   = [0, 0]
    pass_lq_tried    = [0, 0]

    debug_active = debug_led_ids is not None and debug_blob_ids is not None
    debug_led_list  = list(debug_led_ids)  if debug_active else None
    debug_blob_list = list(debug_blob_ids) if debug_active else None

    # ── Two-pass search (shallow then deep) ───────────────────────────────────
    for pass_idx, (lq_mask, max_blob_d) in enumerate((
        (shallow_mask, shallow_blob_depth),
        (deep_mask,    blob_neighbor_depth),
    )):
        if strong_found:
            break

        lq_indices = np.where(lq_mask)[0]
        if len(lq_indices) == 0:
            continue
        lq_indices = lq_indices[rng.permutation(len(lq_indices))]

        for lq_i in lq_indices:
            if strong_found:
                break

            l_ids = led_quad_idx[lq_i]   # [anchor, l1, l2, l3]
            obj3  = positions[l_ids[:3]]  # (3, 3) world points for P3P
            obj4  = positions[l_ids[3]]   # (3,)   world point for 4th-point gate

            had_p3p = False

            for b_anchor in range(n_blobs):
                if strong_found:
                    break
                nbrs   = blob_nbr[b_anchor]
                n_nbrs = min(len(nbrs), max_blob_d)
                if n_nbrs < 3:
                    continue

                for i1, i2, i3 in combinations(range(n_nbrs), 3):
                    if strong_found:
                        break
                    b1 = int(nbrs[i1])
                    b2 = int(nbrs[i2])
                    b3 = int(nbrs[i3])

                    # ── Debug trigger ─────────────────────────────────────────
                    dbg = (
                        debug_active and
                        list(l_ids) == debug_led_list and
                        [b_anchor, b1, b2, b3] == debug_blob_list
                    )
                    if dbg:
                        print(
                            f"\n[DEBUG] Target quad reached — "
                            f"LEDs {list(l_ids)}  blobs [{b_anchor},{b1},{b2},{b3}]  "
                            f"pass={'shallow' if pass_idx == 0 else 'deep'}"
                        )

                    img3 = blobs[[b_anchor, b1, b2]]  # (3, 2) image points for P3P
                    img4 = blobs[b3]                  # (2,)   image point for 4th-point gate

                    # ── 1. P3P → up to 4 pose hypotheses ─────────────────────
                    pass_p3p_calls[pass_idx] += 1
                    had_p3p = True
                    n_sols, rvecs, tvecs = cv2.solveP3P(
                        obj3.reshape(3, 1, 3),
                        img3.reshape(3, 1, 2),
                        K, dc,
                        flags=cv2.SOLVEPNP_P3P,
                    )
                    if dbg:
                        print(f"  [DEBUG] P3P returned {n_sols} solutions")
                    if not n_sols or rvecs is None:
                        continue

                    for sol_i, (rvec_h, tvec_h) in enumerate(zip(rvecs, tvecs)):
                        if strong_found:
                            break
                        rvec_h = rvec_h.reshape(3, 1).astype(np.float32)
                        tvec_h = tvec_h.reshape(3).astype(np.float32)

                        # ── 2. Depth range check (OpenHMD: 0.05 m – 15 m) ────
                        z_ok = _check_z_range(tvec_h)
                        if dbg:
                            print(f"  [DEBUG] sol {sol_i}: z={tvec_h[2]:.3f} m  depth_ok={z_ok}")
                        if not z_ok:
                            continue

                        R_h, _ = cv2.Rodrigues(rvec_h)

                        # ── 3. 4th-point pixel gate ───────────────────────────
                        p4_ok = _gate_fourth_point(R_h, tvec_h, obj4, img4, fx, fy, cx, cy, p4_thresh_sq)
                        if dbg:
                            p4_cam = R_h @ obj4 + tvec_h
                            if p4_cam[2] > 0:
                                iz = 1.0 / p4_cam[2]
                                p4_px = np.array([fx * p4_cam[0] * iz + cx, fy * p4_cam[1] * iz + cy])
                                p4_err = float(np.linalg.norm(p4_px - img4))
                            else:
                                p4_err = float('inf')
                            print(f"  [DEBUG] sol {sol_i}: 4th-point err={p4_err:.2f} px  gate_ok={p4_ok}")
                        if not p4_ok:
                            continue

                        # ── 4. Full inlier count on all visible LEDs ──────────
                        vis_mask_h = _visible_mask(
                            R_h, tvec_h, positions, normals,
                            is_inner, radial_out, ring_axis,
                            ring_centroid,
                            R_frustum_center, frustum_slope,
                            z_frustum_top, z_frustum_bot,
                        )
                        vis_ids = np.where(vis_mask_h)[0]
                        if dbg:
                            print(f"  [DEBUG] sol {sol_i}: {len(vis_ids)} visible LEDs")
                        if len(vis_ids) < min_inliers:
                            continue

                        proj_all = _project_points(rvec_h, tvec_h, positions[vis_ids], K, dc)
                        cost     = cdist(blobs, proj_all)
                        ri, ci   = linear_sum_assignment(cost)

                        inlier_mask = cost[ri, ci] < reprojection_threshold
                        ib = ri[inlier_mask]
                        il = vis_ids[ci[inlier_mask]]

                        if dbg:
                            print(f"  [DEBUG] sol {sol_i}: {len(ib)} inliers after Hungarian "
                                  f"(need {min_inliers_eff})")
                        if len(ib) < min_inliers_eff:
                            continue

                        # ── 5. RANSAC PnP refinement on inliers ───────────────
                        ok_r, rvec_r, tvec_r, ransac_inliers = _ransac_pnp(
                            positions[il], blobs[ib], K, dc,
                            rvec_h, tvec_h.reshape(3, 1),
                        )
                        if dbg:
                            print(f"  [DEBUG] sol {sol_i}: RANSAC ok={ok_r}, "
                                  f"inliers={len(ransac_inliers) if ok_r else 0}")
                        if not ok_r:
                            continue

                        il = il[ransac_inliers]
                        ib = ib[ransac_inliers]

                        if len(ib) < min_inliers_eff:
                            continue

                        # ── 6. Visibility recheck with the refined pose ───────
                        R_r, _ = cv2.Rodrigues(rvec_r.reshape(3, 1).astype(np.float32))
                        vis_sub = _visible_mask(
                            R_r, tvec_r.reshape(3), positions[il], normals[il],
                            is_inner[il], radial_out[il], ring_axis,
                            ring_centroid,
                            R_frustum_center, frustum_slope,
                            z_frustum_top, z_frustum_bot,
                        )
                        il = il[vis_sub]
                        ib = ib[vis_sub]
                        if dbg:
                            print(f"  [DEBUG] sol {sol_i}: {len(ib)} inliers after vis recheck")
                        if len(ib) < min_inliers_eff:
                            continue

                        proj_r = _project_points(rvec_r, tvec_r, positions[il], K, dc)
                        err    = float(np.mean(np.linalg.norm(proj_r - blobs[ib], axis=1)))
                        n_ib   = len(ib)

                        orient_err = np.inf
                        if R_prior is not None:
                            cos_a = np.clip((np.trace(R_r @ R_prior.T) - 1.0) / 2.0, -1.0, 1.0)
                            orient_err = float(np.arccos(cos_a))

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

                        if dbg:
                            print(f"  [DEBUG] sol {sol_i}: err={err:.3f} px  inliers={n_ib}  "
                                  f"is_better={is_better}")

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
                            solution_pass   = "shallow" if pass_idx == 0 else "deep"

                            if best_inliers >= strong_inliers_eff and best_error <= strong_match_error_px:
                                strong_found = True

            if had_p3p:
                pass_lq_tried[pass_idx] += 1

    # ── Summary print ─────────────────────────────────────────────────────────
    total_p3p_tried = pass_p3p_calls[0] + pass_p3p_calls[1]
    total_p3p_possible = total_p3p_shallow + total_p3p_deep
    result_str = (
        f"found in {solution_pass} pass  "
        f"({best_inliers} inliers, {best_error:.2f} px)"
        if best_solution is not None else "not found"
    )
    print(
        f"Brute-force: {result_str}\n"
        f"  shallow — {pass_p3p_calls[0]:>7} / {total_p3p_shallow:>7} P3P calls  "
        f"({pass_lq_tried[0]}/{n_shallow_lq} LED quads reached inner loop)\n"
        f"  deep    — {pass_p3p_calls[1]:>7} / {total_p3p_deep:>7} P3P calls  "
        f"({pass_lq_tried[1]}/{n_deep_lq} LED quads reached inner loop)\n"
        f"  total   — {total_p3p_tried:>7} / {total_p3p_possible:>7} P3P calls"
    )
    return best_solution
