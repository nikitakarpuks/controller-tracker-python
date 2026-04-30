import math

import cv2
import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from src.debug_config import is_deep, get_debug_triple, is_verbose_all, log_best
from src.geometry import ControllerGeometry, tangent_frame


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
    reprojection_px: float = 2.0,
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


def _rays_blocked_by_box(cam: np.ndarray, leds: np.ndarray, box) -> np.ndarray:
    """
    Vectorised OBB slab test.
    Returns bool (N,) — True where the camera→led[i] segment is blocked by box.
    """
    M  = np.asarray(box.axes, float) if box.axes is not None else np.eye(3)
    c  = np.asarray(box.center, float)
    h  = np.asarray(box.half_dims, float)

    D  = leds - cam       # (N, 3) segment vectors
    oc = c - cam          # (3,)
    e  = M.T @ oc         # (3,) center-offset in box local frame
    f  = (M.T @ D.T).T    # (N, 3) directions in box local frame

    t_min = np.full(len(leds), 1e-4)
    t_max = np.full(len(leds), 1.0 - 1e-4)

    for i in range(3):
        nz    = np.abs(f[:, i]) > 1e-12
        safe  = np.where(nz, f[:, i], 1.0)
        t1    = np.where(nz, (e[i] - h[i]) / safe, -np.inf)
        t2    = np.where(nz, (e[i] + h[i]) / safe,  np.inf)
        swap  = t1 > t2
        t1, t2 = np.where(swap, t2, t1), np.where(swap, t1, t2)
        t_min = np.maximum(t_min, t1)
        t_max = np.minimum(t_max, t2)
        t_max = np.where(~nz & (np.abs(e[i]) > h[i]), -np.inf, t_max)

    return t_min < t_max


def _rays_blocked_by_cylinder(cam: np.ndarray, leds: np.ndarray, cy) -> np.ndarray:
    """
    Vectorised ray vs elliptic cylinder (side wall + end caps).
    Returns bool (N,) — True where the camera→led[i] segment is blocked.
    """
    ax = np.asarray(cy.axis, float); ax /= np.linalg.norm(ax) + 1e-9
    u, v = tangent_frame(ax)
    if cy.angle:
        ca, sa = np.cos(cy.angle), np.sin(cy.angle)
        u, v = ca * u + sa * v, -sa * u + ca * v

    r_u = float(cy.radius)
    r_v = float(cy.radius_v) if cy.radius_v is not None else r_u
    c   = np.asarray(cy.center, float)
    hl  = float(cy.half_length)

    D  = leds - cam   # (N, 3)
    oc = cam - c      # (3,)

    oc_ax = float(np.dot(oc, ax))
    oc_u  = float(np.dot(oc, u))
    oc_v  = float(np.dot(oc, v))
    D_ax  = D @ ax    # (N,)
    D_u   = D @ u     # (N,)
    D_v   = D @ v     # (N,)

    eps     = 1e-4
    blocked = np.zeros(len(leds), dtype=bool)

    # ── Side wall: quadratic in t ────────────────────────────────────────────
    a     = (D_u / r_u) ** 2 + (D_v / r_v) ** 2
    b     = 2.0 * (oc_u * D_u / r_u ** 2 + oc_v * D_v / r_v ** 2)
    c_val = oc_u ** 2 / r_u ** 2 + oc_v ** 2 / r_v ** 2 - 1.0

    disc = b ** 2 - 4.0 * a * c_val
    has_roots = disc >= 0.0
    if np.any(has_roots):
        sqrt_d  = np.sqrt(np.maximum(disc, 0.0))
        safe_2a = np.where(np.abs(a) > 1e-12, 2.0 * a, 1.0)
        for sign in (-1.0, 1.0):
            t = np.where(np.abs(a) > 1e-12, (-b + sign * sqrt_d) / safe_2a, 2.0)
            ok = has_roots & (t > eps) & (t < 1.0 - eps)
            if np.any(ok):
                z_t = oc_ax + t * D_ax
                blocked |= ok & (np.abs(z_t) <= hl)

    # ── End caps ─────────────────────────────────────────────────────────────
    for z_cap in (-hl, hl):
        safe_ax = np.where(np.abs(D_ax) > 1e-12, D_ax, 1.0)
        t_cap   = np.where(np.abs(D_ax) > 1e-12, (z_cap - oc_ax) / safe_ax, 2.0)
        ok      = (t_cap > eps) & (t_cap < 1.0 - eps)
        if np.any(ok):
            xu = oc_u + t_cap * D_u
            xv = oc_v + t_cap * D_v
            blocked |= ok & ((xu / r_u) ** 2 + (xv / r_v) ** 2 <= 1.0)

    return blocked


def _box_entry_t(cam: np.ndarray, led: np.ndarray, box) -> Optional[float]:
    """Return the entry t in (0,1) of ray cam→led through box, or None if not blocked."""
    M  = np.asarray(box.axes, float) if box.axes is not None else np.eye(3)
    c  = np.asarray(box.center, float)
    h  = np.asarray(box.half_dims, float)
    D  = led - cam
    oc = c - cam
    e  = M.T @ oc
    f  = M.T @ D
    t_min, t_max = 1e-4, 1.0 - 1e-4
    for i in range(3):
        if abs(f[i]) > 1e-12:
            t1, t2 = (e[i] - h[i]) / f[i], (e[i] + h[i]) / f[i]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
        elif abs(e[i]) > h[i]:
            return None
    return t_min if t_min < t_max else None


def _cylinder_entry_t(cam: np.ndarray, led: np.ndarray, cy) -> Optional[float]:
    """Return the entry t in (0,1) of ray cam→led through cylinder, or None if not blocked."""
    ax = np.asarray(cy.axis, float); ax /= np.linalg.norm(ax) + 1e-9
    u, v = tangent_frame(ax)
    if cy.angle:
        ca, sa = np.cos(cy.angle), np.sin(cy.angle)
        u, v = ca * u + sa * v, -sa * u + ca * v
    r_u = float(cy.radius)
    r_v = float(cy.radius_v) if cy.radius_v is not None else r_u
    c   = np.asarray(cy.center, float)
    hl  = float(cy.half_length)
    D   = led - cam
    oc  = cam - c
    oc_ax = float(np.dot(oc, ax))
    oc_u  = float(np.dot(oc, u))
    oc_v  = float(np.dot(oc, v))
    D_ax  = float(np.dot(D, ax))
    D_u   = float(np.dot(D, u))
    D_v   = float(np.dot(D, v))
    eps = 1e-4
    t_entry = np.inf

    a     = (D_u / r_u) ** 2 + (D_v / r_v) ** 2
    b     = 2.0 * (oc_u * D_u / r_u ** 2 + oc_v * D_v / r_v ** 2)
    c_val = oc_u ** 2 / r_u ** 2 + oc_v ** 2 / r_v ** 2 - 1.0
    disc  = b ** 2 - 4.0 * a * c_val
    if disc >= 0.0 and abs(a) > 1e-12:
        sq = np.sqrt(disc)
        for sign in (-1.0, 1.0):
            t = (-b + sign * sq) / (2.0 * a)
            if eps < t < 1.0 - eps and abs(oc_ax + t * D_ax) <= hl:
                t_entry = min(t_entry, t)

    for z_cap in (-hl, hl):
        if abs(D_ax) > 1e-12:
            t = (z_cap - oc_ax) / D_ax
            if eps < t < 1.0 - eps:
                xu = oc_u + t * D_u
                xv = oc_v + t * D_v
                if (xu / r_u) ** 2 + (xv / r_v) ** 2 <= 1.0:
                    t_entry = min(t_entry, t)

    return t_entry if t_entry < np.inf else None


def _visible_mask(R: np.ndarray, tvec: np.ndarray,
                  positions: np.ndarray, normals: np.ndarray,
                  geom: ControllerGeometry,
                  is_inner: Optional[np.ndarray] = None,
                  cam_K: Optional[np.ndarray] = None,
                  cam_dc: Optional[np.ndarray] = None,
                  cam_w: int = 0,
                  cam_h: int = 0,
                  cam_rpmax: float = 0.0,
                  debug: bool = False) -> np.ndarray:
    """
    Boolean mask: True for each LED that is camera-facing and not occluded.

    Checks (in order):
      1. LED is in front of the camera (positive depth).
      2. LED faces the camera (emission-cone test, 90° half-angle).
      2b. LED projects within image bounds (distorted projection via cam_K/cam_dc).
          rpmax culls LEDs beyond the valid distortion model radius.
      3. Inner LEDs blocked by the frustum truncated-cone wall.
      4. All LEDs blocked by any handle body primitive (boxes + cylinders).

    Parameters
    ----------
    is_inner  : optional subset mask — pass when positions/normals are a subset
                of the full model (e.g. positions[il]).  If None, geom.is_inner
                is used (assumes positions == full model).
    cam_K     : 3×3 camera intrinsic matrix; if None, the in-frame check is skipped.
    cam_dc    : distortion coefficients (radtan8: k1,k2,p1,p2,k3,k4,k5,k6).
    cam_w, cam_h : image dimensions in pixels.
    cam_rpmax : maximum valid normalised radius for the distortion model;
                LEDs beyond this are culled.  0 disables the check.
    """
    # ── Check 1: positive depth ───────────────────────────────────────────────
    led_cam = (R @ positions.T).T + tvec
    z_ok    = led_cam[:, 2] > 0.01

    # ── Check 2: emission-cone facing test ───────────────────────────────────
    view_dir    = led_cam / (np.linalg.norm(led_cam, axis=1, keepdims=True) + 1e-8)
    normals_cam = (R @ normals.T).T
    dot         = (normals_cam * view_dir).sum(axis=1)
    mask        = z_ok & (dot < -0.05)

    # ── Check 2b: within image frame ─────────────────────────────────────────
    # Must use distorted projection (cv2.projectPoints), not pinhole, because
    # wide-angle lenses with barrel distortion compress large-angle points back
    # into the image — pinhole would incorrectly cull those LEDs.
    if cam_K is not None and cam_w > 0 and cam_h > 0:
        active = np.where(mask)[0]
        if len(active) > 0:
            dc = cam_dc if cam_dc is not None else np.zeros(4, dtype=np.float32)
            pts, _ = cv2.projectPoints(
                positions[active].astype(np.float32).reshape(-1, 1, 3),
                _to_rvec(R),
                tvec.astype(np.float32).reshape(3, 1),
                cam_K, dc,
            )
            pts      = pts.reshape(-1, 2)
            in_frame = (pts[:, 0] >= 0) & (pts[:, 0] < cam_w) & \
                       (pts[:, 1] >= 0) & (pts[:, 1] < cam_h)
            if cam_rpmax > 0.0:
                z_a      = led_cam[active, 2]
                rp       = np.hypot(led_cam[active, 0] / z_a, led_cam[active, 1] / z_a)
                in_frame &= rp <= cam_rpmax
            mask[active[~in_frame]] = False
            if debug:
                for k, led_id in enumerate(active):
                    if not in_frame[k]:
                        print(f"  [frame]   LED {led_id:2d}: OUT OF FRAME  "
                              f"proj=({pts[k,0]:.1f}, {pts[k,1]:.1f})")

    cam_world = -(R.T @ tvec)

    # ── Check 3: thick-frustum occlusion (inner LEDs only) ───────────────────
    # Tests all four wall surfaces: outer cone, inner cone, top cap, bottom cap.
    # A ray is blocked if any midpoint of the segments between boundary crossings
    # lies inside the solid wall region (r_inner ≤ r ≤ r_outer, z_bot ≤ z ≤ z_top).
    _is_inner = is_inner if is_inner is not None else geom.is_inner
    if debug and _is_inner is not None:
        inner_ids = np.where(_is_inner)[0].tolist()
        outer_ids = np.where(~_is_inner)[0].tolist()
        print(f"[frustum debug] inner LEDs ({len(inner_ids)}): {inner_ids}")
        print(f"[frustum debug] outer LEDs ({len(outer_ids)}): {outer_ids}")
    if _is_inner is not None and np.any(_is_inner):
        ring_axis      = geom.ring_axis
        ring_centroid  = geom.ring_centroid
        R_fc           = geom.R_fc
        R_fc_inner     = geom.R_fc_inner
        slope          = geom.frustum_slope
        z_top          = geom.z_frustum_top
        z_bot          = geom.z_frustum_bot
        ring_center_ax = geom.ring_center_ax

        inner_active = np.where(_is_inner)[0]
        P     = positions[inner_active]           # (N, 3)
        C_rel = cam_world - ring_centroid
        P_rel = P - ring_centroid

        C_ax  = float(C_rel @ ring_axis)
        P_ax  = (P_rel * ring_axis).sum(axis=1)   # (N,)
        C_rad = C_rel - C_ax * ring_axis           # (3,)
        P_rad = P_rel - np.outer(P_ax, ring_axis)  # (N, 3)

        D_ax  = P_ax - C_ax                        # (N,)
        D_rad = P_rad - C_rad                      # (N, 3)

        C_ax_rel  = C_ax - ring_center_ax
        A0_out    = R_fc       + slope * C_ax_rel
        A0_in     = R_fc_inner + slope * C_ax_rel
        Ad        = slope * D_ax                   # (N,)
        C_rad_sq  = float(np.dot(C_rad, C_rad))
        CdotD_rad = (C_rad * D_rad).sum(axis=1)    # (N,)
        D_rad_sq  = (D_rad ** 2).sum(axis=1)       # (N,)

        # Quadratic coefficients for outer and inner cone (vectorised over N)
        a      = D_rad_sq - Ad ** 2
        b_out  = 2.0 * (CdotD_rad - A0_out * Ad)
        b_in   = 2.0 * (CdotD_rad - A0_in  * Ad)
        c_out  = C_rad_sq - A0_out ** 2
        c_in   = C_rad_sq - A0_in  ** 2

        disc_out = b_out ** 2 - 4.0 * a * c_out   # (N,)
        disc_in  = b_in  ** 2 - 4.0 * a * c_in    # (N,)

        # Top / bottom plane t-values (one per LED; inf when ray is axially parallel)
        safe_D_ax = np.where(np.abs(D_ax) > 1e-12, D_ax, np.inf)
        t_top_arr = (z_top - C_ax_rel) / safe_D_ax  # (N,)
        t_bot_arr = (z_bot - C_ax_rel) / safe_D_ax  # (N,)

        EPS     = 1e-4
        blocked = np.zeros(len(inner_active), dtype=bool)

        # Detect LEDs whose static position is inside the cone wall material —
        # a linear-cone approximation artifact (the real surface is curved).
        # For these LEDs the trailing open interval [last_crossing, 1-EPS] would
        # always fire because the LED endpoint is inside the wall; dropping the
        # 1-EPS sentinel for them means only real pass-through intervals (bounded
        # by actual crossings on both sides) can trigger blocking.
        z_led_arr   = C_ax_rel + D_ax                        # (N,) LED axial z-rel
        r_led_arr   = np.linalg.norm(P_rad, axis=1)          # (N,) LED radial distance
        r_out_led   = R_fc       + slope * z_led_arr
        r_in_led    = R_fc_inner + slope * z_led_arr
        led_in_wall = ((r_in_led  <= r_led_arr) & (r_led_arr <= r_out_led) &
                       (z_bot     <= z_led_arr) & (z_led_arr <= z_top))

        if debug and led_in_wall.any():
            for i in np.where(led_in_wall)[0]:
                excess = r_led_arr[i] - r_in_led[i]
                print(f"  [frustum] LED {inner_active[i]:2d}: inside cone wall"
                      f"  (r={r_led_arr[i]*1000:.2f}mm  r_in={r_in_led[i]*1000:.2f}mm"
                      f"  excess={excess*1000:.2f}mm) — trailing sentinel suppressed")

        for i in range(len(inner_active)):
            t_cands = []

            # Outer cone crossings
            if disc_out[i] >= 0.0:
                sq  = np.sqrt(max(disc_out[i], 0.0))
                ai2 = 2.0 * a[i]
                if abs(a[i]) > 1e-12:
                    for sign in (-1.0, 1.0):
                        tc = (-b_out[i] + sign * sq) / ai2
                        if EPS < tc < 1.0 - EPS:
                            t_cands.append(tc)
                elif abs(b_out[i]) > 1e-12:
                    tc = -c_out / b_out[i]
                    if EPS < tc < 1.0 - EPS:
                        t_cands.append(tc)

            # Inner cone crossings
            if disc_in[i] >= 0.0:
                sq  = np.sqrt(max(disc_in[i], 0.0))
                ai2 = 2.0 * a[i]
                if abs(a[i]) > 1e-12:
                    for sign in (-1.0, 1.0):
                        tc = (-b_in[i] + sign * sq) / ai2
                        if EPS < tc < 1.0 - EPS:
                            t_cands.append(tc)
                elif abs(b_in[i]) > 1e-12:
                    tc = -c_in / b_in[i]
                    if EPS < tc < 1.0 - EPS:
                        t_cands.append(tc)

            # Top / bottom cap crossings
            for tc in (t_top_arr[i], t_bot_arr[i]):
                if EPS < tc < 1.0 - EPS:
                    t_cands.append(tc)

            if not t_cands:
                if debug:
                    print(f"  [frustum] LED {inner_active[i]:2d}: NO t_cands"
                          f"  disc_out={disc_out[i]:+.5f}  disc_in={disc_in[i]:+.5f}"
                          f"  t_top={t_top_arr[i]:.4f}  t_bot={t_bot_arr[i]:.4f}")
                continue

            # Check midpoint of each interval between consecutive boundary crossings.
            # A midpoint inside the wall region means the ray passes through the wall.
            # For LEDs whose position is inside the cone wall (model artifact), drop
            # the trailing 1-EPS sentinel so only intervals bounded by real crossings
            # on both sides can trigger blocking.
            t_cands_sorted = sorted(t_cands)
            # if led_in_wall[i]:
            #     intervals = [EPS] + t_cands_sorted
            # else:
            #     intervals = [EPS] + t_cands_sorted + [1.0 - EPS]

            intervals = [EPS] + t_cands_sorted + [1.0 - EPS]

            for j in range(len(intervals) - 1):
                t_mid     = 0.5 * (intervals[j] + intervals[j + 1])
                z_mid     = C_ax_rel + t_mid * D_ax[i]
                r_out_mid = R_fc       + slope * z_mid
                r_in_mid  = R_fc_inner + slope * z_mid
                r_mid     = float(np.linalg.norm(C_rad + t_mid * D_rad[i]))
                if z_bot <= z_mid <= z_top and r_in_mid <= r_mid <= r_out_mid:
                    blocked[i] = True
                    if debug:
                        seg_len_cm = float(np.linalg.norm(P_rel[i] - C_rel)) * 100.0
                        dist_cm    = (1.0 - intervals[j]) * seg_len_cm
                        print(f"  [frustum] LED {inner_active[i]:2d}: BLOCKED by frustum wall"
                              f"  ({dist_cm:.1f}cm behind wall)")
                    break

            if debug and not blocked[i]:
                tc_strs = "  ".join(f"{tc:.4f}" for tc in sorted(t_cands))
                print(f"  [frustum] LED {inner_active[i]:2d}: NOT blocked"
                      f"  t_cands=[{tc_strs}]"
                      f"  t_top={t_top_arr[i]:.4f}  t_bot={t_bot_arr[i]:.4f}"
                      f"  C_ax_rel={C_ax_rel:.4f}  D_ax={D_ax[i]:.4f}"
                      f"  z_range=[{z_bot:.4f},{z_top:.4f}]")
                for j in range(len(intervals) - 1):
                    t_mid     = 0.5 * (intervals[j] + intervals[j + 1])
                    z_mid     = C_ax_rel + t_mid * D_ax[i]
                    r_out_mid = R_fc       + slope * z_mid
                    r_in_mid  = R_fc_inner + slope * z_mid
                    r_mid     = float(np.linalg.norm(C_rad + t_mid * D_rad[i]))
                    z_ok_s = "z_ok" if z_bot <= z_mid <= z_top else f"z_FAIL({z_mid:.4f})"
                    r_ok_s = "r_ok" if r_in_mid <= r_mid <= r_out_mid else f"r_FAIL(in={r_in_mid:.4f} r={r_mid:.4f} out={r_out_mid:.4f})"
                    print(f"    [{intervals[j]:.4f},{intervals[j+1]:.4f}]"
                          f"  t_mid={t_mid:.4f}  {z_ok_s}  {r_ok_s}")

        mask[inner_active[blocked]] = False

    # ── Check 4: handle body occlusion (boxes + cylinders, all LEDs) ─────────
    active = np.where(mask)[0]
    if len(active) > 0 and (geom.boxes or geom.cylinders):
        body_blocked = np.zeros(len(active), dtype=bool)
        for box in geom.boxes:
            body_blocked |= _rays_blocked_by_box(cam_world, positions[active], box)
        for cy in geom.cylinders:
            body_blocked |= _rays_blocked_by_cylinder(cam_world, positions[active], cy)
        mask[active[body_blocked]] = False

        if debug:
            for k, led_id in enumerate(active):
                if not body_blocked[k]:
                    continue
                led_pos = positions[led_id]
                seg_len_cm = float(np.linalg.norm(led_pos - cam_world)) * 100.0
                blockers = []
                for bi, box in enumerate(geom.boxes):
                    t = _box_entry_t(cam_world, led_pos, box)
                    if t is not None:
                        blockers.append(f"box_{bi}(dist={((1.0 - t) * seg_len_cm):.1f}cm)")
                for ci, cy in enumerate(geom.cylinders):
                    t = _cylinder_entry_t(cam_world, led_pos, cy)
                    if t is not None:
                        blockers.append(f"cylinder_{ci}(dist={((1.0 - t) * seg_len_cm):.1f}cm)")
                label = ", ".join(blockers) if blockers else "unknown"
                print(f"  [body]    LED {led_id:2d}: BLOCKED by {label}")

    return mask


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


def _build_led_neighbor_lists_edge(
    positions: np.ndarray,
    normals: np.ndarray,
    is_inner: np.ndarray,
    z_rel: np.ndarray,
    k: int = 8,
) -> List[np.ndarray]:
    """
    Alternative neighbourhood for grazing-angle views (~30° to the frustum base plane)
    where both inner and outer LEDs are simultaneously visible.

    For each anchor LED the k neighbours are filled with a strict split:
      - n_same  = min(k // 2, 2): at most 2 nearest same-type LEDs (outer→outer or
                                   inner→inner) with dot >= 0, sorted by distance.
      - n_cross = k - n_same     : cross-type LEDs (outer→inner or inner→outer)
                                   with dot >= 0, sorted by normal similarity descending.

    The two halves are interleaved with cross-type first:
      cross[0], same[0], cross[1], same[1], …
    so rank 0 is always the best cross-type match (the grazing-view target), rank 1 the
    nearest same-type, and deeper ranks continue alternating.  Depth-2 triple (0,1)
    therefore pairs the best cross LED with the nearest same LED.

    debug_led_ids: if provided, print the chosen neighbours for each listed LED id.
    """
    n       = len(positions)
    n_same  = min(k // 2, 2)
    n_cross = k - n_same

    dists = cdist(positions, positions)   # (N,N) Euclidean distances
    dots  = normals @ normals.T           # (N,N) pairwise normal cosines

    result = []
    for i in range(n):
        # ── same-type: nearest by distance, dot >= 0, capped at 2 ────────────────
        same_valid = (is_inner == is_inner[i]) & (dots[i] >= 0.0)
        same_valid[i] = False
        same_cands = np.where(same_valid)[0]
        if len(same_cands):
            same_nbrs = same_cands[np.argsort(dists[i, same_cands])[:n_same]]
        else:
            same_nbrs = np.array([], dtype=int)

        # ── cross-type: most normal-similar, dot >= 0 ────────────────────────────
        cross_valid = (is_inner != is_inner[i]) & (dots[i] >= 0.0)
        cross_cands = np.where(cross_valid)[0]
        if len(cross_cands):
            cross_nbrs = cross_cands[np.argsort(-dots[i, cross_cands])[:n_cross]]
        else:
            cross_nbrs = np.array([], dtype=int)

        # ── interleave: cross[0], same[0], cross[1], same[1], … ─────────────────
        nbrs = []
        for slot in range(max(len(same_nbrs), len(cross_nbrs))):
            if slot < len(cross_nbrs):
                nbrs.append(cross_nbrs[slot])
            if slot < len(same_nbrs):
                nbrs.append(same_nbrs[slot])
        nbrs = np.array(nbrs, dtype=int)
        result.append(nbrs)

        if is_deep():
            kind = "inner" if is_inner[i] else "outer"
            lines = [f"LED {i:2d} ({kind}, z_rel={z_rel[i]:+.5f})  →  neighbours:"]
            for rank, j in enumerate(nbrs):
                jkind = "inner" if is_inner[j] else "outer"
                src   = "same " if is_inner[j] == is_inner[i] else "cross"
                lines.append(f"  rank {rank}: LED {j:2d} ({jkind}/{src}, "
                              f"z_rel={z_rel[j]:+.5f}, dot={dots[i,j]:.4f}, "
                              f"dist={dists[i,j]*1000:.1f} mm)")
            # logger.debug("\n".join(lines))

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
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Enumerate LED triples for P3P. Each unique (anchor, l1, l2) appears exactly once,
    eliminating the duplicate P3P calls caused by varying the gate LED under C(k,3).

    Returns
    -------
    triple_idx   : (N, 3) int32      — [anchor, l1, l2]; all three used for P3P
    triple_depth : (N,) int32        — max neighbour rank used (1-based); shallow/deep split
    triple_gates : List[np.ndarray]  — for each triple, remaining neighbour LED indices (gate pool)
    """
    idx_rows:   List[Tuple]       = []
    depth_rows: List[int]         = []
    gate_rows:  List[np.ndarray]  = []
    seen_led:   set               = set()
    n = len(positions)
    for anchor in range(n):
        nbrs   = led_nbr[anchor][:k]
        nb_len = len(nbrs)
        if nb_len < 2:
            continue
        for i1, i2 in combinations(range(nb_len), 2):
            l1, l2 = int(nbrs[i1]), int(nbrs[i2])
            key = tuple(sorted((anchor, l1, l2)))
            if key in seen_led:
                continue
            seen_led.add(key)
            depth  = i2 + 1
            gates  = np.array(
                [int(nbrs[j]) for j in range(nb_len) if j != i1 and j != i2],
                dtype=np.int32,
            )
            idx_rows.append((anchor, l1, l2))
            depth_rows.append(depth)
            gate_rows.append(gates)
    if not idx_rows:
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            [],
        )
    return (
        np.array(idx_rows,   dtype=np.int32),
        np.array(depth_rows, dtype=np.int32),
        gate_rows,
    )



# ---------------------------------------------------------------------------
# brute_match helpers
# ---------------------------------------------------------------------------

def _check_z_range(tvec_h: np.ndarray, z_min: float = 0.05, z_max: float = 15.0) -> bool:
    """Return True if the hypothesis depth is within plausible range (OpenHMD: 0.05–15 m)."""
    return z_min < tvec_h[2] < z_max


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

    geom = self._geometry

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
                    geom,
                    cam_K=K, cam_dc=dc, cam_w=self.camera.width, cam_h=self.camera.height,
                    cam_rpmax=self.camera.rpmax,
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
                              geom,
                              cam_K=K, cam_dc=dc, cam_w=self.camera.width, cam_h=self.camera.height,
                              cam_rpmax=self.camera.rpmax)
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

def _tier_label(t):
    nbr = t[2] if len(t) > 2 else 'standard'
    return f"led≤{t[0]}, blob≤{t[1]}, nbr={nbr}"

# ---------------------------------------------------------------------------
# brute_match
# ---------------------------------------------------------------------------

def brute_match(
    self,
    blobs: np.ndarray,
    depth_tiers: Tuple[Tuple, ...] = ((2, 3), (2, 4), (2, 4, 'edge'), (3, 5), (3, 5, 'edge'), (4, 6)),  # (led_max, blob_max[, 'standard'|'edge'])
    p4_threshold_px: float = 2.0,
    hungarian_threshold_px: float = 5.0,  # pre-filter on the raw P3P hypothesis pose. Loose because P3P poses can be noisy; RANSAC does the real filtering after this
    reprojection_threshold: float = 2.0,  # passed to RANSAC, controls which blobs make it into the final assignment. This is now what the visualization reflects: all shown errors will be ≤ this
    min_inliers: int = 4,
    min_inlier_fraction: Optional[float] = None,
    strong_match_inliers: int = 7,
    strong_match_error_px: float = 1.5,
    min_vis_coverage: float = 0.7,
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

    geom     = self._geometry
    is_inner = geom.is_inner   # kept for subset indexing: is_inner[il]

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

    R_prior = None
    if pose_prior is not None:
        rvec_pr, _ = pose_prior
        R_prior, _ = cv2.Rodrigues(np.asarray(rvec_pr, dtype=np.float32).reshape(3, 1))

    best_solution   = None
    best_inliers    = 0
    best_error      = np.inf
    best_orient_err = np.inf
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
                            )
                            # Drop inliers that became occluded under the refined pose.
                            inlier_still_visible = vis_mask_r[inlier_leds]
                            inlier_leds  = inlier_leds[inlier_still_visible]
                            inlier_blobs = inlier_blobs[inlier_still_visible]
                            if dbg:
                                logger.debug(f"  sol {sol_i}: {len(inlier_blobs)} inliers after vis recheck")
                            if len(inlier_blobs) < min_inliers_eff:
                                continue

                            # ── 7. Visibility coverage check ──────────────────
                            # Weight each visible LED by cos(θ) — the dot product
                            # of its normal with the view direction.  LEDs at
                            # grazing angles (θ → 90°) may be theoretically
                            # visible but are rarely detected reliably, so they
                            # contribute less to both numerator and denominator.
                            # This prevents those LEDs from unfairly failing the
                            # coverage gate when they simply aren't bright enough.
                            vis_ids_r      = np.where(vis_mask_r)[0]
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

                            error_per_inlier      = err  / max(n_inlier_blobs, 1)
                            best_error_per_inlier = best_error / max(best_inliers, 1)

                            # Prefer more inliers, break ties by absolute error, break
                            # further ties by orientation distance to the prior pose.
                            is_better = (
                                (n_inlier_blobs > best_inliers and error_per_inlier < best_error_per_inlier) or
                                (n_inlier_blobs >= best_inliers + 2 and error_per_inlier < best_error_per_inlier * 1.1) or
                                (n_inlier_blobs == best_inliers and err < best_error) or
                                (n_inlier_blobs == best_inliers and
                                 abs(err - best_error) < 0.5 and
                                 orient_err < best_orient_err)
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
