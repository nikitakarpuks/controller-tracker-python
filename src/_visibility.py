import cv2
import numpy as np
from typing import Optional

from loguru import logger

from src.geometry import Box3D, Cylinder3D, ControllerGeometry, tangent_frame
from src._pnp import _to_rvec
from src.debug_config import is_deep


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
                  facing_threshold_deg: float = 86.0,
                  debug: bool = False) -> np.ndarray:
    """
    Boolean mask: True for each LED that is camera-facing and not occluded.

    R, tvec represent T_cam_ctrl (controller frame → camera frame): the native
    output of P3P/PnP.  positions and normals must be in controller frame.

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
    mask        = z_ok & (dot < -np.cos(np.radians(facing_threshold_deg)))

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
            if led_in_wall[i]:
                intervals = [EPS] + t_cands_sorted
            else:
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


# ── Cross-controller occlusion ───────────────────────────────────────────────

def _box_to_cam(box: Box3D, R: np.ndarray, t: np.ndarray) -> Box3D:
    """Return a new Box3D with center and axes transformed into camera frame."""
    axes_mat = np.asarray(box.axes, float) if box.axes is not None else np.eye(3)
    return Box3D(
        center=R @ np.asarray(box.center, float) + t,
        half_dims=box.half_dims,
        axes=R @ axes_mat,
        name=box.name,
    )


def _cylinder_to_cam(cy: Cylinder3D, R: np.ndarray, t: np.ndarray) -> Cylinder3D:
    """Return a new Cylinder3D with center and axis transformed into camera frame."""
    return Cylinder3D(
        center=R @ np.asarray(cy.center, float) + t,
        axis=R @ np.asarray(cy.axis, float),
        radius=cy.radius,
        half_length=cy.half_length,
        radius_v=cy.radius_v,
        angle=cy.angle,
        name=cy.name,
    )


def _occlusion_gate(
    tvec_A: np.ndarray,
    tvec_B: np.ndarray,
    br_A: float,
    br_B: float,
    focal_px: float,
    margin_px: float = 20.0,
) -> bool:
    """
    Fast 2-D projection gate.  Returns True (run the full ray test) when the
    two controllers' bounding circles could overlap on the image plane.

    Projects both controller origins with simplified pinhole and checks whether
    the pixel distance between them is less than the sum of their projected
    bounding radii plus margin_px slack.

    Returns False immediately if either controller is behind the camera.
    """
    if tvec_A[2] <= 0.0 or tvec_B[2] <= 0.0:
        return False
    px_A = tvec_A[:2] / tvec_A[2] * focal_px
    px_B = tvec_B[:2] / tvec_B[2] * focal_px
    dist = float(np.linalg.norm(px_A - px_B))
    r_proj_A = focal_px * br_A / float(tvec_A[2])
    r_proj_B = focal_px * br_B / float(tvec_B[2])
    return dist < r_proj_A + r_proj_B + margin_px


def _cross_occluded_mask(
    R_B: np.ndarray, tvec_B: np.ndarray,
    positions_B: np.ndarray,
    R_A: np.ndarray, tvec_A: np.ndarray,
    geom_A: ControllerGeometry,
    br_A: float,
    br_B: float,
    focal_px: float,
    gate_margin_px: float = 20.0,
    log_tag: str = "",
    vis_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Boolean mask (N,) — True for each LED of controller B whose camera→LED
    ray is blocked by controller A's handle body (boxes + cylinders).

    All geometry is expressed in camera frame: the camera sits at the origin,
    controller A's primitives are rotated/translated by T_cam_A (R_A, tvec_A),
    and controller B's LED positions are rotated/translated by T_cam_B (R_B, tvec_B).

    Returns all-False immediately when the 2-D projection gate determines that
    the two controllers cannot occlude each other from this camera.

    Parameters
    ----------
    R_B, tvec_B   : T_cam_B — controller B pose in camera frame (target).
    positions_B   : (N, 3) LED positions in controller-B frame.
    R_A, tvec_A   : T_cam_A — controller A pose in camera frame (occluder).
    geom_A        : ControllerGeometry of the occluder (boxes + cylinders).
    br_A, br_B    : bounding sphere radii of A and B for the 2-D gate.
    focal_px      : focal length in pixels (max of fx, fy).
    gate_margin_px: extra pixel slack on the gate threshold.
    """
    n = len(positions_B)
    if not _occlusion_gate(tvec_A, tvec_B, br_A, br_B, focal_px, gate_margin_px):
        return np.zeros(n, dtype=bool)

    # Transform A's primitives into camera frame.
    boxes_cam = [_box_to_cam(b,  R_A, tvec_A) for b  in geom_A.boxes]
    cyls_cam  = [_cylinder_to_cam(cy, R_A, tvec_A) for cy in geom_A.cylinders]

    # B's LEDs in camera frame; camera origin is [0, 0, 0].
    led_cam_B  = (R_B @ positions_B.T).T + tvec_B
    cam_origin = np.zeros(3, dtype=np.float64)

    blocked = np.zeros(n, dtype=bool)
    for box in boxes_cam:
        blocked |= _rays_blocked_by_box(cam_origin, led_cam_B, box)
    for cy in cyls_cam:
        blocked |= _rays_blocked_by_cylinder(cam_origin, led_cam_B, cy)

    if log_tag and is_deep():
        n_blocked = int(blocked.sum())
        if vis_mask is not None:
            newly_hidden = blocked & vis_mask
            blocked_ids = np.where(newly_hidden)[0].tolist()
        else:
            blocked_ids = np.where(blocked)[0].tolist()
        logger.debug(
            f"{log_tag} cross-controller occlusion: gate passed — "
            f"{n_blocked}/{n} LEDs masked  self-visible ids={blocked_ids}"
        )

    return blocked
