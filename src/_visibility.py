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


def _rays_box_entry_t(cam: np.ndarray, leds: np.ndarray, box) -> np.ndarray:
    """
    Vectorised OBB slab test.
    Returns float (N,) — entry t value; np.inf where ray is not blocked.
    """
    M  = np.asarray(box.axes, float) if box.axes is not None else np.eye(3)
    c  = np.asarray(box.center, float)
    h  = np.asarray(box.half_dims, float)

    D  = leds - cam
    oc = c - cam
    e  = M.T @ oc
    f  = (M.T @ D.T).T

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

    result = np.full(len(leds), np.inf)
    hit = t_min < t_max
    result[hit] = t_min[hit]
    return result


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


def _rays_cylinder_entry_t(cam: np.ndarray, leds: np.ndarray, cy) -> np.ndarray:
    """
    Vectorised ray vs elliptic cylinder (side wall + end caps).
    Returns float (N,) — minimum entry t; np.inf where ray is not blocked.
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

    D  = leds - cam
    oc = cam - c

    oc_ax = float(np.dot(oc, ax))
    oc_u  = float(np.dot(oc, u))
    oc_v  = float(np.dot(oc, v))
    D_ax  = D @ ax
    D_u   = D @ u
    D_v   = D @ v

    eps    = 1e-4
    result = np.full(len(leds), np.inf)

    # ── Side wall ────────────────────────────────────────────────────────────
    a     = (D_u / r_u) ** 2 + (D_v / r_v) ** 2
    b     = 2.0 * (oc_u * D_u / r_u ** 2 + oc_v * D_v / r_v ** 2)
    c_val = oc_u ** 2 / r_u ** 2 + oc_v ** 2 / r_v ** 2 - 1.0
    disc  = b ** 2 - 4.0 * a * c_val
    has_roots = disc >= 0.0
    if np.any(has_roots):
        sqrt_d  = np.sqrt(np.maximum(disc, 0.0))
        safe_2a = np.where(np.abs(a) > 1e-12, 2.0 * a, 1.0)
        for sign in (-1.0, 1.0):
            t = np.where(np.abs(a) > 1e-12, (-b + sign * sqrt_d) / safe_2a, 2.0)
            ok = has_roots & (t > eps) & (t < 1.0 - eps)
            if np.any(ok):
                z_t   = oc_ax + t * D_ax
                valid = ok & (np.abs(z_t) <= hl)
                result[valid] = np.minimum(result[valid], t[valid])

    # ── End caps ─────────────────────────────────────────────────────────────
    for z_cap in (-hl, hl):
        safe_ax = np.where(np.abs(D_ax) > 1e-12, D_ax, 1.0)
        t_cap   = np.where(np.abs(D_ax) > 1e-12, (z_cap - oc_ax) / safe_ax, 2.0)
        ok      = (t_cap > eps) & (t_cap < 1.0 - eps)
        if np.any(ok):
            xu    = oc_u + t_cap * D_u
            xv    = oc_v + t_cap * D_v
            valid = ok & ((xu / r_u) ** 2 + (xv / r_v) ** 2 <= 1.0)
            result[valid] = np.minimum(result[valid], t_cap[valid])

    return result


def _rays_boxes_entry_t(cam: np.ndarray, leds: np.ndarray, boxes: list) -> np.ndarray:
    """
    Batched OBB slab test over all boxes in one NumPy pass.
    Returns (N,) minimum entry t across all boxes; np.inf where no box is hit.
    """
    B = len(boxes)
    N = len(leds)
    centers   = np.array([np.asarray(b.center,   float) for b in boxes])          # (B, 3)
    half_dims = np.array([np.asarray(b.half_dims, float) for b in boxes])          # (B, 3)
    axes_mats = np.array([np.asarray(b.axes, float) if b.axes is not None
                           else np.eye(3) for b in boxes])                          # (B, 3, 3)

    D  = leds - cam                              # (N, 3)
    oc = centers - cam[None, :]                  # (B, 3)
    # Project offset and direction into each box's local frame
    e  = np.einsum('bi,bik->bk',  oc, axes_mats)    # (B, 3)
    f  = np.einsum('ni,bik->nbk', D,  axes_mats)    # (N, B, 3)

    e_bc = e[None, :, :]         # (1, B, 3)
    h_bc = half_dims[None, :, :] # (1, B, 3)

    nz     = np.abs(f) > 1e-12
    safe_f = np.where(nz, f, 1.0)
    t1 = np.where(nz, (e_bc - h_bc) / safe_f, -np.inf)   # (N, B, 3)
    t2 = np.where(nz, (e_bc + h_bc) / safe_f,  np.inf)   # (N, B, 3)
    swap = t1 > t2
    t1, t2 = np.where(swap, t2, t1), np.where(swap, t1, t2)
    # Parallel ray outside slab → no hit
    t2 = np.where(~nz & (np.abs(e_bc) > h_bc), -np.inf, t2)

    t_min_nb = np.maximum(1e-4,       t1.max(axis=2))   # (N, B)
    t_max_nb = np.minimum(1.0 - 1e-4, t2.min(axis=2))   # (N, B)
    hit_nb   = t_min_nb < t_max_nb
    entry_nb = np.where(hit_nb, t_min_nb, np.inf)
    return entry_nb.min(axis=1)                           # (N,)


def _rays_cylinders_entry_t(cam: np.ndarray, leds: np.ndarray, cylinders: list) -> np.ndarray:
    """
    Batched ray vs elliptic cylinders (side wall + end caps) in one NumPy pass.
    Returns (N,) minimum entry t across all cylinders; np.inf where no cylinder is hit.
    """
    B   = len(cylinders)
    N   = len(leds)
    eps = 1e-4

    ax_arr  = np.zeros((B, 3))
    u_arr   = np.zeros((B, 3))
    v_arr   = np.zeros((B, 3))
    r_u_arr = np.zeros(B)
    r_v_arr = np.zeros(B)
    ctr_arr = np.zeros((B, 3))
    hl_arr  = np.zeros(B)

    for b, cy in enumerate(cylinders):
        ax = np.asarray(cy.axis, float)
        ax /= np.linalg.norm(ax) + 1e-9
        u, v = tangent_frame(ax)
        if cy.angle:
            ca, sa = np.cos(cy.angle), np.sin(cy.angle)
            u, v = ca * u + sa * v, -sa * u + ca * v
        ax_arr[b]  = ax
        u_arr[b]   = u
        v_arr[b]   = v
        r_u_arr[b] = float(cy.radius)
        r_v_arr[b] = float(cy.radius_v) if cy.radius_v is not None else float(cy.radius)
        ctr_arr[b] = np.asarray(cy.center, float)
        hl_arr[b]  = float(cy.half_length)

    D   = leds - cam                      # (N, 3)
    oc  = cam[None, :] - ctr_arr          # (B, 3)

    oc_ax = (oc * ax_arr).sum(axis=1)    # (B,)
    oc_u  = (oc * u_arr).sum(axis=1)     # (B,)
    oc_v  = (oc * v_arr).sum(axis=1)     # (B,)

    D_ax = D @ ax_arr.T    # (N, B)
    D_u  = D @ u_arr.T     # (N, B)
    D_v  = D @ v_arr.T     # (N, B)

    # Side-wall quadratic coefficients; (N, B) except c_val which is (B,)
    a     = (D_u / r_u_arr) ** 2 + (D_v / r_v_arr) ** 2
    b_    = 2.0 * (oc_u * D_u / r_u_arr ** 2 + oc_v * D_v / r_v_arr ** 2)
    c_val = oc_u ** 2 / r_u_arr ** 2 + oc_v ** 2 / r_v_arr ** 2 - 1.0   # (B,)
    disc  = b_ ** 2 - 4.0 * a * c_val                                     # (N, B)

    result = np.full((N, B), np.inf)

    has_disc = disc >= 0.0
    if np.any(has_disc):
        sqrt_d  = np.sqrt(np.maximum(disc, 0.0))
        has_a   = np.abs(a) > 1e-12
        safe_2a = np.where(has_a, 2.0 * a, 1.0)
        for sign in (-1.0, 1.0):
            t = np.where(has_a, (-b_ + sign * sqrt_d) / safe_2a, 2.0)
            ok = has_disc & (t > eps) & (t < 1.0 - eps)
            if np.any(ok):
                z_t   = oc_ax + t * D_ax
                valid = ok & (np.abs(z_t) <= hl_arr)
                result = np.where(valid, np.minimum(result, t), result)

    for sign_cap in (-1.0, 1.0):
        z_cap    = sign_cap * hl_arr                           # (B,)
        safe_ax  = np.where(np.abs(D_ax) > 1e-12, D_ax, 1.0)
        t_cap    = np.where(np.abs(D_ax) > 1e-12,
                            (z_cap - oc_ax) / safe_ax, 2.0)   # (N, B)
        ok       = (t_cap > eps) & (t_cap < 1.0 - eps)
        if np.any(ok):
            xu    = oc_u + t_cap * D_u
            xv    = oc_v + t_cap * D_v
            valid = ok & ((xu / r_u_arr) ** 2 + (xv / r_v_arr) ** 2 <= 1.0)
            result = np.where(valid, np.minimum(result, t_cap), result)

    return result.min(axis=1)   # (N,)


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
                  cam_is_fisheye: bool = False,
                  facing_threshold_deg: float = 86.0,
                  occlusion_margin_m: float = 0.0,
                  debug: bool = False) -> np.ndarray:
    """
    Visibility score (float32, N,) for each LED. 1.0 = cleanly visible; 0.0 = not visible.
    Values in (0, 1) indicate borderline visibility near a geometric threshold.

    When occlusion_margin_m == 0 (default), occlusion checks return 0.0 or 1.0 only,
    matching the original binary behaviour. Callers that need a strict bool mask should
    use ``scores >= 1.0``; callers that want to include borderline LEDs (e.g. proximity
    search with an uncertain predicted pose) use a lower threshold such as 0.95.

    R, tvec represent T_cam_ctrl (controller frame → camera frame): the native
    output of P3P/PnP.  positions and normals must be in controller frame.

    Checks (in order):
      1. LED is in front of the camera (z > 1e-6).
      2. LED faces the camera (soft emission-cone score: threshold_deg / angle_deg,
         clamped to [0, 1]; 0 for angle >= 90°).
      2b. LED projects within image bounds (binary; distorted projection via cam_K/cam_dc).
      3. Inner LEDs blocked by the frustum truncated-cone wall (soft when
         occlusion_margin_m > 0: score from penetration depth at blocking midpoint).
      4. All LEDs blocked by any handle body primitive (soft when occlusion_margin_m > 0:
         score from ray-entry depth before the LED position).

    Parameters
    ----------
    is_inner          : optional subset mask — pass when positions/normals are a subset
                        of the full model (e.g. positions[il]).  If None, geom.is_inner
                        is used (assumes positions == full model).
    cam_K             : 3×3 camera intrinsic matrix; if None, the in-frame check is skipped.
    cam_dc            : distortion coefficients.
    cam_w, cam_h      : image dimensions in pixels.
    cam_rpmax         : max valid normalised radius; 0 disables the check.
    occlusion_margin_m: penetration depth (metres) that maps to score 0 in checks 3 & 4.
                        0 = hard binary (original behaviour).
    """
    N      = len(positions)
    scores = np.zeros(N, dtype=np.float32)

    # ── Check 1: positive depth ───────────────────────────────────────────────
    led_cam = (R @ positions.T).T + tvec

    # ── Check 2: emission-cone facing score ──────────────────────────────────
    view_dir    = led_cam / (np.linalg.norm(led_cam, axis=1, keepdims=True) + 1e-8)
    normals_cam = (R @ normals.T).T
    dot         = (normals_cam * view_dir).sum(axis=1)
    # angle_deg ∈ [0°, 180°]; 0° = normal points directly at camera
    angle_deg   = np.degrees(np.arccos(np.clip(-dot, -1.0, 1.0)))
    # score = 1 when angle < threshold; threshold/angle when threshold ≤ angle < 90°; 0 otherwise
    facing_score = np.where(
        (led_cam[:, 2] > 1e-6) & (angle_deg < 90.0),
        np.clip(facing_threshold_deg / np.maximum(angle_deg, 1e-6), 0.0, 1.0),
        0.0,
    )
    scores[:] = facing_score.astype(np.float32)

    # Working set: LEDs with any chance of being visible
    active = np.where(scores > 0.0)[0]
    if len(active) == 0:
        return scores

    if debug:
        borderline = active[scores[active] < 1.0]
        for led_id in borderline:
            print(f"  [facing]  LED {led_id:2d}: angle={angle_deg[led_id]:.1f}°  "
                  f"threshold={facing_threshold_deg:.1f}°  score={scores[led_id]:.3f}")

    # ── Check 2b: within image frame (binary) ────────────────────────────────
    if cam_K is not None and cam_w > 0 and cam_h > 0:
        dc = cam_dc if cam_dc is not None else np.zeros(4, dtype=np.float32)
        rv = _to_rvec(R)
        tv = tvec.astype(np.float32).reshape(3, 1)
        if cam_is_fisheye:
            pts, _ = cv2.fisheye.projectPoints(
                positions[active].astype(np.float64).reshape(-1, 1, 3),
                rv.astype(np.float64), tv.astype(np.float64),
                cam_K.astype(np.float64), dc,
            )
        else:
            pts, _ = cv2.projectPoints(
                positions[active].astype(np.float32).reshape(-1, 1, 3),
                rv, tv, cam_K, dc,
            )
        pts      = pts.reshape(-1, 2)
        in_frame = (pts[:, 0] >= 0) & (pts[:, 0] < cam_w) & \
                   (pts[:, 1] >= 0) & (pts[:, 1] < cam_h)
        if cam_rpmax > 0.0:
            z_a      = led_cam[active, 2]
            rp       = np.hypot(led_cam[active, 0] / z_a, led_cam[active, 1] / z_a)
            in_frame &= rp <= cam_rpmax
        scores[active[~in_frame]] = 0.0
        if debug:
            for k, led_id in enumerate(active):
                if not in_frame[k]:
                    print(f"  [frame]   LED {led_id:2d}: OUT OF FRAME  "
                          f"proj=({pts[k,0]:.1f}, {pts[k,1]:.1f})")
        active = active[in_frame]
        if len(active) == 0:
            return scores

    cam_world = -(R.T @ tvec)

    # ── Check 3: thick-frustum occlusion (inner LEDs only) ───────────────────
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
        P     = positions[inner_active]
        C_rel = cam_world - ring_centroid
        P_rel = P - ring_centroid

        C_ax  = float(C_rel @ ring_axis)
        P_ax  = (P_rel * ring_axis).sum(axis=1)
        C_rad = C_rel - C_ax * ring_axis
        P_rad = P_rel - np.outer(P_ax, ring_axis)

        D_ax  = P_ax - C_ax
        D_rad = P_rad - C_rad

        C_ax_rel  = C_ax - ring_center_ax
        A0_out    = R_fc       + slope * C_ax_rel
        A0_in     = R_fc_inner + slope * C_ax_rel
        Ad        = slope * D_ax
        C_rad_sq  = float(np.dot(C_rad, C_rad))
        CdotD_rad = (C_rad * D_rad).sum(axis=1)
        D_rad_sq  = (D_rad ** 2).sum(axis=1)

        a      = D_rad_sq - Ad ** 2
        b_out  = 2.0 * (CdotD_rad - A0_out * Ad)
        b_in   = 2.0 * (CdotD_rad - A0_in  * Ad)
        c_out  = C_rad_sq - A0_out ** 2
        c_in   = C_rad_sq - A0_in  ** 2

        disc_out = b_out ** 2 - 4.0 * a * c_out
        disc_in  = b_in  ** 2 - 4.0 * a * c_in

        safe_D_ax = np.where(np.abs(D_ax) > 1e-12, D_ax, np.inf)
        t_top_arr = (z_top - C_ax_rel) / safe_D_ax
        t_bot_arr = (z_bot - C_ax_rel) / safe_D_ax

        EPS     = 1e-4
        # frustum_scores[i]: score for inner_active[i]; starts 1.0, reduced when blocked
        frustum_scores = np.ones(len(inner_active), dtype=np.float32)

        z_led_arr   = C_ax_rel + D_ax
        r_led_arr   = np.linalg.norm(P_rad, axis=1)
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

        # ── Vectorized t-candidate array (N_inner, 6) ────────────────────────
        # Slots: [outer_root-, outer_root+, inner_root-, inner_root+, t_top, t_bot]
        NI     = len(inner_active)
        t_cands = np.full((NI, 6), np.inf)
        has_a   = np.abs(a) > 1e-12
        safe_2a = np.where(has_a, 2.0 * a, 1.0)

        for ci, (disc, b_, c_) in enumerate(
                [(disc_out, b_out, c_out), (disc_in, b_in, c_in)]):
            valid_disc = disc >= 0.0
            sqrt_d     = np.sqrt(np.maximum(disc, 0.0))
            has_b      = np.abs(b_) > 1e-12
            tc_lin     = np.where(~has_a & has_b, -float(c_) / b_, np.inf)
            tc_m       = np.where(has_a & valid_disc, (-b_ - sqrt_d) / safe_2a, np.inf)
            tc_p       = np.where(has_a & valid_disc, (-b_ + sqrt_d) / safe_2a, np.inf)
            t_cands[:, ci * 2]     = np.where(has_a, tc_m, tc_lin)
            t_cands[:, ci * 2 + 1] = np.where(has_a, tc_p, np.inf)

        t_cands[:, 4] = t_top_arr
        t_cands[:, 5] = t_bot_arr

        # Discard out-of-range candidates (same filter as original EPS guard)
        t_cands = np.where((t_cands > EPS) & (t_cands < 1.0 - EPS), t_cands, np.inf)
        t_sorted = np.sort(t_cands, axis=1)   # (NI, 6), inf sorts last

        # Build interval boundaries: left (NI,7), right (NI,7)
        # For led_in_wall rows suppress the trailing [last_t, 1-EPS] sentinel.
        sentinel = np.where(led_in_wall, np.inf, 1.0 - EPS)   # (NI,)
        left  = np.column_stack([np.full(NI, EPS), t_sorted])  # (NI, 7)
        right = np.column_stack([t_sorted, sentinel])           # (NI, 7)

        # Valid intervals: both endpoints finite and properly ordered
        valid_iv = np.isfinite(left) & np.isfinite(right) & (left < right)  # (NI, 7)

        # Midpoints — use 0.0 as safe fill to avoid inf arithmetic in norm
        t_mid_safe = np.where(valid_iv, 0.5 * (left + right), 0.0)   # (NI, 7)

        z_mid     = C_ax_rel + t_mid_safe * D_ax[:, None]             # (NI, 7)
        r_out_mid = R_fc       + slope * z_mid
        r_in_mid  = R_fc_inner + slope * z_mid
        # r_mid[n, j] = ||C_rad + t_mid[n,j] * D_rad[n]||
        mid_pts = C_rad[None, None, :] + t_mid_safe[:, :, None] * D_rad[:, None, :]
        r_mid   = np.linalg.norm(mid_pts, axis=2)                     # (NI, 7)

        in_wall = (
            valid_iv
            & (z_bot <= z_mid) & (z_mid <= z_top)
            & (r_in_mid <= r_mid) & (r_mid <= r_out_mid)
        )   # (NI, 7)

        has_block   = in_wall.any(axis=1)                   # (NI,)
        first_block = np.argmax(in_wall, axis=1)            # (NI,) — 0 when no hit (guarded)
        idx_n       = np.arange(NI)

        if occlusion_margin_m > 0.0:
            bz  = z_mid[idx_n, first_block]
            bro = r_out_mid[idx_n, first_block]
            bri = r_in_mid[idx_n, first_block]
            br  = r_mid[idx_n, first_block]
            radial_pen  = np.minimum(br - bri, bro - br)
            axial_pen   = np.minimum(bz - z_bot, z_top - bz)
            pen         = np.minimum(radial_pen, axial_pen)
            block_score = np.clip(1.0 - pen / occlusion_margin_m, 0.0, 1.0).astype(np.float32)
        else:
            block_score = np.zeros(NI, dtype=np.float32)

        frustum_scores = np.where(has_block, block_score, np.float32(1.0))

        if debug:
            for i in range(NI):
                tc_list = sorted(t_cands[i][np.isfinite(t_cands[i])].tolist())
                if frustum_scores[i] < 1.0:
                    j = int(first_block[i])
                    seg_len_cm = float(np.linalg.norm(P_rel[i] - C_rel)) * 100.0
                    dist_cm    = (1.0 - float(left[i, j])) * seg_len_cm
                    print(f"  [frustum] LED {inner_active[i]:2d}: BLOCKED by frustum wall"
                          f"  ({dist_cm:.1f}cm behind wall)  score={frustum_scores[i]:.3f}")
                elif tc_list:
                    print(f"  [frustum] LED {inner_active[i]:2d}: NOT blocked"
                          f"  t_cands={tc_list}"
                          f"  t_top={t_top_arr[i]:.4f}  t_bot={t_bot_arr[i]:.4f}"
                          f"  C_ax_rel={C_ax_rel:.4f}  D_ax={D_ax[i]:.4f}"
                          f"  z_range=[{z_bot:.4f},{z_top:.4f}]")
                else:
                    print(f"  [frustum] LED {inner_active[i]:2d}: NO t_cands"
                          f"  disc_out={disc_out[i]:+.5f}  disc_in={disc_in[i]:+.5f}"
                          f"  t_top={t_top_arr[i]:.4f}  t_bot={t_bot_arr[i]:.4f}")

        scores[inner_active] = np.minimum(scores[inner_active], frustum_scores)
        active = np.where(scores > 0.0)[0]
        if len(active) == 0:
            return scores

    # ── Check 4: handle body occlusion (boxes + cylinders, all LEDs) ─────────
    if len(active) > 0 and (geom.boxes or geom.cylinders):
        seg_len    = np.linalg.norm(positions[active] - cam_world, axis=1)
        body_score = np.ones(len(active), dtype=np.float32)

        for t in (
            _rays_boxes_entry_t(cam_world, positions[active], geom.boxes)
            if geom.boxes else None,
            _rays_cylinders_entry_t(cam_world, positions[active], geom.cylinders)
            if geom.cylinders else None,
        ):
            if t is None:
                continue
            hit = np.isfinite(t)
            if np.any(hit):
                if occlusion_margin_m > 0.0:
                    pen = (1.0 - t[hit]) * seg_len[hit]
                    body_score[hit] = np.minimum(
                        body_score[hit],
                        np.clip(1.0 - pen / occlusion_margin_m, 0.0, 1.0).astype(np.float32),
                    )
                else:
                    body_score[hit] = 0.0

        scores[active] = np.minimum(scores[active], body_score)

        if debug:
            for k, led_id in enumerate(active):
                s = body_score[k]
                if s >= 1.0:
                    continue
                led_pos = positions[led_id]
                seg_len_cm = float(np.linalg.norm(led_pos - cam_world)) * 100.0
                blockers = []
                for bi, box in enumerate(geom.boxes):
                    t_val = _box_entry_t(cam_world, led_pos, box)
                    if t_val is not None:
                        blockers.append(f"box_{bi}(dist={((1.0 - t_val) * seg_len_cm):.1f}cm)")
                for ci, cy in enumerate(geom.cylinders):
                    t_val = _cylinder_entry_t(cam_world, led_pos, cy)
                    if t_val is not None:
                        blockers.append(f"cylinder_{ci}(dist={((1.0 - t_val) * seg_len_cm):.1f}cm)")
                label = ", ".join(blockers) if blockers else "unknown"
                print(f"  [body]    LED {led_id:2d}: score={s:.3f}  blocked by {label}")

    return scores


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
    """
    n = len(positions_B)
    if not _occlusion_gate(tvec_A, tvec_B, br_A, br_B, focal_px, gate_margin_px):
        return np.zeros(n, dtype=bool)

    boxes_cam = [_box_to_cam(b,  R_A, tvec_A) for b  in geom_A.boxes]
    cyls_cam  = [_cylinder_to_cam(cy, R_A, tvec_A) for cy in geom_A.cylinders]

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
            vis_bool     = np.asarray(vis_mask, dtype=bool)
            newly_hidden = blocked & vis_bool
            blocked_ids  = np.where(newly_hidden)[0].tolist()
        else:
            blocked_ids = np.where(blocked)[0].tolist()
        logger.debug(
            f"{log_tag} cross-controller occlusion: gate passed — "
            f"{n_blocked}/{n} LEDs masked  self-visible ids={blocked_ids}"
        )

    return blocked
