"""Per-camera memory of "known static ceiling lamp" 3D REGIONS (axis-aligned
rectangles, not points), hard-masked out of every future frame via mocap
reprojection -- see src/blob_detector.py's `region_exclude_blobs` H1-style
exclusion check.

Deliberately simpler and coarser than the earlier point-level
`LampAnchorMemory` (removed): that system reprojected individual recognized
points into detect_lamp_blobs' own dim-context pool, hoping the line-finder
(spacing/residual/min_points) re-recognized the row every single frame from
reprojected points whose exact pixel position carries all of the mocap+height
error. That turned out fragile in practice (an entire row could collapse into
one wrong-position blurred anchor, or a reprojected point could land almost
exactly on a still-visible real point and void a whole otherwise-valid line via
the spacing_min_px check). This module instead remembers a lamp's rough
bounding rectangle, ray-casts it to 3D at the assumed ceiling height, and
reprojects it every frame as a hard geometric mask -- no line fitting, no
per-point spacing/residual sensitivity, "rough is fine" by design.

Room frame is Y-up (src/imu_data.py's own empirically-measured
MOCAP_ROOM_G_WORLD = [0, -9.81, 0], not Z) -- every region's 4 corners sit on
the plane Y = ceiling_height_m by construction.

Regions are axis-aligned in room-frame XZ (kept deliberately simple and
stable across merges -- a rotated-rectangle representation was tried and
reverted: fitting cv2.minAreaRect per observation and unioning via convex
hull worked for a single frame, but different observations of the same
static fixture can ray-cast to slightly different apparent orientations
(measurement noise), and merging rotated rects with mismatched orientations
produced a WORSE, more wasteful box than a plain axis-aligned union -- see
project_lamp_anchor_memory.md for the full story). Tightness instead comes
from what gets ray-cast: real recognized POINTS (a thin line of elements,
plus each element's own real visible extent -- see
src/blob_detector.py's `dim_extent_bounds`), not a bounding rectangle's 4
corners -- ray-casting a fat synthetic rectangle and axis-aligning THAT
wastes real area whenever the row is diagonal in room-space (confirmed on a
real recording, cam0, right: a tight 152.75x35.25px row's bbox-corners
approach inflated to a 202x103px room-space box), whereas the real points
themselves trace only the row's true (thin) footprint and stay tight
regardless of orientation.

LIFECYCLE (rewritten 2026-09-13 -- see project_lamp_anchor_memory.md for the
full history of what this replaced): a region is a LIVE, continuously
re-measured thing, not a permanent, ever-growing memory.
  - CREATION: update() is called only when detect_lamp_blobs' own strict
    line-finder (spacing/residual/min_points) recognizes+removes a full
    candidate row this frame -- unchanged, still the only way a brand NEW
    region gets seeded.
  - SUSTAINING: every lamp-region-active frame, sustain_and_expire() checks
    each ALREADY-HELD region against THIS frame's own raw candidate pool
    (both "kept"/bright blobs and "dim" ones -- the same combined pool
    detect_lamp_blobs itself sees, not a fresh strict re-recognition): if
    enough points -- kept AND dim COMBINED, see sustain_min_points -- fall
    inside the region's current reprojected footprint, the region is refit
    TIGHTLY to those points, replacing its bounds outright, not unioning
    with the old ones, since this is a fresh live measurement each time, not
    a permanent accumulation. A region that isn't sustained this frame (too
    few points inside it, OR it's out of view entirely) accumulates a
    "no-support" streak; once that streak reaches
    expire_after_no_support_frames, the region is DELETED outright. This
    deliberately uses a MUCH lower bar to keep a region alive than the bar
    that created it in the first place (detect_lamp_blobs' own min_points,
    e.g. 6): once a fixture is already known, a few points is enough proof
    it's still there -- we don't need to re-clear the strict from-scratch
    recognition bar every single frame.

    An earlier version of this check counted only KEPT (bright) points
    toward the threshold, on the theory that dim points alone are "too
    easily spurious to gate survival on," mirroring detect_lamp_blobs' own
    "dim points are context, never independently sufficient" philosophy.
    Confirmed WRONG via real e2e tracing (cam0, left lamp): a frame with 20
    real dim points and only 2 kept ones still counted as "no support"
    (kept=2 < sustain_min_points's old kept-only bar of 3), expiring a
    region that plainly had overwhelming real evidence, then re-creating it
    from scratch a few frames later, repeating indefinitely. The "dim alone
    is spurious" concern doesn't actually transfer here the way it does for
    detect_lamp_blobs' own from-scratch line-finding: those dim points
    still have to fall inside an ALREADY-established region's own tight
    footprint to count at all (unlike a fresh recognition, which has no
    such anchor) -- 20 unrelated noise points landing inside one small,
    specific, already-known box repeatedly is not a realistic false-positive
    mode. Counting kept+dim together fixes this while keeping the same
    numeric bar (sustain_min_points defaults to 3, same default as the old
    kept-only sustain_min_kept) -- still enough to reject a truly empty
    frame, no longer blind to a fixture that's just mostly dim.
  - No more permanent/frozen memory: an earlier version of this module
    added `freeze_after_hits` (stop a region's bounds from growing once
    "established", to resist single-frame measurement noise) and later
    `regrowth_confirmations` (let a frozen region's bounds move again once
    independently corroborated several separate times, since an early
    freeze from a low-parallax view could just be WRONG). Both are GONE --
    continuous per-frame refitting supersedes them entirely: there's
    nothing to freeze (bounds are re-measured, not accumulated) and nothing
    that needs corroboration to correct (a wrong measurement is simply
    replaced by the next frame's fresh one).

    CORRECTION 2026-09-14 (independent code review + real ground-truth
    evaluation): the claim that once removed `absolute_max_region_size_m`
    was safe to drop because "nothing ever accumulates without bound" was
    WRONG in practice -- quad_room WAS still an ever-growing all-time union
    (of "current bounds" with each sustained frame's own fresh measurement),
    and a single merge bypassing max_region_size_m via size_cap_bypass_iou
    could push it past the cap PERMANENTLY, since nothing in
    sustain_and_expire's own size check ever shrinks quad_room back down --
    it only ever refuses to grow FURTHER once already too big. Confirmed on
    real data (seq1/walk_hard, cam3): a region with hits=80 reached 4.97m x
    4.63m (~23 square meters), visibly distorting into a skewed shape when
    reprojected through this camera's wide-FOV fisheye lens and masking far
    more background as "lamp" than any real fixture. Fixed properly this
    time via LampRegion.recent_quads: quad_room is now always recomputed
    from a BOUNDED trailing window of individual observations (see
    recent_quad_window), so a quad that made a region too big eventually
    ages out as newer, tighter observations replace it in the window --
    self-correcting, not just growth-capped.

Region bookkeeping (merging/eviction) is still NMS-like: every update()
call appends the new recognition as its own region, then _dedupe_and_cap()
repeatedly merges any two regions that look like the same physical fixture
(high XZ IoU -- clearly the same, heavily-overlapping recognition -- OR
close centers -- a real fixture recognized as two disconnected pieces, e.g.
split by a physical gap the line-finder couldn't bridge, see
project_lamp_row_leak_frames1_13 memory -- OR one almost entirely contained
in the other) into their union, until no qualifying pair remains. This is a
real, separate pass over ALL current regions (not just "new box vs. its
nearest neighbor") specifically because two regions can end up overlapping/
duplicating each other over time even without ever being directly compared
before.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.camera import Camera
from src.static_light_geometry import pixel_to_room_ray, room_point_to_pixel
from src.transformations import Transform

_EPS = 1e-6


@dataclass
class LampRegion:
    quad_room: np.ndarray   # (4,3) room-frame meters; quad_room[:,1] == ceiling_height_m by construction
    hits: int = 1           # refresh counter -- how many times this region has been created/merged/sustained
    # CONSECUTIVE count of frames where this region failed to gather enough
    # support (see LampRegionMemory.sustain_and_expire) -- reset to 0 on any
    # single sustained frame in between, region deleted once this reaches
    # expire_after_no_support_frames. (An earlier version of this comment,
    # and of config.yml's own expire_after_no_support_frames comment, claimed
    # this was "not necessarily consecutive" -- that was simply wrong: every
    # sustained branch in sustain_and_expire() resets it to 0, so a region
    # that sustains even once every few frames never approaches expiry. Fixed
    # 2026-09-14 as a documentation-only correction; the lenient consecutive-
    # reset behavior itself is correct and intentional -- see
    # sustain_and_expire's own docstring for why.)
    no_support_streak: int = 0
    # Bounded trailing window of the individual per-observation quads that
    # currently back this region's quad_room (most-recent-last) -- see
    # LampRegionMemory._recent_quad_window. ADDED 2026-09-14 (independent
    # code review + real ground-truth evaluation): quad_room used to be an
    # ever-growing union with NO way back down (sustain_and_expire's own size
    # check only ever blocked FURTHER growth once too big, it never shrank
    # anything back), and a single merge bypassing max_region_size_m (see
    # LampRegionMemory._size_cap_bypass_iou) could push a region's size up
    # permanently. Confirmed on real data (seq1/walk_hard, cam3): a region
    # with hits=80 had grown to a 4.97m x 4.63m (~23 square meter) footprint
    # -- physically absurd for a lamp, and the direct cause of two real
    # complaints: reprojecting a room-space box that size through this
    # camera's wide-FOV fisheye lens puts its 4 corners at wildly different
    # viewing angles (23 deg to 60 deg in the confirmed example), which
    # straight pixel-space edges render as a grotesquely skewed/twisted
    # shape -- NOT a kb4 projection bug, a symptom of the box itself being
    # absurd -- and the huge extra area masked far more background/noise as
    # "lamp" than the real fixture ever covered. Now quad_room is always
    # RECOMPUTED as the union of only this bounded window's own quads (see
    # sustain_and_expire and LampRegionMemory._dedupe_and_cap), so a quad
    # that made the region too wide eventually ages out as newer, presumably
    # tighter, observations push it out of the window -- self-correcting,
    # not just growth-capped.
    recent_quads: List[np.ndarray] = field(default_factory=list)


def _xz_bounds(quad_room: np.ndarray) -> np.ndarray:
    """(4,) [xmin, xmax, zmin, zmax] axis-aligned bounds of a quad's XZ footprint."""
    xz = quad_room[:, [0, 2]]
    xy_min, xy_max = xz.min(axis=0), xz.max(axis=0)
    return np.array([xy_min[0], xy_max[0], xy_min[1], xy_max[1]])


def _iou_xz(quad_a: np.ndarray, quad_b: np.ndarray) -> float:
    """2D IoU of two quads' axis-aligned XZ bounding boxes."""
    ax0, ax1, az0, az1 = _xz_bounds(quad_a)
    bx0, bx1, bz0, bz1 = _xz_bounds(quad_b)
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    iz0, iz1 = max(az0, bz0), min(az1, bz1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iz1 - iz0)
    if inter <= 0.0:
        return 0.0
    area_a = (ax1 - ax0) * (az1 - az0)
    area_b = (bx1 - bx0) * (bz1 - bz0)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _center_xz(quad_room: np.ndarray) -> np.ndarray:
    return quad_room[:, [0, 2]].mean(axis=0)


def _containment_xz(quad_a: np.ndarray, quad_b: np.ndarray) -> float:
    """Fraction of the SMALLER of the two quads' XZ area that overlaps the
    other -- 1.0 when the smaller one sits entirely inside the larger one.
    IoU alone misses exactly this case: a small box fully contained in a
    much bigger one still scores a LOW IoU (the union is dominated by the
    big box's own area), even at 100% containment -- found on real usage
    (cam2, right): a hits=6 region's box was entirely inside a hits=32
    region's box (containment=1.0) yet their IoU was only ~0.19, just under
    merge_iou_threshold, and their centers were 1.2m apart, past
    merge_center_distance_m -- neither existing trigger caught it, so two
    regions that were clearly the same covered area never merged."""
    ax0, ax1, az0, az1 = _xz_bounds(quad_a)
    bx0, bx1, bz0, bz1 = _xz_bounds(quad_b)
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    iz0, iz1 = max(az0, bz0), min(az1, bz1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iz1 - iz0)
    if inter <= 0.0:
        return 0.0
    area_a = (ax1 - ax0) * (az1 - az0)
    area_b = (bx1 - bx0) * (bz1 - bz0)
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0.0 else 0.0


def _union_many(quads: List[np.ndarray], ceiling_height_m: float) -> np.ndarray:
    """Axis-aligned XZ union of a whole list of quads at once -- min/max,
    never an average (a union must be a safe superset of every input, not
    drift toward a blended position) -- for the trailing-window recompute
    (see LampRegion.recent_quads)."""
    xz_all = np.concatenate([q[:, [0, 2]] for q in quads], axis=0)
    x0, z0 = xz_all.min(axis=0)
    x1, z1 = xz_all.max(axis=0)
    return np.array([
        [x0, ceiling_height_m, z0],
        [x1, ceiling_height_m, z0],
        [x1, ceiling_height_m, z1],
        [x0, ceiling_height_m, z1],
    ], dtype=np.float64)


def _xz_size(quad_room: np.ndarray) -> np.ndarray:
    x0, x1, z0, z1 = _xz_bounds(quad_room)
    return np.array([x1 - x0, z1 - z0])


def _largest_cluster_xz(xz: np.ndarray, max_link_m: float) -> np.ndarray:
    """Keep only the SINGLE LARGEST connected cluster of these room-space XZ
    points (an edge links two points within max_link_m of each other,
    transitively). See _ray_cast_points_to_quad's own comment for why this
    runs HERE (room-space meters, post ray-cast) and not in pixel space:
    this camera's fisheye projection is highly anisotropic at oblique
    viewing angles (the same real-world gap spans far more pixels near the
    image edge than near center), so a fixed pixel-space distance threshold
    is ineffective exactly where it matters most -- confirmed directly on
    real data (a first attempt at this filter, in pixel space inside
    src/blob_detector.py, made no measurable difference to a real region
    sitting at 24-58 degree viewing angles). A metric threshold means the
    same real-world gap is judged the same way everywhere in the image."""
    n = len(xz)
    if n < 2:
        return xz
    dists = np.linalg.norm(xz[:, None, :] - xz[None, :, :], axis=2)
    adjacency = dists <= max_link_m
    np.fill_diagonal(adjacency, False)
    unvisited = set(range(n))
    best_component: list = []
    while unvisited:
        start = next(iter(unvisited))
        stack, component = [start], []
        unvisited.discard(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in np.nonzero(adjacency[node])[0]:
                neighbor = int(neighbor)
                if neighbor in unvisited:
                    unvisited.discard(neighbor)
                    stack.append(neighbor)
        if len(component) > len(best_component):
            best_component = component
    return xz[best_component]


def _ray_cast_points_to_quad(points_px: np.ndarray, camera: Camera, T_room_cam: Transform,
                              ceiling_height_m: float, bbox_pad_m: float,
                              cluster_max_m: Optional[float] = None) -> Optional[np.ndarray]:
    """Shared by update() (creation) and sustain_and_expire() (per-frame
    refit): ray-cast pixel-space points through the known ceiling plane,
    take the axis-aligned XZ bounds of the valid ones, pad by bbox_pad_m.
    Returns None if there are no points, or none of their rays can reach
    the ceiling plane in front of the camera (near-zero/negative upward ray
    component, or an intersection behind the camera) -- never fabricates a
    quad from an unusable ray.

    cluster_max_m (ADDED 2026-09-14, independent code review + real
    ground-truth evaluation + classical-ML feature analysis, seq1/walk_hard):
    when given, keeps only the single largest connected cluster of the
    ray-cast room-space points (_largest_cluster_xz) before taking bounds --
    a genuinely different, spatially-separate group of points (a second real
    structure, or a spatially-distinct patch of correlated noise) that
    happens to also satisfy the caller's own point-count/containment test
    gets discarded wholesale instead of blowing out the box's bounds.
    Confirmed on real data: without this, a region with hits=80 had grown to
    a physically-absurd 4.97m x 4.63m (~23 square meter) footprint -- a
    self-reinforcing feedback loop (a bigger region catches more of whatever
    else is nearby, which then keeps it big). None (the default) skips this
    entirely -- used by tests that construct small synthetic point sets with
    no intent to exercise this feature."""
    points_px = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    if len(points_px) == 0:
        return None
    origin, dirs = pixel_to_room_ray(camera, T_room_cam, points_px)
    valid = dirs[:, 1] > _EPS
    if not valid.any():
        return None
    t = np.full(len(dirs), np.nan)
    t[valid] = (ceiling_height_m - origin[1]) / dirs[valid, 1]
    valid &= t > 0
    if not valid.any():
        return None
    room_pts = (origin + t[:, None] * dirs)[valid]
    xz = room_pts[:, [0, 2]]
    if cluster_max_m is not None:
        xz = _largest_cluster_xz(xz, cluster_max_m)
    x0, z0 = xz.min(axis=0)
    x1, z1 = xz.max(axis=0)
    x0 -= bbox_pad_m
    z0 -= bbox_pad_m
    x1 += bbox_pad_m
    z1 += bbox_pad_m
    return np.array([
        [x0, ceiling_height_m, z0],
        [x1, ceiling_height_m, z0],
        [x1, ceiling_height_m, z1],
        [x0, ceiling_height_m, z1],
    ], dtype=np.float64)


class LampRegionMemory:
    """One instance per camera, same ownership model as BlobDetector._memory."""

    def __init__(self, cfg: dict):
        self._ceiling_height_m = float(cfg.get("ceiling_height_m", 4.0))
        # Padding is applied AFTER ray-casting, directly in room-frame meters
        # -- NOT in pixel space before ray-casting (an earlier design).
        # Found on live usage (cam2, right): the same fixed pixel padding
        # translated to wildly different REAL margins depending on viewing
        # angle -- this camera's pixel-to-ceiling-plane projection is highly
        # anisotropic (a near-grazing ray amplifies a few pixels into a huge
        # real distance, a near-vertical ray barely amplifies it at all), so
        # "same pixel padding" != "same real padding". A fixed metric padding
        # here guarantees every lamp gets the same real-world margin
        # regardless of where in the image (and at what angle) it was seen.
        self._bbox_pad_m = float(cfg.get("bbox_pad_m", 0.05))
        # Three independent triggers for "these two regions are the same
        # physical fixture, merge them" -- any one is sufficient:
        #  - IoU: for heavily/partially overlapping duplicates (found on live
        #    usage persisting at ~0.9 IoU under an earlier "new box vs.
        #    nearest only" design -- deliberately a fairly LOW default so a
        #    real but only partial overlap (e.g. IoU ~0.3-0.5) still merges,
        #    not just near-total duplicates).
        #  - center distance: for a fixture recognized as two DISCONNECTED
        #    pieces (IoU == 0, no box overlap at all -- e.g. split by a real
        #    physical gap the line-finder couldn't bridge, see
        #    project_lamp_row_leak_frames1_13 memory) but close enough in
        #    room-space to plausibly be the same fixture.
        #  - containment: for a SMALL region sitting almost entirely inside a
        #    much LARGER one -- IoU alone misses this (dominated by the big
        #    box's own area, so even 100% containment can score a low IoU),
        #    and center distance misses it too when the boxes are very
        #    different sizes (a big box's own center can be far from a small
        #    box fully inside one of its corners). Found on live usage (cam2,
        #    right): a small region's box was 100% inside a much larger
        #    region's box, yet IoU was ~0.19 (just under merge_iou_threshold)
        #    and their centers were 1.2m apart (past merge_center_distance_m)
        #    -- neither trigger caught it, so two regions that were clearly
        #    the same covered area never merged. See _containment_xz.
        self._merge_iou_threshold = float(cfg.get("merge_iou_threshold", 0.2))
        self._merge_center_distance_m = float(cfg.get("merge_center_distance_m", 0.5))
        self._merge_containment_threshold = float(cfg.get("merge_containment_threshold", 0.8))
        self._max_regions = int(cfg.get("max_regions_per_camera", 10))
        # Backstop against a truly implausible single recognition (e.g. a
        # wide wall reflection/glare that happened to satisfy
        # detect_lamp_blobs' own spacing/residual checks) -- used both at
        # creation (update()) and at each per-frame refit
        # (sustain_and_expire()). Deliberately generous: real recognized
        # footprints on this project's own recordings range from ~0.8m (a
        # single clean row, cam3) up past 1.4m (cam2, right) even for a
        # SINGLE legitimate recognition, so a tight cap here mostly just
        # blocks real fixtures (see size_cap_bypass_iou below).
        self._max_region_size_m = float(cfg.get("max_region_size_m", 3.0))
        # A merge triggered by HIGH IoU (or containment) is exempt from
        # max_region_size_m -- two regions overlapping this much are
        # unambiguously the same physical object no matter how "big" that
        # object is. Found on live usage (cam2, right): several regions of
        # the SAME real fixture, each already sitting close to
        # max_region_size_m on their own, could never merge with each other
        # at all -- every attempt got rejected by the size guard,
        # permanently fragmenting one fixture into several near-duplicate,
        # never-consolidating boxes. Lower-confidence merges (IoU below this
        # but still >= merge_iou_threshold, or triggered only by
        # merge_center_distance_m) still respect max_region_size_m as a
        # sanity check, since those are exactly the ones NOT already
        # obviously the same object. (No separate "absolute, never-bypassed"
        # ceiling anymore -- that existed only to bound a CHAIN of
        # ever-growing merges under the old permanent-union/freeze model,
        # which no longer exists now that regions are continuously refit,
        # not permanently accumulated.)
        self._size_cap_bypass_iou = float(cfg.get("size_cap_bypass_iou", 0.5))
        # How many individual per-observation quads (see LampRegion.recent_quads)
        # a region's quad_room is recomputed from -- bounds how long a single
        # oversized observation (a bad merge, a noisy frame) can keep
        # inflating quad_room before it ages out. See LampRegion.recent_quads'
        # own comment for the real 4.97m x 4.63m runaway this fixes. Picked
        # generously (not tiny) on purpose: too small a window would re-shrink
        # the box on every routine partial-visibility frame (the exact
        # regression "union, not replace" was originally fixed for -- see
        # sustain_and_expire's own docstring), defeating that fix. 20 is
        # roughly a middle ground between "recovers within a real recording's
        # own timescale" (confirmed via re-evaluation against ground truth)
        # and "tolerates several consecutive partial-visibility frames
        # without prematurely shrinking a still-real fixture."
        self._recent_quad_window = int(cfg.get("recent_quad_window", 20))
        # A region only actually excludes anything once recognized/sustained
        # this many times (hits >= this) -- see reprojected_contours(). A
        # single-frame recognition (line-finder false positive off a
        # reflection, or a genuine lamp seen only once so far) can seed a
        # region immediately (update() below), but doesn't get to hard-mask
        # anything until reinforced -- a real static lamp gets re-recognized
        # or sustained across many consecutive frames essentially for free,
        # a one-off spurious recognition doesn't.
        # hits starts at 1 (LampRegion's own dataclass default), so a value
        # of 1 here means "< 1" is never true -- i.e. every region excludes
        # from the very first frame it exists, with zero reinforcement,
        # silently defeating the entire point of this parameter (see the
        # comment above). This exact misconfiguration shipped in config.yml
        # for a time (min_hits_to_exclude: 1) despite this class's own test
        # suite validating 3 as the intended behavior. NOT enforced here via
        # assertion (tried, then reverted 2026-09-14): this class is also
        # constructed directly by tests/test_lamp_region_memory.py with
        # min_hits_to_exclude=1 as a deliberate, documented convenience for
        # tests unrelated to this specific gate (creation/merge/sustain
        # tests that don't want hits-based exclusion timing to interfere) --
        # blanket-rejecting it here broke 27 legitimate unit tests. The real
        # production value is instead validated directly against the actual
        # config.yml by tests/test_lamp_mask_config_sanity.py.
        self._min_hits_to_exclude = int(cfg.get("min_hits_to_exclude", 3))
        # Per-frame liveness bar for an ALREADY-HELD region -- deliberately
        # much lower than detect_lamp_blobs' own from-scratch recognition
        # bar (min_points, e.g. 6): once a fixture is already known, seeing
        # a handful of its own elements again is enough proof it's still
        # there; we don't need the full strict spacing/residual line-fit
        # every single frame just to keep tracking something we already
        # trust exists. Counts KEPT (bright) and DIM points TOGETHER --
        # an earlier version counted only kept ones, on the theory that dim
        # points alone are too easily spurious to gate survival on (mirroring
        # detect_lamp_blobs' own "dim points are context, never
        # independently sufficient" philosophy for FRESH recognition). That
        # doesn't transfer to sustaining an already-established region:
        # confirmed via real e2e tracing (cam0, left lamp) that a frame with
        # 20 real dim points and only 2 kept ones was being counted as "no
        # support" under the kept-only rule, expiring a region with
        # overwhelming real evidence and forcing it to be re-created from
        # scratch repeatedly. Dim points landing inside an ALREADY-tight,
        # already-established region's own footprint aren't the same
        # false-positive risk as dim noise trying to seed a recognition from
        # nothing.
        self._sustain_min_points = int(cfg.get("sustain_min_points", 3))
        # See _ray_cast_points_to_quad's own cluster_max_m docstring for the
        # full story: without this, a spatially-separate group of points
        # (falling inside an already-big region's contour, satisfying
        # sustain_min_points on raw count alone) could blow the region's own
        # bounds out to cover BOTH clusters. A real row's own adjacent
        # elements are expected to sit within a small fraction of a meter of
        # each other (this project's own real recordings show roughly
        # 0.1-0.3m bead spacing at typical viewing distance) -- 0.3m is
        # generous enough that single-linkage clustering still keeps one
        # genuinely long, continuous fixture together (chained via its own
        # adjacent-element gaps, not the row's total span), while rejecting
        # a truly separate structure or noise patch sitting elsewhere within
        # the same oversized box.
        self._sustain_cluster_max_m = float(cfg.get("sustain_cluster_max_m", 0.3))
        # How many frames of "not enough support" (see sustain_min_points) a
        # region tolerates, not necessarily consecutive, before it is
        # DELETED outright -- no more permanent mocap-only memory once
        # real evidence stops backing a region up. Out of view counts as
        # zero support too (there's nothing to check), so a region that
        # leaves the frame and never comes back eventually expires instead
        # of being remembered forever. A short grace period (not 1) avoids
        # a single flickery/occluded frame deleting an otherwise-good,
        # well-established region.
        self._expire_after_no_support_frames = int(cfg.get("expire_after_no_support_frames", 3))
        self._regions: List[LampRegion] = []
        # Recording-clock (frame_ts_ns) timestamp of the last WARM-frame
        # "shadow" pass1 refresh (see BlobDetector.detect()'s
        # _run_warm_lamp_shadow_pass) -- None until the first one. Lives here
        # (not on BlobDetector) specifically because LampRegionMemory is
        # already what round-trips across the process boundary under
        # matching.parallel_blob_detection_enabled: true (src/parallel_search.py
        # explicitly extracts/reinjects bd._lamp_region_memory per call, but
        # does NOT round-trip arbitrary other BlobDetector attributes) -- so
        # any new per-camera state that must survive across parallel-pool
        # calls belongs on this object, not the detector.
        self._last_shadow_refresh_ts_ns: Optional[int] = None

    def due_for_shadow_refresh(self, frame_ts_ns: int, interval_s: float) -> bool:
        """True (and internally records this call as the new baseline) once
        at least interval_s of RECORDING time (frame_ts_ns, not wall-clock)
        has passed since the last time this returned True. Uses the
        recording's own clock, not time.time(), because this whole pipeline
        processes prerecorded sequences offline -- wall-clock processing
        speed has nothing to do with how much time actually elapsed in the
        recording between frames, which is what the throttle is meant to
        bound. NOT monotonic-clock-based for the same reason. Always True on
        the first call (nothing to compare against yet -- getting a region's
        lifecycle started at all takes priority over the throttle)."""
        if (self._last_shadow_refresh_ts_ns is None
                or abs(frame_ts_ns - self._last_shadow_refresh_ts_ns) >= interval_s * 1e9):
            self._last_shadow_refresh_ts_ns = frame_ts_ns
            return True
        return False

    def update(self, points_px: np.ndarray, camera: Camera, T_room_cam: Transform) -> None:
        """points_px: (N,2) pixel-space points of a lamp fixture actually
        recognized+removed this frame (by detect_lamp_blobs's own strict
        line-finder) -- the row's own REAL points (including each element's
        own real visible extent, see src/blob_detector.py's
        dim_extent_bounds), not a synthetic bounding rectangle's 4 corners.
        This is the only path that creates a brand NEW region; an existing
        one is grown/refit instead via sustain_and_expire()."""
        quad = _ray_cast_points_to_quad(points_px, camera, T_room_cam,
                                         self._ceiling_height_m, self._bbox_pad_m)
        if quad is None:
            return
        # Reject outright a single recognition whose own footprint is already
        # bigger than a real single fixture could plausibly be -- e.g. a
        # large/noisy structure (wall reflection, glare) that happened to
        # satisfy detect_lamp_blobs' own spacing/residual checks. Never even
        # becomes a candidate region, regardless of merging.
        if (_xz_size(quad) > self._max_region_size_m).any():
            return
        self._regions.append(LampRegion(quad_room=quad, recent_quads=[quad]))
        self._dedupe_and_cap()

    def _should_merge(self, a: LampRegion, b: LampRegion) -> bool:
        return (_iou_xz(a.quad_room, b.quad_room) >= self._merge_iou_threshold
                or np.linalg.norm(_center_xz(a.quad_room) - _center_xz(b.quad_room)) <= self._merge_center_distance_m
                or _containment_xz(a.quad_room, b.quad_room) >= self._merge_containment_threshold)

    def _dedupe_and_cap(self) -> None:
        """NMS-like cleanup, run after every update() and every
        sustain_and_expire(): repeatedly find the pair of regions most
        likely to be the same physical fixture (highest IoU-or-containment
        among qualifying pairs) and merge them into their union (hits
        summed). A pair whose union would exceed max_region_size_m is
        normally left as two separate regions rather than merged -- EXCEPT
        when their IoU already reaches size_cap_bypass_iou, OR one is
        almost entirely contained in the other (>= merge_containment_threshold),
        in which case they're unambiguously the same covered area and get
        merged regardless. Finishes by evicting down to max_regions_per_camera
        (lowest-hits first)."""
        skip: set = set()
        while len(self._regions) > 1:
            best_score, best_pair = -1.0, None
            for i in range(len(self._regions)):
                for j in range(i + 1, len(self._regions)):
                    if (i, j) in skip:
                        continue
                    a, b = self._regions[i], self._regions[j]
                    if not self._should_merge(a, b):
                        continue
                    score = max(_iou_xz(a.quad_room, b.quad_room), _containment_xz(a.quad_room, b.quad_room))
                    if score > best_score:
                        best_score, best_pair = score, (i, j)
            if best_pair is None:
                break
            i, j = best_pair
            a, b = self._regions[i], self._regions[j]
            iou = _iou_xz(a.quad_room, b.quad_room)
            containment = _containment_xz(a.quad_room, b.quad_room)
            # recent_quads: concatenate both inputs' own trailing windows and
            # re-truncate to the same bound (most-recent-last approximation --
            # no per-entry timestamp is tracked, so true chronological
            # interleaving across two independently-aged regions isn't
            # attempted; keeping the tail of the concatenation is enough to
            # guarantee the merged region's size still self-corrects over the
            # next _recent_quad_window sustained frames, which is the
            # property that actually matters -- see LampRegion.recent_quads).
            # `or [x.quad_room]`: a LampRegion constructed directly with no
            # recent_quads (e.g. a test fixture, or any future caller) falls
            # back to treating its current quad_room as its one known
            # observation, rather than crashing _union_many on an empty list.
            a_recent = a.recent_quads or [a.quad_room]
            b_recent = b.recent_quads or [b.quad_room]
            merged_recent = (a_recent + b_recent)[-self._recent_quad_window:]
            union_quad = _union_many(merged_recent, self._ceiling_height_m)
            union_size = _xz_size(union_quad)
            # IoU-triggered bypass: kept UNCONDITIONAL regardless of current
            # size -- this is the original fix for a real bug (cam2, right):
            # several regions of the SAME fixture, each ALREADY close to
            # max_region_size_m on their own, could never merge (every
            # attempt rejected by the size guard, permanently fragmenting
            # one fixture into several near-duplicate boxes). A genuinely
            # HIGH IoU between two comparably-large regions (this bug's own
            # regression test uses IoU~0.71) is strong, size-ratio-aware
            # evidence of "same object" no matter how big either already is.
            #
            # Containment-triggered bypass: additionally requires the
            # RESULTING union to not actually be bigger than whichever input
            # was already bigger -- i.e. a TRUE full/near-full containment,
            # where merging genuinely adds no new area (this bug's own
            # regression test's own reasoning: "the union of a fully
            # contained pair can't exceed the larger box's own size at
            # all"). Fixed 2026-09-14 (independent code review + real
            # ground-truth evaluation): unlike IoU, the raw containment
            # SCORE alone does not discriminate "adds no real area" from
            # "mostly overlaps but still sticks out a bit" -- a tiny,
            # brand-new, single-frame (hits=1) region need only have
            # merge_containment_threshold (0.8) of ITS OWN small area
            # overlap an already-oversized region to trigger the raw score,
            # while still nudging the union slightly wider each time.
            # Confirmed via direct merge-event tracing on real data: an
            # already-oversized region absorbed a fresh hits=1 recognition
            # almost EVERY frame this way (116 merge events in 75 frames,
            # low IoU each time -- confirming containment, not IoU, was the
            # trigger), growing a region with hits=80 to a physically-absurd
            # 4.97m x 4.63m (~23 square meter) footprint via many small
            # nudges, not one big jump. A containment match that would add
            # real area can still merge as long as the RESULT respects
            # max_region_size_m (same as any other non-bypassed pair) -- it
            # just no longer gets a free pass to grow past the cap.
            larger_existing_size = np.maximum(_xz_size(a.quad_room), _xz_size(b.quad_room))
            containment_bypass = (containment >= self._merge_containment_threshold
                                   and not (union_size > larger_existing_size + 1e-6).any())
            size_cap_bypassed = iou >= self._size_cap_bypass_iou or containment_bypass
            if not size_cap_bypassed and (union_size > self._max_region_size_m).any():
                skip.add((i, j))
                continue
            # no_support_streak: take the MIN of the two, not the dataclass
            # default of 0 a plain LampRegion(...) construction would silently
            # fall back to. Fixed 2026-09-14 (independent code review): a
            # naive merge let a stale region sitting one frame from expiry
            # (e.g. streak=9/10) get "revived" for free by merging with any
            # qualifying nearby region -- including a brand-new, one-off,
            # single-frame recognition -- completely bypassing the per-frame
            # sustain_min_points gate expire_after_no_support_frames exists to
            # enforce. min() keeps the merged region exactly as close to
            # expiry as its more-supported input was, so a merge can only ever
            # help a struggling region as much as its partner's own real
            # recent support justifies, never reset it outright.
            self._regions[i] = LampRegion(
                quad_room=union_quad, hits=a.hits + b.hits,
                no_support_streak=min(a.no_support_streak, b.no_support_streak),
                recent_quads=merged_recent)
            del self._regions[j]
            skip = set()   # indices shifted -- start the scan fresh

        while len(self._regions) > self._max_regions:
            # Circuit breaker against a systematically-wrong ceiling height
            # (or a genuinely different scene structure) spawning more
            # distinct, never-merging regions than a real room could hold --
            # evict the least-reinforced one, one at a time (argmin returns
            # the FIRST index achieving the minimum, so an already-tracked
            # region ties in favor of being evicted over a brand new one).
            worst_i = int(np.argmin([r.hits for r in self._regions]))
            del self._regions[worst_i]

    def all_reprojected_contours(self, camera: Camera, T_room_cam: Transform) -> List[Optional[np.ndarray]]:
        """Like reprojected_contours(), but returns exactly one entry per
        CURRENTLY HELD region, in self._regions' own order (len(result) ==
        len(self._regions)) -- None for a region whose reprojected bbox
        falls entirely outside the image this frame, its (4,2) float32
        pixel contour otherwise. Used by sustain_and_expire(), which must
        consider every tracked region (not just ones already reinforced
        enough to exclude, unlike reprojected_contours())."""
        out: List[Optional[np.ndarray]] = []
        for r in self._regions:
            px = room_point_to_pixel(camera, T_room_cam, r.quad_room)
            x0, y0 = px[:, 0].min(), px[:, 1].min()
            x1, y1 = px[:, 0].max(), px[:, 1].max()
            if x1 < 0 or y1 < 0 or x0 >= camera.width or y0 >= camera.height:
                out.append(None)
            else:
                out.append(px.astype(np.float32))
        return out

    def reprojected_contours(self, camera: Camera, T_room_cam: Transform) -> List[np.ndarray]:
        """Project every SUFFICIENTLY-REINFORCED held region's (hits >=
        min_hits_to_exclude) 4 corners into the CURRENT camera view, keeping
        only those whose reprojected bbox overlaps the image at all. Returns
        a list of (4,2) float32 contours, the exact shape
        cv2.pointPolygonTest (and this project's own lamp_exclude_blobs
        convention) already expects. No region is ever deleted here --
        expiry is sustain_and_expire()'s job."""
        contours = []
        for r, px in zip(self._regions, self.all_reprojected_contours(camera, T_room_cam)):
            if px is None or r.hits < self._min_hits_to_exclude:
                continue
            contours.append(px)
        return contours

    def sustain_and_expire(self, camera: Camera, T_room_cam: Transform,
                            region_points: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Called once per lamp-region-active frame, AFTER this frame's own
        candidate detection has run. `region_points` must be parallel to
        self._regions IN THE SAME ORDER as when all_reprojected_contours()
        was called earlier this same frame (before detection ran) -- each
        entry is (kept_pts, dim_pts): the pixel-space points from this
        frame's own combined candidate pool (bright/"kept" and dim
        respectively) that fell inside that region's reprojected contour at
        that time. An entry for a region that was out of view that frame
        should be (empty, empty).

        For each region: if it received at least sustain_min_points
        KEPT+DIM points COMBINED this frame, it is SUSTAINED -- refit by
        taking the UNION of its current bounds with a fresh measurement from
        just this frame's kept+dim points, hits += 1, no_support_streak
        reset to 0 (see "why union, not replace" below). Otherwise (too few
        points, ray-casting failed, or the fresh union would be implausibly
        big), it is NOT sustained this frame: no_support_streak += 1, and the
        region is DELETED once that streak reaches
        expire_after_no_support_frames. See this module's own docstring for
        why continuous refit+expiry replaces the earlier
        freeze/corroborated-regrowth machinery entirely, and for why kept and
        dim points count EQUALLY here (an earlier, kept-only version of this
        check was a real bug, not a safety feature).

        Why UNION, not a fresh replace: an earlier version of this method
        replaced a region's bounds outright with just that frame's own
        support each sustained frame -- confirmed via real e2e testing to
        be a real regression (cam3: 0/214 leaked baseline back up to
        191/214). Root cause: `sustain_and_expire`'s own containment test
        only counts a point as support if it already falls inside the
        region's CURRENT contour, so replacing outright can only ever
        shrink or hold steady, never recover -- any frame where only PART
        of a genuinely-still-there fixture is recognized (very common; the
        line-finder/brightness threshold doesn't re-find every element
        every single frame) permanently shrinks the box, and it can never
        grow back on its own. Taking the union instead means a region can
        still expand to include a newly (or intermittently) visible part of
        the SAME fixture, while the actual "let a wrong or gone box go"
        behavior the user asked for comes from expire_after_no_support_frames
        -- not from ever discarding known-good extent on a routine partial-
        visibility frame."""
        if not self._regions:
            return
        survivors: List[LampRegion] = []
        for i, region in enumerate(self._regions):
            kept_pts, dim_pts = region_points[i] if i < len(region_points) else (
                np.empty((0, 2)), np.empty((0, 2)))
            kept_pts = np.asarray(kept_pts, dtype=np.float64).reshape(-1, 2)
            dim_pts = np.asarray(dim_pts, dtype=np.float64).reshape(-1, 2)
            sustained = (len(kept_pts) + len(dim_pts)) >= self._sustain_min_points
            if sustained:
                all_pts = np.vstack([kept_pts, dim_pts]) if len(dim_pts) else kept_pts
                fresh_quad = _ray_cast_points_to_quad(
                    all_pts, camera, T_room_cam, self._ceiling_height_m, self._bbox_pad_m,
                    cluster_max_m=self._sustain_cluster_max_m)
                if fresh_quad is None:
                    # Couldn't ray-cast this frame's points at all -- still
                    # real support (the point-count test passed), just
                    # nothing new to fold in.
                    region.no_support_streak = 0
                    survivors.append(region)
                    continue
                # Recompute quad_room from a BOUNDED trailing window of
                # per-observation quads (this frame's fresh_quad included),
                # not an ever-growing all-time union -- see
                # LampRegion.recent_quads' own comment for why: the old
                # all-time-union model had no way back down once a single bad
                # observation (or bypassed merge) made a region too big.
                candidate_recent = ((region.recent_quads or [region.quad_room]) + [fresh_quad]
                                     )[-self._recent_quad_window:]
                unioned_quad = _union_many(candidate_recent, self._ceiling_height_m)
                if (_xz_size(unioned_quad) > self._max_region_size_m).any():
                    # Even the bounded recent window is implausibly large --
                    # distrust THIS frame's fresh_quad specifically (don't add
                    # it to the window), but the point-count test above still
                    # counts as real support this frame.
                    region.no_support_streak = 0
                    survivors.append(region)
                    continue
                region.recent_quads = candidate_recent
                region.quad_room = unioned_quad
                region.hits += 1
                region.no_support_streak = 0
                survivors.append(region)
            else:
                region.no_support_streak += 1
                if region.no_support_streak < self._expire_after_no_support_frames:
                    survivors.append(region)
                # else: dropped -- expired.
        self._regions = survivors
        self._dedupe_and_cap()
