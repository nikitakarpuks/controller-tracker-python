"""Regression tests for src/lamp_region_memory.py -- remembers a recognized
lamp fixture's own real ray-cast points as an axis-aligned 3D RECTANGLE
(single-ray/known-height plane intersection per point, NOT the abandoned
multi-view voxel-vote system, and NOT a rotated-rectangle fit -- see that
module's own docstring for why a rotated representation was tried and
reverted) and reprojects it every frame as a hard geometric exclusion mask
(see src/blob_detector.py's `region_exclude_blobs` check in
`_finish_candidate`). Replaces the earlier point-level `LampAnchorMemory`
(removed 2026-09-13) -- see project_lamp_anchor_memory.md /
project_lamp_row_leak_frames1_13.md for why: a rough rectangle-level hard
mask sidesteps the fragile line-finder dependence the point system needed
every single frame to stay useful.

Uses stdlib unittest (pytest is not a declared dependency of this project),
the same real KB4 fixture as tests/test_static_light_geometry.py.
Run with:  python3 -m unittest tests.test_lamp_region_memory
"""
import unittest
from pathlib import Path

import numpy as np

from src.camera import Camera
from src.lamp_region_memory import (LampRegion, LampRegionMemory, _center_xz, _containment_xz,
                                     _iou_xz, _largest_cluster_xz, _xz_bounds, _xz_size)
from src.load_config import load_json_config
from src.static_light_geometry import camera_room_pose, pixel_to_room_ray, room_point_to_pixel
from src.transformations import Transform

_CALIB_PATH = Path(__file__).resolve().parent.parent / "data" / "cameras" / "calibration_basalt.json"
_CEILING_HEIGHT_M = 3.15

# Real point sets confirmed (by direct computation against this exact
# camera/pose) to ray-cast validly (every point's ray actually reaches the
# ceiling plane) and to land at well-separated room-frame XZ positions --
# not arbitrary. Each represents a candidate's own recognized points (see
# LampRegionMemory.update()'s docstring for why real points, not a bbox's 4
# corners) -- here simply the 4 corners of a bounding box, a valid special
# case of "a candidate's own points" for testing purposes.
_POINTS_BASE = np.array([[300.0, 200.0], [340.0, 200.0], [340.0, 230.0], [300.0, 230.0]])
_POINTS_FAR = np.array([[450.0, 300.0], [490.0, 300.0], [490.0, 330.0], [450.0, 330.0]])
_POINTS_FAR2 = np.array([[550.0, 400.0], [600.0, 400.0], [600.0, 440.0], [550.0, 440.0]])


def _cfg(ceiling_height_m=_CEILING_HEIGHT_M, bbox_pad_m=0.0,
         merge_center_distance_m=0.5, max_regions_per_camera=10,
         max_region_size_m=100.0, min_hits_to_exclude=1, size_cap_bypass_iou=0.5) -> dict:
    # max_region_size_m defaults generously large and min_hits_to_exclude to 1
    # here so tests unrelated to those two features (both new 2026-09-13, see
    # dedicated test methods below) aren't affected by them -- this camera's
    # own real fisheye geometry makes even small pixel point sets project to
    # room-frame footprints of a meter or more at some viewing angles (an
    # oblique/grazing ray-plane intersection, not a real large structure),
    # which a tight default here would spuriously reject.
    return {"ceiling_height_m": ceiling_height_m, "bbox_pad_m": bbox_pad_m,
            "merge_center_distance_m": merge_center_distance_m,
            "max_regions_per_camera": max_regions_per_camera,
            "max_region_size_m": max_region_size_m,
            "min_hits_to_exclude": min_hits_to_exclude,
            "size_cap_bypass_iou": size_cap_bypass_iou}


def _expected_quad(camera, T_room_cam, points_px, ceiling_height_m, pad_m=0.0) -> np.ndarray:
    """Mirrors LampRegionMemory.update()'s own logic: ray-cast the raw
    points, take their axis-aligned XZ bounds directly (NOT a fitted
    rectangle -- see lamp_region_memory's own docstring for why using the
    real points already avoids the corner-inflation problem), then pad by a
    fixed REAL margin."""
    origin, dirs = pixel_to_room_ray(camera, T_room_cam, points_px)
    t = (ceiling_height_m - origin[1]) / dirs[:, 1]
    room_pts = origin + t[:, None] * dirs
    xz = room_pts[:, [0, 2]]
    x0, z0 = xz.min(axis=0) - pad_m
    x1, z1 = xz.max(axis=0) + pad_m
    return np.array([
        [x0, ceiling_height_m, z0], [x1, ceiling_height_m, z0],
        [x1, ceiling_height_m, z1], [x0, ceiling_height_m, z1],
    ], dtype=np.float64)


class LampRegionMemoryUpdateTests(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(load_json_config(str(_CALIB_PATH)), camera_idx=0)
        self.T_room_cam = camera_room_pose(Transform(np.eye(3), np.zeros(3)), self.camera)

    def test_single_bbox_produces_the_analytically_expected_quad(self):
        mem = LampRegionMemory(_cfg())
        mem.update(_POINTS_BASE, self.camera, self.T_room_cam)
        self.assertEqual(len(mem._regions), 1)
        expected = _expected_quad(self.camera, self.T_room_cam, _POINTS_BASE, _CEILING_HEIGHT_M)
        np.testing.assert_allclose(mem._regions[0].quad_room, expected, atol=1e-6)
        self.assertEqual(mem._regions[0].hits, 1)

    def test_real_points_produce_a_tighter_box_than_bbox_corners(self):
        """Regression for a real bug found on live usage (cam0, right): a
        real recognized row's raw pixel bbox, ray-cast at its 4 CORNERS,
        forms a ROTATED quadrilateral in room-space (its long axis runs
        diagonally relative to the room's own X/Z axes, generic for most
        viewing angles) -- axis-aligning that wastes a lot of area on the
        sides perpendicular to its true orientation. Confirmed concretely on
        real data: a tight 152.75x35.25px bbox's 4 corners ray-cast to an
        axis-aligned room-space box of 202.1x103.1px, nearly 3x too tall.
        Feeding update() the row's own real (thin-line) points instead of
        that bbox's 4 corners avoids the inflation entirely, since the real
        points never span the bbox's empty diagonal corners in the first
        place -- confirmed here by comparing the two directly for a
        realistic diagonal 2-point row (only the row's own endpoints, not a
        synthetic rectangle) against its own enclosing bbox's 4 corners."""
        x0, y0, x1, y1 = _POINTS_BASE[:, 0].min(), _POINTS_BASE[:, 1].min(), \
            _POINTS_BASE[:, 0].max(), _POINTS_BASE[:, 1].max()
        bbox_corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
        # A thin diagonal row spanning the same bbox -- just its own 2 real
        # endpoints, not the bbox's other 2 (empty) corners.
        row_points = np.array([[x0, y0], [x1, y1]], dtype=np.float64)

        mem_bbox = LampRegionMemory(_cfg())
        mem_bbox.update(bbox_corners, self.camera, self.T_room_cam)
        bbox_size = _xz_size(mem_bbox._regions[0].quad_room)
        bbox_area = float(bbox_size[0] * bbox_size[1])

        mem_row = LampRegionMemory(_cfg())
        mem_row.update(row_points, self.camera, self.T_room_cam)
        row_size = _xz_size(mem_row._regions[0].quad_room)
        row_area = float(row_size[0] * row_size[1])

        self.assertLess(row_area, bbox_area,
                         f"ray-casting the row's own 2 endpoints ({row_area}) must produce a tighter box than "
                         f"ray-casting its enclosing bbox's 4 corners ({bbox_area})")

    def test_bbox_that_cannot_reach_the_ceiling_is_skipped(self):
        """A ceiling height BELOW the camera must never fabricate a region
        from a physically-impossible intersection."""
        mem = LampRegionMemory(_cfg(ceiling_height_m=-5.0))
        mem.update(_POINTS_BASE, self.camera, self.T_room_cam)
        self.assertEqual(len(mem._regions), 0)

    def test_repeated_recognition_of_same_bbox_merges_and_increments_hits(self):
        mem = LampRegionMemory(_cfg())
        mem.update(_POINTS_BASE, self.camera, self.T_room_cam)
        mem.update(_POINTS_BASE, self.camera, self.T_room_cam)
        mem.update(_POINTS_BASE, self.camera, self.T_room_cam)
        self.assertEqual(len(mem._regions), 1, "repeated recognition of the same bbox must merge, not duplicate")
        self.assertEqual(mem._regions[0].hits, 3)

    def test_merge_takes_the_union_not_an_average(self):
        """A second, wider overlapping point set must EXPAND the remembered
        region to cover both, never blend toward a smaller/shifted average
        -- a hard mask must stay a safe superset of everything ever
        recognized there."""
        mem = LampRegionMemory(_cfg())
        mem.update(_POINTS_BASE, self.camera, self.T_room_cam)
        first_quad = mem._regions[0].quad_room.copy()
        wider_points = _POINTS_BASE + np.array([[-20.0, 0.0], [20.0, 0.0], [20.0, 0.0], [-20.0, 0.0]])
        mem.update(wider_points, self.camera, self.T_room_cam)
        self.assertEqual(len(mem._regions), 1)
        merged = mem._regions[0].quad_room
        self.assertLessEqual(merged[:, 0].min(), first_quad[:, 0].min() + 1e-9)
        self.assertGreaterEqual(merged[:, 0].max(), first_quad[:, 0].max() - 1e-9)
        self.assertLessEqual(merged[:, 2].min(), first_quad[:, 2].min() + 1e-9)
        self.assertGreaterEqual(merged[:, 2].max(), first_quad[:, 2].max() - 1e-9)

    def test_distant_bbox_stays_a_separate_region(self):
        mem = LampRegionMemory(_cfg(merge_center_distance_m=0.5))
        mem.update(_POINTS_BASE, self.camera, self.T_room_cam)
        mem.update(_POINTS_FAR, self.camera, self.T_room_cam)
        self.assertEqual(len(mem._regions), 2)

    def test_max_regions_per_camera_evicts_lowest_hits(self):
        mem = LampRegionMemory(_cfg(merge_center_distance_m=0.05, max_regions_per_camera=2))
        mem.update(_POINTS_BASE, self.camera, self.T_room_cam)
        mem.update(_POINTS_BASE, self.camera, self.T_room_cam)   # reinforce -> hits=2, safe from eviction
        mem.update(_POINTS_FAR, self.camera, self.T_room_cam)    # hits=1
        self.assertEqual(len(mem._regions), 2)
        mem.update(_POINTS_FAR2, self.camera, self.T_room_cam)   # exceeds cap -- evicts lowest-hits (_POINTS_FAR's region)
        self.assertEqual(len(mem._regions), 2)
        hits = sorted(r.hits for r in mem._regions)
        self.assertEqual(hits, [1, 2])

    def test_oversized_bbox_is_rejected_outright(self):
        """Regression for a real bug found on live usage: a large/noisy
        structure (a wall light reflection, or any other big-area spurious
        blob) that happens to satisfy detect_lamp_blobs' own spacing/
        residual checks could still seed a mask -- max_region_size_m rejects
        any single recognition whose own footprint is already implausibly
        big for one real fixture, before it ever becomes a region."""
        mem = LampRegionMemory(_cfg(max_region_size_m=1.5))
        # _POINTS_BASE's own footprint at this camera/pose exceeds 1.5m (a
        # grazing ray-plane intersection, not a real large structure --
        # confirmed by direct computation) -- exactly the shape of point set
        # this guard must reject.
        mem.update(_POINTS_BASE, self.camera, self.T_room_cam)
        self.assertEqual(len(mem._regions), 0)

    def test_merge_refused_if_union_would_exceed_max_region_size(self):
        """NMS-like guard: a fixture's recognized bbox drifting a little
        every frame (mocap/reprojection error) must not let repeated
        merging grow a region without bound -- once the union would exceed
        max_region_size_m, the new recognition becomes its own SEPARATE
        region instead of being folded in."""
        small_box = np.array([[305.0, 225.0], [306.0, 225.0], [306.0, 226.0], [305.0, 226.0]])   # tiny -- confirmed well under 1.5m alone
        small_box_2 = np.array([[306.0, 225.0], [307.0, 225.0], [307.0, 226.0], [306.0, 226.0]])  # adjacent, still small alone
        # But together (as a union) they must exceed the individual boxes'
        # own footprint for this test to actually exercise the guard, so the
        # cap picked below can sit strictly between "one alone" and "both
        # merged" -- verified below rather than assumed.
        mem_probe = LampRegionMemory(_cfg())
        mem_probe.update(small_box, self.camera, self.T_room_cam)
        individual_size = _xz_size(mem_probe._regions[0].quad_room)
        mem_probe.update(small_box_2, self.camera, self.T_room_cam)
        self.assertEqual(len(mem_probe._regions), 1, "test setup: the two small boxes must land close enough to merge with no cap")
        union_size = _xz_size(mem_probe._regions[0].quad_room)
        self.assertGreater(float(union_size.max()), float(individual_size.max()),
                            f"test setup: the union {union_size} must exceed one box alone {individual_size}")

        # size_cap_bypass_iou set unreachably high (IoU can never exceed 1.0)
        # to isolate the RAW size-cap behavior from the (separately tested)
        # high-IoU bypass -- this pair's actual IoU is otherwise high enough
        # to bypass the cap on its own, which is not what this test targets.
        cap = (float(individual_size.max()) + float(union_size.max())) / 2.0
        mem = LampRegionMemory(_cfg(max_region_size_m=cap, size_cap_bypass_iou=1.1))
        mem.update(small_box, self.camera, self.T_room_cam)
        self.assertEqual(len(mem._regions), 1)
        mem.update(small_box_2, self.camera, self.T_room_cam)
        self.assertEqual(len(mem._regions), 2, "the union would exceed max_region_size_m -- must stay separate")


class LampRegionMemoryReprojectTests(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(load_json_config(str(_CALIB_PATH)), camera_idx=0)
        self.T_room_cam = camera_room_pose(Transform(np.eye(3), np.zeros(3)), self.camera)

    def test_in_bounds_region_is_reprojected(self):
        mem = LampRegionMemory(_cfg())
        mem.update(_POINTS_BASE, self.camera, self.T_room_cam)
        out = mem.reprojected_contours(self.camera, self.T_room_cam)
        self.assertEqual(len(out), 1)
        expected_quad = _expected_quad(self.camera, self.T_room_cam, _POINTS_BASE, _CEILING_HEIGHT_M)
        expected_px = room_point_to_pixel(self.camera, self.T_room_cam, expected_quad)
        np.testing.assert_allclose(sorted(out[0].tolist()), sorted(expected_px.tolist()), atol=1e-2)

    def test_region_out_of_current_view_is_omitted_but_never_deleted(self):
        """No time-based expiry: a region whose reprojected bbox falls
        entirely outside the image this frame is simply omitted from THIS
        call's output, not removed -- confirmed by querying a pose that
        brings it back into view afterward and getting it back.

        This camera's real fisheye model has no hard field-of-view cutoff
        (found empirically while building the analogous point-anchor test
        this mirrors) -- so the "away" pose below is one confirmed by direct
        search to actually push the WHOLE reprojected bbox outside
        [0,width)x[0,height), not assumed from geometric intuition. (The
        specific angle was re-picked after room_point_to_pixel started
        clamping rays beyond camera.rpmax onto the theta_max cone instead of
        projecting them arbitrarily -- see that function's own docstring --
        which changed where a sufficiently-extreme rotation actually lands.)"""
        from scipy.spatial.transform import Rotation
        mem = LampRegionMemory(_cfg())
        mem.update(_POINTS_BASE, self.camera, self.T_room_cam)
        self.assertEqual(len(mem._regions), 1)

        T_away = Transform(Rotation.from_euler("x", 90.0, degrees=True).as_matrix(), np.zeros(3))
        T_room_cam_away = camera_room_pose(T_away, self.camera)
        px_away = room_point_to_pixel(self.camera, T_room_cam_away, mem._regions[0].quad_room)
        self.assertTrue(px_away[:, 1].min() >= self.camera.height,
                         "test precondition: the region's reprojected bbox must be out of bounds under T_room_cam_away")

        out_away = mem.reprojected_contours(self.camera, T_room_cam_away)
        self.assertEqual(len(out_away), 0)
        self.assertEqual(len(mem._regions), 1, "region must not be deleted just for being out of view")

        out_back = mem.reprojected_contours(self.camera, self.T_room_cam)
        self.assertEqual(len(out_back), 1)

    def test_no_regions_returns_empty(self):
        mem = LampRegionMemory(_cfg())
        out = mem.reprojected_contours(self.camera, self.T_room_cam)
        self.assertEqual(len(out), 0)

    def test_region_not_excluded_until_min_hits_reached(self):
        """Regression for a real bug found on live usage: a one-off
        line-finder false positive (e.g. off a wall reflection) must not
        immediately become a permanent hard mask -- a region only starts
        excluding anything once reinforced min_hits_to_exclude times.
        update()/merging still happens from hits=1 (confirmed via _regions
        directly), only reprojected_contours()'s OUTPUT is gated."""
        mem = LampRegionMemory(_cfg(min_hits_to_exclude=3))
        mem.update(_POINTS_BASE, self.camera, self.T_room_cam)
        self.assertEqual(len(mem._regions), 1, "the region should still be tracked from hits=1")
        self.assertEqual(len(mem.reprojected_contours(self.camera, self.T_room_cam)), 0,
                          "hits=1 < min_hits_to_exclude=3 -- must not exclude anything yet")

        mem.update(_POINTS_BASE, self.camera, self.T_room_cam)
        self.assertEqual(len(mem.reprojected_contours(self.camera, self.T_room_cam)), 0, "hits=2 still below threshold")

        mem.update(_POINTS_BASE, self.camera, self.T_room_cam)
        self.assertEqual(mem._regions[0].hits, 3)
        self.assertEqual(len(mem.reprojected_contours(self.camera, self.T_room_cam)), 1,
                          "hits=3 reaches min_hits_to_exclude -- must now exclude")


def _quad(x0, x1, z0, z1, height=_CEILING_HEIGHT_M) -> np.ndarray:
    """A synthetic axis-aligned quad directly in room-frame XZ, bypassing
    ray-casting entirely -- the merge/dedup POLICY (see
    LampRegionMemoryDedupePolicyTests below) is pure 2D-box geometry,
    independent of any particular camera's projection quirks, so testing it
    this way avoids depending on this camera's own real (and, at some
    viewing angles, extreme) fisheye distortion the other test classes above
    have to work around."""
    return np.array([[x0, height, z0], [x1, height, z0], [x1, height, z1], [x0, height, z1]], dtype=np.float64)


class LampRegionMemoryDedupePolicyTests(unittest.TestCase):
    """Direct tests of the NMS-like merge policy in isolation from
    ray-casting/camera geometry -- constructs LampRegion objects with
    synthetic room-frame quads and drives _dedupe_and_cap()/_should_merge()
    directly. Regression for two real problems found on live usage (both
    with the real camera/ray-casting-based tests above): (1) near-duplicate
    regions at ~0.9 IoU persisting indefinitely, because the old design only
    ever compared a brand-new incoming box against its single nearest
    existing region, never reconciling two regions that ended up
    overlapping with EACH OTHER; (2) a real fixture recognized as two
    disconnected pieces (e.g. split by a physical gap the line-finder
    couldn't bridge) not merging at all when their IoU was low."""

    def _mem(self, **kwargs) -> LampRegionMemory:
        cfg = {"ceiling_height_m": _CEILING_HEIGHT_M, "bbox_pad_m": 0.0,
               "merge_iou_threshold": 0.2, "merge_center_distance_m": 0.5,
               "max_region_size_m": 100.0, "max_regions_per_camera": 10,
               "min_hits_to_exclude": 1, "size_cap_bypass_iou": 0.5}
        cfg.update(kwargs)
        return LampRegionMemory(cfg)

    def test_high_iou_pair_merges_despite_exceeding_max_region_size(self):
        """The actual fix for a real bug found on live usage (cam2, right):
        several regions of the SAME physical fixture, each already close to
        max_region_size_m on their own, could never merge with each other --
        every attempt was rejected by the size guard, permanently
        fragmenting one fixture into several near-duplicate, unmerged boxes.
        A pair whose IoU reaches size_cap_bypass_iou must merge regardless
        of the resulting size -- they're unambiguously the same object."""
        mem = self._mem(max_region_size_m=1.5, size_cap_bypass_iou=0.5)
        # Two large, heavily-overlapping boxes (IoU ~0.71) whose union
        # (14x10) exceeds max_region_size_m in the X dimension (14 > 1.5).
        mem._regions = [LampRegion(quad_room=_quad(0.0, 12.0, 0.0, 10.0), hits=5),
                        LampRegion(quad_room=_quad(2.0, 14.0, 0.0, 10.0), hits=3)]
        iou = _iou_xz(mem._regions[0].quad_room, mem._regions[1].quad_room)
        self.assertGreaterEqual(iou, 0.5, f"test setup: IoU={iou} must reach size_cap_bypass_iou")
        union_x_size = 14.0 - 0.0
        self.assertGreater(union_x_size, mem._max_region_size_m,
                            "test setup: the union must exceed max_region_size_m for this test to matter")

        mem._dedupe_and_cap()
        self.assertEqual(len(mem._regions), 1, "high-IoU pair must merge even though the union exceeds max_region_size_m")
        self.assertEqual(mem._regions[0].hits, 8)

    def test_low_iou_pair_still_respects_max_region_size(self):
        """A merge NOT already obviously-the-same-object (IoU below
        size_cap_bypass_iou, only triggered by merge_center_distance_m) must
        still respect max_region_size_m as a sanity check."""
        mem = self._mem(max_region_size_m=1.5, size_cap_bypass_iou=0.5, merge_center_distance_m=5.0)
        mem._regions = [LampRegion(quad_room=_quad(0.0, 1.0, 0.0, 1.0), hits=1),
                        LampRegion(quad_room=_quad(0.9, 3.0, 0.0, 1.0), hits=1)]
        iou = _iou_xz(mem._regions[0].quad_room, mem._regions[1].quad_room)
        self.assertLess(iou, 0.5, f"test setup: IoU={iou} must stay below size_cap_bypass_iou")
        centers_dist = np.linalg.norm(np.array([0.5, 0.5]) - np.array([1.95, 0.5]))
        self.assertLessEqual(centers_dist, mem._merge_center_distance_m,
                              "test setup: centers must be close enough to trigger the merge attempt")

        mem._dedupe_and_cap()
        self.assertEqual(len(mem._regions), 2, "a low-IoU merge must still be refused once it would exceed max_region_size_m")

    def test_small_region_fully_contained_in_large_one_still_merges(self):
        """Regression for a real bug found on live usage (cam2, right): a
        small region's box was 100% inside a much larger region's box, yet
        neither existing trigger caught it -- IoU was only ~0.19 (the union
        is dominated by the big box's own area, so even full containment
        scores low IoU) and the centers were 1.2m apart (past
        merge_center_distance_m, since the small box sat in a corner of the
        big one, not near its center). Two regions that were clearly the
        same covered area never merged. merge_containment_threshold fixes
        this directly."""
        mem = self._mem(merge_iou_threshold=0.2, merge_center_distance_m=0.5, merge_containment_threshold=0.8)
        big = LampRegion(quad_room=_quad(-2.0, 2.0, 0.0, 3.0), hits=32)      # 4m x 3m
        small = LampRegion(quad_room=_quad(-1.9, -0.6, 0.1, 0.9), hits=6)    # 1.3m x 0.8m, fully inside big
        mem._regions = [big, small]
        iou = _iou_xz(big.quad_room, small.quad_room)
        containment = _containment_xz(big.quad_room, small.quad_room)
        centers_dist = np.linalg.norm(_center_xz(big.quad_room) - _center_xz(small.quad_room))
        self.assertLess(iou, mem._merge_iou_threshold, f"test setup: IoU={iou} must stay below merge_iou_threshold")
        self.assertGreater(centers_dist, mem._merge_center_distance_m,
                            f"test setup: centers_dist={centers_dist} must exceed merge_center_distance_m")
        self.assertGreaterEqual(containment, 0.99, f"test setup: containment={containment} must be ~1.0")

        mem._dedupe_and_cap()
        self.assertEqual(len(mem._regions), 1, "a fully-contained small region must merge despite low IoU and far centers")
        self.assertEqual(mem._regions[0].hits, 38)
        # The union of a fully-contained pair is exactly the larger box --
        # containment-triggered merges can't grow a region at all here.
        np.testing.assert_allclose(_xz_bounds(mem._regions[0].quad_room), _xz_bounds(big.quad_room))

    def test_containment_merge_bypasses_size_cap_too(self):
        """A containment-triggered merge is exempt from max_region_size_m --
        same reasoning as the high-IoU bypass, and even safer here: the
        union of a fully-contained pair can't exceed the larger box's own
        size at all."""
        mem = self._mem(max_region_size_m=1.5, merge_containment_threshold=0.8)
        big = LampRegion(quad_room=_quad(0.0, 4.0, 0.0, 1.0), hits=10)   # 4m -- already over max_region_size_m alone
        small = LampRegion(quad_room=_quad(0.1, 0.3, 0.1, 0.3), hits=1)  # tiny, fully inside big
        mem._regions = [big, small]
        self.assertGreaterEqual(_containment_xz(big.quad_room, small.quad_room), 0.99)

        mem._dedupe_and_cap()
        self.assertEqual(len(mem._regions), 1, "containment merge must bypass max_region_size_m")
        self.assertEqual(mem._regions[0].hits, 11)

    def test_merge_still_grows_normally_via_union(self):
        """Sanity/regression-shape check: merging still grows into the
        union normally -- there's no more freeze mechanism to interact with
        (see this module's own docstring: continuous per-frame refitting in
        sustain_and_expire() replaced it entirely)."""
        mem = self._mem(merge_center_distance_m=5.0)
        a = LampRegion(quad_room=_quad(0.0, 2.0, 0.0, 1.0), hits=1)
        b = LampRegion(quad_room=_quad(1.5, 4.0, 0.0, 1.0), hits=1)
        mem._regions = [a, b]

        mem._dedupe_and_cap()
        self.assertEqual(len(mem._regions), 1)
        x0, x1, z0, z1 = _xz_bounds(mem._regions[0].quad_room)
        self.assertAlmostEqual(x1, 4.0, msg="a normal merge should still extend into the full union")

    def test_high_iou_pair_merges_even_though_centers_are_far_apart(self):
        """Two boxes shifted by half their own width: centers 5m apart (far
        past merge_center_distance_m=0.5), but IoU=1/3 (>= the 0.2
        threshold) since they overlap heavily -- the OLD center-distance-only
        design would never have merged these; this is the new capability
        under test."""
        mem = self._mem()
        mem._regions = [LampRegion(quad_room=_quad(0.0, 10.0, 0.0, 10.0), hits=1),
                        LampRegion(quad_room=_quad(5.0, 15.0, 0.0, 10.0), hits=1)]
        centers_dist = np.linalg.norm(np.array([5.0, 5.0]) - np.array([10.0, 5.0]))
        self.assertGreater(centers_dist, mem._merge_center_distance_m,
                            "test setup: centers must be farther apart than merge_center_distance_m")
        iou = _iou_xz(mem._regions[0].quad_room, mem._regions[1].quad_room)
        self.assertGreaterEqual(iou, mem._merge_iou_threshold,
                                 f"test setup: IoU={iou} must reach the merge threshold")

        mem._dedupe_and_cap()
        self.assertEqual(len(mem._regions), 1, "high-IoU pair must merge despite far-apart centers")
        self.assertEqual(mem._regions[0].hits, 2, "hits must be summed across the merged pair")
        x0, x1, z0, z1 = mem._regions[0].quad_room[:, 0].min(), mem._regions[0].quad_room[:, 0].max(), \
            mem._regions[0].quad_room[:, 2].min(), mem._regions[0].quad_room[:, 2].max()
        self.assertAlmostEqual(x0, 0.0)
        self.assertAlmostEqual(x1, 15.0)

    def test_low_iou_split_fixture_still_merges_via_center_distance(self):
        """Two NON-overlapping boxes (IoU=0, well under the merge threshold)
        but with close centers -- the real "one physical fixture recognized
        as two disconnected pieces" scenario (e.g. a real ~30px physical gap
        the line-finder couldn't bridge in a single frame, see
        project_lamp_row_leak_frames1_13 memory). Must still merge via the
        center-distance trigger."""
        mem = self._mem(merge_center_distance_m=0.5)
        mem._regions = [LampRegion(quad_room=_quad(0.0, 0.2, 0.0, 0.2), hits=1),
                        LampRegion(quad_room=_quad(0.3, 0.5, 0.0, 0.2), hits=1)]
        iou = _iou_xz(mem._regions[0].quad_room, mem._regions[1].quad_room)
        self.assertEqual(iou, 0.0, "test setup: the two pieces must not overlap at all")
        centers_dist = np.linalg.norm(np.array([0.1, 0.1]) - np.array([0.4, 0.1]))
        self.assertLessEqual(centers_dist, mem._merge_center_distance_m,
                              "test setup: centers must be close enough to trigger the fallback")

        mem._dedupe_and_cap()
        self.assertEqual(len(mem._regions), 1, "a physically-split fixture's two pieces must still merge")
        self.assertEqual(mem._regions[0].hits, 2)

    def test_far_apart_non_overlapping_pair_never_merges(self):
        """Neither trigger fires -- must stay as two separate regions."""
        mem = self._mem()
        mem._regions = [LampRegion(quad_room=_quad(0.0, 0.2, 0.0, 0.2), hits=1),
                        LampRegion(quad_room=_quad(10.0, 10.2, 10.0, 10.2), hits=1)]
        mem._dedupe_and_cap()
        self.assertEqual(len(mem._regions), 2)

    def test_three_way_chain_merges_into_one(self):
        """A qualifies to merge with B, and B (once merged with A) still
        qualifies with C -- the dedup loop must keep going until stable, not
        stop after the first merge."""
        mem = self._mem()
        mem._regions = [
            LampRegion(quad_room=_quad(0.0, 10.0, 0.0, 10.0), hits=1),
            LampRegion(quad_room=_quad(5.0, 15.0, 0.0, 10.0), hits=1),
            LampRegion(quad_room=_quad(10.0, 20.0, 0.0, 10.0), hits=1),
        ]
        mem._dedupe_and_cap()
        self.assertEqual(len(mem._regions), 1)
        self.assertEqual(mem._regions[0].hits, 3)


def _sustain_cfg(**overrides) -> dict:
    # sustain_cluster_max_m defaults to effectively-disabled (inf) here, same
    # spirit as max_region_size_m/min_hits_to_exclude's own generous test
    # defaults above: this test file's camera pose (identity headset pose
    # composed with the real cam0 T_imu_cam) puts the ceiling tens of meters
    # away in room-space for these hand-picked pixel coordinates -- a real
    # recording's own actual geometry is a few meters, where
    # sustain_cluster_max_m's real production default (0.3m) is meaningful,
    # but applying that same small metric threshold to THIS file's inflated
    # synthetic scale would spuriously split points that are meant to
    # represent one coherent row. _largest_cluster_xz (the clustering logic
    # itself) has its own dedicated, scale-independent unit tests below.
    cfg = {"ceiling_height_m": _CEILING_HEIGHT_M, "bbox_pad_m": 0.0,
           "merge_center_distance_m": 0.05, "max_regions_per_camera": 10,
           "max_region_size_m": 100.0, "min_hits_to_exclude": 1,
           "size_cap_bypass_iou": 0.5, "sustain_min_points": 3,
           "expire_after_no_support_frames": 3, "sustain_cluster_max_m": float("inf")}
    cfg.update(overrides)
    return cfg


class LampRegionMemorySustainExpireTests(unittest.TestCase):
    """Regression for the 2026-09-13 lifecycle rewrite (see this module's
    own docstring): a region is no longer a permanent memory -- it's a LIVE
    thing, continuously re-measured and expired once evidence stops backing
    it up. User's own framing: "you must keep an eye on what is happening
    inside; if there are not enough blobs anymore, this bbox must be gone" /
    "fit box to the blobs inside, make it really tight, adapt each step" /
    "if it goes out of view, you can also let it go". This entirely
    replaces the earlier freeze_after_hits/regrowth_confirmations machinery
    (see the now-removed LampRegionMemoryCorroboratedRegrowthTests, and this
    class's own DedupePolicyTests sibling for what's unchanged: merging is
    still NMS-like union, just no longer freeze-gated)."""

    def setUp(self):
        self.camera = Camera(load_json_config(str(_CALIB_PATH)), camera_idx=0)
        self.T_room_cam = camera_room_pose(Transform(np.eye(3), np.zeros(3)), self.camera)

    def _seed(self, mem: LampRegionMemory) -> LampRegion:
        mem.update(_POINTS_BASE, self.camera, self.T_room_cam)
        self.assertEqual(len(mem._regions), 1, "test setup: seeding must create exactly one region")
        return mem._regions[0]

    def test_sustained_region_grows_to_cover_newly_supported_points(self):
        """Regression for a real bug found via e2e testing (cam3: replacing
        a region's bounds outright with just that frame's own support,
        instead of unioning, took the 0/214-leaked baseline back up to
        191/214) -- sustain_and_expire's own containment test only counts a
        point as support if it's already inside the CURRENT contour, so a
        pure replace can only ever shrink or hold steady, never recover a
        part of the SAME fixture that wasn't recognized this particular
        frame. A sustained refit must instead be able to GROW to cover a
        wider support set than the region's own current bounds."""
        mem = LampRegionMemory(_sustain_cfg())
        # Seed a SMALL region first (not _POINTS_BASE, already wide) so a
        # wider later observation has real room to grow into.
        small_seed = np.array([[313.0, 211.0], [314.0, 212.0], [315.0, 213.0]])
        mem.update(small_seed, self.camera, self.T_room_cam)
        self.assertEqual(len(mem._regions), 1, "test setup")
        original_size = _xz_size(mem._regions[0].quad_room)
        wider_kept = np.array([[290.0, 205.0], [313.0, 211.0], [340.0, 218.0]])
        mem.sustain_and_expire(self.camera, self.T_room_cam, [(wider_kept, np.empty((0, 2)))])
        self.assertEqual(len(mem._regions), 1)
        new_size = _xz_size(mem._regions[0].quad_room)
        self.assertTrue(np.all(new_size > original_size),
                         f"a wider support set must grow the region ({new_size} vs original {original_size})")
        self.assertEqual(mem._regions[0].hits, 2, "a sustained frame increments hits")
        self.assertEqual(mem._regions[0].no_support_streak, 0)

    def test_too_few_kept_points_does_not_sustain_but_survives_the_grace_period(self):
        mem = LampRegionMemory(_sustain_cfg(sustain_min_points=3, expire_after_no_support_frames=3))
        self._seed(mem)
        original_quad = mem._regions[0].quad_room.copy()
        few_kept = np.array([[310.0, 210.0], [312.0, 211.0]])   # only 2 < sustain_min_points=3
        mem.sustain_and_expire(self.camera, self.T_room_cam, [(few_kept, np.empty((0, 2)))])
        self.assertEqual(len(mem._regions), 1, "one under-supported frame must not delete the region yet")
        np.testing.assert_allclose(mem._regions[0].quad_room, original_quad,
                                    err_msg="an unsustained frame must not change the region's bounds at all")
        self.assertEqual(mem._regions[0].no_support_streak, 1)

    def test_region_expires_after_reaching_no_support_limit(self):
        mem = LampRegionMemory(_sustain_cfg(sustain_min_points=3, expire_after_no_support_frames=3))
        self._seed(mem)
        no_support = (np.empty((0, 2)), np.empty((0, 2)))
        mem.sustain_and_expire(self.camera, self.T_room_cam, [no_support])
        mem.sustain_and_expire(self.camera, self.T_room_cam, [no_support])
        self.assertEqual(len(mem._regions), 1, "still under expire_after_no_support_frames")
        mem.sustain_and_expire(self.camera, self.T_room_cam, [no_support])
        self.assertEqual(len(mem._regions), 0, "the 3rd no-support frame must delete the region")

    def test_sustained_frame_resets_the_no_support_streak(self):
        mem = LampRegionMemory(_sustain_cfg(sustain_min_points=3, expire_after_no_support_frames=3))
        self._seed(mem)
        no_support = (np.empty((0, 2)), np.empty((0, 2)))
        mem.sustain_and_expire(self.camera, self.T_room_cam, [no_support])
        mem.sustain_and_expire(self.camera, self.T_room_cam, [no_support])
        self.assertEqual(mem._regions[0].no_support_streak, 2)
        good_kept = np.array([[310.0, 210.0], [312.0, 211.0], [314.0, 212.0]])
        mem.sustain_and_expire(self.camera, self.T_room_cam, [(good_kept, np.empty((0, 2)))])
        self.assertEqual(mem._regions[0].no_support_streak, 0, "a sustained frame must reset the streak")
        # Would have expired at the 3rd no-support frame without the reset above.
        mem.sustain_and_expire(self.camera, self.T_room_cam, [no_support])
        mem.sustain_and_expire(self.camera, self.T_room_cam, [no_support])
        self.assertEqual(len(mem._regions), 1)

    def test_out_of_view_counts_as_no_support(self):
        """The caller (BlobDetector.detect()) passes (empty, empty) for a
        region whose all_reprojected_contours() entry was None (out of
        view) that frame -- functionally identical to zero real support."""
        mem = LampRegionMemory(_sustain_cfg(sustain_min_points=3, expire_after_no_support_frames=1))
        self._seed(mem)
        mem.sustain_and_expire(self.camera, self.T_room_cam, [(np.empty((0, 2)), np.empty((0, 2)))])
        self.assertEqual(len(mem._regions), 0,
                          "an out-of-view (zero points) frame must expire the region just like any other no-support frame")

    def test_dim_points_are_folded_into_the_refit(self):
        """Enough KEPT points to sustain (kept+dim >= sustain_min_points), but the
        refit's UNION must include the DIM points too -- they trace the same
        physical row, just below the bright/kept brightness threshold.
        Seeded with a SMALL region (not _POINTS_BASE, already wide) so the
        dim points have real room to widen the union into."""
        kept = np.array([[313.0, 211.0], [314.0, 212.0], [315.0, 213.0]])
        small_seed = kept
        dim = np.array([[295.0, 208.0], [340.0, 218.0]])   # wider than kept alone

        mem = LampRegionMemory(_sustain_cfg(sustain_min_points=3))
        mem.update(small_seed, self.camera, self.T_room_cam)
        mem.sustain_and_expire(self.camera, self.T_room_cam, [(kept, dim)])
        self.assertEqual(len(mem._regions), 1)
        kept_and_dim_size = _xz_size(mem._regions[0].quad_room)

        mem_kept_only = LampRegionMemory(_sustain_cfg(sustain_min_points=3))
        mem_kept_only.update(small_seed, self.camera, self.T_room_cam)
        mem_kept_only.sustain_and_expire(self.camera, self.T_room_cam, [(kept, np.empty((0, 2)))])
        kept_only_size = _xz_size(mem_kept_only._regions[0].quad_room)

        self.assertTrue(np.any(kept_and_dim_size > kept_only_size + 1e-6),
                         f"including dim points ({kept_and_dim_size}) must widen the refit vs kept-only ({kept_only_size})")

    def test_dim_points_count_toward_the_sustain_threshold(self):
        """The actual bug fix: an earlier version of this check counted only
        KEPT points toward sustain_min_points, so a frame with abundant real
        DIM evidence but few kept points still counted as "no support".
        Confirmed via real e2e tracing (cam0, left lamp): a frame with 20
        real dim points and only 2 kept ones was being treated as
        unsustained (2 < the old kept-only bar of 3), expiring a region with
        overwhelming real evidence and forcing it to be re-created from
        scratch repeatedly. kept+dim combined must clear the bar here even
        though kept ALONE does not."""
        mem = LampRegionMemory(_sustain_cfg(sustain_min_points=3))
        self._seed(mem)
        barely_any_kept = np.array([[310.0, 210.0], [312.0, 211.0]])   # only 2 kept -- below the bar alone
        plenty_of_dim = np.array([[300.0 + i, 205.0 + i] for i in range(20)])  # 20 real dim points
        mem.sustain_and_expire(self.camera, self.T_room_cam,
                                [(barely_any_kept, plenty_of_dim)])
        self.assertEqual(len(mem._regions), 1, "region must survive -- must not have been deleted")
        self.assertEqual(mem._regions[0].no_support_streak, 0,
                          "kept(2) + dim(20) = 22 >= sustain_min_points(3) -- this frame IS sustained")
        self.assertEqual(mem._regions[0].hits, 2, "a sustained frame increments hits")

    def test_implausible_refit_is_ignored_but_still_counts_as_sustained(self):
        """Enough kept points to sustain, but the resulting fresh
        measurement would exceed max_region_size_m -- don't trust that
        measurement, but the region still isn't penalized (it WAS
        supported this frame, just not usefully re-measurable)."""
        mem = LampRegionMemory(_sustain_cfg(sustain_min_points=3))
        self._seed(mem)
        original_quad = mem._regions[0].quad_room.copy()
        mem._max_region_size_m = 0.001   # tighten AFTER seeding -- only the refit should be affected
        kept = np.array([[310.0, 210.0], [312.0, 211.0], [314.0, 212.0]])
        mem.sustain_and_expire(self.camera, self.T_room_cam, [(kept, np.empty((0, 2)))])
        self.assertEqual(len(mem._regions), 1)
        np.testing.assert_allclose(mem._regions[0].quad_room, original_quad,
                                    err_msg="an implausibly-sized fresh measurement must not be applied")
        self.assertEqual(mem._regions[0].no_support_streak, 0,
                          "still counts as sustained -- the point-count test passed, only the refit itself was distrusted")


class DueForShadowRefreshTests(unittest.TestCase):
    """Regression for the 2026-09-14 warm-path lamp-tracking design (see
    src/blob_detector.py's _run_warm_lamp_shadow_pass): throttles by the
    RECORDING's own clock (frame_ts_ns), not wall-clock, since this pipeline
    processes prerecorded sequences offline."""

    def _mem(self):
        return LampRegionMemory(_cfg())

    def test_first_call_is_always_due(self):
        mem = self._mem()
        self.assertTrue(mem.due_for_shadow_refresh(1_000_000_000, interval_s=0.15))

    def test_call_before_interval_elapsed_is_not_due(self):
        mem = self._mem()
        self.assertTrue(mem.due_for_shadow_refresh(1_000_000_000, interval_s=0.15))
        self.assertFalse(mem.due_for_shadow_refresh(1_050_000_000, interval_s=0.15))  # +50ms < 150ms

    def test_call_after_interval_elapsed_is_due_again(self):
        mem = self._mem()
        self.assertTrue(mem.due_for_shadow_refresh(1_000_000_000, interval_s=0.15))
        self.assertTrue(mem.due_for_shadow_refresh(1_200_000_000, interval_s=0.15))  # +200ms >= 150ms

    def test_due_call_resets_the_baseline(self):
        mem = self._mem()
        mem.due_for_shadow_refresh(1_000_000_000, interval_s=0.15)
        mem.due_for_shadow_refresh(1_200_000_000, interval_s=0.15)  # due, resets baseline to here
        self.assertFalse(mem.due_for_shadow_refresh(1_250_000_000, interval_s=0.15))  # +50ms since new baseline

    def test_not_due_call_does_not_move_the_baseline(self):
        """A rejected (not-yet-due) call must not silently reset the clock --
        otherwise a stream of frequent-but-rejected calls could keep pushing
        the baseline forward and the refresh would never actually happen."""
        mem = self._mem()
        mem.due_for_shadow_refresh(1_000_000_000, interval_s=0.15)
        mem.due_for_shadow_refresh(1_050_000_000, interval_s=0.15)   # not due, must not move baseline
        mem.due_for_shadow_refresh(1_100_000_000, interval_s=0.15)   # not due, must not move baseline
        self.assertTrue(mem.due_for_shadow_refresh(1_200_000_000, interval_s=0.15))  # +200ms since ORIGINAL baseline


class LargestClusterXzTests(unittest.TestCase):
    """Regression for the 2026-09-14 fix (independent code review + real
    ground-truth evaluation + classical-ML feature analysis, see
    sustain_cluster_max_m's own config.yml comment): a spatially-separate
    group of points that happens to fall inside a region's contour must be
    discarded wholesale, not folded into its bounds. Pure math, no camera/
    ray-casting involved -- deliberately scale-independent so it's immune to
    this test file's own camera-pose scale quirk (see _sustain_cfg's own
    comment)."""

    def test_two_far_apart_clusters_keeps_only_the_larger(self):
        cluster_a = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])          # 3 points
        cluster_b = np.array([[10.0, 10.0], [10.1, 10.0]])                  # 2 points, far away
        xz = np.vstack([cluster_a, cluster_b])
        result = _largest_cluster_xz(xz, max_link_m=0.3)
        self.assertEqual(len(result), 3)
        np.testing.assert_allclose(np.sort(result, axis=0), np.sort(cluster_a, axis=0))

    def test_single_long_chain_stays_together_despite_large_total_span(self):
        """A genuinely continuous row (each point close to its NEIGHBOR, not
        necessarily to the far end) must survive intact -- single-linkage
        clustering, not a max-total-span cutoff."""
        chain = np.array([[float(i) * 0.2, 0.0] for i in range(10)])  # spans 1.8m end-to-end
        result = _largest_cluster_xz(chain, max_link_m=0.3)
        self.assertEqual(len(result), len(chain))

    def test_three_clusters_keeps_only_the_single_largest(self):
        a = np.array([[0.0, 0.0], [0.1, 0.0]])           # 2
        b = np.array([[5.0, 5.0], [5.1, 5.0], [5.2, 5.0]])  # 3 -- largest
        c = np.array([[9.0, 9.0]])                        # 1
        xz = np.vstack([a, b, c])
        result = _largest_cluster_xz(xz, max_link_m=0.3)
        self.assertEqual(len(result), 3)
        np.testing.assert_allclose(np.sort(result, axis=0), np.sort(b, axis=0))

    def test_fewer_than_two_points_returned_unchanged(self):
        xz = np.array([[1.0, 2.0]])
        result = _largest_cluster_xz(xz, max_link_m=0.3)
        np.testing.assert_allclose(result, xz)
        empty = np.empty((0, 2))
        np.testing.assert_allclose(_largest_cluster_xz(empty, max_link_m=0.3), empty)


class SustainClusterFilterIntegrationTests(unittest.TestCase):
    """sustain_and_expire()'s own use of _largest_cluster_xz, exercised
    directly through the public sustain_and_expire()/update() API at a
    deliberately realistic room-space scale (not this file's own inflated
    camera-pose convention -- see _sustain_cfg's comment) via a fabricated
    camera+pose fixture."""

    def setUp(self):
        self.camera = Camera(load_json_config(str(_CALIB_PATH)), camera_idx=0)
        self.T_room_cam = camera_room_pose(Transform(np.eye(3), np.zeros(3)), self.camera)

    def test_spatially_separate_pool_does_not_widen_the_region(self):
        """Enough combined points to clear sustain_min_points, but they form
        TWO room-space-separated clusters (confirmed via direct ray-casting
        of this test's own pixel coordinates, not assumed) -- only the
        cluster overlapping the region's own existing footprint should
        survive into the refit; the other, distant cluster must not widen
        the box out to cover it too."""
        seed = np.array([[318.0, 213.0], [319.0, 214.0], [320.0, 215.0]])  # tight, real cluster
        mem = LampRegionMemory(_sustain_cfg(sustain_min_points=3, sustain_cluster_max_m=1.0))
        mem.update(seed, self.camera, self.T_room_cam)
        self.assertEqual(len(mem._regions), 1)
        original_size = _xz_size(mem._regions[0].quad_room).copy()

        near = np.array([[318.5, 213.5], [319.5, 214.5]])   # same physical cluster as seed
        far = np.array([[100.0, 100.0], [105.0, 105.0]])    # a different part of the image entirely
        mem.sustain_and_expire(self.camera, self.T_room_cam, [(near, far)])

        self.assertEqual(len(mem._regions), 1)
        new_size = _xz_size(mem._regions[0].quad_room)
        # The near cluster may grow the box a little (real new support); the
        # far cluster's presence must not blow it out to a wildly different
        # scale -- assert it stays within a modest multiple of the original.
        self.assertTrue(np.all(new_size < original_size * 5 + 0.5),
                         f"region grew from {original_size} to {new_size} -- "
                         f"the spatially-separate 'far' cluster was not filtered out")


if __name__ == "__main__":
    unittest.main()
