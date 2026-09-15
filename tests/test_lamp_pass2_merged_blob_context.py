"""Regression coverage for src/blob_detector.py's split-peak lamp-fixture
context, both variants (fixes landed 2026-09-11): too-large-blob rejection
(MergedTooLargeBlobContextTests) and non-circular-blob rejection
(MergedNonCircularBlobContextTests) -- same underlying gap, two different
upstream rejection reasons a real merged multi-element lamp blob can hit.

Bug: a real ceiling-lamp fixture's individual elements sit only a few px
apart (spacing_min_px). At pass 2's lower/wider re-detection threshold
(BlobDetector.detect's own adaptive pass-2 pixel/required thresholds), several
adjacent elements routinely touch and merge into ONE connected component --
confirmed on a real recording (cam3, frame_range 600-700 relative frame 0):
a genuine lamp column's bridging element merged with 5 neighbors into one
128px^2 blob with 6 distinct internal peaks. `_detect_blobs`'s own
"too_large" area filter correctly rejects that merged blob (it's far bigger
than any single real LED/lamp element), but the rejection used to be a
silent drop: unlike dim (sub-threshold) or non-circular-blended blobs, which
both already feed the lamp-fixture filter's "dim context" pool (see
src/lamp_blob_filter.py's module docstring), a too-large blob contributed
NOTHING -- leaving a real, wide physical gap in the fixture's point pattern
that the line-finder could never bridge, so neither side of the merged
region (each individually below min_points) ever got recognized/removed.

Fix: too-large blobs now run the same `_find_split_maxima` peak-finder
already used by the 2-seed merged-blob splitter (just not restricted to the
2-maxima case, and never touching the real filtered_centroids/matching
candidate pool -- only the lamp filter's own dim-context list), contributing
one context point per internal local maximum.

Uses stdlib unittest (pytest is not a declared dependency of this project).
Run with:  python3 -m unittest tests.test_lamp_pass2_merged_blob_context
"""
import time
import unittest

import cv2
import numpy as np

import src.blob_detector as bd
from src.blob_detector import _detect_blobs

_IMG_H, _IMG_W = 100, 120
_Y = 50


def _make_cfg(max_area: float = 20.0) -> dict:
    return {
        "min_area": 1.0,
        "max_area": max_area,
        "min_circularity": 0.5,
        "min_split_dist": 4.0,
        "split_valley_ratio": 0.6,
        "outlier_factor": 3.0,
        "interior_edge_margin_px": 10.0,
        "lamp_blob_filter": {
            "enabled": True,
            "dim_context_radius_px": 150.0,
            "max_pool_size": 150,
            "area_min": 0.4,
            "area_max": 30.0,
            "spacing_min_px": 3.0,
            "spacing_max_px": 10.0,
            "min_points": 6,
            "max_lines": 20,
            "max_brightness": 60,
            "max_line_residual_px": 2.0,
        },
    }


def _draw_lamp_row_with_merged_gap(peak_val: int = 30, bridge_val: int = 15) -> np.ndarray:
    """A straight row of 10 real lamp elements: 3 kept (bright, correctly
    sized) on each side of a 4-element cluster so densely packed it forms
    one connected, oversized blob -- the exact real-recording pattern this
    fix targets. All 10 elements are collinear at y=_Y with 5-7px consecutive
    spacing (well inside spacing_min_px=3/spacing_max_px=10 above)."""
    img = np.zeros((_IMG_H, _IMG_W), dtype=np.uint8)

    for cx in (20, 27, 34):
        cv2.circle(img, (cx, _Y), 1, peak_val, -1)

    # Connecting bridge: one filled rectangle spanning the 4 merged peaks so
    # they form a single 8-connected component well past max_area, then 4
    # brighter peaks drawn on top -- same construction as a real fixture row
    # whose individual elements bleed into their neighbors at a lower
    # threshold, exactly what pass 2's own (lower/wider) re-detection does.
    cv2.rectangle(img, (39, _Y - 2), (58, _Y + 2), bridge_val, -1)
    for cx in (41, 46, 51, 56):
        cv2.circle(img, (cx, _Y), 1, peak_val, -1)

    for cx in (63, 70, 77):
        cv2.circle(img, (cx, _Y), 1, peak_val, -1)

    return img


class MergedTooLargeBlobContextTests(unittest.TestCase):
    def test_merged_blob_is_rejected_as_too_large(self):
        """Control: confirms the synthetic bridge really does form one
        connected component whose area exceeds max_area, matching the real
        recording's own merged-blob shape -- otherwise this test would prove
        nothing about the fix under test."""
        img = _draw_lamp_row_with_merged_gap()
        cfg = _make_cfg(max_area=20.0)
        result = _detect_blobs(img, pixel_threshold=10, required_threshold=15, cfg=cfg)
        large_rejected = result[6]
        self.assertEqual(len(large_rejected), 1,
                          "expected exactly one too-large merged blob bridging the gap")

    def test_full_row_removed_once_merged_blob_contributes_split_peak_context(self):
        """The fix under test: with the merged blob's internal peaks feeding
        lamp-fixture context, all 10 collinear elements (3 kept + 4
        peaks-from-merged-blob + 3 kept) form one contiguous run >= min_points,
        so every real lamp element on BOTH sides of the merged region gets
        removed -- not just an isolated 3-point stub on each side."""
        img = _draw_lamp_row_with_merged_gap()
        cfg = _make_cfg(max_area=20.0)
        result = _detect_blobs(img, pixel_threshold=10, required_threshold=15, cfg=cfg)
        filtered_centroids = result[0]
        self.assertEqual(len(filtered_centroids), 0,
                          "every real lamp element in the row should be filtered, "
                          f"but {len(filtered_centroids)} survived: {filtered_centroids}")

    def test_without_merged_blob_context_each_side_would_be_too_short(self):
        """Sanity/regression-shape check: directly on JUST the 3+3 kept
        elements (no merged-blob context at all -- the pre-fix situation),
        detect_lamp_blobs must NOT find a qualifying line, since each side is
        only 3 points and the gap between them (29px) breaks spacing_max_px.
        Proves the merged blob's split-peak context is actually load-bearing
        for the row to qualify, not incidental."""
        from src.lamp_blob_filter import detect_lamp_blobs
        from src.blob_detector import BlobResult

        centroids = np.array([[20., _Y], [27., _Y], [34., _Y],
                               [63., _Y], [70., _Y], [77., _Y]], dtype=np.float32)
        radii = np.full(6, 1.0, dtype=np.float32)
        brightnesses = np.full(6, 30.0, dtype=np.float32)
        blobs = BlobResult(centroids=centroids, radii=radii,
                            brightnesses=brightnesses, contours=[None] * 6)
        lamp_cfg = _make_cfg()["lamp_blob_filter"]
        result = detect_lamp_blobs(blobs, lamp_cfg)
        self.assertTrue(result.keep_mask.all(),
                         "without the merged-blob's split-peak context, neither "
                         "3-point side should reach min_points -- nothing should be removed")


class MergedNonCircularBlobContextTests(unittest.TestCase):
    """Same physical bug, different rejection reason (fix landed 2026-09-11,
    same day as the too-large-blob one above): a real merged multi-element
    blob doesn't always exceed blob_detection's own (generic) `max_area` --
    confirmed on a real recording (cam3, frames 0-1, bbox ~[190,425,330,465]):
    a 93px/118px 4-5-peak blended blob, comfortably under
    blob_detection.max_area=500, so it was never "too large", only
    non-circular (0.126-0.156 vs min_circularity=0.5). The single-centroid
    contribution the non-circular context path already made was not just
    imprecise there, it was actively wrong: that whole merged blob's own
    contourArea (60-90px^2, the same measure detect_lamp_blobs itself uses)
    exceeds the lamp filter's own area_max (25.0 in production) outright, so
    detect_lamp_blobs' own area_ok filter silently dropped that single
    context point before line-finding ever saw it -- reproduced directly
    with the real recorded centroid/radius (see this session's own
    investigation). Fix: same one-point-per-internal-peak treatment as the
    too-large case, applied at the circularity-rejection context site
    instead of the area-rejection one.

    Uses the EXACT same synthetic row as MergedTooLargeBlobContextTests
    above, just with `max_area` raised so the merged blob clears the
    too-large check and is rejected by circularity instead -- proving this
    is a different rejection path hitting the identical underlying gap, not
    a different bug shape."""

    def test_merged_blob_is_rejected_as_non_circular_not_too_large(self):
        """Control: with a generous max_area, the same synthetic bridge
        blob is no longer too-large (large_rejected is empty) -- it must
        reach the circularity path instead, otherwise this test would be
        exercising the other fix, not this one."""
        img = _draw_lamp_row_with_merged_gap()
        cfg = _make_cfg(max_area=200.0)
        result = _detect_blobs(img, pixel_threshold=10, required_threshold=15, cfg=cfg)
        self.assertEqual(len(result[6]), 0,
                          "merged blob should clear the too-large check with this max_area")

    def test_full_row_removed_once_noncircular_blob_contributes_split_peak_context(self):
        """The fix under test: the merged blob's internal peaks (not its
        single whole-blob centroid, whose own contour area exceeds the lamp
        filter's area_max) bridge the row into one qualifying >= min_points
        run, so every real lamp element gets removed."""
        img = _draw_lamp_row_with_merged_gap()
        cfg = _make_cfg(max_area=200.0)
        result = _detect_blobs(img, pixel_threshold=10, required_threshold=15, cfg=cfg)
        filtered_centroids = result[0]
        self.assertEqual(len(filtered_centroids), 0,
                          "every real lamp element in the row should be filtered, "
                          f"but {len(filtered_centroids)} survived: {filtered_centroids}")


class TooSmallBlobContextTests(unittest.TestCase):
    """Regression for a real bug found 2026-09-13 (cam3, right, frame_range
    5350-5400, region ~[500,350,635,400], frames 4/6/9/11 of that window):
    a genuine single-pixel lamp-row element (area=1, just under
    blob_detection's own min_area) sat exactly where a real ~20px physical
    gap needed bridging. Unlike too-large and non-circular rejections (both
    already fed into the dim-context pool above), a too-small blob
    contributed NOTHING at all -- the SAME underlying gap as
    MergedTooLargeBlobContextTests/MergedNonCircularBlobContextTests above,
    a fourth distinct rejection reason hitting it. Fix: a too-small blob
    that is also dim (never a bright single-pixel hot/dead-sensor-pixel --
    those stay fully invisible, per this function's own "single-pixel blobs
    are always dropped" docstring) now joins the same dim-context pool via
    its own intensity-weighted centroid."""

    _Y = 50

    def _make_cfg(self):
        return {
            "min_area": 2.0,   # so a literal 1px blob is genuinely "too small", not just "small"
            "max_area": 20.0,
            "min_circularity": 0.5,
            "min_split_dist": 4.0,
            "split_valley_ratio": 0.6,
            "outlier_factor": 3.0,
            "interior_edge_margin_px": 10.0,
            "lamp_blob_filter": {
                "enabled": True,
                "dim_context_radius_px": 150.0,
                "max_pool_size": 150,
                "area_min": 0.4,
                "area_max": 30.0,
                "spacing_min_px": 3.0,
                "spacing_max_px": 10.0,
                "min_points": 6,
                "max_lines": 20,
                "max_brightness": 60,
                "max_line_residual_px": 2.0,
            },
        }

    def _draw_row_with_single_pixel_bridge(self, bridge_val):
        """3 real elements + a 1px bridge pixel + 3 more real elements, all
        collinear at 7px spacing (20,27,34,41,48,55,62) -- without the
        bridge (x=41), the two 3-element halves are 14px apart, past
        spacing_max_px=10, and each half (3 points) is below min_points=6."""
        img = np.zeros((100, 100), dtype=np.uint8)
        for cx in (20, 27, 34, 48, 55, 62):
            cv2.circle(img, (cx, self._Y), 1, 30, -1)
        img[self._Y, 41] = bridge_val
        return img

    def test_row_split_by_missing_bridge_without_the_fix_shape(self):
        """Control: with NOTHING at x=41 at all (not even a dim pixel), the
        two 3-point halves must NOT be recognized -- confirms this row
        genuinely needs the bridge, isolating what the fix under test adds."""
        img = np.zeros((100, 100), dtype=np.uint8)
        for cx in (20, 27, 34, 48, 55, 62):
            cv2.circle(img, (cx, self._Y), 1, 30, -1)
        cfg = self._make_cfg()
        result = _detect_blobs(img, pixel_threshold=10, required_threshold=15, cfg=cfg)
        self.assertEqual(len(result[0]), 6, "without any bridge, neither 3-point half should qualify")

    def test_dim_single_pixel_bridge_is_recognized_and_removed(self):
        """The fix under test: a single dim (value=12, between pixel_threshold=10
        and required_threshold=15) pixel at x=41 -- too small (area=1) to ever
        be a real candidate -- now feeds the dim-context pool, bridging the
        row into one qualifying >= min_points run. Every real (circle) element
        gets removed; the bridge pixel itself was never a candidate to begin
        with, so it can't appear as "removed" either."""
        img = self._draw_row_with_single_pixel_bridge(bridge_val=12)
        cfg = self._make_cfg()
        result = _detect_blobs(img, pixel_threshold=10, required_threshold=15, cfg=cfg)
        self.assertEqual(len(result[0]), 0,
                          f"all 6 real elements should be filtered, but {len(result[0])} survived")
        removed = {tuple(np.round(p, 1)) for p in result[9]}
        for cx in (20, 27, 34, 48, 55, 62):
            self.assertIn((float(cx), float(self._Y)), removed)

    def test_bright_single_pixel_stays_fully_invisible(self):
        """A too-small blob that is ALSO bright (>= required_threshold) must
        NOT be treated as trustworthy dim context -- it stays exactly as
        invisible as before this fix (matches _detect_blobs' own docstring:
        single-pixel blobs are always dropped, bright or not). With no
        bridge at all in practice, the row must stay split, same as the
        no-pixel control above."""
        img = self._draw_row_with_single_pixel_bridge(bridge_val=20)  # >= required_threshold=15
        cfg = self._make_cfg()
        result = _detect_blobs(img, pixel_threshold=10, required_threshold=15, cfg=cfg)
        self.assertEqual(len(result[0]), 6,
                          "a bright single pixel must not bridge the row -- it should stay invisible")


class RegionExcludeBlobsTests(unittest.TestCase):
    """Regression coverage for src/blob_detector.py's `region_exclude_blobs`
    H1-style exclusion check in `_finish_candidate` (added 2026-09-13,
    replacing the earlier point-level LampAnchorMemory/`extra_lamp_context`
    system -- see project_lamp_row_leak_frames1_13.md). Feeds
    src/lamp_region_memory.py's reprojected hard-mask rectangles into the
    SAME `cv2.pointPolygonTest`-based mechanism already used for pass-1/
    pass-2 same-frame lamp propagation (`lamp_exclude_blobs`), but with its
    own margin (`interior_margin_px`) and brightness guard
    (`max_brightness`), since this one can persist across many frames
    instead of being derived fresh every frame from the current frame's own
    recognition."""

    _REGION_CONTOUR = np.array([[30., 30.], [70., 30.], [70., 70.], [30., 70.]], dtype=np.float32)

    def _cfg(self, interior_margin_px=15.0, max_brightness=210.0):
        return {
            "min_area": 1.0,
            "max_area": 500.0,
            "min_circularity": 0.5,
            "min_split_dist": 4.0,
            "split_valley_ratio": 0.6,
            "outlier_factor": 3.0,
            "interior_edge_margin_px": 10.0,
            "lamp_blob_filter": {
                "enabled": False,   # irrelevant here -- region_exclude_blobs doesn't depend on it
                "static_lamp_mask": {
                    "interior_margin_px": interior_margin_px,
                    "max_brightness": max_brightness,
                },
            },
        }

    def test_candidate_deep_inside_region_is_excluded(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (50, 50), 2, 30, -1)   # 20px from every edge of the region below
        result = _detect_blobs(img, pixel_threshold=10, required_threshold=15, cfg=self._cfg(),
                                region_exclude_blobs=[self._REGION_CONTOUR])
        self.assertEqual(len(result[0]), 0, "a candidate deep inside the remembered region must be excluded")

    def test_candidate_outside_region_is_kept(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (90, 90), 2, 30, -1)
        result = _detect_blobs(img, pixel_threshold=10, required_threshold=15, cfg=self._cfg(),
                                region_exclude_blobs=[self._REGION_CONTOUR])
        self.assertEqual(len(result[0]), 1, "a candidate outside the remembered region must survive")

    def test_bright_candidate_inside_region_is_spared(self):
        """The safety guard this design is built around: a real, bright
        controller LED that later wanders through the remembered region must
        still survive -- this hard mask bypasses detect_lamp_blobs (and its
        own brightness check) entirely, so it needs its own."""
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (50, 50), 2, 250, -1)   # far above max_brightness=210
        result = _detect_blobs(img, pixel_threshold=10, required_threshold=15,
                                cfg=self._cfg(max_brightness=210.0),
                                region_exclude_blobs=[self._REGION_CONTOUR])
        self.assertEqual(len(result[0]), 1, "a too-bright candidate must be spared even inside the region")

    def test_candidate_within_margin_of_region_edge_is_spared(self):
        """interior_margin_px protects a real element grazing the mask's
        edge, given this mask's own position is a rougher (mocap + assumed
        height) estimate than same-frame pixel geometry -- only a candidate
        MORE than this deep inside gets excluded."""
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (32, 50), 2, 30, -1)   # only ~2px inside the region's x=30 edge
        result = _detect_blobs(img, pixel_threshold=10, required_threshold=15,
                                cfg=self._cfg(interior_margin_px=15.0),
                                region_exclude_blobs=[self._REGION_CONTOUR])
        self.assertEqual(len(result[0]), 1,
                          "a candidate within interior_margin_px of the region's edge must survive")


class SplitMaximaCandidateCapTests(unittest.TestCase):
    """Regression for the 2026-09-11 fix: too-large-blob split-peak context
    used to be gated by a flat PIXEL-COUNT pre-filter on the whole blob
    (_TOO_LARGE_SPLIT_MAX_PIXELS, since removed), which turned out to be the
    wrong grain -- a real, well-structured lamp-fixture merge can legitimately
    be huge (confirmed on a real 1661px merge, cam3 frame_range 1750-1800
    relative frame 15: a whole two-row fixture merged at pass 2's threshold,
    only 50 raw candidate maxima, whole call 1.1ms) while a large near-flat/
    saturated region (the actually-expensive case, a window or overexposed
    light) has candidate count scaling with pixel count instead. The fix
    moved the cap inside _find_split_maxima itself, keyed on the number of
    raw candidate maxima (cheap to compute, correctly small for any real
    blob regardless of size) rather than the blob's own pixel footprint."""

    def test_large_real_structured_blob_finds_all_its_peaks_uncapped(self):
        """A big (>1000px) blob with genuine, well-separated brightness
        peaks -- the shape of a real multi-element lamp-fixture merge --
        must find every one of its real peaks, not be silently skipped for
        being large. This would have been dropped entirely by the OLD
        pixel-count pre-filter (>400px)."""
        img = np.zeros((120, 120), dtype=np.uint8)
        # A big, dim connecting field (still >= peak_threshold isn't required
        # for non-peak pixels) with 6 well-separated, genuinely brighter peaks.
        cv2.rectangle(img, (5, 55), (115, 65), 10, -1)
        # Single-pixel peaks (not small filled circles, which have a flat
        # multi-pixel top and so tie across several rows/cols -- not a real
        # local-maximum shape) so each element has exactly one clear apex.
        peak_xy = [(10, 60), (30, 60), (50, 60), (70, 60), (90, 60), (110, 60)]
        for px, py in peak_xy:
            img[py, px] = 30
        ys, xs = np.nonzero(img > 0)
        self.assertGreater(len(ys), 1000, "test setup: blob should exceed the old 400px cap")

        maxima = bd._find_split_maxima(img, ys, xs, peak_threshold=15, min_split_dist=4.0)
        self.assertEqual(len(maxima), len(peak_xy),
                          f"expected all {len(peak_xy)} real peaks, got {len(maxima)}: {maxima}")

    def test_large_flat_saturated_blob_stays_fast(self):
        """Worst case: a large, fully flat/saturated region (every pixel
        ties with its neighbors, so nearly all of them are raw is_max
        candidates) -- must stay fast regardless of size, via the internal
        candidate cap, not a size-based pre-filter."""
        size = 200
        img = np.full((size + 10, size + 10), 255, dtype=np.uint8)
        ys, xs = np.mgrid[0:size, 0:size]
        ys, xs = ys.ravel(), xs.ravel()

        t0 = time.perf_counter()
        bd._find_split_maxima(img, ys, xs, peak_threshold=15, min_split_dist=4.0)
        dt = time.perf_counter() - t0
        self.assertLess(dt, 0.5,
                         f"a {size}x{size} fully-flat blob must stay well under the "
                         f"multi-second pathological cost seen before capping candidates "
                         f"(took {dt*1000:.0f}ms)")


if __name__ == "__main__":
    unittest.main()
