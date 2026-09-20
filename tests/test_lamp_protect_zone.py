"""Regression coverage for the lamp-protect zone around a failed warm attempt's
predicted pose (BlobDetector.detect / _detect_blobs `lamp_protect_rects`,
main.py's mid-Phase-2 cold redetect, blob_detection.lamp_blob_filter.
predicted_pose_protect).

Real case (static_medium, right_controller, cam3, relative frames 27-30): the
warm attempt found 9 real LED blobs; it failed, main.py re-detected cold for
brute-force, and the lamp filter swept the LED cluster into a nearby lamp
column's line and removed it -- brute-force got 1 blob instead of 9. Inside
the zone drawn around the LEDs the failed warm attempt predicted, no lamp
mechanism may remove anything now:
  * detect_lamp_blobs' structural line-finder (protected points are still
    seen as context, so a lamp row crossing the zone is recognised exactly as
    before, but they are never removed and never seed a lamp region),
  * pass 2's same-frame propagation of it (H1b),
  * the remembered-region mask (H1c).

Uses stdlib unittest (pytest is not a declared dependency of this project).
Run with:  python3 -m unittest tests.test_lamp_protect_zone
"""
import unittest

import numpy as np

from src.blob_detector import (BlobDetector, _detect_blobs, _legend_rows, _points_in_rects,
                               _render_legend_strip,
                               predicted_leds_protect_rects)

_H, _W = 150, 280
_ROW = [(20 + 8 * i, 60) for i in range(14)]           # one long, dim, lamp-like row
_ZONE = (55.0, 40.0, 95.0, 80.0)                        # covers row points x = 60, 68, 76, 84, 92
_INSIDE = [p for p in _ROW if _ZONE[0] <= p[0] <= _ZONE[2]]
_OUTSIDE = [p for p in _ROW if p not in _INSIDE]


def _cfg() -> dict:
    return {
        "min_threshold": 10, "min_area": 1.0, "max_area": 2000.0, "min_circularity": 0.3,
        "min_split_dist": 4.0, "split_valley_ratio": 0.6, "outlier_factor": 2.5,
        "interior_edge_margin_px": 10.0,
        "lamp_blob_filter": {
            "enabled": True, "dim_context_radius_px": 150.0, "max_pool_size": 150,
            "area_min": 0.4, "area_max": 400.0, "spacing_min_px": 4.0, "spacing_max_px": 25.0,
            "min_points": 5, "max_lines": 20, "max_brightness": 60, "max_line_residual_px": 2.0,
            "static_lamp_mask": {"interior_margin_px": -6.0, "max_brightness": 210},
        },
    }


def _image() -> np.ndarray:
    img = np.zeros((_H, _W), dtype=np.uint8)
    for x, y in _ROW:
        img[y - 1:y + 2, x - 1:x + 2] = 40
    return img


def _run(**kw):
    return _detect_blobs(_image(), pixel_threshold=10, required_threshold=15, cfg=_cfg(), **kw)


def _n_near(points, targets, tol=1.5) -> int:
    pts = np.asarray(points).reshape(-1, 2)
    if len(pts) == 0:
        return 0
    return sum(bool((np.linalg.norm(pts - np.array(t), axis=1) < tol).any()) for t in targets)


class RectHelperTests(unittest.TestCase):
    def test_none_or_empty_gives_no_rects(self):
        self.assertEqual(predicted_leds_protect_rects(None, 20, 640, 480), [])
        self.assertEqual(predicted_leds_protect_rects(np.empty((0, 5)), 20, 640, 480), [])

    def test_one_square_per_led_ignoring_extra_columns(self):
        leds = np.array([[100.0, 200.0, 0.4, 0.9, 4.0], [150.0, 230.0, 0.4, 0.5, 6.0]])
        self.assertEqual(predicted_leds_protect_rects(leds, 10, 640, 480),
                         [(90.0, 190.0, 110.0, 210.0), (140.0, 220.0, 160.0, 240.0)])

    def test_zone_hugs_the_ring_not_its_bounding_box(self):
        """Two LEDs far apart: the empty space between them is NOT protected
        (a bounding box would cover it, and any lamp element sitting there)."""
        leds = np.array([[100.0, 100.0], [300.0, 100.0]])
        rects = predicted_leds_protect_rects(leds, 15, 640, 480)
        self.assertEqual(_points_in_rects(np.array([[200.0, 100.0]]), rects).tolist(), [False])
        self.assertEqual(_points_in_rects(np.array([[110.0, 108.0], [290.0, 95.0]]), rects).tolist(), [True, True])

    def test_clipped_to_image(self):
        leds = np.array([[5.0, 5.0], [630.0, 470.0]])
        self.assertEqual(predicted_leds_protect_rects(leds, 50, 640, 480),
                         [(0.0, 0.0, 55.0, 55.0), (580.0, 420.0, 639.0, 479.0)])

    def test_points_in_rects(self):
        pts = np.array([[10.0, 10.0], [50.0, 50.0], [200.0, 200.0]])
        self.assertEqual(_points_in_rects(pts, [(0, 0, 60, 60)]).tolist(), [True, True, False])
        self.assertEqual(_points_in_rects(pts, []).tolist(), [False, False, False])
        self.assertEqual(_points_in_rects(pts, None).tolist(), [False, False, False])


class StructuralLampFilterTests(unittest.TestCase):
    def test_control_without_zone_the_whole_row_is_removed(self):
        """Guards the suite against going vacuous: the row IS recognised."""
        res = _run()
        self.assertEqual(_n_near(res[0], _ROW), 0)

    def test_zone_keeps_points_inside_and_still_removes_the_rest(self):
        res = _run(lamp_protect_rects=[_ZONE])
        self.assertEqual(_n_near(res[0], _INSIDE), len(_INSIDE), f"kept: {np.asarray(res[0]).tolist()}")
        self.assertEqual(_n_near(res[0], _OUTSIDE), 0, "lamp points outside the zone must still be removed")

    def test_protected_points_never_reach_removed_lists_or_region_seeding(self):
        """result[9] = removed centroids, result[10] = per-candidate point sets
        that seed LampRegionMemory: a protected LED there would grow a lamp
        region over the controller."""
        res = _run(lamp_protect_rects=[_ZONE])
        self.assertEqual(_n_near(res[9], _INSIDE), 0)
        for pts in res[10]:
            self.assertEqual(_n_near(pts, _INSIDE), 0)
        self.assertGreater(len(res[10]), 0, "the rest of the row should still seed a region")

    def test_empty_zone_list_is_a_no_op(self):
        self.assertEqual(_n_near(_run(lamp_protect_rects=[])[0], _ROW), 0)
        self.assertEqual(_n_near(_run(lamp_protect_rects=None)[0], _ROW), 0)


_BIG_CONTOUR = np.array([[0, 0], [_W, 0], [_W, _H], [0, _H]], dtype=np.float32).reshape(-1, 1, 2)


class SameFrameAndRegionMaskTests(unittest.TestCase):
    """Pass 2 re-uses pass 1's removals (H1b) and the persistent region mask
    (H1c) as exclusion contours -- both must also spare the zone."""

    def _cfg_no_line_finder(self):
        cfg = _cfg(); cfg["lamp_blob_filter"]["enabled"] = False
        return cfg

    def _run(self, **kw):
        return _detect_blobs(_image(), pixel_threshold=10, required_threshold=15, cfg=self.cfg, **kw)

    def setUp(self):
        self.cfg = _cfg()

    def test_h1b_pass2_propagation_spares_zone(self):
        self.assertEqual(_n_near(self._run(lamp_exclude_blobs=[_BIG_CONTOUR])[0], _ROW), 0)
        res = self._run(lamp_exclude_blobs=[_BIG_CONTOUR], lamp_protect_rects=[_ZONE])
        self.assertEqual(_n_near(res[0], _INSIDE), len(_INSIDE))

    def test_h1c_region_mask_spares_zone(self):
        self.assertEqual(_n_near(self._run(region_exclude_blobs=[_BIG_CONTOUR])[0], _ROW), 0)
        res = self._run(region_exclude_blobs=[_BIG_CONTOUR], lamp_protect_rects=[_ZONE])
        self.assertEqual(_n_near(res[0], _INSIDE), len(_INSIDE))


class DetectParameterTests(unittest.TestCase):
    def test_detect_accepts_the_parameter_and_none_changes_nothing(self):
        cfg = _cfg()
        a, _ = BlobDetector(0, cfg).detect(_image())
        b, _ = BlobDetector(0, cfg).detect(_image(), lamp_protect_rects=None)
        np.testing.assert_allclose(np.asarray(a.centroids), np.asarray(b.centroids))


class LegendLayoutTests(unittest.TestCase):
    """The saved/rerun blob-detection canvas legend used a fixed 2-row split
    and ran labels off the right edge whenever the cold canvas had many
    entries. It now wraps by measured width."""
    _ENTRIES = [((255, 255, 255), "kept"), ((0, 255, 128), "split (black seam divides pair)"),
                ((0, 255, 255), "not circular (< 0.4, >= 4px)"), ((0, 0, 255), "spatial outlier"),
                ((100, 180, 255), "deep in large blob"), ((180, 0, 0), "too dim (< 12)"),
                ((0, 140, 255), "area > 500px"), ((255, 0, 255), "pixels < 2"),
                ((255, 0, 140), "lamp (removed)"), ((255, 255, 0), "lamp-protected zone (predicted pose)"),
                ((0, 255, 0), "static lamp mask (excluding)"),
                ((0, 200, 255), "static lamp mask (held: not confirmed)"),
                ((255, 0, 0), "static lamp mask (held: has_recent_memory)")]

    def test_every_label_fits_the_width(self):
        import cv2
        for width in (640, 480, 320):
            rows = _legend_rows(self._ENTRIES, width)
            self.assertEqual(sum(len(r) for r in rows), len(self._ENTRIES), "no entry may be dropped")
            for row in rows:
                used = 6 + sum(13 + cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0][0] + 10 for _, l in row)
                if len(row) > 1:
                    self.assertLessEqual(used, width, f"row overflows at width {width}: {[l for _, l in row]}")

    def test_more_than_two_rows_when_needed_and_strip_height_follows(self):
        strip = _render_legend_strip(self._ENTRIES, 640, "info")
        rows = _legend_rows(self._ENTRIES, 640)
        self.assertGreater(len(rows), 2)
        self.assertEqual(strip.shape, (20 * (len(rows) + 1), 640, 3))

    def test_few_entries_stay_on_one_row(self):
        self.assertEqual(len(_legend_rows(self._ENTRIES[:3], 640)), 1)


if __name__ == "__main__":
    unittest.main()
