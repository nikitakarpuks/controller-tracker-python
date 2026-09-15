"""Regression coverage for src/blob_detector.py's motion-blur "tail" rescue
(fix landed 2026-09-12): a circularity-failed blob whose own minAreaRect
shape (elongation + fill-ratio) matches a genuine LED motion streak is kept
instead of discarded, UNLESS the lamp-fixture line-finder (detect_lamp_blobs)
actually recognizes it as part of a real, removed lamp line -- see
_looks_like_motion_streak and the "Motion-streak promotion" block in
_detect_blobs for the full mechanism and why line-context (not shape alone)
is required: some individual, non-merged lamp elements in this project's own
recordings are themselves naturally elongated ovals, indistinguishable from a
real streak by shape alone (see config.yml's own min_streak_* comments for
the empirical derivation).

Uses stdlib unittest (pytest is not a declared dependency of this project).
Run with:  python3 -m unittest tests.test_motion_streak_elongation_rescue
"""
import unittest

import cv2
import numpy as np

import src.blob_detector as bd
from src.blob_detector import _detect_blobs, _looks_like_motion_streak

_H, _W = 100, 120
_Y = 50


def _base_cfg(lamp_enabled: bool = False, min_points: int = 5) -> dict:
    cfg = {
        "min_threshold": 10,
        "min_area": 1.0,
        "max_area": 2000.0,
        "min_circularity": 0.5,
        "min_streak_elongation": 2.0,
        "max_streak_elongation": 4.5,
        "min_streak_fill_ratio": 0.30,
        "min_split_dist": 4.0,
        "split_valley_ratio": 0.6,
        "outlier_factor": 3.0,
        "interior_edge_margin_px": 10.0,
    }
    if lamp_enabled:
        cfg["lamp_blob_filter"] = {
            "enabled": True,
            "dim_context_radius_px": 150.0,
            "max_pool_size": 150,
            "area_min": 0.4,
            "area_max": 400.0,
            "spacing_min_px": 4.0,
            "spacing_max_px": 25.0,
            "min_points": min_points,
            "max_lines": 20,
            "max_brightness": 60,
            "max_line_residual_px": 2.0,
        }
    return cfg


def _draw_streak(cx: int = 60, cy: int = _Y) -> np.ndarray:
    """A solid, elongated blob (filled ellipse) matching a genuine LED
    motion-blur streak's own signature: circularity < 0.5, elongation and
    fill-ratio comfortably inside the config's min/max_streak_* window
    (verified: area=316, circularity=0.48, elongation=3.79, fill=0.73)."""
    img = np.zeros((_H, _W), dtype=np.uint8)
    cv2.ellipse(img, (cx, cy), (20, 5), 30, 0, 360, 255, -1)
    return img


def _draw_too_long_streak(cx: int = 60, cy: int = _Y) -> np.ndarray:
    """Same solid-ellipse construction as _draw_streak, stretched well past
    max_streak_elongation (verified: circularity=0.21, elongation=8.60,
    fill=0.67) -- stands in for a lamp-fixture edge or a badly merged
    multi-LED blur that must NOT be rescued just for being elongated+solid."""
    img = np.zeros((_H, _W), dtype=np.uint8)
    cv2.ellipse(img, (cx, cy), (30, 3), 30, 0, 360, 255, -1)
    return img


def _draw_irregular_compact(cx: int = 60, cy: int = _Y) -> np.ndarray:
    """A thin plus/cross: fails circularity (0.29) but is NOT elongated
    (aspect ratio 1.0, below min_streak_elongation) -- a non-circular blob
    that isn't streak-shaped at all must stay rejected."""
    img = np.zeros((_H, _W), dtype=np.uint8)
    cv2.rectangle(img, (cx - 5, cy - 35), (cx + 5, cy + 15), 255, -1)
    cv2.rectangle(img, (cx - 25, cy - 15), (cx + 25, cy - 5), 255, -1)
    return img


def _draw_sparse_elongated(ox: int = 30, oy: int = 15) -> np.ndarray:
    """A thin zigzag stroke: elongated (aspect ratio ~3.7, inside the
    elongation window) but sparse (fill=0.18, well below
    min_streak_fill_ratio) -- proves the fill-ratio guard is load-bearing,
    not incidental, for shapes that are elongated without being solid."""
    img = np.zeros((_H, _W), dtype=np.uint8)
    pts = np.array([[ox, oy], [ox + 20, oy], [ox + 20, oy + 20], [ox + 40, oy + 20],
                    [ox + 40, oy + 40]], dtype=np.int32)
    cv2.polylines(img, [pts], False, 255, thickness=2)
    return img


def _draw_comet(head=(84, _Y), length: int = 24, width_sigma: float = 1.2,
                peak_val: int = 60, thresh: int = 10) -> np.ndarray:
    """A single-peaked "comet": bright compact head smoothly tapering into a
    dimmer tail via a real intensity gradient (not a flat fill) -- matches a
    real motion-blurred LED's own brightness profile and, critically, is
    correctly found as exactly ONE local maximum by _find_split_maxima
    (verified), unlike a flat-fill ellipse of the same footprint (which
    tends to register several tied interior maxima and can spuriously
    "self-satisfy" the lamp line-finder's own spacing/count rule -- found
    empirically while writing this test). Verified: area=34, circularity=
    0.34, elongation=3.75, fill=0.57, n_peaks=1."""
    hx, hy = head
    ys, xs = np.mgrid[0:_H, 0:_W]
    along = np.clip(hx - xs, 0, None)
    perp = ys - hy
    intensity = peak_val * np.exp(-along / (length / 3.0)) * np.exp(-(perp ** 2) / (2 * width_sigma ** 2))
    intensity = np.where(hx - xs >= -1, intensity, 0)
    img = np.clip(intensity, 0, 255).astype(np.uint8)
    img[img < thresh] = 0
    return img


class StreakShapeGateTests(unittest.TestCase):
    """No lamp_blob_filter in scope (disabled): the streak-rescue gate in
    section 3a applies immediately and unconditionally, based on shape alone.
    """

    def test_elongated_solid_streak_is_rescued(self):
        img = _draw_streak()
        cfg = _base_cfg(lamp_enabled=False)
        cfg_disabled = dict(cfg, min_streak_elongation=999.0, max_streak_elongation=0.0)
        without = _detect_blobs(img, pixel_threshold=10, required_threshold=15, cfg=cfg_disabled)
        self.assertEqual(len(without[0]), 0,
                          "control: without the rescue thresholds, the streak must fail circularity")
        with_rescue = _detect_blobs(img, pixel_threshold=10, required_threshold=15, cfg=cfg)
        self.assertEqual(len(with_rescue[0]), 1,
                          "a solid, elongated motion-blur-shaped blob should be rescued")

    def test_irregular_non_elongated_blob_stays_rejected(self):
        img = _draw_irregular_compact()
        cfg = _base_cfg(lamp_enabled=False)
        result = _detect_blobs(img, pixel_threshold=10, required_threshold=15, cfg=cfg)
        self.assertEqual(len(result[0]), 0,
                          "a non-circular but non-elongated blob must not be rescued as a streak")

    def test_too_long_streak_stays_rejected(self):
        img = _draw_too_long_streak()
        cfg = _base_cfg(lamp_enabled=False)
        result = _detect_blobs(img, pixel_threshold=10, required_threshold=15, cfg=cfg)
        self.assertEqual(len(result[0]), 0,
                          "elongation past max_streak_elongation must still be rejected")

    def test_sparse_elongated_shape_stays_rejected(self):
        img = _draw_sparse_elongated()
        cfg = _base_cfg(lamp_enabled=False)
        result = _detect_blobs(img, pixel_threshold=10, required_threshold=15, cfg=cfg)
        self.assertEqual(len(result[0]), 0,
                          "an elongated but sparse/hollow shape must not be rescued")


class StreakLampLineContextTests(unittest.TestCase):
    """lamp_blob_filter enabled: a streak-shaped blob's rescue is deferred
    until detect_lamp_blobs has had a chance to recognize it as part of a
    real lamp line -- same shape, opposite outcome depending on whether real
    collinear structure exists around it."""

    def test_isolated_streak_is_rescued_even_with_lamp_filter_active(self):
        img = _draw_comet(head=(84, _Y))
        cfg = _base_cfg(lamp_enabled=True)
        result = _detect_blobs(img, pixel_threshold=10, required_threshold=15, cfg=cfg)
        self.assertEqual(len(result[0]), 1,
                          "a streak with no nearby collinear structure has nothing to be "
                          "claimed by, and should still be rescued")

    def test_streak_absorbed_into_a_real_lamp_row_is_not_rescued(self):
        img = np.zeros((_H, _W), dtype=np.uint8)
        for cx in (20, 27, 34, 48, 55):
            cv2.circle(img, (cx, _Y), 1, 40, -1)
        comet = _draw_comet(head=(45, _Y))
        img = np.maximum(img, comet)
        cfg = _base_cfg(lamp_enabled=True, min_points=5)
        result = _detect_blobs(img, pixel_threshold=10, required_threshold=15, cfg=cfg)
        self.assertEqual(len(result[0]), 0,
                          "the comet sits exactly at real lamp-row spacing among 5 other "
                          "collinear elements (>= min_points); detect_lamp_blobs should "
                          "recognize and remove the whole row, comet included, even though "
                          "the comet alone looks streak-shaped")


class LooksLikeMotionStreakUnitTests(unittest.TestCase):
    """Direct, pipeline-independent checks of the helper itself."""

    def _contour_and_area(self, img):
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea).astype(np.float32)
        return cnt, float(cv2.contourArea(cnt))

    def test_solid_streak_passes(self):
        cnt, area = self._contour_and_area(_draw_streak())
        self.assertTrue(_looks_like_motion_streak(cnt, area, _base_cfg()))

    def test_compact_irregular_fails(self):
        cnt, area = self._contour_and_area(_draw_irregular_compact())
        self.assertFalse(_looks_like_motion_streak(cnt, area, _base_cfg()))

    def test_too_long_fails(self):
        cnt, area = self._contour_and_area(_draw_too_long_streak())
        self.assertFalse(_looks_like_motion_streak(cnt, area, _base_cfg()))

    def test_sparse_fails(self):
        cnt, area = self._contour_and_area(_draw_sparse_elongated())
        self.assertFalse(_looks_like_motion_streak(cnt, area, _base_cfg()))

    def test_disabled_by_default_config(self):
        """No min_streak_* keys at all -> the helper's own safe defaults
        (min > max) mean it never rescues anything."""
        cnt, area = self._contour_and_area(_draw_streak())
        self.assertFalse(_looks_like_motion_streak(cnt, area, {}))


if __name__ == "__main__":
    unittest.main()
