"""Regression coverage for blob_detection.min_circularity_pixels: the
circularity filter (4*pi*area/perimeter^2) is only applied to blobs with at
least that many FILLED pixels (default 4).

Before: a 2-3 pixel blob lying along a single axis has a degenerate contour
(shoelace area 0, circularity 0) and was always rejected as "not circular",
even though a run that short carries no meaningful shape information and is
a perfectly plausible tiny LED. Blobs of 4+ pixels are unchanged.

Uses stdlib unittest (pytest is not a declared dependency of this project).
Run with:  python3 -m unittest tests.test_circularity_min_pixels
"""
import unittest

import cv2
import numpy as np

from src.blob_detector import _detect_blobs

_H, _W = 60, 80


def _cfg(**over) -> dict:
    cfg = {
        "min_threshold": 10,
        "min_area": 1.0,
        "max_area": 2000.0,
        "min_circularity": 0.5,
        "min_split_dist": 4.0,
        "split_valley_ratio": 0.6,
        "outlier_factor": 3.0,
        "interior_edge_margin_px": 10.0,
    }
    cfg.update(over)
    return cfg


def _img(pixels) -> np.ndarray:
    img = np.zeros((_H, _W), dtype=np.uint8)
    for x, y in pixels:
        img[y, x] = 200
    return img


def _n_kept(img, **over) -> int:
    result = _detect_blobs(img, pixel_threshold=10, required_threshold=15, cfg=_cfg(**over))
    return len(result[0])


class CircularityMinPixelsTests(unittest.TestCase):
    def test_two_pixel_run_is_kept(self):
        self.assertEqual(_n_kept(_img([(30, 30), (31, 30)])), 1)

    def test_three_pixel_straight_run_is_kept(self):
        """The case that was always rejected: 3 pixels along one axis."""
        for pixels in ([(30, 30), (31, 30), (32, 30)],      # horizontal
                       [(30, 30), (30, 31), (30, 32)]):     # vertical
            self.assertEqual(_n_kept(_img(pixels)), 1, pixels)

    def test_three_pixel_l_shape_is_kept(self):
        self.assertEqual(_n_kept(_img([(30, 30), (31, 30), (30, 31)])), 1)

    def test_control_old_behaviour_rejects_three_pixel_run(self):
        """min_circularity_pixels=1 restores 'apply circularity to everything'.
        Guards against these tests going vacuous."""
        self.assertEqual(_n_kept(_img([(30, 30), (31, 30), (32, 30)]), min_circularity_pixels=1), 0)

    def test_four_pixel_straight_run_still_filtered(self):
        """Circularity applies from 4 pixels up -- a straight 4 px run is
        still rejected as non-circular."""
        self.assertEqual(_n_kept(_img([(30, 30), (31, 30), (32, 30), (33, 30)])), 0)

    def test_long_streak_still_filtered(self):
        img = np.zeros((_H, _W), dtype=np.uint8)
        cv2.line(img, (10, 30), (60, 30), 200, 1)
        self.assertEqual(_n_kept(img), 0)

    def test_round_blob_kept_regardless(self):
        img = np.zeros((_H, _W), dtype=np.uint8)
        cv2.circle(img, (40, 30), 4, 200, -1)
        self.assertEqual(_n_kept(img), 1)
        self.assertEqual(_n_kept(img, min_circularity_pixels=1), 1)

    def test_threshold_is_configurable(self):
        run3 = _img([(30, 30), (31, 30), (32, 30)])
        self.assertEqual(_n_kept(run3, min_circularity_pixels=4), 1)
        self.assertEqual(_n_kept(run3, min_circularity_pixels=3), 0)   # 3 px now subject to circularity


if __name__ == "__main__":
    unittest.main()
