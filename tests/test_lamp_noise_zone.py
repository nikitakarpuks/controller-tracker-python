"""Noise-zone guard in detect_lamp_blobs (lamp_blob_filter.noise_zone_min_points / noise_brightness_max)."""
import unittest

import numpy as np

from src.blob_detector import BlobResult
from src.lamp_blob_filter import detect_lamp_blobs

CFG = dict(area_min=0.4, area_max=70.0, spacing_min_px=4.0, spacing_max_px=32.0, min_points=6, max_lines=20,
           max_brightness=210, max_line_residual_px=2.0)


def _blobs(points):
    """points: [(x, y, radius, brightness)]"""
    a = np.array(points, dtype=np.float32)
    return BlobResult(centroids=a[:, :2], radii=a[:, 2], brightnesses=a[:, 3], contours=[None] * len(a))


def _noise_cloud(n=40, seed=0, x0=230, y0=240, w=30, h=40):
    rng = np.random.default_rng(seed)
    return [(float(x0 + rng.integers(0, w)), float(y0 + rng.integers(0, h)), 0.6, 7.0) for _ in range(n)]


# 4 real, clearly-brighter blobs on an exact diagonal (spacing ~7 px) + 3 noise-level points exactly on the same
# diagonal, inside a noise cloud: legacy joins them into one 7-point "lamp line" and removes the 4 real blobs too
REAL4 = [(200.0 + 7 * i, 200.0 + 7 * i, 1.2, 20.0) for i in range(4)]
ON_LINE_NOISE = [(228.0 + 7 * i, 228.0 + 7 * i, 0.6, 7.0) for i in range(3)]
RING = REAL4 + ON_LINE_NOISE
# a genuine lamp row: 8 small bright points, exactly collinear
LAMP = [(400.0 + 12 * i, 30.0, 0.8, 40.0) for i in range(8)]


class NoiseZoneTests(unittest.TestCase):
    @staticmethod
    def _mixed_count(seed, cfg):
        blobs = _blobs(REAL4 + ON_LINE_NOISE + _noise_cloud(seed=seed))
        res = detect_lamp_blobs(blobs, cfg)
        noise_level = blobs.brightnesses <= 8.5
        # removed line candidates that contain BOTH a real (brighter) blob and a noise-level point
        return sum(1 for c in res.rejected_candidates
                   if noise_level[c.blob_indices].any() and (~noise_level[c.blob_indices]).any())

    def test_legacy_builds_mixed_lines_through_a_noise_cloud(self):
        self.assertGreater(sum(self._mixed_count(seed, CFG) for seed in range(12)), 0,
                           "sanity: without the guard, noise clouds do produce mixed (real + noise) lamp lines")

    def test_no_mixed_lines_in_a_noise_zone_when_enabled(self):
        cfg = dict(CFG, noise_zone_min_points=20, noise_brightness_max=8.5)
        for seed in range(12):
            blobs = _blobs(REAL4 + ON_LINE_NOISE + _noise_cloud(seed=seed))
            res = detect_lamp_blobs(blobs, cfg)
            noise_level = blobs.brightnesses <= 8.5
            for c in res.rejected_candidates:
                self.assertFalse(noise_level[c.blob_indices].any(), f"seed {seed}: noise point inside a removed line")
        self.assertEqual(sum(self._mixed_count(seed, cfg) for seed in range(12)), 0)

    def test_real_lamp_row_still_removed_with_guard_on(self):
        cfg = dict(CFG, noise_zone_min_points=20, noise_brightness_max=8.5)
        res = detect_lamp_blobs(_blobs(LAMP + _noise_cloud()), cfg)
        self.assertFalse(res.keep_mask[:8].any(), "an isolated bright lamp row must still be recognised and removed")

    def test_lamp_row_inside_a_noise_cloud_is_still_removed(self):
        cloud = _noise_cloud(n=40, x0=380, y0=10, w=110, h=40)     # cloud overlapping the lamp row's cluster
        cfg = dict(CFG, noise_zone_min_points=20, noise_brightness_max=8.5)
        res = detect_lamp_blobs(_blobs(LAMP + cloud), cfg)
        self.assertFalse(res.keep_mask[:8].any())

    def test_small_noise_count_below_threshold_is_unchanged(self):
        blobs = _blobs(REAL4 + ON_LINE_NOISE + _noise_cloud(n=8))
        off = detect_lamp_blobs(blobs, CFG).keep_mask
        on = detect_lamp_blobs(blobs, dict(CFG, noise_zone_min_points=20, noise_brightness_max=8.5)).keep_mask
        np.testing.assert_array_equal(off, on)


if __name__ == "__main__":
    unittest.main()
