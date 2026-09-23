"""Regression coverage for the spatial-outlier (DBSCAN 1-NN) filter running
AFTER the lamp-fixture filter (src/blob_detector.py, sections 3b/3c).

The filter rejects any blob whose nearest-neighbour distance exceeds
outlier_factor * median(nearest-neighbour distances) of the frame's blobs.
Lamp rows are the tightest structures in a frame (5-9px spacing), so when
they were still in the population the median -- and the cutoff -- collapsed:
real case (static_medium, right_controller, cam3, frame 89): 11 lamp blobs
set a 19.7px cutoff, and 6 of the 9 real controller LEDs (11-31px apart,
controller close to the camera) were rejected as "isolated" in pass 1. Run
after the lamp filter has removed the rows, the median describes the
controller candidates and all 9 survive.

Synthetic reproduction of the same geometry: a 10-point lamp row 7px apart
plus a 9-LED controller cloud 22-30px apart.

Uses stdlib unittest (pytest is not a declared dependency of this project).
Run with:  python3 -m unittest tests.test_spatial_outlier_after_lamp
"""
import unittest

import numpy as np

from src.blob_detector import _detect_blobs, _spatial_outlier_keep

_H, _W = 150, 280
_LAMP_ROW = [(20 + 7 * i, 125) for i in range(10)]
_CLOUD = [(150, 20), (175, 35), (200, 15), (222, 44), (160, 62),
          (190, 76), (215, 96), (150, 100), (240, 20)]
_ISOLATED = (30, 20)          # ~120px from everything else: genuine outlier


def _cfg(lamp_enabled: bool) -> dict:
    cfg = {
        "min_threshold": 10, "min_area": 1.0, "max_area": 2000.0, "min_circularity": 0.3,
        "min_split_dist": 4.0, "split_valley_ratio": 0.6,
        "outlier_factor": 2.5, "interior_edge_margin_px": 10.0,
    }
    if lamp_enabled:
        cfg["lamp_blob_filter"] = {
            "enabled": True, "dim_context_radius_px": 150.0, "max_pool_size": 150,
            "area_min": 0.4, "area_max": 400.0, "spacing_min_px": 4.0, "spacing_max_px": 25.0,
            "min_points": 5, "max_lines": 20, "max_brightness": 60, "max_line_residual_px": 2.0,
        }
    return cfg


def _image(lamp_row=True, cloud=True, isolated=False) -> np.ndarray:
    img = np.zeros((_H, _W), dtype=np.uint8)
    def dot(x, y, v):
        img[y - 1:y + 2, x - 1:x + 2] = v
    if lamp_row:
        for x, y in _LAMP_ROW: dot(x, y, 40)       # dim: within the lamp filter's max_brightness
    if cloud:
        for x, y in _CLOUD: dot(x, y, 200)         # bright controller LEDs
    if isolated:
        dot(*_ISOLATED, 200)
    return img


def _kept(img, lamp_enabled=True):
    res = _detect_blobs(img, pixel_threshold=10, required_threshold=15, cfg=_cfg(lamp_enabled))
    return res, np.asarray(res[0]).reshape(-1, 2)


def _count_near(points, targets, tol=1.5) -> int:
    if len(points) == 0:
        return 0
    return sum(bool((np.linalg.norm(points - np.array(t), axis=1) < tol).any()) for t in targets)


class SpatialOutlierAfterLampTests(unittest.TestCase):
    def test_controller_cloud_survives_next_to_a_lamp_row(self):
        """The fix: lamp row removed first, so the median reflects the cloud."""
        _, pts = _kept(_image())
        self.assertEqual(_count_near(pts, _CLOUD), len(_CLOUD), f"kept: {pts.tolist()}")

    def test_lamp_row_itself_is_still_removed(self):
        _, pts = _kept(_image())
        self.assertEqual(_count_near(pts, _LAMP_ROW), 0)

    def test_control_without_lamp_filter_the_row_poisons_the_median(self):
        """Documents the failure the ordering avoids: with the lamp filter
        off nothing removes the row before the spatial filter, its 7px
        spacing sets a ~17px cutoff, and the 22-30px-spaced cloud is
        rejected. Also guards this suite against going vacuous."""
        _, pts = _kept(_image(), lamp_enabled=False)
        self.assertEqual(_count_near(pts, _CLOUD), 0)

    def test_genuinely_isolated_blob_is_still_rejected(self):
        res, pts = _kept(_image(isolated=True))
        self.assertEqual(_count_near(pts, _CLOUD), len(_CLOUD))
        self.assertEqual(_count_near(pts, [_ISOLATED]), 0, "isolated noise must still be filtered")
        rejected = np.asarray(res[4]).reshape(-1, 2)
        self.assertEqual(_count_near(rejected, [_ISOLATED]), 1,
                         "deferred spatial rejects must still reach the rejected list (debug canvas)")

    def test_isolated_blob_rejected_without_any_lamp_row(self):
        _, pts = _kept(_image(lamp_row=False, isolated=True))
        self.assertEqual(_count_near(pts, _CLOUD), len(_CLOUD))
        self.assertEqual(_count_near(pts, [_ISOLATED]), 0)

    def test_lamp_filter_disabled_still_applies_spatial_filter_in_place(self):
        """No lamp filter -> nothing to wait for; behaviour unchanged."""
        _, pts = _kept(_image(lamp_row=False, isolated=True), lamp_enabled=False)
        self.assertEqual(_count_near(pts, _CLOUD), len(_CLOUD))
        self.assertEqual(_count_near(pts, [_ISOLATED]), 0)


class SpatialOutlierKeepHelperTests(unittest.TestCase):
    def test_fewer_than_two_points_all_kept(self):
        self.assertEqual(_spatial_outlier_keep(np.empty((0, 2)), 2.5).tolist(), [])
        self.assertEqual(_spatial_outlier_keep(np.array([[5.0, 5.0]]), 2.5).tolist(), [True])

    def test_far_point_rejected_cluster_kept(self):
        pts = np.array([[0, 0], [5, 0], [10, 0], [15, 0], [100, 0]], dtype=float)
        self.assertEqual(_spatial_outlier_keep(pts, 2.5).tolist(), [True, True, True, True, False])

    def test_factor_controls_the_cutoff(self):
        pts = np.array([[0, 0], [5, 0], [10, 0], [15, 0], [45, 0]], dtype=float)   # last is 30px from its neighbour
        self.assertFalse(_spatial_outlier_keep(pts, 2.5)[-1])    # cutoff 12.5
        self.assertTrue(_spatial_outlier_keep(pts, 7.0)[-1])     # cutoff 35


if __name__ == "__main__":
    unittest.main()
