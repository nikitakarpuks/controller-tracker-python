"""Regression coverage for velocity-relaxed required_threshold in the hybrid
warm blob-detection path (src/blob_detector.py's BlobDetector.detect).

Bug fixed 2026-09-09: a fast-moving controller's real LED can motion-blur
below required_threshold (the "at least one pixel must reach this level"
dim-blob rejection floor) and get dropped as background noise, purely
because the shutter integrates its light over a wider blurred footprint,
lowering its peak pixel value -- not because it's actually a dim/fake blob.
threshold_scale (from velocity_threshold_k/velocity_threshold_min_factor,
already computed per-controller per-camera from real tracked velocity in
main.py and threaded into BlobDetector.detect) was already relaxing
_detect_blobs_local's per-LED threshold for warm_strategy: "fit", but had NO
effect on required_threshold for warm_strategy: "hybrid" -- the strategy
this project's config.yml actually has active -- so real fast-moving LEDs
were being filtered out in production. The fix applies the same
threshold_scale to required_threshold in the hybrid warm path only (gated on
has_prior, i.e. a real predicted-LED prior -- never the cold/no-prior path,
which has no single controller's velocity to relax against), floored at
pixel_threshold (a blob can never contain a pixel below pixel_threshold in
the first place, so scaling required_threshold under that floor would be a
no-op relaxation, not a real one).

Uses stdlib unittest (pytest is not a declared dependency of this project).
Run with:  python3 -m unittest tests.test_blob_detector_velocity_threshold
"""
import unittest

import cv2
import numpy as np

from src.blob_detector import BlobDetector

_IMG_SIZE = 60
_CENTER = (30, 30)


def _make_cfg(pass2_enabled: bool = False) -> dict:
    return {
        "min_threshold": 7,                 # pixel_threshold: floor for blob inclusion
        "required_threshold_factor": 2.0,   # base required_threshold = 7*2.0 = 14
        "min_area": 2,
        "max_area": 500,
        "warm_strategy": "hybrid",
        "pass2_threshold_factor": 0.7 if pass2_enabled else 0.0,
        "pass2_required_factor": 1.2,
        "min_circularity": 0.5,
        "min_split_dist": 4.0,
        "split_valley_ratio": 0.6,
        "outlier_factor": 3.0,
        "interior_edge_margin_px": 10.0,
        "local_search_depth_k": 0.0,
    }


def _synthetic_dim_led(peak: int = 9) -> np.ndarray:
    """A small circular blob: outer ring at pixel_threshold (7, included in
    the blob but not "bright"), middle ring at 8, center at `peak` -- the
    single brightest pixel required_threshold is checked against. Concentric
    filled circles keep the overall contour genuinely circular (real
    LEDs pass min_circularity easily; a jagged synthetic shape wouldn't)."""
    img = np.zeros((_IMG_SIZE, _IMG_SIZE), dtype=np.uint8)
    cv2.circle(img, _CENTER, 3, 7, -1)
    cv2.circle(img, _CENTER, 2, 8, -1)
    cv2.circle(img, _CENTER, 1, int(peak), -1)
    return img


def _predicted_leds() -> np.ndarray:
    # columns: x, y, depth_m, facing_cos -- only x/y/depth are read on this
    # code path (pose_guided_thresholds.max_area is omitted from _make_cfg,
    # so facing_cos is never consulted).
    return np.array([[float(_CENTER[0]), float(_CENTER[1]), 1.0, 1.0]])


class HybridWarmVelocityThresholdTests(unittest.TestCase):
    def test_base_required_threshold_still_filters_a_genuinely_dim_blob(self):
        """Sanity/control: with threshold_scale=1.0 (stationary/slow), the
        unscaled required_threshold=14 must still reject a blob peaking at 9
        -- confirms the fix didn't accidentally weaken the no-motion case."""
        det = BlobDetector(camera_idx=0, cfg=_make_cfg())
        image = _synthetic_dim_led(peak=9)
        result, _ = det.detect(image, predicted_leds=_predicted_leds(),
                               local_search_radius_px=15.0, threshold_scale=1.0, velocity_px=0.0)
        self.assertEqual(len(result.centroids), 0,
                          "peak=9 must still fail the unscaled required_threshold=14")

    def test_velocity_relaxed_threshold_recovers_a_motion_blurred_led(self):
        """The actual fix: threshold_scale=0.6 -> required_threshold =
        max(7, round(14*0.6)) = 8 -- a real LED peaking at 9 (motion-blurred
        below the base 14) must now be detected instead of dropped as dim."""
        det = BlobDetector(camera_idx=0, cfg=_make_cfg())
        image = _synthetic_dim_led(peak=9)
        result, _ = det.detect(image, predicted_leds=_predicted_leds(),
                               local_search_radius_px=15.0, threshold_scale=0.6, velocity_px=50.0)
        self.assertEqual(len(result.centroids), 1,
                          "peak=9 should now clear the velocity-relaxed required_threshold=8")
        np.testing.assert_allclose(result.centroids[0], _CENTER, atol=1.0)

    def test_relaxation_never_drops_below_pixel_threshold_floor(self):
        """An extreme (unrealistically low) threshold_scale must still floor
        required_threshold at pixel_threshold=7, not go below it -- going
        below would make the dim-blob check a no-op rather than "relaxed"."""
        det = BlobDetector(camera_idx=0, cfg=_make_cfg())
        image = _synthetic_dim_led(peak=7)  # peak exactly at the floor
        result, _ = det.detect(image, predicted_leds=_predicted_leds(),
                               local_search_radius_px=15.0, threshold_scale=0.01, velocity_px=1000.0)
        self.assertEqual(len(result.centroids), 1,
                          "required_threshold floored at pixel_threshold=7 -- peak=7 clears it")

    def test_cold_path_ignores_threshold_scale(self):
        """No predicted_leds (has_prior=False, i.e. a true cold-start search)
        must use the unscaled required_threshold regardless of
        threshold_scale -- there's no single controller's velocity to relax
        against during a blind full-frame search."""
        det = BlobDetector(camera_idx=0, cfg=_make_cfg())
        image = _synthetic_dim_led(peak=9)
        result, _ = det.detect(image, predicted_leds=None,
                               threshold_scale=0.1, velocity_px=200.0)
        self.assertEqual(len(result.centroids), 0,
                          "cold path (no prior) must not relax required_threshold")


if __name__ == "__main__":
    unittest.main()
