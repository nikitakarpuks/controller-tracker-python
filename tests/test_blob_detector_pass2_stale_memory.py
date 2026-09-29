"""Regression coverage for BlobDetector's pass-2 (adaptive re-threshold) memory.

Real bug (walk_dark, right controller, cam3, abs frames ~1276-1330; found 2026-09-25):
the per-camera pass-2 memory (`_memory`) got stuck at pixel_threshold=12 /
required_threshold=21 / blob_count=2 (written by the OTHER controller's 2-blob
cold detection). Every later frame of the right controller (6-12 blobs) tripped
the count gate (ratio > pass2_count_gate_max_factor), which reused those stale
thresholds AND skipped the memory write, so the count reference never moved:
a permanent deadlock. Pass 2 then thresholded away dim but real LEDs (frame 1328:
7 correct pass-1 blobs -> 1), leaving 4-5 matches and a wrong thin lock at 1296.

Fixes covered here:
  * pass2_cap_at_frame_stats: pass-2 thresholds never exceed THIS frame's own
    statistics-derived thresholds (raw p25*factor), whatever the memory holds;
  * the count gate compares frame to frame (blob_count recorded every call);
  * pass2_min_survival_fraction guard (default 0 = off): pass 2 keeping too few
    of pass 1's blobs falls back to the pass-1 result and clears the memory.

Uses stdlib unittest. Run with:  python3 -m unittest tests.test_blob_detector_pass2_stale_memory
"""
import copy
import unittest
from pathlib import Path

import numpy as np

from src.blob_detector import BlobDetector
from src.camera import Camera
from src.load_config import load_json_config, load_yaml_config

_ROOT = Path(__file__).resolve().parent.parent
_CALIB = _ROOT / "data" / "cameras" / "calibration_basalt.json"
_CONFIG = _ROOT / "config" / "config.yml"

# An 8-LED cluster (~25 px pitch) with mixed brightness: a few bright LEDs and
# several dim ones whose peaks (17-22) sit BELOW the stale required_threshold 21.
_PTS = [(300 + dx, 200 + dy) for dx, dy in
        [(0, 0), (24, 3), (50, -2), (75, 4), (12, 26), (38, 28), (62, 25), (88, 27)]]
_PEAKS = [60, 30, 20, 19, 45, 22, 17, 19]
_STALE = {"pixel_threshold": 12, "required_threshold": 21, "blob_count": 2, "max_area": 30.0}


def _image(cam):
    img = np.zeros((cam.height, cam.width), np.uint8)
    yy, xx = np.mgrid[-4:5, -4:5]
    for (x, y), p in zip(_PTS, _PEAKS):
        g = (p * np.exp(-(xx ** 2 + yy ** 2) / (2 * 1.1 ** 2))).astype(np.uint8)
        sl = (slice(y - 4, y + 5), slice(x - 4, x + 5))
        img[sl] = np.maximum(img[sl], g)
    return img


class Pass2StaleMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.camera = Camera(load_json_config(str(_CALIB)), camera_idx=3)
        cls.base_cfg = load_yaml_config(str(_CONFIG))["blob_detection"]
        cls.image = _image(cls.camera)

    def _detector(self, memory=None, **cfg_over):
        cfg = copy.deepcopy(self.base_cfg)
        cfg.update(cfg_over)
        d = BlobDetector(3, cfg)
        d._memory = dict(memory or {})
        return d

    def _detect(self, d):
        res, _ = d.detect(self.image, ctrl_label="right", predicted_leds=None, camera=self.camera,
                          pose_source=None, frame_ts_ns=1)
        return res

    def test_fresh_memory_keeps_all_blobs(self):
        d = self._detector()
        self.assertEqual(len(self._detect(d).centroids), 8)
        self.assertEqual(d._memory["blob_count"], 8)

    def test_stale_high_memory_does_not_discard_real_blobs(self):
        # T1: today only 4 of the 8 survive (dim LEDs fall under the stale required_threshold 21).
        d = self._detector(_STALE)
        self.assertEqual(len(self._detect(d).centroids), 8)

    def test_count_gate_deadlock_resolves(self):
        # T2: memory written from a 2-blob detection; an 8-blob population must not stay
        # locked out of the memory (count reference must follow the frame-to-frame count).
        d = self._detector(_STALE)
        for _ in range(5):
            self.assertEqual(len(self._detect(d).centroids), 8)
        self.assertEqual(d._memory["blob_count"], 8)

    def test_cap_off_reproduces_old_loss(self):
        # sanity: the failure is real and the toggle really is what fixes it
        d = self._detector(_STALE, pass2_cap_at_frame_stats=False)
        self.assertLess(len(self._detect(d).centroids), 8)

    def test_cap_only_lowers_thresholds(self):
        # T3: EMA lag when the frame's own thresholds are HIGHER than memory is preserved:
        # memory (stable count 8) with low thresholds must stay the effective threshold.
        d = self._detector({"pixel_threshold": 7, "required_threshold": 13, "blob_count": 8, "max_area": 30.0})
        self.assertEqual(len(self._detect(d).centroids), 8)
        self.assertLessEqual(d._memory["pixel_threshold"], 9)  # blended toward raw (9), never above it

    def test_guard_off_by_default(self):
        self.assertEqual(float(self.base_cfg.get("pass2_min_survival_fraction", 0.0)), 0.0)

    def test_guard_falls_back_to_pass1_and_clears_memory(self):
        # T4: force pass 2 to kill most blobs via an absurd required factor, guard on.
        common = dict(pass2_required_factor=4.0, pass2_cap_at_frame_stats=False)
        off = self._detector(**common)
        n_off = len(self._detect(off).centroids)
        self.assertLess(n_off, 4, "test setup: pass 2 must discard most blobs when the guard is off")
        on = self._detector(pass2_min_survival_fraction=0.5, **common)
        res = self._detect(on)
        self.assertEqual(len(res.centroids), 8)   # pass-1 result
        self.assertEqual(on._memory, {})           # memory cleared


if __name__ == "__main__":
    unittest.main()
