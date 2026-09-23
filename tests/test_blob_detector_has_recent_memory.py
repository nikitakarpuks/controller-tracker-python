"""Regression coverage for BlobDetector.detect()'s has_recent_memory
parameter (added 2026-09-15, user-proposed): static_lamp_mask's EXCLUSION
(region_exclude_blobs, actually removing a candidate in _finish_candidate)
is skipped whenever has_recent_memory=True, even on an otherwise-eligible
cold call (predicted_leds=None) -- but TRACKING (region creation via
update(), and per-frame refit via sustain_and_expire()) keeps running
regardless, exactly as it does when has_recent_memory=False.

Why this is safe, not just convenient (see BlobDetector.detect()'s own
docstring for the full reasoning, verified against the real code before
implementing): a re-acquisition candidate with a usable last_good_pose
already gets an independent, position/rotation-based corroboration check in
ControllerTracker._check_vs_last_good (src/controller.py) -- a lamp-
mistaken solve would very likely register as an implausible 3D jump there,
regardless of whether this module also masked the lamp's own pixels. A true
cold start (has_recent_memory=False) has no such reference pose to check
against, so masking is the only protection available there, and stays on.

Uses stdlib unittest (pytest is not a declared dependency of this project),
the same real KB4 fixture as tests/test_lamp_region_memory.py.
Run with:  python3 -m unittest tests.test_blob_detector_has_recent_memory
"""
import copy
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.blob_detector import BlobDetector
from src.camera import Camera
from src.load_config import load_json_config, load_yaml_config
from src.static_light_geometry import camera_room_pose
from src.transformations import Transform

_CALIB_PATH = Path(__file__).resolve().parent.parent / "data" / "cameras" / "calibration_basalt.json"
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yml"

# A real recognized row's own points (see LampRegionMemory.update()'s own
# docstring for why real points, not a bbox's 4 corners) -- reused verbatim
# from tests/test_lamp_region_memory.py's own _POINTS_BASE, confirmed there
# to ray-cast validly against this exact camera/pose.
_LAMP_POINTS_PX = np.array([[300.0, 200.0], [340.0, 200.0], [340.0, 230.0], [300.0, 230.0]])


class _StubPoseSource:
    """Always returns the same room pose, regardless of timestamp -- avoids
    needing a real mocap trajectory file for this unit test; the geometry
    itself is still real (a genuine Camera + camera_room_pose call)."""

    def __init__(self, T_room_headsetImu):
        self._T = T_room_headsetImu

    def room_pose_at(self, frame_ts_ns):
        return self._T


class HasRecentMemoryTests(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(load_json_config(str(_CALIB_PATH)), camera_idx=0)
        self.T_room_cam = camera_room_pose(Transform(np.eye(3), np.zeros(3)), self.camera)
        self.pose_source = _StubPoseSource(Transform(np.eye(3), np.zeros(3)))

        full_cfg = load_yaml_config(str(_CONFIG_PATH))
        self.blob_cfg = copy.deepcopy(full_cfg["blob_detection"])
        self.blob_cfg["lamp_blob_filter"]["static_lamp_mask"]["enabled"] = True
        # max_region_size_m overridden generously, same reasoning as
        # tests/test_lamp_region_memory.py's own _cfg() helper: this
        # camera's real fisheye geometry projects even a small synthetic
        # pixel point set to a room-frame footprint of a meter or more at
        # some viewing angles (anisotropic, not a bug) -- the real shipped
        # value (3.0) is tuned against real recognized-row point sets, not
        # this test's synthetic 4-corner rectangle, and isn't what this test
        # is checking anyway (has_recent_memory's own gating logic, not
        # region-size tuning).
        self.blob_cfg["lamp_blob_filter"]["static_lamp_mask"]["max_region_size_m"] = 100.0

        self.detector = BlobDetector(camera_idx=0, cfg=self.blob_cfg)
        # Seed a region directly via update() (the real creation path) using
        # this camera/pose, then force it well past min_hits_to_exclude so
        # it's unambiguously "active" regardless of the shipped config value.
        self.detector._lamp_region_memory = None  # let detect() lazily create it with the real cfg
        from src.lamp_region_memory import LampRegionMemory
        _lamp_region_cfg = self.blob_cfg["lamp_blob_filter"]["static_lamp_mask"]
        mem = LampRegionMemory(_lamp_region_cfg)
        mem.update(_LAMP_POINTS_PX, self.camera, self.T_room_cam)
        self.assertEqual(len(mem._regions), 1, "test setup: seeding must create exactly one region")
        mem._regions[0].hits = 1000  # unambiguously past min_hits_to_exclude
        self.detector._lamp_region_memory = mem

        # THREE bright synthetic candidates near the seeded row's own
        # centroid -- deep inside the resulting region by construction, and
        # >= sustain_min_points (3, the real shipped default) so a
        # has_recent_memory=True frame's tracking-still-runs check below has
        # enough real support to actually sustain the region, not just
        # "count as no support" for an unrelated reason.
        self.image = np.zeros((self.camera.height, self.camera.width), dtype=np.uint8)
        cx, cy = _LAMP_POINTS_PX.mean(axis=0)
        self.n_candidates = 3
        for dx in (-15, 0, 15):
            cv2.circle(self.image, (int(cx + dx), int(cy)), 2, 60, -1)
        self.ts = 12345

    def _n_kept(self, has_recent_memory: bool) -> int:
        result, _ = self.detector.detect(
            self.image, predicted_leds=None, camera=self.camera, pose_source=self.pose_source,
            frame_ts_ns=self.ts, has_recent_memory=has_recent_memory, visualize=False,
        )
        return len(result.centroids)

    def test_true_cold_start_excludes_the_candidate(self):
        """has_recent_memory=False (the default, and today's only behavior
        before this feature) -- a candidate deep inside an active region
        must still be excluded, exactly as before."""
        self.assertEqual(self._n_kept(has_recent_memory=False), 0)

    def test_recent_memory_spares_the_candidate(self):
        """The new behavior: with a usable last_good_pose (has_recent_memory
        =True), the SAME candidates at the SAME positions must survive --
        exclusion is skipped, relying on ControllerTracker's own jump-gate
        instead."""
        self.assertEqual(self._n_kept(has_recent_memory=True), self.n_candidates)

    def test_recent_memory_does_not_disable_tracking(self):
        """Skipping EXCLUSION must not also stop TRACKING -- the region
        must still be sustained (no_support_streak reset, hits incremented)
        on a has_recent_memory=True frame, so it doesn't go stale for the
        next true-cold-start frame that DOES need it."""
        mem = self.detector._lamp_region_memory
        hits_before = mem._regions[0].hits
        self._n_kept(has_recent_memory=True)
        self.assertEqual(len(mem._regions), 1, "region must not be deleted")
        self.assertGreater(mem._regions[0].hits, hits_before,
                            "region must still be sustained (hits incremented) even though exclusion was skipped")
        self.assertEqual(mem._regions[0].no_support_streak, 0)

    def test_default_is_false_backward_compatible(self):
        """Any caller not yet updated to pass has_recent_memory (existing
        tests, throwaway scripts) must see EXACTLY today's behavior --
        default must be False, not True."""
        result, _ = self.detector.detect(
            self.image, predicted_leds=None, camera=self.camera, pose_source=self.pose_source,
            frame_ts_ns=self.ts, visualize=False,
        )
        self.assertEqual(len(result.centroids), 0)


if __name__ == "__main__":
    unittest.main()
