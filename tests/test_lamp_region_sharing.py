"""Regression coverage for cross-camera lamp-region sharing
(blob_detection.lamp_blob_filter.static_lamp_mask.share_across_cameras,
BlobDetector.detect's foreign_lamp_quads, LampRegionMemory.confirmed_quads,
reproject_room_quads).

Real case (walk_medium, relative frames 24-32, right_controller): a real
ceiling lamp was recognised and confirmed in cam3, but its blobs also appear
in cam1, where nothing ever recognised it -- so they stayed candidates, and
brute-force fitted the controller model to them (a pose 1.3 m / 171 deg off
mocap). Regions are stored in the ROOM frame, so a region one camera has
confirmed can simply be reprojected into another camera's view and used for
exclusion there (read-only: the other camera's own memory is never seeded or
sustained by it).

Uses stdlib unittest (pytest is not a declared dependency of this project).
Run with:  python3 -m unittest tests.test_lamp_region_sharing
"""
import copy
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.blob_detector import BlobDetector
from src.camera import Camera
from src.lamp_region_memory import LampRegionMemory, reproject_room_quads
from src.load_config import load_json_config, load_yaml_config
from src.static_light_geometry import camera_room_pose, room_point_to_pixel
from src.transformations import Transform

_ROOT = Path(__file__).resolve().parent.parent
_CALIB = load_json_config(str(_ROOT / "data" / "cameras" / "calibration_basalt.json"))
_FULL_CFG = load_yaml_config(str(_ROOT / "config" / "config.yml"))

# A pixel patch in cam3 whose lamp region lands inside cam1's view (verified
# against this exact calibration: cam1 bbox ~ x[409,498] y[315,362]).
_SEED_PX = np.array([[540.0, 130.0], [580.0, 130.0], [580.0, 150.0], [540.0, 150.0]])


class _StubPoseSource:
    def __init__(self, T):
        self._T = T

    def room_pose_at(self, frame_ts_ns):
        return self._T


def _blob_cfg():
    cfg = copy.deepcopy(_FULL_CFG["blob_detection"])
    cfg["lamp_blob_filter"]["static_lamp_mask"]["enabled"] = True
    cfg["lamp_blob_filter"]["static_lamp_mask"]["max_region_size_m"] = 100.0   # synthetic points, see test_lamp_region_memory
    return cfg


class SharingTests(unittest.TestCase):
    def setUp(self):
        self.T_head = Transform(np.eye(3), np.zeros(3))
        self.pose_source = _StubPoseSource(self.T_head)
        self.cam_src = Camera(_CALIB, camera_idx=3)
        self.cam_dst = Camera(_CALIB, camera_idx=1)
        self.cfg = _blob_cfg()
        lamp_cfg = self.cfg["lamp_blob_filter"]["static_lamp_mask"]
        self.mem_src = LampRegionMemory(lamp_cfg)
        self.mem_src.update(_SEED_PX, self.cam_src, camera_room_pose(self.T_head, self.cam_src))
        self.assertEqual(len(self.mem_src._regions), 1, "test setup: seeding must create one region")
        self.mem_src._regions[0].hits = 1000

        # Where the region lands in the destination camera, and 3 dim blobs there.
        self.T_room_dst = camera_room_pose(self.T_head, self.cam_dst)
        contours = reproject_room_quads(self.mem_src.confirmed_quads(), self.cam_dst, self.T_room_dst)
        self.assertEqual(len(contours), 1, "test setup: region must be in the destination camera's view")
        cx, cy = contours[0].mean(axis=0)
        self.image = np.zeros((self.cam_dst.height, self.cam_dst.width), dtype=np.uint8)
        for dx in (-15, 0, 15):
            cv2.circle(self.image, (int(cx + dx), int(cy)), 2, 60, -1)

    def _detect(self, **kw):
        det = BlobDetector(camera_idx=1, cfg=self.cfg)
        res, _ = det.detect(self.image, predicted_leds=None, camera=self.cam_dst,
                            pose_source=self.pose_source, frame_ts_ns=1, visualize=False, **kw)
        return det, len(res.centroids)

    def test_control_without_sharing_the_lamp_blobs_survive_in_the_neighbour(self):
        """The bug: the destination camera never recognised the lamp."""
        _, n = self._detect()
        self.assertEqual(n, 3)

    def test_shared_region_masks_the_same_lamp_in_the_neighbour(self):
        _, n = self._detect(foreign_lamp_quads=self.mem_src.confirmed_quads())
        self.assertEqual(n, 0)

    def test_sharing_is_read_only_for_the_neighbours_own_memory(self):
        det, _ = self._detect(foreign_lamp_quads=self.mem_src.confirmed_quads())
        mem = det._lamp_region_memory
        self.assertTrue(mem is None or len(mem._regions) == 0,
                        "a foreign region must never be seeded into this camera's own memory")

    def test_has_recent_memory_still_disables_exclusion_including_foreign(self):
        _, n = self._detect(foreign_lamp_quads=self.mem_src.confirmed_quads(), has_recent_memory=True)
        self.assertEqual(n, 3)

    def test_protect_zone_spares_blobs_from_a_foreign_region_too(self):
        _, n = self._detect(foreign_lamp_quads=self.mem_src.confirmed_quads(),
                            lamp_protect_rects=[(0.0, 0.0, float(self.cam_dst.width - 1), float(self.cam_dst.height - 1))])
        self.assertEqual(n, 3)

    def test_none_or_empty_foreign_quads_change_nothing(self):
        self.assertEqual(self._detect(foreign_lamp_quads=None)[1], 3)
        self.assertEqual(self._detect(foreign_lamp_quads=[])[1], 3)


class ConfirmedQuadsAndReprojectionTests(unittest.TestCase):
    def setUp(self):
        self.T_head = Transform(np.eye(3), np.zeros(3))
        self.cam3 = Camera(_CALIB, camera_idx=3)
        self.cam1 = Camera(_CALIB, camera_idx=1)
        self.lamp_cfg = _blob_cfg()["lamp_blob_filter"]["static_lamp_mask"]
        self.mem = LampRegionMemory(self.lamp_cfg)
        self.mem.update(_SEED_PX, self.cam3, camera_room_pose(self.T_head, self.cam3))

    def test_only_confirmed_regions_are_shared(self):
        self.assertEqual(self.mem._regions[0].hits, 1)
        self.assertEqual(self.mem.confirmed_quads(), [], "an unconfirmed region must not be shared")
        self.mem._regions[0].hits = int(self.lamp_cfg.get("min_hits_to_exclude", 3))
        self.assertEqual(len(self.mem.confirmed_quads()), 1)

    def test_confirmed_quads_are_copies(self):
        self.mem._regions[0].hits = 1000
        q = self.mem.confirmed_quads()[0]
        q[:] = 0.0
        self.assertFalse(np.allclose(self.mem._regions[0].quad_room, 0.0))

    def test_reprojection_covers_the_direct_projection(self):
        """In-view region: the sampled hull must contain the 4 directly
        projected corners (it may be slightly larger: fisheye edges bow)."""
        self.mem._regions[0].hits = 1000
        T_room_cam1 = camera_room_pose(self.T_head, self.cam1)
        (px,) = reproject_room_quads(self.mem.confirmed_quads(), self.cam1, T_room_cam1)
        direct = room_point_to_pixel(self.cam1, T_room_cam1, self.mem._regions[0].quad_room)
        hull = px.reshape(-1, 1, 2).astype(np.float32)
        for c in direct:
            self.assertGreaterEqual(cv2.pointPolygonTest(hull, (float(c[0]), float(c[1])), True), -2.0)
        self.assertLess(cv2.contourArea(hull), 4 * cv2.contourArea(direct.astype(np.float32)))

    def test_quad_behind_the_camera_is_dropped_not_smeared(self):
        """KB4 theta_max clamping used to turn a quad wholly outside the FOV
        into a huge/degenerate boundary polygon. It must now yield nothing."""
        T = camera_room_pose(self.T_head, self.cam1)
        cam_pos = np.asarray(T.t, dtype=float)
        behind = np.array([[-1, 3.15, -1], [1, 3.15, -1], [1, 3.15, 1], [-1, 3.15, 1]], dtype=float)
        pc = T.inverse().apply(behind)
        # Flip the test quad across the camera so every corner is behind it.
        flipped = T.apply(np.stack([pc[:, 0], pc[:, 1], -np.abs(pc[:, 2]) - 1.0], axis=1))
        self.assertEqual(reproject_room_quads([flipped], self.cam1, T), [])
        del cam_pos

    def test_no_contour_covers_a_large_part_of_the_image(self):
        """Sweep quads around the camera: nothing may ever mask a big chunk."""
        T = camera_room_pose(self.T_head, self.cam1)
        area = float(self.cam1.width * self.cam1.height)
        rng = np.random.default_rng(0)
        for _ in range(200):
            c = rng.uniform(-6, 6, 2)
            h = rng.uniform(0.1, 0.5)
            q = np.array([[c[0]-h, 3.15, c[1]-h], [c[0]+h, 3.15, c[1]-h],
                          [c[0]+h, 3.15, c[1]+h], [c[0]-h, 3.15, c[1]+h]])
            for px in reproject_room_quads([q], self.cam1, T):
                self.assertLess(cv2.contourArea(px.reshape(-1, 1, 2)), 0.15 * area)

    def test_empty_or_none_input(self):
        T = camera_room_pose(self.T_head, self.cam1)
        self.assertEqual(reproject_room_quads(None, self.cam1, T), [])
        self.assertEqual(reproject_room_quads([], self.cam1, T), [])


if __name__ == "__main__":
    unittest.main()
