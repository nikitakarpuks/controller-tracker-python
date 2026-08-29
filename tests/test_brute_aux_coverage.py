"""Regression coverage for the cross-camera coverage-gate activation
(PoseSearcher.brute_search_tier's other_cameras_blobs mechanism, src/pose_search.py).

Builds two real Camera objects (same kb4 lens/intrinsics as data/cameras/kb4_calib.json,
distinct hand-chosen extrinsics so they view the controller from different angles) and a
real ControllerModel (data/controllers/right_controller_A85K6081930636R.json), places the
controller in an edge-on orientation relative to camera A where only 5 of its 32 LEDs are
geometrically visible there -- exactly the "thin, single-camera-only" fit pattern that let
a coincidental brute-force P3P candidate pass balanced_coverage trivially before this
mechanism was activated (see controller.py's TrackingSystem.update_cold_batch /
ControllerTracker.update, and pose_search.py's brute_search_tier aux block).

Three scenarios exercise the same ground-truth pose/blobs in camera A, varying only what
other_cameras_blobs supplies for camera B:
  1. other_cameras_blobs=None            -- today's pre-activation baseline: found.
  2. corroborating camera-B blobs        -- real multi-camera support: found, with
                                             aux_inliers reflecting the extra evidence.
  3. contradicting camera-B blobs        -- camera B predicts several LEDs should be
                                             visible there but none of its blobs match
                                             (shifted far away) -- the SAME camera-A fit
                                             now fails the coverage gate: not found.

Scenario 3 vs 1 is the actual regression test: identical camera-A data, only the aux
evidence differs, and the outcome flips -- proving the gate is doing real cross-camera
work, not just passing data through inertly.

Uses stdlib unittest rather than pytest since pytest is not a declared dependency of this
project (see tests/test_cold_conflict_resolution.py).
Run with:  python3 -m unittest tests.test_brute_aux_coverage
"""
import json
import unittest
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.camera import Camera
from src.controller import ControllerModel, create_leds_from_config
from src.pose_search import PoseSearcher
from src.transformations import Transform

REPO_ROOT = Path(__file__).resolve().parent.parent
CALIB_PATH = REPO_ROOT / "data" / "cameras" / "kb4_calib.json"
CTRL_PATH = REPO_ROOT / "data" / "controllers" / "right_controller_A85K6081930636R.json"

# LEDs geometrically visible from camera A (edge-on: rotate the controller 90 deg about Y)
# and from camera B (a second view ~20 deg off camera A, also seeing 5 more LEDs beyond the
# 5 camera A sees) -- found by sweeping orientations and camera-B placement until both
# cameras' projections land inside their [0,640)x[0,480) frame; see the exploration this
# file's design was validated against.
_VIS_IDS_A = np.array([2, 7, 9, 11, 13])
_VIS_IDS_B = np.array([2, 7, 9, 11, 13, 22, 23, 24, 27, 29])
_CAM_B_LOOK_DEG = -20.556045219583467  # rotation about Y so camera B's +Z axis points at
                                        # the controller from its offset position (0.15,0,-0.05)


def _make_two_camera_calib():
    with open(CALIB_PATH) as f:
        calib = json.load(f)
    intr0 = calib["value0"]["intrinsics"][0]
    res0 = calib["value0"]["resolution"][0]

    t_a = {"px": 0.0, "py": 0.0, "pz": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    qb = R.from_euler('y', _CAM_B_LOOK_DEG, degrees=True).as_quat()
    t_b = {"px": 0.15, "py": 0.0, "pz": -0.05,
           "qx": float(qb[0]), "qy": float(qb[1]), "qz": float(qb[2]), "qw": float(qb[3])}
    return {"value0": {"T_imu_cam": [t_a, t_b], "intrinsics": [intr0, intr0], "resolution": [res0, res0]}}


class BruteAuxCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = _make_two_camera_calib()
        cls.cam_a = Camera(cfg, camera_idx=0)
        cls.cam_b = Camera(cfg, camera_idx=1)

        with open(CTRL_PATH) as f:
            ctrl_cfg = json.load(f)
        cls.model = ControllerModel(create_leds_from_config(ctrl_cfg), "right")
        cls.pose_searcher = PoseSearcher(cls.cam_a, cls.model, geometry_cfg={}, matching_cfg={})

        # Ground-truth controller pose: edge-on to camera A (rotated 90 deg about Y),
        # 0.35 m along camera A's own optical axis (world frame == camera A frame here).
        R_ctrl = R.from_euler('xy', [0, 90], degrees=True).as_matrix().astype(np.float32)
        t_ctrl = np.array([0.0, 0.0, 0.35], dtype=np.float32)
        T_world_ctrl = Transform(R_ctrl, t_ctrl)
        T_camB_ctrl = cls.cam_b.T_world_cam.inverse().compose(T_world_ctrl)

        rvec_a, _ = cv2.Rodrigues(R_ctrl)
        rvec_b, _ = cv2.Rodrigues(T_camB_ctrl.R.astype(np.float32))
        blobs_a, _ = cls.cam_a.project_points(cls.model.positions[_VIS_IDS_A], rvec_a, t_ctrl)
        blobs_b, _ = cls.cam_b.project_points(cls.model.positions[_VIS_IDS_B], rvec_b,
                                               T_camB_ctrl.t.astype(np.float32))
        cls.blobs_a = blobs_a.astype(np.float32)
        cls.blobs_b = blobs_b.astype(np.float32)

    def _run(self, other_cameras_blobs):
        mask = np.ones(len(self.blobs_a), dtype=bool)
        state = self.pose_searcher.new_brute_state(
            self.blobs_a, pose_prior=None, other_cameras_blobs=other_cameras_blobs,
            blob_mask=mask, occluders_per_cam=None,
        )
        for tier_idx in range(len(self.pose_searcher._c_brute_depth_tiers)):
            self.pose_searcher.brute_search_tier(state, tier_idx)
            if state.strong_found:
                break
        return self.pose_searcher.finalize_brute_state(state)

    def test_no_aux_data_finds_ground_truth(self):
        sol = self._run(None)
        self.assertIsNotNone(sol)
        self.assertEqual(sol["inliers"], 5)
        self.assertEqual(sol["aux_inliers"], 0)
        self.assertLess(sol["error"], 0.01)

    def test_corroborating_aux_data_found_with_aux_support(self):
        sol = self._run([(self.cam_b, self.blobs_b, None)])
        self.assertIsNotNone(sol)
        self.assertEqual(sol["inliers"], 5)
        self.assertEqual(sol["aux_inliers"], len(_VIS_IDS_B))
        self.assertLess(sol["error"], 0.01)

    def test_contradicting_aux_data_rejects_the_same_camera_a_fit(self):
        # Same camera-A ground truth blobs; camera B still predicts the same 10 LEDs
        # visible, but its "detected" blobs are shifted far away so none match --
        # the identical camera-A-only fit that succeeded above must now be rejected.
        blobs_b_shifted = self.blobs_b.copy()
        blobs_b_shifted[:, 0] += 150.0
        sol = self._run([(self.cam_b, blobs_b_shifted, None)])
        self.assertIsNone(sol)


if __name__ == "__main__":
    unittest.main()
