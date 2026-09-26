"""Cameras whose pairs the returned fused pose does not explain are dropped from the candidate (inliers, aux evidence,
claimed blobs), so a single-camera fusion fallback no longer drags along other cameras' wrong pairs.
Real case: static_hard frame 5872 -- pose from cam2 (7 pairs) but cam0/cam1 pairs (5+5, 336-464 px off) were still
attached (n_inliers 17, quality 1.0), drawn in the visualization with huge errors, and registered as claimed blobs."""
import unittest
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from src.controller import ControllerTracker
from src.pose_search import fused_pose_consistent_cameras
from src.transformations import Transform

K = np.array([[300.0, 0, 320.0], [0, 300.0, 240.0], [0, 0, 1.0]])
LEDS = np.array([[0.04, 0, 0], [-0.04, 0.01, 0], [0, 0.04, 0.02], [0, -0.04, -0.01], [0.02, 0.02, 0.04],
                 [-0.02, -0.02, -0.04], [0.03, -0.03, 0.03], [-0.03, 0.03, -0.03]], dtype=np.float32)


class Cam:
    def __init__(self, idx, R, t):
        self.camera_idx = idx
        self.T_world_cam = Transform(R, np.asarray(t, float))
        self.camera_matrix = K
        self.dist_coeffs = np.zeros(5)
        self.is_fisheye = False

    def project_points(self, pts3d, rvec, tvec):
        px, jac = cv2.projectPoints(np.asarray(pts3d, np.float32).reshape(-1, 1, 3), np.asarray(rvec, np.float64),
                                    np.asarray(tvec, np.float64).reshape(3, 1), K, np.zeros(5))
        return px.reshape(-1, 2), jac


def _blobs(cam, T):
    Tcc = cam.T_world_cam.inverse().compose(T)
    rv, _ = cv2.Rodrigues(Tcc.R)
    px, _ = cam.project_points(LEDS, rv, Tcc.t)
    return px


T_RIGHT = Transform(Rotation.from_euler("xyz", [10, 20, 5], degrees=True).as_matrix(), np.array([0.05, 0.0, 0.5]))
T_WRONG = Transform(Rotation.from_euler("xyz", [150, -40, 100], degrees=True).as_matrix(), np.array([-0.30, 0.2, 0.6]))
CAMS = {0: Cam(0, np.eye(3), [0, 0, 0]),
        1: Cam(1, Rotation.from_euler("y", -20, degrees=True).as_matrix(), [0.3, 0, 0.05]),
        2: Cam(2, Rotation.from_euler("y", 25, degrees=True).as_matrix(), [-0.3, 0, 0.1])}


def _fuse_in(spec):
    """spec: {cam_id: (pose the camera's blobs were generated from, n_pairs, error, confidence)}"""
    out = []
    for cid, (T, n, err, conf) in spec.items():
        out.append(dict(camera=CAMS[cid], blobs=_blobs(CAMS[cid], T), pairs=[(i, i) for i in range(n)],
                        T_world_ctrl=T, error=err, confidence=conf))
    return out


class ConsistentCamerasTests(unittest.TestCase):
    def test_all_consistent_keeps_everything(self):
        kept, rms = fused_pose_consistent_cameras(_fuse_in({0: (T_RIGHT, 6, .05, .3), 2: (T_RIGHT, 7, .06, .3)}),
                                                  T_RIGHT, LEDS)
        self.assertEqual(kept, {0, 2})
        self.assertLess(max(rms.values()), 1e-3)

    def test_camera_contradicting_the_returned_pose_is_dropped(self):
        ins = _fuse_in({0: (T_WRONG, 5, .059, .118), 1: (T_WRONG, 5, .7, 0.0), 2: (T_RIGHT, 7, .091, .299)})
        kept, rms = fused_pose_consistent_cameras(ins, T_RIGHT, LEDS)
        self.assertEqual(kept, {2})
        self.assertGreater(rms[0], 50.0)
        self.assertGreater(rms[1], 50.0)

    def test_threshold_is_configurable_and_empty_pairs_are_consistent(self):
        ins = _fuse_in({0: (T_RIGHT, 6, .05, .3), 2: (T_RIGHT, 0, .06, .3)})
        ins[0]["blobs"] = ins[0]["blobs"] + 2.0          # 2 px offset on both axes -> rms ~2.83 px
        self.assertEqual(fused_pose_consistent_cameras(ins, T_RIGHT, LEDS, max_reproj_px=3.5)[0], {0, 2})
        self.assertEqual(fused_pose_consistent_cameras(ins, T_RIGHT, LEDS, max_reproj_px=2.0)[0], {2})


def _tracker(cfg=None):
    t = object.__new__(ControllerTracker)
    t.cameras = CAMS
    t.trackers = {cid: SimpleNamespace(model=SimpleNamespace(positions=LEDS)) for cid in CAMS}
    t._matching_cfg = cfg or {}
    t.ctrl_name = "left_controller"
    return t


def _cam_solutions(spec):
    return [dict(cam_id=cid, solution=dict(assignment=[(i, i) for i in range(n)], T_world_ctrl=T, error=err,
                                            confidence=conf, method="p3p_systematic"))
            for cid, (T, n, err, conf) in spec.items()], {cid: _blobs(CAMS[cid], T) for cid, (T, *_) in spec.items()}


class ComputeFusedSolutionTests(unittest.TestCase):
    def test_single_camera_fallback_drops_the_inconsistent_cameras_pairs(self):
        spec = {0: (T_WRONG, 5, .059, .118), 1: (T_WRONG, 5, .7, 0.0), 2: (T_RIGHT, 7, .091, .299)}
        sols, obs = _cam_solutions(spec)
        sol = _tracker()._compute_fused_solution(sols, obs)
        np.testing.assert_allclose(sol["T_world_ctrl"].t, T_RIGHT.t, atol=1e-9)   # seed (cam2), see fallback fix
        self.assertEqual(sol["primary_cam"], 2)
        self.assertEqual(sol["fused_cam_ids"], [2])
        self.assertEqual(sol["aux_assignments"], {})
        self.assertEqual(sol["aux_cameras"], [])
        self.assertEqual(set(sol["camera_importance"]), {2})
        self.assertEqual(set(sol["camera_method"]), {2})
        self.assertEqual(set(sol["dropped_cam_rms_px"]), {0, 1})
        self.assertEqual(len(sol["assignment"]), 7)

    def test_consistent_cameras_are_all_kept_as_before(self):
        spec = {0: (T_RIGHT, 6, .05, .3), 2: (T_RIGHT, 7, .06, .3)}
        sols, obs = _cam_solutions(spec)
        sol = _tracker()._compute_fused_solution(sols, obs)
        self.assertEqual(sol["fused_cam_ids"], [0, 2])
        self.assertEqual(set(sol["aux_assignments"]), {0})          # primary = cam2 (most pairs); cam0 is aux
        self.assertEqual(sol["dropped_cam_rms_px"], {})

    def test_primary_is_chosen_among_consistent_cameras(self):
        # cam0 has the most pairs but its pairs contradict the returned pose -> must not be the anchor
        spec = {0: (T_WRONG, 8, .05, .01), 2: (T_RIGHT, 6, .09, .5)}
        sols, obs = _cam_solutions(spec)
        sol = _tracker()._compute_fused_solution(sols, obs)
        self.assertEqual(sol["primary_cam"], 2)


if __name__ == "__main__":
    unittest.main()
