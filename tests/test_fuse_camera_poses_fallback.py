"""fuse_camera_poses single-camera fallback: when the joint fit is much worse than the best single camera's own
solve (the per-camera solutions are inconsistent), the camera returned must be the SEED (highest confidence, lowest
error breaks ties), not simply the lowest-error one. Real case: static_hard frame 5872, left controller."""
import unittest
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from src.pose_search import fuse_camera_poses
from src.transformations import Transform

K = np.array([[300.0, 0, 320.0], [0, 300.0, 240.0], [0, 0, 1.0]])
LEDS = np.array([[0.04, 0, 0], [-0.04, 0.01, 0], [0, 0.04, 0.02], [0, -0.04, -0.01], [0.02, 0.02, 0.04],
                 [-0.02, -0.02, -0.04], [0.03, -0.03, 0.03], [-0.03, 0.03, -0.03]], dtype=np.float32)


def _cam(idx, R, t):
    return SimpleNamespace(camera_idx=idx, T_world_cam=Transform(R, np.asarray(t, float)),
                           camera_matrix=K, dist_coeffs=np.zeros(5), is_fisheye=False)


def _blobs(cam, T_world_ctrl):
    Tcc = cam.T_world_cam.inverse().compose(T_world_ctrl)
    rv, _ = cv2.Rodrigues(Tcc.R)
    px, _ = cv2.projectPoints(LEDS.reshape(-1, 1, 3), rv, Tcc.t.reshape(3, 1), K, np.zeros(5))
    return px.reshape(-1, 2)


def _solution(cam, T, n_pairs, error, confidence):
    return dict(camera=cam, blobs=_blobs(cam, T), pairs=[(i, i) for i in range(n_pairs)],
                T_world_ctrl=T, error=error, confidence=confidence)


T_RIGHT = Transform(Rotation.from_euler("xyz", [10, 20, 5], degrees=True).as_matrix(), np.array([0.05, 0.0, 0.5]))
T_WRONG = Transform(Rotation.from_euler("xyz", [150, -40, 100], degrees=True).as_matrix(), np.array([-0.30, 0.2, 0.6]))
CAM_A = _cam(0, np.eye(3), [0.0, 0.0, 0.0])
CAM_B = _cam(2, Rotation.from_euler("y", 25, degrees=True).as_matrix(), [-0.3, 0.0, 0.1])


class FuseFallbackTests(unittest.TestCase):
    def test_inconsistent_solutions_return_the_confident_seed_not_the_min_error_one(self):
        wrong = _solution(CAM_A, T_WRONG, 5, 0.059, 0.118)     # tiny error, few pairs, low confidence
        right = _solution(CAM_B, T_RIGHT, 7, 0.091, 0.299)     # slightly larger error, more pairs, confident
        T, err = fuse_camera_poses([wrong, right], LEDS)
        np.testing.assert_allclose(T.t, T_RIGHT.t, atol=1e-9)
        self.assertAlmostEqual(err, 0.091)

    def test_order_of_the_solutions_does_not_matter(self):
        wrong = _solution(CAM_A, T_WRONG, 5, 0.059, 0.118)
        right = _solution(CAM_B, T_RIGHT, 7, 0.091, 0.299)
        T, _ = fuse_camera_poses([right, wrong], LEDS)
        np.testing.assert_allclose(T.t, T_RIGHT.t, atol=1e-9)

    def test_equal_confidence_still_returns_the_lowest_error_solution(self):
        a = _solution(CAM_A, T_WRONG, 5, 0.059, 0.0)
        b = _solution(CAM_B, T_RIGHT, 7, 0.091, 0.0)
        T, err = fuse_camera_poses([a, b], LEDS)
        np.testing.assert_allclose(T.t, T_WRONG.t, atol=1e-9)   # seed == min error when confidences tie
        self.assertAlmostEqual(err, 0.059)

    def test_consistent_solutions_use_the_joint_fit_not_the_fallback(self):
        a = _solution(CAM_A, T_RIGHT, 6, 0.05, 0.3)
        b = _solution(CAM_B, T_RIGHT, 7, 0.06, 0.3)
        T, err = fuse_camera_poses([a, b], LEDS)
        np.testing.assert_allclose(T.t, T_RIGHT.t, atol=1e-3)
        self.assertLess(err, 0.5)

    def test_single_camera_returned_unchanged(self):
        s = _solution(CAM_A, T_RIGHT, 6, 0.05, 0.3)
        T, err = fuse_camera_poses([s], LEDS)
        self.assertIs(T, s["T_world_ctrl"])


if __name__ == "__main__":
    unittest.main()
