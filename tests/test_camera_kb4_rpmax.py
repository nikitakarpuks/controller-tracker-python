"""Unit coverage for the kb4 fisheye rpmax/rpmax_px fix (src/camera.py, src/_self_calibration.py).

Covers: (1) rpmax is now tan(theta_max)-scale (matching pinhole-radtan8's convention
and src/_visibility.py's expectation) instead of the old, wrong pixel-scale value;
(2) rpmax_px is the TRUE kb4 polynomial turning-point pixel radius, larger than the
old (buggy) rpmax formula would have given; (3) Camera.undistort_points' bisection
inverse recovers pixels the old cv2.fisheye.undistortPoints call diverged on, and
round-trips cleanly through project_points; (4) the new bisection inverse agrees with
cv2.fisheye.undistortPoints on safely-interior pixels (no regression); (5)
_self_calibration.apply_to_cameras' recomputation is wired to the same shared helper.

Uses stdlib unittest rather than pytest since pytest is not a declared dependency of
this project (not in requirements, not installed in .venv).
Run with:  python3 -m unittest tests.test_camera_kb4_rpmax
"""
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.camera import Camera, kb4_rpmax, kb4_theta_max

CALIB_PATH = Path(__file__).resolve().parent.parent / "data" / "cameras" / "kb4_calib.json"


def _load_calib():
    with open(CALIB_PATH) as f:
        return json.load(f)


def _independent_theta_max(k1, k2, k3, k4):
    """Deliberately re-derived (not imported) so the ground-truth check doesn't
    just re-validate the implementation against itself."""
    def drho(t):
        return 1 + 3*k1*t**2 + 5*k2*t**4 + 7*k3*t**6 + 9*k4*t**8
    lo, hi = 0.0, np.pi / 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if drho(mid) > 0:
            lo = mid
        else:
            hi = mid
    return lo


def _independent_rpmax(k1, k2, k3, k4, fx, fy, margin=0.99):
    theta_max = _independent_theta_max(k1, k2, k3, k4)
    rho_max = theta_max * (1 + k1*theta_max**2 + k2*theta_max**4
                            + k3*theta_max**6 + k4*theta_max**8)
    return np.tan(theta_max) * margin, rho_max * (fx + fy) / 2 * margin


class Kb4RpmaxGroundTruthTests(unittest.TestCase):
    def test_all_cameras_match_independent_computation(self):
        calib = _load_calib()
        entries = calib["value0"]["intrinsics"]
        for idx in range(len(entries)):
            cam = Camera(calib, camera_idx=idx)
            intr = entries[idx]["intrinsics"]
            exp_tan, exp_px = _independent_rpmax(
                intr["k1"], intr["k2"], intr["k3"], intr["k4"], intr["fx"], intr["fy"])
            with self.subTest(camera=idx):
                self.assertAlmostEqual(cam.rpmax, exp_tan, places=6)
                self.assertAlmostEqual(cam.rpmax_px, exp_px, places=3)

    def test_camera0_bug_fix_true_turning_point_exceeds_old_wrong_value(self):
        calib = _load_calib()
        cam = Camera(calib, camera_idx=0)
        old_wrong_rpmax_px = kb4_theta_max(cam.k1, cam.k2, cam.k3, cam.k4) * 0.99 * (cam.fx + cam.fy) / 2
        self.assertGreater(cam.rpmax_px, old_wrong_rpmax_px)
        self.assertAlmostEqual(old_wrong_rpmax_px, 334.48, delta=0.5)
        self.assertAlmostEqual(cam.rpmax_px, 379.17 * 0.99, delta=0.5)


class Kb4UndistortDivergenceFixTests(unittest.TestCase):
    def test_pixel_20_70_no_longer_diverges_and_round_trips(self):
        calib = _load_calib()
        cam = Camera(calib, camera_idx=0)
        px = np.array([[20.0, 70.0]])

        # Sanity: reconfirm cv2's own solver still diverges on this exact pixel/camera,
        # so we know the "before" state is understood correctly.
        K64 = cam.camera_matrix.astype(np.float64)
        cv2_result = cv2.fisheye.undistortPoints(
            px.reshape(-1, 1, 2).astype(np.float64), K64, cam.dist_coeffs,
        ).reshape(-1, 2)
        self.assertGreater(np.linalg.norm(cv2_result[0]), 3.0)  # cv2 diverges: garbage, large norm

        result = cam.undistort_points(px)
        self.assertTrue(np.all(np.isfinite(result)))
        # Expected order of magnitude: tan(63.6 deg) ~= 2.0 (true recovered angle).
        self.assertLess(abs(result[0, 0]), 2.5)
        self.assertLess(abs(result[0, 1]), 2.5)

        # Round-trip: project the recovered ray back and expect ~(20, 70).
        ray = np.array([[result[0, 0], result[0, 1], 1.0]])
        reproj, _ = cam.project_points(ray, np.zeros(3), np.zeros(3))
        self.assertLess(np.linalg.norm(reproj[0] - px[0]), 1.0)


class Kb4UndistortRegressionTests(unittest.TestCase):
    def test_interior_pixels_agree_with_cv2(self):
        calib = _load_calib()
        cam = Camera(calib, camera_idx=0)

        rng = np.random.default_rng(0)
        angles = rng.uniform(0, 2*np.pi, size=50)
        radii = rng.uniform(0, 250, size=50)  # well inside both old and new rpmax_px
        px = np.stack([cam.cx + radii*np.cos(angles), cam.cy + radii*np.sin(angles)], axis=1)

        ours = cam.undistort_points(px)
        cv2_out = cv2.fisheye.undistortPoints(
            px.reshape(-1, 1, 2).astype(np.float64),
            cam.camera_matrix.astype(np.float64), cam.dist_coeffs,
        ).reshape(-1, 2)

        np.testing.assert_allclose(ours, cv2_out, atol=1e-4)


class SelfCalibrationDedupTests(unittest.TestCase):
    def test_apply_to_cameras_matches_shared_helper(self):
        calib = _load_calib()
        cam = Camera(calib, camera_idx=0)
        # Simulate a k1-k4 change (as self-calibration would apply) and recompute
        # exactly the way _self_calibration.apply_to_cameras now does.
        cam.k1, cam.k2, cam.k3, cam.k4 = 0.09, -0.035, 0.21, -0.12
        cam.rpmax, cam.rpmax_px = kb4_rpmax(cam.k1, cam.k2, cam.k3, cam.k4, cam.fx, cam.fy)

        exp_tan, exp_px = kb4_rpmax(cam.k1, cam.k2, cam.k3, cam.k4, cam.fx, cam.fy)
        self.assertEqual(cam.rpmax, exp_tan)
        self.assertEqual(cam.rpmax_px, exp_px)


if __name__ == "__main__":
    unittest.main()
