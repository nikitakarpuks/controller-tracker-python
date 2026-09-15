"""Regression tests for src/static_light_geometry.py's room-frame ray-casting
math -- Phase 1 of the mocap-based static-light (ceiling-lamp) exclusion
feature (see the approved implementation plan). Frame-composition math is
exactly the kind of change that has caused real, silent bugs in this project
before (see tests/test_pose_jump_rotation_convention.py) -- verified
numerically here before any offline map-building or live wiring is built on
top of it.

Uses stdlib unittest (pytest is not a declared dependency of this project).
Run with:  python3 -m unittest tests.test_static_light_geometry
"""
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.camera import Camera
from src.load_config import load_json_config
from src.static_light_geometry import camera_room_pose, pixel_to_room_ray, room_point_to_pixel
from src.transformations import Transform

_CALIB_PATH = Path(__file__).resolve().parent.parent / "data" / "cameras" / "calibration_basalt.json"

# A point ~1.8m straight ahead of camera 0's optical axis, expressed in the
# headset-IMU frame under an identity headset pose (derived once via
# cam.T_imu_cam.apply([0,0,1.8]) and hardcoded here for a deterministic,
# dependency-free test point that's confirmed to land near image center).
_POINT_ROOM = np.array([-0.371, 0.474, 1.717])


def _point_line_distance(point: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    """Perpendicular distance from `point` to the line through `origin` along
    unit `direction`."""
    v = point - origin
    t = np.dot(v, direction)
    closest = origin + t * direction
    return float(np.linalg.norm(point - closest))


def _headset_pose(yaw_deg: float, pos) -> Transform:
    R_room_headset = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    return Transform(R_room_headset, np.asarray(pos, dtype=np.float64))


class TestCameraRoomPose(unittest.TestCase):
    def setUp(self):
        cfg = load_json_config(str(_CALIB_PATH))
        self.camera = Camera(cfg, camera_idx=0)

    def test_matches_manual_composition(self):
        T_room_headsetImu = _headset_pose(20.0, [1.0, 0.5, 2.0])

        T_room_cam = camera_room_pose(T_room_headsetImu, self.camera)

        R_expected = T_room_headsetImu.R @ self.camera.T_imu_cam.R
        t_expected = T_room_headsetImu.R @ self.camera.T_imu_cam.t + T_room_headsetImu.t
        np.testing.assert_allclose(T_room_cam.R, R_expected, atol=1e-12)
        np.testing.assert_allclose(T_room_cam.t, t_expected, atol=1e-12)


class TestPixelRoomRayRoundTrip(unittest.TestCase):
    """A known room-frame point, projected from several distinct synthetic
    headset poses, must round-trip back to a ray that passes through it."""

    def setUp(self):
        cfg = load_json_config(str(_CALIB_PATH))
        self.camera = Camera(cfg, camera_idx=0)
        self.headset_poses = [
            _headset_pose(0.0, [0.0, 0.0, 0.0]),
            _headset_pose(8.0, [0.1, -0.05, 0.02]),
            _headset_pose(-6.0, [-0.15, 0.1, -0.03]),
        ]

    def test_round_trip_recovers_the_point_across_viewpoints(self):
        for T_room_headsetImu in self.headset_poses:
            T_room_cam = camera_room_pose(T_room_headsetImu, self.camera)
            px = room_point_to_pixel(self.camera, T_room_cam, _POINT_ROOM.reshape(1, 3))
            self.assertTrue(np.all(np.isfinite(px)))

            origin, dirs = pixel_to_room_ray(self.camera, T_room_cam, px)
            dist = _point_line_distance(_POINT_ROOM, origin, dirs[0])
            self.assertLess(
                dist, 1e-6,
                f"ray from headset pose t={T_room_headsetImu.t} misses the known point by {dist}m")

    def test_moving_headset_gives_angularly_distinct_rays(self):
        """Sanity precondition for the future voxel accumulator: if these
        synthetic headset poses didn't actually produce angularly distinct
        rays toward the same point, no triangulation-style confirmation would
        ever be possible -- that would be a test-data problem, not a
        geometry-code problem."""
        dirs = []
        for T_room_headsetImu in self.headset_poses[:2]:
            T_room_cam = camera_room_pose(T_room_headsetImu, self.camera)
            px = room_point_to_pixel(self.camera, T_room_cam, _POINT_ROOM.reshape(1, 3))
            _, d = pixel_to_room_ray(self.camera, T_room_cam, px)
            dirs.append(d[0])
        cos_angle = np.clip(np.dot(dirs[0], dirs[1]), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        self.assertGreater(angle_deg, 1.0)


class TestForwardProjectionNeverFoldsBackPastRpmax(unittest.TestCase):
    """Regression for a real bug found on live usage (cam0, right, frame 55
    of a real recording): a tracked lamp region's corner near the image's
    right edge had its reprojected pixel x1 DRIFT LEFT frame-over-frame
    instead of staying at/past the boundary, corrupting the region's
    exclusion contour into a non-bbox shape. Root cause: the kb4 forward
    mapping rho(theta) is only monotonic up to theta_max (camera.rpmax ==
    tan(theta_max), see src/camera.py's kb4_theta_max/kb4_rpmax) --
    cv2.fisheye.projectPoints has no awareness of this and folds a ray
    PAST theta_max back toward the image centre instead of continuing
    outward. room_point_to_pixel now clamps any such ray onto the
    theta_max cone (same azimuth, angle capped) before projecting."""

    def setUp(self):
        cfg = load_json_config(str(_CALIB_PATH))
        self.camera = Camera(cfg, camera_idx=0)
        self.T_room_cam = Transform(np.eye(3), np.zeros(3))  # identity: room frame == camera frame

    def test_pixel_x_is_monotonic_and_saturates_beyond_theta_max(self):
        # A family of room points at increasing angle off-axis (z=1 forward,
        # x growing), simulating a lamp corner sweeping toward/through the
        # image's right edge -- spans well past camera.rpmax (confirmed
        # below) so both sides of the turning point are exercised.
        tans = np.linspace(0.5, 6.0, 40)
        self.assertGreater(tans.max(), self.camera.rpmax,
                            "test setup: the sweep must cross camera.rpmax")
        pts_room = np.stack([tans, np.zeros_like(tans), np.ones_like(tans)], axis=1)

        px = room_point_to_pixel(self.camera, self.T_room_cam, pts_room)
        x = px[:, 0]

        # Never decreasing: once saturated, must hold or keep rising, never
        # fold back toward the image centre.
        diffs = np.diff(x)
        self.assertTrue(np.all(diffs >= -1e-6),
                         f"pixel x must never decrease as the ray sweeps further off-axis, got diffs={diffs}")

        # The clamped tail (well past rpmax) must all land at the SAME
        # saturated pixel value (same azimuth, angle capped to theta_max).
        beyond = tans > self.camera.rpmax * 1.2  # comfortably past the clamp threshold
        self.assertGreater(int(beyond.sum()), 5, "test setup: need several points comfortably past rpmax")
        np.testing.assert_allclose(x[beyond], x[beyond][0], atol=1e-2)

    def test_unclamped_forward_model_actually_folds_back(self):
        """Confirms the bug is real (not a misdiagnosis): calling the raw
        cv2-backed Camera.project_points directly (bypassing
        room_point_to_pixel's clamp) DOES fold back past rpmax -- otherwise
        this whole fix would be solving a non-problem."""
        tans = np.linspace(0.5, 6.0, 40)
        pts_room = np.stack([tans, np.zeros_like(tans), np.ones_like(tans)], axis=1)
        px_raw, _ = self.camera.project_points(pts_room, rvec=np.zeros(3), tvec=np.zeros(3))
        x_raw = px_raw[:, 0]
        self.assertLess(x_raw[-1], x_raw.max() - 10.0,
                         "test setup: the unpatched forward model must fold back for this sweep")

    def test_pixel_x_is_continuous_across_the_behind_camera_crossing(self):
        """Regression for a second, real bug found on live usage (cam2,
        left, frame ~40-44 of a real recording): a tracked region's corner
        swinging all the way to BEHIND the camera plane (z <= 0) as the
        headset kept turning produced sudden, non-monotonic pixel jumps in
        the reprojected bbox (observed: y0 347->277 across one frame, x0
        -47->+62 three frames later) despite the region's own room-space
        quad staying exactly frozen throughout -- a pure projection
        artifact. An earlier version of this fix only clamped rays still in
        front of the camera (z > 0, via an x/z-style ratio), leaving this
        case entirely unclamped. Sweeping theta from just under 90 degrees
        to just over it (crossing z=0) must stay continuous."""
        theta_max = np.arctan(self.camera.rpmax)
        self.assertLess(theta_max, np.pi / 2, "test setup: theta_max must be under 90 degrees for this crossing to be exercised")
        thetas = np.linspace(theta_max + 0.05, np.pi / 2 + 0.4, 60)  # sweeps straight through z=0
        x = np.sin(thetas)   # unit ray: (sin theta, 0, cos theta) -- z crosses zero mid-sweep
        z = np.cos(thetas)
        pts_room = np.stack([x, np.zeros_like(x), z], axis=1)

        px = room_point_to_pixel(self.camera, self.T_room_cam, pts_room)
        diffs = np.diff(px[:, 0])
        self.assertTrue(np.all(np.abs(diffs) < 5.0),
                         f"pixel x must stay continuous (no jump) as theta sweeps through 90 degrees, got diffs={diffs}")
        # Already-clamped (both sides of the crossing are past theta_max), so
        # the whole sweep should sit at the same saturated pixel value.
        np.testing.assert_allclose(px[:, 0], px[0, 0], atol=1e-2)


class TestMovingPointMissesTheOriginalRay(unittest.TestCase):
    """Complementary sanity check to the static-point round trip: a point
    displaced between two observations must NOT satisfy the first
    observation's ray -- baseline confirmation that the geometry isn't
    degenerate before anything downstream tries to use it to tell a static
    lamp apart from a moving controller LED."""

    def setUp(self):
        cfg = load_json_config(str(_CALIB_PATH))
        self.camera = Camera(cfg, camera_idx=0)
        self.T_room_cam = camera_room_pose(Transform(np.eye(3), np.zeros(3)), self.camera)

    def test_displaced_point_misses_the_original_ray(self):
        point_t1 = _POINT_ROOM + np.array([0.5, 0.0, 0.0])  # 0.5m lateral displacement

        px_t0 = room_point_to_pixel(self.camera, self.T_room_cam, _POINT_ROOM.reshape(1, 3))
        origin, dirs = pixel_to_room_ray(self.camera, self.T_room_cam, px_t0)

        dist = _point_line_distance(point_t1, origin, dirs[0])
        self.assertGreater(dist, 0.1)


if __name__ == "__main__":
    unittest.main()
