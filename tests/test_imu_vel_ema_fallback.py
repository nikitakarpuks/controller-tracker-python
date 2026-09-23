"""Regression coverage for the IMU-derived vel_ema fallback (src/controller.py:
_predicted_world_for_vel_ema / _imu_vel_ema_override / _vel_ema_with_imu_fallback).

Found 2026-09-09 investigating a real recording: under fast controller motion,
right after a fresh (re)acquisition, CameraTracker._predict_pose's translation
prediction (n=1 constant-position / n=2 fractional-velocity-from-2-noisy-
points) was bad enough to fail the warm proximity search's tight neighbourhood
radius on the next TWO frames running, forcing two extra expensive cold
brute-force (p3p_systematic) re-detects before enough real vision samples
(n=3) accumulated for _predict_pose's own linear-fit branch to average the
noise out. Fix: when the vision-only vel_ema is unavailable (tracker.vel_ema
is None) AND pose_history is still short (<=2), fall back to a camera-frame
velocity derived from the fusion filter's OWN accel+gyro world-frame
dead-reckoning (HeuristicPoseFusionFilter/PoseFusionFilter.predict(), already
validated elsewhere) instead of leaving _predict_pose to guess from
vision alone. Verified on the real recording: a frame that previously forced
a cold p3p_systematic retry now succeeds via warm proximity instead (one
fewer expensive brute-force search per fast-motion reacquisition).

Uses stdlib unittest (pytest is not a declared dependency of this project).
Run with:  python3 -m unittest tests.test_imu_vel_ema_fallback
"""
import unittest
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

from src.controller import (
    _predicted_world_for_vel_ema, _imu_vel_ema_override, _vel_ema_with_imu_fallback,
)
from src.transformations import Transform

_NS = 1_000_000_000


def _tracker(vel_ema=None, pose_history=()):
    t = Mock()
    t.vel_ema = vel_ema
    t.pose_history = list(pose_history)
    return t


def _pose_history_entry(tvec, ts_ns):
    return (np.zeros((3, 1), dtype=np.float32), np.asarray(tvec, dtype=np.float32), ts_ns)


class ImuVelEmaOverrideTests(unittest.TestCase):
    def test_none_predicted_world_returns_none(self):
        camera = Mock(T_world_cam=Transform(np.eye(3), np.zeros(3)))
        result = _imu_vel_ema_override(None, camera, [_pose_history_entry([0, 0, 0], 0)], _NS)
        self.assertIsNone(result)

    def test_empty_pose_history_returns_none(self):
        camera = Mock(T_world_cam=Transform(np.eye(3), np.zeros(3)))
        predicted = (np.eye(3), np.array([1.0, 0.0, 0.0]))
        result = _imu_vel_ema_override(predicted, camera, [], _NS)
        self.assertIsNone(result)

    def test_non_positive_dt_returns_none(self):
        camera = Mock(T_world_cam=Transform(np.eye(3), np.zeros(3)))
        predicted = (np.eye(3), np.array([1.0, 0.0, 0.0]))
        # target ts EQUAL to pose_history[0]'s own ts -> dt_s == 0
        result = _imu_vel_ema_override(predicted, camera, [_pose_history_entry([0, 0, 0], _NS)], _NS)
        self.assertIsNone(result)

    def test_identity_camera_gives_direct_world_rate(self):
        """Camera at the world origin, no rotation -- camera frame == world
        frame, so the resulting rate should be exactly
        (p_pred_world - tvec0) / dt_s with no rotation applied."""
        camera = Mock(T_world_cam=Transform(np.eye(3), np.zeros(3)))
        R_pred_world = np.eye(3)
        p_pred_world = np.array([1.0, 0.0, 0.0])
        ts0 = 0
        target_ts = int(0.5 * _NS)  # 0.5s later
        pose_history = [_pose_history_entry([0.0, 0.0, 0.0], ts0)]
        rate = _imu_vel_ema_override((R_pred_world, p_pred_world), camera, pose_history, target_ts)
        self.assertIsNotNone(rate)
        np.testing.assert_allclose(rate, [2.0, 0.0, 0.0], atol=1e-5)  # 1.0m / 0.5s

    def test_camera_rotation_is_correctly_applied(self):
        """Camera rotated 90deg about Z relative to world -- a world-frame
        +X world_cam offset must project onto the camera's own -Y axis (or
        +Y, depending on convention) via T_world_cam.inverse(), NOT come out
        unrotated -- this is the whole point of doing the conversion via a
        real Transform instead of just subtracting world positions."""
        R_world_cam = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        camera = Mock(T_world_cam=Transform(R_world_cam, np.zeros(3)))
        R_pred_world = np.eye(3)
        p_pred_world = np.array([1.0, 0.0, 0.0])
        ts0 = 0
        target_ts = int(1.0 * _NS)
        pose_history = [_pose_history_entry([0.0, 0.0, 0.0], ts0)]
        rate = _imu_vel_ema_override((R_pred_world, p_pred_world), camera, pose_history, target_ts)
        # Manually compute expected: T_cam_world = T_world_cam.inverse();
        # t_cam_ctrl_pred = T_cam_world.apply(p_pred_world)
        T_cam_ctrl_pred = Transform(R_world_cam, np.zeros(3)).inverse().compose(
            Transform(R_pred_world, p_pred_world))
        expected = (T_cam_ctrl_pred.t - np.zeros(3)) / 1.0
        np.testing.assert_allclose(rate, expected, atol=1e-5)
        # And confirm it's NOT the naive (wrong) unrotated answer.
        self.assertFalse(np.allclose(rate, [1.0, 0.0, 0.0], atol=1e-3))


class VelEmaWithImuFallbackTests(unittest.TestCase):
    def test_real_vel_ema_always_wins_never_overridden(self):
        """A populated vision-only vel_ema must never be touched by the IMU
        fallback, even if pose_history is short -- vision over IMU."""
        real_vel_ema = np.array([9.0, 9.0, 9.0], dtype=np.float32)
        tracker = _tracker(vel_ema=real_vel_ema, pose_history=[_pose_history_entry([0, 0, 0], 0)])
        camera = Mock(T_world_cam=Transform(np.eye(3), np.zeros(3)))
        predicted_world = (np.eye(3), np.array([100.0, 100.0, 100.0]))  # would give a wildly different rate
        result = _vel_ema_with_imu_fallback(tracker, predicted_world, camera, _NS)
        self.assertIs(result, real_vel_ema)

    def test_long_pose_history_never_falls_back_even_if_vel_ema_is_none(self):
        """n>2 real vision samples -- _predict_pose's own linear fit already
        averages out noise (see its own docstring); this must return the
        (None) vel_ema unchanged, NOT synthesize an IMU-derived one."""
        tracker = _tracker(vel_ema=None, pose_history=[
            _pose_history_entry([0, 0, 0], 0),
            _pose_history_entry([0, 0, 0], -1),
            _pose_history_entry([0, 0, 0], -2),
        ])
        camera = Mock(T_world_cam=Transform(np.eye(3), np.zeros(3)))
        predicted_world = (np.eye(3), np.array([5.0, 0.0, 0.0]))
        result = _vel_ema_with_imu_fallback(tracker, predicted_world, camera, _NS)
        self.assertIsNone(result)

    def test_falls_back_to_imu_when_vel_ema_none_and_pose_history_short(self):
        """The actual fix: vel_ema is None AND pose_history has only 1 real
        entry -- must use the IMU-derived rate instead of leaving
        _predict_pose to guess from a single constant-position sample."""
        tracker = _tracker(vel_ema=None, pose_history=[_pose_history_entry([0.0, 0.0, 0.0], 0)])
        camera = Mock(T_world_cam=Transform(np.eye(3), np.zeros(3)))
        predicted_world = (np.eye(3), np.array([1.0, 0.0, 0.0]))
        result = _vel_ema_with_imu_fallback(tracker, predicted_world, camera, int(0.5 * _NS))
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result, [2.0, 0.0, 0.0], atol=1e-5)

    def test_two_entries_still_eligible_for_fallback(self):
        tracker = _tracker(vel_ema=None, pose_history=[
            _pose_history_entry([0.0, 0.0, 0.0], 0),
            _pose_history_entry([0.0, 0.0, 0.0], -int(0.1 * _NS)),
        ])
        camera = Mock(T_world_cam=Transform(np.eye(3), np.zeros(3)))
        predicted_world = (np.eye(3), np.array([3.0, 0.0, 0.0]))
        result = _vel_ema_with_imu_fallback(tracker, predicted_world, camera, int(1.0 * _NS))
        self.assertIsNotNone(result)


class PredictedWorldForVelEmaTests(unittest.TestCase):
    def test_no_fusion_filter_never_calls_predict(self):
        trackers = [_tracker(vel_ema=None, pose_history=[])]
        result = _predicted_world_for_vel_ema(None, trackers, _NS)
        self.assertIsNone(result)

    def test_no_tracker_needs_it_predict_never_called(self):
        """Every tracker already has a real vel_ema or long enough history --
        predict() (real accel/gyro integration work) must not be called at
        all in the common steady-state case."""
        fusion_filter = Mock()
        trackers = [
            _tracker(vel_ema=np.zeros(3), pose_history=[_pose_history_entry([0, 0, 0], 0)]),
            _tracker(vel_ema=None, pose_history=[
                _pose_history_entry([0, 0, 0], 0),
                _pose_history_entry([0, 0, 0], -1),
                _pose_history_entry([0, 0, 0], -2),
            ]),
        ]
        result = _predicted_world_for_vel_ema(fusion_filter, trackers, _NS)
        self.assertIsNone(result)
        fusion_filter.predict.assert_not_called()

    def test_one_tracker_needs_it_predict_called_exactly_once(self):
        """Multiple cameras of the same controller sharing the identical
        world-frame prediction must not each trigger their own predict()
        call -- see this function's own docstring on why that would be
        wasteful, redundant work."""
        fusion_filter = Mock()
        sentinel = (np.eye(3), np.array([1.0, 2.0, 3.0]))
        fusion_filter.predict.return_value = sentinel
        trackers = [
            _tracker(vel_ema=np.zeros(3), pose_history=[_pose_history_entry([0, 0, 0], 0)]),  # doesn't need it
            _tracker(vel_ema=None, pose_history=[_pose_history_entry([0, 0, 0], 0)]),           # needs it
            _tracker(vel_ema=None, pose_history=[_pose_history_entry([0, 0, 0], 0)]),           # also needs it
        ]
        result = _predicted_world_for_vel_ema(fusion_filter, trackers, _NS)
        self.assertEqual(result, sentinel)
        fusion_filter.predict.assert_called_once_with(_NS)


if __name__ == "__main__":
    unittest.main()
