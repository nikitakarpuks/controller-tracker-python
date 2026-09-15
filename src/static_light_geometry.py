"""Pure room-frame ray-casting geometry for the static-light (ceiling-lamp)
exclusion feature. No mocap-specific concepts here -- callers supply a
T_room_headsetImu Transform (see src/headset_pose_source.py) from whatever
pose source is currently authoritative.

Naming: this codebase's existing Camera.T_world_cam/T_imu_cam already means
"headset-IMU rig frame" (src/camera.py). To avoid colliding with that, every
transform here that involves the NEW absolute mocap-room frame is named
T_room_*, never T_world_*.

The undistort/redistort pair below is the same fisheye-correct (KB4
bisection, not a pinhole linear approximation) pattern probe_static_blobs.py
already validated for VIO-derotation residuals -- reused here composed with
an arbitrary room-frame headset pose instead of a pure inter-frame rotation
delta.
"""
from typing import Tuple

import numpy as np

from src.camera import Camera
from src.transformations import Transform


def camera_room_pose(T_room_headsetImu: Transform, camera: Camera) -> Transform:
    """T_room_cam(t) = T_room_headsetImu(t) . camera.T_imu_cam (the camera's
    own fixed rig extrinsic)."""
    return T_room_headsetImu.compose(camera.T_imu_cam)


def pixel_to_room_ray(camera: Camera, T_room_cam: Transform,
                       px: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """px: (N,2) distorted pixel coords -> (origin (3,), directions (N,3)
    unit vectors), both in the room frame. Uses the camera's real per-model
    inverse (KB4 bisection for fisheye, cv2.undistortPoints for radtan8) via
    Camera.undistort_points."""
    px = np.asarray(px, dtype=np.float64).reshape(-1, 2)
    norm = camera.undistort_points(px)  # (N,2) normalized cam-frame (x/z, y/z)
    dirs_cam = np.hstack([norm, np.ones((norm.shape[0], 1))])
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)
    dirs_room = (T_room_cam.R @ dirs_cam.T).T
    origin_room = T_room_cam.t
    return origin_room, dirs_room


def room_point_to_pixel(camera: Camera, T_room_cam: Transform,
                         pts_room: np.ndarray) -> np.ndarray:
    """(N,3) room-frame points -> (N,2) pixels, via the real forward model
    (cv2.fisheye.projectPoints for kb4, cv2.projectPoints for radtan8).
    Inverse of pixel_to_room_ray's geometry.

    For kb4 (fisheye): the forward mapping rho(theta) is only monotonically
    increasing up to theta_max (see src/camera.py's kb4_theta_max/kb4_rpmax
    -- camera.rpmax == tan(theta_max), already used by src/_visibility.py's
    in-frame check). cv2.fisheye.projectPoints has no awareness of this and
    will happily project a ray PAST theta_max, where rho(theta) has already
    turned over and is now DECREASING -- folding the projected pixel back
    toward the image centre instead of continuing outward. Confirmed on live
    usage (cam0, right, frame 55, a tracked lamp region near the image's
    right edge): as the region's corner drifted further right in reality
    (larger theta), its reprojected x1 started drifting LEFT once past this
    angular limit, corrupting the exclusion contour into a self-intersecting
    shape that isn't a valid bbox anymore.

    Fix: clamp any ray whose angle from the optical axis (theta, measured
    via atan2 over the FULL [0, pi] range -- not x/z or y/z, which blow up
    or flip sign once z crosses zero) exceeds theta_max onto the theta_max
    cone first (same azimuth, angle capped) before projecting -- mirrors
    Camera.undistort_points' own rpmax_px clamp for the inverse direction,
    so the projected pixel instead saturates at the image's true edge and
    never folds back or jumps. Using atan2 (rather than an earlier version
    of this fix that only handled 0 < theta, still in front of the camera)
    matters: a region whose corner swings all the way to BEHIND the camera
    plane (z <= 0) as the headset keeps turning is a real, observed case
    (cam2, left, frame ~40-44 of a real recording) -- not a same-frame edge
    case, but a real region already tracked for dozens of frames whose
    corner keeps moving as the headset yaws. The earlier x/z-ratio version
    left z<=0 entirely unclamped, so cv2.fisheye.projectPoints' own
    behaviour for those points (undefined/discontinuous once theta exceeds
    the model's real domain) produced sudden, non-monotonic pixel jumps
    (observed: y0 347->277 across one frame, x0 -47->+62 three frames
    later) despite the region's own room-space quad staying exactly frozen
    the whole time -- i.e. a pure projection artifact, not a data/merge
    bug. Clamping via atan2 handles the full range continuously: as theta
    sweeps through 90 degrees and beyond (genuinely behind the camera), the
    clamped ray saturates at theta_max in the correct azimuthal direction
    and stays there, with no discontinuity at any point along the way."""
    pts_room = np.asarray(pts_room, dtype=np.float64).reshape(-1, 3)
    pts_cam = T_room_cam.inverse().apply(pts_room)
    if camera.is_fisheye and camera.rpmax > 0.0:
        theta_max = np.arctan(camera.rpmax)
        x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
        theta = np.arctan2(np.hypot(x, y), z)   # angle from +Z axis, range [0, pi] -- robust for z <= 0
        beyond = theta > theta_max
        if beyond.any():
            phi = np.arctan2(y[beyond], x[beyond])
            pts_cam = pts_cam.copy()
            pts_cam[beyond, 0] = np.sin(theta_max) * np.cos(phi)
            pts_cam[beyond, 1] = np.sin(theta_max) * np.sin(phi)
            pts_cam[beyond, 2] = np.cos(theta_max)
    pts_px, _ = camera.project_points(pts_cam, rvec=np.zeros(3), tvec=np.zeros(3))
    return pts_px
