"""Loaders for the IMU-integration data sources under an euroc_recording's mav0/ root:
imu0 (HMD IMU), imu1/imu2 (left/right controller IMU), vio (headset VIO pose estimate),
and the controllers' factory IMU calibration (InertialSensors block in each controller's
config JSON, alongside the ControllerLeds block create_leds_from_config already reads).

Entry selection and calibration application match Monado's actual WMR driver
(src/xrt/drivers/wmr/wmr_config.c wmr_inertial_sensors_config_parse,
wmr_controller_hp.c wmr_controller_hp_packet_parse -- verified against a GitHub
mirror of Monado main; gitlab.freedesktop.org itself blocks automated fetches):

  - InertialSensors holds TWO Gyro + TWO Accelerometer entries per controller
    (Id=ICM20602 then Id=Undefined, in that array order). Monado's parser has no
    Id-based selection at all -- it just iterates the array and each same-typed
    entry overwrites the previous one, so whichever comes LAST in the array wins.
    For this file that's the Undefined-Id entries (index 1 below), not the
    ICM20602 ones -- confirmed from source, but plausibly an oversight in Monado
    rather than a deliberate choice, so treat as "this is what the code does",
    not necessarily "this is correct".
  - correct() applies mix then ADDS bias (mix @ raw + bias), matching
    math_matrix_3x3_transform_vec3 + math_vec3_accum(bias, imu.acc) in
    wmr_controller_hp_packet_parse (math_vec3_accum(a, b) means b += a, confirmed
    in m_api.h).
  - Rt is NOT applied in that raw-sample path in Monado (only mix+bias+an
    OXR-axis-remap quaternion are) -- Rt becomes a separate pose used downstream
    in pose composition. Kept un-composed here (T_rt) for the same reason;
    direction (body->imu vs imu->body) still isn't documented in the file itself.

Axis transform (sensor frame -> controller body/LED frame): a single, SHARED
transform for gyro and accel -- both live on the same physical chip package,
same mounting -- diag(1,-1,-1), a precise 180deg flip about X (X unchanged,
Y/Z reversed). Physically unremarkable: a chip mounted with two axes reversed
relative to the device's logical convention is a common, deliberate PCB-layout
choice, not a bug.

SUPERSEDED (2026-09-02) an earlier, per-sensor-DISTINCT transform this
docstring used to document:
  - gyro:  R_body_gyro  = diag(1,-1,1) @ T_rt.R.T   (transpose)
  - accel: R_body_accel = diag(1,-1,1) @ T_rt.R      (no transpose)
That version was itself validated by two independent checks (gyro:
cross-correlated against vision-derived angular velocity via
imu_vision_sync_check.py, ranked #1 of 16 transpose x sign-flip candidates;
accel: gravity-direction self-consistency via imu_accel_exhaustive_search.py)
-- but on euroc_recording_20260826173103_static_dark specifically, a decisive
re-check (full-recording gyro-vs-vision rotation error, not just a handful of
frames: 2509/2610 samples, left/right) found diag(1,-1,-1) alone gives median
0.342°/0.510° error, roughly 8x better than the old per-sensor transform's
2.644°/2.480° and better than every other transpose/flip combination tried.
Neither imu_vision_sync_check.py nor imu_accel_exhaustive_search.py were ever
committed to this repo (lost scratch scripts, confirmed via git log), so it's
unknown which recording that original validation ran against -- but given
this project's lag_ns constant was separately confirmed (git archaeology) to
have been measured on a DIFFERENT, older recording
(euroc_recording_20260729173447_still_easy) and only turned out to still be
correct for static_dark by luck, the far more likely explanation here is that
the old per-sensor transform was validated against that same older recording
and, unlike lag_ns, does NOT hold up on static_dark. Cross-validated
independently three ways in visualization/controller_calibration_for_basalt/
README.md findings 5-7 (a separate Claude session's vision+mocap bundle-
adjustment chain, that session's fresh EPnP-only reproduction, and this
project's own joint bias+rotation solve in bias_estimation_check.py).
Do not assume diag(1,-1,-1) generalizes to other controller hardware/
calibration files or recordings without re-running this check.

imu0 (the HMD's own IMU) needs NO axis transform at all -- it IS the reference
frame T_imu_cam/VIO are already expressed against, unlike the controllers
(unknown wireless-chip mounting orientation). Confirmed via the same
cross-correlation technique: identity transform gave 0.997 mean axis
correlation against VIO's own derived rotation (imu0_vio_sync_check.py),
alongside a tiny (~3ms) timing offset that is NOT currently corrected for
anywhere in this codebase (interpolate_vio's callers use raw VIO timestamps).
"""
import csv

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from src.transformations import Transform


def load_imu_csv(path):
    """EuRoC-style imu data.csv -> (t_ns int64[N], gyro float32[N,3] rad/s, accel float32[N,3] m/s^2)."""
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[float(x) for x in row] for row in reader]
    arr = np.array(rows, dtype=np.float64)
    t_ns = arr[:, 0].astype(np.int64)
    gyro = arr[:, 1:4].astype(np.float32)
    accel = arr[:, 4:7].astype(np.float32)
    return t_ns, gyro, accel


def load_vio_csv(path):
    """EuRoC-style vio data.csv -> (t_ns int64[N], position float32[N,3], quat_xyzw float32[N,4]).

    File stores quaternion as w,x,y,z; reordered here to x,y,z,w to match
    scipy.spatial.transform.Rotation's convention used elsewhere in this codebase
    (see src/camera.py's Camera.__init__).
    """
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[float(x) for x in row] for row in reader]
    arr = np.array(rows, dtype=np.float64)
    t_ns = arr[:, 0].astype(np.int64)
    position = arr[:, 1:4].astype(np.float32)
    quat_wxyz = arr[:, 4:8]
    quat_xyzw = np.roll(quat_wxyz, -1, axis=1).astype(np.float32)
    return t_ns, position, quat_xyzw


def interpolate_vio(t_query_ns: np.ndarray, t_vio: np.ndarray, position_vio: np.ndarray,
                     quat_xyzw_vio: np.ndarray):
    """Interpolate VIO's (sparse, 30Hz) headset-in-world pose to arbitrary query
    timestamps (t_query_ns must lie within [t_vio[0], t_vio[-1]] -- no extrapolation).
    SLERP for rotation, linear for position.

    Returns (R_interp (N,3,3), pos_interp (N,3)) -- a plain array pair, not a
    Transform: Transform.compose/inverse assume a single (3,3)/(3,) pose, not a
    batch, so batched composition is done explicitly by the caller (see
    imu_vision_sync_check-style scripts) instead of overloading Transform's contract.
    """
    if np.any(t_query_ns < t_vio[0]) or np.any(t_query_ns > t_vio[-1]):
        raise ValueError("interpolate_vio: query timestamp outside VIO's covered range")

    slerp = Slerp(t_vio, Rotation.from_quat(quat_xyzw_vio))
    R_interp = slerp(t_query_ns).as_matrix()

    pos_interp = np.empty((len(t_query_ns), 3), dtype=np.float64)
    for i in range(3):
        pos_interp[:, i] = np.interp(t_query_ns, t_vio, position_vio[:, i])

    return R_interp, pos_interp


def integrate_gyro_segment(t_gyro: np.ndarray, gyro_body: np.ndarray,
                            ts0, ts1: int):
    """Integrate calibrated, body-frame gyro samples over [ts0, ts1] into a single
    relative rotation (body frame at ts0 -> body frame at ts1), by composing each
    inter-sample step as a small-angle rotation (midpoint angular velocity x dt).

    Returns a (3,3) rotation matrix, or None if ts0 is None, the window is
    degenerate (ts1 <= ts0), or it falls outside [t_gyro[0], t_gyro[-1]] (no
    extrapolation -- same no-extrapolation policy as interpolate_vio).

    The returned matrix is meant to be right-composed onto a body-frame pose
    (R_new = R_old @ R_rel), matching CameraTracker._predict_pose's own
    rel_rvecs convention (R_0.T @ R_i) -- see its docstring for why that's the
    correct composition side for a body-frame-relative delta.
    """
    if ts0 is None or ts1 <= ts0:
        return None
    if ts0 < t_gyro[0] or ts1 > t_gyro[-1]:
        return None

    mid_mask = (t_gyro > ts0) & (t_gyro < ts1)
    ts = np.concatenate(([ts0], t_gyro[mid_mask], [ts1])).astype(np.int64)
    omega = np.empty((len(ts), 3), dtype=np.float64)
    for i in range(3):
        omega[:, i] = np.interp(ts, t_gyro, gyro_body[:, i])

    R_rel = Rotation.identity()
    for i in range(len(ts) - 1):
        dt = (ts[i + 1] - ts[i]) / 1e9
        w_mid = 0.5 * (omega[i] + omega[i + 1])
        R_rel = R_rel * Rotation.from_rotvec(w_mid * dt)
    return R_rel.as_matrix()


def gyro_preint_residual(t_gyro: np.ndarray, gyro_body: np.ndarray, ts0, ts1: int,
                          R0: np.ndarray, R1: np.ndarray, bias0: np.ndarray = None,
                          bias1: np.ndarray = None, bias_random_walk_std: np.ndarray = None):
    """Gyro preintegration factor for one reference-node gap [ts0, ts1] --
    r_gyro(k) = Log(DeltaR_gyro(ts0,ts1)^-1 . (R0^-1 R1)) -- wraps the
    already-validated integrate_gyro_segment as a residual generator between
    consecutive REFERENCE-NODE timestamps (e.g. vision-frame cadence) rather
    than per-sample.

    R0/R1 (3,3): two orientations in a common, non-rotating reference frame
    at ts0/ts1 -- gyro measures angular velocity in an inertial frame, so
    these must NOT be headset-relative vision poses (those rotate with the
    headset -- see src/mocap_data.world_pose, which is what removes that).
    They must also share gyro_body's own frame (controller LED/body frame,
    see load_and_calibrate_controller_imu) -- i.e. the UN-bridged world-frame
    vision pose, not compare_vision_mocap.py's mocap-accel-IMU-frame bridge
    output.

    bias0 (3,) is subtracted from gyro_body before integration -- a
    first-order correction, exact under the same constant-bias-over-the-
    segment assumption integrate_gyro_segment's midpoint rule already makes
    about gyro noise. Defaults to zero (no bias correction/optimization yet).

    If bias1 and bias_random_walk_std (3,) are both given, also returns the
    bias random-walk residual r_bias = (bias1-bias0)/bias_random_walk_std/
    sqrt(dt) -- the other half of a standard IMU preintegration factor;
    unused until per-node bias states exist (joint batch solve).

    Returns (r_gyro (3,) rad or None, dt_s or None, r_bias (3,) or None) --
    None for r_gyro/dt if the segment falls outside gyro coverage or is
    degenerate (see integrate_gyro_segment)."""
    if bias0 is None:
        bias0 = np.zeros(3)
    R_gyro = integrate_gyro_segment(t_gyro, gyro_body - bias0, ts0, ts1)
    if R_gyro is None:
        return None, None, None
    R_vision_rel = R0.T @ R1
    r_gyro = Rotation.from_matrix(R_gyro.T @ R_vision_rel).as_rotvec()
    dt = (ts1 - ts0) / 1e9
    r_bias = None
    if bias1 is not None and bias_random_walk_std is not None:
        r_bias = (bias1 - bias0) / bias_random_walk_std / np.sqrt(dt)
    return r_gyro, dt, r_bias


def _lever_arm_correction(ts: np.ndarray, t_gyro: np.ndarray, gyro_body: np.ndarray,
                           r: np.ndarray) -> np.ndarray:
    """Per-sample body-frame correction (N,3) subtracted from a raw accel reading to
    recover the acceleration of the body origin vision tracks, given the
    accelerometer's lever arm r (3,) from that origin (rigid-body kinematics --
    see accel_lever_arm_solve.py's module docstring for the derivation this
    mirrors, and Finding: 2026-09-02 lever-arm confirmation in
    visualization/controller_calibration_for_basalt/README.md for where r itself
    comes from):

        correction(t) = alpha(t) x r + omega(t) x (omega(t) x r)

    omega(t) is gyro_body interpolated onto ts (same np.interp approach the
    caller already uses for accel samples). alpha(t) (angular acceleration) is
    its numerical derivative: 3-point central difference at interior ts, one-
    sided at the two endpoints -- ts is typically only a handful of samples
    spanning one vision-frame gap (~15-30ms), so a higher-order scheme isn't
    worth the complexity."""
    omega = np.empty((len(ts), 3), dtype=np.float64)
    for i in range(3):
        omega[:, i] = np.interp(ts, t_gyro, gyro_body[:, i])

    ts_s = ts.astype(np.float64) / 1e9
    alpha = np.empty_like(omega)
    alpha[0] = (omega[1] - omega[0]) / (ts_s[1] - ts_s[0])
    alpha[-1] = (omega[-1] - omega[-2]) / (ts_s[-1] - ts_s[-2])
    if len(ts) > 2:
        alpha[1:-1] = (omega[2:] - omega[:-2]) / (ts_s[2:] - ts_s[:-2])[:, None]

    r_b = np.broadcast_to(r, omega.shape)
    return np.cross(alpha, r_b) + np.cross(omega, np.cross(omega, r_b))


def integrate_accel_segment(t_accel: np.ndarray, accel_body: np.ndarray, ts0, ts1: int,
                             R0: np.ndarray, R1: np.ndarray, g_world: np.ndarray,
                             t_gyro: np.ndarray = None, gyro_body: np.ndarray = None,
                             r: np.ndarray = None):
    """World-frame velocity change (Delta_v, m/s) over [ts0, ts1] from raw
    (mix+bias corrected, body-frame) accel samples.

    LEVER ARM: pass t_gyro/gyro_body (the same controller's calibrated gyro
    stream) and r (3,) -- the accelerometer's body-frame offset from the body
    origin vision tracks, e.g. accel.Rt composed with gyro.Rt^-1's translation,
    see _lever_arm_correction's docstring -- to correct for the accelerometer
    NOT being co-located with that origin. r=None (default) keeps the original
    LEVER-ARM-FREE behavior (accelerometer assumed co-located) -- Step 3's
    sign/convention sanity check was deliberately run this way first, before
    lever-arm complexity was introduced, so a bug wouldn't get misattributed;
    now that the lever arm is confirmed (see module docstring reference above),
    callers should pass it.

    a_world(t) = R(t) @ (accel_body(t) - lever_arm_correction(t)) + g_world,
    trapezoidal-integrated over [ts0, ts1] (same midpoint-rule spirit as
    integrate_gyro_segment). R(t) is SLERP-interpolated between the two known
    endpoint orientations R0 (at ts0) and R1 (at ts1) -- adequate for the short
    (single vision-frame-gap) windows this is used over (Step 1 found ~2-3deg
    median inter-frame rotation on this recording).

    Returns delta_v (3,) world frame, or None if ts0 is None, the window is
    degenerate (ts1 <= ts0), or it falls outside [t_accel[0], t_accel[-1]]
    (no extrapolation -- same policy as integrate_gyro_segment)."""
    if ts0 is None or ts1 <= ts0:
        return None
    if ts0 < t_accel[0] or ts1 > t_accel[-1]:
        return None

    mid_mask = (t_accel > ts0) & (t_accel < ts1)
    ts = np.concatenate(([ts0], t_accel[mid_mask], [ts1])).astype(np.int64)
    acc = np.empty((len(ts), 3), dtype=np.float64)
    for i in range(3):
        acc[:, i] = np.interp(ts, t_accel, accel_body[:, i])
    if r is not None:
        acc = acc - _lever_arm_correction(ts, t_gyro, gyro_body, r)

    frac = (ts - ts0) / (ts1 - ts0)
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([R0, R1])))
    R_t = slerp(frac).as_matrix()

    a_world = np.einsum("nij,nj->ni", R_t, acc) + g_world
    dv = np.zeros(3)
    for i in range(len(ts) - 1):
        dt = (ts[i + 1] - ts[i]) / 1e9
        dv += 0.5 * (a_world[i] + a_world[i + 1]) * dt
    return dv


def integrate_accel_to_position(t_accel: np.ndarray, accel_body: np.ndarray, ts0, ts1: int,
                                 R0: np.ndarray, R1: np.ndarray, v0: np.ndarray, g_world: np.ndarray,
                                 t_gyro: np.ndarray = None, gyro_body: np.ndarray = None,
                                 r: np.ndarray = None):
    """SHORT-HORIZON forward integration: given a known velocity v0 at ts0,
    double-trapezoidal-integrates raw accel over [ts0, ts1] to predict the
    world-frame position change Delta_p (m) -- the actual "propagate with
    accel between vision frames, then let vision correct" use this data is
    for, as opposed to integrate_accel_segment's Delta_v (used there only for
    a sign/correlation sanity check). The distinction matters: over a SHORT
    window (single vision-frame gap, ~15-30ms here) integration SHRINKS a
    calibration error by ~dt^2 rather than amplifying it -- e.g. a 2.8 m/s^2
    gravity error integrated twice over 15ms is only ~0.3mm, well under
    vision's own ~3-5mm noise floor -- unlike a a_center_world estimate
    built by DIFFERENTIATING vision position (Step 3 sub-stage 2's approach),
    where the same 15-30ms window AMPLIFIES mm-level position noise into
    several m/s^2. So this function does not need a precisely-calibrated
    lever arm/bias/gravity to be useful; a rough estimate suffices.

    LEVER ARM: same t_gyro/gyro_body/r convention as integrate_accel_segment
    (see its docstring and _lever_arm_correction) -- r=None keeps the
    original LEVER-ARM-FREE behavior; g_world here should come from a
    low-motion-frame bootstrap (or similar), not the whole-recording average
    that sub-stage 1 found biased by the omitted lever arm.

    Same SLERP-interpolated-rotation approach as integrate_accel_segment.

    Returns delta_p (3,) world frame, or None under the same conditions
    integrate_accel_segment returns None for."""
    if ts0 is None or ts1 <= ts0:
        return None
    if ts0 < t_accel[0] or ts1 > t_accel[-1]:
        return None

    mid_mask = (t_accel > ts0) & (t_accel < ts1)
    ts = np.concatenate(([ts0], t_accel[mid_mask], [ts1])).astype(np.int64)
    acc = np.empty((len(ts), 3), dtype=np.float64)
    for i in range(3):
        acc[:, i] = np.interp(ts, t_accel, accel_body[:, i])
    if r is not None:
        acc = acc - _lever_arm_correction(ts, t_gyro, gyro_body, r)

    frac = (ts - ts0) / (ts1 - ts0)
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([R0, R1])))
    R_t = slerp(frac).as_matrix()

    a_world = np.einsum("nij,nj->ni", R_t, acc) + g_world
    v = np.empty((len(ts), 3), dtype=np.float64)
    v[0] = v0
    dp = np.zeros(3)
    for i in range(len(ts) - 1):
        dt = (ts[i + 1] - ts[i]) / 1e9
        v[i + 1] = v[i] + 0.5 * (a_world[i] + a_world[i + 1]) * dt
        dp += 0.5 * (v[i] + v[i + 1]) * dt
    return dp


def slice_imu_to_window(t: np.ndarray, data: np.ndarray, ts_lo: int, ts_hi: int, pad_ns: int = 200_000_000):
    """(t_slice, data_slice) restricted to roughly [ts_lo-pad_ns, ts_hi+pad_ns], via
    searchsorted (t is already sorted ascending -- true for load_and_calibrate_
    controller_imu's output). Performance fix, not a math change: integrate_gyro_
    segment/integrate_accel_segment/integrate_accel_to_position each mask/interpolate
    over the WHOLE array passed in, independent of how short [ts0, ts1] actually is --
    fine for a single call, but expensive when called many times per short window
    (bias_estimation_check.py's solver Jacobian; PoseFusionFilter.predict, called every
    frame against the full-recording-length live gyro/accel arrays once wired into
    live tracking -- found in code review, see src/pose_fusion.py). pad_ns keeps
    enough margin that every gap's mid-sample interpolation still has real samples on
    both sides. Single shared copy -- was previously duplicated in
    bias_estimation_check.py, which now imports this instead."""
    lo = max(t[0], ts_lo - pad_ns)
    hi = min(t[-1], ts_hi + pad_ns)
    i0 = int(np.searchsorted(t, lo, side="left"))
    i1 = int(np.searchsorted(t, hi, side="right"))
    return t[i0:i1], data[i0:i1]


def predict_world_pose(t_gyro, gyro_body, t_accel, accel_body, g_world, lever_arm,
                        ts0, ts1, R0, p0, v0):
    """Single-endpoint BLIND dead-reckoning prediction at ts1, given a known state
    (R0, p0, v0) at ts0 -- no peeking at any ground truth at ts1. The single-hop
    primitive dead_reckon_dense (below) samples repeatedly to draw a dense curve
    (validated against 13 real tracking-loss gaps, see
    visualization/controller_calibration_for_basalt/README.md finding 11); this
    is just the final point, for callers (e.g. PoseFusionFilter.predict,
    src/pose_fusion.py) that only need the endpoint, not a dense intermediate curve.
    Bias fixed at 0 throughout (this session's own validated finding -- bias choice
    doesn't measurably change real dead-reckoning outcomes on this hardware).

    Callers with long-lived, full-recording-length t_gyro/t_accel arrays (anything
    called repeatedly against a short [ts0, ts1] window, e.g. PoseFusionFilter.predict)
    should pre-slice via slice_imu_to_window first -- this function itself does not,
    since single-call use sites (offline scripts) don't need it and slicing has its
    own (small) overhead.

    Returns (R1 (3,3), p1 (3,)), or None if gyro/accel coverage doesn't span
    [ts0, ts1] (see integrate_gyro_segment/integrate_accel_to_position)."""
    R_gyro = integrate_gyro_segment(t_gyro, gyro_body, ts0, ts1)
    if R_gyro is None:
        return None
    R1 = R0 @ R_gyro
    dp = integrate_accel_to_position(t_accel, accel_body, ts0, ts1, R0, R1, v0, g_world,
                                      t_gyro=t_gyro, gyro_body=gyro_body, r=lever_arm)
    if dp is None:
        return None
    return R1, p0 + dp


def predict_headset_relative_pose(t_gyro, gyro_body, t_accel, accel_body, g_world_abs, lever_arm,
                                   ts0, ts1, R_hc0, p_hc0, v_hc0,
                                   R_wh0, p_wh0, omega_h0, v_wh0, R_wh1, p_wh1):
    """Same contract as predict_world_pose, EXCEPT (R_hc0, p_hc0, v_hc0) are HEADSET-RELATIVE --
    this pipeline's own "world" frame is really the headset-IMU rig frame (Camera.T_world_cam is
    a fixed rig extrinsic, see src/camera.py), not any inertial frame, so predict_world_pose's own
    blind dead-reckoning silently mis-treats the headset as non-rotating/non-accelerating during
    [ts0,ts1]. This wraps it with an exact frame conversion at the boundary instead of touching
    its (validated) internals: lift the headset-relative state to the absolute frame using the
    caller-supplied headset ego-motion, run predict_world_pose completely unchanged, then project
    the result back to headset-relative using the headset's absolute pose at ts1.

    R_wh0/p_wh0, R_wh1/p_wh1: the headset's own absolute pose (e.g. mocap-derived, see
    src.mocap_data.world_pose) at ts0/ts1. omega_h0 (rad/s): headset body-frame angular velocity
    at ts0 (see src.mocap_data.headset_angular_velocity). v_wh0 (m/s): headset WORLD-frame linear
    velocity at ts0 (see src.mocap_data.headset_linear_velocity). g_world_abs: gravity already
    expressed in the ABSOLUTE frame (NOT the headset-relative g_world predict_world_pose normally
    takes -- rotating a headset-relative gravity estimate into this frame, if that's your source,
    is the caller's responsibility; this function does no such rotation itself).

    Velocity-lift derivation: differentiating p_wc(t) = R_wh(t) @ p_hc(t) + p_wh(t) using this
    codebase's own body-frame angular-velocity convention (dR/dt = R @ [omega]_x -- the
    continuous-time limit of integrate_gyro_segment's own R_new = R_old @ R_rel right-composition)
    gives v_wc = R_wh @ (v_hc + omega_h x p_hc) + v_wh -- the cross term is the "arm's-length
    lever" effect of the headset itself spinning while the controller sits at a nonzero
    headset-relative offset; omitting it (as naively reusing v_hc0 as an absolute velocity would)
    is exactly the kind of error this function exists to avoid.

    Returns (R_hc1, p_hc1), or None under the same conditions predict_world_pose returns None for
    (propagated, not duplicated)."""
    R_wc0 = R_wh0 @ R_hc0
    p_wc0 = R_wh0 @ p_hc0 + p_wh0
    v_wc0 = R_wh0 @ (v_hc0 + np.cross(omega_h0, p_hc0)) + v_wh0

    predicted = predict_world_pose(t_gyro, gyro_body, t_accel, accel_body, g_world_abs, lever_arm,
                                    ts0, ts1, R_wc0, p_wc0, v_wc0)
    if predicted is None:
        return None
    R_wc1, p_wc1 = predicted

    R_hc1 = R_wh1.T @ R_wc1
    p_hc1 = R_wh1.T @ (p_wc1 - p_wh1)
    return R_hc1, p_hc1


def dead_reckon_dense(t_gyro, gyro_body, t_accel, accel_body, g_world, lever_arm,
                       t0, t1, R0, p0, v0, sample_every_n: int = 1):
    """Dense, BLIND (no true-endpoint peeking) dead-reckoning through [t0, t1]:
    one (t, R, p) sample per raw IMU timestamp in the gap (or every
    sample_every_n-th, for very dense IMU streams / many repeated calls), each
    computed via a fresh predict_world_pose call over [t0, t_i] -- the same
    single-hop primitive PoseFusionFilter.predict uses, so a future change to
    the single-hop formula can't silently diverge from what this draws. Shared
    by visualize_position_orientation.py's offline gap-bridging plot and
    PoseFusionFilter.predict_dense's live debug-visualization curve (moved
    here from the former's own inlined copy, found in code review -- see this
    project's history of exactly this kind of duplication).

    COST: O(samples^2) -- each sample re-integrates the whole [t0, t_i]
    prefix from scratch rather than incrementally extending the previous
    sample's result (predict_world_pose's own docstring notes the same
    tradeoff). Fine for a one-shot offline plot or a debug-viz curve bounded
    to a few hundred samples; NOT suitable for anything called every frame
    over an unbounded window -- callers wanting the latter should raise
    sample_every_n or cap the sample count some other way first.

    Returns (ts (N,) int64, positions (N,3), rotations: list of (3,3), one
    per ts) -- empty arrays/list if no sample in [t0, t1] has coverage."""
    mask = (t_gyro > t0) & (t_gyro < t1)
    inner_ts = t_gyro[mask][::sample_every_n]
    sample_ts = np.concatenate(([t0], inner_ts, [t1])).astype(np.int64)
    sample_ts = np.unique(sample_ts)

    out_ts, out_p, out_R = [], [], []
    for t_i in sample_ts:
        if t_i == t0:
            R_i, p_i = R0, p0
        else:
            predicted = predict_world_pose(t_gyro, gyro_body, t_accel, accel_body, g_world, lever_arm,
                                            t0, t_i, R0, p0, v0)
            if predicted is None:
                continue
            R_i, p_i = predicted
        out_ts.append(t_i)
        out_p.append(p_i)
        out_R.append(R_i)
    if len(out_ts) <= 1:
        # Only the t0 anchor itself was ever appended -- every real sample beyond
        # it failed predict_world_pose's own coverage check (e.g. a genuine IMU
        # dropout wider than slice_imu_to_window's pad). A 1-point "path" isn't a
        # real prediction; signal unavailability the same way predict_world_pose/
        # PoseFusionFilter.predict() do (found in review) instead of returning a
        # trivially-"successful" single point frozen at the start position.
        return np.array([]), np.zeros((0, 3)), []
    return np.array(out_ts), np.array(out_p), out_R


def accel_preint_residual(t_accel: np.ndarray, accel_body: np.ndarray, ts0, ts1: int,
                           R0: np.ndarray, R1: np.ndarray, p0: np.ndarray, p1: np.ndarray,
                           v0: np.ndarray, v1: np.ndarray, g_world: np.ndarray,
                           bias0: np.ndarray = None, bias1: np.ndarray = None,
                           bias_random_walk_std: np.ndarray = None,
                           t_gyro: np.ndarray = None, gyro_body: np.ndarray = None,
                           r: np.ndarray = None):
    """Accel preintegration factor for one reference-node gap [ts0, ts1] --
    the accel counterpart to gyro_preint_residual (Step 4: joint batch solve
    with per-node bias states). Gives TWO residuals instead of one, since
    accel constrains both velocity and (via v0) position:

        r_vel(k) = integrate_accel_segment(ts0,ts1) - (v1 - v0)
        r_pos(k) = integrate_accel_to_position(ts0,ts1, v0=v0) - (p1 - p0)

    r_pos is what actually ties the free velocity state v0 to the world-frame
    vision positions p0/p1 -- r_vel alone would leave v0 unconstrained by
    anything but the bias random-walk/anchor terms.

    bias0 (3,) is subtracted from accel_body before integration -- same
    first-order-correction convention as gyro_preint_residual's bias0.
    Defaults to zero (no bias correction).

    LEVER ARM: t_gyro/gyro_body/r are forwarded as-is to integrate_accel_
    segment/integrate_accel_to_position -- see their docstrings and
    _lever_arm_correction. r should be a FIXED, known constant (this
    project's factory-JSON-derived accel/gyro lever arm -- confirmed
    2026-09-02, see visualization/controller_calibration_for_basalt/
    README.md finding 5), not a per-node unknown for this solve to estimate:
    it's a static hardware property, unlike bias, which genuinely drifts and
    is what this joint solve's per-node states are for. r=None keeps the
    original lever-arm-free behavior.

    If bias1 and bias_random_walk_std (3,) are both given, also returns the
    bias random-walk residual r_bias = (bias1-bias0)/bias_random_walk_std/
    sqrt(dt) -- identical formula to gyro_preint_residual's, applied to the
    accel bias chain instead.

    Returns (r_vel (3,) m/s or None, r_pos (3,) m or None, dt_s or None,
    r_bias (3,) or None) -- None for r_vel/r_pos/dt if the segment falls
    outside accel coverage or is degenerate (see integrate_accel_segment)."""
    if bias0 is None:
        bias0 = np.zeros(3)
    accel_corrected = accel_body - bias0
    dv = integrate_accel_segment(t_accel, accel_corrected, ts0, ts1, R0, R1, g_world,
                                  t_gyro=t_gyro, gyro_body=gyro_body, r=r)
    if dv is None:
        return None, None, None, None
    dp = integrate_accel_to_position(t_accel, accel_corrected, ts0, ts1, R0, R1, v0, g_world,
                                      t_gyro=t_gyro, gyro_body=gyro_body, r=r)
    r_vel = dv - (v1 - v0)
    r_pos = dp - (p1 - p0)
    dt = (ts1 - ts0) / 1e9
    r_bias = None
    if bias1 is not None and bias_random_walk_std is not None:
        r_bias = (bias1 - bias0) / bias_random_walk_std / np.sqrt(dt)
    return r_vel, r_pos, dt, r_bias


def load_T_imu_cam(cfg, camera_idx: int = 0) -> Transform:
    """cfg: an already-loaded camera calibration JSON (see load_json_config).
    Reads cfg['value0']['T_imu_cam'][camera_idx] directly -- unlike src/camera.py's
    Camera class, this does NOT depend on the config's extrinsics_convention setting;
    it always returns whatever transform is literally stored under that JSON key.
    Only meaningful if this particular calibration file was produced under the true
    "T_imu_cam" convention (cam0 entry non-identity) -- e.g. data/cameras/backup/
    mateosss.reverbg2v1.kleineinzeigen.bslt.json, NOT the currently-active
    kb4_calib_data_tuned.json, whose cam0 T_imu_cam entry is identity (that file
    was tuned under T_cam0_camN and never carried a real imu0 relationship).
    """
    e = cfg["value0"]["T_imu_cam"][camera_idx]
    R_imu_cam = Rotation.from_quat([e["qx"], e["qy"], e["qz"], e["qw"]]).as_matrix()
    t_imu_cam = np.array([e["px"], e["py"], e["pz"]], dtype=np.float64)
    return Transform(R_imu_cam, t_imu_cam)


class ControllerImuAxisCalib:
    """Factory calibration for one IMU axis type (gyro or accel) of one controller.

    bias0 / mix0 are the T=0 (reference-temperature) term of each model's
    per-axis/per-element temperature polynomial (BiasTemperatureModel: 3 axes x
    4 coeffs; MixingMatrixTemperatureModel: 3x3 elements x 4 coeffs). Real operating
    temperature is unknown -- these recordings have no temperature channel -- so
    correct() is only a T=0 approximation, not full temperature compensation.
    """

    def __init__(self, entry: dict):
        rt = entry["Rt"]
        # T_rt: rotation+translation as given in the file. Direction (body->imu or
        # imu->body) is not documented here the way Camera.T_imu_cam is -- do not
        # assume a direction until confirmed.
        R_rt = np.array(rt["Rotation"], dtype=np.float64).reshape(3, 3)
        t_rt = np.array(rt["Translation"], dtype=np.float64)
        self.T_rt = Transform(R_rt, t_rt)

        self.bias0 = np.array(entry["BiasTemperatureModel"][0::4], dtype=np.float64)
        self.mix0 = np.array(entry["MixingMatrixTemperatureModel"][0::4], dtype=np.float64).reshape(3, 3)
        self.noise_std = np.array(entry["Noise"][:3], dtype=np.float64)
        self.bias_uncertainty = np.array(entry["BiasUncertainty"], dtype=np.float64)

    def correct(self, raw: np.ndarray) -> np.ndarray:
        """raw: (N,3) raw samples in sensor frame -> T=0 mixing+bias corrected (N,3).
        mix then ADD bias -- matches Monado's wmr_controller_hp_packet_parse (see
        module docstring), not the more common "subtract bias" convention."""
        return (self.mix0 @ raw.T).T + self.bias0


class ControllerImuCalib:
    def __init__(self, gyro: ControllerImuAxisCalib, accel: ControllerImuAxisCalib):
        self.gyro = gyro
        self.accel = accel


def create_imu_calib_from_config(cfg, entry_index: int = 1) -> ControllerImuCalib:
    """cfg: an already-loaded controller config JSON (see load_json_config), same
    input create_leds_from_config takes. entry_index selects which of the two
    Gyro/Accelerometer entries in InertialSensors to use -- default 1 matches
    Monado's actual (last-array-entry-wins) behavior, i.e. the Id=Undefined
    entries, not the Id=ICM20602 ones -- see module docstring for the caveat.
    """
    sensors = cfg["CalibrationInformation"]["InertialSensors"]
    gyro_entries = [s for s in sensors if s["SensorType"] == "CALIBRATION_InertialSensorType_Gyro"]
    accel_entries = [s for s in sensors if s["SensorType"] == "CALIBRATION_InertialSensorType_Accelerometer"]
    return ControllerImuCalib(
        gyro=ControllerImuAxisCalib(gyro_entries[entry_index]),
        accel=ControllerImuAxisCalib(accel_entries[entry_index]),
    )


# Sensor-frame -> body-frame transform, SAME for gyro and accel (both live on the
# same physical chip package, same mounting) -- a precise 180deg flip about X (X
# unchanged, Y/Z reversed). Superseded the old per-sensor-distinct _Y_FLIP @
# Rt.R^(±1) transform this module's docstring used to document (see docstring's
# "SUPERSEDED" note for the full story and the decisive full-recording evidence).
_DIAG_FLIP = np.diag([1.0, -1.0, -1.0])


def load_and_calibrate_controller_imu(imu_path, controller_cfg: dict, lag_ns: int = 0,
                                       entry_index: int = 1):
    """Load one controller's raw imu*.csv and apply the full correction chain:
    mix+bias calibration, the confirmed sensor->body axis transform (_DIAG_FLIP
    -- see module docstring), and an optional clock-offset correction (lag_ns,
    added to the raw timestamps -- e.g. Stage 1's measured controller<->camera
    offset).

    Single source for this chain -- main.py and visualize_imu.py both call this
    rather than each keeping their own copy of the transform logic.

    Returns (t_ns int64[N], gyro_body float64[N,3], accel_body float64[N,3]).
    """
    t_imu, gyro_raw, accel_raw = load_imu_csv(imu_path)
    calib = create_imu_calib_from_config(controller_cfg, entry_index=entry_index)

    gyro_corr = calib.gyro.correct(gyro_raw.astype(np.float64))
    gyro_body = (_DIAG_FLIP @ gyro_corr.T).T

    accel_corr = calib.accel.correct(accel_raw.astype(np.float64))
    accel_body = (_DIAG_FLIP @ accel_corr.T).T

    return t_imu + lag_ns, gyro_body, accel_body


# "Near-stationary" gate for a low-motion-frame gravity bootstrap -- shared canonical
# constant (found duplicated as a re-typed magic number in 2 places in code review:
# accel_short_horizon_check.py's own module-level copy, which now imports this instead,
# and this same file's LiveGravityEstimator, below).
LOW_OMEGA_THRESH_RAD_S = 0.5


class LiveGravityEstimator:
    """Online counterpart to accel_short_horizon_check.low_motion_bootstrap_g_world
    (repo root) for LIVE operation, which has no pre-recorded pose log to run the
    batch version against -- "world frame" here is a fixed-but-not-gravity-aligned
    per-session camera-calibration extrinsic (Camera.T_world_cam), so g_world (gravity
    expressed in it) is a genuine per-session unknown, exactly the same problem the
    offline scripts solve with a full pose log. Same low-motion gate (|gyro| <=
    omega_thresh) and formula (-mean(R_world_ctrl @ accel_sample)) as the batch
    version, accumulated incrementally on every accepted commit instead of over a
    whole pre-recorded recording -- chosen over a fixed startup calibration window
    because it converges whenever the first ~20 low-motion frames occur, anywhere in
    the session, rather than requiring the user to hold still at boot specifically."""

    def __init__(self, omega_thresh: float = LOW_OMEGA_THRESH_RAD_S, min_samples: int = 20):
        self._sum = np.zeros(3, dtype=np.float64)
        self._n = 0
        self._omega_thresh = omega_thresh
        self._min_samples = min_samples

    def observe(self, R_world_ctrl: np.ndarray, gyro_sample: np.ndarray, accel_sample: np.ndarray) -> None:
        if np.linalg.norm(gyro_sample) > self._omega_thresh:
            return
        self._sum += R_world_ctrl @ accel_sample
        self._n += 1

    @property
    def g_world(self):
        if self._n < self._min_samples:
            return None
        return -self._sum / self._n

    @property
    def n_samples(self) -> int:
        return self._n


# Mocap room-frame gravity, ADDITIVE convention matching LiveGravityEstimator.g_world
# (a_world = R(t) @ accel_body + g_world). Empirically validated, not assumed: rotating
# raw headset-IMU (imu0) accelerometer samples into mocap room frame via the mocap-
# tracked headset orientation at each timestamp, over 7852 low-motion samples (|gyro|
# <= LOW_OMEGA_THRESH_RAD_S) spread across a real walk_medium recording, gives a mean
# accel of [-0.005, 9.997, 0.155] m/s^2 -- 99.99% of magnitude on the Y axis (0.01%
# off-axis leakage) -- confirming the mocap room's Y axis is gravity-aligned by the
# room's own manual leveling, to within ~0.06 degrees. Unlike the rig-frame g_world
# (which rotates WITH the headset and genuinely has no session-constant value), the
# room/absolute frame's gravity vector is a true constant once that leveling is trusted
# -- no 20-sample convergence wait needed. Replaces g_world_estimator_abs entirely (see
# main.py / src/controller.py / src/pose_fusion_heuristic.py -- that estimator is
# commented out, not deleted, in case a future recording's room leveling needs
# reverifying against this same empirical check before trusting this constant again).
MOCAP_ROOM_G_WORLD = np.array([0.0, -9.81, 0.0])
