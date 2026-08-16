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

Axis transform (sensor frame -> controller body/LED frame), resolved empirically,
NOT from documentation -- and NOT the same for gyro and accel, despite both
living on the same physical ICM20602 package:
  - gyro:  R_body_gyro  = diag(1,-1,1) @ T_rt.R.T   (transpose)
  - accel: R_body_accel = diag(1,-1,1) @ T_rt.R      (no transpose)
  Both use the same (1,-1,1) Y-flip (consistent with a documented WMR-internal
  Y-down vs. consumer/OpenXR Y-up convention difference), but the accelerometer
  needs T_rt AS-IS while the gyro needs its transpose -- confirmed by two
  independent checks, not assumed:
    * gyro: cross-correlated against vision-derived controller angular velocity
      (imu_vision_sync_check.py) -- transpose+flip ranked #1 of all 16
      transpose x sign-flip combinations.
    * accel: gravity-direction self-consistency across world-composed
      orientation (imu_accel_exhaustive_search.py) -- assuming accel used the
      SAME transform as gyro ranked #15 of 16 (nearly the worst option); the
      actual best (no-transpose) was 4-5x more self-consistent. The remaining
      sign ambiguity in that method (a candidate and its exact negation score
      identically) was resolved against imu0's own gravity direction (imu0
      needs no transform at all -- see below), which matched the no-transpose
      +Y-flip candidate unambiguously (dot +0.74 vs -0.74 for both controllers).
  Do not assume this generalizes to other controller hardware/calibration
  files without re-running both checks.

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


# Empirically-resolved sensor-frame -> body-frame transforms -- see module
# docstring for how each was determined and why they're NOT the same despite
# sharing one physical chip. _Y_FLIP is common to both; the transpose is not.
_Y_FLIP = np.diag([1.0, -1.0, 1.0])


def _gyro_body_transform(R_rt: np.ndarray) -> np.ndarray:
    return _Y_FLIP @ R_rt.T


def _accel_body_transform(R_rt: np.ndarray) -> np.ndarray:
    return _Y_FLIP @ R_rt


def load_and_calibrate_controller_imu(imu_path, controller_cfg: dict, lag_ns: int = 0,
                                       entry_index: int = 1):
    """Load one controller's raw imu*.csv and apply the full Stage 1/3 correction
    chain: mix+bias calibration, the empirically-resolved (and per-sensor-distinct
    -- see module docstring) axis transform, and an optional clock-offset
    correction (lag_ns, added to the raw timestamps -- e.g. Stage 1's measured
    controller<->camera offset).

    Single source for this chain -- main.py and visualize_imu.py both call this
    rather than each keeping their own copy of the transform logic.

    Returns (t_ns int64[N], gyro_body float64[N,3], accel_body float64[N,3]).
    """
    t_imu, gyro_raw, accel_raw = load_imu_csv(imu_path)
    calib = create_imu_calib_from_config(controller_cfg, entry_index=entry_index)

    gyro_corr = calib.gyro.correct(gyro_raw.astype(np.float64))
    gyro_body = (_gyro_body_transform(calib.gyro.T_rt.R) @ gyro_corr.T).T

    accel_corr = calib.accel.correct(accel_raw.astype(np.float64))
    accel_body = (_accel_body_transform(calib.accel.T_rt.R) @ accel_corr.T).T

    return t_imu + lag_ns, gyro_body, accel_body
