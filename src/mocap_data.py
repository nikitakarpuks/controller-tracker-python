"""Loaders for per-recording mocap ground truth (mocap_filtered/<device>/data.csv,
sibling to mav0/) and per-device mocap calibration (mocap_calibrations_for_each_device/
*.json) -- see the imu/mocap data-organization discussion this module implements:

  - Devices are "headset", "left_controller", "right_controller" -- same keys
    config.yml already uses for controllers; headset is implicit (no config
    entry of its own yet beyond cameras.mocap_calib_path).
  - Per-recording, each device's filtered/aligned trajectory lives at
    <recording_root>/mocap_filtered/<disk_name>/data.csv, where disk_name is
    "headset" / "ctrlleft" / "ctrlright" (not the config device keys --
    see main.py's _MOCAP_DISK_NAMES map). Schema is byte-identical to
    src/imu_data.py's load_vio_csv (#timestamp_ns,p_x,p_y,p_z,q_w,q_x,q_y,q_z),
    so that loader is reused directly rather than duplicated here.
  - Per-device persistent calibration (mocap_calibrations_for_each_device/*.json)
    holds far more than this pipeline needs -- only value0.T_imu_marker (the
    rigid mocap-marker <-> device-IMU spatial extrinsic) is read. Every other
    field (T_mocap_world, mocap_time_offset_ns/mocap_to_algorithm_offset_ns,
    the all-zero headset_static_*/residual_* stats) is intentionally ignored:
    this pipeline only ever needs controller-pose-relative-to-headset ground
    truth (see relative_pose below), and the shared mocap-world frame cancels
    out of that composition entirely, so no world-frame alignment is needed.

Time offset -- confirmed empirically against real recordings, NOT assumed:
    data.csv's timestamps already share mav0/imu*/data.csv's numerical epoch,
    but that shared epoch only means the COARSE (multi-second, "which system
    I hit record on first") component of the device's IMU<->mocap offset is
    already baked in -- confirmed by comparing timestamp ranges directly
    against a real recording, where naive epoch alignment recovers exactly
    the coarse_offset_ns component, not zero. The FINE (sub-second, ~100-150ms)
    residual from basalt_mocap_time_sync is NOT yet applied and must still be
    added at lookup time (see DeviceMocap.pose_at). final_offset_ns/
    final_offset_ms in a device's drift_check/*/final_offset.txt is the
    coarse+fine TOTAL -- useful only as a sanity-check log line ("did I
    roughly start Motive N seconds before this device"), never as a
    correction applied to the data.
"""
import json

import numpy as np
from scipy.spatial.transform import Rotation

from src.imu_data import load_vio_csv, interpolate_vio
from src.transformations import Transform

# mocap_filtered/<device>/drift_check/<variant>/drift_check.json -- 1chunk is a
# single flat/weighted-mean fit over the whole recording (no drift tracking);
# 10chunks additionally reports a std across chunks. Config-selected, not
# hardcoded, in case a recording only has one variant computed.
DRIFT_CHECK_VARIANT = "1chunk"


def load_mocap_csv(path):
    """mocap_filtered/<device>/data.csv -- identical EuRoC-ish schema to
    src/imu_data.py's vio/data.csv, so reuse that parser directly."""
    return load_vio_csv(path)


def load_mocap_fine_offset_ns(drift_check_json_path) -> float:
    """The FINE (sub-second) component of one device's IMU<->mocap time offset
    from a basalt_mocap_time_sync drift_check.json -- see this module's
    docstring for why this, not final_total.final_mean_offset_ns, is the
    value to apply at runtime.

    Convention: device_imu_time + fine_offset_ns ~= this device's
    mocap_filtered/<device>/data.csv timestamp, i.e. to look up this device's
    mocap trajectory at a given device-clock timestamp t, query it at
    (t + fine_offset_ns) -- see DeviceMocap.pose_at.
    """
    with open(drift_check_json_path) as f:
        d = json.load(f)
    return float(d["flat_stats"]["mean_offset_ns"])


def load_T_imu_marker(calib_json_path) -> Transform:
    """value0.T_imu_marker from a per-device mocap calibration file -- the
    rigid mocap-marker <-> device-IMU spatial extrinsic. See module docstring
    for why nothing else in that file is read."""
    with open(calib_json_path) as f:
        e = json.load(f)["value0"]["T_imu_marker"]
    R = Rotation.from_quat([e["qx"], e["qy"], e["qz"], e["qw"]]).as_matrix()
    t = np.array([e["px"], e["py"], e["pz"]], dtype=np.float64)
    return Transform(R, t)


def load_mocap_bridge(path) -> Transform:
    """value0.T_ledRef_mocapAccelImu from a controller's mocap-bridge file
    (data/mocap_calib/controller_{left,right}_mocap_bridge.json) -- the
    empirically-fit Transform s.t. T_world_ctrl(t).compose(bridge) ~=
    relative_pose(headset_mocap, ctrl_mocap, t) (see compare_vision_mocap.py,
    which is what produces these files and what to re-run to refresh one).

    Deliberately NOT derived from the controller's factory InertialSensors
    Rt -- that path was tried and found to be off by several mm/degrees
    (see the file's own "comment" field for the specific numbers from
    whichever fit produced it) -- this is fit directly against real vision +
    mocap data instead, which is what makes it usable with no further
    per-recording fitting (see load_or_fit_mocap_bridge)."""
    with open(path) as f:
        e = json.load(f)["value0"]["T_ledRef_mocapAccelImu"]
    R = Rotation.from_quat([e["qx"], e["qy"], e["qz"], e["qw"]]).as_matrix()
    t = np.array([e["px"], e["py"], e["pz"]], dtype=np.float64)
    return Transform(R, t)


DEFAULT_MAX_INTERP_GAP_NS = 30_000_000  # 30ms -- see DeviceMocap.pose_at


class DeviceMocap:
    """One device's mocap trajectory + fine time offset + marker<->IMU spatial
    extrinsic, all loaded for a single recording.

    max_interp_gap_ns guards against motive_to_gt.py's own filtering step
    (mocap_pipeline.py's convert_from_refit): it DROPS -- not blanks -- any
    frame that's N/A, over the Refit_RMS threshold, or an implausible jump
    from the last SURVIVING frame, so a burst of consecutive bad frames (real
    marker occlusion, not just one noisy sample) can leave a much larger gap
    in data.csv's timestamps than the nominal ~8.3ms mocap cadence -- 158ms
    gaps (~19x nominal) were confirmed empirically on a real recording, not
    hypothetical. Silently SLERP/linearly interpolating across a gap that
    size would fabricate a smooth ground-truth transition through a window
    where the real motion is actually unknown. pose_at() instead checks the
    LOCAL gap size around the specific query timestamp (not just whether it
    falls within the trajectory's overall covered range) and returns None --
    "no mocap ground truth here" -- once that gap exceeds this threshold,
    same no-fabrication policy as the existing start/end range check."""

    def __init__(self, t_ns: np.ndarray, position: np.ndarray, quat_xyzw: np.ndarray,
                 fine_offset_ns: float, T_imu_marker: Transform,
                 max_interp_gap_ns: float = DEFAULT_MAX_INTERP_GAP_NS):
        self.t_ns              = t_ns
        self.position           = position
        self.quat_xyzw          = quat_xyzw
        self.fine_offset_ns     = fine_offset_ns
        self.T_imu_marker       = T_imu_marker
        self.max_interp_gap_ns  = max_interp_gap_ns

    def pose_at(self, query_ts_ns: int):
        """Interpolated (R (3,3), t (3,)) marker pose in the shared mocap-world
        frame at query_ts_ns (device/vision clock domain) -- None if that falls
        outside the trajectory's covered range once the fine offset (see
        module docstring) is applied, OR if the two real samples bracketing it
        are more than max_interp_gap_ns apart (see class docstring)."""
        t_lookup = int(query_ts_ns) + int(round(self.fine_offset_ns))
        if t_lookup < self.t_ns[0] or t_lookup > self.t_ns[-1]:
            return None
        idx0 = min(int(np.searchsorted(self.t_ns, t_lookup, side="right")) - 1, len(self.t_ns) - 2)
        idx0 = max(idx0, 0)
        if self.t_ns[idx0 + 1] - self.t_ns[idx0] > self.max_interp_gap_ns:
            return None
        # Slice to just the two bracketing samples before interpolating -- t_lookup is guaranteed
        # (by idx0's own clamping above) to fall within [t_ns[idx0], t_ns[idx0+1]], and SLERP/
        # linear interpolation between two points depends only on those two points, so this is
        # numerically identical to interpolating against the full trajectory. Performance fix,
        # not a math change: interpolate_vio rebuilds a fresh Rotation+Slerp over WHATEVER array
        # it's given on every call -- passing the full ~8.3ms-cadence, whole-recording trajectory
        # (thousands of samples) on every single query made this call's cost scale with recording
        # length instead of being ~O(1), which went from a minor pre-existing inefficiency (this
        # was previously called at most twice per frame, for the ground-truth CSV export) to a
        # dominant one once headset_angular_velocity/headset_linear_velocity/
        # predict_headset_relative_pose's wiring started calling this several times per predict()
        # call, itself called multiple times per frame per controller (found from a real ~5x
        # slowdown report after that feature landed).
        R, pos = interpolate_vio(np.array([t_lookup]), self.t_ns[idx0:idx0 + 2],
                                  self.position[idx0:idx0 + 2], self.quat_xyzw[idx0:idx0 + 2])
        return R[0], pos[0]


def world_pose(device: DeviceMocap, query_ts_ns: int):
    """T_world_deviceImu(query_ts_ns) -- device's own IMU pose expressed in the
    shared mocap-world frame, i.e. device.pose_at's marker pose chained through
    T_imu_marker the same way relative_pose does, but WITHOUT cancelling the
    world frame against a second device. Needed for world-frame fusion (T_world_
    ctrl_vision = world_pose(headset, t).compose(T_headsetImu_ctrl_vision)) --
    relative_pose alone can't provide this since its whole point is that the
    world frame cancels out. Returns a Transform, or None if device has no
    mocap coverage at this timestamp (see DeviceMocap.pose_at)."""
    m = device.pose_at(query_ts_ns)
    if m is None:
        return None
    T_world_marker = Transform(*m)
    return T_world_marker.compose(device.T_imu_marker.inverse())


DEFAULT_EGO_MOTION_WINDOW_S = 0.02  # +-10ms central finite-difference window, see the two
                                    # functions below -- an implementation-numerical constant
                                    # (mocap's own ~8.3ms nominal cadence is stable across
                                    # recordings), not a per-recording tuning knob, so a function
                                    # default rather than a config.yml key.


def _bracket_world_poses(device: DeviceMocap, query_ts_ns: int, window_s: float):
    """(T_wh_a, T_wh_b) = world_pose(device, t) at query_ts_ns -+ window_s/2, or None if either
    endpoint lacks mocap coverage (propagates world_pose's/pose_at's own None contract -- a real,
    non-rare occurrence: marker-occlusion gaps up to ~158ms are documented in DeviceMocap's own
    docstring). Shared by headset_angular_velocity/headset_linear_velocity below so both draw
    from the exact same pair of pose_at lookups rather than risking two independently-rounded
    brackets."""
    half_ns = int(window_s * 1e9 / 2)
    T_a = world_pose(device, query_ts_ns - half_ns)
    T_b = world_pose(device, query_ts_ns + half_ns)
    if T_a is None or T_b is None:
        return None
    return T_a, T_b


def headset_angular_velocity(device: DeviceMocap, query_ts_ns: int,
                              window_s: float = DEFAULT_EGO_MOTION_WINDOW_S):
    """Body-frame angular velocity (rad/s) of device's IMU frame at query_ts_ns, via a central
    finite difference of world_pose(device, t)'s ROTATION across a small window straddling
    query_ts_ns. Returns None on any coverage gap (see _bracket_world_poses).

    Differences world_pose's own (IMU-frame) rotation, NOT device.pose_at's raw marker rotation
    directly -- the two differ by the fixed T_imu_marker rotation, and only world_pose's is
    already expressed in the IMU's own axes. Differencing the raw marker rotation would give the
    same physical rate in different, wrongly-oriented coordinates, silently corrupting every
    caller of this function (in particular src.imu_data.predict_headset_relative_pose, which
    needs this in the SAME frame as the gyro_body it composes against).

    dR/dt = R @ [omega]_x (this codebase's own body-frame angular-velocity convention -- see
    src.imu_data.integrate_gyro_segment's R_new = R_old @ R_rel right-composition, of which this
    is the continuous-time limit), so omega = Log(R_a.T @ R_b) / window_s, evaluated at the
    window's start (treated as ~constant over window_s, matching integrate_gyro_segment's own
    midpoint-rule small-angle assumption over a comparably short span)."""
    bracket = _bracket_world_poses(device, query_ts_ns, window_s)
    if bracket is None:
        return None
    T_a, T_b = bracket
    rotvec = Rotation.from_matrix(T_a.R.T @ T_b.R).as_rotvec()
    return rotvec / window_s


def headset_linear_velocity(device: DeviceMocap, query_ts_ns: int,
                             window_s: float = DEFAULT_EGO_MOTION_WINDOW_S):
    """World-frame (mocap-world) linear velocity (m/s) of device's IMU-frame ORIGIN at
    query_ts_ns, via a central finite difference of world_pose(device, t)'s POSITION. Returns
    None on any coverage gap (see _bracket_world_poses).

    Differences world_pose's IMU-origin position, NOT the raw marker position -- this correctly
    captures the velocity induced by the marker<->IMU lever arm sweeping through rotation
    whenever the device is rotating (the mocap-side analog of src.imu_data's existing
    controller-side _lever_arm_correction); differencing the raw marker position would miss
    exactly that component."""
    bracket = _bracket_world_poses(device, query_ts_ns, window_s)
    if bracket is None:
        return None
    T_a, T_b = bracket
    return (T_b.t - T_a.t) / window_s


def relative_pose(headset: DeviceMocap, device: DeviceMocap, query_ts_ns: int):
    """Ground-truth T_headsetImu_deviceImu(query_ts_ns) -- device pose expressed
    in the headset-IMU frame, chained through each device's own T_imu_marker so
    the result is comparable to the vision pipeline's own IMU/LED-reference-frame
    poses (T_world_ctrl), not left in the mocap rig's arbitrary per-device marker
    mounting frame. The shared mocap-world frame both devices' raw trajectories
    are expressed in cancels out of this composition, so no world-frame
    alignment (T_mocap_world or similar) is needed, only each device's own
    trajectory + T_imu_marker. Returns a Transform, or None if either device
    has no mocap coverage at this timestamp."""
    h = headset.pose_at(query_ts_ns)
    d = device.pose_at(query_ts_ns)
    if h is None or d is None:
        return None
    T_world_headsetMarker = Transform(*h)
    T_world_deviceMarker  = Transform(*d)
    T_headsetMarker_deviceMarker = T_world_headsetMarker.inverse().compose(T_world_deviceMarker)
    return headset.T_imu_marker.compose(T_headsetMarker_deviceMarker).compose(device.T_imu_marker.inverse())
