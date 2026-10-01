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

Vision-clock offset (DeviceMocap.vision_offset_ns) -- the SECOND time link, easy to
    confuse with the fine offset above:
    the fine offset maps a device's OWN IMU clock -> mocap clock (basalt_mocap_time_sync,
    gyro vs mocap angular velocity). Vision poses, though, are stamped with the CAMERA
    frame time, and the controller's IMU stamps are NOT on that clock: measured against
    the vision poses of all 8 recordings-aug26 recordings (vision-vs-controller-gyro and
    vision-vs-mocap angular-velocity cross-correlation, both agree), controller raw IMU
    stamp = camera stamp + ~7.6ms for both controllers, std 0.24ms across 16
    recording/controller cases, uncorrelated with the per-device fine offsets. main.py's
    lag_ns (imu_data.load_and_calibrate_controller_imu) already applies this link to the
    IMU stream; without vision_offset_ns the mocap lookup silently skipped it, leaving a
    constant ~7.6ms vision<->mocap lag that showed up as a speed-proportional residual
    (~speed * 7ms) in every vision-vs-mocap comparison. Lookup is therefore
    frame_ts + vision_offset_ns + fine_offset_ns. Defaults to 0 (old behavior); set per
    controller via config.yml's mocap_vision_offset_ns (see load_vision_offset_ns).

Vision-clock DRIFT (DeviceMocap.drift_offset_ns / drift_rate_ns_per_ns) -- a THIRD, independent
    correction, on top of vision_offset_ns above, not a replacement for it:
    the vision<->mocap lag is not perfectly constant -- a per-recording sweep of the best extra
    lookup shift in time quarters, and a held-out linear-drift fit (train on time blocks, score on
    held-out blocks, bridge refit under each model), both found a consistent residual drift of
    roughly +2.5 to +3.5 ms/min between the mocap clock and the camera/controller-IMU clock,
    the SAME SIGN in essentially every recording-aug26 recording. A held-out linear correction
    measurably reduces the STATIC_MEDIUM/STATIC_HARD residual (roughly 5-25%, largest on rotation)
    with no held-out case made worse; on the calmer static_dark it helps the right controller and
    is close to neutral for the left. Fit as one shared (offset, rate) pair per controller, pooled
    (n-weighted) over the 3 recordings with fresh, current-pipeline vision (static_dark,
    static_medium, static_hard) -- see analysis/drift_test/fit_deploy_drift.py.

    Deliberately kept SEPARATE from vision_offset_ns/mocap_vision_offset_ns (not folded into that
    same number) even though both are additive constants at the pivot: mocap_vision_offset_ns also
    drives controller_imu_lag_ns (the constant used for the LIVE controller IMU stream, gyro/accel
    integration, via -mocap_vision_offset_ns) -- an unrelated, separately-validated constant this
    drift correction must NOT silently perturb. This drift correction only ever touches
    DeviceMocap.pose_at, i.e. mocap ground truth comparison and (when mocap.enabled) the headset
    ego-motion correction in predict_headset_relative_pose -- never the IMU stream's own lag.

    Pivot: each device's OWN mocap-track start (t_ns[0], set at construction) -- not the mean time
    of whichever frames a given run happens to track, which would make the fitted intercept
    dependent on run-to-run coverage. This needs no extra file or config lookup (t_ns is always
    already loaded) and makes the correction a pure function of (query_ts_ns - t_ns[0]), so it
    behaves identically regardless of which frame_range subset of the recording is processed.
    Lookup is now frame_ts + vision_offset_ns + drift_offset_ns +
    drift_rate_ns_per_ns*(frame_ts - t_ns[0]) + fine_offset_ns. Defaults to (0, 0) -- old behavior.
    Set per controller via config.yml's mocap_vision_drift_offset_ns /
    mocap_vision_drift_rate_ms_per_min (see load_vision_drift_params). Headset is deliberately left
    at (0, 0): the drift fit above is controller-mocap-specific (pivoted on the CONTROLLER's own
    track), and no comparably-validated headset-side drift fit exists.
"""
import json
from pathlib import Path
from typing import Optional, Tuple

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


def load_vision_offset_ns(device_cfg) -> float:
    """config.yml's <device>.mocap_vision_offset_ns (vision-frame-stamp -> this device's
    IMU clock, see module docstring), or 0.0 if unset/None -- old behavior."""
    v = (device_cfg or {}).get("mocap_vision_offset_ns")
    return 0.0 if v is None else float(v)


def load_vision_drift_params(device_cfg) -> tuple:
    """config.yml's <device>.mocap_vision_drift_offset_ns / mocap_vision_drift_rate_ms_per_min
    (see module docstring's "Vision-clock DRIFT" section) as (drift_offset_ns, drift_rate_ns_per_ns)
    -- the rate is converted from human-readable ms/min to a dimensionless ns-correction-per-ns-
    elapsed ratio (ms_per_min * 1e6 / 60e9 = ms_per_min / 60_000). (0.0, 0.0) if either is unset --
    old behavior, no drift term. Deliberately independent of load_vision_offset_ns/
    controller_imu_lag_ns -- see module docstring."""
    cfg = device_cfg or {}
    offset_ns = cfg.get("mocap_vision_drift_offset_ns")
    rate_ms_per_min = cfg.get("mocap_vision_drift_rate_ms_per_min")
    offset_ns = 0.0 if offset_ns is None else float(offset_ns)
    rate_ns_per_ns = 0.0 if rate_ms_per_min is None else float(rate_ms_per_min) / 60_000.0
    return offset_ns, rate_ns_per_ns


# Legacy per-controller lag (t_imu + lag_ns), measured on one older clip -- ONLY used when config.yml's
# mocap_vision_offset_ns is unset for that controller.
_LEGACY_CONTROLLER_LAG_NS = {"left_controller": -5_000_000, "right_controller": -7_000_000}
_CONTROLLER_IMU_REL_PATH  = {"left_controller": "imu1/data.csv", "right_controller": "imu2/data.csv"}


def controller_imu_lag_ns(ctrl_key: str, config: dict = None) -> int:
    """lag_ns for load_and_calibrate_controller_imu (t_imu + lag_ns puts the raw controller IMU stamps
    on the camera clock) = -mocap_vision_offset_ns from config.yml -- the SAME physical link the mocap
    lookup applies via DeviceMocap.vision_offset_ns (see module docstring), so the IMU stream and the
    mocap lookup can't drift apart. config=None loads config/config.yml. Falls back to the legacy
    -5ms/-7ms constants only if the key is unset for that controller."""
    if config is None:
        from pathlib import Path
        from src.load_config import load_yaml_config
        config = load_yaml_config(str(Path(__file__).resolve().parent.parent / "config" / "config.yml"))
    offset_ns = load_vision_offset_ns(config["controllers"].get(ctrl_key))
    return -int(round(offset_ns)) if offset_ns else _LEGACY_CONTROLLER_LAG_NS[ctrl_key]


def controller_imu_files(config: dict = None) -> dict:
    """{ctrl_key: (imu csv path relative to mav0/, lag_ns)} -- the (path, lag) table every script that
    loads the controller IMUs used to hardcode. See controller_imu_lag_ns."""
    return {k: (rel, controller_imu_lag_ns(k, config)) for k, rel in _CONTROLLER_IMU_REL_PATH.items()}


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
                 max_interp_gap_ns: float = DEFAULT_MAX_INTERP_GAP_NS,
                 vision_offset_ns: float = 0.0,
                 drift_offset_ns: float = 0.0, drift_rate_ns_per_ns: float = 0.0):
        self.t_ns              = t_ns
        self.position           = position
        self.quat_xyzw          = quat_xyzw
        self.fine_offset_ns     = fine_offset_ns
        self.T_imu_marker       = T_imu_marker
        self.max_interp_gap_ns  = max_interp_gap_ns
        self.vision_offset_ns   = vision_offset_ns
        self.drift_offset_ns       = drift_offset_ns
        self.drift_rate_ns_per_ns  = drift_rate_ns_per_ns
        # Pivot for the drift term: THIS device's own mocap-track start (see module docstring's
        # "Vision-clock DRIFT" section for why -- a fixed, always-available, run-coverage-
        # independent anchor). int() up front since t_ns entries are queried against int query
        # timestamps below; harmless (and pivot is unused) when drift_rate_ns_per_ns == 0.
        self._drift_pivot_ns = int(t_ns[0]) if len(t_ns) else 0

    def pose_at(self, query_ts_ns: int):
        """Interpolated (R (3,3), t (3,)) marker pose in the shared mocap-world
        frame at query_ts_ns (device/vision clock domain) -- None if that falls
        outside the trajectory's covered range once the fine offset (see
        module docstring) is applied, OR if the two real samples bracketing it
        are more than max_interp_gap_ns apart (see class docstring)."""
        query_ts_ns = int(query_ts_ns)
        drift_ns = self.drift_offset_ns + self.drift_rate_ns_per_ns * (query_ts_ns - self._drift_pivot_ns)
        t_lookup = query_ts_ns + int(round(self.vision_offset_ns + drift_ns + self.fine_offset_ns))
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


MOCAP_DISK_NAMES = {"headset": "headset", "left_controller": "ctrlleft", "right_controller": "ctrlright"}


def load_device_mocap_from_config(recording_root, device_key: str, config: dict) -> Tuple[Optional[DeviceMocap], Optional[str]]:
    """THE one place a recording's per-device DeviceMocap is built from config.yml -- shared by
    main.py (live pipeline) and xrtslam-metrics' make_xrtslam_targets.py, so the ground truth the
    metrics are computed against can never again disagree with the mocap lookup the pipeline
    uses (the targets script used to build DeviceMocap with only the fine offset, silently missing
    mocap_vision_offset_ns and the vision<->mocap drift -- ~2-9 ms of time error growing over a
    recording, i.e. 10-25 mm of apparent error at 2-3 m/s).

    recording_root: the recording directory that CONTAINS mav0/ and mocap_filtered/.
    device_key: "headset" | "left_controller" | "right_controller".
    Returns (DeviceMocap, None), or (None, reason) when the device's mocap data/calibration is
    incomplete (caller decides whether to warn or fail). The returned object also carries
    `.fine_offset_source` (str, for logging only).

    Per-device settings come from config["cameras"] (headset) or config["controllers"][key]:
    mocap_calib_path, mocap_fine_offset_override_ns, mocap_vision_offset_ns and
    mocap_vision_drift_offset_ns / mocap_vision_drift_rate_ms_per_min (headset has none -> (0, 0)
    drift, deliberately, see module docstring). max_interp_gap_ms comes from config["mocap"]."""
    recording_root = Path(recording_root)
    mocap_cfg = config.get("mocap", {})
    dev_cfg = config["cameras"] if device_key == "headset" else config["controllers"][device_key]
    calib_path = dev_cfg.get("mocap_calib_path")
    offset_override_ns = dev_cfg.get("mocap_fine_offset_override_ns")
    device_dir = recording_root / "mocap_filtered" / MOCAP_DISK_NAMES[device_key]
    data_path = device_dir / "data.csv"
    drift_path = device_dir / "drift_check" / DRIFT_CHECK_VARIANT / "drift_check.json"
    # drift_path is only required when no manual override is configured -- an override lets a
    # device be used before its drift_check has even been run (see config.yml's
    # mocap_fine_offset_override_ns comment).
    if not calib_path or not data_path.exists() or (offset_override_ns is None and not drift_path.exists()):
        return None, f"mocap data/calibration incomplete ({device_dir})"
    t_mocap, position, quat_xyzw = load_mocap_csv(data_path)
    if offset_override_ns is not None:
        fine_offset_ns = float(offset_override_ns)
        source = "config override"
    else:
        fine_offset_ns = load_mocap_fine_offset_ns(drift_path)
        source = f"{DRIFT_CHECK_VARIANT}/drift_check.json"
    T_imu_marker = load_T_imu_marker(calib_path)
    max_gap_ns = float(mocap_cfg.get("max_interp_gap_ms", 30.0)) * 1e6
    vision_offset_ns = load_vision_offset_ns(dev_cfg)
    drift_offset_ns, drift_rate_ns_per_ns = load_vision_drift_params(dev_cfg)
    dm = DeviceMocap(t_mocap, position, quat_xyzw, fine_offset_ns, T_imu_marker,
                     max_interp_gap_ns=max_gap_ns, vision_offset_ns=vision_offset_ns,
                     drift_offset_ns=drift_offset_ns, drift_rate_ns_per_ns=drift_rate_ns_per_ns)
    dm.fine_offset_source = source
    dm.data_path = data_path
    return dm, None
