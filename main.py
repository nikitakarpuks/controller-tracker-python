import csv
from pathlib import Path
from shutil import copy
from time import time

import numpy as np
import rerun as rr

from loguru import logger
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from src import debug_config
from src.blob_detector import (BlobDetector, BlobResult, _blackout_neighborhoods,
                               _compute_led_search_radii)
from src.camera import Camera
from src.controller import ControllerModel, TrackingSystem, create_leds_from_config, mirror_primitives
from src.imu_data import load_and_calibrate_controller_imu, create_imu_calib_from_config, _DIAG_FLIP, \
    LiveGravityEstimator
from src.mocap_data import DeviceMocap, load_mocap_csv, load_mocap_fine_offset_ns, load_T_imu_marker, \
                            relative_pose, DRIFT_CHECK_VARIANT
from src.load_config import load_yaml_config, load_json_config
from src.preprocess_data import get_data, count_images
from src.transformations import Transform
from src.visualization import (ControllerAnimatorRerun, prepare_model_geometry,
                               fine_tune_alignment, load_trimesh)

SLOW_MATCH_THRESHOLD_S = 1.5


def main():
    # Initialise rerun before any other native libraries (numpy BLAS/LAPACK,
    # cv2, scipy) are loaded to prevent a Windows DLL heap-state conflict that
    # causes rrb.EyeControls3D() to crash with 0xC0000005 ACCESS_VIOLATION.
    rr.init("controller_animator", spawn=False)

    config = load_yaml_config('./config/config.yml')

    data_root = Path(config["data"]["root"])

    debug_cfg = config.get("debug", {})

    continuous_frames = bool(debug_cfg.get("assume_continuous_frames", True))

    debug_config.configure(
        continuous_frames = continuous_frames,
        log_all_triples = bool(debug_cfg.get("log_all_triples", False)),
        log_all_proximity_hyps = bool(debug_cfg.get("log_all_proximity_hyps", False)),
        log_best       = bool(debug_cfg.get("log_best", True)),
        debug_led_ids  = debug_cfg.get("debug_led_ids") or None,
        debug_blob_ids = debug_cfg.get("debug_blob_ids") or None,
        log_categories = {
            "startup":             bool(debug_cfg.get("log_startup", True)),
            "frame_summary":       bool(debug_cfg.get("log_frame_summary", True)),
            "timings":             bool(debug_cfg.get("log_timings", True)),
            "blob_detection":      bool(debug_cfg.get("log_blob_detection", False)),
            "blob_diag":           bool(debug_cfg.get("log_blob_diag", False)),
            "matching_decisions":  bool(debug_cfg.get("log_matching_decisions", False)),
            "batch_orchestration": bool(debug_cfg.get("log_batch_orchestration", False)),
            "pose_fusion":         bool(debug_cfg.get("log_pose_fusion", False)),
            "occlusion":           bool(debug_cfg.get("log_occlusion", False)),
            "proximity_match":     bool(debug_cfg.get("log_proximity_match", False)),
            "hypothesis_testing":  bool(debug_cfg.get("log_hypothesis_testing", False)),
            "ransac":              bool(debug_cfg.get("log_ransac", False)),
            "self_calibration":    bool(debug_cfg.get("log_self_calibration", False)),
        },
    )
    debug_config.setup_logging()

    logger.bind(cat="startup").info(f"continuous_frames={continuous_frames}  data={data_root}")

    # ── Output directories (harvest problem frames from a continuous run) ──
    out_slow = out_tracking_lost = None
    if config["debug"]["split_to_folders"]:
        out_slow          = data_root / "deep_search_required"
        out_tracking_lost = data_root / "tracking_lost"
        out_slow.mkdir(parents=True, exist_ok=True)
        out_tracking_lost.mkdir(parents=True, exist_ok=True)

    # ── Camera & controller setup ──────────────────────────────────────────
    calib_cfg              = load_json_config(config["cameras"]["intrinsics_path"])
    extrinsics_convention  = config["cameras"].get("extrinsics_convention", "T_imu_cam")
    cameras   = {idx: Camera(calib_cfg, camera_idx=idx,
                             extrinsics_convention=extrinsics_convention)
                 for idx in config["data"]["selected_cameras"]}
    blob_detectors = {idx: BlobDetector(idx, config["blob_detection"])
                      for idx in cameras}

    # Load all enabled controllers; build per-controller geometry configs.
    enabled_ctrls    = {}   # {ctrl_name: ControllerModel}
    ctrl_leds        = {}   # {ctrl_name: [ControllerLED, ...]}
    ctrl_geom        = {}   # {ctrl_name: (positions_model, normals_model, T_model_ctrl)}
    geo_cfg_per_ctrl = {}   # {ctrl_name: geometry_cfg dict with handle_primitives}
    ctrl_json_cfg    = {}   # {ctrl_name: loaded controller calibration JSON (leds + InertialSensors)}
    right_ctrl_cfg   = config["controllers"]["right_controller"]

    for ctrl_key in ["right_controller", "left_controller"]:
        ctrl_cfg = config["controllers"].get(ctrl_key, {})
        if not ctrl_cfg.get("enabled", False):
            continue
        json_cfg = load_json_config(ctrl_cfg["config_path"])
        ctrl_json_cfg[ctrl_key] = json_cfg
        leds = create_leds_from_config(json_cfg)
        ctrl_leds[ctrl_key]    = leds
        enabled_ctrls[ctrl_key] = ControllerModel(leds, ctrl_key)

        side = "right" if ctrl_key == "right_controller" else "left"
        ctrl_geom[ctrl_key] = prepare_model_geometry(leds, right_ctrl_cfg, side=side)

        geo = dict(config.get("geometry", {}))
        if "handle_primitives" in ctrl_cfg:
            geo["handle_primitives"] = ctrl_cfg["handle_primitives"]
        elif ctrl_key == "left_controller":
            right_prim = right_ctrl_cfg.get("handle_primitives")
            if right_prim is not None:
                geo["handle_primitives"] = mirror_primitives(right_prim)
        geo_cfg_per_ctrl[ctrl_key] = geo

    # ── IMU (Stage 3): load + calibrate controller gyro/accel for pose prediction,
    # the gravity-alignment diagnostic, and (when fusion.enabled) the pose-fusion
    # filter's dead-reckoning ────────────────────────────────────────────────────
    # Confirmed mapping (mentor + Stage 1 cross-correlation validation, this
    # recording only): imu1.csv = left controller, imu2.csv = right controller.
    # _DIAG_FLIP is the confirmed sensor->body axis transform (see
    # src/imu_data.py's module docstring for the full story). Per-controller lag
    # is Stage 1's measured controller<->camera clock offset on THIS recording
    # (imu_vision_sync_check.py) — a single-clip estimate, re-measure if this ever
    # runs against different data.
    imu_cfg = config.get("imu", {})
    gyro_data:  dict = {}
    accel_data: dict = {}
    lever_arm:  dict = {}   # {ctrl_key: (3,) accel<->gyro lever arm}, see PoseFusionFilter
    g_world_estimator = None
    if imu_cfg.get("enabled", False):
        _mav0_root = Path(config["data"]["root"])
        _IMU_FILES = {"left_controller":  ("imu1/data.csv", -5_000_000),
                      "right_controller": ("imu2/data.csv", -7_000_000)}
        for ctrl_key, (imu_rel_path, lag_ns) in _IMU_FILES.items():
            if ctrl_key not in enabled_ctrls:
                continue
            imu_path = _mav0_root / imu_rel_path
            if not imu_path.exists():
                logger.bind(cat="startup").warning(
                    f"[{ctrl_key}] IMU file not found ({imu_path}) — gyro prediction "
                    f"and gravity-check diagnostic disabled for this controller")
                continue
            t_imu, gyro_body, accel_body = load_and_calibrate_controller_imu(
                imu_path, load_json_config(config["controllers"][ctrl_key]["config_path"]), lag_ns=lag_ns,
            )
            gyro_data[ctrl_key]  = (t_imu, gyro_body)
            accel_data[ctrl_key] = (t_imu, accel_body)

            _imu_calib = create_imu_calib_from_config(ctrl_json_cfg[ctrl_key])
            lever_arm[ctrl_key] = _imu_calib.accel.T_rt.compose(_imu_calib.gyro.T_rt.inverse()).t

            logger.bind(cat="startup").info(f"[{ctrl_key}] IMU loaded: {len(t_imu)} samples from {imu_path.name}")

        fusion_cfg = config.get("fusion", {})
        g_world_estimator = LiveGravityEstimator(
            omega_thresh=float(fusion_cfg.get("g_world_low_omega_thresh_rad_s", 0.5)),
            min_samples=int(fusion_cfg.get("g_world_min_samples", 20)),
        )

    # ── Mocap ground truth (see src/mocap_data.py + the imu/mocap organization
    # discussion): per-device filtered/aligned trajectories live under
    # mocap_filtered/, a SIBLING of mav0/ (not inside it) at the recording root.
    # headset is loaded unconditionally when enabled -- every controller's
    # ground truth is expressed relative to it (see relative_pose), independent
    # of which controllers happen to be enabled this run.
    mocap_cfg = config.get("mocap", {})
    device_mocap: dict = {}   # {"headset": DeviceMocap, ctrl_key: DeviceMocap, ...}
    if mocap_cfg.get("enabled", False):
        _recording_root    = Path(config["data"]["root"]).parent
        _MOCAP_DISK_NAMES  = {"headset": "headset", "left_controller": "ctrlleft", "right_controller": "ctrlright"}
        _mocap_device_cfg  = {"headset": config["cameras"],
                               "left_controller":  config["controllers"]["left_controller"],
                               "right_controller": config["controllers"]["right_controller"]}
        for device_key in ["headset", *enabled_ctrls]:
            _dev_cfg    = _mocap_device_cfg[device_key]
            calib_path  = _dev_cfg.get("mocap_calib_path")
            offset_override_ns = _dev_cfg.get("mocap_fine_offset_override_ns")
            device_dir  = _recording_root / "mocap_filtered" / _MOCAP_DISK_NAMES[device_key]
            data_path   = device_dir / "data.csv"
            drift_path  = device_dir / "drift_check" / DRIFT_CHECK_VARIANT / "drift_check.json"
            # drift_path is only required when no manual override is configured --
            # an override lets a device be used before its drift_check has even
            # been run (see config.yml's mocap_fine_offset_override_ns comment).
            if not calib_path or not data_path.exists() or (offset_override_ns is None and not drift_path.exists()):
                logger.bind(cat="startup").warning(
                    f"[{device_key}] mocap data/calibration incomplete ({device_dir}) — "
                    f"mocap ground truth disabled for this device")
                continue
            t_mocap, position, quat_xyzw = load_mocap_csv(data_path)
            if offset_override_ns is not None:
                fine_offset_ns = float(offset_override_ns)
                _offset_source = "config override"
            else:
                fine_offset_ns = load_mocap_fine_offset_ns(drift_path)
                _offset_source = f"{DRIFT_CHECK_VARIANT}/drift_check.json"
            T_imu_marker = load_T_imu_marker(calib_path)
            _max_gap_ns  = float(mocap_cfg.get("max_interp_gap_ms", 30.0)) * 1e6
            device_mocap[device_key] = DeviceMocap(t_mocap, position, quat_xyzw, fine_offset_ns, T_imu_marker,
                                                    max_interp_gap_ns=_max_gap_ns)
            logger.bind(cat="startup").info(
                f"[{device_key}] mocap loaded: {len(t_mocap)} samples from {data_path} "
                f"(fine offset {fine_offset_ns / 1e6:.1f} ms, from {_offset_source})")

    tracking_system = TrackingSystem(
        list(enabled_ctrls.values()), list(cameras.values()),
        matching_cfg=config.get("matching", {}),
        geometry_cfg=config.get("geometry", {}),
        geometry_cfg_per_ctrl=geo_cfg_per_ctrl,
        self_calibration_cfg=config.get("self_calibration", {}),
        blob_detection_cfg=config["blob_detection"],
        gyro_data=gyro_data,
        accel_data=accel_data,
        lever_arm=lever_arm,
        g_world_estimator=g_world_estimator,
        fusion_cfg=config.get("fusion", {}),
        debug_pose_fusion_cfg=config["visualization"].get("pose_fusion_debug", {}),
    )
    pool          = tracking_system.get_pool()
    blob_parallel = tracking_system.blob_parallel_enabled

    if config["visualization"].get("fine_tune_alignment") and "right_controller" in enabled_ctrls:
        mesh = load_trimesh(config["visualization"]["3d_model_path"])
        fine_tune_alignment(ctrl_leds["right_controller"], mesh, right_ctrl_cfg)

    # ── Visualiser setup (streamed, one log_frame() call per tracked frame —
    # see ControllerAnimatorRerun.begin()'s docstring for why this must not be
    # replaced with buffer-then-replay) ─────────────────────────────────────
    animator = None
    if enabled_ctrls and config["visualization"].get("enabled", True):
        controllers_vis = {}
        for ctrl_name in enabled_ctrls:
            pos, nrm, T = ctrl_geom[ctrl_name]
            side = "right" if ctrl_name == "right_controller" else "left"
            controllers_vis[ctrl_name] = {
                "positions":    pos,
                "normals":      nrm,
                "T_model_ctrl": T,
                "side":         side,
                "geometry_cfg": geo_cfg_per_ctrl[ctrl_name],
            }
        animator = ControllerAnimatorRerun(
            config["visualization"]["3d_model_path"],
            controllers_vis,
            matching_cfg=config.get("matching", {}),
            pose_fusion_debug_enabled=bool(
                config["visualization"].get("pose_fusion_debug", {}).get("enabled", False)),
        )
        animator.begin(cameras, save_path=config["visualization"].get("save_recording"))

    # ── Tracking loop ──────────────────────────────────────────────────────
    any_valid_pose    = {n: False for n in enabled_ctrls}
    last_good_T_world = {n: None for n in enabled_ctrls}
    # Consecutive lost frames per controller (any_valid_pose sanity check below).
    lost_streak       = {n: 0 for n in enabled_ctrls}
    # Cold-path BlobDetector EMA-threshold memory, round-tripped explicitly through
    # run_blob_detect() rather than left as worker-resident state (a pool task isn't
    # pinned to the same worker every call) — keyed per (cam_idx, ctrl_name) so two
    # controllers cold-starting on the same camera in the same frame no longer
    # clobber each other's memory (see run_blob_detect's docstring).
    _cold_memory: dict = {}

    _csv_path = debug_cfg.get("calibration_csv")
    _csv_file = _csv_writer = None
    if _csv_path:
        Path(_csv_path).parent.mkdir(parents=True, exist_ok=True)
        _csv_file = open(_csv_path, "w", newline="")
        _csv_writer = csv.writer(_csv_file)
        _csv_writer.writerow(["frame", "ctrl_name", "cam_idx",
                               "led_id", "depth_m", "facing_cos", "velocity_px",
                               "brightness", "area"])
        logger.bind(cat="startup").info(f"Calibration CSV → {_csv_path}")

    _pose_csv_path = debug_cfg.get("pose_csv")
    _pose_csv_file = _pose_csv_writer = None
    if _pose_csv_path:
        Path(_pose_csv_path).parent.mkdir(parents=True, exist_ok=True)
        _pose_csv_file = open(_pose_csv_path, "w", newline="")
        _pose_csv_writer = csv.writer(_pose_csv_file)
        _pose_csv_writer.writerow(["timestamp_ns", "ctrl_name", "qx", "qy", "qz", "qw", "px", "py", "pz", "reproj_err_px", "inlier_count"])
        logger.bind(cat="startup").info(f"Pose CSV → {_pose_csv_path}")

    _vision_pose_csv_path = debug_cfg.get("vision_pose_csv")
    _vision_pose_csv_file = _vision_pose_csv_writer = None
    if _vision_pose_csv_path:
        Path(_vision_pose_csv_path).parent.mkdir(parents=True, exist_ok=True)
        _vision_pose_csv_file = open(_vision_pose_csv_path, "w", newline="")
        _vision_pose_csv_writer = csv.writer(_vision_pose_csv_file)
        _vision_pose_csv_writer.writerow(["timestamp_ns", "ctrl_name", "qx", "qy", "qz", "qw", "px", "py", "pz",
                                           "confidence", "error_px", "n_inliers"])
        logger.bind(cat="startup").info(f"Vision pose CSV → {_vision_pose_csv_path}")

    # ── Raw per-LED 2D observations (pre-PnP-solve point data) -- unlike pose_csv
    # above (already-solved 6-DOF pose) or calibration_csv (primary-camera-only),
    # this is every LED actually matched THIS frame across ALL cameras that
    # contributed to the accepted solve -- what a bundle-adjustment-style external
    # tool needs and can't recover by working backward from a solved pose.
    _led_csv_path = debug_cfg.get("led_detections_csv")
    _led_csv_file = _led_csv_writer = None
    if _led_csv_path:
        Path(_led_csv_path).parent.mkdir(parents=True, exist_ok=True)
        _led_csv_file = open(_led_csv_path, "w", newline="")
        _led_csv_writer = csv.writer(_led_csv_file)
        _led_csv_writer.writerow(["timestamp_ns", "camera_id", "ctrl_name", "led_id", "pixel_x", "pixel_y",
                                   "blob_radius_px", "brightness"])
        logger.bind(cat="startup").info(f"LED-detections CSV → {_led_csv_path}")

    # ── basalt_controller_mocap_calib input: one T_Ih_Ic(t) log per controller ──
    # T_world_ctrl is already T_Ih_ref (headset-IMU <- LED reference frame): LED
    # Position/Normal in each controller's config JSON, and the InertialSensors
    # Rt entries' near-zero translation, are all given relative to that same
    # reference frame, whose origin is defined to sit exactly at the (Id=Undefined)
    # gyro's physical location -- so T_world_ctrl.t is already the IMU's position,
    # no correction needed. What's needed for T_Ih_Ic's ROTATION is R_ref_ic: the
    # rotation taking a vector expressed in the controller-IMU's own native sensor
    # axes into reference-frame axes (T_Ih_Ic = T_world_ctrl.compose(Transform(
    # R_ref_ic, 0))).
    #
    # RESOLVED 2026-09-02 (previously computed from the controller config's
    # InertialSensors Rt, with an UNVERIFIED transpose direction -- see git
    # history for that version): R_ref_ic is exactly _DIAG_FLIP, the same
    # sensor-frame->body-frame transform load_and_calibrate_controller_imu now
    # applies to raw gyro/accel samples to land them in this same reference
    # frame (see src/imu_data.py's module docstring) -- gyro_body IS gyro data
    # expressed in reference-frame axes, by construction, so the same transform
    # is what T_Ih_Ic's rotation needs too. Confirmed Rt's rotation is NOT the
    # right quantity for this either direction: both Rt.R and Rt.R.T are
    # ~137-139° away from _DIAG_FLIP (both controllers) -- the same ~140°
    # mismatch found independently twice elsewhere in this investigation (see
    # visualization/controller_calibration_for_basalt/README.md findings 5/7).
    # Rt's rotation just isn't the sensor<->reference-frame axis relationship
    # for this hardware; only its TRANSLATION (used for the lever arm) is.
    _algo_log_dir = debug_cfg.get("algorithm_log_dir")
    _algo_log_writers = {}   # {ctrl_name: csv.writer}
    _algo_log_files   = {}   # {ctrl_name: file handle}
    _algo_log_T_ref_ic = {}  # {ctrl_name: Transform}  ref-frame -> controller-IMU
    if _algo_log_dir:
        _algo_log_path = Path(_algo_log_dir)
        _algo_log_path.mkdir(parents=True, exist_ok=True)
        for ctrl_name in enabled_ctrls:
            _algo_log_T_ref_ic[ctrl_name] = Transform(_DIAG_FLIP, np.zeros(3))

            f = open(_algo_log_path / f"{ctrl_name}_algorithm_log.csv", "w", newline="")
            w = csv.writer(f)
            w.writerow(["#timestamp_ns", "p_x", "p_y", "p_z", "q_w", "q_x", "q_y", "q_z"])
            _algo_log_files[ctrl_name]   = f
            _algo_log_writers[ctrl_name] = w
        logger.bind(cat="startup").info(
            f"Algorithm-log CSVs → {_algo_log_path}/<ctrl_name>_algorithm_log.csv")

    # ── Mocap ground-truth export: one T_headsetImu_ctrlImu(t) log per
    # controller with both its own and the headset's mocap loaded (see
    # src/mocap_data.py.relative_pose) -- written every processed frame,
    # independent of whether vision tracking accepted a pose that frame, since
    # it's derived purely from mocap.
    _mocap_log_dir     = debug_cfg.get("mocap_log_dir")
    _mocap_log_writers = {}   # {ctrl_name: csv.writer}
    _mocap_log_files   = {}   # {ctrl_name: file handle}
    if _mocap_log_dir and "headset" in device_mocap:
        _mocap_log_path = Path(_mocap_log_dir)
        _mocap_log_path.mkdir(parents=True, exist_ok=True)
        for ctrl_name in enabled_ctrls:
            if ctrl_name not in device_mocap:
                continue
            f = open(_mocap_log_path / f"{ctrl_name}_mocap_gt.csv", "w", newline="")
            w = csv.writer(f)
            w.writerow(["#timestamp_ns", "p_x", "p_y", "p_z", "q_w", "q_x", "q_y", "q_z"])
            _mocap_log_files[ctrl_name]   = f
            _mocap_log_writers[ctrl_name] = w
        if _mocap_log_writers:
            logger.bind(cat="startup").info(
                f"Mocap ground-truth CSVs → {_mocap_log_path}/<ctrl_name>_mocap_gt.csv")

    _n_frames = count_images(config["data"])
    for frame_idx, batch in enumerate(tqdm(get_data(config["data"]), total=_n_frames)):
        img_path, cam_images = batch[0][0], batch[0][1]
        if img_path.name == "58750954068441.png":
            pass
        # cam_images: {cam_idx: numpy array}

        # Real capture timestamp (nanoseconds) — filenames encode it directly, and
        # consecutive frames are NOT uniformly spaced (confirmed: alternates between
        # substantially different gaps), so pose extrapolation uses this exact
        # elapsed time rather than assuming one frame = one uniform step.
        frame_ts_ns = int(img_path.stem)

        for ctrl_name, _mocap_writer in _mocap_log_writers.items():
            T_gt = relative_pose(device_mocap["headset"], device_mocap[ctrl_name], frame_ts_ns)
            if T_gt is None:
                continue
            _gqx, _gqy, _gqz, _gqw = Rotation.from_matrix(T_gt.R).as_quat()
            _mocap_writer.writerow([
                frame_ts_ns,
                f"{T_gt.t[0]:.6f}", f"{T_gt.t[1]:.6f}", f"{T_gt.t[2]:.6f}",
                f"{_gqw:.8f}", f"{_gqx:.8f}", f"{_gqy:.8f}", f"{_gqz:.8f}",
            ])

        proj_hints, vel_hints, radius_hints, search_eligible = tracking_system.get_predicted_led_projections_per_camera(frame_ts_ns)
        primary_cams       = tracking_system.get_designated_primary_cameras()
        ctrl_names_ordered = tracking_system.get_ctrl_processing_order()
        _mask_margin       = int(config["blob_detection"].get("blob_cross_mask_margin_px", 5))

        # per_ctrl_blobs: {ctrl_name: {cam_idx: BlobResult}}
        per_ctrl_blobs: dict = {}
        frame_blob_vis: dict = {}
        # {ctrl_name: [cam_idx, ...]} — cameras skipped this frame (out-of-
        # scope: no prediction while the controller has one via another
        # camera). No detect() call means no canvas, so Rerun would otherwise
        # keep showing whatever was last logged for that camera — frozen,
        # possibly for hundreds of frames — which looks indistinguishable
        # from a permanently-cold detection. See _log_blob_debug's use of this.
        skipped_cams_per_ctrl: dict = {}

        # ── Phase 1: detect blobs for every controller on the original images ──
        _match_cfg = config["matching"]
        _blob_cfg  = config["blob_detection"]
        _base_r    = float(_match_cfg.get("proximity_expansion_px", 8.0))
        # Building the annotated debug canvas has a real per-frame cost, so only
        # pay for it when a sink (local save and/or Rerun logging) wants it.
        # Lives under visualization: (not blob_detection:) alongside `enabled`
        # for convenience — all the viewer/debug-output toggles in one place.
        _visualize_save    = bool(config["visualization"].get("visualize_save", False))
        _visualize_rerun   = bool(config["visualization"].get("visualize_rerun", False))
        _visualize_compute = _visualize_save or _visualize_rerun
        # Independent of the blob-canvas toggles above: vision-vs-fused-vs-IMU-predicted
        # trajectories plus PoseFusionFilter's internal state, logged into the same
        # Rerun recording (see ControllerAnimatorRerun's "Pose Fusion" tab). Needs
        # fusion.enabled: true to have anything meaningful to show; TrackingSystem was
        # already constructed with this same flag (debug_pose_fusion_cfg above), which
        # is what actually gates PoseFusionFilter.debug_snapshot()/predict_dense()'s
        # extra per-frame cost -- this local copy only gates what main.py forwards to
        # the animator.
        _pose_fusion_debug = bool(config["visualization"].get("pose_fusion_debug", {}).get("enabled", False))

        def _run_blob_detect_batch_multi(cam_kwargs_per_ctrl: dict, images_override: dict = None):
            """Detect blobs for every controller across their cameras in a
            single batched pool round — flattens {ctrl_name: {cam_idx: kwargs}}
            into one {(ctrl_name, cam_idx): future} submission so multiple
            controllers' per-camera detection work overlaps in the pool
            instead of round-tripping it once per controller. Blob detection
            has no cross-controller coupling (no blob claiming happens here),
            and the worker registry is already keyed only by cam_idx (shared
            safely — see run_blob_detect's docstring on _memory round-
            tripping), so this batching is safe unconditionally, every frame.

            One task per (controller, camera) pair, not one per distinct
            camera — tried grouping multiple controllers sharing a camera
            into a single task (one image transmission, but their detect()
            calls then run sequentially inside that one worker instead of
            concurrently on separate workers) and measured it: ~45ms/frame
            vs ~26-29ms/frame on the two-controller dataset where both
            controllers commonly share a camera — the parallelism lost by
            serializing two detect() calls onto one worker outweighs the
            (comparatively small, ~300KB crop) IPC saved by not re-sending
            the image. Reverted; kept per-pair submission.

            Per-pair does NOT mean one detect() call per pair, though: pairs
            whose inputs are provably identical (same camera, no image
            override, no established cold-path memory, same kwargs — see the
            dedup grouping below) collapse to a single submitted task, with
            its result copied to every other controller in the group. This is
            distinct from the reverted grouping above — that merged two
            *different* detect() calls into one serialized task; this skips
            the second call entirely, since two simultaneously cold
            controllers sharing a camera are running the exact same
            computation, not independent work.

            cam_kwargs_per_ctrl: {ctrl_name: {cam_idx: {predicted_leds,
            local_search_radius_px, threshold_scale, velocity_px}}} — only
            predicted_leds is required per camera, the rest default to the
            cold-path values.

            images_override: optional {ctrl_name: {cam_idx: image}} — used by
            the cross-controller blackout (_build_blackout_images) to hand a
            specific (controller, camera) call a modified image instead of
            the raw cam_images[cam_idx]; any pair not present here falls
            through to cam_images unchanged.

            Returns ({ctrl_name: {cam_idx: (BlobResult, canvases)}},
            {ctrl_name: {cam_idx: elapsed_ms}}).
            """
            img_path_arg = img_path if _visualize_save else None
            results_by_ctrl: dict = {c: {} for c in cam_kwargs_per_ctrl}
            ms_by_ctrl: dict = {c: {} for c in cam_kwargs_per_ctrl}

            def _image_for(ctrl_name, cam_idx):
                return (images_override or {}).get(ctrl_name, {}).get(cam_idx, cam_images[cam_idx])

            # ── Dedup: two-or-more controllers sharing a camera whose detect()
            # inputs are provably identical — no image override, the same
            # predicted_leds/radius/threshold/velocity kwargs, and equal
            # cold-path EMA memory (both empty/never-established, or both
            # holding the same pixel_threshold/required_threshold/max_area/
            # blob_count — which is what two controllers that have been
            # deduped together since they went cold end up with, since
            # BlobDetector.detect is a pure function of (image,
            # predicted_leds, radius, threshold_scale, velocity_px, memory)
            # and identical inputs produce identical memory going forward)
            # — get ONE detect() call instead of one per controller, with
            # that single result copied to every member below. ctrl_label
            # only affects a debug-canvas filename.
            #
            # Comparing memory *by value* (not just "has any memory been
            # established") matters: an earlier version treated any
            # established memory as disqualifying, which meant two
            # controllers that started cold-cold and got deduped on their
            # first frame would permanently fall back to solo (duplicated)
            # detection from their second cold frame onward, even though
            # their fanned-out memory was still identical — silently
            # doubling blob-detection cost for the rest of a cold-cold run.
            # Only large_blobs is left out of the comparison below — it's
            # write-only in BlobDetector (never read back via _mem), so it
            # can't affect detect()'s output and including it would just
            # risk an unnecessary split (or an elementwise-comparison error
            # on the numpy contour arrays it holds).
            def _mem_key(cam_idx, ctrl_name):
                mem = _cold_memory.get((cam_idx, ctrl_name))
                if not mem:
                    return None
                return (mem.get("pixel_threshold"), mem.get("required_threshold"),
                        mem.get("max_area"), mem.get("blob_count"))

            groups: dict = {}   # dedup_key -> [(ctrl_name, cam_idx, kwargs), ...]
            for ctrl_name, cam_kwargs in cam_kwargs_per_ctrl.items():
                for cam_idx, kwargs in cam_kwargs.items():
                    _has_override = (images_override or {}).get(ctrl_name, {}).get(cam_idx) is not None
                    _has_prior    = kwargs.get("predicted_leds") is not None
                    if _has_override or _has_prior:
                        key = ("solo", ctrl_name, cam_idx)
                    else:
                        key = (
                            "cold", cam_idx,
                            kwargs.get("local_search_radius_px", 0.0),
                            kwargs.get("threshold_scale", 1.0),
                            kwargs.get("velocity_px", 0.0),
                            _mem_key(cam_idx, ctrl_name),
                        )
                    groups.setdefault(key, []).append((ctrl_name, cam_idx, kwargs))

            # One task per group — its first member is the representative
            # actually submitted below; every other member reuses that single
            # result (see the fan-out after each result is collected).
            flat = [members[0] for members in groups.values()]
            _group_by_rep = {(members[0][0], members[0][1]): members for members in groups.values()}
            _dupe_groups = [members for members in groups.values() if len(members) > 1]
            if _dupe_groups:
                logger.bind(cat="blob_detection").debug(
                    "[blob-dedup] " + "  ".join(
                        f"cam{members[0][1]}="
                        f"[{', '.join(c.replace('_controller', '') for c, _, _ in members)}]"
                        for members in _dupe_groups
                    )
                )

            def _fan_out(ctrl_name, cam_idx, det_result, memory_out=None):
                for _mc, _mcam, _ in _group_by_rep[(ctrl_name, cam_idx)][1:]:
                    results_by_ctrl[_mc][_mcam] = det_result
                    ms_by_ctrl[_mc][_mcam] = 0.0
                    if memory_out is not None:
                        _cold_memory[(_mcam, _mc)] = memory_out

            if pool is not None and blob_parallel:
                from src.parallel_search import run_blob_detect
                futures = {}
                t0_by_key = {}
                for ctrl_name, cam_idx, kwargs in flat:
                    ctrl_label = ctrl_name.replace("_controller", "")
                    t0_by_key[(ctrl_name, cam_idx)] = time()
                    futures[(ctrl_name, cam_idx)] = pool.submit(
                        run_blob_detect, cam_idx, ctrl_label, _image_for(ctrl_name, cam_idx),
                        kwargs.get("predicted_leds"),
                        kwargs.get("local_search_radius_px", 0.0),
                        kwargs.get("threshold_scale", 1.0),
                        kwargs.get("velocity_px", 0.0),
                        _visualize_compute, img_path_arg, img_path.name,
                        _cold_memory.get((cam_idx, ctrl_name)),
                    )
                for (ctrl_name, cam_idx), fut in futures.items():
                    result, canvases, memory_out, diag = fut.result()
                    t_result = time()
                    _cold_memory[(cam_idx, ctrl_name)] = memory_out
                    results_by_ctrl[ctrl_name][cam_idx] = (result, canvases)
                    ms_by_ctrl[ctrl_name][cam_idx] = (time() - t0_by_key[(ctrl_name, cam_idx)]) * 1000
                    if debug_config.log_enabled("blob_diag"):
                        t_submit = t0_by_key[(ctrl_name, cam_idx)]
                        t_worker_start, t_compute_start, t_compute_end = diag
                        dispatch_ms = (t_worker_start - t_submit) * 1000
                        compute_ms  = (t_compute_end - t_compute_start) * 1000
                        return_ms   = (t_result - t_compute_end) * 1000
                        total_ms    = (t_result - t_submit) * 1000
                        logger.bind(cat="blob_diag").debug(
                            f"[{ctrl_name} | cam {cam_idx}] blob diag: "
                            f"dispatch={dispatch_ms:.2f}ms  compute={compute_ms:.2f}ms  "
                            f"return={return_ms:.2f}ms  total={total_ms:.2f}ms"
                        )
                    _fan_out(ctrl_name, cam_idx, (result, canvases), memory_out=memory_out)
            else:
                for ctrl_name, cam_idx, kwargs in flat:
                    ctrl_label = ctrl_name.replace("_controller", "")
                    _t0 = time()
                    det_result = blob_detectors[cam_idx].detect(
                        _image_for(ctrl_name, cam_idx),
                        ctrl_label=ctrl_label,
                        predicted_leds=kwargs.get("predicted_leds"),
                        local_search_radius_px=kwargs.get("local_search_radius_px", 0.0),
                        threshold_scale=kwargs.get("threshold_scale", 1.0),
                        velocity_px=kwargs.get("velocity_px", 0.0),
                        visualize=_visualize_compute,
                        img_path=img_path_arg,
                        frame_name=img_path.name,
                    )
                    results_by_ctrl[ctrl_name][cam_idx] = det_result
                    ms_by_ctrl[ctrl_name][cam_idx] = (time() - _t0) * 1000
                    _fan_out(ctrl_name, cam_idx, det_result)
            return results_by_ctrl, ms_by_ctrl

        def _run_blob_detect_batch(ctrl_name, cam_kwargs: dict, images_override: dict = None):
            """Single-controller convenience wrapper around
            _run_blob_detect_batch_multi — used by the mid-Phase-2 cold-
            redetect fallback, which only ever re-detects one controller at
            a time (a single-key batch is a no-op flatten, same behavior as
            before this was extracted).

            images_override here is {cam_idx: image} (single-controller
            shape) — wrapped into the multi form before forwarding.

            Returns ({cam_idx: (BlobResult, canvases)}, {cam_idx: elapsed_ms}).
            """
            _override_multi = {ctrl_name: images_override} if images_override else None
            results_by_ctrl, ms_by_ctrl = _run_blob_detect_batch_multi(
                {ctrl_name: cam_kwargs}, images_override=_override_multi)
            return results_by_ctrl[ctrl_name], ms_by_ctrl[ctrl_name]

        def _build_blackout_images(cold_ctrl_name, cam_ids):
            """Black out, per camera, the union of every *other* enabled
            controller's expected LED neighborhoods (wherever that other
            controller has a real geometric prediction — proj_hints is not
            None — regardless of whether it's actually searching that camera
            itself this frame) before cold_ctrl_name runs full-image cold
            detection there — reuses the exact predicted-LED data and
            per-LED radius formula that other controller's own hybrid-warm
            detection already computes for itself (proj_hints/radius_hints,
            _compute_led_search_radii), so the excluded region is identical
            to what the other controller's own search already covers (or
            would have covered, for a camera the warm-cam-cap demoted but
            didn't null out — see get_predicted_led_projections_per_camera's
            search_eligible docstring) — no separate margin. Sourced from
            each other controller's *extrapolated* prediction (proj_hints,
            computed once at the top of this frame, before any detection or
            matching runs) — no same-frame confirmed-result dependency, so
            this has no ordering requirement between controllers, and works
            identically whether called from Phase 1's initial cold-start or
            Phase 2's mid-frame cold-redetect.

            Returns {cam_idx: image} — only for cameras actually modified; a
            camera with no other controller's prediction on it at all is
            simply absent, so callers fall through to cam_images[cam_idx]
            unchanged (zero extra copy cost).
            """
            if not bool(_blob_cfg.get("cross_controller_blackout", True)):
                return {}
            _depth_k = float(_blob_cfg.get("local_search_depth_k", 0.0))
            out: dict = {}
            for cam_idx in cam_ids:
                img = None
                for other_ctrl in ctrl_names_ordered:
                    if other_ctrl == cold_ctrl_name:
                        continue
                    other_pred = proj_hints.get(cam_idx, {}).get(other_ctrl)
                    if other_pred is None or len(other_pred) == 0:
                        continue
                    base_px = radius_hints.get(cam_idx, {}).get(other_ctrl, _base_r)
                    depths  = other_pred[:, 2].astype(np.float64)
                    search_radii = _compute_led_search_radii(depths, base_px, _depth_k)
                    img = _blackout_neighborhoods(
                        img if img is not None else cam_images[cam_idx],
                        other_pred, search_radii,
                    )
                if img is not None:
                    out[cam_idx] = img
            return out

        # {ctrl_name: bool} — whether ANY camera has a predicted pose for this
        # controller this frame. False means a true cold-start (frame 1, or a
        # fully lost track) where no camera has anything to be "warm" about;
        # Phase 2 uses this to skip a cheap/proximity pass that's guaranteed
        # to fail, and the per-camera out-of-scope skip below only applies
        # when True (a per-camera gap, not a global cold-start where every
        # camera is needed to reacquire track).
        ctrl_has_prior: dict = {}
        # {ctrl_name: ms} — wall-clock blob-detection time for this controller
        # this frame (Phase 1's initial batch, plus Phase 2's cold-re-detect
        # fallback if it fires) — reported alongside pose-search time in the
        # per-frame summary line, since elapsed_per_ctrl alone only ever
        # covered Phase 2 and was easy to mistake for the whole per-frame cost.
        blob_ms_per_ctrl: dict = {}

        # Pass A: figure out each controller's per-camera detection kwargs
        # (and out-of-scope skips) independently — no cross-controller
        # dependency here, so this loop stays per-controller; only the actual
        # pool dispatch below is batched across controllers.
        _cam_kwargs_per_ctrl: dict = {}
        for ctrl_name in ctrl_names_ordered:
            per_ctrl_blobs[ctrl_name] = {}

            _thr_k   = float(_blob_cfg.get("velocity_threshold_k", 0.0))
            _thr_min = float(_blob_cfg.get("velocity_threshold_min_factor", 0.4))
            _cam_kwargs = {}
            _skipped_cams = []
            _ctrl_has_prior = any(
                search_eligible.get(c, {}).get(ctrl_name, False)
                for c in cameras if c in cam_images
            )
            ctrl_has_prior[ctrl_name] = _ctrl_has_prior
            for cam_idx in cameras:
                if cam_idx not in cam_images:
                    continue
                _pred = proj_hints.get(cam_idx, {}).get(ctrl_name)
                if not search_eligible.get(cam_idx, {}).get(ctrl_name, False) and _ctrl_has_prior:
                    # Out of scope for this extrapolated view this frame — other
                    # cameras DO have a prior, so this isn't a cold-start; skip
                    # blob detection for this camera entirely (and, via
                    # _filter_cam's None/empty handling in TrackingSystem.update,
                    # pose search too) rather than paying for a cold/hybrid
                    # detection pass whose predicted LEDs are known unusable.
                    # search_eligible (not proj_hints-is-None) is the source of
                    # truth here: a camera the warm-cam-cap demoted still has a
                    # real, non-None proj_hints entry (kept so
                    # _build_blackout_images can use it below), but is not
                    # itself searched — see get_predicted_led_projections_per_camera's
                    # docstring.
                    per_ctrl_blobs[ctrl_name][cam_idx] = BlobResult.empty()
                    _skipped_cams.append(cam_idx)
                    continue
                _v_px = vel_hints.get(cam_idx, {}).get(ctrl_name, 0.0)
                _cam_kwargs[cam_idx] = dict(
                    predicted_leds=_pred,
                    local_search_radius_px=radius_hints.get(cam_idx, {}).get(ctrl_name, _base_r),
                    threshold_scale=(max(1.0 / (1.0 + _thr_k * _v_px), _thr_min) if _thr_k > 0 else 1.0),
                    velocity_px=_v_px,
                )

            skipped_cams_per_ctrl[ctrl_name] = list(_skipped_cams)
            _cam_kwargs_per_ctrl[ctrl_name] = _cam_kwargs

        # Cold-warm blackout: a controller with no prior at all this frame
        # (true cold-start) gets every *other* warm controller's expected LED
        # neighborhoods blacked out of its own cameras before it runs
        # full-image cold detection — see _build_blackout_images.
        _images_override: dict = {}
        for ctrl_name in ctrl_names_ordered:
            if ctrl_has_prior[ctrl_name]:
                continue
            _blackout = _build_blackout_images(ctrl_name, _cam_kwargs_per_ctrl[ctrl_name].keys())
            if _blackout:
                _images_override[ctrl_name] = _blackout

        # Pass B: one combined pool round across every controller's cameras —
        # this is what actually lets two controllers' blob detection overlap
        # instead of round-tripping the pool once per controller.
        _t_blob0 = time()
        _phase1_results_per_ctrl, _warm_ms_per_ctrl = _run_blob_detect_batch_multi(
            _cam_kwargs_per_ctrl, images_override=_images_override)
        _blob_batch_ms = (time() - _t_blob0) * 1000

        for ctrl_name in ctrl_names_ordered:
            _phase1_results  = _phase1_results_per_ctrl[ctrl_name]
            _warm_ms_per_cam = _warm_ms_per_ctrl[ctrl_name]
            # Per-controller display estimate only (that controller's slowest
            # camera this round) — the true combined wall-clock for the whole
            # batch is _blob_batch_ms, used below for the slow-frame check so
            # concurrent controllers' time isn't double-counted.
            blob_ms_per_ctrl[ctrl_name] = max(_warm_ms_per_cam.values(), default=0.0)
            for cam_idx, (det_result_0, det_result_1) in _phase1_results.items():
                per_ctrl_blobs[ctrl_name][cam_idx] = det_result_0
                if det_result_1:
                    frame_blob_vis.setdefault(ctrl_name, {})[cam_idx] = det_result_1
            _warm_str = "  ".join(f"cam{c}={ms:.1f}ms" for c, ms in _warm_ms_per_cam.items())
            _skipped_cams = skipped_cams_per_ctrl[ctrl_name]
            if _skipped_cams:
                _skip_str = "  ".join(f"cam{c}=skip(out-of-scope)" for c in _skipped_cams)
                _warm_str = f"{_warm_str}  {_skip_str}" if _warm_str else _skip_str
            _mode_label = "warm detect" if ctrl_has_prior[ctrl_name] else "cold detect"
            logger.bind(cat="timings").info(f"[{ctrl_name}] {_mode_label}: {_warm_str}")

        # ── Phase 2: track controllers in order, filtering matched blobs at the
        # centroid level (no image copy / pixel drawing needed) ─────────────────
        results         = {}
        elapsed_per_ctrl = {}

        def _exclude_claimed_blobs(ctrl_idx, ctrl_name):
            """Drop blobs from ctrl_name's per-camera detections that already
            matched an earlier-processed controller's LEDs this frame — must be
            re-run after any re-detection of ctrl_name's own blobs, since a fresh
            detection pass isn't aware of other controllers' claims."""
            if ctrl_idx == 0:
                return
            for cam_idx in list(per_ctrl_blobs[ctrl_name]):
                curr = per_ctrl_blobs[ctrl_name][cam_idx]
                if len(curr) == 0:
                    continue
                keep = np.ones(len(curr), dtype=bool)
                for prev_ctrl in ctrl_names_ordered[:ctrl_idx]:
                    sol = results.get(prev_ctrl)
                    if sol is None:
                        continue
                    primary_cam = sol["primary_cam"]
                    if cam_idx == primary_cam:
                        matched_pairs = sol["assignment"]
                    elif cam_idx in (sol.get("aux_assignments") or {}):
                        matched_pairs = sol["aux_assignments"][cam_idx]
                    else:
                        continue
                    src = per_ctrl_blobs[prev_ctrl].get(cam_idx)
                    if not matched_pairs or src is None:
                        continue
                    m_idx = [b for b, _ in matched_pairs]
                    dists = np.linalg.norm(
                        curr.centroids[:, None, :] - src.centroids[m_idx][None, :, :], axis=2
                    )
                    too_close = (dists < (src.radii[m_idx] + _mask_margin)[None, :]).any(axis=1)
                    keep &= ~too_close
                per_ctrl_blobs[ctrl_name][cam_idx] = curr.filter(keep)

        def _update_ctrl(ctrl_name, allow_brute=True, force_brute=False):
            _ctrl_blobs = per_ctrl_blobs[ctrl_name]
            sol_map = tracking_system.update(
                {},
                frame_ts_ns=frame_ts_ns,
                per_ctrl_observations={ctrl_name: {c: r.centroids    for c, r in _ctrl_blobs.items()}},
                per_ctrl_radii=        {ctrl_name: {c: r.radii        for c, r in _ctrl_blobs.items()}},
                per_ctrl_brightnesses= {ctrl_name: {c: r.brightnesses for c, r in _ctrl_blobs.items()}},
                ctrl_name_filter=ctrl_name,
                allow_brute=allow_brute,
                force_brute=force_brute,
            )
            return sol_map.get(ctrl_name)

        # ── Warm-warm fast path: when every enabled controller starts this
        # frame with a prior, batch their cheap-search work into one pool
        # round instead of processing controllers one at a time — this is
        # what actually lets two warm controllers' pose search overlap. Any
        # controller whose cheap pass doesn't fully succeed here (sol is
        # None) falls through to the existing sequential loop below,
        # unchanged, exactly as it would on any other cheap-search failure.
        # A frame with any cold controller — including every frame with only
        # one controller enabled, today's production config — never takes
        # this branch. The batch call itself never triggers brute-force or
        # cross-controller occlusion reasoning; both stay explicitly out of
        # scope for the warm-warm fast path.
        _all_warm = len(ctrl_names_ordered) >= 2 and all(
            ctrl_has_prior[c] for c in ctrl_names_ordered
        )
        _fallback_ctrls = list(ctrl_names_ordered)
        _pose_phase_wall_s = 0.0
        if _all_warm:
            _t_pose_batch0 = time()
            _batch_results = tracking_system.update_warm_batch(
                ctrl_names_ordered,
                per_ctrl_observations={
                    c: {cm: r.centroids for cm, r in per_ctrl_blobs[c].items()}
                    for c in ctrl_names_ordered
                },
                per_ctrl_radii={
                    c: {cm: r.radii for cm, r in per_ctrl_blobs[c].items()}
                    for c in ctrl_names_ordered
                },
                per_ctrl_brightnesses={
                    c: {cm: r.brightnesses for cm, r in per_ctrl_blobs[c].items()}
                    for c in ctrl_names_ordered
                },
                frame_ts_ns=frame_ts_ns,
            )
            _pose_batch_ms = (time() - _t_pose_batch0) * 1000
            _pose_phase_wall_s += _pose_batch_ms / 1000
            _fallback_ctrls = []
            logger.bind(cat="batch_orchestration").debug(
                f"[warm-batch] {len(ctrl_names_ordered)} controllers in one pool round: "
                f"{_pose_batch_ms:.1f}ms  succeeded=[{', '.join(c for c in ctrl_names_ordered if _batch_results.get(c) is not None)}]"
            )
            for ctrl_name in ctrl_names_ordered:
                sol = _batch_results.get(ctrl_name)
                results[ctrl_name] = sol
                # Display estimate only (even split of the shared batch time)
                # — _pose_phase_wall_s above already holds the real combined
                # wall-clock, so this split isn't double-counted in the
                # slow-frame total below.
                elapsed_per_ctrl[ctrl_name] = _pose_batch_ms / 1000 / len(ctrl_names_ordered)
                if sol is None:
                    _fallback_ctrls.append(ctrl_name)

        _redetect_ms_total = 0.0
        _true_cold_ctrls: list = []
        for ctrl_idx, ctrl_name in enumerate(ctrl_names_ordered):
            if ctrl_name not in _fallback_ctrls:
                continue
            _exclude_claimed_blobs(ctrl_idx, ctrl_name)

            if not ctrl_has_prior[ctrl_name]:
                # True cold-start (frame 1, or a fully lost track): no camera
                # had a prediction, so a cheap/proximity pass is guaranteed to
                # fail (search_cheap has nothing to search against) and blobs
                # are already cold from Phase 1. Defer to the batched pass
                # below instead of a sequential _update_ctrl(force_brute=True)
                # call here, so every simultaneously-cold controller's tier
                # rounds share one pool round-trip (see
                # TrackingSystem.update_cold_batch) instead of each paying for
                # its own full round-trip sequence one after another.
                # get_ctrl_processing_order sorts every has-prior controller
                # ahead of every no-prior one, so ctrl_names_ordered's cold
                # controllers are always a contiguous suffix — nothing
                # processed earlier in this loop can depend on a cold
                # controller's result, so deferring is safe. The
                # _exclude_claimed_blobs call above still runs as a cheap
                # blob-level pre-filter against any controller already
                # committed earlier in this same frame; the batch call below
                # additionally passes every such committed solution as a
                # fixed candidate to update_cold_batch's post-hoc conflict
                # resolution, so a cold controller's brute search losing to
                # (sharing a blob with, or geometrically occluding) an
                # already-tracked warm controller is caught even when the
                # cheap pre-filter alone wouldn't have excluded it — not just
                # among cold-cold controllers themselves.
                _true_cold_ctrls.append(ctrl_name)
                continue

            t0 = time()
            _redetect_s = 0.0
            sol = _update_ctrl(ctrl_name, allow_brute=False)

            if sol is None:
                # Every camera's cheap (proximity/prior_constrained) search failed —
                # the warm-path per-LED ROIs were centered on an extrapolated pose
                # that just proved untrustworthy, so brute-force against those SAME
                # blobs has no better chance: a blob outside a wrong ROI was never
                # detected at all. Re-detect this controller's blobs cold (full-image,
                # no prior — same as frame 1) before falling back to brute, and add the
                # cold-path (pass1/pass2) canvases to this frame's debug view alongside
                # the failed warm-path ("local") one, instead of replacing it — seeing
                # what the failed proximity attempt looked at is exactly what's needed
                # to understand why it missed.
                _cold_cams = [c for c in cameras if c in cam_images]
                _warm_canvases = {
                    cam_idx: frame_blob_vis.get(ctrl_name, {}).get(cam_idx)
                    for cam_idx in _cold_cams
                }
                _t_redetect0 = time()
                _cold_results, _cold_ms_per_cam = _run_blob_detect_batch(
                    ctrl_name, {c: {"predicted_leds": None} for c in _cold_cams},
                    images_override=_build_blackout_images(ctrl_name, _cold_cams),
                )
                # Extra blob-detection work triggered mid-Phase-2 — counts
                # toward this controller's total blob time, not pose time
                # (subtracted from elapsed_per_ctrl below).
                _redetect_s = time() - _t_redetect0
                blob_ms_per_ctrl[ctrl_name] = blob_ms_per_ctrl.get(ctrl_name, 0.0) + _redetect_s * 1000
                _redetect_ms_total += _redetect_s * 1000
                for cam_idx, (det_result_0, det_result_1) in _cold_results.items():
                    per_ctrl_blobs[ctrl_name][cam_idx] = det_result_0
                    _merged_canvases = dict(_warm_canvases.get(cam_idx) or {})
                    _merged_canvases.update(det_result_1 or {})
                    if _merged_canvases:
                        frame_blob_vis.setdefault(ctrl_name, {})[cam_idx] = _merged_canvases
                    else:
                        frame_blob_vis.get(ctrl_name, {}).pop(cam_idx, None)
                _cold_str = "  ".join(f"cam{c}={ms:.1f}ms" for c, ms in _cold_ms_per_cam.items())
                logger.bind(cat="timings").info(f"[{ctrl_name}] cold re-detect (warm proximity lost): {_cold_str}")

                _exclude_claimed_blobs(ctrl_idx, ctrl_name)
                # Straight to brute — no pose_prior, same as a cold-start first
                # frame. Retrying cheap search here would just fail again for the
                # same reason it failed above: cheap depends on the same
                # extrapolated pose that's already proven untrustworthy, cold
                # blobs or not.
                sol = _update_ctrl(ctrl_name, force_brute=True)

            elapsed_per_ctrl[ctrl_name] = (time() - t0) - _redetect_s
            _pose_phase_wall_s += elapsed_per_ctrl[ctrl_name]
            results[ctrl_name] = sol

        # ── Cold-cold batch: every controller deferred above (true cold-start,
        # no prior anywhere) shares one set of pool tier-rounds instead of the
        # one-controller-at-a-time sequence the loop above used to run for
        # each. Post-hoc conflict resolution (shared blobs / cross-occlusion
        # between simultaneously-solved candidates, AND against every
        # controller already committed earlier this same frame via
        # committed_solutions) lives inside update_cold_batch — see
        # TrackingSystem._resolve_cold_conflicts.
        if _true_cold_ctrls:
            _t_cold_batch0 = time()
            _cold_results = tracking_system.update_cold_batch(
                _true_cold_ctrls,
                per_ctrl_observations={
                    c: {cm: r.centroids for cm, r in per_ctrl_blobs[c].items()}
                    for c in _true_cold_ctrls
                },
                per_ctrl_radii={
                    c: {cm: r.radii for cm, r in per_ctrl_blobs[c].items()}
                    for c in _true_cold_ctrls
                },
                per_ctrl_brightnesses={
                    c: {cm: r.brightnesses for cm, r in per_ctrl_blobs[c].items()}
                    for c in _true_cold_ctrls
                },
                frame_ts_ns=frame_ts_ns,
                committed_solutions={
                    c: results[c] for c in ctrl_names_ordered
                    if c not in _true_cold_ctrls and results.get(c) is not None
                },
                committed_observations={
                    c: {cm: r.centroids for cm, r in per_ctrl_blobs[c].items()}
                    for c in ctrl_names_ordered
                    if c not in _true_cold_ctrls and results.get(c) is not None
                },
                committed_radii={
                    c: {cm: r.radii for cm, r in per_ctrl_blobs[c].items()}
                    for c in ctrl_names_ordered
                    if c not in _true_cold_ctrls and results.get(c) is not None
                },
            )
            _cold_batch_ms = (time() - _t_cold_batch0) * 1000
            _pose_phase_wall_s += _cold_batch_ms / 1000
            logger.bind(cat="batch_orchestration").debug(
                f"[cold-batch] {len(_true_cold_ctrls)} controllers in one pool round: "
                f"{_cold_batch_ms:.1f}ms  "
                f"succeeded=[{', '.join(c for c in _true_cold_ctrls if _cold_results.get(c) is not None)}]"
            )
            for ctrl_name in _true_cold_ctrls:
                sol = _cold_results.get(ctrl_name)
                results[ctrl_name] = sol
                # Display estimate only (even split of the shared batch time)
                # — _pose_phase_wall_s above already holds the real combined
                # wall-clock, same convention as the warm-batch split above.
                elapsed_per_ctrl[ctrl_name] = _cold_batch_ms / 1000 / len(_true_cold_ctrls)

        blobs_frame = {ctrl: {cam: r.centroids for cam, r in cb.items()}
                      for ctrl, cb in per_ctrl_blobs.items()}
        contours_frame = {ctrl: {cam: r.contours for cam, r in cb.items()}
                         for ctrl, cb in per_ctrl_blobs.items()}

        total_blobs = sum(
            len(r)
            for ctrl_blobs in per_ctrl_blobs.values()
            for r in ctrl_blobs.values()
        )

        T_world_ctrl_frame        = {}
        assignments_frame_out     = {}
        primary_cams_frame_out    = {}
        aux_assignments_frame_out = {}
        camera_importance_frame_out = {}
        frozen_T_world_ctrl_frame = {}
        # Pose-fusion debug tool (visualization.pose_fusion_debug) -- populated only
        # when a controller's sol actually carries these (fusion enabled + the debug
        # flag on, see ControllerTracker._commit_fused_solution); left empty per
        # controller otherwise, same convention as the dicts above.
        vision_T_world_ctrl_frame   = {}
        fusion_debug_frame          = {}
        fusion_imu_path_frame       = {}
        fusion_forced_cold_start_frame = {}

        for ctrl_name in enabled_ctrls:
            sol = results.get(ctrl_name)
            _blob_ms  = blob_ms_per_ctrl.get(ctrl_name, 0.0)
            _pose_ms  = elapsed_per_ctrl.get(ctrl_name, 0.0) * 1000
            _time_str = f"{_blob_ms:.1f}ms[blob] + {_pose_ms:.1f}ms[pose] = {_blob_ms + _pose_ms:.1f}ms"
            if sol:
                T_world_ctrl    = sol["T_world_ctrl"]
                primary_cam_idx = sol.get("primary_cam", 0)
                # This controller's OWN escape hatch just fired this frame (see
                # ControllerTracker._commit_fused_solution's abs_reject-triggered
                # persistent-reject escape hatch): T_world_ctrl here is still just the
                # coasted IMU-only prediction from a rejected candidate, not a real
                # accepted pose -- showing it would be exactly the confident-looking-
                # but-wrong display the user flagged (frame 354: "this should be right
                # away marked as unreliable ... tracking should be lost already").
                # Skip populating the three pose-tracking dicts that drive the 3D mesh
                # display so this controller is hidden THIS SAME FRAME, same as a real
                # tracking-loss frame (T_world_ctrl_frame/frozen_T_world_ctrl_frame
                # absent -- see src/visualization.py's _log_frame). Everything else in
                # this block (CSV writers, frame-summary logging, etc.) is left as-is.
                _forced_cold_start_this_frame = bool(sol.get("fusion_forced_cold_start"))
                if not _forced_cold_start_this_frame:
                    T_world_ctrl_frame[ctrl_name]        = T_world_ctrl
                # Always populated (cheap -- same Transform object, no copy), not just
                # under _pose_fusion_debug: the rerun 3D view's own LED-projection/
                # error overlay needs vision's own pose too (see _log_frame), any time
                # fusion.enabled is on, independent of the separate debug-tool toggle.
                vision_T_world_ctrl_frame[ctrl_name] = sol.get("vision_T_world_ctrl", T_world_ctrl)
                if _vision_pose_csv_writer:
                    _v_T = vision_T_world_ctrl_frame[ctrl_name]
                    _vqx, _vqy, _vqz, _vqw = Rotation.from_matrix(_v_T.R).as_quat()
                    # n_inliers: total matched LED-blob pairs across every camera that
                    # contributed to this solve -- same definition HeuristicPoseFusionFilter's
                    # vision-weight cost term uses (see src/pose_fusion_heuristic.py).
                    _n_inliers = (len(sol.get("assignment") or [])
                                  + sum(len(v) for v in (sol.get("aux_assignments") or {}).values()))
                    _vision_pose_csv_writer.writerow([
                        frame_ts_ns, ctrl_name,
                        f"{_vqx:.8f}", f"{_vqy:.8f}", f"{_vqz:.8f}", f"{_vqw:.8f}",
                        f"{_v_T.t[0]:.6f}", f"{_v_T.t[1]:.6f}", f"{_v_T.t[2]:.6f}",
                        f"{float(sol.get('confidence', 1.0)):.4f}", f"{float(sol.get('error', 0.0)):.4f}",
                        _n_inliers,
                    ])
                if _pose_fusion_debug:
                    if "fusion_debug" in sol:
                        fusion_debug_frame[ctrl_name] = sol["fusion_debug"]
                    if sol.get("fusion_imu_path") is not None:
                        fusion_imu_path_frame[ctrl_name] = sol["fusion_imu_path"]
                    if sol.get("fusion_forced_cold_start"):
                        fusion_forced_cold_start_frame[ctrl_name] = True
                assignments_frame_out[ctrl_name]     = sol["assignment"].copy()
                primary_cams_frame_out[ctrl_name]    = primary_cam_idx
                aux_assignments_frame_out[ctrl_name] = sol.get("aux_assignments")
                camera_importance_frame_out[ctrl_name] = sol.get("camera_importance")
                if not _forced_cold_start_this_frame:
                    last_good_T_world[ctrl_name] = T_world_ctrl
                    frozen_T_world_ctrl_frame[ctrl_name] = T_world_ctrl
                # Gated on fusion_accepted (found in review): a fusion-rejected frame still
                # has a non-None sol (T_world_ctrl is then the filter's own IMU-only
                # prediction), so counting it as "a real valid pose"/"not lost" here would
                # let a controller stuck in a persistent-reject loop (vision keeps finding
                # candidates, the filter keeps failing them) never register as lost --
                # any_valid_pose's end-of-run sanity check and lost_streak's tracking_lost
                # grace-period bookkeeping would both silently treat it as healthy.
                if sol.get("fusion_accepted", True):
                    any_valid_pose[ctrl_name] = True
                    lost_streak[ctrl_name] = 0
                if _pose_csv_writer:
                    # sol['error']/assignment describe the ORIGINAL vision candidate, not
                    # necessarily this row's pose -- on a fusion-rejected frame T_world_ctrl
                    # is the filter's own IMU-only prediction (see _commit_fused_solution),
                    # so pairing it with the discarded candidate's error/inlier-count would
                    # misrepresent this row as a real vision fit (found in review).
                    _fusion_accepted = sol.get("fusion_accepted", True)
                    _qx, _qy, _qz, _qw = Rotation.from_matrix(T_world_ctrl.R).as_quat()
                    _pose_csv_writer.writerow([
                        int(img_path.stem), ctrl_name,
                        f"{_qx:.8f}", f"{_qy:.8f}", f"{_qz:.8f}", f"{_qw:.8f}",
                        f"{T_world_ctrl.t[0]:.6f}", f"{T_world_ctrl.t[1]:.6f}", f"{T_world_ctrl.t[2]:.6f}",
                        # "nan", not "" -- downstream readers (compare_vision_mocap.py,
                        # pnp_certainty_check.py) unconditionally float() this column;
                        # an empty string crashes them the first time they load a
                        # fusion-enabled pose_csv (found in review). float("nan")
                        # parses cleanly and is still an honest "not a real fit" value.
                        (f"{sol['error']:.4f}" if _fusion_accepted else "nan"),
                        (len(sol["assignment"]) if _fusion_accepted else 0),
                    ])
                if ctrl_name in _algo_log_writers:
                    T_Ih_Ic = T_world_ctrl.compose(_algo_log_T_ref_ic[ctrl_name])
                    _aqx, _aqy, _aqz, _aqw = Rotation.from_matrix(T_Ih_Ic.R).as_quat()
                    _algo_log_writers[ctrl_name].writerow([
                        frame_ts_ns,
                        f"{T_Ih_Ic.t[0]:.6f}", f"{T_Ih_Ic.t[1]:.6f}", f"{T_Ih_Ic.t[2]:.6f}",
                        f"{_aqw:.8f}", f"{_aqx:.8f}", f"{_aqy:.8f}", f"{_aqz:.8f}",
                    ])
                if _led_csv_writer:
                    # (cam_idx, matched_pairs) for every camera that actually contributed
                    # to this frame's accepted solve -- primary plus every aux camera,
                    # not just primary (unlike calibration_csv above).
                    _cams_matched = [(primary_cam_idx, sol["assignment"])]
                    for _aux_cam_idx, _aux_pairs in (sol.get("aux_assignments") or {}).items():
                        if _aux_pairs:
                            _cams_matched.append((_aux_cam_idx, _aux_pairs))
                    for _cam_idx, _pairs in _cams_matched:
                        _cam_result = per_ctrl_blobs[ctrl_name].get(_cam_idx)
                        if _cam_result is None:
                            continue
                        for _blob_idx, _led_id in _pairs:
                            _px, _py = _cam_result.centroids[_blob_idx]
                            _led_csv_writer.writerow([
                                frame_ts_ns, _cam_idx, ctrl_name, _led_id,
                                f"{_px:.3f}", f"{_py:.3f}",
                                f"{float(_cam_result.radii[_blob_idx]):.3f}",
                                f"{float(_cam_result.brightnesses[_blob_idx]):.1f}",
                            ])
                primary_cam = sol.get("primary_cam", "?")
                aux_cameras = sol.get("aux_cameras")
                if aux_cameras:
                    _aux_parts = [f"cam{c}:{n}" for c, n in aux_cameras if n > 0]
                    aux_str = ("  aux=[" + ",".join(_aux_parts) + "]") if _aux_parts else ""
                elif sol.get("aux_inliers", 0):
                    aux_str = f"  +{sol['aux_inliers']}aux"
                else:
                    aux_str = ""
                # err/matches gated on fusion_accepted (found in review, same reason as
                # pose_csv above): on a fusion-rejected frame these still describe the
                # DISCARDED vision candidate, not the IMU-predicted pose actually reported.
                _fs_accepted = sol.get("fusion_accepted", True)
                _fs_err_str = f"{sol['error']:.2f}px" if _fs_accepted else "n/a (fusion-rejected)"
                _fs_matches = len(sol["assignment"]) if _fs_accepted else 0
                logger.bind(cat="frame_summary").info(
                            f"[{img_path.name}]  [{ctrl_name}]  {_time_str}  "
                            f"cam={primary_cam}  err={_fs_err_str}  "
                            f"matches={_fs_matches}{aux_str}  "
                            f"method={sol.get('method', '?')}")
                if _csv_writer and sol.get("fusion_accepted", True):
                    # Skipped on a fusion-rejected frame: T_world_ctrl would then be the
                    # filter's own IMU-only prediction while sol["assignment"] still lists
                    # LEDs matched under the DISCARDED vision candidate -- computing
                    # depth/facing_cos from that mismatched pairing would corrupt
                    # calibration-threshold data (found in review).
                    _proj = proj_hints.get(primary_cam_idx, {}).get(ctrl_name)
                    if _proj is not None:  # warm path was active for this camera
                        _cam_result = per_ctrl_blobs[ctrl_name].get(primary_cam_idx)
                        _brts  = _cam_result.brightnesses if _cam_result is not None else None
                        _radii = _cam_result.radii        if _cam_result is not None else None
                        if _brts is not None and _radii is not None:
                            # Depth/facing_cos from the pose actually solved this
                            # frame, not proj_hints' pre-match extrapolation —
                            # the extrapolation is stale exactly when the
                            # controller is rotating/accelerating fast.
                            _led_ids = [led_id for _, led_id in sol["assignment"]]
                            _geom = tracking_system.solved_led_geometry(
                                ctrl_name, primary_cam_idx, T_world_ctrl, _led_ids)
                            _vel_px = vel_hints.get(primary_cam_idx, {}).get(ctrl_name, 0.0)
                            for blob_idx, led_id in sol["assignment"]:
                                if led_id not in _geom:
                                    continue
                                _depth_m, _facing_cos = _geom[led_id]
                                _csv_writer.writerow([
                                    img_path.name, ctrl_name, primary_cam_idx,
                                    led_id,
                                    f"{_depth_m:.5f}",
                                    f"{_facing_cos:.5f}",
                                    f"{_vel_px:.3f}",
                                    f"{float(_brts[blob_idx]):.1f}",
                                    f"{float(np.pi * _radii[blob_idx] ** 2):.2f}",
                                ])
            else:
                # No sol this frame -- report the IMU-only dead-reckoned pose
                # instead of hiding the controller outright, so a short vision
                # gap (occlusion, a missed detection) still shows where the
                # controller actually is. Fails open to the old hide-immediately
                # behavior whenever IMU coverage/g_world/prior state aren't
                # there yet (imu_only_predicted_pose returns None).
                lost_streak[ctrl_name] += 1
                _imu_T = tracking_system.ctrl_trackers[ctrl_name].imu_only_predicted_pose(frame_ts_ns)
                if _imu_T is not None:
                    T_world_ctrl_frame[ctrl_name] = _imu_T
                frozen_T_world_ctrl_frame[ctrl_name] = None
                logger.bind(cat="frame_summary").info(f"[{img_path.name}]  [{ctrl_name}]  {_time_str}  TRACKING LOST")
                # No sol at all this frame -- _commit_fused_solution never ran, so it
                # never got a chance to capture fusion_debug/fusion_imu_path itself.
                # Captured here instead so the debug tool's IMU-predicted-path curve
                # keeps growing across a real full-occlusion gap too, not just a
                # reject streak where vision keeps finding (rejected) candidates
                # (found in review).
                if _pose_fusion_debug:
                    _dbg, _path = tracking_system.ctrl_trackers[ctrl_name].debug_fusion_state(frame_ts_ns)
                    if _dbg is not None:
                        fusion_debug_frame[ctrl_name] = _dbg
                    if _path is not None:
                        fusion_imu_path_frame[ctrl_name] = _path

        if animator is not None:
            _t_rerun0 = time()
            animator.log_frame(
                frame_idx,
                T_world_ctrl_frame,
                assignments_per_ctrl=assignments_frame_out,
                blobs_per_ctrl=blobs_frame,
                contours_per_ctrl=contours_frame,
                primary_cam_per_ctrl=primary_cams_frame_out,
                aux_assignments_per_ctrl=aux_assignments_frame_out,
                camera_importance_per_ctrl=camera_importance_frame_out,
                frozen_T_world_ctrl_per_ctrl=frozen_T_world_ctrl_frame,
                blob_vis_frame=(frame_blob_vis if _visualize_rerun else {}),
                blob_vis_skipped=(skipped_cams_per_ctrl if _visualize_rerun else {}),
                vision_T_world_ctrl_per_ctrl=vision_T_world_ctrl_frame,
                fusion_debug_per_ctrl=fusion_debug_frame,
                fusion_imu_path_per_ctrl=fusion_imu_path_frame,
                fusion_forced_cold_start_per_ctrl=fusion_forced_cold_start_frame,
            )
            logger.bind(cat="timings").info(f"[{img_path.name}]  rerun log_frame: {(time() - _t_rerun0) * 1000:.1f}ms")

        # Built from the true batched wall-clock numbers (_blob_batch_ms,
        # _pose_phase_wall_s), not summed per-controller estimates — summing
        # blob_ms_per_ctrl/elapsed_per_ctrl directly would double-count the
        # shared batch time across controllers once Phase 1 is always
        # batched and Phase 2 batches on warm-warm frames (see _all_warm
        # above; those two dicts stay purely for the per-controller display
        # line, where a small approximation is fine).
        _total_frame_s = _blob_batch_ms / 1000 + _redetect_ms_total / 1000 + _pose_phase_wall_s
        if out_slow is not None and _total_frame_s > SLOW_MATCH_THRESHOLD_S:
            copy(img_path, out_slow / img_path.name)
            logger.bind(cat="timings").info(f"  → saved to deep_search_required (slow: {_total_frame_s:.1f}s)")
        if out_tracking_lost is not None and any(n not in T_world_ctrl_frame for n in enabled_ctrls):
            copy(img_path, out_tracking_lost / img_path.name)

    if _csv_file:
        _csv_file.close()
        logger.bind(cat="startup").info(f"Calibration CSV saved → {_csv_path}")

    if _pose_csv_file:
        _pose_csv_file.close()
        logger.bind(cat="startup").info(f"Pose CSV saved → {_pose_csv_path}")

    if _vision_pose_csv_file:
        _vision_pose_csv_file.close()
        logger.bind(cat="startup").info(f"Vision pose CSV saved → {_vision_pose_csv_path}")

    if _led_csv_file:
        _led_csv_file.close()
        logger.bind(cat="startup").info(f"LED-detections CSV saved → {_led_csv_path}")

    for f in _algo_log_files.values():
        f.close()
    if _algo_log_files:
        logger.bind(cat="startup").info(f"Algorithm-log CSVs saved → {_algo_log_dir}")

    for f in _mocap_log_files.values():
        f.close()
    if _mocap_log_files:
        logger.bind(cat="startup").info(f"Mocap ground-truth CSVs saved → {_mocap_log_dir}")

    if tracking_system._self_cal is not None:
        tracking_system._self_cal.run()

    # ── Sanity check ───────────────────────────────────────────────────────
    for ctrl_name in enabled_ctrls:
        if not any_valid_pose[ctrl_name]:
            logger.warning(f"[{ctrl_name}] No valid poses found in the entire sequence.")

    if animator is not None:
        animator.finish()

    tracking_system.shutdown()


if __name__ == "__main__":
    main()
