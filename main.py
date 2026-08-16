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
from src.imu_data import load_and_calibrate_controller_imu
from src.load_config import load_yaml_config, load_json_config
from src.preprocess_data import get_data, count_images
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
    right_ctrl_cfg   = config["controllers"]["right_controller"]

    for ctrl_key in ["right_controller", "left_controller"]:
        ctrl_cfg = config["controllers"].get(ctrl_key, {})
        if not ctrl_cfg.get("enabled", False):
            continue
        leds = create_leds_from_config(load_json_config(ctrl_cfg["config_path"]))
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

    # ── IMU (Stage 3): load + calibrate controller gyro/accel for pose prediction
    # and the gravity-alignment diagnostic ───────────────────────────────────────
    # Confirmed mapping (mentor + Stage 1 cross-correlation validation, this
    # recording only): imu1.csv = left controller, imu2.csv = right controller.
    # flip_y @ T_rt.R.T is the axis transform Stage 1 empirically resolved (see
    # src/imu_data.py's module docstring — T_rt's own direction is undocumented
    # in the calibration file). Per-controller lag is Stage 1's measured
    # controller<->camera clock offset on THIS recording (imu_vision_sync_check.py)
    # — a single-clip estimate, re-measure if this ever runs against different data.
    imu_cfg = config.get("imu", {})
    gyro_data:  dict = {}
    accel_data: dict = {}
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

            logger.bind(cat="startup").info(f"[{ctrl_key}] IMU loaded: {len(t_imu)} samples from {imu_path.name}")

    tracking_system = TrackingSystem(
        list(enabled_ctrls.values()), list(cameras.values()),
        matching_cfg=config.get("matching", {}),
        geometry_cfg=config.get("geometry", {}),
        geometry_cfg_per_ctrl=geo_cfg_per_ctrl,
        self_calibration_cfg=config.get("self_calibration", {}),
        blob_detection_cfg=config["blob_detection"],
        gyro_data=gyro_data,
        accel_data=accel_data,
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
        )
        animator.begin(cameras, save_path=config["visualization"].get("save_recording"))

    # ── Tracking loop ──────────────────────────────────────────────────────
    any_valid_pose    = {n: False for n in enabled_ctrls}
    last_good_T_world = {n: None for n in enabled_ctrls}
    # Consecutive lost frames per controller, for capping how long the ghost
    # (frozen last-known pose) stays visible in the visualizer — see the
    # tracking_lost_grace_frames use below.
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
        _pose_csv_writer.writerow(["timestamp_ns", "ctrl_name", "qx", "qy", "qz", "qw", "px", "py", "pz"])
        logger.bind(cat="startup").info(f"Pose CSV → {_pose_csv_path}")

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

        for ctrl_name in enabled_ctrls:
            sol = results.get(ctrl_name)
            _blob_ms  = blob_ms_per_ctrl.get(ctrl_name, 0.0)
            _pose_ms  = elapsed_per_ctrl.get(ctrl_name, 0.0) * 1000
            _time_str = f"{_blob_ms:.1f}ms[blob] + {_pose_ms:.1f}ms[pose] = {_blob_ms + _pose_ms:.1f}ms"
            if sol:
                T_world_ctrl    = sol["T_world_ctrl"]
                primary_cam_idx = sol.get("primary_cam", 0)
                T_world_ctrl_frame[ctrl_name]        = T_world_ctrl
                assignments_frame_out[ctrl_name]     = sol["assignment"].copy()
                primary_cams_frame_out[ctrl_name]    = primary_cam_idx
                aux_assignments_frame_out[ctrl_name] = sol.get("aux_assignments")
                camera_importance_frame_out[ctrl_name] = sol.get("camera_importance")
                last_good_T_world[ctrl_name] = T_world_ctrl
                frozen_T_world_ctrl_frame[ctrl_name] = T_world_ctrl
                any_valid_pose[ctrl_name] = True
                lost_streak[ctrl_name] = 0
                if _pose_csv_writer:
                    _qx, _qy, _qz, _qw = Rotation.from_matrix(T_world_ctrl.R).as_quat()
                    _pose_csv_writer.writerow([
                        int(img_path.stem), ctrl_name,
                        f"{_qx:.8f}", f"{_qy:.8f}", f"{_qz:.8f}", f"{_qw:.8f}",
                        f"{T_world_ctrl.t[0]:.6f}", f"{T_world_ctrl.t[1]:.6f}", f"{T_world_ctrl.t[2]:.6f}",
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
                logger.bind(cat="frame_summary").info(
                            f"[{img_path.name}]  [{ctrl_name}]  {_time_str}  "
                            f"cam={primary_cam}  err={sol['error']:.2f}px  "
                            f"matches={len(sol['assignment'])}{aux_str}  "
                            f"method={sol.get('method', '?')}")
                if _csv_writer:
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
                # Case 3: had a prior good pose and cameras still see blobs →
                # controller is between cameras or ambiguous; freeze last pose,
                # but only for up to tracking_lost_grace_frames consecutive
                # lost frames — beyond that the track is really gone and the
                # ghost must disappear rather than hang around indefinitely.
                # Cases 1/2: never tracked or truly out of view → None (hidden).
                lost_streak[ctrl_name] += 1
                _grace = int(_match_cfg.get('tracking_lost_grace_frames', 1))
                last_good = last_good_T_world[ctrl_name]
                frozen_T_world_ctrl_frame[ctrl_name] = (
                    last_good if last_good is not None and total_blobs > 0
                    and lost_streak[ctrl_name] <= _grace else None
                )
                logger.bind(cat="frame_summary").info(f"[{img_path.name}]  [{ctrl_name}]  {_time_str}  TRACKING LOST")

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
