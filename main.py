import csv
import sys
from pathlib import Path
from shutil import copy
from time import time

import numpy as np
import rerun as rr

from loguru import logger
from tqdm import tqdm

from src import debug_config
from src.blob_detector import BlobDetector, BlobResult
from src.camera import Camera
from src.controller import ControllerModel, TrackingSystem, create_leds_from_config, mirror_primitives
from src.debug_config import DebugMode
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

    # ── Debug mode: auto-detect from the data path ─────────────────────────
    # SEQUENTIAL: full sequential repo  → minimal logs, copy slow/lost frames
    # DEEP:       tracking_lost or deep_search_required → verbose matching logs
    data_root = Path(config["data"]["root"])

    debug_cfg = config.get("debug", {})

    mode = (DebugMode.DEEP
            if config["debug"]["mode_active"]
            else DebugMode.SEQUENTIAL)

    debug_config.configure(
        mode           = mode,
        verbose_all    = bool(debug_cfg.get("verbose_all", False)),
        log_best       = bool(debug_cfg.get("log_best", True)),
        debug_led_ids  = debug_cfg.get("debug_led_ids") or None,
        debug_blob_ids = debug_cfg.get("debug_blob_ids") or None,
    )

    logger.remove()
    if mode == DebugMode.SEQUENTIAL:
        logger.add(sys.stderr, level="INFO",
                   format="<green>{time:HH:mm:ss}</green> | {message}")
    else:
        logger.add(sys.stderr, level="DEBUG",
                   format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}")

    logger.info(f"mode={mode.value}  data={data_root}")

    # ── Output directories (sequential mode only) ──────────────────────────
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

    tracking_system = TrackingSystem(
        list(enabled_ctrls.values()), list(cameras.values()),
        matching_cfg=config.get("matching", {}),
        geometry_cfg=config.get("geometry", {}),
        geometry_cfg_per_ctrl=geo_cfg_per_ctrl,
        self_calibration_cfg=config.get("self_calibration", {}),
        blob_detection_cfg=config["blob_detection"],
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
        logger.info(f"Calibration CSV → {_csv_path}")

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

        proj_hints, vel_hints, radius_hints = tracking_system.get_predicted_led_projections_per_camera(frame_ts_ns)
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
        _visualize_save    = bool(_blob_cfg.get("visualize_save", False))
        _visualize_rerun   = bool(_blob_cfg.get("visualize_rerun", False))
        _visualize_compute = _visualize_save or _visualize_rerun

        def _run_blob_detect_batch(ctrl_name, cam_kwargs: dict):
            """Detect blobs for one controller across the given cameras — in
            parallel (one task per camera, submitted to the shared pool) when
            available, else sequentially against the local `blob_detectors`.

            cam_kwargs: {cam_idx: {predicted_leds, local_search_radius_px,
            threshold_scale}} — only predicted_leds is required, the rest default
            to the cold-path values.

            Returns ({cam_idx: (BlobResult, canvases)}, {cam_idx: elapsed_ms}).
            """
            ctrl_label   = ctrl_name.replace("_controller", "")
            img_path_arg = img_path if _visualize_save else None
            results_by_cam: dict = {}
            ms_by_cam: dict = {}
            if pool is not None and blob_parallel:
                from src.parallel_search import run_blob_detect
                futures = {}
                t0_by_cam = {}
                for cam_idx, kwargs in cam_kwargs.items():
                    t0_by_cam[cam_idx] = time()
                    futures[cam_idx] = pool.submit(
                        run_blob_detect, cam_idx, ctrl_label, cam_images[cam_idx],
                        kwargs.get("predicted_leds"),
                        kwargs.get("local_search_radius_px", 0.0),
                        kwargs.get("threshold_scale", 1.0),
                        kwargs.get("velocity_px", 0.0),
                        _visualize_compute, img_path_arg, img_path.name,
                        _cold_memory.get((cam_idx, ctrl_name)),
                    )
                for cam_idx, fut in futures.items():
                    result, canvases, memory_out = fut.result()
                    _cold_memory[(cam_idx, ctrl_name)] = memory_out
                    results_by_cam[cam_idx] = (result, canvases)
                    ms_by_cam[cam_idx] = (time() - t0_by_cam[cam_idx]) * 1000
            else:
                for cam_idx, kwargs in cam_kwargs.items():
                    t0 = time()
                    det_result = blob_detectors[cam_idx].detect(
                        cam_images[cam_idx],
                        ctrl_label=ctrl_label,
                        predicted_leds=kwargs.get("predicted_leds"),
                        local_search_radius_px=kwargs.get("local_search_radius_px", 0.0),
                        threshold_scale=kwargs.get("threshold_scale", 1.0),
                        velocity_px=kwargs.get("velocity_px", 0.0),
                        visualize=_visualize_compute,
                        img_path=img_path_arg,
                        frame_name=img_path.name,
                    )
                    results_by_cam[cam_idx] = det_result
                    ms_by_cam[cam_idx] = (time() - t0) * 1000
            return results_by_cam, ms_by_cam

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

        for ctrl_name in ctrl_names_ordered:
            per_ctrl_blobs[ctrl_name] = {}

            _thr_k   = float(_blob_cfg.get("velocity_threshold_k", 0.0))
            _thr_min = float(_blob_cfg.get("velocity_threshold_min_factor", 0.4))
            _cam_kwargs = {}
            _skipped_cams = []
            _ctrl_has_prior = any(
                proj_hints.get(c, {}).get(ctrl_name) is not None
                for c in cameras if c in cam_images
            )
            ctrl_has_prior[ctrl_name] = _ctrl_has_prior
            for cam_idx in cameras:
                if cam_idx not in cam_images:
                    continue
                _pred = proj_hints.get(cam_idx, {}).get(ctrl_name)
                if _pred is None and _ctrl_has_prior:
                    # Out of scope for this extrapolated view this frame — other
                    # cameras DO have a prior, so this isn't a cold-start; skip
                    # blob detection for this camera entirely (and, via
                    # _filter_cam's None/empty handling in TrackingSystem.update,
                    # pose search too) rather than paying for a cold/hybrid
                    # detection pass whose predicted LEDs are known unusable.
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

            _t_blob0 = time()
            _phase1_results, _warm_ms_per_cam = _run_blob_detect_batch(ctrl_name, _cam_kwargs)
            blob_ms_per_ctrl[ctrl_name] = (time() - _t_blob0) * 1000
            for cam_idx, (det_result_0, det_result_1) in _phase1_results.items():
                per_ctrl_blobs[ctrl_name][cam_idx] = det_result_0
                if det_result_1:
                    frame_blob_vis.setdefault(ctrl_name, {})[cam_idx] = det_result_1
            _warm_str = "  ".join(f"cam{c}={ms:.1f}ms" for c, ms in _warm_ms_per_cam.items())
            if _skipped_cams:
                _skip_str = "  ".join(f"cam{c}=skip(out-of-scope)" for c in _skipped_cams)
                _warm_str = f"{_warm_str}  {_skip_str}" if _warm_str else _skip_str
            _mode_label = "warm detect" if _ctrl_has_prior else "cold detect"
            logger.info(f"[{ctrl_name}] {_mode_label}: {_warm_str}")

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

        for ctrl_idx, ctrl_name in enumerate(ctrl_names_ordered):
            _exclude_claimed_blobs(ctrl_idx, ctrl_name)

            t0 = time()
            _redetect_s = 0.0

            if not ctrl_has_prior[ctrl_name]:
                # True cold-start (frame 1, or a fully lost track): no camera
                # had a prediction, so a cheap/proximity pass is guaranteed to
                # fail (search_cheap has nothing to search against) and blobs
                # are already cold from Phase 1 — skip straight to brute
                # instead of paying for a doomed cheap pass plus an identical
                # redundant re-detect of the same cold blobs.
                sol = _update_ctrl(ctrl_name, force_brute=True)
            else:
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
                    )
                    # Extra blob-detection work triggered mid-Phase-2 — counts
                    # toward this controller's total blob time, not pose time
                    # (subtracted from elapsed_per_ctrl below).
                    _redetect_s = time() - _t_redetect0
                    blob_ms_per_ctrl[ctrl_name] = blob_ms_per_ctrl.get(ctrl_name, 0.0) + _redetect_s * 1000
                    for cam_idx, (det_result_0, det_result_1) in _cold_results.items():
                        per_ctrl_blobs[ctrl_name][cam_idx] = det_result_0
                        _merged_canvases = dict(_warm_canvases.get(cam_idx) or {})
                        _merged_canvases.update(det_result_1 or {})
                        if _merged_canvases:
                            frame_blob_vis.setdefault(ctrl_name, {})[cam_idx] = _merged_canvases
                        else:
                            frame_blob_vis.get(ctrl_name, {}).pop(cam_idx, None)
                    _cold_str = "  ".join(f"cam{c}={ms:.1f}ms" for c, ms in _cold_ms_per_cam.items())
                    logger.info(f"[{ctrl_name}] cold re-detect (warm proximity lost): {_cold_str}")

                    _exclude_claimed_blobs(ctrl_idx, ctrl_name)
                    # Straight to brute — no pose_prior, same as a cold-start first
                    # frame. Retrying cheap search here would just fail again for the
                    # same reason it failed above: cheap depends on the same
                    # extrapolated pose that's already proven untrustworthy, cold
                    # blobs or not.
                    sol = _update_ctrl(ctrl_name, force_brute=True)

            elapsed_per_ctrl[ctrl_name] = (time() - t0) - _redetect_s
            results[ctrl_name] = sol

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
                last_good_T_world[ctrl_name] = T_world_ctrl
                frozen_T_world_ctrl_frame[ctrl_name] = T_world_ctrl
                any_valid_pose[ctrl_name] = True
                primary_cam = sol.get("primary_cam", "?")
                aux_cameras = sol.get("aux_cameras")
                if aux_cameras:
                    _aux_parts = [f"cam{c}:{n}" for c, n in aux_cameras if n > 0]
                    aux_str = ("  aux=[" + ",".join(_aux_parts) + "]") if _aux_parts else ""
                elif sol.get("aux_inliers", 0):
                    aux_str = f"  +{sol['aux_inliers']}aux"
                else:
                    aux_str = ""
                logger.info(f"[{img_path.name}]  [{ctrl_name}]  {_time_str}  "
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
                # controller is between cameras or ambiguous; freeze last pose.
                # Cases 1/2: never tracked or truly out of view → None (hidden).
                last_good = last_good_T_world[ctrl_name]
                frozen_T_world_ctrl_frame[ctrl_name] = (
                    last_good if last_good is not None and total_blobs > 0 else None
                )
                logger.info(f"[{img_path.name}]  [{ctrl_name}]  {_time_str}  TRACKING LOST")

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
                frozen_T_world_ctrl_per_ctrl=frozen_T_world_ctrl_frame,
                blob_vis_frame=(frame_blob_vis if _visualize_rerun else {}),
                blob_vis_skipped=(skipped_cams_per_ctrl if _visualize_rerun else {}),
            )
            logger.info(f"[{img_path.name}]  rerun log_frame: {(time() - _t_rerun0) * 1000:.1f}ms")

        _total_frame_s = sum(elapsed_per_ctrl.values()) + sum(blob_ms_per_ctrl.values()) / 1000
        if out_slow is not None and _total_frame_s > SLOW_MATCH_THRESHOLD_S:
            copy(img_path, out_slow / img_path.name)
            logger.info(f"  → saved to deep_search_required (slow: {_total_frame_s:.1f}s)")
        if out_tracking_lost is not None and any(n not in T_world_ctrl_frame for n in enabled_ctrls):
            copy(img_path, out_tracking_lost / img_path.name)

    if _csv_file:
        _csv_file.close()
        logger.info(f"Calibration CSV saved → {_csv_path}")

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
