import csv
import sys
from pathlib import Path
from shutil import copy
from time import time

import cv2
import numpy as np

from loguru import logger
from tqdm import tqdm

from src import debug_config
from src.blob_detector import BlobDetector, BlobResult
from src.camera import Camera
from src.controller import ControllerModel, TrackingSystem, create_leds_from_config, mirror_primitives
from src.debug_config import DebugMode
from src.load_config import load_yaml_config, load_json_config
from src.preprocess_data import get_data
from src.visualization import (ControllerAnimatorRerun, prepare_model_geometry,
                               fine_tune_alignment, load_trimesh)

SLOW_MATCH_THRESHOLD_S = 1.5


def main():
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
    calib_cfg = load_json_config(config["cameras"]["intrinsics_path"])
    cameras   = {idx: Camera(calib_cfg, camera_idx=idx)
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
    )

    if config["visualization"].get("fine_tune_alignment") and "right_controller" in enabled_ctrls:
        mesh = load_trimesh(config["visualization"]["3d_model_path"])
        fine_tune_alignment(ctrl_leds["right_controller"], mesh, right_ctrl_cfg)

    # ── Tracking loop ──────────────────────────────────────────────────────
    poses_all            = {n: [] for n in enabled_ctrls}
    assignments_all      = {n: [] for n in enabled_ctrls}
    primary_cams_all     = {n: [] for n in enabled_ctrls}
    aux_assignments_all  = {n: [] for n in enabled_ctrls}
    # Per-frame T_world_ctrl for frames where tracking failed but blobs were
    # present (case 3: between cameras / ambiguous).  None for truly invisible
    # frames (case 1/2).  Visualiser holds the last known pose frozen.
    frozen_poses_all     = {n: [] for n in enabled_ctrls}
    last_good_T_world    = {n: None for n in enabled_ctrls}
    blobs        = []
    contours_all = []

    _csv_path = debug_cfg.get("calibration_csv")
    _csv_file = _csv_writer = None
    if _csv_path:
        Path(_csv_path).parent.mkdir(parents=True, exist_ok=True)
        _csv_file = open(_csv_path, "w", newline="")
        _csv_writer = csv.writer(_csv_file)
        _csv_writer.writerow(["frame", "ctrl_name", "cam_idx",
                               "led_id", "depth_m", "facing_cos",
                               "brightness", "area"])
        logger.info(f"Calibration CSV → {_csv_path}")

    for batch in tqdm(get_data(config["data"])):
        img_path, cam_images = batch[0][0], batch[0][1]
        # if img_path.name == "30633726268015.png":
        #     pass
        # cam_images: {cam_idx: numpy array}

        proj_hints, vel_hints = tracking_system.get_predicted_led_projections_per_camera()
        primary_cams       = tracking_system.get_designated_primary_cameras()
        ctrl_names_ordered = tracking_system.get_ctrl_processing_order()
        _mask_margin       = int(config["blob_detection"].get("blob_cross_mask_margin_px", 5))

        # per_ctrl_blobs: {ctrl_name: {cam_idx: BlobResult}}
        per_ctrl_blobs: dict = {}

        # ── Phase 1: detect blobs for every controller on the original images ──
        _match_cfg = config["matching"]
        _blob_cfg  = config["blob_detection"]
        for ctrl_name in ctrl_names_ordered:
            per_ctrl_blobs[ctrl_name] = {}

            for cam_idx in cameras:
                if cam_idx not in cam_images:
                    continue
                predicted_leds = proj_hints.get(cam_idx, {}).get(ctrl_name)
                _base_r     = float(_match_cfg.get("proximity_expansion_px", 8.0))
                _vel_k      = float(_match_cfg.get("proximity_expansion_velocity_k", 0.0))
                _v_px       = vel_hints.get(cam_idx, {}).get(ctrl_name, 0.0)
                _local_r_px = _base_r + _vel_k * _v_px
                _thr_k      = float(_blob_cfg.get("velocity_threshold_k", 0.0))
                _thr_min    = float(_blob_cfg.get("velocity_threshold_min_factor", 0.4))
                _thr_scale  = max(1.0 / (1.0 + _thr_k * _v_px), _thr_min) if _thr_k > 0 else 1.0
                t0 = time()
                per_ctrl_blobs[ctrl_name][cam_idx] = blob_detectors[cam_idx].detect(
                    cam_images[cam_idx],
                    ctrl_label=ctrl_name.replace("_controller", ""),
                    predicted_leds=predicted_leds,
                    local_search_radius_px=_local_r_px,
                    threshold_scale=_thr_scale,
                    visualize=_blob_cfg["visualize"],
                    img_path=img_path,
                )
                logger.info(f"blob detection took {time() - t0} seconds")

        # ── Phase 2: track controllers in order, filtering matched blobs at the
        # centroid level (no image copy / pixel drawing needed) ─────────────────
        results         = {}
        elapsed_per_ctrl = {}

        for ctrl_idx, ctrl_name in enumerate(ctrl_names_ordered):
            # Remove blobs from preceding controllers' LED-matched positions.
            if ctrl_idx > 0:
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

            t0 = time()
            _ctrl_blobs = per_ctrl_blobs[ctrl_name]
            sol_map = tracking_system.update(
                {},
                per_ctrl_observations={ctrl_name: {c: r.centroids    for c, r in _ctrl_blobs.items()}},
                per_ctrl_radii=        {ctrl_name: {c: r.radii        for c, r in _ctrl_blobs.items()}},
                per_ctrl_brightnesses= {ctrl_name: {c: r.brightnesses for c, r in _ctrl_blobs.items()}},
                ctrl_name_filter=ctrl_name,
            )
            elapsed_per_ctrl[ctrl_name] = time() - t0
            results[ctrl_name] = sol_map.get(ctrl_name)

        blobs.append({ctrl: {cam: r.centroids for cam, r in cb.items()}
                      for ctrl, cb in per_ctrl_blobs.items()})
        contours_all.append({ctrl: {cam: r.contours for cam, r in cb.items()}
                             for ctrl, cb in per_ctrl_blobs.items()})

        total_blobs = sum(
            len(r)
            for ctrl_blobs in per_ctrl_blobs.values()
            for r in ctrl_blobs.values()
        )

        for ctrl_name in enabled_ctrls:
            sol = results.get(ctrl_name)
            if sol:
                T_world_ctrl    = sol["T_world_ctrl"]
                primary_cam_idx = sol.get("primary_cam", 0)
                primary_camera  = cameras[primary_cam_idx]
                T_primary_ctrl  = primary_camera.T_world_cam.inverse().compose(T_world_ctrl)
                rvec_primary, _ = cv2.Rodrigues(T_primary_ctrl.R.astype(np.float32))
                poses_all[ctrl_name].append((rvec_primary.reshape(3), T_primary_ctrl.t.astype(np.float32)))
                assignments_all[ctrl_name].append(sol["assignment"].copy())
                primary_cams_all[ctrl_name].append(primary_cam_idx)
                aux_assignments_all[ctrl_name].append(sol.get("aux_assignments"))
                last_good_T_world[ctrl_name] = T_world_ctrl
                frozen_poses_all[ctrl_name].append(T_world_ctrl)
                primary_cam = sol.get("primary_cam", "?")
                aux_cameras = sol.get("aux_cameras")
                if aux_cameras:
                    _aux_parts = [f"cam{c}:{n}" for c, n in aux_cameras if n > 0]
                    aux_str = ("  aux=[" + ",".join(_aux_parts) + "]") if _aux_parts else ""
                elif sol.get("aux_inliers", 0):
                    aux_str = f"  +{sol['aux_inliers']}aux"
                else:
                    aux_str = ""
                logger.info(f"[{img_path.name}]  [{ctrl_name}]  {elapsed_per_ctrl.get(ctrl_name, 0.0):.3f}s  "
                            f"cam={primary_cam}  err={sol['error']:.2f}px  "
                            f"matches={len(sol['assignment'])}{aux_str}  "
                            f"method={sol.get('method', '?')}")
                if _csv_writer:
                    _proj = proj_hints.get(primary_cam_idx, {}).get(ctrl_name)
                    if _proj is not None:  # warm path was active for this camera
                        led_lookup = {
                            int(row[4]): (float(row[2]), float(row[3]))
                            for row in _proj
                        }
                        _cam_result = per_ctrl_blobs[ctrl_name].get(primary_cam_idx)
                        _brts  = _cam_result.brightnesses if _cam_result is not None else None
                        _radii = _cam_result.radii        if _cam_result is not None else None
                        if _brts is not None and _radii is not None:
                            for blob_idx, led_id in sol["assignment"]:
                                if led_id not in led_lookup:
                                    continue
                                _depth_m, _facing_cos = led_lookup[led_id]
                                _csv_writer.writerow([
                                    img_path.name, ctrl_name, primary_cam_idx,
                                    led_id,
                                    f"{_depth_m:.5f}",
                                    f"{_facing_cos:.5f}",
                                    f"{float(_brts[blob_idx]):.1f}",
                                    f"{float(np.pi * _radii[blob_idx] ** 2):.2f}",
                                ])
            else:
                poses_all[ctrl_name].append(None)
                assignments_all[ctrl_name].append(None)
                primary_cams_all[ctrl_name].append(None)
                aux_assignments_all[ctrl_name].append(None)
                # Case 3: had a prior good pose and cameras still see blobs →
                # controller is between cameras or ambiguous; freeze last pose.
                # Cases 1/2: never tracked or truly out of view → None (hidden).
                last_good = last_good_T_world[ctrl_name]
                if last_good is not None and total_blobs > 0:
                    frozen_poses_all[ctrl_name].append(last_good)
                else:
                    frozen_poses_all[ctrl_name].append(None)
                logger.info(f"[{img_path.name}]  [{ctrl_name}]  {elapsed_per_ctrl.get(ctrl_name, 0.0):.3f}s  TRACKING LOST")

        if out_slow is not None and sum(elapsed_per_ctrl.values()) > SLOW_MATCH_THRESHOLD_S:
            copy(img_path, out_slow / img_path.name)
            logger.info(f"  → saved to deep_search_required (slow: {elapsed:.1f}s)")
        if out_tracking_lost is not None and any(poses_all[n][-1] is None for n in enabled_ctrls):
            copy(img_path, out_tracking_lost / img_path.name)

    if _csv_file:
        _csv_file.close()
        logger.info(f"Calibration CSV saved → {_csv_path}")

    if tracking_system._self_cal is not None:
        tracking_system._self_cal.run()

    # ── Sanity check ───────────────────────────────────────────────────────
    for ctrl_name in enabled_ctrls:
        if all(p is None for p in poses_all[ctrl_name]):
            logger.warning(f"[{ctrl_name}] No valid poses found in the entire sequence.")

    # ── Visualisation ──────────────────────────────────────────────────────
    if enabled_ctrls:
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
        animator.start(
            poses_all,
            assignments_all,
            blobs, cameras,
            contours_all=contours_all,
            save_path=config["visualization"].get("save_recording"),
            primary_cams_all=primary_cams_all,
            aux_assignments_all=aux_assignments_all,
            frozen_poses_all=frozen_poses_all,
        )


if __name__ == "__main__":
    main()
