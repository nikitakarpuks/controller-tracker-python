import sys
from pathlib import Path
from shutil import copy
from time import time

import cv2
import numpy as np

from loguru import logger
from tqdm import tqdm

from src import debug_config
from src.blobs_detection import get_centroids
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
    )

    if config["visualization"].get("fine_tune_alignment") and "right_controller" in enabled_ctrls:
        mesh = load_trimesh(config["visualization"]["3d_model_path"])
        fine_tune_alignment(ctrl_leds["right_controller"], mesh, right_ctrl_cfg)

    # ── Tracking loop ──────────────────────────────────────────────────────
    poses_all           = {n: [] for n in enabled_ctrls}
    assignments_all     = {n: [] for n in enabled_ctrls}
    primary_cams_all    = {n: [] for n in enabled_ctrls}
    aux_assignments_all = {n: [] for n in enabled_ctrls}
    blobs        = []
    contours_all = []

    for batch in tqdm(get_data(config["data"])):
        img_path, cam_images = batch[0][0], batch[0][1]
        # cam_images: {cam_idx: numpy array}

        cam_blobs        = {}
        cam_contours     = {}
        cam_radii        = {}
        cam_brightnesses = {}

        for cam_idx, image in cam_images.items():
            t0 = time()
            blob_centroids, blob_contours, blob_radii, blob_brightnesses, _, _, _ = get_centroids(
                image, config["blob_detection"], visualize=True, img_path=img_path
            )
            t1 = time()
            logger.info(f"blob detection took {t1 - t0} seconds")

            cam_blobs[cam_idx]        = blob_centroids.copy()
            cam_contours[cam_idx]     = blob_contours
            cam_radii[cam_idx]        = blob_radii
            cam_brightnesses[cam_idx] = blob_brightnesses

        blobs.append(cam_blobs)
        contours_all.append(cam_contours)

        t0      = time()
        results = tracking_system.update(cam_blobs, radii_per_camera=cam_radii)
        elapsed = time() - t0

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
                primary_cam = sol.get("primary_cam", "?")
                aux_cameras = sol.get("aux_cameras")
                if aux_cameras:
                    _aux_parts = [f"cam{c}:{n}" for c, n in aux_cameras if n > 0]
                    aux_str = ("  aux=[" + ",".join(_aux_parts) + "]") if _aux_parts else ""
                elif sol.get("aux_inliers", 0):
                    aux_str = f"  +{sol['aux_inliers']}aux"
                else:
                    aux_str = ""
                logger.info(f"[{img_path.name}]  [{ctrl_name}]  {elapsed:.3f}s  "
                            f"cam={primary_cam}  err={sol['error']:.2f}px  "
                            f"matches={len(sol['assignment'])}{aux_str}  "
                            f"method={sol.get('method', '?')}")
            else:
                poses_all[ctrl_name].append(None)
                assignments_all[ctrl_name].append(None)
                primary_cams_all[ctrl_name].append(None)
                aux_assignments_all[ctrl_name].append(None)
                logger.info(f"[{img_path.name}]  [{ctrl_name}]  {elapsed:.3f}s  TRACKING LOST")

        if out_slow is not None and elapsed > SLOW_MATCH_THRESHOLD_S:
            copy(img_path, out_slow / img_path.name)
            logger.info(f"  → saved to deep_search_required (slow: {elapsed:.1f}s)")
        if out_tracking_lost is not None and any(poses_all[n][-1] is None for n in enabled_ctrls):
            copy(img_path, out_tracking_lost / img_path.name)

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
        )


if __name__ == "__main__":
    main()
