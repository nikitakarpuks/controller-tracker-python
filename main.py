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
from src.controller import ControllerModel, TrackingSystem, create_leds_from_config
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
    camera_0  = cameras[0]

    calibration_config    = load_json_config(config["controllers"]["right_controller"]["config_path"])
    right_controller_leds = create_leds_from_config(calibration_config)
    right_controller      = ControllerModel(right_controller_leds, "right_controller")

    tracking_system = TrackingSystem(
        [right_controller], list(cameras.values()),
        matching_cfg=config.get("matching", {}),
        geometry_cfg=config.get("geometry", {}),
    )

    positions_model, normals_model, T_model_ctrl = prepare_model_geometry(
        right_controller_leds, config["visualization"]
    )

    if config["visualization"].get("fine_tune_alignment"):
        mesh = load_trimesh(config["visualization"]["3d_model_path"])
        fine_tune_alignment(right_controller_leds, mesh, config["visualization"])

    # ── Tracking loop ──────────────────────────────────────────────────────
    poses               = []
    assignments         = []
    primary_cams        = []
    aux_assignments_all = []
    blobs               = []
    contours_all        = []
    raw_blobs           = []
    raw_contours_all    = []

    for batch in tqdm(get_data(config["data"])):
        img_path, cam_images = batch[0][0], batch[0][1]
        # cam_images: {cam_idx: numpy array}

        cam_blobs        = {}
        cam_contours     = {}
        cam_radii        = {}
        cam_raw_blobs    = {}
        cam_raw_contours = {}

        for cam_idx, image in cam_images.items():
            blob_centroids, blob_contours, blob_radii, raw_centroids, raw_contours = get_centroids(
                image, config["blob_detection"], visualize=True, img_path=img_path
            )
            cam_blobs[cam_idx]        = blob_centroids.copy()
            cam_contours[cam_idx]     = blob_contours
            cam_radii[cam_idx]        = blob_radii
            cam_raw_blobs[cam_idx]    = raw_centroids.copy()
            cam_raw_contours[cam_idx] = raw_contours

        blobs.append(cam_blobs)
        contours_all.append(cam_contours)
        raw_blobs.append(cam_raw_blobs)
        raw_contours_all.append(cam_raw_contours)

        t0       = time()
        solution = tracking_system.update(
            cam_blobs,
            radii_per_camera=cam_radii,
        ).get("right_controller")
        elapsed  = time() - t0

        if solution:
            # Express pose in the primary camera's frame so the animator uses the
            # correct intrinsics and blobs for projection and error visualization.
            T_world_ctrl    = solution["T_world_ctrl"]
            primary_cam_idx = solution.get("primary_cam", 0)
            primary_camera  = cameras[primary_cam_idx]
            T_primary_ctrl  = primary_camera.T_world_cam.inverse().compose(T_world_ctrl)
            rvec_primary, _ = cv2.Rodrigues(T_primary_ctrl.R.astype(np.float32))
            poses.append((rvec_primary.reshape(3), T_primary_ctrl.t.astype(np.float32)))
            assignments.append(solution["assignment"].copy())
            primary_cams.append(primary_cam_idx)
            aux_assignments_all.append(solution.get("aux_assignments"))
            primary_cam = solution.get("primary_cam", "?")
            aux_cameras = solution.get("aux_cameras")
            if aux_cameras:
                _aux_parts = [f"cam{c}:{n}" for c, n in aux_cameras if n > 0]
                aux_str = ("  aux=[" + ",".join(_aux_parts) + "]") if _aux_parts else ""
            elif solution.get("aux_inliers", 0):
                aux_str = f"  +{solution['aux_inliers']}aux"
            else:
                aux_str = ""
            logger.info(f"[{img_path.name}]  {elapsed:.3f}s  "
                        f"cam={primary_cam}  "
                        f"err={solution['error']:.2f}px  "
                        f"matches={len(solution['assignment'])}{aux_str}  "
                        f"method={solution.get('method', '?')}")
            if out_slow is not None and elapsed > SLOW_MATCH_THRESHOLD_S:
                copy(img_path, out_slow / img_path.name)
                logger.info(f"  → saved to deep_search_required (slow: {elapsed:.1f}s)")
        else:
            poses.append(None)
            assignments.append(None)
            primary_cams.append(None)
            aux_assignments_all.append(None)
            logger.info(f"[{img_path.name}]  {elapsed:.3f}s  TRACKING LOST")
            if out_tracking_lost is not None:
                copy(img_path, out_tracking_lost / img_path.name)

    # ── Sanity check ───────────────────────────────────────────────────────
    if all(p is None for p in poses):
        raise RuntimeError("No valid poses found in the entire sequence.")

    # ── Visualisation ──────────────────────────────────────────────────────
    animator = ControllerAnimatorRerun(
        config["visualization"]["3d_model_path"],
        positions_model,
        normals_model,
        matching_cfg=config.get("matching", {}),
    )
    animator.start(
        poses, assignments, blobs, cameras, T_model_ctrl,
        contours_all=contours_all,
        raw_blobs_all=raw_blobs,
        raw_contours_all=raw_contours_all,
        save_path=config["visualization"].get("save_recording"),
        primary_cams_all=primary_cams,
        aux_assignments_all=aux_assignments_all,
    )


if __name__ == "__main__":
    main()
