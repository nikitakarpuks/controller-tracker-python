import sys
from pathlib import Path
from shutil import copy
from time import time

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
    mode = (DebugMode.DEEP
            if config["debug_mode"]
            else DebugMode.SEQUENTIAL)

    logger.remove()
    if mode == DebugMode.SEQUENTIAL:
        logger.add(sys.stderr, level="INFO",
                   format="<green>{time:HH:mm:ss}</green> | {message}")
    else:
        logger.add(sys.stderr, level="DEBUG",
                   format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}")

    # Debugging options (all independent, mix as needed):
    #   verbose_all=True  → log every P3P hypothesis (no LED/blob filter needed)
    #   log_best=True     → log each new best solution update  (default: on)
    #   debug_led_ids / debug_blob_ids → also log a specific triple in detail
    debug_config.configure(mode)
    # debug_config.configure(mode, verbose_all=True)
    # debug_config.configure(mode, debug_led_ids=[19, 31, 28], debug_blob_ids=[4, 1, 3])

    logger.info(f"mode={mode.value}  data={data_root}")

    # ── Output directories (sequential mode only) ──────────────────────────
    out_slow = out_tracking_lost = None
    if config["split_to_folders"]:
        out_slow          = data_root / "deep_search_required"
        out_tracking_lost = data_root / "tracking_lost"
        out_slow.mkdir(parents=True, exist_ok=True)
        out_tracking_lost.mkdir(parents=True, exist_ok=True)

    # ── Camera & controller setup ──────────────────────────────────────────
    camera_0 = Camera(
        load_json_config(config["cameras"]["intrinsics_path"]),
        camera_idx=0,
    )

    calibration_config    = load_json_config(config["controllers"]["right_controller"]["config_path"])
    right_controller_leds = create_leds_from_config(calibration_config)
    right_controller      = ControllerModel(right_controller_leds, "right_controller")

    tracking_system = TrackingSystem([right_controller], [camera_0])

    positions_model, normals_model, T_model_ctrl = prepare_model_geometry(
        right_controller_leds, config["visualization"]
    )

    if config["visualization"].get("fine_tune_alignment"):
        mesh = load_trimesh(config["visualization"]["3d_model_path"])
        fine_tune_alignment(right_controller_leds, mesh, config["visualization"])

    # ── Tracking loop ──────────────────────────────────────────────────────
    poses             = []
    assignments       = []
    blobs             = []
    contours_all      = []
    raw_blobs         = []
    raw_contours_all  = []

    for batch in tqdm(get_data(config["data"])):
        img_path, image = batch[0][0], batch[0][1]

        blob_centroids, blob_contours, raw_centroids, raw_contours = get_centroids(
            image, config["blob_detection"], visualize=True, img_path=img_path
        )
        blobs.append(blob_centroids.copy())
        contours_all.append(blob_contours)
        raw_blobs.append(raw_centroids.copy())
        raw_contours_all.append(raw_contours)

        t0       = time()
        solution = tracking_system.update({0: blob_centroids}).get(
            ("right_controller", camera_0.camera_idx)
        )
        elapsed  = time() - t0

        if solution:
            poses.append((solution["rvec"].copy(), solution["tvec"].copy()))
            assignments.append(solution["assignment"].copy())
            logger.info(f"[{img_path.name}]  {elapsed:.3f}s  "
                        f"err={solution['error']:.2f}px  "
                        f"matches={len(solution['assignment'])}")
            if out_slow is not None and elapsed > SLOW_MATCH_THRESHOLD_S:
                copy(img_path, out_slow / img_path.name)
                logger.info(f"  → saved to deep_search_required (slow: {elapsed:.1f}s)")
        else:
            poses.append(None)
            assignments.append(None)
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
    )
    animator.start(
        poses, assignments, blobs, camera_0, T_model_ctrl,
        contours_all=contours_all,
        raw_blobs_all=raw_blobs,
        raw_contours_all=raw_contours_all,
        save_path=config["visualization"].get("save_recording"),
    )


if __name__ == "__main__":
    main()
