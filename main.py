from pathlib import Path
from shutil import copy
from time import time

from tqdm import tqdm

from src.blobs_detection import get_centroids
from src.camera import Camera
from src.controller import ControllerModel, TrackingSystem, create_leds_from_config
from src.load_config import load_yaml_config, load_json_config
from src.preprocess_data import get_data
from src.visualization import (ControllerAnimatorRerun, prepare_model_geometry,
                               fine_tune_alignment, load_trimesh)

SLOW_MATCH_THRESHOLD_S = 1.0


def main():
    config = load_yaml_config('./config/config.yml')

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

    # ── Debug output directories ───────────────────────────────────────────
    data_root         = Path(config["data"]["root"])
    out_slow          = data_root / "deep_search_required"
    out_tracking_lost = data_root / "tracking_lost"
    out_slow.mkdir(parents=True, exist_ok=True)
    out_tracking_lost.mkdir(parents=True, exist_ok=True)

    # ── Tracking loop ──────────────────────────────────────────────────────
    poses        = []
    assignments  = []
    blobs        = []
    contours_all = []

    for batch in tqdm(get_data(config["data"])):
        img_path, image = batch[0][0], batch[0][1]

        blob_centroids, blob_contours = get_centroids(
            image, config["blob_detection"], visualize=True, img_path=img_path
        )
        blobs.append(blob_centroids.copy())
        contours_all.append(blob_contours)

        t0       = time()
        solution = tracking_system.update({0: blob_centroids}).get(
            ("right_controller", camera_0.camera_idx)
        )
        elapsed  = time() - t0

        if solution:
            poses.append((solution["rvec"].copy(), solution["tvec"].copy()))
            assignments.append(solution["assignment"].copy())
            # print(f"[{img_path.name}]  {elapsed:.2f}s  "
            #       f"err={solution['error']:.2f}px  "
            #       f"matches={len(solution['assignment'])}")
            if elapsed > SLOW_MATCH_THRESHOLD_S:
                copy(img_path, out_slow / img_path.name)
                print(f"[{img_path.name}]  {elapsed:.2f}s  "
                      f"err={solution['error']:.2f}px  "
                      f"matches={len(solution['assignment'])}")
        else:
            poses.append(None)
            assignments.append(None)
            print(f"[{img_path.name}]  {elapsed:.2f}s  TRACKING LOST")
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
        save_path=config["visualization"].get("save_recording"),
    )


if __name__ == "__main__":
    main()
