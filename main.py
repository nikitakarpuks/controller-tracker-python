from src.load_config import load_yaml_config, load_json_config
from src.preprocess_data import get_data
from src.blobs_detection import get_centroids
from src.camera import Camera
from src.controller import ControllerModel, TrackingSystem, create_leds_from_config
from src.visualization import ControllerAnimatorRerun, prepare_model_geometry, show_initial_alignment
from time import time


def main():
    config = load_yaml_config('./config/config.yml')

    camera_0 = Camera(
        load_json_config(config["cameras"]["intrinsics_path"]),
        camera_idx = 0
    )

    calibration_config = load_json_config(config["controllers"]["right_controller"]["config_path"])
    right_controller_leds = create_leds_from_config(calibration_config)

    right_controller = ControllerModel(right_controller_leds, "right_controller")

    tracking_system = TrackingSystem([right_controller], [camera_0])

    dataloader = get_data(config["data"])

    poses = []
    assignments = []
    blobs = []

    positions_model, normals_model, T_model_ctrl = prepare_model_geometry(right_controller_leds, config["visualization"])

    # show_initial_alignment(positions_model, normals_model, config["visualization"]["3d_model_path"])

    # Iterate
    for batch in dataloader:
        img_path, image = batch[0][0], batch[0][1]
        blob_centroids = get_centroids(image, config["blob_detection"], visualize=True, img_path=img_path)
        blobs.append(blob_centroids.copy())

        t0 = time()

        # Track controller
        solution_ = tracking_system.update({0: blob_centroids})

        solution = solution_.get(("right_controller", camera_0.camera_idx))

        t1 = time()

        if solution:

            poses.append((solution['rvec'].copy(), solution['tvec'].copy()))
            assignments.append(solution["assignment"].copy())

            print(f"Tracking method: {solution['method']}")
            print(f"Reprojection error: {solution['error']:.2f} pixels")
            print(f"Matching took {t1 - t0:.2f} seconds\n")
            # print(f"Rotation vector: {solution['rvec'].flatten()}")
            # print(f"Translation vector: {solution['tvec'].flatten()}")
            # print(f"Assignment: {solution['assignment']}")
        else:
            poses.append(None)
            print("Tracking lost")

    # --- sanity check ---
    if all(p is None for p in poses):
        raise RuntimeError("No valid poses found")

    # --- start interactive viewer ---

    controler_animator = ControllerAnimatorRerun(
        config["visualization"]["3d_model_path"],
        positions_model,
        normals_model
    )

    controler_animator.start(poses, assignments, blobs, camera_0, T_model_ctrl)



if __name__ == '__main__':
    main()
