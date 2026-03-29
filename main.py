from src.load_config import load_yaml_config, load_json_config
from src.preprocess_data import get_data
from src.blobs_detection import get_centroids
from src.camera import Camera
from src.controller import ControllerModel, TrackingSystem, create_leds_from_config
# from src.visualization import visualize_leds, get_aligned_geometry, ControllerAnimator # run_full_pipeline


def main():
    config = load_yaml_config('./config/config.yml')

    camera_0 = Camera(
        load_json_config(config["cameras"]["intrinsics_path"]),
        camera_idx = 0
    )

    calibration_config = load_json_config(config["controllers"]["right_controller"]["config_path"])
    right_controller_leds = create_leds_from_config(calibration_config)

    # visualize_leds(right_controller_leds)
    # run_full_pipeline(right_controller_leds, config["visualization"])

    # # Step 1: get aligned geometry
    # positions_aligned, normals_aligned = get_aligned_geometry(right_controller_leds, config["visualization"],
    #                                                           visualize=True)
    #
    # # Step 2: create animator
    # anim = ControllerAnimator(
    #     config["visualization"]["3d_model_path"],
    #     positions_aligned,
    #     normals_aligned
    # )

    right_controller = ControllerModel(right_controller_leds, "right_controller")

    tracking_system = TrackingSystem([right_controller], [camera_0])

    dataloader = get_data(config["data"])

    poses = []

    # Iterate
    for batch in dataloader:
        img_path, image = batch[0][0], batch[0][1]
        blob_centroids = get_centroids(image, config["blob_detection"], visualize=True, img_path=img_path)

        # Track controller
        solution_ = tracking_system.update({0: blob_centroids})
        solution = solution_.get(("right_controller", camera_0.camera_idx))

        if solution:

            poses.append((solution['rvec'], solution['tvec']))

            print(f"Tracking method: {solution['method']}")
            print(f"Reprojection error: {solution['error']:.2f} pixels")
            print(f"Rotation vector: {solution['rvec'].flatten()}")
            print(f"Translation vector: {solution['tvec'].flatten()}")
            print(f"Assignment: {solution['assignment']}")
        else:
            print("Tracking lost")

    # # Step 3: run
    # anim.run(poses, fps=30, output="controller.mp4")



if __name__ == '__main__':
    main()
