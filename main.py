from src.load_config import load_yaml_config, load_json_config
from src.preprocess_data import get_data
from src.blobs_detection import get_centroids
from src.camera import CameraCalibration
from src.controller import ControllerTracker, create_leds_from_config

from src.visualization import visualize_leds, visualize_leds_with_controller

import matplotlib.pyplot as plt


def main():
    config = load_yaml_config('./config/config.yml')

    camera_calibration = CameraCalibration(
        load_json_config(config["cameras"]["intrinsics_path"]),
        camera_idx = 0
    )

    calibration_config = load_json_config(config["controllers"]["right_controller"]["config_path"])
    right_controller_leds = create_leds_from_config(calibration_config)
    # visualize_leds(right_controller_leds)

    # Path to your STL file
    stl_file = './data/quest2_contollers_div0_right.stl'

    # Visualize with controller model at origin
    visualize_leds_with_controller(
        leds=right_controller_leds,
        stl_path=stl_file,
        position=(-0.87, -2.30, 0.5),  # Position in meters
        orientation=(0.75, -0.2, 1.6)  # Rotation in radians (rx, ry, rz)
    )

    # controller_tracker = ControllerTracker(camera_calibration, right_controller_leds)
    #
    # dataloader = get_data(config["data"])
    #
    # # Iterate
    # for image_batch in dataloader:
    #     image = image_batch[0]
    #     blob_centroids = get_centroids(image, config["blob_detection"])
    #
    #     # Track controller
    #     solution = controller_tracker.track(blob_centroids)
    #
    #     if solution:
    #         print(f"Tracking method: {solution['method']}")
    #         print(f"Reprojection error: {solution['error']:.2f} pixels")
    #         print(f"Rotation vector: {solution['rvec'].flatten()}")
    #         print(f"Translation vector: {solution['tvec'].flatten()}")
    #         print(f"Assignment: {solution['assignment']}")
    #     else:
    #         print("Tracking lost")
    #
    #     print(1)


if __name__ == '__main__':
    main()
