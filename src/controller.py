import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_leds(leds_data):
    """Visualize LED positions and normals"""
    positions = np.array([led['Position'] for led in leds_data])
    normals = np.array([led['Normal'] for led in leds_data])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot LED positions
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c='red', s=50, label='LED Positions')

    # Plot normal vectors (scaled for visibility)
    scale = 0.03  # Scale factor to make normals visible
    for pos, norm in zip(positions, normals):
        ax.quiver(pos[0], pos[1], pos[2],
                  norm[0] * scale, norm[1] * scale, norm[2] * scale,
                  color='blue', alpha=0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Controller LED Constellation')
    ax.legend()
    plt.show()


def get_controller_positions(cfg):

    with open(cfg["right_controller"]["config_path"]) as f:
        controller_config = json.load(f)
    calibration_information = controller_config["CalibrationInformation"]["ControllerLeds"]
    # visualize_leds(calibration_information)
    positions_3d = np.array([led['Position'] for led in calibration_information])
    return positions_3d





