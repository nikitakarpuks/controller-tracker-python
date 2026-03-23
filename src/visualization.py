import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh


class ControllerVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.controller_mesh = None

    def load_controller_model(self, stl_path):
        self.controller_mesh = trimesh.load(stl_path)

        print(f"Original: {len(self.controller_mesh.faces)} faces")

        # 🔥 Simplify mesh (CRITICAL)
        target_faces = 3000
        if len(self.controller_mesh.faces) > target_faces:
            self.controller_mesh = self.controller_mesh.simplify_quadric_decimation(face_count=target_faces)

        print(f"Simplified: {len(self.controller_mesh.faces)} faces")

    def add_controller_model(self, position=(0, 0, 0), orientation=(0, 0, 0)):
        mesh = self.controller_mesh.copy()

        # Apply transform
        mesh.apply_translation(position)

        if any(orientation):
            rx, ry, rz = orientation
            R = trimesh.transformations.euler_matrix(rx, ry, rz)
            mesh.apply_transform(R)

        vertices = mesh.vertices
        faces = mesh.faces

        collection = Poly3DCollection(
            vertices[faces],
            facecolor='lightgray',
            edgecolor='none',
            alpha=0.6
        )

        self.ax.add_collection3d(collection)

    def add_leds(self, led_coords, normals=None, colors=None):
        if colors is None:
            colors = ['red'] * len(led_coords)

        led_coords = np.array(led_coords)
        led_coords *= 80  # Scale up for better visibility

        # 🔥 BIGGER + always visible
        self.ax.scatter(
            led_coords[:, 0],
            led_coords[:, 1],
            led_coords[:, 2],
            c=colors,
            s=50,  # bigger
            depthshade=False,  # 🔥 important
            edgecolors='black'
        )

        # Draw origin point (0,0,0)
        self.ax.scatter(
            0, 0, 0,
            c='black',
            s=100,
            depthshade=False,
            edgecolors='white',
            linewidth=1.5,
            label='Origin'
        )

        # Draw X, Y, Z unit vectors from origin
        # X-axis (red)
        self.ax.quiver(
            0, 0, 0, 1, 0, 0,
            color='red',
            length=1.0,
            linewidth=3,
            arrow_length_ratio=0.2
        )
        self.ax.text(1.1, 0, 0, 'X', color='red', fontsize=12, fontweight='bold')

        # Y-axis (green)
        self.ax.quiver(
            0, 0, 0, 0, 1, 0,
            color='green',
            length=1.0,
            linewidth=3,
            arrow_length_ratio=0.2
        )
        self.ax.text(0, 1.1, 0, 'Y', color='green', fontsize=12, fontweight='bold')

        # Z-axis (blue)
        self.ax.quiver(
            0, 0, 0, 0, 0, 1,
            color='blue',
            length=1.0,
            linewidth=3,
            arrow_length_ratio=0.2
        )
        self.ax.text(0, 0, 1.1, 'Z', color='blue', fontsize=12, fontweight='bold')

        # Normals
        if normals is not None:
            normals = np.array(normals)

            self.ax.quiver(
                led_coords[:, 0],
                led_coords[:, 1],
                led_coords[:, 2],
                normals[:, 0],
                normals[:, 1],
                normals[:, 2],
                length=0.03,
                color='blue',
                linewidth=2
            )

    # def set_equal_axes(self, points=None):
    #     if self.controller_mesh is not None:
    #         bounds = self.controller_mesh.bounds
    #     else:
    #         bounds = np.array([[0, 0, 0], [1, 1, 1]])
    #
    #     if points is not None:
    #         pts = np.array(points)
    #         bounds = np.vstack([bounds, [pts.min(axis=0), pts.max(axis=0)]])
    #
    #     min_b = bounds.min(axis=0)
    #     max_b = bounds.max(axis=0)
    #
    #     center = (min_b + max_b) / 2
    #     size = (max_b - min_b).max() / 2
    #
    #     self.ax.set_xlim(center[0] - size, center[0] + size)
    #     self.ax.set_ylim(center[1] - size, center[1] + size)
    #     self.ax.set_zlim(center[2] - size, center[2] + size)
    #
    #     self.ax.set_box_aspect([1, 1, 1])

    def show(self):
        """Display the visualization"""
        plt.tight_layout()
        plt.show()


def visualize_leds_with_controller(leds: list, cfg: dict):
    """Visualize LED positions and normals with controller model"""

    positions = []
    normals = []
    for led in leds:
        positions.append(led.position)
        normals.append(led.normal)

    # Create visualizer
    viz = ControllerVisualizer()
    viz.load_controller_model(cfg["3d_model_path"])

    translation_cfg = cfg["initial_position_change"]["translation"]
    translation = (translation_cfg["x"], translation_cfg["y"], translation_cfg["z"])

    rotation_cfg = cfg["initial_position_change"]["rotation"]
    rotation = (rotation_cfg["rx"], rotation_cfg["ry"], rotation_cfg["rz"])

    viz.add_controller_model(translation,
                             rotation)
    viz.add_leds(positions)# normals)
    # viz.set_equal_axes(leds)
    viz.show()


def visualize_leds(leds: list):
    """Visualize LED positions and normals"""

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    positions = np.array([led.position for led in leds])
    normals = np.array([led.normal for led in leds])

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
