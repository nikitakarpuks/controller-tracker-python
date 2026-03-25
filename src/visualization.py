import matplotlib.pyplot as plt
import numpy as np
import trimesh
import open3d as o3d


class ControllerVisualizer3D:
    def __init__(self):
        self.mesh = None
        self.geometries = []

    # =========================
    # 1) LOAD RIGHT CONTROLLER
    # =========================
    def load_controller_model(self, path):
        scene = trimesh.load(path, force='scene')

        if "REVERB_G2_CONTROLLER_RIGHT_HAND" not in scene.geometry:
            raise ValueError(
                f"Right controller not found. Available keys: {list(scene.geometry.keys())}"
            )

        mesh = scene.geometry["REVERB_G2_CONTROLLER_RIGHT_HAND"].copy()

        print(f"Loaded right controller: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # # 🔥 Simplify (critical for performance)
        # if len(mesh.faces) > 5000:
        #     mesh = mesh.simplify_quadric_decimation(5000)
        #     print(f"Simplified to {len(mesh.faces)} faces")

        # Convert to Open3D
        o3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices),
            o3d.utility.Vector3iVector(mesh.faces)
        )

        o3d_mesh.compute_vertex_normals()
        o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])

        self.mesh = o3d_mesh
        self.geometries.append(o3d_mesh)

    # =========================
    # APPLY TRANSFORM (optional)
    # =========================
    def transform_model(self, position=(0, 0, 0), orientation=(0, 0, 0)):
        if self.mesh is None:
            return

        rx, ry, rz = orientation
        R = trimesh.transformations.euler_matrix(rx, ry, rz)[:3, :3]

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.array(position)

        self.mesh.transform(T)

    # =========================
    # 2) ADD LEDS
    # =========================
    def add_leds(self, led_coords, normals=None, scale=1.0):
        pts = np.array(led_coords) * scale

        # --- LED points ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile([1, 0, 0], (len(pts), 1))
        )

        self.geometries.append(pcd)

        # --- Normals ---
        if normals is not None:
            normals = np.array(normals)

            line_points = []
            lines = []

            for i, (p, n) in enumerate(zip(pts, normals)):
                line_points.append(p)
                line_points.append(p + n * 0.03)

                lines.append([2*i, 2*i + 1])

            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(line_points),
                lines=o3d.utility.Vector2iVector(lines)
            )

            line_set.colors = o3d.utility.Vector3dVector(
                [[0, 0, 1]] * len(lines)
            )

            self.geometries.append(line_set)

    # =========================
    # COORDINATE FRAME
    # =========================
    def add_frame(self, size=0.05):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        self.geometries.append(frame)

    # =========================
    # SHOW
    # =========================
    def show(self):
        o3d.visualization.draw_geometries(self.geometries)


# =========================
# USAGE FUNCTION
# =========================
def visualize_leds_with_controller(leds, cfg):
    positions = [led.position.squeeze(-1) for led in leds]
    normals = [led.normal.squeeze(-1) for led in leds]

    viz = ControllerVisualizer3D()
    viz.load_controller_model(cfg["3d_model_path"])

    # Optional initial transform (if you already have it)
    translation_cfg = cfg["initial_position_change"]["translation"]
    translation = (
        translation_cfg["x"],
        translation_cfg["y"],
        translation_cfg["z"]
    )

    rotation_cfg = cfg["initial_position_change"]["rotation"]
    rotation = (
        rotation_cfg["rx"],
        rotation_cfg["ry"],
        rotation_cfg["rz"]
    )

    viz.transform_model(translation, rotation)

    viz.add_leds(positions, normals, scale=1.0)
    viz.add_frame()

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
