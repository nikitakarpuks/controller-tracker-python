import matplotlib.pyplot as plt
import numpy as np
import trimesh
import open3d as o3d
from scipy.optimize import least_squares


def refine_translation(mesh, positions, rx, ry, rz, t_init):
    """
    Optimize translation t so that transformed LEDs lie on mesh surface
    """

    # --- rotations ---
    R_rz = trimesh.transformations.euler_matrix(0, 0, rz)[:3, :3]
    R_base = trimesh.transformations.euler_matrix(rx, ry, 0)[:3, :3]
    R_base_inv = R_base.T

    # --- precompute constant part ---
    A = (R_base_inv @ (R_rz @ positions.T)).T  # Nx3

    def residual(t):
        t = np.asarray(t)

        # transform points (same model as your pipeline)
        pts = A - (R_base_inv @ t)

        # project to mesh
        closest, _, _ = mesh.nearest.on_surface(pts)

        return (pts - closest).ravel()

    result = least_squares(
        residual,
        t_init,
        verbose=2
    )

    return result.x


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
    positions = np.array(
        [led.position.reshape(3) for led in leds],
        dtype=np.float64
    )

    normals = np.array(
        [led.normal.reshape(3) for led in leds],
        dtype=np.float64
    )

    viz = ControllerVisualizer3D()
    viz.load_controller_model(cfg["3d_model_path"])

    # parameters
    t = np.array([
        cfg["initial_position_change"]["translation"]["x"],
        cfg["initial_position_change"]["translation"]["y"],
        cfg["initial_position_change"]["translation"]["z"]
    ])

    rx = cfg["initial_position_change"]["rotation"]["rx"]
    ry = cfg["initial_position_change"]["rotation"]["ry"]
    rz = cfg["initial_position_change"]["rotation"]["rz"]

    # Step 1: First, apply rz rotation to the original point cloud (centered near origin)
    R_rz = trimesh.transformations.euler_matrix(0, 0, rz)[:3, :3]
    positions_rotated = (R_rz @ positions.T).T
    normals_rotated = (R_rz @ normals.T).T

    # Step 2: Transform points to controller's local coordinate system
    # This is the inverse of what viz.transform_model would do
    # Original transformation would be: p_world = R_base @ p_local + t
    # So p_local = R_base.T @ (p_world - t)

    R_base = trimesh.transformations.euler_matrix(rx, ry, 0)[:3, :3]

    # First center at controller position, then apply inverse rotation
    positions_final = (R_base.T @ (positions_rotated - t).T).T
    normals_final = (R_base.T @ normals_rotated.T).T

    # Add LEDs (now in controller's local coordinate system, origin at controller center)
    viz.add_leds(positions_final, normals_final)

    # Add coordinate frame at origin (controller center)
    viz.add_frame()

    # Controller model stays at origin (no transformation)
    viz.show()

def run_full_pipeline(leds, cfg):
    # --- prepare data ---
    positions = np.array(
        [led.position.reshape(3) for led in leds],
        dtype=np.float64
    )

    # --- load mesh ---
    scene = trimesh.load(cfg["3d_model_path"], force='scene')
    mesh = scene.geometry["REVERB_G2_CONTROLLER_RIGHT_HAND"].copy()

    # OPTIONAL: scale fix (uncomment if needed)
    # mesh.apply_scale(0.001)

    # --- parameters ---
    rx = cfg["initial_position_change"]["rotation"]["rx"]
    ry = cfg["initial_position_change"]["rotation"]["ry"]
    rz = cfg["initial_position_change"]["rotation"]["rz"]

    t_init = np.array([
        cfg["initial_position_change"]["translation"]["x"],
        cfg["initial_position_change"]["translation"]["y"],
        cfg["initial_position_change"]["translation"]["z"]
    ])

    # --- refine translation ---
    print("Refining translation...")
    t_refined = refine_translation(mesh, positions, rx, ry, rz, t_init)

    print("\n=== RESULT ===")
    print("Refined translation:", t_refined)

    # --- visualize ---
    visualize_leds_with_controller(leds, cfg, refined_t=t_refined)

    return t_refined


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
