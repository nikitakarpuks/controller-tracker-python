import matplotlib.pyplot as plt
import trimesh
from scipy.optimize import least_squares


import open3d as o3d
import numpy as np
import copy
import cv2


class SceneVisualizer:
    def draw_controller(self, model: ControllerModel, pose: Transform):
        # Add LEDs (now in controller's local coordinate system, origin at controller center)
        viz.add_leds(positions_final, normals_final)

        # Add coordinate frame at origin (controller center)
        viz.add_frame()

        # Controller model stays at origin (no transformation)
        viz.show()

    def add_frame(self, size=0.05):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        self.geometries.append(frame)

    # =========================
    # SHOW
    # =========================
    def show(self):
        o3d.visualization.draw_geometries(self.geometries)













def rt_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


class ControllerAnimator:
    def __init__(self, mesh_path, positions, normals):
        self.base_mesh = self.load_mesh(mesh_path)

        self.base_leds = o3d.geometry.PointCloud()
        self.base_leds.points = o3d.utility.Vector3dVector(positions)
        self.base_leds.colors = o3d.utility.Vector3dVector(
            np.tile([1, 0, 0], (len(positions), 1))
        )

        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    def load_mesh(self, path):
        scene = trimesh.load(path, force='scene')
        mesh = scene.geometry["REVERB_G2_CONTROLLER_RIGHT_HAND"]

        o3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices),
            o3d.utility.Vector3iVector(mesh.faces)
        )
        o3d_mesh.compute_vertex_normals()
        o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])

        return o3d_mesh

    def apply_pose(self, rvec, tvec):
        import cv2

        R, _ = cv2.Rodrigues(rvec)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.reshape(3)

        mesh = copy.deepcopy(self.base_mesh)
        leds = copy.deepcopy(self.base_leds)

        mesh.transform(T)
        leds.transform(T)

        return mesh, leds

    def apply_pose(self, rvec, tvec):
        import cv2

        R, _ = cv2.Rodrigues(rvec)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.reshape(3)

        mesh = copy.deepcopy(self.base_mesh)
        leds = copy.deepcopy(self.base_leds)

        mesh.transform(T)
        leds.transform(T)

        return mesh, leds

    def run(self, poses, fps=30, output="controller.mp4"):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        width, height = 640, 480
        video = cv2.VideoWriter(
            output,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        last_pose = None

        for pose in poses:
            if pose is not None:
                last_pose = pose
            elif last_pose is None:
                continue

            rvec, tvec = last_pose

            mesh, leds = self.apply_pose(rvec, tvec)

            vis.clear_geometries()
            vis.add_geometry(mesh)
            vis.add_geometry(leds)
            vis.add_geometry(self.frame)

            vis.poll_events()
            vis.update_renderer()

            img = vis.capture_screen_float_buffer(False)

            if img is None:
                continue

            img = np.asarray(img)

            # skip invalid frames
            if img.size == 0:
                continue

            # convert to uint8
            img = (255 * img).astype(np.uint8)

            # ensure 3 channels
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            # resize to writer resolution
            img = cv2.resize(img, (width, height))

            # convert RGB → BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            video.write(img)

        video.release()
        # vis.destroy_window()










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
def get_aligned_geometry(leds, cfg, visualize=True):
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

    # Build rotations
    R_rz = trimesh.transformations.euler_matrix(0, 0, rz)[:3, :3]
    R_base = trimesh.transformations.euler_matrix(rx, ry, 0)[:3, :3]

    # Combine
    R_combined = R_base.T @ R_rz
    t_combined = -R_base.T @ t

    # Apply once
    positions_final = (R_combined @ positions.T).T + t_combined
    normals_final = (R_combined @ normals.T).T

    if visualize:
        # Add LEDs (now in controller's local coordinate system, origin at controller center)
        viz.add_leds(positions_final, normals_final)

        # Add coordinate frame at origin (controller center)
        viz.add_frame()

        # Controller model stays at origin (no transformation)
        viz.show()

    return positions_final, normals_final

# def run_full_pipeline(leds, cfg):
#     # --- prepare data ---
#     positions = np.array(
#         [led.position.reshape(3) for led in leds],
#         dtype=np.float64
#     )
#
#     # --- load mesh ---
#     scene = trimesh.load(cfg["3d_model_path"], force='scene')
#     mesh = scene.geometry["REVERB_G2_CONTROLLER_RIGHT_HAND"].copy()
#
#     # OPTIONAL: scale fix (uncomment if needed)
#     # mesh.apply_scale(0.001)
#
#     # --- parameters ---
#     rx = cfg["initial_position_change"]["rotation"]["rx"]
#     ry = cfg["initial_position_change"]["rotation"]["ry"]
#     rz = cfg["initial_position_change"]["rotation"]["rz"]
#
#     t_init = np.array([
#         cfg["initial_position_change"]["translation"]["x"],
#         cfg["initial_position_change"]["translation"]["y"],
#         cfg["initial_position_change"]["translation"]["z"]
#     ])
#
#     # --- refine translation ---
#     print("Refining translation...")
#     t_refined = refine_translation(mesh, positions, rx, ry, rz, t_init)
#
#     print("\n=== RESULT ===")
#     print("Refined translation:", t_refined)
#
#     # --- visualize ---
#     visualize_leds_with_controller(leds, cfg, refined_t=t_refined)
#
#     return t_refined
