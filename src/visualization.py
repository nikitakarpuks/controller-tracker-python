import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import trimesh
import cv2
import copy

from typing import List, Tuple, Optional
from src.transformations import Transform


# =========================================================
# Utility
# =========================================================

def create_rays(points_cam: np.ndarray):
    """
    Create rays from camera origin to 3D points (in camera frame)
    """
    origin = np.zeros((len(points_cam), 3))

    line_points = []
    lines = []

    for i, (o, p) in enumerate(zip(origin, points_cam)):
        line_points.append(o)
        line_points.append(p)
        lines.append([2*i, 2*i + 1])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(lines)
    )

    # color: blue rays
    line_set.colors = o3d.utility.Vector3dVector(
        [[0, 0, 1]] * len(lines)
    )

    return line_set

def rt_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> Transform:
    R, _ = cv2.Rodrigues(rvec)
    return Transform(R, tvec.reshape(3))


def create_camera_frustum(scale=0.1):
    points = np.array([
        [0, 0, 0],      # camera center
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ]) * scale

    lines = [
        [0,1],[0,2],[0,3],[0,4],
        [1,2],[2,3],[3,4],[4,1]
    ]

    return o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )


def transform_to_matrix(T: Transform) -> np.ndarray:
    T4 = np.eye(4)
    T4[:3, :3] = T.R
    T4[:3, 3] = T.t
    return T4


# =========================================================
# Alignment (controller → model)
# =========================================================

def build_alignment_transform(cfg) -> Transform:
    rx = cfg["initial_position_change"]["rotation"]["rx"]
    ry = cfg["initial_position_change"]["rotation"]["ry"]
    rz = cfg["initial_position_change"]["rotation"]["rz"]

    t = np.array([
        cfg["initial_position_change"]["translation"]["x"],
        cfg["initial_position_change"]["translation"]["y"],
        cfg["initial_position_change"]["translation"]["z"]
    ])

    R_rz = trimesh.transformations.euler_matrix(0, 0, rz)[:3, :3]
    R_base = trimesh.transformations.euler_matrix(rx, ry, 0)[:3, :3]

    R = R_base.T @ R_rz
    t_final = -R_base.T @ t

    return Transform(R, t_final)  # controller → model


# =========================================================
# Static visualization (frame 0)
# =========================================================

def show_initial_alignment(model_positions: np.ndarray,
                           model_normals: np.ndarray,
                           mesh_path: str):

    scene = trimesh.load(mesh_path, force='scene')
    mesh = scene.geometry["REVERB_G2_CONTROLLER_RIGHT_HAND"]

    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces)
    )
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])

    # LEDs
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(model_positions)
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile([1, 0, 0], (len(model_positions), 1))
    )

    # normals
    line_points = []
    lines = []
    for i, (p, n) in enumerate(zip(model_positions, model_normals)):
        line_points.append(p)
        line_points.append(p + n * 0.03)
        lines.append([2*i, 2*i + 1])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1]] * len(lines))

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    o3d.visualization.draw_geometries([o3d_mesh, pcd, line_set, frame])


# =========================================================
# Animator
# =========================================================

def create_normals(points: np.ndarray, normals: np.ndarray, scale=0.03):
    line_points = []
    lines = []

    for i, (p, n) in enumerate(zip(points, normals)):
        line_points.append(p)
        line_points.append(p + n * scale)
        lines.append([2*i, 2*i + 1])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(lines)
    )

    # blue normals
    line_set.colors = o3d.utility.Vector3dVector(
        [[0, 0, 1]] * len(lines)
    )

    return line_set

def create_image_plane(width=640, height=480, fx=500, fy=500, scale=1.0):
    """
    Create a plane at z=1 in camera coords
    """
    z = 1.0 * scale

    corners = np.array([
        [0, 0, z],
        [width, 0, z],
        [width, height, z],
        [0, height, z]
    ], dtype=np.float32)

    # normalize using intrinsics
    corners[:, 0] = (corners[:, 0] - width / 2) / fx * z
    corners[:, 1] = (corners[:, 1] - height / 2) / fy * z

    lines = [[0,1],[1,2],[2,3],[3,0]]

    plane = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines)
    )

    plane.colors = o3d.utility.Vector3dVector([[1,1,1]] * 4)

    return plane

def backproject_to_plane(blobs, fx, fy, cx, cy, z=1.0):
    """
    Convert 2D image points → 3D points on plane z
    """
    pts = []

    for u, v in blobs:
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        pts.append([x, y, z])

    return np.array(pts, dtype=np.float32)

class ControllerAnimatorInteractive:

    def __init__(self, mesh_path, model_positions, model_normals):
        self.base_mesh = self._load_mesh(mesh_path)
        self.base_leds = o3d.geometry.PointCloud()
        self.base_normals = model_normals
        self.base_leds.points = o3d.utility.Vector3dVector(model_positions)

        self.base_leds.colors = o3d.utility.Vector3dVector(
            np.tile([1, 0, 0], (len(model_positions), 1))
        )

        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

        self.poses = None
        self.T_model_ctrl = None

        self.idx = 0
        self.paused = False
        self.window = None
        self.scene = None
        self.control_window = None
        self._tick_callback = None

    def _load_mesh(self, path):
        """Load mesh from file"""
        import trimesh
        try:
            scene = trimesh.load(path, force='scene')
            # Try different possible mesh names
            mesh = None
            possible_names = ["REVERB_G2_CONTROLLER_RIGHT_HAND", "mesh", "geometry"]
            for name in possible_names:
                if name in scene.geometry:
                    mesh = scene.geometry[name]
                    break

            if mesh is None:
                # If no named mesh found, use the first one
                mesh = list(scene.geometry.values())[0]

            o3d_mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(mesh.vertices),
                o3d.utility.Vector3iVector(mesh.faces)
            )
            o3d_mesh.compute_vertex_normals()
            o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
            return o3d_mesh
        except Exception as e:
            print(f"Error loading mesh: {e}")
            # Return a simple cube as fallback
            return o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 0.1)

    def start(self, poses, assignments, blobs_all, camera, T_model_ctrl):
        """Start the visualization"""

        self.blobs_all = blobs_all
        self.camera = camera

        # Initialize GUI
        gui.Application.instance.initialize()

        # Create main window
        self.window = gui.Application.instance.create_window(
            "Controller Viewer", 1024, 768
        )

        # Create control panel window (separate window)
        self.control_window = gui.Application.instance.create_window(
            "Controls", 300, 200, x=1024, y=1024
        )

        # Create control panel layout
        panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))

        # Create buttons with proper spacing
        play_btn = gui.Button("Play / Pause (SPACE)")
        next_btn = gui.Button("Next Frame (→)")
        prev_btn = gui.Button("Prev Frame (←)")
        reset_btn = gui.Button("Reset to Start")

        # Add status label
        self.status_label = gui.Label("Frame: 0 / 0")

        # Add some spacing
        panel.add_child(gui.Label(""))  # spacing
        panel.add_child(play_btn)
        panel.add_child(gui.Label(""))  # spacing
        panel.add_child(next_btn)
        panel.add_child(prev_btn)
        panel.add_child(gui.Label(""))  # spacing
        panel.add_child(reset_btn)
        panel.add_child(gui.Label(""))  # spacing
        panel.add_child(self.status_label)

        # Create scene widget
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)

        # Add geometries to scene
        self.mesh = copy.deepcopy(self.base_mesh)
        self.leds = copy.deepcopy(self.base_leds)

        # Create material records
        self.mat_mesh = rendering.MaterialRecord()
        self.mat_mesh.shader = "defaultLit"

        self.mat_leds = rendering.MaterialRecord()
        self.mat_leds.shader = "defaultUnlit"
        self.mat_leds.point_size = 5.0

        self.mat_lines = rendering.MaterialRecord()
        self.mat_lines.shader = "unlitLine"
        self.mat_lines.line_width = 2.0  # optional, makes lines thicker

        self.normals_vis = None

        self.scene.scene.add_geometry("mesh", self.mesh, self.mat_mesh)
        self.scene.scene.add_geometry("leds", self.leds, self.mat_leds)
        self.rays = None
        # self.scene.scene.add_geometry("frame", self.frame, mesh_material)

        # --- CAMERA (static) ---
        self.camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.camera_frustum = create_camera_frustum(scale=0.15)
        self.camera_frustum.colors = o3d.utility.Vector3dVector(
            [[0, 1, 0]] * len(self.camera_frustum.lines)  # green
        )

        cam_mat = rendering.MaterialRecord()
        cam_mat.shader = "defaultUnlit"

        self.scene.scene.add_geometry("camera_frame", self.camera_frame, cam_mat)
        self.scene.scene.add_geometry("camera_frustum", self.camera_frustum, cam_mat)

        # Setup camera
        bounds = self.mesh.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60, bounds, bounds.get_center())

        self.image_plane = create_image_plane(
            width=self.camera.width,
            height=self.camera.height,
            fx=self.camera.fx,
            fy=self.camera.fy,
        )

        self.scene.scene.add_geometry("image_plane", self.image_plane, self.mat_lines)

        self.scene.scene.add_geometry("image_plane", self.image_plane, self.mat_lines)

        # Add scene to main window
        self.window.add_child(self.scene)

        # Add panel to control window
        self.control_window.add_child(panel)

        # Set callbacks
        play_btn.set_on_clicked(self._toggle)
        next_btn.set_on_clicked(lambda: self._step(1))
        prev_btn.set_on_clicked(lambda: self._step(-1))
        reset_btn.set_on_clicked(self._reset)

        # Store data
        self.poses = poses
        self.assignments = assignments
        self.T_model_ctrl = T_model_ctrl

        # Keyboard controls
        self.window.set_on_key(self._on_key)

        # Update status
        self._update_status()

        # Start animation
        self._tick()

        # Run application
        gui.Application.instance.run()

    def _toggle(self):
        """Toggle pause state"""
        self.paused = not self.paused

    def _step(self, direction):
        """Step forward or backward by one frame"""
        new_idx = self.idx + direction
        if 0 <= new_idx < len(self.poses):
            self.idx = new_idx
            self._update_frame()
            self._update_status()

    def _reset(self):
        """Reset to first frame"""
        self.idx = 0
        self.paused = True
        self._update_frame()
        self._update_status()

    def _on_key(self, event):
        """Handle keyboard events"""
        if event.type == gui.KeyEvent.Type.UP:
            if event.key == gui.KeyName.SPACE:
                self._toggle()
                return gui.Widget.EventCallbackResult.HANDLED

            elif event.key == gui.KeyName.RIGHT:
                self._step(1)
                return gui.Widget.EventCallbackResult.HANDLED

            elif event.key == gui.KeyName.LEFT:
                self._step(-1)
                return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    def _update_status(self):
        """Update status label text"""
        if hasattr(self, 'status_label') and self.status_label:
            total = len(self.poses) if self.poses else 0
            current = self.idx + 1 if total > 0 else 0
            self.status_label.text = f"Frame: {current} / {total}"

    def _tick(self):
        """Animation tick function"""
        if not self.paused and self.poses is not None and len(self.poses) > 0:
            self.idx = (self.idx + 1) % len(self.poses)
            self._update_frame()
            self._update_status()

        # Schedule next tick if window still exists
        if self.window:
            self._tick_callback = gui.Application.instance.post_to_main_thread(
                self.window, self._tick
            )

    def _update_frame(self):
        """Update the current frame based on pose"""
        if self.poses is None or self.idx >= len(self.poses):
            return

        pose = self.poses[self.idx]

        if pose is None:
            return

        try:
            rvec, tvec = pose

            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            tvec = tvec.reshape(3)

            # Compute transformation
            T_cam_ctrl = Transform(R, tvec)
            T_ctrl_model = self.T_model_ctrl.inverse()
            T_cam_model = T_cam_ctrl.compose(T_ctrl_model)

            # =========================
            # BLOBS + PROJECTIONS
            # =========================

            if self.blobs_all is not None and self.idx < len(self.blobs_all):

                blobs = self.blobs_all[self.idx]

                # --- blobs → plane ---
                pts_plane = backproject_to_plane(
                    blobs,
                    self.camera.fx,
                    self.camera.fy,
                    self.camera.cx,
                    self.camera.cy,
                    z=1.0
                )

                blob_pcd = o3d.geometry.PointCloud()
                blob_pcd.points = o3d.utility.Vector3dVector(pts_plane)
                blob_pcd.colors = o3d.utility.Vector3dVector(
                    np.tile([1, 1, 0], (len(pts_plane), 1))  # yellow blobs
                )

                if hasattr(self, "blob_vis") and self.blob_vis is not None:
                    self.scene.scene.remove_geometry("blobs")

                self.scene.scene.add_geometry("blobs", blob_pcd, self.mat_leds)
                self.blob_vis = blob_pcd

                # --- project LEDs (PnP result) ---
                object_points = np.asarray(self.base_leds.points)

                proj_2d, _ = cv2.projectPoints(
                    object_points,
                    rvec,
                    tvec,
                    self.camera.camera_matrix,
                    self.camera.dist_coeffs
                )

                proj_2d = proj_2d.reshape(-1, 2)

                proj_plane = backproject_to_plane(
                    proj_2d,
                    self.camera.fx,
                    self.camera.fy,
                    self.camera.cx,
                    self.camera.cy,
                    z=1.0
                )

                proj_pcd = o3d.geometry.PointCloud()
                proj_pcd.points = o3d.utility.Vector3dVector(proj_plane)
                proj_pcd.colors = o3d.utility.Vector3dVector(
                    np.tile([0, 1, 0], (len(proj_plane), 1))  # green projected LEDs
                )

                if hasattr(self, "proj_vis") and self.proj_vis is not None:
                    self.scene.scene.remove_geometry("proj")

                self.scene.scene.add_geometry("proj", proj_pcd, self.mat_leds)
                self.proj_vis = proj_pcd

            error_lines = []
            error_edges = []

            for i, (p, b) in enumerate(zip(proj_plane, pts_plane)):
                error_lines.append(p)
                error_lines.append(b)
                error_edges.append([2 * i, 2 * i + 1])

            err_ls = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(error_lines),
                lines=o3d.utility.Vector2iVector(error_edges)
            )

            err_ls.colors = o3d.utility.Vector3dVector(
                [[1, 0, 0]] * len(error_edges)  # red error lines
            )

            if hasattr(self, "err_vis") and self.err_vis is not None:
                self.scene.scene.remove_geometry("err")

            self.scene.scene.add_geometry("err", err_ls, self.mat_lines)
            self.err_vis = err_ls

            # --- normals ---
            pts_model = np.asarray(self.base_leds.points)
            normals_model = self.base_normals

            # transform normals (rotation only!)
            normals_cam = (T_cam_model.R @ normals_model.T).T

            # transform points
            pts_cam = (T_cam_model.R @ pts_model.T).T + T_cam_model.t

            normals_vis = create_normals(pts_cam, normals_cam, scale=0.03)

            # update scene
            if hasattr(self, "normals_vis") and self.normals_vis is not None:
                self.scene.scene.remove_geometry("normals")

            self.scene.scene.add_geometry("normals", normals_vis, self.mat_lines)
            self.normals_vis = normals_vis

            # Create 4x4 transformation matrix
            T4 = np.eye(4)
            T4[:3, :3] = T_cam_model.R
            T4[:3, 3] = T_cam_model.t

            # --- rays ---
            assignment = None
            if self.assignments is not None and self.idx < len(self.assignments):
                assignment = self.assignments[self.idx]

            if assignment is not None and len(assignment) > 0:

                # get LED positions in MODEL frame
                led_ids = [lid for _, lid in assignment]

                pts_model = self.base_leds.points

                pts_model = np.asarray(pts_model)[led_ids]

                # transform to camera frame
                pts_cam = (T_cam_model.R @ pts_model.T).T + T_cam_model.t

                rays = create_rays(pts_cam)

                # update scene
                if hasattr(self, "rays") and self.rays is not None:
                    self.scene.scene.remove_geometry("rays")

                self.scene.scene.add_geometry("rays", rays, self.mat_lines)
                self.rays = rays

            # Update mesh and point cloud transformations
            # Create transformed copies
            self.mesh = copy.deepcopy(self.base_mesh)
            self.mesh.transform(T4)

            self.leds = copy.deepcopy(self.base_leds)
            self.leds.transform(T4)

            # --- color LEDs ---
            colors = np.tile([1, 0, 0], (len(self.base_leds.points), 1))  # red

            if assignment:
                for _, lid in assignment:
                    colors[lid] = [0, 1, 0]  # green matched

            self.leds.colors = o3d.utility.Vector3dVector(colors)

            # # Update geometries in scene

            self.scene.scene.remove_geometry("mesh")
            self.scene.scene.remove_geometry("leds")

            self.scene.scene.add_geometry("mesh", self.mesh, self.mat_mesh)
            self.scene.scene.add_geometry("leds", self.leds, self.mat_leds)

        except Exception as e:
            print(f"Error updating frame {self.idx}: {e}")


# =========================================================
# Convenience wrapper
# =========================================================

def prepare_model_geometry(leds, cfg):
    positions = np.array([led.position for led in leds], dtype=np.float64)
    normals = np.array([led.normal for led in leds], dtype=np.float64)

    T_model_ctrl = build_alignment_transform(cfg)

    # transform LEDs into model space
    positions_model = T_model_ctrl.apply(positions)
    normals_model = (T_model_ctrl.R @ normals.T).T

    return positions_model, normals_model, T_model_ctrl
