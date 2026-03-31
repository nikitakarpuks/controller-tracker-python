import numpy as np
import trimesh
import cv2
import rerun as rr
import rerun.blueprint as rrb

from src.transformations import Transform


# =========================================================
# Visualization config (toggle objects on/off)
# =========================================================

VIS_CONFIG = {
    "show_mesh":        True,
    "show_leds":        True,
    "show_normals":     True,
    "show_rays":        True,
    "show_blobs":       True,
    "show_projected":   True,
    "show_errors":      True,
    "show_frustum":     True,
    "show_image_plane": True,
    "show_camera_frame":True,
    "fps":              30,        # playback speed
}


# =========================================================
# Utility
# =========================================================

def rt_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> Transform:
    R, _ = cv2.Rodrigues(rvec)
    return Transform(R, tvec.reshape(3))


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
# Geometry helpers
# =========================================================

def compute_frustum_corners(cam, z: float):
    """
    Returns the 4 corner points of the image plane at depth z,
    in camera frame.
    """
    corners_px = np.array([
        [0,          0         ],
        [cam.width,  0         ],
        [cam.width,  cam.height],
        [0,          cam.height],
    ], dtype=np.float32)

    pts = []
    for u, v in corners_px:
        x = (u - cam.cx) / cam.fx * z
        y = (v - cam.cy) / cam.fy * z
        pts.append([x, y, z])

    return np.array(pts, dtype=np.float32)


def backproject_to_plane(blobs: np.ndarray, cam, z: float = 0.2) -> np.ndarray:
    pts = []
    for u, v in blobs:
        x = (u - cam.cx) / cam.fx * z
        y = (v - cam.cy) / cam.fy * z
        pts.append([x, y, z])
    return np.array(pts, dtype=np.float32)


# =========================================================
# Static one-shot visualization (frame 0 equivalent)
# =========================================================

def show_initial_alignment(model_positions: np.ndarray,
                           model_normals: np.ndarray,
                           mesh_path: str,
                           vis_cfg: dict = None):
    if vis_cfg is None:
        vis_cfg = VIS_CONFIG

    rr.init("controller_initial_alignment", spawn=True)

    scene = trimesh.load(mesh_path, force='scene')
    mesh = scene.geometry.get(
        "REVERB_G2_CONTROLLER_RIGHT_HAND",
        list(scene.geometry.values())[0]
    )

    if vis_cfg.get("show_mesh", True):
        rr.log(
            "world/mesh",
            rr.Mesh3D(
                vertex_positions=mesh.vertices,
                triangle_indices=mesh.faces,
                vertex_normals=mesh.vertex_normals if hasattr(mesh, "vertex_normals") else None,
                albedo_factor=[0.7, 0.7, 0.7, 1.0],
            )
        )

    if vis_cfg.get("show_leds", True):
        rr.log(
            "world/leds",
            rr.Points3D(
                positions=model_positions,
                colors=np.tile([255, 0, 0], (len(model_positions), 1)),
                radii=0.005,
            )
        )

    if vis_cfg.get("show_normals", True):
        normal_starts = model_positions
        normal_ends   = model_positions + model_normals * 0.03
        rr.log(
            "world/normals",
            rr.LineStrips3D(
                strips=[[s, e] for s, e in zip(normal_starts, normal_ends)],
                colors=[0, 0, 255],
            )
        )

    rr.log(
        "world/frame",
        rr.Arrows3D(
            origins=[[0, 0, 0]] * 3,
            vectors=[[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        )
    )


# =========================================================
# Mesh loader
# =========================================================

def load_trimesh(path: str):
    scene = trimesh.load(path, force='scene')
    possible_names = ["REVERB_G2_CONTROLLER_RIGHT_HAND", "mesh", "geometry"]
    mesh = None
    for name in possible_names:
        if name in scene.geometry:
            mesh = scene.geometry[name]
            break
    if mesh is None:
        mesh = list(scene.geometry.values())[0]
    return mesh


# =========================================================
# Main animator (rerun version)
# =========================================================

class ControllerAnimatorRerun:
    """
    Rerun-based replacement for ControllerAnimatorInteractive.

    Instead of an Open3D GUI event loop, we log every frame to
    rerun with a timeline.  Open the Rerun viewer and scrub/play
    freely.
    """

    def __init__(self, mesh_path: str,
                 model_positions: np.ndarray,
                 model_normals: np.ndarray,
                 vis_cfg: dict = None):

        self.mesh_path       = mesh_path
        self.model_positions = model_positions   # (N,3) in model space
        self.model_normals   = model_normals     # (N,3) in model space
        self.vis_cfg         = vis_cfg if vis_cfg is not None else dict(VIS_CONFIG)

        self._trimesh        = load_trimesh(mesh_path)
        self.visual_offset   = np.array([0.0, 0.0, 0.3])

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def start(self, poses, assignments, blobs_all, camera, T_model_ctrl):
        """
        Log all frames to rerun.
        Opens the viewer automatically (spawn=True).
        """
        rr.init("controller_animator", spawn=True)

        # Set up a blueprint so the viewer looks reasonable on first open
        blueprint = rrb.Blueprint(
            rrb.Spatial3DView(name="3-D Scene", origin="/world"),
            collapse_panels=False,
        )
        rr.send_blueprint(blueprint)

        self._log_static_camera(camera)

        n_frames = len(poses)

        for idx in range(n_frames):
            rr.set_time("frame", sequence=idx)

            pose = poses[idx]
            if pose is None:
                continue

            rvec, tvec = pose
            R, _  = cv2.Rodrigues(rvec)
            tvec  = tvec.reshape(3)

            T_cam_ctrl  = Transform(R, tvec)
            T_ctrl_model = T_model_ctrl.inverse()
            T_cam_model  = T_cam_ctrl.compose(T_ctrl_model)

            assignment = (assignments[idx]
                          if assignments is not None and idx < len(assignments)
                          else None)
            blobs = (blobs_all[idx]
                     if blobs_all is not None and idx < len(blobs_all)
                     else None)

            self._log_frame(idx, T_cam_model, assignment, blobs, camera)

        print(f"[rerun] Logged {n_frames} frames. Open the Rerun viewer to inspect.")

    # ------------------------------------------------------------------
    # Static geometry (logged once, no timeline)
    # ------------------------------------------------------------------

    def _log_static_camera(self, cam):
        """Camera frustum and image-plane outline — static, logged once."""
        frustum_z = 0.2

        corners = compute_frustum_corners(cam, z=frustum_z)
        origin  = np.zeros(3)

        # Frustum edges: 4 rays from origin + 4 edges of the image plane
        frustum_strips = [
            [origin, corners[0]],
            [origin, corners[1]],
            [origin, corners[2]],
            [origin, corners[3]],
        ]
        plane_strip = [corners[0], corners[1], corners[2], corners[3], corners[0]]

        if self.vis_cfg.get("show_frustum", True):
            rr.log(
                "world/camera/frustum",
                rr.LineStrips3D(
                    strips=frustum_strips,
                    colors=[0, 255, 0],
                ),
                static=True,
            )

        if self.vis_cfg.get("show_image_plane", True):
            rr.log(
                "world/camera/image_plane",
                rr.LineStrips3D(
                    strips=[plane_strip],
                    colors=[255, 255, 255],
                ),
                static=True,
            )

        if self.vis_cfg.get("show_camera_frame", True):
            rr.log(
                "world/camera/axes",
                rr.Arrows3D(
                    origins=[[0, 0, 0]] * 3,
                    vectors=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                ),
                static=True,
            )

        self._frustum_z = frustum_z

    # ------------------------------------------------------------------
    # Per-frame logging
    # ------------------------------------------------------------------

    def _log_frame(self, idx: int, T_cam_model: Transform,
                   assignment, blobs, camera):

        offset = self.visual_offset
        R      = T_cam_model.R
        t      = T_cam_model.t

        # ---- transformed positions of model LEDs in camera space ----
        pts_cam = (R @ self.model_positions.T).T + t + offset

        # ---- 4x4 matrix for mesh ----
        T4 = np.eye(4)
        T4[:3, :3] = R
        T4[:3,  3] = t + offset

        # ---- mesh ----
        if self.vis_cfg.get("show_mesh", True):
            verts_world = (T4[:3, :3] @ self._trimesh.vertices.T).T + T4[:3, 3]
            rr.log(
                "world/mesh",
                rr.Mesh3D(
                    vertex_positions=verts_world,
                    triangle_indices=self._trimesh.faces,
                    albedo_factor=[0.7, 0.7, 0.7, 1.0],
                )
            )

        # ---- LEDs ----
        if self.vis_cfg.get("show_leds", True):
            colors = np.tile([255, 0, 0], (len(pts_cam), 1))
            if assignment:
                for _, lid in assignment:
                    colors[lid] = [0, 255, 0]

            rr.log(
                "world/leds",
                rr.Points3D(
                    positions=pts_cam,
                    colors=colors,
                    radii=0.003,
                )
            )

        # ---- normals ----
        if self.vis_cfg.get("show_normals", True):
            normals_cam   = (R @ self.model_normals.T).T
            normal_starts = pts_cam
            normal_ends   = pts_cam + normals_cam * 0.03

            rr.log(
                "world/normals",
                rr.LineStrips3D(
                    strips=[[s, e] for s, e in zip(normal_starts, normal_ends)],
                    colors=[255, 0, 255],
                )
            )

        # ---- rays ----
        if self.vis_cfg.get("show_rays", True) and assignment:
            led_ids  = [lid for _, lid in assignment]
            pts_sel  = pts_cam[led_ids]
            origin   = np.zeros(3)
            ray_strips = [[origin, p] for p in pts_sel]

            rr.log(
                "world/rays",
                rr.LineStrips3D(
                    strips=ray_strips,
                    colors=[0, 0, 255],
                    radii=0.0005
                )
            )

        # ---- blobs & projections & errors ----
        frustum_z = self._frustum_z

        if blobs is not None and len(blobs) > 0:
            pts_plane = backproject_to_plane(blobs, camera, z=frustum_z)

            if self.vis_cfg.get("show_blobs", True):
                rr.log(
                    "world/blobs",
                    rr.Points3D(
                        positions=pts_plane,
                        colors=np.tile([255, 255, 0], (len(pts_plane), 1)),
                        radii=0.0015,
                    )
                )

            # projected LED positions onto image plane
            proj_plane = []
            for X, Y, Z in pts_cam:
                if Z <= 1e-6:
                    continue
                scale = frustum_z / Z
                proj_plane.append([X * scale, Y * scale, frustum_z])
            proj_plane = np.array(proj_plane, dtype=np.float32)

            if self.vis_cfg.get("show_projected", True) and len(proj_plane):
                rr.log(
                    "world/projected_leds",
                    rr.Points3D(
                        positions=proj_plane,
                        colors=np.tile([0, 255, 0], (len(proj_plane), 1)),
                        radii=0.0015,
                    )
                )

            # error lines between projected LEDs and blobs
            if (self.vis_cfg.get("show_errors", True)
                    and len(proj_plane)
                    and len(pts_plane)):

                n = min(len(proj_plane), len(pts_plane))
                error_strips = [
                    [proj_plane[i], pts_plane[i]]
                    for i in range(n)
                ]
                rr.log(
                    "world/errors",
                    rr.LineStrips3D(
                        strips=error_strips,
                        colors=[255, 0, 0],
                        radii=0.0005
                    )
                )


# =========================================================
# Convenience wrapper
# =========================================================

def prepare_model_geometry(leds, cfg):
    positions = np.array([led.position for led in leds], dtype=np.float64)
    normals   = np.array([led.normal   for led in leds], dtype=np.float64)

    T_model_ctrl = build_alignment_transform(cfg)

    positions_model = T_model_ctrl.apply(positions)
    normals_model   = (T_model_ctrl.R @ normals.T).T

    return positions_model, normals_model, T_model_ctrl
