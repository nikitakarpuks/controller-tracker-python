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
    "frustum_z":        0.05,      # depth of the virtual projection screen (metres).
                                   # Must be less than the closest expected controller depth.
                                   # Pure display parameter — has no effect on matching.
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
    """
    Convert 2D pixel blob positions to 3D points on the visualization plane.

    Uses normalized image coordinates so the result is independent of z
    (changing z just scales the whole plane uniformly, ratios preserved).
    """
    pts = []
    for u, v in blobs:
        x = (u - cam.cx) / cam.fx * z
        y = (v - cam.cy) / cam.fy * z
        pts.append([x, y, z])
    return np.array(pts, dtype=np.float32)


def project_to_plane(pts_cam: np.ndarray, z: float = 0.2) -> np.ndarray:
    """
    Project 3D camera-space LED positions onto the visualization plane at depth z.

    Divides by Z to get normalized coords, then scales by z — same formula
    as backproject_to_plane, so blobs and projected LEDs are comparable.
    Filters points behind the camera.
    """
    out = []
    indices = []
    for i, (X, Y, Z) in enumerate(pts_cam):
        if Z <= 1e-6:
            continue
        x_n = X / Z   # normalized image coords
        y_n = Y / Z
        out.append([x_n * z, y_n * z, z])
        indices.append(i)

    return np.array(out, dtype=np.float32), indices

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
        self.visual_offset   = np.array([0.0, 0.0, 0.0])

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
        frustum_z = self.vis_cfg.get("frustum_z", 0.05)

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

        # Real camera-space positions (no visual offset) — used for projection
        pts_cam_real = (R @ self.model_positions.T).T + t
        # Offset positions for 3-D display only
        pts_cam = pts_cam_real + offset

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

        # normals_cam needed for both display and projection — compute unconditionally
        normals_cam = (R @ self.model_normals.T).T   # (N_leds, 3)

        # ---- normals ----
        if self.vis_cfg.get("show_normals", True):
            rr.log(
                "world/normals",
                rr.LineStrips3D(
                    strips=[[s, e] for s, e in zip(pts_cam,
                                                    pts_cam + normals_cam * 0.03)],
                    colors=[255, 0, 255],
                )
            )

        # ---- blobs, projections, rays (LED→proj), errors ----
        # proj_pt = pinhole projection of the LED's 3D position onto the frustum plane at z=frustum_z.
        # Formula: proj_pt = (X/Z * fz,  Y/Z * fz,  fz)  — same geometry as backproject_to_plane.
        # The red error lines therefore show actual reprojection error, matching the console output.
        # (The LED emission direction / beam-aim is shown separately by world/normals arrows.)
        frustum_z = self._frustum_z

        if blobs is not None and len(blobs) > 0:
            pts_plane = backproject_to_plane(blobs, camera, z=frustum_z)

            if self.vis_cfg.get("show_blobs", True):
                rr.log("world/blobs", rr.Points3D(
                    positions=pts_plane,
                    colors=np.tile([255, 255, 0], (len(pts_plane), 1)),
                    radii=0.0015,
                ))

            if assignment:
                ray_strips   = []
                error_strips = []
                proj_matched = []

                for bid, lid in assignment:
                    if bid >= len(pts_plane):
                        continue
                    real_pos = pts_cam_real[lid]

                    # Pinhole projection of LED onto the frustum plane:
                    # same formula as backproject_to_plane (inverse pinhole at depth fz)
                    X, Y, Z = real_pos
                    if Z <= 1e-6:
                        continue
                    proj_pt = np.array([X / Z * frustum_z,
                                        Y / Z * frustum_z,
                                        frustum_z])
                    proj_matched.append(proj_pt)

                    # Ray: displayed LED (with offset) → projection green dot
                    # Ray: real (unshifted) LED position → its pinhole projection on the frustum plane.
                    # Using real position (not display position) keeps this segment on the true
                    # viewing ray, which passes through the camera origin [0,0,0].
                    if self.vis_cfg.get("show_rays", True):
                        ray_strips.append([pts_cam_real[lid], proj_pt])

                    # Error: projection → blob (shows matching residual on image plane)
                    if self.vis_cfg.get("show_errors", True):
                        error_strips.append([proj_pt, pts_plane[bid]])

                if self.vis_cfg.get("show_projected", True) and proj_matched:
                    rr.log("world/projected_leds", rr.Points3D(
                        positions=np.array(proj_matched),
                        colors=np.tile([0, 255, 0], (len(proj_matched), 1)),
                        radii=0.0015,
                    ))

                if ray_strips:
                    rr.log("world/rays", rr.LineStrips3D(
                        strips=ray_strips,
                        colors=[0, 0, 255],
                        radii=0.0005,
                    ))

                if error_strips:
                    rr.log("world/errors", rr.LineStrips3D(
                        strips=error_strips,
                        colors=[255, 0, 0],
                        radii=0.0005,
                    ))


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
