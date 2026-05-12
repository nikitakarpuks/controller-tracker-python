import numpy as np
import trimesh
import cv2
import rerun as rr
import rerun.blueprint as rrb

from src.transformations import Transform
from src._visibility import _visible_mask
from src.controller import _compute_geometry


# =========================================================
# Visualization config (toggle objects on/off)
# =========================================================

VIS_CONFIG = {
    "show_mesh":        True,
    "show_leds":        True,
    "show_normals":     False,
    "show_rays":        True,
    "show_blobs":       True,
    "show_projected":   True,
    "show_errors":      True,
    "show_image_plane": True,
    "show_camera_frame": False, # unit vector lines
    "show_led_ids":       True,   # LED index labels next to projected disks
    "show_blob_ids":      True,   # blob index labels next to blob contours
    "fps":              30,        # playback speed
    "frustum_z":        0.05,      # depth of the virtual projection screen (metres).
                                   # Must be less than the closest expected controller depth.
                                   # Pure display parameter — has no effect on matching.
    "ray_radius":       0.0002,    # thickness of all rays and normals (metres)
    "led_disk_radius":  0.0015,     # radius of LED disks on the controller body (metres)
    "proj_disk_radius": 0.0004,    # radius of projection disks on the frustum plane (metres)
    "blob_z_offset":         0.001,  # how far blob plane sits in front of frustum_z (metres)
    "matched_proj_z_offset": 0.000,  # how far matched projection disks sit in front of frustum_z (metres)
    "error_z_offset":        0.0005, # how far the 2-D error-line plane sits in front of frustum_z (metres)
    "error_radius":          0.0001, # thickness of reprojection error lines (metres)
    # Error value label appearance
    "show_error_values":       True,
    "error_value_label_size":  12,
    "error_value_label_color": [255, 0, 0],          # RGB — red
    "error_value_label_offset": [0.0006, 0.0006],    # [dx, dy] shift in error-plane metres
    # LED ID label appearance
    "led_id_label_size":     12,            # point marker size that anchors the label (Rerun units)
    "led_id_label_color":    [180, 0, 255], # RGB — violet by default
    "led_id_label_offset":   [0.0006, -0.0006],  # [dx, dy] shift in frustum-plane metres
    # Blob ID label appearance
    "blob_id_label_size":    12,            # point marker size that anchors the label (Rerun units)
    "blob_id_label_color":   [255, 210, 0], # RGB — yellow, matching blob colour
    "blob_id_label_offset":  [-0.0013, 0.0008],  # [dx, dy] shift in blob-plane metres
    # Camera centre / world origin markers
    "camera_center_radius":  0.004,   # metres
    "world_origin_radius":   0.006,   # metres
    "world_origin_color":    [255, 255, 255],
}

# One distinct colour per camera index (cyan / orange / purple / green).
# Used for frustum outlines, filled image planes, centre balls, and blob contours.
CAMERA_COLORS = [
    [  0, 220, 255],   # cam 0 — cyan
    [255, 120,   0],   # cam 1 — orange
    [180,   0, 255],   # cam 2 — purple
    [ 50, 255,  50],   # cam 3 — green
]

def _camera_color(cam_idx: int) -> list:
    return CAMERA_COLORS[cam_idx % len(CAMERA_COLORS)]


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
    rx = cfg["rotation"]["rx"]
    ry = cfg["rotation"]["ry"]
    rz = cfg["rotation"]["rz"]

    t = np.array([
        cfg["translation"]["x"],
        cfg["translation"]["y"],
        cfg["translation"]["z"]
    ])

    R_rz = trimesh.transformations.euler_matrix(0, 0, rz)[:3, :3]
    R_base = trimesh.transformations.euler_matrix(rx, ry, 0)[:3, :3]

    R = R_base.T @ R_rz
    t_final = -R_base.T @ t

    return Transform(R, t_final)  # controller → model


# =========================================================
# Geometry helpers
# =========================================================

def compute_frustum_boundary(cam, z: float, edge_samples: int = 20):
    """
    Returns (N, 3) points tracing the full image boundary at depth z, in camera
    frame, with distortion correctly inverted.  Each of the 4 edges is sampled
    at `edge_samples` points (endpoint=False), so N = 4 * edge_samples.
    Corners are at indices 0, edge_samples, 2*edge_samples, 3*edge_samples.

    Sampling the edges (not just corners) captures the curvature introduced by
    lens distortion: with barrel distortion the edges bow inward, with pincushion
    they bow outward — straight-line corner interpolation misses this.
    """
    w, h = float(cam.width), float(cam.height)
    t = np.linspace(0.0, 1.0, edge_samples, endpoint=False, dtype=np.float32)
    edges_px = np.concatenate([
        np.column_stack([t * w,            np.zeros_like(t)]),   # top
        np.column_stack([np.full_like(t, w), t * h]),             # right
        np.column_stack([(1.0 - t) * w,   np.full_like(t, h)]), # bottom
        np.column_stack([np.zeros_like(t), (1.0 - t) * h]),     # left
    ], axis=0)

    norm = cv2.undistortPoints(
        edges_px.reshape(-1, 1, 2),
        cam.camera_matrix, cam.dist_coeffs,
    ).reshape(-1, 2)

    pts = np.column_stack([norm[:, 0] * z, norm[:, 1] * z, np.full(len(norm), z)])
    return pts.astype(np.float32)


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


def make_disk_mesh(positions: np.ndarray, normals: np.ndarray,
                   colors: np.ndarray, radius: float = 0.003,
                   n_segments: int = 12, surface_offset: float = 0.0005):
    """
    Build a combined triangle mesh of flat disks, one per LED.
    Each disk is centered at position + surface_offset*normal and lies
    perpendicular to the normal (so it sits flush on the controller surface).

    Returns (vertices, faces, vertex_colors).
    """
    all_verts   = []
    all_faces   = []
    all_colors  = []
    vert_offset = 0

    for pos, normal, color in zip(positions, normals, colors):
        n = normal / (np.linalg.norm(normal) + 1e-9)
        center = pos + n * surface_offset

        # Local tangent frame perpendicular to n
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(n, ref);  u /= np.linalg.norm(u)
        v = np.cross(n, u)

        angles = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
        rim = [center + radius * (np.cos(a) * u + np.sin(a) * v) for a in angles]

        all_verts.extend([center] + rim)
        all_colors.extend([color] * (1 + n_segments))

        for i in range(n_segments):
            j = (i + 1) % n_segments
            all_faces.append([vert_offset, vert_offset + 1 + i, vert_offset + 1 + j])

        vert_offset += 1 + n_segments

    return (np.array(all_verts,  dtype=np.float32),
            np.array(all_faces,  dtype=np.int32),
            np.array(all_colors, dtype=np.uint8))


def make_contour_mesh_3d(contours_px: list, cam, frustum_z: float,
                          colors: np.ndarray, undistort: bool = False):
    """
    Back-project blob pixel contours onto the frustum plane and triangulate.

    contours_px : list of (M_i, 2) float32 pixel arrays, one per blob.
    colors      : (N_blobs, 3or4) uint8 colour per blob.
    undistort   : if True, undistort contour points before backprojecting so
                  they align with the pinhole LED projections in the 3-D view.
    Returns (vertices, faces, vertex_colors) or (None, None, None) if empty.
    """
    all_verts  = []
    all_faces  = []
    all_colors = []
    vert_offset = 0

    for cnt, color in zip(contours_px, colors):
        n = len(cnt)
        if n < 3:
            continue
        if undistort:
            cnt = cv2.undistortPoints(
                cnt.reshape(-1, 1, 2).astype(np.float32),
                cam.camera_matrix, cam.dist_coeffs,
                P=cam.camera_matrix,
            ).reshape(-1, 2)
        pts = np.column_stack([
            (cnt[:, 0] - cam.cx) / cam.fx * frustum_z,
            (cnt[:, 1] - cam.cy) / cam.fy * frustum_z,
            np.full(n, frustum_z, dtype=np.float32),
        ]).astype(np.float32)

        center = pts.mean(axis=0)
        all_verts.append(center)
        all_verts.extend(pts)
        all_colors.extend([color] * (1 + n))
        for i in range(n):
            j = (i + 1) % n
            all_faces.append([vert_offset, vert_offset + 1 + i, vert_offset + 1 + j])
        vert_offset += 1 + n

    if not all_verts:
        return None, None, None
    return (np.array(all_verts,  dtype=np.float32),
            np.array(all_faces,  dtype=np.int32),
            np.array(all_colors, dtype=np.uint8))


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
        colors = np.tile([255, 0, 0], (len(model_positions), 1))
        verts, faces, vcols = make_disk_mesh(model_positions, model_normals, colors)
        rr.log(
            "world/leds",
            rr.Mesh3D(
                vertex_positions=verts,
                triangle_indices=faces,
                vertex_colors=vcols,
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
                 vis_cfg: dict = None,
                 matching_cfg: dict = None):

        self.mesh_path       = mesh_path
        self.model_positions = model_positions.astype(np.float32)
        self.model_normals   = model_normals.astype(np.float32)
        self.vis_cfg         = vis_cfg if vis_cfg is not None else dict(VIS_CONFIG)
        self._matching_cfg   = matching_cfg or {}

        self._trimesh        = load_trimesh(mesh_path)
        self.visual_offset   = np.array([0.0, 0.0, 0.0])

        self._geom = None  # computed in start() once ctrl-space positions are available

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Blueprint: per-camera tabs + global overview
    # ------------------------------------------------------------------

    def _build_blueprint(self, cameras: dict, frustum_z: float) -> rrb.Blueprint:
        """
        Build a tabbed blueprint with one global "World" view and one
        per-camera view.  Each camera tab:
          - positions the 3-D eye at that camera's centre, looking along its
            optical axis toward the projection surface at frustum_z depth;
          - hides every other camera's frustum, blobs, and markers.
        """
        all_cam_indices = sorted(cameras.keys())
        views: list = [rrb.Spatial3DView(name="World", origin="/world")]

        for cam_idx, cam in sorted(cameras.items()):
            T_wc = cam.T_world_cam
            if T_wc is None:
                continue

            # Camera centre and optical axis in world frame.
            cam_pos_w = T_wc.t.copy()
            fwd_w = T_wc.R @ np.array([0.0, 0.0, 1.0])
            # OpenCV Y is down → world-up is -Y in camera frame.
            up_w  = T_wc.R @ np.array([0.0, -1.0, 0.0])

            # Look-at: centre of the projection surface at depth frustum_z.
            look_target_w = cam_pos_w + fwd_w * frustum_z

            eye_ctrl = rrb.EyeControls3D(
                kind=rrb.Eye3DKind.FirstPerson,
                position=cam_pos_w.tolist(),
                look_target=look_target_w.tolist(),
                eye_up=up_w.tolist(),
            )

            # Include everything, then exclude every other camera's subtree.
            contents = ["$origin/**"] + [
                f"- world/camera_{other}/**"
                for other in all_cam_indices
                if other != cam_idx
            ]

            color = _camera_color(cam_idx)
            views.append(rrb.Spatial3DView(
                name=f"Camera {cam_idx}",
                origin="/world",
                contents=contents,
                eye_controls=eye_ctrl,
                background=rrb.Background(
                    color=[int(c * 0.08) for c in color] + [255],
                ),
            ))

        return rrb.Blueprint(
            rrb.Tabs(*views),
            collapse_panels=False,
        )

    def start(self, poses, assignments, blobs_all, cameras: dict, T_model_ctrl,
              contours_all=None, raw_blobs_all=None, raw_contours_all=None,
              save_path: str = None,
              primary_cams_all: list = None,
              aux_assignments_all: list = None):
        """
        Log all frames to rerun.
        Opens the viewer automatically (spawn=True).

        Args:
            cameras   : {cam_idx: Camera} for all selected cameras.
            save_path : Optional path for an .rrd recording file.
                        When set, the session is saved to disk so it can be
                        replayed later with:  rerun <save_path>
        """
        self._cameras     = cameras
        self._T_world_cam = cameras[0].T_world_cam  # cam-0 only — used for tracked geometry

        # When a file is being saved the viewer is not spawned — replay manually.
        spawn_viewer = save_path is None
        rr.init("controller_animator", spawn=spawn_viewer)

        if save_path:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            rr.save(save_path)
            print(f"[rerun] Saving recording to: {save_path}  (no live viewer)")
            print(f"[rerun] Replay later with:   rerun {save_path}")

        # Per-camera tabbed blueprint — one tab per camera + global "World" tab.
        # Each camera tab snaps the eye to that camera's optical axis.
        blueprint = self._build_blueprint(cameras, frustum_z=self.vis_cfg.get("frustum_z", 0.05))
        rr.send_blueprint(blueprint)

        T_ctrl_model = T_model_ctrl.inverse()
        self._ctrl_positions = T_ctrl_model.apply(self.model_positions.astype(np.float64)).astype(np.float32)
        self._ctrl_normals   = (T_ctrl_model.R @ self.model_normals.astype(np.float64).T).T.astype(np.float32)
        self._T_model_ctrl   = T_model_ctrl
        self._geom           = _compute_geometry(self._ctrl_positions, self._ctrl_normals)

        self._log_static_cameras(cameras)

        if self.vis_cfg.get("show_mesh", True):
            rr.log(
                "world/mesh",
                rr.Mesh3D(
                    vertex_positions=self._trimesh.vertices,
                    triangle_indices=self._trimesh.faces,
                    vertex_normals=self._trimesh.vertex_normals if hasattr(self._trimesh, "vertex_normals") else None,
                    albedo_factor=[0.7, 0.7, 0.7, 1.0],
                ),
                static=True,
            )

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
            contours = (contours_all[idx]
                        if contours_all is not None and idx < len(contours_all)
                        else None)
            raw_blobs = (raw_blobs_all[idx]
                         if raw_blobs_all is not None and idx < len(raw_blobs_all)
                         else None)
            raw_contours = (raw_contours_all[idx]
                            if raw_contours_all is not None and idx < len(raw_contours_all)
                            else None)
            primary_cam_idx = (
                primary_cams_all[idx]
                if primary_cams_all is not None and idx < len(primary_cams_all)
                   and primary_cams_all[idx] is not None
                else 0
            )
            aux_assignments = (
                aux_assignments_all[idx]
                if aux_assignments_all is not None and idx < len(aux_assignments_all)
                else None
            )

            self._log_frame(idx, T_cam_model, assignment, blobs, contours,
                            raw_blobs_per_cam=raw_blobs,
                            raw_contours_per_cam=raw_contours,
                            primary_cam_idx=primary_cam_idx,
                            aux_assignments=aux_assignments)

        msg = f"[rerun] Logged {n_frames} frames."
        if save_path:
            msg += f" Recording saved to: {save_path}"
            msg += f"\n[rerun] Replay with:  rerun {save_path}"
        print(msg)

    # ------------------------------------------------------------------
    # Static geometry (logged once, no timeline)
    # ------------------------------------------------------------------

    def _log_static_cameras(self, cameras: dict):
        """Frustum image planes and camera markers — logged once."""
        frustum_z    = self.vis_cfg.get("frustum_z", 0.05)
        edge_samples = 20
        cam_r        = self.vis_cfg.get("camera_center_radius", 0.004)

        for cam_idx, cam in cameras.items():
            color       = _camera_color(cam_idx)
            color_dim   = [int(c * 0.25) for c in color]  # dim fill for the plane mesh
            T_world_cam = cam.T_world_cam
            boundary    = compute_frustum_boundary(cam, z=frustum_z, edge_samples=edge_samples)
            origin      = np.zeros(3)

            if T_world_cam is not None:
                boundary = T_world_cam.apply(boundary)
                origin   = T_world_cam.t.copy()

            path = f"world/camera_{cam_idx}"

            # ---- image-plane boundary outline (camera colour) ----
            if self.vis_cfg.get("show_image_plane", True):
                plane_strip = np.concatenate([boundary, boundary[:1]], axis=0).tolist()
                rr.log(f"{path}/image_plane",
                       rr.LineStrips3D(strips=[plane_strip], colors=[color]),
                       static=True)

            # ---- camera-centre ball + ID label ----
            rr.log(f"{path}/center",
                   rr.Points3D(positions=[origin.tolist()], colors=[color], radii=cam_r),
                   static=True)
            rr.log(f"{path}/id_label",
                   rr.Points3D(positions=[origin.tolist()],
                               labels=[f"cam {cam_idx}"],
                               colors=[color],
                               radii=0.0),
                   static=True)

            # ---- optional camera-frame axes ----
            if self.vis_cfg.get("show_camera_frame", True):
                axes_w = (T_world_cam.R @ np.eye(3) * 0.1).T if T_world_cam is not None else np.eye(3) * 0.1
                rr.log(f"{path}/axes",
                       rr.Arrows3D(origins=[origin.tolist()] * 3, vectors=axes_w.tolist(),
                                   colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
                       static=True)

        # ---- world-origin ball ----
        wo_color  = self.vis_cfg.get("world_origin_color",  [255, 255, 255])
        wo_radius = self.vis_cfg.get("world_origin_radius", 0.006)
        rr.log("world/origin",
               rr.Points3D(positions=[[0, 0, 0]], colors=[wo_color], radii=wo_radius),
               static=True)

        self._frustum_z = frustum_z

    # ------------------------------------------------------------------
    # Per-camera blob logging (used for cameras other than cam-0)
    # ------------------------------------------------------------------

    def _log_camera_blobs(self, cam_idx: int, cam,
                          blobs, contours, raw_blobs, raw_contours,
                          blob_z: float, raw_blob_z: float):
        """Log blob contours, IDs, and raw blobs for a single camera."""
        T_world_cam = cam.T_world_cam
        color       = _camera_color(cam_idx)
        color_dim   = [int(c * 0.5) for c in color]

        def _wc(pts):
            if T_world_cam is None:
                return pts
            return T_world_cam.apply(np.asarray(pts, dtype=np.float64))

        path = f"world/camera_{cam_idx}"

        if blobs is not None and len(blobs) > 0 and self.vis_cfg.get("show_blobs", True) and contours is not None:
            blob_colors = np.array([color] * len(contours), dtype=np.uint8)
            bv, bf, bc = make_contour_mesh_3d(contours, cam, blob_z, blob_colors, undistort=True)
            if bv is not None:
                rr.log(f"{path}/blobs", rr.Mesh3D(vertex_positions=_wc(bv),
                                                   triangle_indices=bf, vertex_colors=bc))
            else:
                rr.log(f"{path}/blobs", rr.Clear(recursive=False))
        else:
            rr.log(f"{path}/blobs", rr.Clear(recursive=False))

        if blobs is not None and len(blobs) > 0 and self.vis_cfg.get("show_blob_ids", True):
            blobs_ud = cv2.undistortPoints(
                blobs.reshape(-1, 1, 2).astype(np.float32),
                cam.camera_matrix, cam.dist_coeffs, P=cam.camera_matrix,
            ).reshape(-1, 2)
            bdx, bdy = self.vis_cfg.get("blob_id_label_offset", [-0.0013, 0.0008])
            label_pts = backproject_to_plane(blobs_ud, cam, z=blob_z).copy()
            label_pts[:, 0] += bdx
            label_pts[:, 1] += bdy
            rr.log(f"{path}/blob_ids", rr.Points3D(
                positions=_wc(label_pts),
                labels=[str(i) for i in range(len(blobs))],
                colors=[color] * len(blobs),
                radii=0.0,
            ))
        else:
            rr.log(f"{path}/blob_ids", rr.Clear(recursive=False))

        if self.vis_cfg.get("show_raw_blobs", True) and raw_blobs is not None and len(raw_blobs) > 0 and raw_contours is not None:
            raw_colors = np.array([[255, 192, 203]] * len(raw_contours), dtype=np.uint8)
            rv, rf, rc = make_contour_mesh_3d(raw_contours, cam, raw_blob_z, raw_colors, undistort=True)
            if rv is not None:
                rr.log(f"{path}/blobs_raw", rr.Mesh3D(vertex_positions=_wc(rv),
                                                       triangle_indices=rf, vertex_colors=rc))
            else:
                rr.log(f"{path}/blobs_raw", rr.Clear(recursive=False))
        else:
            rr.log(f"{path}/blobs_raw", rr.Clear(recursive=False))

    # ------------------------------------------------------------------
    # Per-frame logging
    # ------------------------------------------------------------------

    def _log_frame(self, idx: int, T_cam_model: Transform,
                   assignment, blobs_per_cam: dict, contours_per_cam: dict = None,
                   raw_blobs_per_cam: dict = None, raw_contours_per_cam: dict = None,
                   primary_cam_idx: int = 0, aux_assignments: dict = None):

        # Fall back to first available camera if primary is missing (e.g. during handoff).
        if primary_cam_idx not in self._cameras:
            primary_cam_idx = next(iter(self._cameras))
        cam_path     = f"world/camera_{primary_cam_idx}"

        camera       = self._cameras[primary_cam_idx]
        T_world_cam  = camera.T_world_cam
        blobs        = blobs_per_cam.get(primary_cam_idx) if blobs_per_cam else None
        contours     = contours_per_cam.get(primary_cam_idx) if contours_per_cam else None
        raw_blobs    = raw_blobs_per_cam.get(primary_cam_idx) if raw_blobs_per_cam else None
        raw_contours = raw_contours_per_cam.get(primary_cam_idx) if raw_contours_per_cam else None

        offset       = self.visual_offset

        # Camera-frame rotation/translation — used for projection math only.
        R_cam = T_cam_model.R
        t_cam = T_cam_model.t

        # Display (world) frame transform — all 3D Rerun logs use this.
        # When T_world_cam is None, camera frame is used as world (backward compat).
        if T_world_cam is not None:
            T_disp_model = T_world_cam.compose(T_cam_model)
        else:
            T_disp_model = T_cam_model
        R_disp = T_disp_model.R
        t_disp = T_disp_model.t

        # Helper: transform a set of camera-frame 3D points to display/world frame.
        def _w(pts):
            if T_world_cam is None:
                return pts
            pts_arr = np.asarray(pts, dtype=np.float64)
            if pts_arr.ndim == 1:
                return T_world_cam.apply(pts_arr.reshape(1, 3))[0]
            return T_world_cam.apply(pts_arr)

        ray_radius            = self.vis_cfg.get("ray_radius",            0.0002)
        led_disk_radius       = self.vis_cfg.get("led_disk_radius",       0.003)
        proj_disk_radius      = self.vis_cfg.get("proj_disk_radius",      0.0008)
        blob_z_offset         = self.vis_cfg.get("blob_z_offset",         0.002)
        matched_proj_z_offset = self.vis_cfg.get("matched_proj_z_offset", 0.001)
        error_z_offset        = self.vis_cfg.get("error_z_offset",        0.0015)
        error_radius          = self.vis_cfg.get("error_radius",          0.0001)

        # LED positions in camera frame (for projection) and display frame (for logging).
        pts_cam_real  = (R_cam @ self.model_positions.T).T + t_cam
        pts_disp_real = (R_disp @ self.model_positions.T).T + t_disp
        pts_disp      = pts_disp_real + offset

        # ---- 4x4 matrix for mesh (display frame) ----
        T4 = np.eye(4)
        T4[:3, :3] = R_disp
        T4[:3,  3] = t_disp + offset

        # ---- mesh (transform only — geometry is logged once as static) ----
        if self.vis_cfg.get("show_mesh", True):
            rr.log(
                "world/mesh",
                rr.Transform3D(
                    translation=T4[:3, 3],
                    mat3x3=T4[:3, :3],
                )
            )

        # Normals in camera frame (for visibility mask) and display frame (for LED disks).
        normals_cam  = (R_cam  @ self.model_normals.T).T
        normals_disp = (R_disp @ self.model_normals.T).T

        # ---- per-frame visibility mask ----
        # _visible_mask takes camera-frame R, t — use R_cam / t_cam.
        T_cam_ctrl = T_cam_model.compose(self._T_model_ctrl)
        facing_threshold_deg = float(self._matching_cfg.get('led_facing_angle_deg', 86.0))
        vis_mask = _visible_mask(
            T_cam_ctrl.R, T_cam_ctrl.t,
            self._ctrl_positions, self._ctrl_normals,
            self._geom,
            cam_K=camera.camera_matrix, cam_dc=camera.dist_coeffs,
            cam_w=camera.width, cam_h=camera.height,
            cam_rpmax=camera.rpmax,
            facing_threshold_deg=facing_threshold_deg,
        )
        vis_set = set(np.where(vis_mask)[0].tolist())

        # ---- LEDs (flat disks lying on the controller surface) ----
        # green  : matched and model-visible
        # pink   : matched but not model-visible (vis_mask approximation overridden by blob evidence)
        # red    : unmatched
        if self.vis_cfg.get("show_leds", True):
            colors = np.tile([255, 192, 203], (len(pts_disp), 1))
            verts, faces, vcols = make_disk_mesh(pts_disp, normals_disp, colors,
                                                 radius=led_disk_radius)
            rr.log(
                "world/leds",
                rr.Mesh3D(
                    vertex_positions=verts,
                    triangle_indices=faces,
                    vertex_colors=vcols,
                )
            )

        # ---- normals ----
        if self.vis_cfg.get("show_normals", True):
            rr.log(
                "world/normals",
                rr.LineStrips3D(
                    strips=[[s, e] for s, e in zip(pts_disp,
                                                    pts_disp + normals_disp * 0.03)],
                    colors=[255, 0, 255],
                    radii=ray_radius,
                )
            )

        # ---- blobs, projections, rays (LED→proj), errors ----
        # Z-layer ordering (front → back), all depths in camera frame:
        #   blob contours        : frustum_z - blob_z_offset
        #   matched projections  : frustum_z - matched_proj_z_offset
        #   unmatched projections: frustum_z
        frustum_z        = self._frustum_z
        blob_z           = frustum_z - blob_z_offset
        matched_proj_z   = frustum_z - matched_proj_z_offset
        error_z          = frustum_z - error_z_offset

        # Matched indices
        matched_bids = set()
        matched_lids = set()
        if assignment:
            for bid, lid in assignment:
                matched_bids.add(bid)
                matched_lids.add(lid)

        # ---- blobs as actual contour shapes ----
        if blobs is not None and len(blobs) > 0:
            # Undistort blob centroids so they live in the same space as the
            # pinhole LED projections (proj_flat = px/pz * z).
            blobs_undist_disp = cv2.undistortPoints(
                blobs.reshape(-1, 1, 2).astype(np.float32),
                camera.camera_matrix, camera.dist_coeffs,
                P=camera.camera_matrix,
            ).reshape(-1, 2)

            # pts_plane: blob positions on the error-z plane, camera frame.
            # Transformed to display frame for logging.
            pts_plane_cam  = backproject_to_plane(blobs_undist_disp, camera, z=error_z)
            pts_plane_disp = _w(pts_plane_cam)

            if self.vis_cfg.get("show_blobs", True) and contours is not None:
                primary_color     = _camera_color(primary_cam_idx)
                primary_color_dim = [int(c * 0.5) for c in primary_color]
                blob_colors = np.array(
                    [primary_color if i in matched_bids else primary_color_dim
                     for i in range(len(contours))],
                    dtype=np.uint8,
                )
                bv, bf, bc = make_contour_mesh_3d(
                    contours, camera, blob_z, blob_colors,
                    undistort=True,
                )
                if bv is not None:
                    rr.log(f"{cam_path}/blobs", rr.Mesh3D(
                        vertex_positions=_w(bv) if bv is not None else bv,
                        triangle_indices=bf,
                        vertex_colors=bc,
                    ))
                else:
                    rr.log(f"{cam_path}/blobs", rr.Clear(recursive=False))
            elif self.vis_cfg.get("show_blobs", True):
                rr.log(f"{cam_path}/blobs", rr.Clear(recursive=False))

            # ---- blob ID labels on the blob plane ----
            if self.vis_cfg.get("show_blob_ids", True):
                blabel_color  = self.vis_cfg.get("blob_id_label_color",  [255, 210, 0])
                blabel_offset = self.vis_cfg.get("blob_id_label_offset", [0.0006, -0.0006])
                bdx, bdy = blabel_offset[0], blabel_offset[1]
                blob_label_pts_cam = backproject_to_plane(blobs_undist_disp, camera, z=blob_z).copy()
                blob_label_pts_cam[:, 0] += bdx
                blob_label_pts_cam[:, 1] += bdy
                rr.log(f"{cam_path}/blob_ids", rr.Points3D(
                    positions=_w(blob_label_pts_cam),
                    labels=[str(i) for i in range(len(blobs))],
                    colors=[blabel_color] * len(blobs),
                    radii=0.0,
                ))
        else:
            pts_plane_disp = None
            rr.log(f"{cam_path}/blobs",    rr.Clear(recursive=False))
            rr.log(f"{cam_path}/blob_ids", rr.Clear(recursive=False))

        # ---- raw (pre-filter) blobs ----
        if self.vis_cfg.get("show_raw_blobs", True) and raw_blobs is not None and len(raw_blobs) > 0:
            if raw_contours is not None:
                raw_colors = np.array([[255,192,203]] * len(raw_contours), dtype=np.uint8)
                rv, rf, rc = make_contour_mesh_3d(
                    raw_contours, camera, blob_z - 0.001, raw_colors,
                    undistort=True,
                )
                if rv is not None:
                    rr.log(f"{cam_path}/blobs_raw", rr.Mesh3D(
                        vertex_positions=_w(rv),
                        triangle_indices=rf,
                        vertex_colors=rc,
                    ))
                else:
                    rr.log(f"{cam_path}/blobs_raw", rr.Clear(recursive=False))
            else:
                rr.log(f"{cam_path}/blobs_raw", rr.Clear(recursive=False))
        else:
            rr.log(f"{cam_path}/blobs_raw", rr.Clear(recursive=False))

        # Clear stale per-camera dynamic geometry for all non-primary cameras so old
        # data does not bleed through when the primary camera index changes frame to frame.
        for _ci in self._cameras:
            if _ci != primary_cam_idx:
                for _sub in ("rays", "projected_leds", "led_ids", "errors", "error_values"):
                    rr.log(f"world/camera_{_ci}/{_sub}", rr.Clear(recursive=False))

        # ---- visible LED projections (matched + physically visible unmatched) ----
        # Compute projection in camera frame, then transform to display frame.
        # lid_to_proj_pt stores display-frame points (used for rays and error lines).
        all_proj_pts   = []   # display frame
        all_proj_lids  = []
        lid_to_proj_pt = {}   # display frame
        for i, (px, py, pz) in enumerate(pts_cam_real):
            if pz > 1e-6 and (i in matched_lids or i in vis_set):
                z_val = matched_proj_z if i in matched_lids else frustum_z
                proj_pt_cam  = np.array([px / pz * z_val, py / pz * z_val, z_val])
                proj_pt_disp = _w(proj_pt_cam)
                all_proj_pts.append(proj_pt_disp)
                all_proj_lids.append(i)
                lid_to_proj_pt[i] = proj_pt_disp

        if self.vis_cfg.get("show_projected", True):
            if all_proj_pts:
                proj_colors = np.array(
                    [[70, 130, 255] if lid in matched_lids else [230, 80, 50]
                     for lid in all_proj_lids],
                    dtype=np.uint8,
                )
                # Project normal [0,0,-1] (camera backward) into display frame.
                proj_normal_cam  = np.array([0.0, 0.0, -1.0])
                proj_normal_disp = (T_world_cam.R @ proj_normal_cam
                                    if T_world_cam is not None else proj_normal_cam)
                proj_normals = np.tile(proj_normal_disp,
                                       (len(all_proj_pts), 1)).astype(np.float32)
                pv, pf, pc = make_disk_mesh(
                    np.array(all_proj_pts, dtype=np.float32),
                    proj_normals, proj_colors,
                    radius=proj_disk_radius, n_segments=16, surface_offset=0.0,
                )
                rr.log(f"{cam_path}/projected_leds", rr.Mesh3D(
                    vertex_positions=pv,
                    triangle_indices=pf,
                    vertex_colors=pc,
                ))
            else:
                rr.log(f"{cam_path}/projected_leds", rr.Clear(recursive=False))

        # ---- LED ID labels on the frustum plane ----
        if self.vis_cfg.get("show_led_ids", True):
            if all_proj_pts:
                label_color  = self.vis_cfg.get("led_id_label_color",  [180, 0, 255])
                label_offset = self.vis_cfg.get("led_id_label_offset", [0.0006, -0.0006])
                dx, dy = label_offset[0], label_offset[1]
                # Apply offset in camera frame before transforming to display frame.
                label_pts_cam = np.array([
                    [pts_cam_real[lid, 0] / pts_cam_real[lid, 2] * frustum_z + dx,
                     pts_cam_real[lid, 1] / pts_cam_real[lid, 2] * frustum_z + dy,
                     frustum_z]
                    for lid in all_proj_lids
                ], dtype=np.float32)
                rr.log(f"{cam_path}/led_ids", rr.Points3D(
                    positions=_w(label_pts_cam),
                    labels=[str(lid) for lid in all_proj_lids],
                    colors=[label_color] * len(all_proj_lids),
                    radii=0.0,
                ))
            else:
                rr.log(f"{cam_path}/led_ids", rr.Clear(recursive=False))

        # ---- rays: matched (blue) and unmatched-but-visible (orange-red) ----
        ray_strips = []
        ray_colors = []
        error_strips    = []
        error_values    = []   # pixel reprojection errors, one per matched pair
        error_label_pts = []   # 3-D positions for error value labels

        # Pre-project all matched LEDs at once using the real K + distortion,
        # matching exactly how _matching.py computes its reported error.
        rvec_model, _ = cv2.Rodrigues(R_cam)
        if assignment and pts_plane_disp is not None:
            matched_lids_ord = [lid for _, lid in assignment]
            proj_matched, _ = cv2.projectPoints(
                self.model_positions[matched_lids_ord].astype(np.float32),
                rvec_model,
                t_cam.astype(np.float32).reshape(3, 1),
                camera.camera_matrix,
                camera.dist_coeffs,
            )
            proj_matched = proj_matched.reshape(-1, 2)

            for pair_i, (bid, lid) in enumerate(assignment):
                if bid >= len(pts_plane_disp) or lid not in lid_to_proj_pt:
                    continue
                if self.vis_cfg.get("show_rays", True):
                    ray_strips.append([pts_disp_real[lid].tolist(), lid_to_proj_pt[lid].tolist()])
                    ray_colors.append([70, 130, 255])
                px, py, pz = pts_cam_real[lid]
                if pz > 1e-6:
                    proj_flat_cam  = np.array([px / pz * error_z, py / pz * error_z, error_z])
                    proj_flat_disp = _w(proj_flat_cam)
                    if self.vis_cfg.get("show_errors", True):
                        error_strips.append([proj_flat_disp.tolist(), pts_plane_disp[bid].tolist()])
                    u, v = blobs[bid]
                    pu, pv = proj_matched[pair_i]
                    err_px = np.sqrt((u - pu) ** 2 + (v - pv) ** 2)
                    error_values.append(err_px)
                    error_label_pts.append((proj_flat_disp + pts_plane_disp[bid]) / 2)

        # Unmatched rays: visible LEDs not in the assignment
        if self.vis_cfg.get("show_rays", True):
            for lid, proj_pt in lid_to_proj_pt.items():
                if lid not in matched_lids:
                    ray_strips.append([pts_disp_real[lid].tolist(), proj_pt.tolist()])
                    ray_colors.append([230, 80, 50])

        # Primary camera rays.
        rr.log(f"{cam_path}/rays", rr.LineStrips3D(
            strips=ray_strips,
            colors=ray_colors if ray_strips else [[0, 0, 0]],
            radii=ray_radius,
        ) if ray_strips else rr.Clear(recursive=False))

        # Aux cameras: full overlay — projected LEDs, LED IDs, rays, errors, error values.
        for _aux_ci, _aux_pairs in (aux_assignments or {}).items():
            _acp     = f"world/camera_{_aux_ci}"
            _aux_cam = self._cameras.get(_aux_ci)
            if _aux_cam is None or not _aux_pairs:
                for _sub in ("rays", "projected_leds", "led_ids", "errors", "error_values"):
                    rr.log(f"{_acp}/{_sub}", rr.Clear(recursive=False))
                continue

            _T_world_aux = _aux_cam.T_world_cam
            _T_aux_model = (_T_world_aux.inverse().compose(T_disp_model)
                            if _T_world_aux is not None else T_disp_model)
            _pts_aux_cam = (_T_aux_model.R @ self.model_positions.T).T + _T_aux_model.t
            _aux_color   = _camera_color(_aux_ci)
            _aux_blobs_raw = blobs_per_cam.get(_aux_ci) if blobs_per_cam else None
            _aux_blobs_arr = (np.asarray(_aux_blobs_raw, dtype=np.float32)
                              if _aux_blobs_raw is not None and len(_aux_blobs_raw) > 0 else None)

            # Shorthand: single point or array → world frame.
            _R_aw = _T_world_aux.R if _T_world_aux is not None else np.eye(3)
            _t_aw = _T_world_aux.t if _T_world_aux is not None else np.zeros(3)

            _aux_matched_lids = {_lid for _, _lid in _aux_pairs}

            # ── projected LEDs ──────────────────────────────────────────────
            _aux_proj_pts    = []
            _aux_proj_lids   = []
            _aux_lid_to_proj = {}
            for _i, (_ipx, _ipy, _ipz) in enumerate(_pts_aux_cam):
                if _ipz > 1e-6 and _i in _aux_matched_lids:
                    _pp  = np.array([_ipx / _ipz * matched_proj_z,
                                     _ipy / _ipz * matched_proj_z,
                                     matched_proj_z])
                    _pw  = _R_aw @ _pp + _t_aw
                    _aux_proj_pts.append(_pw)
                    _aux_proj_lids.append(_i)
                    _aux_lid_to_proj[_i] = _pw

            if self.vis_cfg.get("show_projected", True):
                if _aux_proj_pts:
                    _pn_world = _R_aw @ np.array([0.0, 0.0, -1.0])
                    _pnormals = np.tile(_pn_world, (len(_aux_proj_pts), 1)).astype(np.float32)
                    _apv, _apf, _apc = make_disk_mesh(
                        np.array(_aux_proj_pts, dtype=np.float32), _pnormals,
                        np.array([_aux_color] * len(_aux_proj_pts), dtype=np.uint8),
                        radius=proj_disk_radius, n_segments=16, surface_offset=0.0,
                    )
                    rr.log(f"{_acp}/projected_leds",
                           rr.Mesh3D(vertex_positions=_apv, triangle_indices=_apf, vertex_colors=_apc))
                else:
                    rr.log(f"{_acp}/projected_leds", rr.Clear(recursive=False))

            # ── LED ID labels ───────────────────────────────────────────────
            if self.vis_cfg.get("show_led_ids", True):
                if _aux_proj_lids:
                    _lbl_color = self.vis_cfg.get("led_id_label_color",  [180, 0, 255])
                    _ldx, _ldy = self.vis_cfg.get("led_id_label_offset", [0.0006, -0.0006])
                    _lbl_pts_cam = np.array([
                        [_pts_aux_cam[_lid, 0] / _pts_aux_cam[_lid, 2] * frustum_z + _ldx,
                         _pts_aux_cam[_lid, 1] / _pts_aux_cam[_lid, 2] * frustum_z + _ldy,
                         frustum_z]
                        for _lid in _aux_proj_lids
                    ], dtype=np.float32)
                    _lbl_pts_world = (_R_aw @ _lbl_pts_cam.T).T + _t_aw
                    rr.log(f"{_acp}/led_ids", rr.Points3D(
                        positions=_lbl_pts_world,
                        labels=[str(_lid) for _lid in _aux_proj_lids],
                        colors=[_lbl_color] * len(_aux_proj_lids),
                        radii=0.0,
                    ))
                else:
                    rr.log(f"{_acp}/led_ids", rr.Clear(recursive=False))

            # ── rays, errors, error values ──────────────────────────────────
            _aux_ray_strips  = []
            _aux_err_strips  = []
            _aux_err_vals    = []
            _aux_err_lbl_pts = []
            _aux_error_z     = frustum_z - error_z_offset

            if _aux_blobs_arr is not None:
                _rv_aux, _ = cv2.Rodrigues(_T_aux_model.R.astype(np.float32))
                _tv_aux    = _T_aux_model.t.astype(np.float32).reshape(3, 1)
                _aux_lids_ord = [_lid for _, _lid in _aux_pairs]
                _proj_px_aux, _ = cv2.projectPoints(
                    self.model_positions[_aux_lids_ord].astype(np.float32),
                    _rv_aux, _tv_aux,
                    _aux_cam.camera_matrix, _aux_cam.dist_coeffs,
                )
                _proj_px_aux = _proj_px_aux.reshape(-1, 2)

                for _pi, (_blob_j, _lid) in enumerate(_aux_pairs):
                    if _blob_j >= len(_aux_blobs_arr) or _lid not in _aux_lid_to_proj:
                        continue

                    if self.vis_cfg.get("show_rays", True):
                        _aux_ray_strips.append(
                            [pts_disp_real[_lid].tolist(), _aux_lid_to_proj[_lid].tolist()]
                        )

                    _lp = _pts_aux_cam[_lid]
                    if _lp[2] > 1e-6 and self.vis_cfg.get("show_errors", True):
                        _bu, _bv = float(_aux_blobs_arr[_blob_j, 0]), float(_aux_blobs_arr[_blob_j, 1])
                        _b_ud = cv2.undistortPoints(
                            np.array([[_bu, _bv]], dtype=np.float32).reshape(1, 1, 2),
                            _aux_cam.camera_matrix, _aux_cam.dist_coeffs,
                            P=_aux_cam.camera_matrix,
                        ).reshape(2)
                        _blob_plane_aux = backproject_to_plane(
                            _b_ud.reshape(1, 2), _aux_cam, z=_aux_error_z
                        )[0]
                        _blob_plane_w = _R_aw @ _blob_plane_aux + _t_aw

                        _proj_flat_aux = np.array([
                            _lp[0] / _lp[2] * _aux_error_z,
                            _lp[1] / _lp[2] * _aux_error_z,
                            _aux_error_z,
                        ])
                        _proj_flat_w = _R_aw @ _proj_flat_aux + _t_aw
                        _aux_err_strips.append([_proj_flat_w.tolist(), _blob_plane_w.tolist()])

                        _pu, _pv_val = float(_proj_px_aux[_pi, 0]), float(_proj_px_aux[_pi, 1])
                        _aux_err_vals.append(float(np.sqrt((_bu - _pu)**2 + (_bv - _pv_val)**2)))
                        _aux_err_lbl_pts.append((_proj_flat_w + _blob_plane_w) / 2)

            rr.log(f"{_acp}/rays",
                   rr.LineStrips3D(strips=_aux_ray_strips,
                                   colors=[_aux_color] * len(_aux_ray_strips),
                                   radii=ray_radius)
                   if _aux_ray_strips and self.vis_cfg.get("show_rays", True)
                   else rr.Clear(recursive=False))

            rr.log(f"{_acp}/errors",
                   rr.LineStrips3D(strips=_aux_err_strips, colors=[255, 0, 0], radii=error_radius)
                   if _aux_err_strips else rr.Clear(recursive=False))

            if self.vis_cfg.get("show_error_values", True):
                if _aux_err_lbl_pts:
                    _ev_color = self.vis_cfg.get("error_value_label_color", [255, 0, 0])
                    _ev_off   = self.vis_cfg.get("error_value_label_offset", [0.0006, 0.0006])
                    _ev_pts   = np.array(_aux_err_lbl_pts, dtype=np.float32).copy()
                    _ev_pts[:, 0] += _ev_off[0]
                    _ev_pts[:, 1] += _ev_off[1]
                    rr.log(f"{_acp}/error_values", rr.Points3D(
                        positions=_ev_pts,
                        labels=[f"{_e:.2f}" for _e in _aux_err_vals],
                        colors=[_ev_color] * len(_ev_pts),
                        radii=0.0,
                    ))
                else:
                    rr.log(f"{_acp}/error_values", rr.Clear(recursive=False))

        rr.log(f"{cam_path}/errors", rr.LineStrips3D(
            strips=error_strips,
            colors=[255, 0, 0],
            radii=error_radius,
        ) if error_strips else rr.Clear(recursive=False))

        if self.vis_cfg.get("show_error_values", True):
            if error_label_pts:
                ev_color  = self.vis_cfg.get("error_value_label_color",  [255, 0, 0])
                ev_offset = self.vis_cfg.get("error_value_label_offset", [0.0006, 0.0006])
                ev_pts = np.array(error_label_pts, dtype=np.float32).copy()
                ev_pts[:, 0] += ev_offset[0]
                ev_pts[:, 1] += ev_offset[1]
                rr.log(f"{cam_path}/error_values", rr.Points3D(
                    positions=ev_pts,
                    labels=[f"{e:.2f}" for e in error_values],
                    colors=[ev_color] * len(ev_pts),
                    radii=0.0,
                ))
            else:
                rr.log(f"{cam_path}/error_values", rr.Clear(recursive=False))

        # ---- blobs for all cameras except the primary (handled above) ----
        for cam_idx, cam in self._cameras.items():
            if cam_idx == primary_cam_idx:
                continue
            self._log_camera_blobs(
                cam_idx, cam,
                blobs_per_cam.get(cam_idx) if blobs_per_cam else None,
                contours_per_cam.get(cam_idx) if contours_per_cam else None,
                raw_blobs_per_cam.get(cam_idx) if raw_blobs_per_cam else None,
                raw_contours_per_cam.get(cam_idx) if raw_contours_per_cam else None,
                blob_z, blob_z - 0.001,
            )


# =========================================================
# Alignment fine-tuning
# =========================================================

def fine_tune_alignment(leds, mesh, cfg) -> dict:
    """
    Refine the 6 alignment parameters so every LED disk sits exactly on the
    mesh surface with its normal pointing outward.

    Key corrections over the naive approach
    ----------------------------------------
    1. **Signed distance** — we distinguish LEDs that are INSIDE the mesh
       (wrong side) from those that are outside.  Being inside is penalised
       20× more than being the same distance outside, so the optimiser is
       forced to push LEDs to the correct (outer) side of the surface.

    2. **Oriented normal alignment** — abs() is NOT used.  LED normals are
       expected to point outward (same direction as mesh face normals).
       Using the raw dot product gives a real gradient signal: a normal
       pointing 180° the wrong way (into the surface) now has maximum loss
       rather than zero.

    3. **Tight search bounds** — the optimiser is constrained to ±5° / ±5 mm
       from the initial estimate so it cannot diverge when the landscape
       is non-smooth near mesh edges.

    Starting point is taken from cfg["initial_position_change"]["right"].
    Refined parameters are printed in copy-paste YAML format and returned.
    """
    from scipy.optimize import minimize, differential_evolution

    # Work on a copy and ensure face normals are consistently outward-pointing.
    mesh = mesh.copy()
    trimesh.repair.fix_normals(mesh, multibody=True)

    positions = np.array([led.position for led in leds], dtype=np.float64)
    normals   = np.array([led.normal   for led in leds], dtype=np.float64)
    normals  /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9

    cfg_r = cfg["initial_position_change"]["right"]
    x0 = np.array([
        cfg_r["rotation"]["rx"],
        cfg_r["rotation"]["ry"],
        cfg_r["rotation"]["rz"],
        cfg_r["translation"]["x"],
        cfg_r["translation"]["y"],
        cfg_r["translation"]["z"],
    ])

    def _build_R_t(params):
        rx, ry, rz, tx, ty, tz = params
        R_rz   = trimesh.transformations.euler_matrix(0, 0, rz)[:3, :3]
        R_base = trimesh.transformations.euler_matrix(rx, ry, 0)[:3, :3]
        R      = R_base.T @ R_rz
        t      = -R_base.T @ np.array([tx, ty, tz])
        return R, t

    # ------------------------------------------------------------------
    # Determine whether LED normals are outward or inward at the initial
    # solution, so the normal loss is oriented correctly regardless.
    # ------------------------------------------------------------------
    R0, t0 = _build_R_t(x0)
    pos0   = (R0 @ positions.T).T + t0
    nor0   = (R0 @ normals.T).T
    nor0  /= np.linalg.norm(nor0, axis=1, keepdims=True) + 1e-9
    _, _, fids0   = trimesh.proximity.closest_point(mesh, pos0)
    mean_cos_init = np.mean((nor0 * mesh.face_normals[fids0]).sum(axis=1))
    normal_sign   = 1.0 if mean_cos_init >= 0.0 else -1.0
    print(f"[alignment] LED normals are "
          f"{'outward' if normal_sign > 0 else 'INWARD'}-facing "
          f"(mean cos at initial pose = {mean_cos_init:.3f})")

    # ------------------------------------------------------------------
    # Diagnostic: how many LEDs are currently inside the mesh?
    # ------------------------------------------------------------------
    diff0        = pos0 - trimesh.proximity.closest_point(mesh, pos0)[0]
    signed_dot0  = (diff0 * mesh.face_normals[fids0]).sum(axis=1)
    n_inside_init = int((signed_dot0 < 0).sum())
    print(f"[alignment] LEDs inside mesh at start: "
          f"{n_inside_init} / {len(leds)}")

    # ------------------------------------------------------------------
    # Search bounds: ±5° / ±5 mm from the initial estimate.
    # ------------------------------------------------------------------
    delta_rot = np.deg2rad(5.0)   # 5 degrees
    delta_t   = 0.005             # 5 mm
    bounds = [
        (x0[0] - delta_rot, x0[0] + delta_rot),
        (x0[1] - delta_rot, x0[1] + delta_rot),
        (x0[2] - delta_rot, x0[2] + delta_rot),
        (x0[3] - delta_t,   x0[3] + delta_t),
        (x0[4] - delta_t,   x0[4] + delta_t),
        (x0[5] - delta_t,   x0[5] + delta_t),
    ]
    bounds_lo = np.array([b[0] for b in bounds])
    bounds_hi = np.array([b[1] for b in bounds])

    # ------------------------------------------------------------------
    # Loss function
    # A large out-of-bounds penalty keeps Nelder-Mead (which does not
    # accept bounds natively) inside the valid search region.
    # ------------------------------------------------------------------
    def _loss(params):
        # Hard wall for any method that ignores bounds (e.g. Nelder-Mead).
        if np.any(params < bounds_lo) or np.any(params > bounds_hi):
            return 1e12

        R, t = _build_R_t(params)
        pos_model = (R @ positions.T).T + t
        nor_model = (R @ normals.T).T
        nor_model /= np.linalg.norm(nor_model, axis=1, keepdims=True) + 1e-9

        closest, distances, face_ids = trimesh.proximity.closest_point(mesh, pos_model)
        mesh_normals = mesh.face_normals[face_ids]

        # --- signed distance ---
        # dot(pos - closest, face_normal):
        #   > 0  →  LED is on the OUTSIDE  (correct)
        #   < 0  →  LED is on the INSIDE   (penalise 20×)
        diff       = pos_model - closest
        signed_dot = (diff * mesh_normals).sum(axis=1)
        inside     = signed_dot < 0

        # Distances are in metres (≈1e-3); ×1e6 puts them in mm² range
        # so the term is O(1) and comparable to the normal term.
        dist_loss = np.mean(
            np.where(inside, 20.0 * distances ** 2, distances ** 2)
        ) * 1e6

        # --- oriented normal alignment (no abs) ---
        # LED normals should point the SAME way as mesh face normals (both outward).
        # 1 - cos = 0 when perfectly aligned, 2 when anti-parallel.
        cos_sim     = normal_sign * (nor_model * mesh_normals).sum(axis=1)
        normal_loss = np.mean(1.0 - cos_sim)

        return dist_loss + normal_loss

    # ------------------------------------------------------------------
    # Helper: run one full stage-cascade from a given starting point,
    # returning (best_x, best_f).
    #
    # Stage order (derivative-free methods are preferred because the
    # closest-point function is non-smooth at mesh face edges):
    #   1. L-BFGS-B      — fast gradient descent near smooth regions
    #   2. Powell        — direction-set, bounded, handles mild non-smoothness
    #   3. Nelder-Mead   — simplex, fully derivative-free, tightest tolerances
    #   4. L-BFGS-B      — final gradient polish from Nelder-Mead result
    # ------------------------------------------------------------------
    def _run_cascade(start, f_start):
        best_x, best_f = start.copy(), f_start

        # --- Stage 1: L-BFGS-B (bounded) ---
        r = minimize(
            _loss, best_x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100_000, "ftol": 1e-15, "gtol": 1e-12},
        )
        if r.fun < best_f:
            best_x, best_f = r.x.copy(), r.fun

        # --- Stage 2: Powell (bounded, scipy ≥ 1.7) ---
        r = minimize(
            _loss, best_x,
            method="Powell",
            bounds=bounds,
            options={"maxiter": 100_000, "xtol": 1e-12, "ftol": 1e-12},
        )
        if r.fun < best_f:
            best_x, best_f = r.x.copy(), r.fun

        # --- Stage 3: Nelder-Mead (adaptive simplex, very tight tolerances) ---
        r = minimize(
            _loss, best_x,
            method="Nelder-Mead",
            options={
                "maxiter": 500_000,
                "xatol": 1e-12,
                "fatol": 1e-12,
                "adaptive": True,   # rescales simplex for 6-D parameter space
            },
        )
        if r.fun < best_f:
            best_x, best_f = r.x.copy(), r.fun

        # --- Stage 4: final L-BFGS-B polish ---
        r = minimize(
            _loss, best_x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100_000, "ftol": 1e-15, "gtol": 1e-12},
        )
        if r.fun < best_f:
            best_x, best_f = r.x.copy(), r.fun

        return best_x, best_f

    # ------------------------------------------------------------------
    # Optimise
    # ------------------------------------------------------------------
    loss_before = _loss(x0)
    print(f"[alignment] Optimising …  initial loss = {loss_before:.6f}")

    # Primary run from the given initial estimate.
    best_x, best_f = _run_cascade(x0, loss_before)
    print(f"[alignment] Primary cascade → loss = {best_f:.10f}")

    # Multi-start: perturb the current best and re-run the cascade.
    # This helps escape local minima created by mesh face discontinuities.
    rng = np.random.default_rng(42)
    n_restarts = 2
    for i in range(n_restarts):
        # Gaussian perturbation (σ = 10 % of the allowed range)
        sigma = np.array([delta_rot, delta_rot, delta_rot,
                          delta_t,   delta_t,   delta_t]) * 0.10
        perturbed = best_x + rng.normal(0.0, sigma)
        perturbed = np.clip(perturbed, bounds_lo, bounds_hi)
        f_start   = _loss(perturbed)
        cx, cf    = _run_cascade(perturbed, f_start)
        if cf < best_f:
            best_x, best_f = cx, cf
            print(f"[alignment] Restart {i+1:2d}/{n_restarts} improved → "
                  f"loss = {best_f:.10f}")
        else:
            print(f"[alignment] Restart {i+1:2d}/{n_restarts} no improvement "
                  f"(loss = {cf:.10f})")

    if best_f >= loss_before:
        print("[alignment] WARNING: optimiser could not improve the initial estimate.")
        print("[alignment] Returning initial parameters — no change recommended.")
        rx, ry, rz, tx, ty, tz = x0
        loss_after = loss_before
    else:
        rx, ry, rz, tx, ty, tz = best_x
        loss_after = best_f

    # ------------------------------------------------------------------
    # Post-opt diagnostics
    # ------------------------------------------------------------------
    R1, t1 = _build_R_t(np.array([rx, ry, rz, tx, ty, tz]))
    pos1 = (R1 @ positions.T).T + t1
    closest1, _, fids1 = trimesh.proximity.closest_point(mesh, pos1)
    diff1       = pos1 - closest1
    signed_dot1 = (diff1 * mesh.face_normals[fids1]).sum(axis=1)
    n_inside_final = int((signed_dot1 < 0).sum())
    mean_dist_mm   = float(np.mean(np.abs(signed_dot1)) * 1000)

    print(f"[alignment] Done.  loss {loss_before:.6f} → {loss_after:.6f}")
    print(f"[alignment] LEDs inside mesh after opt: {n_inside_final} / {len(leds)}")
    print(f"[alignment] Mean signed distance to surface: {mean_dist_mm:.3f} mm")
    print()
    print("=" * 52)
    print("  Copy-paste into config.yml:")
    print("=" * 52)
    print("  initial_position_change:")
    print("    right:")
    print("      translation: # in meters")
    print(f"        x: {tx:.8f}")
    print(f"        y: {ty:.8f}")
    print(f"        z: {tz:.8f}")
    print("      rotation: # in radians")
    print(f"        rx: {rx:.8f}")
    print(f"        ry: {ry:.8f}")
    print(f"        rz: {rz:.8f}")
    print("=" * 52)
    print()

    return {
        "translation": {"x": float(tx), "y": float(ty), "z": float(tz)},
        "rotation":    {"rx": float(rx), "ry": float(ry), "rz": float(rz)},
    }


# =========================================================
# Convenience wrapper
# =========================================================

def prepare_model_geometry(leds, cfg):
    positions = np.array([led.position for led in leds], dtype=np.float64)
    normals   = np.array([led.normal   for led in leds], dtype=np.float64)

    T_model_ctrl = build_alignment_transform(cfg["initial_position_change"]["right"])

    positions_model = T_model_ctrl.apply(positions)
    normals_model   = (T_model_ctrl.R @ normals.T).T

    return positions_model, normals_model, T_model_ctrl
