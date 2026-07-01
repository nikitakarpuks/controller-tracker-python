import numpy as np
import trimesh
import cv2
import rerun as rr
import rerun.blueprint as rrb

from src.transformations import Transform
from src._visibility import _visible_mask, _cross_occluded_mask
from src.geometry import _compute_geometry


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
    "show_image_plane":  True,
    "show_camera_frame": False, # unit vector lines
    "show_led_ids":       True,   # LED index labels next to projected disks
    "show_blob_ids":      True,   # blob index labels next to blob contours
    "fps":              30,        # playback speed
    "frustum_z":        0.05,      # depth of the virtual projection screen (metres).
                                   # Must be less than the closest expected controller depth.
                                   # Pure display parameter — has no effect on matching.
    "frustum_edge_samples": 100,   # samples per image edge for the frustum outline polygon.
                                   # Higher = smoother curve, fewer blobs appearing outside.
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
    # Ghost (frozen last-known pose) appearance
    "ghost_led_color":       [110, 110, 130],  # dim blue-gray LEDs
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


# One distinct colour per controller index — used for blob contours.
# Warm / cool alternation makes the two sets easy to tell apart at a glance.
CONTROLLER_BLOB_COLORS = [
    [255, 160,  40],   # ctrl 0 — amber
    [ 40, 200, 255],   # ctrl 1 — sky blue
    [160, 255,  40],   # ctrl 2 — lime
    [255,  60, 200],   # ctrl 3 — magenta
]

def _controller_blob_color(ctrl_idx: int) -> list:
    return CONTROLLER_BLOB_COLORS[ctrl_idx % len(CONTROLLER_BLOB_COLORS)]


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

    SPECIAL HANDLING FOR KB4 FISHEYE CAMERAS:
    For KB4, undistorted points can blow up near image edges. We need to:
    1. Use more edge samples to capture the true boundary
    2. Filter out invalid (NaN/infinite) points
    3. Clip extreme values to prevent visualization breakage
    """
    w, h = float(cam.width), float(cam.height)

    # For KB4, use more samples to capture the distortion curve better
    if hasattr(cam, 'is_fisheye') and cam.is_fisheye:
        edge_samples = max(edge_samples, 50)  # More samples for fisheye

    t = np.linspace(0.0, 1.0, edge_samples, endpoint=False, dtype=np.float32)
    edges_px = np.concatenate([
        np.column_stack([t * w,            np.zeros_like(t)]),   # top
        np.column_stack([np.full_like(t, w), t * h]),             # right
        np.column_stack([(1.0 - t) * w,   np.full_like(t, h)]), # bottom
        np.column_stack([np.zeros_like(t), (1.0 - t) * h]),     # left
    ], axis=0)

    if getattr(cam, 'is_fisheye', False) and getattr(cam, 'rpmax', 0) > 0:
        # Only keep pixels within the valid fisheye radius so the outline stays convex.
        r_px = np.sqrt((edges_px[:, 0] - cam.cx)**2 + (edges_px[:, 1] - cam.cy)**2)
        edges_px = edges_px[r_px <= cam.rpmax]

    if len(edges_px) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    norm = cam.undistort_points(edges_px)
    pts = np.column_stack([norm[:, 0] * z, norm[:, 1] * z, np.full(len(norm), z)])
    return pts.astype(np.float32)



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
            cnt_n = cam.undistort_points(cnt)   # normalized (X/Z, Y/Z)
            pts = np.column_stack([
                cnt_n[:, 0] * frustum_z,
                cnt_n[:, 1] * frustum_z,
                np.full(n, frustum_z, dtype=np.float32),
            ]).astype(np.float32)
        else:
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
                 controllers_vis: dict,
                 vis_cfg: dict = None,
                 matching_cfg: dict = None):
        """
        controllers_vis: {ctrl_name: {"positions": np.ndarray,   # model-frame LED positions
                                      "normals":   np.ndarray,   # model-frame LED normals
                                      "T_model_ctrl": Transform, # controller→model transform
                                      "side": "right"|"left"}}   # informational only
        """
        self.vis_cfg         = vis_cfg if vis_cfg is not None else dict(VIS_CONFIG)
        self._matching_cfg   = matching_cfg or {}
        self._controllers_vis = controllers_vis
        self.visual_offset   = np.array([0.0, 0.0, 0.0])

        raw_mesh = load_trimesh(mesh_path)
        self._mesh_faces = raw_mesh.faces
        self._mesh_vertex_normals = (raw_mesh.vertex_normals
                                     if hasattr(raw_mesh, "vertex_normals") else None)
        # T_model_ctrl_left already embeds the X-reflection (Rx factor),
        # so model-space vertices are the same STP verts for both controllers.
        self._mesh_verts: dict = {}
        for ctrl_name in controllers_vis:
            self._mesh_verts[ctrl_name] = raw_mesh.vertices.astype(np.float32).copy()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Blueprint: per-camera tabs + global overview
    # ------------------------------------------------------------------

    def _build_blueprint(self, cameras: dict, frustum_z: float) -> rrb.Blueprint:
        """
        Build a tabbed blueprint with one global "World" view, one per-camera
        view, and a "Blob Debug" tab showing blob detection canvases per
        camera × controller combination.
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

        # ── Blob Debug tab ─────────────────────────────────────────────────
        ctrl_labels = [cn.replace("_controller", "") for cn in self._controllers_vis]
        image_views: list = []
        for ci in all_cam_indices:
            for cl in ctrl_labels:
                image_views.append(rrb.Spatial2DView(
                    name=f"cam{ci} / {cl}",
                    origin=f"blob_debug/cam{ci}/{cl}",
                ))
        views.append(rrb.Vertical(
            rrb.Grid(*image_views, grid_columns=len(ctrl_labels)),
            rrb.TextLogView(name="Detection modes", origin="blob_debug/modes"),
            name="Blob Debug",
            row_shares=[10, 1],
        ))

        return rrb.Blueprint(
            rrb.Tabs(*views),
            collapse_panels=False,
        )

    def start(self, poses_per_ctrl: dict, assignments_per_ctrl: dict,
              blobs_all, cameras: dict,
              contours_all=None,
              save_path: str = None,
              primary_cams_all: dict = None,
              aux_assignments_all: dict = None,
              frozen_poses_all: dict = None,
              blob_vis_all: list = None):
        """
        Log all frames to rerun.
        Opens the viewer automatically (spawn=True).

        Args:
            poses_per_ctrl:       {ctrl_name: [pose_or_None, ...]}
            assignments_per_ctrl: {ctrl_name: [assignment_or_None, ...]}
            cameras:              {cam_idx: Camera} for all selected cameras.
            primary_cams_all:     {ctrl_name: [cam_idx_or_None, ...]}
            aux_assignments_all:  {ctrl_name: [aux_asgn_or_None, ...]}
            save_path:            Optional path for an .rrd recording file.
            frozen_poses_all:     {ctrl_name: [T_world_ctrl_or_None, ...]}
                                  Non-None entries are used when the current
                                  pose is None but blobs were visible (case 3:
                                  between cameras).  None hides the controller.
        """
        self._cameras = cameras

        spawn_viewer = save_path is None
        rr.init("controller_animator", spawn=spawn_viewer)

        if save_path:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            rr.save(save_path)
            print(f"[rerun] Saving recording to: {save_path}  (no live viewer)", flush=True)
            print(f"[rerun] Replay later with:   rerun {save_path}", flush=True)

        blueprint = self._build_blueprint(cameras, frustum_z=self.vis_cfg.get("frustum_z", 0.05))
        rr.send_blueprint(blueprint)

        # Per-controller derived state (controller-frame positions/normals + geometry).
        self._ctrl_state: dict = {}
        for ctrl_name, cv in self._controllers_vis.items():
            T_model_ctrl = cv["T_model_ctrl"]
            T_ctrl_model = T_model_ctrl.inverse()
            pos_model = cv["positions"].astype(np.float64)
            nrm_model = cv["normals"].astype(np.float64)
            ctrl_pos  = T_ctrl_model.apply(pos_model).astype(np.float32)
            ctrl_nrm  = (T_ctrl_model.R @ nrm_model.T).T.astype(np.float32)
            self._ctrl_state[ctrl_name] = {
                "model_positions": cv["positions"].astype(np.float32),
                "model_normals":   cv["normals"].astype(np.float32),
                "ctrl_positions":  ctrl_pos,
                "ctrl_normals":    ctrl_nrm,
                "T_model_ctrl":    T_model_ctrl,
                "T_ctrl_model":    T_ctrl_model,
                "geom":            _compute_geometry(ctrl_pos, ctrl_nrm, cv.get("geometry_cfg")),
            }

        self._log_static_cameras(cameras)

        # Static mesh per controller.
        if self.vis_cfg.get("show_mesh", True):
            for ctrl_name in self._controllers_vis:
                rr.log(
                    f"world/{ctrl_name}/mesh",
                    rr.Mesh3D(
                        vertex_positions=self._mesh_verts[ctrl_name],
                        triangle_indices=self._mesh_faces,
                        vertex_normals=self._mesh_vertex_normals,
                        albedo_factor=[0.7, 0.7, 0.7, 1.0],
                    ),
                    static=True,
                )

        n_frames = len(next(iter(poses_per_ctrl.values())))

        for idx in range(n_frames):
            rr.set_time("frame", sequence=idx)

            if blob_vis_all is not None and idx < len(blob_vis_all):
                self._log_blob_debug(blob_vis_all[idx])

            # Build per-controller T_world_ctrl for this frame.
            T_world_ctrl_per_ctrl: dict = {}
            for ctrl_name, poses in poses_per_ctrl.items():
                pose = poses[idx] if idx < len(poses) else None
                if pose is None:
                    continue
                T_world_ctrl_per_ctrl[ctrl_name] = pose  # T_world_ctrl stored directly

            # Build ghost world-frame transforms for case-3 lost frames.
            ghost_T_world_model_per_ctrl: dict = {}
            for ctrl_name in self._controllers_vis:
                if ctrl_name in T_world_ctrl_per_ctrl:
                    continue
                frozen_list = (frozen_poses_all or {}).get(ctrl_name, [])
                T_world_ctrl = frozen_list[idx] if idx < len(frozen_list) else None
                if T_world_ctrl is not None:
                    T_ctrl_model = self._ctrl_state[ctrl_name]["T_ctrl_model"]
                    ghost_T_world_model_per_ctrl[ctrl_name] = T_world_ctrl.compose(T_ctrl_model)

            blobs    = blobs_all[idx]    if blobs_all    is not None and idx < len(blobs_all)    else None
            contours = contours_all[idx] if contours_all is not None and idx < len(contours_all) else None

            # Slice per-controller lists for this frame.
            assignments_frame = {
                n: (assignments_per_ctrl[n][idx]
                    if assignments_per_ctrl is not None and idx < len(assignments_per_ctrl[n])
                    else None)
                for n in self._controllers_vis
            }
            primary_cams_frame = {
                n: (primary_cams_all[n][idx]
                    if primary_cams_all is not None and idx < len(primary_cams_all[n])
                       and primary_cams_all[n][idx] is not None
                    else 0)
                for n in self._controllers_vis
            }
            aux_assignments_frame = {
                n: (aux_assignments_all[n][idx]
                    if aux_assignments_all is not None and idx < len(aux_assignments_all[n])
                    else None)
                for n in self._controllers_vis
            }

            self._log_frame(idx, T_world_ctrl_per_ctrl, assignments_frame,
                            blobs_per_ctrl=blobs, contours_per_ctrl=contours,
                            primary_cam_per_ctrl=primary_cams_frame,
                            aux_assignments_per_ctrl=aux_assignments_frame,
                            ghost_T_world_model_per_ctrl=ghost_T_world_model_per_ctrl)

        msg = f"[rerun] Logged {n_frames} frames."
        if save_path:
            msg += f" Recording saved to: {save_path}"
            msg += f"\n[rerun] Replay with:  rerun {save_path}"
        print(msg)

    # ------------------------------------------------------------------
    # Blob debug images (logged per frame under blob_debug/ subtree)
    # ------------------------------------------------------------------

    def _log_blob_debug(self, frame_vis: dict) -> None:
        """Log blob detection canvases and a mode summary for the current frame.

        frame_vis: {ctrl_name: {cam_idx: {mode_key: canvas_bgr}}}
        mode_key is "local", "pass1", or "pass2".

        For 2-pass frames (pass1 + pass2 present) the two canvases are placed
        side-by-side in a single image so both are visible in one panel.
        The composite image is logged directly at the panel origin path so that
        Rerun updates it reliably when scrubbing the timeline.
        """
        mode_lines = []
        for ctrl_name, cam_dict in frame_vis.items():
            ctrl_label = ctrl_name.replace("_controller", "")
            for cam_idx, canvases in cam_dict.items():
                if not canvases:
                    continue

                if "pass1" in canvases and "pass2" in canvases:
                    p1 = canvases["pass1"]
                    p2 = canvases["pass2"]
                    if p1.shape[0] != p2.shape[0]:
                        # Pad shorter canvas to match height
                        h = max(p1.shape[0], p2.shape[0])
                        def _pad(img, h):
                            if img.shape[0] < h:
                                pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
                                return np.vstack([img, pad])
                            return img
                        p1, p2 = _pad(p1, h), _pad(p2, h)
                    # Thin separator between the two passes
                    sep = np.full((p1.shape[0], 3, 3), 80, dtype=np.uint8)
                    composite = np.hstack([p1, sep, p2])
                    mode_str = "pass1|pass2"
                elif "local" in canvases:
                    composite = canvases["local"]
                    mode_str = "local"
                else:
                    # pass1 only (cold start, pass2 skipped)
                    composite = next(iter(canvases.values()))
                    mode_str = next(iter(canvases))

                rr.log(
                    f"blob_debug/cam{cam_idx}/{ctrl_label}",
                    rr.Image(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)),
                )
                mode_lines.append(f"cam{cam_idx}/{ctrl_label}: {mode_str}")

        if mode_lines:
            rr.log("blob_debug/modes", rr.TextLog("  |  ".join(mode_lines)))

    # ------------------------------------------------------------------
    # Static geometry (logged once, no timeline)
    # ------------------------------------------------------------------

    def _log_static_cameras(self, cameras: dict):
        """Frustum image planes and camera markers — logged once."""
        frustum_z    = self.vis_cfg.get("frustum_z", 0.05)
        edge_samples = self.vis_cfg.get("frustum_edge_samples", 100)
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
                          blobs_per_ctrl: dict, contours_per_ctrl: dict,
                          blob_z: float,
                          matched_bids_per_ctrl_cam: dict = None):
        """Log per-controller blob contours with distinct colours per controller.

        Each controller's blobs are rendered in their own colour layer so the
        user can immediately see which blobs each controller detected and which
        are matched (bright) vs unmatched (dim).
        """
        T_world_cam = cam.T_world_cam

        def _wc(pts):
            if T_world_cam is None:
                return pts
            return T_world_cam.apply(np.asarray(pts, dtype=np.float64))

        path      = f"world/camera_{cam_idx}"
        ctrl_list = list(self._controllers_vis.keys())

        for ctrl_idx, ctrl_name in enumerate(ctrl_list):
            ctrl_path    = f"{path}/ctrl_{ctrl_idx}"
            color        = _controller_blob_color(ctrl_idx)
            color_dim    = [int(c * 0.45) for c in color]
            # Tiny z-stagger per controller avoids z-fighting between layers.
            ctrl_z       = blob_z - ctrl_idx * 0.00005

            blobs        = (blobs_per_ctrl    or {}).get(ctrl_name, {}).get(cam_idx)
            contours     = (contours_per_ctrl or {}).get(ctrl_name, {}).get(cam_idx)
            matched_bids = ((matched_bids_per_ctrl_cam or {}).get(ctrl_name) or {}).get(cam_idx, set())

            if blobs is not None and len(blobs) > 0 and self.vis_cfg.get("show_blobs", True) and contours:
                blob_colors = np.array(
                    [color if i in matched_bids else color_dim
                     for i in range(len(contours))],
                    dtype=np.uint8,
                )
                bv, bf, bc = make_contour_mesh_3d(contours, cam, ctrl_z, blob_colors, undistort=True)
                if bv is not None:
                    rr.log(f"{ctrl_path}/blobs",
                           rr.Mesh3D(vertex_positions=_wc(bv), triangle_indices=bf, vertex_colors=bc))
                else:
                    rr.log(f"{ctrl_path}/blobs", rr.Clear(recursive=False))
            else:
                rr.log(f"{ctrl_path}/blobs", rr.Clear(recursive=False))

            if blobs is not None and len(blobs) > 0 and self.vis_cfg.get("show_blob_ids", True):
                _norm = cam.undistort_points(blobs)   # normalized (X/Z, Y/Z)
                label_pts = np.column_stack([
                    _norm[:, 0] * ctrl_z,
                    _norm[:, 1] * ctrl_z,
                    np.full(len(_norm), ctrl_z, dtype=np.float32),
                ]).copy()
                bdx, bdy = self.vis_cfg.get("blob_id_label_offset", [-0.0013, 0.0008])
                label_pts[:, 0] += bdx
                label_pts[:, 1] += bdy
                rr.log(f"{ctrl_path}/blob_ids", rr.Points3D(
                    positions=_wc(label_pts),
                    labels=[str(i) for i in range(len(blobs))],
                    colors=[color] * len(blobs),
                    radii=0.0,
                ))
            else:
                rr.log(f"{ctrl_path}/blob_ids", rr.Clear(recursive=False))

    # ------------------------------------------------------------------
    # Per-frame logging
    # ------------------------------------------------------------------

    def _log_frame(self, idx: int, T_world_ctrl_per_ctrl: dict,
                   assignments_per_ctrl: dict, blobs_per_ctrl: dict,
                   contours_per_ctrl: dict = None,
                   primary_cam_per_ctrl: dict = None,
                   aux_assignments_per_ctrl: dict = None,
                   ghost_T_world_model_per_ctrl: dict = None):

        ray_radius            = self.vis_cfg.get("ray_radius",            0.0002)
        led_disk_radius       = self.vis_cfg.get("led_disk_radius",       0.003)
        proj_disk_radius      = self.vis_cfg.get("proj_disk_radius",      0.0008)
        blob_z_offset         = self.vis_cfg.get("blob_z_offset",         0.002)
        matched_proj_z_offset = self.vis_cfg.get("matched_proj_z_offset", 0.001)
        error_z_offset        = self.vis_cfg.get("error_z_offset",        0.0015)
        error_radius          = self.vis_cfg.get("error_radius",          0.0001)
        frustum_z             = self._frustum_z
        blob_z                = frustum_z - blob_z_offset
        matched_proj_z        = frustum_z - matched_proj_z_offset
        error_z               = frustum_z - error_z_offset
        offset                = self.visual_offset

        # ── matched_bids per controller per camera ───────────────────────────
        matched_bids_per_ctrl_cam: dict = {}
        for ctrl_name in self._controllers_vis:
            pri = (primary_cam_per_ctrl or {}).get(ctrl_name, next(iter(self._cameras)))
            if pri not in self._cameras:
                pri = next(iter(self._cameras))
            ctrl_bids: dict = {pri: set()}
            for bid, _ in ((assignments_per_ctrl or {}).get(ctrl_name) or []):
                ctrl_bids[pri].add(bid)
            for aux_ci, pairs in ((aux_assignments_per_ctrl or {}).get(ctrl_name) or {}).items():
                if aux_ci != pri:
                    ctrl_bids[aux_ci] = {bid for bid, _ in pairs}
            matched_bids_per_ctrl_cam[ctrl_name] = ctrl_bids

        # ── Blobs per controller per camera ─────────────────────────────────
        for cam_idx, cam in self._cameras.items():
            self._log_camera_blobs(
                cam_idx, cam,
                blobs_per_ctrl,
                contours_per_ctrl,
                blob_z,
                matched_bids_per_ctrl_cam=matched_bids_per_ctrl_cam,
            )

        # ── Hide or ghost controllers with no valid pose this frame ─────────
        ghost_led_color = self.vis_cfg.get("ghost_led_color", [110, 110, 130])
        for ctrl_name in self._controllers_vis:
            if ctrl_name in T_world_ctrl_per_ctrl:
                continue
            ghost_T_world_model = (ghost_T_world_model_per_ctrl or {}).get(ctrl_name)
            if ghost_T_world_model is not None:
                # Case 3: blobs present but pose unsolvable → hold last known pose.
                # Show mesh + dim LEDs at frozen position; no rays/projections.
                cs        = self._ctrl_state[ctrl_name]
                ctrl_path = f"world/{ctrl_name}"
                R_disp = ghost_T_world_model.R
                t_disp = ghost_T_world_model.t
                pts_disp     = (R_disp @ cs["model_positions"].T).T + t_disp + offset
                normals_disp = (R_disp @ cs["model_normals"].T).T

                if self.vis_cfg.get("show_mesh", True):
                    T4 = np.eye(4)
                    T4[:3, :3] = R_disp
                    T4[:3,  3] = t_disp + offset
                    rr.log(f"{ctrl_path}/mesh",
                           rr.Transform3D(translation=T4[:3, 3], mat3x3=T4[:3, :3]))

                if self.vis_cfg.get("show_leds", True):
                    colors = np.tile(ghost_led_color, (len(pts_disp), 1)).astype(np.uint8)
                    verts, faces, vcols = make_disk_mesh(pts_disp, normals_disp, colors,
                                                         radius=led_disk_radius)
                    rr.log(f"{ctrl_path}/leds",
                           rr.Mesh3D(vertex_positions=verts, triangle_indices=faces,
                                     vertex_colors=vcols))

                rr.log(f"{ctrl_path}/normals", rr.Clear(recursive=False))
                for _ci in self._cameras:
                    for _sub in ("rays", "projected_leds", "led_ids", "errors", "error_values"):
                        rr.log(f"{ctrl_path}/camera_{_ci}/{_sub}", rr.Clear(recursive=False))
            else:
                # Cases 1/2: never tracked or out of all cameras → collapse to invisible.
                rr.log(f"world/{ctrl_name}/mesh",
                       rr.Transform3D(mat3x3=np.eye(3, dtype=np.float32) * 1e-9))
                for _sub in ("leds", "normals"):
                    rr.log(f"world/{ctrl_name}/{_sub}", rr.Clear(recursive=False))
                for _ci in self._cameras:
                    for _sub in ("rays", "projected_leds", "led_ids", "errors", "error_values"):
                        rr.log(f"world/{ctrl_name}/camera_{_ci}/{_sub}", rr.Clear(recursive=False))

        # ── Per-controller loop ─────────────────────────────────────────────
        for ctrl_name, T_world_ctrl in T_world_ctrl_per_ctrl.items():
            cs          = self._ctrl_state[ctrl_name]
            T_ctrl_model = cs["T_ctrl_model"]
            model_positions = cs["model_positions"]
            model_normals   = cs["model_normals"]
            ctrl_path   = f"world/{ctrl_name}"

            primary_cam_idx = (primary_cam_per_ctrl or {}).get(ctrl_name, 0)
            if primary_cam_idx not in self._cameras:
                primary_cam_idx = next(iter(self._cameras))
            cam_path  = f"{ctrl_path}/camera_{primary_cam_idx}"

            camera = self._cameras[primary_cam_idx]

            # Primary camera frame pose (for projection and visibility)
            T_cam_ctrl  = camera.T_world_cam.inverse().compose(T_world_ctrl)
            T_cam_model = T_cam_ctrl.compose(T_ctrl_model)
            R_cam = T_cam_model.R
            t_cam = T_cam_model.t

            # World frame model pose (for 3D display)
            T_world_model = T_world_ctrl.compose(T_ctrl_model)
            R_disp = T_world_model.R
            t_disp = T_world_model.t

            def _w(pts, _cam=camera):
                pts_arr = np.asarray(pts, dtype=np.float64)
                if pts_arr.ndim == 1:
                    return _cam.T_world_cam.apply(pts_arr.reshape(1, 3))[0]
                return _cam.T_world_cam.apply(pts_arr)

            # LED positions in camera frame and display frame.
            pts_cam_real  = (R_cam  @ model_positions.T).T + t_cam
            pts_disp_real = (R_disp @ model_positions.T).T + t_disp
            pts_disp      = pts_disp_real + offset

            # ── Mesh transform (geometry is static, just update the pose) ──
            T4 = np.eye(4)
            T4[:3, :3] = R_disp
            T4[:3,  3] = t_disp + offset
            if self.vis_cfg.get("show_mesh", True):
                rr.log(f"{ctrl_path}/mesh",
                       rr.Transform3D(translation=T4[:3, 3], mat3x3=T4[:3, :3]))

            normals_cam  = (R_cam  @ model_normals.T).T
            normals_disp = (R_disp @ model_normals.T).T

            # ── Visibility mask ─────────────────────────────────────────────
            vis_scores = _visible_mask(
                T_cam_ctrl.R, T_cam_ctrl.t,
                cs["ctrl_positions"], cs["ctrl_normals"],
                cs["geom"],
                cam_K=camera.camera_matrix, cam_dc=camera.dist_coeffs,
                cam_w=camera.width, cam_h=camera.height,
                cam_rpmax=camera.rpmax,
                cam_is_fisheye=camera.is_fisheye,
                facing_threshold_deg=float(self._matching_cfg.get('led_facing_angle_deg', 86.0)),
            )
            if bool(self._matching_cfg.get('cross_controller_occlusion', False)):
                _br          = float(self._matching_cfg.get('cross_occlusion_bounding_radius_m', 0.18))
                _gate_margin = float(self._matching_cfg.get('cross_occlusion_gate_margin_px', 20.0))
                _focal_px    = float(max(camera.camera_matrix[0, 0], camera.camera_matrix[1, 1]))
                for _occ_name, _T_world_ctrl_occ in T_world_ctrl_per_ctrl.items():
                    if _occ_name == ctrl_name:
                        continue
                    _occ_cs = self._ctrl_state[_occ_name]
                    _T_cam_ctrl_occ = camera.T_world_cam.inverse().compose(_T_world_ctrl_occ)
                    _cross_occ_vis = _cross_occluded_mask(
                        T_cam_ctrl.R.astype(np.float32), T_cam_ctrl.t.astype(np.float32),
                        cs["ctrl_positions"],
                        _T_cam_ctrl_occ.R.astype(np.float32), _T_cam_ctrl_occ.t.astype(np.float32),
                        _occ_cs["geom"],
                        _br, _br, _focal_px, _gate_margin,
                    )
                    vis_scores[_cross_occ_vis] = 0.0
            vis_set = set(np.where(vis_scores >= 1.0)[0].tolist())

            # ── LED disks ───────────────────────────────────────────────────
            if self.vis_cfg.get("show_leds", True):
                colors = np.tile([255, 192, 203], (len(pts_disp), 1))
                verts, faces, vcols = make_disk_mesh(pts_disp, normals_disp, colors,
                                                     radius=led_disk_radius)
                rr.log(f"{ctrl_path}/leds",
                       rr.Mesh3D(vertex_positions=verts, triangle_indices=faces, vertex_colors=vcols))

            # ── Normals ─────────────────────────────────────────────────────
            if self.vis_cfg.get("show_normals", True):
                rr.log(f"{ctrl_path}/normals",
                       rr.LineStrips3D(
                           strips=[[s, e] for s, e in zip(pts_disp, pts_disp + normals_disp * 0.03)],
                           colors=[255, 0, 255], radii=ray_radius))

            assignment    = (assignments_per_ctrl or {}).get(ctrl_name)
            aux_assignments = (aux_assignments_per_ctrl or {}).get(ctrl_name)

            matched_bids: set = set()
            matched_lids: set = set()
            if assignment:
                for bid, lid in assignment:
                    matched_bids.add(bid)
                    matched_lids.add(lid)

            # ── blob positions on the error plane (used by error lines below) ─
            blobs = (blobs_per_ctrl or {}).get(ctrl_name, {}).get(primary_cam_idx)
            if blobs is not None and len(blobs) > 0:
                blobs_norm = camera.undistort_points(blobs)   # (N,2) normalized (X/Z, Y/Z)
                pts_plane_cam = np.column_stack([
                    blobs_norm[:, 0] * error_z,
                    blobs_norm[:, 1] * error_z,
                    np.full(len(blobs_norm), error_z, dtype=np.float32),
                ])
                pts_plane_disp = _w(pts_plane_cam)
            else:
                pts_plane_disp = None

            # ── Clear stale per-camera assets for non-primary cameras ───────
            for _ci in self._cameras:
                if _ci != primary_cam_idx:
                    for _sub in ("rays", "projected_leds", "led_ids", "errors", "error_values"):
                        rr.log(f"{ctrl_path}/camera_{_ci}/{_sub}", rr.Clear(recursive=False))

            # ── Projected LEDs (matched + visible unmatched) ────────────────
            all_proj_pts   = []
            all_proj_lids  = []
            lid_to_proj_pt = {}
            for i, (px, py, pz) in enumerate(pts_cam_real):
                if pz >= frustum_z and (i in matched_lids or i in vis_set):
                    z_val        = matched_proj_z if i in matched_lids else frustum_z
                    proj_pt_cam  = np.array([px / pz * z_val, py / pz * z_val, z_val])
                    proj_pt_disp = _w(proj_pt_cam)
                    all_proj_pts.append(proj_pt_disp)
                    all_proj_lids.append(i)
                    lid_to_proj_pt[i] = proj_pt_disp

            if self.vis_cfg.get("show_projected", True):
                if all_proj_pts:
                    proj_colors = np.array(
                        [[70, 130, 255] if lid in matched_lids else [230, 80, 50]
                         for lid in all_proj_lids], dtype=np.uint8)
                    proj_normal_cam  = np.array([0.0, 0.0, -1.0])
                    proj_normal_disp = camera.T_world_cam.R @ proj_normal_cam
                    proj_normals = np.tile(proj_normal_disp, (len(all_proj_pts), 1)).astype(np.float32)
                    pv, pf, pc = make_disk_mesh(np.array(all_proj_pts, dtype=np.float32),
                                                proj_normals, proj_colors,
                                                radius=proj_disk_radius, n_segments=16, surface_offset=0.0)
                    rr.log(f"{cam_path}/projected_leds",
                           rr.Mesh3D(vertex_positions=pv, triangle_indices=pf, vertex_colors=pc))
                else:
                    rr.log(f"{cam_path}/projected_leds", rr.Clear(recursive=False))

            if self.vis_cfg.get("show_led_ids", True):
                if all_proj_pts:
                    label_color  = self.vis_cfg.get("led_id_label_color",  [180, 0, 255])
                    dx, dy       = self.vis_cfg.get("led_id_label_offset", [0.0006, -0.0006])
                    label_pts_cam = np.array([
                        [pts_cam_real[lid, 0] / pts_cam_real[lid, 2] * frustum_z + dx,
                         pts_cam_real[lid, 1] / pts_cam_real[lid, 2] * frustum_z + dy,
                         frustum_z]
                        for lid in all_proj_lids], dtype=np.float32)
                    rr.log(f"{cam_path}/led_ids", rr.Points3D(
                        positions=_w(label_pts_cam),
                        labels=[str(lid) for lid in all_proj_lids],
                        colors=[label_color] * len(all_proj_lids), radii=0.0))
                else:
                    rr.log(f"{cam_path}/led_ids", rr.Clear(recursive=False))

            # ── Rays + errors (primary camera) ──────────────────────────────
            ray_strips      = []
            ray_colors_list = []
            error_strips    = []
            error_values    = []
            error_label_pts = []

            # T_cam_ctrl has det=+1 (proper rotation); T_cam_model.R may have det=-1
            # for the left controller (T_model_ctrl embeds an Rx reflection), so
            # cv2.Rodrigues(R_cam) would give garbage.  Use ctrl_positions + T_cam_ctrl.
            rvec_ctrl, _ = cv2.Rodrigues(T_cam_ctrl.R.astype(np.float32))
            tvec_ctrl    = T_cam_ctrl.t.astype(np.float32)
            ctrl_positions = cs["ctrl_positions"]
            if assignment and pts_plane_disp is not None:
                matched_lids_ord = [lid for _, lid in assignment]
                proj_matched, _ = camera.project_points(
                    ctrl_positions[matched_lids_ord], rvec_ctrl, tvec_ctrl,
                )
                proj_matched = proj_matched.reshape(-1, 2)

                for pair_i, (bid, lid) in enumerate(assignment):
                    if bid >= len(pts_plane_disp) or lid not in lid_to_proj_pt:
                        continue
                    if self.vis_cfg.get("show_rays", True):
                        ray_strips.append([pts_disp_real[lid].tolist(), lid_to_proj_pt[lid].tolist()])
                        ray_colors_list.append([70, 130, 255])
                    px, py, pz = pts_cam_real[lid]
                    if pz > 1e-6:
                        proj_flat_cam  = np.array([px / pz * error_z, py / pz * error_z, error_z])
                        proj_flat_disp = _w(proj_flat_cam)
                        if self.vis_cfg.get("show_errors", True):
                            error_strips.append([proj_flat_disp.tolist(), pts_plane_disp[bid].tolist()])
                        u, v   = blobs[bid]
                        pu, pv = proj_matched[pair_i]
                        error_values.append(float(np.sqrt((u - pu)**2 + (v - pv)**2)))
                        error_label_pts.append((proj_flat_disp + pts_plane_disp[bid]) / 2)

            if self.vis_cfg.get("show_rays", True):
                for lid, proj_pt in lid_to_proj_pt.items():
                    if lid not in matched_lids:
                        ray_strips.append([pts_disp_real[lid].tolist(), proj_pt.tolist()])
                        ray_colors_list.append([230, 80, 50])

            rr.log(f"{cam_path}/rays",
                   rr.LineStrips3D(strips=ray_strips,
                                   colors=ray_colors_list if ray_strips else [[0, 0, 0]],
                                   radii=ray_radius)
                   if ray_strips else rr.Clear(recursive=False))

            rr.log(f"{cam_path}/errors",
                   rr.LineStrips3D(strips=error_strips, colors=[255, 0, 0], radii=error_radius)
                   if error_strips else rr.Clear(recursive=False))

            if self.vis_cfg.get("show_error_values", True):
                if error_label_pts:
                    ev_color  = self.vis_cfg.get("error_value_label_color",  [255, 0, 0])
                    ev_offset = self.vis_cfg.get("error_value_label_offset", [0.0006, 0.0006])
                    ev_pts    = np.array(error_label_pts, dtype=np.float32).copy()
                    ev_pts[:, 0] += ev_offset[0]
                    ev_pts[:, 1] += ev_offset[1]
                    rr.log(f"{cam_path}/error_values", rr.Points3D(
                        positions=ev_pts,
                        labels=[f"{e:.2f}" for e in error_values],
                        colors=[ev_color] * len(ev_pts), radii=0.0))
                else:
                    rr.log(f"{cam_path}/error_values", rr.Clear(recursive=False))

            # ── Aux cameras overlay ─────────────────────────────────────────
            for _aux_ci, _aux_pairs in (aux_assignments or {}).items():
                if _aux_ci == primary_cam_idx:
                    continue  # primary already rendered above
                _acp     = f"{ctrl_path}/camera_{_aux_ci}"
                _aux_cam = self._cameras.get(_aux_ci)
                if _aux_cam is None or not _aux_pairs:
                    for _sub in ("rays", "projected_leds", "led_ids", "errors", "error_values"):
                        rr.log(f"{_acp}/{_sub}", rr.Clear(recursive=False))
                    continue

                # Uniform formula for any camera: pose in aux camera frame
                _T_aux_ctrl  = _aux_cam.T_world_cam.inverse().compose(T_world_ctrl)
                _T_aux_model = _T_aux_ctrl.compose(T_ctrl_model)
                _pts_aux_cam = _T_aux_model.apply(model_positions)
                _aux_color   = _camera_color(_aux_ci)
                _aux_blobs_raw = (blobs_per_ctrl or {}).get(ctrl_name, {}).get(_aux_ci)
                _aux_blobs_arr = (np.asarray(_aux_blobs_raw, dtype=np.float32)
                                  if _aux_blobs_raw is not None and len(_aux_blobs_raw) > 0 else None)

                _aux_matched_lids = {_lid for _, _lid in _aux_pairs}

                _aux_proj_pts    = []
                _aux_proj_lids   = []
                _aux_lid_to_proj = {}
                for _i, (_ipx, _ipy, _ipz) in enumerate(_pts_aux_cam):
                    if _ipz >= frustum_z and _i in _aux_matched_lids:
                        _pp = np.array([_ipx / _ipz * matched_proj_z,
                                        _ipy / _ipz * matched_proj_z,
                                        matched_proj_z])
                        _pw = _aux_cam.T_world_cam.apply(_pp.reshape(1, 3))[0]
                        _aux_proj_pts.append(_pw)
                        _aux_proj_lids.append(_i)
                        _aux_lid_to_proj[_i] = _pw

                if self.vis_cfg.get("show_projected", True):
                    if _aux_proj_pts:
                        _pn_world = _aux_cam.T_world_cam.R @ np.array([0.0, 0.0, -1.0])
                        _pnormals = np.tile(_pn_world, (len(_aux_proj_pts), 1)).astype(np.float32)
                        _apv, _apf, _apc = make_disk_mesh(
                            np.array(_aux_proj_pts, dtype=np.float32), _pnormals,
                            np.array([_aux_color] * len(_aux_proj_pts), dtype=np.uint8),
                            radius=proj_disk_radius, n_segments=16, surface_offset=0.0)
                        rr.log(f"{_acp}/projected_leds",
                               rr.Mesh3D(vertex_positions=_apv, triangle_indices=_apf, vertex_colors=_apc))
                    else:
                        rr.log(f"{_acp}/projected_leds", rr.Clear(recursive=False))

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
                        _lbl_pts_world = _aux_cam.T_world_cam.apply(_lbl_pts_cam)
                        rr.log(f"{_acp}/led_ids", rr.Points3D(
                            positions=_lbl_pts_world,
                            labels=[str(_lid) for _lid in _aux_proj_lids],
                            colors=[_lbl_color] * len(_aux_proj_lids),
                            radii=0.0,
                        ))
                    else:
                        rr.log(f"{_acp}/led_ids", rr.Clear(recursive=False))

                # ── aux rays, errors, error values ──────────────────────────
                _aux_ray_strips  = []
                _aux_err_strips  = []
                _aux_err_vals    = []
                _aux_err_lbl_pts = []
                _aux_error_z     = frustum_z - error_z_offset

                if _aux_blobs_arr is not None:
                    _rv_aux, _ = cv2.Rodrigues(_T_aux_ctrl.R.astype(np.float32))
                    _tv_aux    = _T_aux_ctrl.t.astype(np.float32).reshape(3, 1)
                    _aux_lids_ord = [_lid for _, _lid in _aux_pairs]
                    _proj_px_aux, _ = _aux_cam.project_points(
                        ctrl_positions[_aux_lids_ord], _rv_aux, _tv_aux,
                    )
                    _proj_px_aux = _proj_px_aux.reshape(-1, 2)

                    for _pi, (_blob_j, _lid) in enumerate(_aux_pairs):
                        if _blob_j >= len(_aux_blobs_arr) or _lid not in _aux_lid_to_proj:
                            continue
                        if self.vis_cfg.get("show_rays", True):
                            _aux_ray_strips.append(
                                [pts_disp_real[_lid].tolist(), _aux_lid_to_proj[_lid].tolist()])
                        _lp = _pts_aux_cam[_lid]
                        if _lp[2] > 1e-6 and self.vis_cfg.get("show_errors", True):
                            _bu, _bv = float(_aux_blobs_arr[_blob_j, 0]), float(_aux_blobs_arr[_blob_j, 1])
                            _b_norm = _aux_cam.undistort_points(
                                np.array([[_bu, _bv]], dtype=np.float32),
                            ).reshape(2)
                            _blob_plane_cam = np.array([_b_norm[0] * _aux_error_z,
                                                        _b_norm[1] * _aux_error_z, _aux_error_z])
                            _blob_plane_w = _aux_cam.T_world_cam.apply(_blob_plane_cam.reshape(1, 3))[0]
                            _proj_flat_cam = np.array([_lp[0] / _lp[2] * _aux_error_z,
                                                       _lp[1] / _lp[2] * _aux_error_z, _aux_error_z])
                            _proj_flat_w = _aux_cam.T_world_cam.apply(_proj_flat_cam.reshape(1, 3))[0]
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
                            positions=_ev_pts, labels=[f"{_e:.2f}" for _e in _aux_err_vals],
                            colors=[_ev_color] * len(_ev_pts), radii=0.0))
                    else:
                        rr.log(f"{_acp}/error_values", rr.Clear(recursive=False))
                # end aux cameras loop
            # end per-controller loop


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

    Starting point is taken from cfg["mesh_alignment"].
    Refined parameters are printed in copy-paste YAML format and returned.
    """
    from scipy.optimize import minimize, differential_evolution

    # Work on a copy and ensure face normals are consistently outward-pointing.
    mesh = mesh.copy()
    trimesh.repair.fix_normals(mesh, multibody=True)

    positions = np.array([led.position for led in leds], dtype=np.float64)
    normals   = np.array([led.normal   for led in leds], dtype=np.float64)
    normals  /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9

    cfg_r = cfg["mesh_alignment"]
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
    print("  mesh_alignment:")
    print("    translation: # in meters")
    print(f"      x: {tx:.8f}")
    print(f"      y: {ty:.8f}")
    print(f"      z: {tz:.8f}")
    print("    rotation: # in radians")
    print(f"      rx: {rx:.8f}")
    print(f"      ry: {ry:.8f}")
    print(f"      rz: {rz:.8f}")
    print("=" * 52)
    print()

    return {
        "translation": {"x": float(tx), "y": float(ty), "z": float(tz)},
        "rotation":    {"rx": float(rx), "ry": float(ry), "rz": float(rz)},
    }


# =========================================================
# Convenience wrapper
# =========================================================

def prepare_model_geometry(leds, cfg, side="right"):
    """
    cfg must be the right_controller config (contains mesh_alignment).
    side="left": T_model_ctrl_left = T_model_ctrl_right @ Rx, which correctly maps
    left-controller-space LED positions into the (X-mirrored) model space.
    """
    positions = np.array([led.position for led in leds], dtype=np.float64)
    normals   = np.array([led.normal   for led in leds], dtype=np.float64)

    T_right = build_alignment_transform(cfg["mesh_alignment"])

    if side == "left":
        Rx = np.diag([-1., 1., 1.])
        T_model_ctrl = Transform(T_right.R @ Rx, T_right.t)
    else:
        T_model_ctrl = T_right

    positions_model = T_model_ctrl.apply(positions)
    normals_model   = (T_model_ctrl.R @ normals.T).T

    return positions_model, normals_model, T_model_ctrl
