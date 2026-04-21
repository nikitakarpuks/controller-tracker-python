import numpy as np
import trimesh
import cv2
import rerun as rr
import rerun.blueprint as rrb

from src.transformations import Transform
from src._matching import _visible_mask, _compute_frustum_geometry


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
    "show_frustum":     True,
    "show_image_plane": True,
    "show_camera_frame": False, # unit vector lines
    "show_led_ids":     True,   # LED index labels next to projected disks
    "show_blob_ids":    True,   # blob index labels next to blob contours
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
                 vis_cfg: dict = None):

        self.mesh_path       = mesh_path
        self.model_positions = model_positions.astype(np.float32)
        self.model_normals   = model_normals.astype(np.float32)
        self.vis_cfg         = vis_cfg if vis_cfg is not None else dict(VIS_CONFIG)

        self._trimesh        = load_trimesh(mesh_path)
        self.visual_offset   = np.array([0.0, 0.0, 0.0])

        # Precompute torus geometry for per-frame visibility testing
        (self._geo_ring_axis, self._geo_is_inner, self._geo_radial_out,
         self._geo_ring_centroid, self._geo_R_frustum_center,
         self._geo_frustum_slope, self._geo_z_frustum_top,
         self._geo_z_frustum_bot, _
         ) = _compute_frustum_geometry(self.model_positions, self.model_normals)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def start(self, poses, assignments, blobs_all, camera, T_model_ctrl,
              contours_all=None, save_path: str = None):
        """
        Log all frames to rerun.
        Opens the viewer automatically (spawn=True).

        Args:
            save_path: Optional path for an .rrd recording file.
                       When set, the session is saved to disk so it can be
                       replayed later with:  rerun <save_path>
        """
        # When a file is being saved the viewer is not spawned — replay manually.
        spawn_viewer = save_path is None
        rr.init("controller_animator", spawn=spawn_viewer)

        if save_path:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            rr.save(save_path)
            print(f"[rerun] Saving recording to: {save_path}  (no live viewer)")
            print(f"[rerun] Replay later with:   rerun {save_path}")

        # Set up a blueprint so the viewer looks reasonable on first open
        blueprint = rrb.Blueprint(
            rrb.Spatial3DView(name="3-D Scene", origin="/world"),
            collapse_panels=False,
        )
        rr.send_blueprint(blueprint)

        self._log_static_camera(camera)

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

            self._log_frame(idx, T_cam_model, assignment, blobs, camera, contours)

        msg = f"[rerun] Logged {n_frames} frames."
        if save_path:
            msg += f" Recording saved to: {save_path}"
            msg += f"\n[rerun] Replay with:  rerun {save_path}"
        print(msg)

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
                   assignment, blobs, camera, contours=None):

        offset           = self.visual_offset
        R                = T_cam_model.R
        t                = T_cam_model.t
        ray_radius            = self.vis_cfg.get("ray_radius",            0.0002)
        led_disk_radius       = self.vis_cfg.get("led_disk_radius",       0.003)
        proj_disk_radius      = self.vis_cfg.get("proj_disk_radius",      0.0008)
        blob_z_offset         = self.vis_cfg.get("blob_z_offset",         0.002)
        matched_proj_z_offset = self.vis_cfg.get("matched_proj_z_offset", 0.001)
        error_z_offset        = self.vis_cfg.get("error_z_offset",        0.0015)
        error_radius          = self.vis_cfg.get("error_radius",          0.0001)

        # Real camera-space positions (no visual offset) — used for projection
        pts_cam_real = (R @ self.model_positions.T).T + t
        # Offset positions for 3-D display only
        pts_cam = pts_cam_real + offset

        # ---- 4x4 matrix for mesh ----
        T4 = np.eye(4)
        T4[:3, :3] = R
        T4[:3,  3] = t + offset

        # ---- mesh (transform only — geometry is logged once as static) ----
        if self.vis_cfg.get("show_mesh", True):
            rr.log(
                "world/mesh",
                rr.Transform3D(
                    translation=T4[:3, 3],
                    mat3x3=T4[:3, :3],
                )
            )

        # normals_cam needed for both display and projection — compute unconditionally
        normals_cam = (R @ self.model_normals.T).T   # (N_leds, 3)

        # ---- LEDs (flat disks lying on the controller surface) ----
        if self.vis_cfg.get("show_leds", True):
            colors = np.tile([255, 0, 0], (len(pts_cam), 1))
            if assignment:
                for _, lid in assignment:
                    colors[lid] = [0, 255, 0]

            verts, faces, vcols = make_disk_mesh(pts_cam, normals_cam, colors,
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
                    strips=[[s, e] for s, e in zip(pts_cam,
                                                    pts_cam + normals_cam * 0.03)],
                    colors=[255, 0, 255],
                    radii=ray_radius,
                )
            )

        # ---- blobs, projections, rays (LED→proj), errors ----
        # Z-layer ordering (front → back):
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

        # ---- per-frame visibility mask (same logic as brute_match) ----
        vis_mask = _visible_mask(
            R, t,
            self.model_positions, self.model_normals,
            self._geo_is_inner, self._geo_radial_out, self._geo_ring_axis,
            self._geo_ring_centroid, self._geo_R_frustum_center,
            self._geo_frustum_slope, self._geo_z_frustum_top,
            self._geo_z_frustum_bot,
        )
        vis_set = set(np.where(vis_mask)[0].tolist())

        # ---- blobs as actual contour shapes ----
        if blobs is not None and len(blobs) > 0:
            # Undistort blob centroids so they live in the same space as the
            # pinhole LED projections (proj_flat = px/pz * z).  The raw
            # distorted pixel coordinate must not be used for 3-D display
            # because backproject_to_plane applies a pinhole model, which only
            # matches the LED projection when distortion has been removed first.
            blobs_undist_disp = cv2.undistortPoints(
                blobs.reshape(-1, 1, 2).astype(np.float32),
                camera.camera_matrix, camera.dist_coeffs,
                P=camera.camera_matrix,
            ).reshape(-1, 2)

            # pts_plane at error_z: both error-line endpoints live on this plane
            pts_plane = backproject_to_plane(blobs_undist_disp, camera, z=error_z)

            if self.vis_cfg.get("show_blobs", True) and contours is not None:
                blob_colors = np.array(
                    [[255, 210, 0] if i in matched_bids else [130, 100, 0]
                     for i in range(len(contours))],
                    dtype=np.uint8,
                )
                bv, bf, bc = make_contour_mesh_3d(
                    contours, camera, blob_z, blob_colors,
                    undistort=True,
                )
                if bv is not None:
                    rr.log("world/blobs", rr.Mesh3D(
                        vertex_positions=bv,
                        triangle_indices=bf,
                        vertex_colors=bc,
                    ))
                else:
                    rr.log("world/blobs", rr.Clear(recursive=False))
            elif self.vis_cfg.get("show_blobs", True):
                rr.log("world/blobs", rr.Clear(recursive=False))

            # ---- blob ID labels on the blob plane ----
            if self.vis_cfg.get("show_blob_ids", True):
                blabel_color  = self.vis_cfg.get("blob_id_label_color",  [255, 210, 0])
                blabel_offset = self.vis_cfg.get("blob_id_label_offset", [0.0006, -0.0006])
                bdx, bdy = blabel_offset[0], blabel_offset[1]
                blob_label_pts = backproject_to_plane(blobs_undist_disp, camera, z=blob_z).copy()
                blob_label_pts[:, 0] += bdx
                blob_label_pts[:, 1] += bdy
                rr.log("world/blob_ids", rr.Points3D(
                    positions=blob_label_pts,
                    labels=[str(i) for i in range(len(blobs))],
                    colors=[blabel_color] * len(blobs),
                    radii=0.0,
                ))
        else:
            pts_plane = None
            rr.log("world/blobs",    rr.Clear(recursive=False))
            rr.log("world/blob_ids", rr.Clear(recursive=False))

        # ---- visible LED projections (matched + physically visible unmatched) ----
        # Matched LEDs use matched_proj_z, unmatched use frustum_z.
        # lid_to_proj_pt maps lid → 3D point for rays and error lines.
        all_proj_pts   = []
        all_proj_lids  = []
        lid_to_proj_pt = {}
        for i, (px, py, pz) in enumerate(pts_cam_real):
            if pz > 1e-6 and (i in matched_lids or i in vis_set):
                z_val = matched_proj_z if i in matched_lids else frustum_z
                proj_pt = np.array([px / pz * z_val, py / pz * z_val, z_val])
                all_proj_pts.append(proj_pt)
                all_proj_lids.append(i)
                lid_to_proj_pt[i] = proj_pt

        if self.vis_cfg.get("show_projected", True):
            if all_proj_pts:
                proj_colors = np.array(
                    [[70, 130, 255] if lid in matched_lids else [230, 80, 50]
                     for lid in all_proj_lids],
                    dtype=np.uint8,
                )
                proj_normals = np.tile([0.0, 0.0, -1.0],
                                       (len(all_proj_pts), 1)).astype(np.float32)
                pv, pf, pc = make_disk_mesh(
                    np.array(all_proj_pts, dtype=np.float32),
                    proj_normals, proj_colors,
                    radius=proj_disk_radius, n_segments=16, surface_offset=0.0,
                )
                rr.log("world/projected_leds", rr.Mesh3D(
                    vertex_positions=pv,
                    triangle_indices=pf,
                    vertex_colors=pc,
                ))
            else:
                rr.log("world/projected_leds", rr.Clear(recursive=False))

        # ---- LED ID labels on the frustum plane ----
        if self.vis_cfg.get("show_led_ids", True):
            if all_proj_pts:
                label_color  = self.vis_cfg.get("led_id_label_color",  [180, 0, 255])
                label_offset = self.vis_cfg.get("led_id_label_offset", [0.0006, -0.0006])
                dx, dy = label_offset[0], label_offset[1]
                label_pts = np.array(all_proj_pts, dtype=np.float32).copy()
                label_pts[:, 0] += dx
                label_pts[:, 1] += dy
                rr.log("world/led_ids", rr.Points3D(
                    positions=label_pts,
                    labels=[str(lid) for lid in all_proj_lids],
                    colors=[label_color] * len(all_proj_lids),
                    radii=0.0,
                ))
            else:
                rr.log("world/led_ids", rr.Clear(recursive=False))

        # ---- rays: matched (blue) and unmatched-but-visible (orange-red) ----
        # Both sets are merged into one path so there is never a stale second
        # path persisting from a previous frame with a leftover color.
        ray_strips = []
        ray_colors = []
        error_strips    = []
        error_values    = []   # pixel reprojection errors, one per matched pair
        error_label_pts = []   # 3-D positions for error value labels

        # Pre-project all matched LEDs at once using the real K + distortion,
        # matching exactly how _matching.py computes its reported error.
        rvec_model, _ = cv2.Rodrigues(R)
        if assignment and pts_plane is not None:
            matched_lids_ord = [lid for _, lid in assignment]
            proj_matched, _ = cv2.projectPoints(
                self.model_positions[matched_lids_ord].astype(np.float32),
                rvec_model,
                t.astype(np.float32).reshape(3, 1),
                camera.camera_matrix,
                camera.dist_coeffs,
            )
            proj_matched = proj_matched.reshape(-1, 2)

            for pair_i, (bid, lid) in enumerate(assignment):
                if bid >= len(pts_plane) or lid not in lid_to_proj_pt:
                    continue
                if self.vis_cfg.get("show_rays", True):
                    ray_strips.append([pts_cam_real[lid], lid_to_proj_pt[lid]])
                    ray_colors.append([70, 130, 255])
                px, py, pz = pts_cam_real[lid]
                if pz > 1e-6:
                    proj_flat = np.array([px / pz * error_z, py / pz * error_z, error_z])
                    if self.vis_cfg.get("show_errors", True):
                        error_strips.append([proj_flat, pts_plane[bid]])
                    u, v = blobs[bid]
                    pu, pv = proj_matched[pair_i]
                    err_px = np.sqrt((u - pu) ** 2 + (v - pv) ** 2)
                    error_values.append(err_px)
                    error_label_pts.append((proj_flat + pts_plane[bid]) / 2)

        # Unmatched rays: visible LEDs not in the assignment
        if self.vis_cfg.get("show_rays", True):
            for lid, proj_pt in lid_to_proj_pt.items():
                if lid not in matched_lids:
                    ray_strips.append([pts_cam_real[lid], proj_pt])
                    ray_colors.append([230, 80, 50])

        # Always log every frame — empty strips clears the path, preventing
        # stale geometry from a previous frame bleeding through.
        rr.log("world/rays", rr.LineStrips3D(
            strips=ray_strips,
            colors=ray_colors if ray_strips else [[0, 0, 0]],
            radii=ray_radius,
        ) if ray_strips else rr.Clear(recursive=False))

        rr.log("world/errors", rr.LineStrips3D(
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
                rr.log("world/error_values", rr.Points3D(
                    positions=ev_pts,
                    labels=[f"{e:.2f}" for e in error_values],
                    colors=[ev_color] * len(ev_pts),
                    radii=0.0,
                ))
            else:
                rr.log("world/error_values", rr.Clear(recursive=False))


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
