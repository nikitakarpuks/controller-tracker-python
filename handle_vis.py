#!/usr/bin/env python3
"""
handle_vis.py — Controller handle model visualization.

PURPOSE
-------
Helps you build a geometric handle model for occlusion testing.
Add primitive shapes (Rect3D, Disk3D, Cylinder3D) to PRIMITIVES below,
run the script, and compare them side-by-side with the LED frustum and
the precise 3-D mesh in Rerun.

WORKFLOW
--------
1. python handle_vis.py          # see frustum + mesh
2. Uncomment / add shapes in PRIMITIVES.
3. Re-run → tweak → repeat.
4. Transfer finalized parameters to the occlusion tester.

COORDINATE SYSTEM
-----------------
Everything is in controller-local space (the same frame as the raw LED
positions in the JSON calibration file).  The LED ring axis is ~along Z;
the handle extends in the –Z direction.

Colour conventions
------------------
  red   LEDs  — outer (face outward on the frustum)
  blue  LEDs  — inner (face inward, can be self-occluded by the frustum wall)
  blue-grey   — frustum truncated-cone mesh
  mesh        — semi-transparent grey
  primitives  — whatever colour you set per shape
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from src.load_config import load_yaml_config, load_json_config
from src.controller import create_leds_from_config, _compute_geometry
from src.geometry import Box3D, Cylinder3D, tangent_frame
from src.visualization import build_alignment_transform, load_trimesh, make_disk_mesh
from src._matching import _visible_mask


# ═══════════════════════════════════════════════════════════════════════════════
#  Box3D / Cylinder3D are defined in src/geometry.py (imported above).
# ═══════════════════════════════════════════════════════════════════════════════
#  ↓↓↓  EDIT HERE: define your handle primitives  ↓↓↓
# ═══════════════════════════════════════════════════════════════════════════════
#
# Shift the precise 3-D mesh along X so it sits beside the primitives.
# Set to 0.0 to overlap both objects for a direct comparison.
MESH_X_OFFSET: float = 0#0.15   # metres

# LED ID label appearance
LED_LABEL_SHOW:          bool  = True
LED_LABEL_COLOR:         list  = [255, 220, 0]   # RGB — bright yellow
LED_LABEL_NORMAL_OFFSET: float = 0.010           # metres above LED surface along its normal
LED_LABEL_RADIUS:        float = 0.0005             # anchor dot radius (0 = invisible, text only)

# Debug visibility overlay: set to a camera position (controller-local, metres)
# to see green=visible / red=occluded rays from that viewpoint.
# Set to None to disable.
DEBUG_CAM_POS: Optional[np.ndarray] = None #np.array([0.0, 0.0, 0.0])  # np.array([0.2, -0.2, -0.20])
# Try also: np.array([0.2, 0.2, -0.2]) for diagonal, or None to disable.
#
# Tips:
#   • Start with a rough Cylinder3D for the grip, compare with the mesh.
#   • Add Box3D patches for flat/boxy parts (trigger guard, shoulder etc.).
#   • axes=None keeps the box world-aligned; pass a (3,3) column matrix to
#     rotate it — columns are the local x, y, z unit vectors.
#   • Re-run after each change — Rerun opens automatically.
#   • LED coordinate ranges are printed to stdout on every run.
#
PRIMITIVES: list = [
    Box3D(
        name="box_vertical",
        center=np.array([-0.009, -0.012, -0.015]),
        half_dims=np.array([0.021, 0.0028, 0.013]),   # half x, y, z extents
        # axes=None  →  world-aligned; example with custom orientation:
        # axes=np.column_stack([[1,0,0],[0,0,-1],[0,1,0]]),
        color=[220, 130, 60],
    ),
    Box3D(
        name="box_horizontal",
        center=np.array([-0.009, -0.029, -0.0048]), # normal dims
        half_dims=np.array([0.021, 0.0032, 0.015]),  # half x, y, z extents
        # axes=None  →  world-aligned; example with custom orientation:
        axes=np.column_stack([[1,0,0],[0,0,-1],[0,1,0]]),
        color=[130, 60, 220],
    ),
    Cylinder3D(
        name="control_panel",
        center=np.array([-0.008, 0.0, -0.05]),  # centre of grip, metres
        axis=np.array([0.0, 1.0, 0.0]),
        radius=0.031,  # extent along tangent u (~Z)
        radius_v=0.033,  # extent along tangent v (~X) — oval cross-section
        half_length=0.017,
        angle=np.pi / 10,
        color=[100, 180, 255],
    ),
    Cylinder3D(
        name="handle",
        center=np.array([0.001, 0.022, -0.098]),   # centre of grip, metres
        axis=np.array([0.0, 1.0, -1.5]),
        radius=0.021,    # extent along tangent u (~Z)
        half_length=0.037,
        color=[255, 55, 55],
    ),

]
# ═══════════════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ───────────────────────────────────────────────────────────────────────────────


def build_box_mesh(b: Box3D):
    c = np.asarray(b.center,    float)
    d = np.asarray(b.half_dims, float)

    if b.axes is not None:
        M = np.asarray(b.axes, float)   # (3,3), columns = local x,y,z
    else:
        M = np.eye(3)

    # 8 corners; index encodes sign per axis (bit0=x, bit1=y, bit2=z; 0→-1, 1→+1)
    signs = np.array([
        [-1,-1,-1], [+1,-1,-1], [-1,+1,-1], [+1,+1,-1],
        [-1,-1,+1], [+1,-1,+1], [-1,+1,+1], [+1,+1,+1],
    ], dtype=float)
    verts = (c + (signs * d) @ M.T).astype(np.float32)

    faces = np.array([
        [0,2,3],[0,3,1],   # -z face
        [4,5,7],[4,7,6],   # +z face
        [0,4,5],[0,5,1],   # -y face
        [2,7,3],[2,6,7],   # +y face
        [0,4,6],[0,6,2],   # -x face
        [1,3,7],[1,7,5],   # +x face
    ], dtype=np.int32)

    colors = np.tile(np.array(b.color[:3], dtype=np.uint8), (8, 1))
    return verts, faces, colors


def build_cylinder_mesh(cy: Cylinder3D):
    ax = np.asarray(cy.axis, float); ax /= np.linalg.norm(ax) + 1e-9
    u, v = tangent_frame(ax)
    c    = np.asarray(cy.center, float)
    ns   = cy.n_segments
    angles  = np.linspace(0, 2*np.pi, ns, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    if cy.angle:
        ca, sa = np.cos(cy.angle), np.sin(cy.angle)
        u, v = ca*u + sa*v, -sa*u + ca*v
    r_u  = cy.radius
    r_v  = cy.radius_v if cy.radius_v is not None else cy.radius
    p0   = c - cy.half_length * ax   # bottom disc centre
    p1   = c + cy.half_length * ax   # top    disc centre
    rim0 = p0 + r_u * cos_a[:, None]*u + r_v * sin_a[:, None]*v  # (ns,3)
    rim1 = p1 + r_u * cos_a[:, None]*u + r_v * sin_a[:, None]*v  # (ns,3)

    # indices: 0..ns-1 = bottom rim, ns..2ns-1 = top rim
    verts_list = [rim0, rim1]
    faces_list = []
    for i in range(ns):
        j = (i + 1) % ns
        faces_list += [[i, j, ns+j], [i, ns+j, ns+i]]

    if cy.closed:
        bot_idx = 2 * ns       # centre of bottom cap
        top_idx = 2 * ns + 1   # centre of top cap
        verts_list += [p0[None], p1[None]]
        for i in range(ns):
            j = (i + 1) % ns
            faces_list.append([bot_idx, j, i])        # bottom cap (CW → normal points down)
            faces_list.append([top_idx, ns+i, ns+j])  # top cap    (CCW → normal points up)

    verts  = np.vstack(verts_list).astype(np.float32)
    faces  = np.array(faces_list, dtype=np.int32)
    colors = np.tile(np.array(cy.color[:3], dtype=np.uint8), (len(verts), 1))
    return verts, faces, colors


def build_frustum_mesh(ring_axis: np.ndarray, ring_centroid: np.ndarray,
                       positions: np.ndarray, is_inner: np.ndarray, z_rel: np.ndarray,
                       R_fc: float, slope: float,
                       z_top: float, z_bot: float,
                       n_rings: int = 24, n_segments: int = 60):
    """
    Thick-walled truncated-cone mesh for the LED ring frustum.

    Wall thickness is derived from the h_corpus of the inner LEDs — the radial
    gap between each inner LED and the outer cone surface (same calculation that
    was commented out in _compute_frustum_geometry).  The inner cone has the same
    slope as the outer one, offset inward by the mean gap.

    Surfaces generated:
      • outer side wall  (faces outward)
      • inner side wall  (faces inward)
      • top annular cap  (z_top edge)
      • bottom annular cap (z_bot edge)
    """
    ax = np.asarray(ring_axis, float)
    c  = np.asarray(ring_centroid, float)
    u, v = tangent_frame(ax)

    # ── wall thickness from inner LED h_corpus ────────────────────────────────
    rel      = positions - c
    rel_proj = rel - np.outer(rel @ ax, ax)          # radial component of each LED (N, 3)
    r_led    = np.linalg.norm(rel_proj, axis=1)      # radial distance from axis   (N,)

    if is_inner.any():
        # h_corpus[i] = gap from inner LED to outer wall at the same axial slice
        h_corpus       = np.maximum(0.0, R_fc + slope * z_rel[is_inner] - r_led[is_inner])
        wall_thickness = float(h_corpus.mean())
    else:
        wall_thickness = 0.003   # 3 mm fallback

    wall_thickness = 0.007

    R_fc_inner     = R_fc - wall_thickness
    ring_center_ax = float((positions @ ax).mean())   # z_rel=0 offset in world space

    z_vals  = np.linspace(z_bot, z_top, n_rings)
    angles  = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    outer_rings, inner_rings = [], []
    for z in z_vals:
        origin  = c + (ring_center_ax + z) * ax
        r_out   = R_fc       + slope * z
        r_in    = R_fc_inner + slope * z
        radial  = cos_a[:, None]*u + sin_a[:, None]*v    # (ns, 3) unit radial dirs
        outer_rings.append(origin + r_out * radial)
        inner_rings.append(origin + r_in  * radial)

    # vertex layout
    # outer: [0,          n_rings*n_segments)
    # inner: [n_rings*ns, 2*n_rings*n_segments)
    ns    = n_segments
    OFF_I = n_rings * ns
    verts = np.vstack(outer_rings + inner_rings).astype(np.float32)

    def oi(ri, i): return ri * ns + i           # outer ring ri, segment i
    def ii(ri, i): return OFF_I + ri * ns + i   # inner ring ri, segment i

    faces = []

    # outer side wall (normals face outward)
    for ri in range(n_rings - 1):
        for i in range(ns):
            j = (i + 1) % ns
            faces += [[oi(ri,i), oi(ri,j), oi(ri+1,j)],
                      [oi(ri,i), oi(ri+1,j), oi(ri+1,i)]]

    # inner side wall (reversed winding → normals face inward)
    for ri in range(n_rings - 1):
        for i in range(ns):
            j = (i + 1) % ns
            faces += [[ii(ri,i), ii(ri+1,j), ii(ri,j)],
                      [ii(ri,i), ii(ri+1,i), ii(ri+1,j)]]

    # top annular cap  (z_top end, ri = n_rings-1)
    ri_t = n_rings - 1
    for i in range(ns):
        j = (i + 1) % ns
        faces += [[oi(ri_t,i), oi(ri_t,j), ii(ri_t,j)],
                  [oi(ri_t,i), ii(ri_t,j), ii(ri_t,i)]]

    # bottom annular cap (z_bot end, ri = 0)  — reversed winding
    ri_b = 0
    for i in range(ns):
        j = (i + 1) % ns
        faces += [[oi(ri_b,i), ii(ri_b,j), oi(ri_b,j)],
                  [oi(ri_b,i), ii(ri_b,i), ii(ri_b,j)]]

    faces  = np.array(faces, dtype=np.int32)
    colors = np.tile(np.array([130, 130, 220], dtype=np.uint8), (len(verts), 1))
    return verts, faces, colors, wall_thickness


def _log_mesh(path: str, verts: np.ndarray, faces: np.ndarray, colors: np.ndarray):
    rr.log(path, rr.Mesh3D(
        vertex_positions=verts,
        triangle_indices=faces,
        vertex_colors=colors,
    ), static=True)


# ───────────────────────────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────────────────────────

def main():
    config = load_yaml_config('./config/config.yml')

    # ── Load LEDs (controller-local space) ────────────────────────────────────
    cal       = load_json_config(config["controllers"]["right_controller"]["config_path"])
    leds      = create_leds_from_config(cal)
    positions = np.array([l.position for l in leds], dtype=np.float64)
    normals   = np.array([l.normal   for l in leds], dtype=np.float64)

    # ── Full controller geometry (frustum + handle primitives) ───────────────
    geom        = _compute_geometry(positions, normals)
    ring_axis   = geom.ring_axis
    is_inner    = geom.is_inner
    ring_centroid = geom.ring_centroid
    R_fc        = geom.R_fc
    slope       = geom.frustum_slope
    z_top       = geom.z_frustum_top
    z_bot       = geom.z_frustum_bot
    z_rel       = geom.z_rel

    # ── Load 3-D mesh → transform to controller-local space ──────────────────
    mesh_verts = None
    mesh_faces = None
    mesh_path  = config["visualization"].get("3d_model_path")
    if mesh_path:
        try:
            raw_mesh     = load_trimesh(mesh_path)
            T_model_ctrl = build_alignment_transform(
                config["visualization"]["initial_position_change"]["right"]
            )
            T_ctrl_model = T_model_ctrl.inverse()   # model-space → controller-space
            mesh_verts   = T_ctrl_model.apply(raw_mesh.vertices.astype(np.float64))
            mesh_faces   = raw_mesh.faces
            print(f"[handle_vis] mesh loaded: {len(mesh_verts)} verts, {len(mesh_faces)} faces")
        except Exception as e:
            print(f"[handle_vis] could not load mesh ({e}) — skipping")

    # ── Rerun init ────────────────────────────────────────────────────────────
    rr.init("handle_model_vis", spawn=True)
    rr.send_blueprint(rrb.Blueprint(
        rrb.Spatial3DView(name="Controller handle model", origin="/world"),
        collapse_panels=False,
    ))

    # Coordinate axes
    rr.log("world/frame", rr.Arrows3D(
        origins=[[0,0,0]]*3,
        vectors=[[0.05,0,0],[0,0.05,0],[0,0,0.05]],
        colors=[[255,0,0],[0,255,0],[0,0,255]],
    ), static=True)
    rr.log("world/frame_labels", rr.Points3D(
        positions=[[0.057,0,0],[0,0.057,0],[0,0,0.057]],
        labels=["X", "Y", "Z"],
        colors=[[255,0,0],[0,255,0],[0,0,255]],
        radii=0.0,
    ), static=True)

    # Ring axis arrow
    rr.log("world/ring_axis", rr.Arrows3D(
        origins=[ring_centroid],
        vectors=[ring_axis * 0.04],
        colors=[[255, 255, 0]],
    ), static=True)

    # Frustum cone
    fv, ff, fc, wall_mm = build_frustum_mesh(
        ring_axis, ring_centroid, positions, is_inner, z_rel,
        R_fc, slope, z_top, z_bot,
    )
    _log_mesh("world/frustum", fv, ff, fc)

    # LEDs: outer = red, inner = blue
    led_colors = np.where(
        is_inner[:, None],
        np.array([[80, 80, 255]], dtype=np.uint8),
        np.array([[255, 80, 80]], dtype=np.uint8),
    )
    lv, lf, lc = make_disk_mesh(positions, normals, led_colors.astype(np.uint8),
                                 radius=0.0015, surface_offset=0.0)
    _log_mesh("world/leds", lv, lf, lc)

    # LED normals (short stubs)
    rr.log("world/normals", rr.LineStrips3D(
        strips=[[p, p + n * 0.008] for p, n in zip(positions, normals)],
        colors=[200, 200, 200],
        radii=0.00015,
    ), static=True)

    # LED index labels (useful for debugging specific LEDs)
    if LED_LABEL_SHOW:
        rr.log("world/led_ids", rr.Points3D(
            positions=positions + normals * LED_LABEL_NORMAL_OFFSET,
            labels=[str(i) for i in range(len(positions))],
            colors=[LED_LABEL_COLOR] * len(positions),
            radii=LED_LABEL_RADIUS,
        ), static=True)

    # 3-D mesh (semi-transparent grey), optionally shifted along X for comparison
    if mesh_verts is not None:
        shifted = mesh_verts.copy()
        shifted[:, 0] += MESH_X_OFFSET
        rr.log("world/mesh", rr.Mesh3D(
            vertex_positions=shifted.astype(np.float32),
            triangle_indices=mesh_faces,
            albedo_factor=[0.6, 0.6, 0.6, 0.5],
        ), static=True)

    # User-defined primitives
    counts: dict = {}
    for prim in PRIMITIVES:
        n = prim.name
        counts[n] = counts.get(n, 0) + 1
        path = f"world/primitives/{n}_{counts[n]}"

        if isinstance(prim, Box3D):
            pv, pf, pc = build_box_mesh(prim)
        elif isinstance(prim, Cylinder3D):
            pv, pf, pc = build_cylinder_mesh(prim)
        else:
            print(f"[handle_vis] unknown primitive type: {type(prim)}")
            continue
        _log_mesh(path, pv, pf, pc)
        print(f"[handle_vis] logged {path}")

    # ── Debug visibility overlay ──────────────────────────────────────────────
    # Colors each LED green (visible) or red (occluded) from DEBUG_CAM_POS.
    # Also draws rays and marks the virtual camera so you can verify the shapes
    # transferred from PRIMITIVES to _compute_geometry match.
    # if DEBUG_CAM_POS is not None:
    from src._matching import _rays_blocked_by_box, _rays_blocked_by_cylinder

    # R_dbg = cv2.Rodrigues(np.array([[0.26485262], [1.42825671], [-2.72708413]]))[0]
    # t_dbg = np.array([0.05591549, -0.08879955, 0.29305191])

    R_dbg = cv2.Rodrigues(np.array([[0.0], [0.0], [0.0]]))[0]
    t_dbg = np.array([0.0, 0.0, -0.1])

    # R_dbg    = np.eye(3, dtype=np.float64)
    # t_dbg    = -DEBUG_CAM_POS.astype(np.float64)   # cam_world = -R^T t = cam_pos
    cam_world = -(R_dbg.T @ t_dbg)

    vis = _visible_mask(R_dbg, t_dbg, positions, normals, geom)

    dbg_colors = np.where(
        vis[:, None],
        np.array([[0, 220, 80]],  dtype=np.uint8),   # green  = visible
        np.array([[220, 40, 40]], dtype=np.uint8),   # red    = occluded
    )
    dv, df, dc = make_disk_mesh(positions, normals, dbg_colors.astype(np.uint8),
                                radius=0.003, surface_offset=0.0015)
    _log_mesh("world/debug/led_visibility", dv, df, dc)

    rr.log("world/debug/camera", rr.Points3D(
        positions=[cam_world.tolist()],#[DEBUG_CAM_POS.tolist()],
        colors=[[255, 255, 0]],
        radii=0.006,
        labels=["cam"],
    ), static=True)

    rr.log("world/debug/rays", rr.LineStrips3D(
        strips=[[cam_world.tolist(), p.tolist()] for p in positions], # [[DEBUG_CAM_POS.tolist(), p.tolist()] for p in positions],
        colors=[[0, 200, 80] if v else [200, 40, 40] for v in vis],
        radii=0.00018,
    ), static=True)

    n_vis = int(vis.sum())
    # print(f"\n[debug] camera at {DEBUG_CAM_POS}")
    print(f"  total visible: {n_vis} / {len(vis)}")

    # Per-shape breakdown (run each blocker independently on ALL LEDs)
    for box in geom.boxes:
        blocked_idx = np.where(_rays_blocked_by_box(cam_world, positions, box))[0]
        if len(blocked_idx):
            print(f"  [{box.name}] blocks LEDs: {blocked_idx.tolist()}")
    for cy in geom.cylinders:
        blocked_idx = np.where(_rays_blocked_by_cylinder(cam_world, positions, cy))[0]
        if len(blocked_idx):
            print(f"  [{cy.name}] blocks LEDs: {blocked_idx.tolist()}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[handle_vis] {len(positions)} LEDs  |  "
          f"{is_inner.sum()} inner (blue), {(~is_inner).sum()} outer (red)")
    print(f"[handle_vis] {len(PRIMITIVES)} user primitive(s)")
    print(f"\nLED bounding box (controller-local, metres):")
    print(f"  X: [{positions[:,0].min():.4f},  {positions[:,0].max():.4f}]")
    print(f"  Y: [{positions[:,1].min():.4f},  {positions[:,1].max():.4f}]")
    print(f"  Z: [{positions[:,2].min():.4f},  {positions[:,2].max():.4f}]")
    print(f"\nFrustum params:")
    print(f"  ring_axis      = {ring_axis.round(4)}")
    print(f"  R_fc           = {R_fc*1000:.2f} mm  (outer radius at ring centroid)")
    print(f"  wall_thickness = {wall_mm*1000:.2f} mm  (mean h_corpus of inner LEDs)")
    print(f"  slope          = {slope:.4f}  (dR/dz_rel)")
    print(f"  z_bot..z_top   = [{z_bot*1000:.2f},  {z_top*1000:.2f}] mm")


if __name__ == "__main__":
    main()
