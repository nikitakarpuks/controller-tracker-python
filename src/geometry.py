"""
Shared geometric primitives for controller body modelling.

Used by:
  • handle_vis.py  — mesh building and interactive tuning
  • pose_search.py — PoseSearcher.__init__ calls _compute_geometry
  • _matching.py   — _visible_mask ray-intersection tests
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ── Primitive shapes ────────────────────────────────────────────────────────

@dataclass
class Box3D:
    """Oriented rectangular box."""
    center:    np.ndarray
    half_dims: np.ndarray
    axes:      Optional[np.ndarray] = None   # (3,3) columns = local x,y,z unit vecs; None → world-aligned
    color:     list = field(default_factory=lambda: [220, 180, 80])
    name:      str  = "box"


@dataclass
class Cylinder3D:
    """Cylinder with optional elliptical cross-section."""
    center:      np.ndarray
    axis:        np.ndarray
    radius:      float                       # radius along tangent u
    half_length: float
    radius_v:    Optional[float] = None      # radius along tangent v; None → circular
    angle:       float = 0.0                 # roll around axis (rad), rotates the ellipse
    closed:      bool  = True
    n_segments:  int   = 40
    color:       list  = field(default_factory=lambda: [100, 180, 255])
    name:        str   = "cylinder"


# ── Tangent-frame helper ─────────────────────────────────────────────────────

def tangent_frame(axis: np.ndarray):
    """Return two orthogonal unit vectors (u, v) perpendicular to *axis*."""
    n = axis / (np.linalg.norm(axis) + 1e-9)
    ref = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    u = np.cross(n, ref); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


# ── All-in-one geometry bundle ───────────────────────────────────────────────

class ControllerGeometry:
    """
    Frustum parameters (computed from LED positions) + handle body primitives
    (hardcoded from handle_vis tuning).  Passed to _visible_mask for occlusion
    testing.
    """

    __slots__ = (
        "ring_axis", "is_inner", "radial_out", "ring_centroid",
        "R_fc", "R_fc_inner", "frustum_slope", "z_frustum_top", "z_frustum_bot", "z_rel",
        "ring_center_ax",
        "boxes", "cylinders",
    )

    def __init__(
        self,
        ring_axis:      np.ndarray,
        is_inner:       np.ndarray,
        radial_out:     np.ndarray,
        ring_centroid:  np.ndarray,
        R_fc:           float,
        R_fc_inner:     float,
        frustum_slope:  float,
        z_frustum_top:  float,
        z_frustum_bot:  float,
        z_rel:          np.ndarray,
        ring_center_ax: float,
        boxes:          List[Box3D],
        cylinders:      List[Cylinder3D],
    ):
        self.ring_axis      = ring_axis
        self.is_inner       = is_inner
        self.radial_out     = radial_out
        self.ring_centroid  = ring_centroid
        self.R_fc           = R_fc
        self.R_fc_inner     = R_fc_inner
        self.frustum_slope  = frustum_slope
        self.z_frustum_top  = z_frustum_top
        self.z_frustum_bot  = z_frustum_bot
        self.z_rel          = z_rel
        self.ring_center_ax = ring_center_ax
        self.boxes          = boxes
        self.cylinders      = cylinders


# ── Factory helpers (moved from controller.py) ───────────────────────────────

def _primitives_from_cfg(prim_cfg: dict):
    """Parse Box3D / Cylinder3D lists from a handle_primitives config dict."""
    boxes = []
    for b in prim_cfg.get("boxes", []):
        axes = np.asarray(b["axes"], float) if "axes" in b else None
        boxes.append(Box3D(
            name=b["name"],
            center=np.array(b["center"], float),
            half_dims=np.array(b["half_dims"], float),
            axes=axes,
            color=b.get("color", [220, 180, 80]),
        ))
    cylinders = []
    for cy in prim_cfg.get("cylinders", []):
        cylinders.append(Cylinder3D(
            name=cy["name"],
            center=np.array(cy["center"], float),
            axis=np.array(cy["axis"], float),
            radius=float(cy["radius"]),
            radius_v=float(cy["radius_v"]) if cy.get("radius_v") is not None else None,
            half_length=float(cy["half_length"]),
            angle=float(cy.get("angle", 0.0)),
            color=cy.get("color", [100, 180, 255]),
        ))
    return boxes, cylinders


def _compute_geometry(
    positions: np.ndarray,
    normals: np.ndarray,
    geometry_cfg: dict = None,
) -> ControllerGeometry:
    """
    Fit the LED-ring frustum and bundle it with handle body primitives into one
    ControllerGeometry object.  Called once per (camera, controller) pair during
    PoseSearcher initialisation.
    """
    centroid  = np.array([0.0, 0.0, 0.0])
    ring_axis = np.array([0.0, 0.0, -1.0])

    rel        = positions - centroid
    rel_proj   = rel - np.outer(rel @ ring_axis, ring_axis)
    radial_out = rel_proj / (np.linalg.norm(rel_proj, axis=1, keepdims=True) + 1e-8)

    is_inner   = (normals * radial_out).sum(axis=1) < 0
    outer_mask = ~is_inner

    big_ring_axis = -ring_axis if float((normals[outer_mask] @ ring_axis).mean()) > 0 else ring_axis

    axial_projs    = positions @ big_ring_axis
    ring_center_ax = float(axial_projs.mean())
    z_rel          = axial_projs - ring_center_ax

    outer_idx    = np.where(outer_mask)[0]
    outer_radial = np.linalg.norm(rel_proj[outer_idx], axis=1)
    outer_z_rel  = z_rel[outer_idx]

    A = np.column_stack([np.ones(len(outer_idx)), outer_z_rel])
    coeffs, _, _, _ = np.linalg.lstsq(A, outer_radial, rcond=None)
    R_fc          = float(coeffs[0])
    frustum_slope = float(coeffs[1])

    cfg = geometry_cfg or {}
    z_frustum_top = float(outer_z_rel.max()) + float(cfg.get("z_frustum_top_padding", 0.0045))
    z_frustum_bot = float(outer_z_rel.min()) - float(cfg.get("z_frustum_bot_padding", 0.0055))

    wall_thickness = float(cfg.get("wall_thickness", 0.007))
    R_fc_inner = R_fc - wall_thickness

    prim_cfg = cfg.get("handle_primitives")
    if prim_cfg is not None:
        boxes, cylinders = _primitives_from_cfg(prim_cfg)
    else:
        boxes = [
            Box3D(
                name="box_vertical",
                center=np.array([-0.009, -0.012, -0.015]),
                half_dims=np.array([0.021, 0.0028, 0.013]),
                color=[220, 130, 60],
            ),
            Box3D(
                name="box_horizontal",
                center=np.array([-0.009, -0.029, -0.0048]),
                half_dims=np.array([0.021, 0.0032, 0.015]),
                axes=np.column_stack([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).astype(float),
                color=[130, 60, 220],
            ),
        ]
        cylinders = [
            Cylinder3D(
                name="control_panel",
                center=np.array([-0.008, 0.0, -0.05]),
                axis=np.array([0.0, 1.0, 0.0]),
                radius=0.031,
                radius_v=0.033,
                half_length=0.017,
                angle=np.pi / 10,
                color=[100, 180, 255],
            ),
            Cylinder3D(
                name="handle",
                center=np.array([0.001, 0.022, -0.098]),
                axis=np.array([0.0, 1.0, -1.5]),
                radius=0.021,
                half_length=0.037,
                color=[255, 55, 55],
            ),
        ]

    return ControllerGeometry(
        ring_axis=big_ring_axis,
        is_inner=is_inner,
        radial_out=radial_out,
        ring_centroid=centroid,
        R_fc=R_fc,
        R_fc_inner=R_fc_inner,
        frustum_slope=frustum_slope,
        z_frustum_top=z_frustum_top,
        z_frustum_bot=z_frustum_bot,
        z_rel=z_rel,
        ring_center_ax=ring_center_ax,
        boxes=boxes,
        cylinders=cylinders,
    )
