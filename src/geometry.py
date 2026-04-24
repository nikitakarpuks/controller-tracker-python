"""
Shared geometric primitives for controller body modelling.

Used by:
  • handle_vis.py  — mesh building and interactive tuning
  • controller.py  — _compute_geometry (bakes finalized shapes)
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
        "R_fc", "frustum_slope", "z_frustum_top", "z_frustum_bot", "z_rel",
        "boxes", "cylinders",
    )

    def __init__(
        self,
        ring_axis:      np.ndarray,
        is_inner:       np.ndarray,
        radial_out:     np.ndarray,
        ring_centroid:  np.ndarray,
        R_fc:           float,
        frustum_slope:  float,
        z_frustum_top:  float,
        z_frustum_bot:  float,
        z_rel:          np.ndarray,
        boxes:          List[Box3D],
        cylinders:      List[Cylinder3D],
    ):
        self.ring_axis     = ring_axis
        self.is_inner      = is_inner
        self.radial_out    = radial_out
        self.ring_centroid = ring_centroid
        self.R_fc          = R_fc
        self.frustum_slope = frustum_slope
        self.z_frustum_top = z_frustum_top
        self.z_frustum_bot = z_frustum_bot
        self.z_rel         = z_rel
        self.boxes         = boxes
        self.cylinders     = cylinders
