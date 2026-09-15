"""Partial-visibility-tolerant lattice fit for ceiling-lamp fixtures --
extends src/lamp_structure_detector.py's line/pair finder with the fixture's
known physical layout (a rigid n_rows x n_cols grid of light elements, this
project's real fixtures are 2x20) to solve the flicker problem: which
elements are bright enough to detect changes frame to frame (camera motion
changes viewing angle/distance), so a raw centroid-of-visible-blobs or
raw-blob bearing check moves around even though the physical fixture does
not.

Fix: fit the INFINITE line(s) each frame (direction + perpendicular offset),
not a discrete segment -- a line's own parameters don't depend on WHICH
subset of its points happen to be visible, only on there being enough of
them to regress. Report a fixed invariant anchor point on that line (its
closest point to a fixed image reference) as the per-frame reference for
downstream cross-frame association/triangulation, instead of a raw centroid.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.blob_detector import BlobResult
from src.lamp_structure_detector import FixtureCandidate, _blob_areas


@dataclass
class LatticeFit:
    direction: np.ndarray          # (2,) unit vector along a row
    normal: np.ndarray             # (2,) unit vector, row-to-row direction
    origin_px: np.ndarray           # (2,) fitted reference point (row0, slot 0 -- not necessarily observed)
    spacing_px: float              # column (element-to-element) spacing
    row_gap_px: float               # row-to-row gap; 0.0 if only one row found
    n_rows_found: int              # 1 or 2
    anchor_px: np.ndarray           # (2,) invariant reference point
    slot_span: int                 # columns spanned by inliers (both rows combined)
    n_inliers: int
    residual_rms_px: float
    inlier_blob_indices: List[int]
    inlier_rows_slots: List[Tuple[int, int]]  # parallel to inlier_blob_indices

    def predict_px(self, row: int, slot: int) -> np.ndarray:
        """Predicted pixel for a given (row, slot), whether or not it was
        actually observed -- used to render the fixture's inferred FULL
        extent, including elements that dropped out this frame."""
        row_offset = 0.0 if row == 0 else self.row_gap_px
        return self.origin_px + slot * self.spacing_px * self.direction + row_offset * self.normal


def _perp_dist_to_line(origin: np.ndarray, direction: np.ndarray, pt: np.ndarray) -> float:
    normal = np.array([-direction[1], direction[0]])
    return float(np.dot(pt - origin, normal))


def fit_lattice(blobs: BlobResult, seed: FixtureCandidate,
                 area_min: float = 0.4, area_max: float = 15.0,
                 spacing_range: Tuple[float, float] = (4.0, 11.0),
                 row_gap_range: Tuple[float, float] = (4.0, 11.0),
                 snap_tol_px: float = 2.2, n_cols: int = 20,
                 max_slot_span: int = 24, n_iters: int = 3) -> Optional[LatticeFit]:
    """Refit `seed` (from find_fixture_candidates) as a rigid lattice,
    snapping EVERY size-filtered blob in the frame (not just the ones the
    greedy chain caught) onto the nearest slot of a direction/spacing/
    row_gap hypothesis, then re-solving that hypothesis by linear regression
    over the inliers. Tolerates partial visibility by design: as few as ~3
    collinear points are enough to constrain a line, and any inlier subset
    of the SAME physical row regresses to the same direction/spacing/offset.
    Returns None if the refined fit doesn't look like a real fixture (too
    few inliers, or a slot span wider than the physical fixture allows --
    n_cols=20 with `max_slot_span` slack for edge/index-origin ambiguity)."""
    areas = _blob_areas(blobs)
    cand_idx = np.where((areas >= area_min) & (areas <= area_max))[0]
    if len(cand_idx) < 3:
        return None
    pts = blobs.centroids[cand_idx].astype(np.float64)
    idx_pos = {int(k): i for i, k in enumerate(cand_idx)}

    primary = max(seed.lines, key=lambda ln: len(ln.blob_indices))
    direction = primary.direction / (np.linalg.norm(primary.direction) + 1e-12)
    normal = np.array([-direction[1], direction[0]])

    sorted_t = np.sort(primary.points @ direction)
    gaps = np.diff(sorted_t)
    gaps = gaps[gaps > 0.5]
    spacing = float(np.median(gaps)) if len(gaps) else float(np.mean(spacing_range))
    spacing = float(np.clip(spacing, *spacing_range))
    origin = primary.points[0].copy()

    if len(seed.lines) == 2:
        other = min(seed.lines, key=lambda ln: len(ln.blob_indices))
        row_gap = abs(_perp_dist_to_line(origin, direction, other.points.mean(axis=0)))
        row_gap = float(np.clip(row_gap, *row_gap_range))
    else:
        row_gap = float(np.mean(row_gap_range))

    inliers = {}  # blob_idx -> (row, slot, residual)
    for _ in range(n_iters):
        inliers = {}
        for k, p in zip(cand_idx, pts):
            t = float(np.dot(p - origin, direction))
            r = float(np.dot(p - origin, normal))
            row = 0 if abs(r) <= abs(r - row_gap) else 1
            row_offset = 0.0 if row == 0 else row_gap
            slot = round(t / spacing)
            predicted = origin + slot * spacing * direction + row_offset * normal
            resid = float(np.linalg.norm(p - predicted))
            if resid <= snap_tol_px and abs(slot) <= max_slot_span:
                inliers[k] = (row, slot, resid)

        if len(inliers) < 3:
            return None

        t_vals = np.array([np.dot(pts[idx_pos[k]] - origin, direction) for k in inliers])
        slots = np.array([v[1] for v in inliers.values()], dtype=np.float64)
        if np.ptp(slots) > 0:
            spacing_fit = float(np.cov(t_vals, slots)[0, 1] / np.var(slots))
            if spacing_range[0] * 0.6 <= abs(spacing_fit) <= spacing_range[1] * 1.6:
                t0 = float(np.mean(t_vals) - spacing_fit * np.mean(slots))
                origin = origin + t0 * direction
                spacing = float(np.clip(abs(spacing_fit), *spacing_range))

        row1_r = [np.dot(pts[idx_pos[k]] - origin, normal)
                  for k, v in inliers.items() if v[0] == 1]
        if len(row1_r) >= 2:
            row_gap = float(np.clip(np.mean(row1_r), *row_gap_range))

    n_rows_found = 2 if any(v[0] == 1 for v in inliers.values()) else 1
    all_slots = [v[1] for v in inliers.values()]
    slot_span = int(max(all_slots) - min(all_slots) + 1) if all_slots else 0
    if slot_span > n_cols + 4:  # physical fixture only has n_cols columns, plus slack for fit noise
        return None

    residual_rms = float(np.sqrt(np.mean([v[2] ** 2 for v in inliers.values()])))
    # Anchor = midpoint of the OBSERVED slot span, not an arbitrary extrapolated
    # point -- projecting onto "closest point to image center" was tried first and
    # rejected: two genuinely separate real fixtures with similar line direction can
    # have that projection collapse to nearby pixels purely because they're both
    # roughly parallel lines equidistant from the image center, even though their
    # actually-visible chains sit far apart -- confirmed on real cam3 data, where it
    # visually collapsed 3 distinct line candidates' anchors into one small region.
    mid_slot = 0.5 * (min(all_slots) + max(all_slots))
    anchor = origin + mid_slot * spacing * direction
    if n_rows_found == 2:
        anchor = anchor + 0.5 * row_gap * normal

    return LatticeFit(
        direction=direction, normal=normal, origin_px=origin,
        spacing_px=spacing, row_gap_px=row_gap if n_rows_found == 2 else 0.0,
        n_rows_found=n_rows_found, anchor_px=anchor, slot_span=slot_span,
        n_inliers=len(inliers), residual_rms_px=residual_rms,
        inlier_blob_indices=list(inliers.keys()),
        inlier_rows_slots=[(v[0], v[1]) for v in inliers.values()],
    )


def triangulate_rays(origins: np.ndarray, directions: np.ndarray) -> Tuple[np.ndarray, float]:
    """Least-squares closest point to N rays (origin_i, unit direction_i) in
    3D -- classic multi-view triangulation. Returns (point, rms_perp_residual_m)."""
    origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(directions, dtype=np.float64)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, directions):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    point = np.linalg.solve(A, b)

    resids = []
    for o, d in zip(origins, directions):
        v = point - o
        perp = v - np.dot(v, d) * d
        resids.append(np.linalg.norm(perp))
    return point, float(np.sqrt(np.mean(np.square(resids))))
