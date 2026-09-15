"""Targeted, single-frame structural detector for ceiling-lamp fixtures --
an alternative to the multi-frame 3D voxel-vote approach in
src/static_light_map.py, aimed at this project's specific fixture geometry
(confirmed on real cam3 data): small dim blobs (~0.5-15px area) arranged in
1-2 nearly-straight, nearly-parallel lines, ~5-10px spacing between
consecutive blobs along a line, ~5-10px perpendicular gap between the two
lines when both are visible. One line is typically sparser (farther/dimmer,
more of its blobs drop below the detection floor).

This module only finds the 2D STRUCTURE in one frame's blob set -- it has no
opinion on whether a candidate is actually static in the room; pair with
src/static_light_geometry.py (mocap ray casting) to verify that separately
(see find_lamp_structures.py).
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.blob_detector import BlobResult


@dataclass
class LineCandidate:
    blob_indices: List[int]           # indices into the BlobResult this was found from
    points: np.ndarray                # (N,2) centroids, in chain order
    direction: np.ndarray             # (2,) unit vector, fitted


@dataclass
class FixtureCandidate:
    lines: List[LineCandidate]        # 1 or 2 lines
    blob_indices: List[int]           # union of all lines' blob indices
    centroid_px: np.ndarray           # (2,) mean of all member blob centroids


def _blob_areas(blobs: BlobResult) -> np.ndarray:
    areas = np.empty(len(blobs), dtype=np.float64)
    for i, c in enumerate(blobs.contours):
        c = np.asarray(c, dtype=np.float32)
        areas[i] = cv2.contourArea(c) if len(c) >= 3 else float(len(c))
    return areas


def _fit_direction(points: np.ndarray) -> np.ndarray:
    """Unit direction of the best-fit line through points (PCA principal axis)."""
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered)
    d = vt[0]
    return d / (np.linalg.norm(d) + 1e-12)


def find_lines(blobs: BlobResult, area_min: float = 0.4, area_max: float = 15.0,
               spacing_min_px: float = 4.0, spacing_max_px: float = 11.0,
               angle_tol_deg: float = 15.0, min_line_len: int = 3) -> List[LineCandidate]:
    """Greedy chain-growing line finder. Only considers blobs within
    [area_min, area_max]; only extends a chain to a candidate within
    [spacing_min_px, spacing_max_px] of the chain's current end AND within
    angle_tol_deg of the chain's running direction estimate."""
    areas = _blob_areas(blobs)
    cand_idx = np.where((areas >= area_min) & (areas <= area_max))[0]
    if len(cand_idx) < min_line_len:
        return []
    pts = blobs.centroids[cand_idx].astype(np.float64)
    n = len(cand_idx)
    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)

    used = np.zeros(n, dtype=bool)
    lines: List[LineCandidate] = []
    angle_tol_cos = np.cos(np.radians(angle_tol_deg))

    # Seed chains from every unused edge whose length is in range, longest-first
    # so a chain claims its blobs before a weaker overlapping seed tries to.
    edges = [(dist[i, j], i, j) for i in range(n) for j in range(i + 1, n)
             if spacing_min_px <= dist[i, j] <= spacing_max_px]
    edges.sort()

    for _, i0, j0 in edges:
        if used[i0] or used[j0]:
            continue
        chain = [i0, j0]
        chain_used = {i0, j0}
        direction = (pts[j0] - pts[i0])
        direction /= (np.linalg.norm(direction) + 1e-12)

        def _extend(chain, direction, forward: bool):
            grown = True
            while grown:
                grown = False
                end = chain[-1] if forward else chain[0]
                best = None
                for k in range(n):
                    if used[k] or k in chain_used:
                        continue
                    d = pts[k] - pts[end]
                    dn = np.linalg.norm(d)
                    if not (spacing_min_px <= dn <= spacing_max_px):
                        continue
                    cos_sim = abs(float(np.dot(d / dn, direction)))
                    if cos_sim < angle_tol_cos:
                        continue
                    if best is None or dn < best[0]:
                        best = (dn, k)
                if best is not None:
                    k = best[1]
                    if forward:
                        chain.append(k)
                    else:
                        chain.insert(0, k)
                    chain_used.add(k)
                    grown = True
            return chain

        chain = _extend(chain, direction, forward=True)
        chain = _extend(chain, direction, forward=False)

        if len(chain) >= min_line_len:
            for k in chain:
                used[k] = True
            chain_pts = pts[chain]
            lines.append(LineCandidate(
                blob_indices=[int(cand_idx[k]) for k in chain],
                points=chain_pts,
                direction=_fit_direction(chain_pts),
            ))

    return lines


def find_fixture_candidates(blobs: BlobResult, lines: List[LineCandidate],
                             parallel_angle_tol_deg: float = 12.0,
                             perp_dist_min_px: float = 4.0,
                             perp_dist_max_px: float = 11.0,
                             min_single_line_len: int = 5) -> List[FixtureCandidate]:
    """Pair up near-parallel lines separated by perp_dist_min_px..perp_dist_max_px
    (the fixture's characteristic two-row structure) into FixtureCandidates.
    A single line with >= min_single_line_len blobs and no matching pair is
    ALSO accepted (lower confidence -- covers the case where the second row
    is entirely below the detection floor this frame), tagged by having only
    one entry in `.lines`."""
    used = [False] * len(lines)
    candidates: List[FixtureCandidate] = []

    def _perp_gap(a: LineCandidate, b: LineCandidate) -> float:
        # distance from b's centroid to a's infinite line
        c_a = a.points.mean(axis=0)
        n_a = np.array([-a.direction[1], a.direction[0]])
        return abs(float(np.dot(b.points.mean(axis=0) - c_a, n_a)))

    pairs = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            cos_sim = abs(float(np.dot(lines[i].direction, lines[j].direction)))
            if cos_sim < np.cos(np.radians(parallel_angle_tol_deg)):
                continue
            gap = _perp_gap(lines[i], lines[j])
            if perp_dist_min_px <= gap <= perp_dist_max_px:
                pairs.append((len(lines[i].blob_indices) + len(lines[j].blob_indices), i, j))
    pairs.sort(reverse=True)

    for _, i, j in pairs:
        if used[i] or used[j]:
            continue
        used[i] = used[j] = True
        idx = lines[i].blob_indices + lines[j].blob_indices
        candidates.append(FixtureCandidate(
            lines=[lines[i], lines[j]], blob_indices=idx,
            centroid_px=blobs.centroids[idx].mean(axis=0),
        ))

    for i, ln in enumerate(lines):
        if used[i]:
            continue
        if len(ln.blob_indices) >= min_single_line_len:
            candidates.append(FixtureCandidate(
                lines=[ln], blob_indices=list(ln.blob_indices),
                centroid_px=blobs.centroids[ln.blob_indices].mean(axis=0),
            ))

    return candidates
