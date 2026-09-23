"""Persisted static-light (ceiling-lamp) map + offline voxel-vote accumulator
-- Phase 2 of the mocap-based static-light exclusion feature. Consumes
(ray_origin, ray_direction) pairs in the room frame (see
src/static_light_geometry.py) without needing any cross-frame blob identity:
a real static point accumulates votes in the same voxel across many frames
regardless of LED flicker dropout; a moving point's rays spread across many
voxels and never cross the vote threshold anywhere.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StaticLightMap:
    voxel_size_m: float
    voxel_room_positions: np.ndarray   # (M,3) float64 -- confirmed static-light centers, room frame
    voxel_radii_m: np.ndarray          # (M,) float64 -- exclusion radius per confirmed light
    vote_counts: np.ndarray            # (M,) int64 -- diagnostic, total votes behind each entry
    camera_calib_id: str = ""          # ties the map to a specific calibration/rig
    created_from_recordings: List[str] = field(default_factory=list)  # provenance


def save_static_light_map(m: StaticLightMap, path) -> None:
    payload = {
        "voxel_size_m": m.voxel_size_m,
        "voxel_room_positions": np.asarray(m.voxel_room_positions, dtype=np.float64).tolist(),
        "voxel_radii_m": np.asarray(m.voxel_radii_m, dtype=np.float64).tolist(),
        "vote_counts": np.asarray(m.vote_counts, dtype=np.int64).tolist(),
        "camera_calib_id": m.camera_calib_id,
        "created_from_recordings": list(m.created_from_recordings),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_static_light_map(path) -> StaticLightMap:
    with open(path) as f:
        payload = json.load(f)
    return StaticLightMap(
        voxel_size_m=float(payload["voxel_size_m"]),
        voxel_room_positions=np.asarray(payload["voxel_room_positions"], dtype=np.float64).reshape(-1, 3),
        voxel_radii_m=np.asarray(payload["voxel_radii_m"], dtype=np.float64).reshape(-1),
        vote_counts=np.asarray(payload["vote_counts"], dtype=np.int64).reshape(-1),
        camera_calib_id=str(payload["camera_calib_id"]),
        created_from_recordings=list(payload["created_from_recordings"]),
    )


_VoxelKey = Tuple[int, int, int]


class StaticLightVoxelAccumulator:
    """Coarse 3D voxel-grid vote accumulator over an entire recording.

    Two independent guards are required in `finalize()`, not vote count
    alone -- vote count by itself is fooled by a near-stationary headset
    watching a slowly-drifting controller, which would also stack votes in
    one voxel without representing a genuine multi-viewpoint confirmation:
      - positional spread: contributing ray origins must span at least
        `min_positional_spread_m` (their bounding-box diagonal).
      - angular spread: the bearing from origin to voxel center must span at
        least `min_angular_spread_deg` across observations -- catches
        straight-line headset motion toward/away from a moving object, which
        can satisfy positional spread alone.
    """

    def __init__(self, voxel_size_m: float = 0.10, min_votes: int = 30,
                 min_positional_spread_m: float = 0.15,
                 min_angular_spread_deg: float = 15.0,
                 radius_margin_m: float = 0.05,
                 min_confirmed_median_radius_px: float = 0.0,
                 min_confirmed_median_brightness: float = 0.0,
                 cluster_merge_distance_m: Optional[float] = None):
        """`min_confirmed_median_radius_px`/`min_confirmed_median_brightness`
        (both 0.0 = disabled) reject a voxel at confirmation time whose
        contributing blobs' MEDIAN radius/brightness falls below the given
        floor -- median, not mean/max, so a handful of coincidentally larger/
        brighter stray blobs can't rescue a voxel whose typical contributor
        is a small dim one (e.g. a wall reflection of a real lamp, as
        opposed to the lamp itself). Only applied when the caller actually
        passes `blob_radius_px`/`blob_brightness` into `add_observation` --
        votes recorded without them are treated as passing (median over an
        all-None list is skipped).

        `cluster_merge_distance_m` (None = `voxel_size_m * 2.35`): two
        confirmed voxels whose CENTERS are within this distance are merged
        into one cluster (see `_merge_clusters`) -- a plain Euclidean
        distance test on voxel centers, not 26-connectivity grid adjacency,
        because two real sub-features of one physical fixture (e.g. a
        fixture's two bulbs) can genuinely triangulate to centers more than
        one voxel-step apart (confirmed on real data: two confirmed voxels
        for the same fixture landed 0.2236m apart against a 0.10m voxel
        size, whose max grid-diagonal adjacency is only sqrt(3)*0.10=0.173m
        -- no grid-adjacency radius could have bridged that gap). The 2.35x
        default (0.235m at the default voxel size) is deliberately threaded
        between that real same-fixture gap (0.2236m, must merge) and the
        nearest distinct-feature gap observed in the same dataset (0.2449m,
        must NOT merge, or a still-unconfirmed candidate voxel silently
        contaminates a real light's position/vote count) -- a narrow,
        dataset-specific margin, not yet swept against the full 8-recording
        dataset; revisit if a future recording's real fixtures sit closer
        together than ~0.245m."""
        self.voxel_size_m = float(voxel_size_m)
        self.min_votes = int(min_votes)
        self.min_positional_spread_m = float(min_positional_spread_m)
        self.min_angular_spread_deg = float(min_angular_spread_deg)
        self.radius_margin_m = float(radius_margin_m)
        self.min_confirmed_median_radius_px = float(min_confirmed_median_radius_px)
        self.min_confirmed_median_brightness = float(min_confirmed_median_brightness)
        self.cluster_merge_distance_m = (float(cluster_merge_distance_m)
                                          if cluster_merge_distance_m is not None
                                          else self.voxel_size_m * 2.35)
        self._voxels: Dict[_VoxelKey, dict] = {}

    def _voxel_center(self, vox: _VoxelKey) -> np.ndarray:
        return (np.array(vox, dtype=np.float64) + 0.5) * self.voxel_size_m

    def add_observation(self, ray_origin: np.ndarray, ray_dir: np.ndarray,
                         max_range_m: float = 8.0, step_m: float = 0.05,
                         frame_ts_ns: Optional[int] = None,
                         blob_radius_px: Optional[float] = None,
                         blob_brightness: Optional[float] = None,
                         blob_px: Optional[np.ndarray] = None) -> None:
        """Walk the ray from ray_origin out to max_range_m in step_m
        increments, registering one vote (not one per sample) in every
        voxel it passes through -- votes count independent rays, never
        point-samples along the same ray.

        `frame_ts_ns`/`blob_radius_px`/`blob_brightness`/`blob_px` (all
        optional) are stored per vote purely for diagnostics and the
        optional size/brightness confirmation gate (see
        `finalize(return_debug=True)`, `min_confirmed_*`) -- none of them
        affect voxel-count confirmation itself. `blob_px` (the original 2D
        detected pixel this ray was cast from) plus `frame_ts_ns` together
        let a caller with camera/pose access re-derive each vote's own
        reprojection residual against the cluster's fitted 3D centroid
        after the fact -- a real static light's votes should all land
        within centroid/detection noise of that reprojection; a voxel that
        only accumulated by coincidence (its "votes" aren't really the same
        physical point) should show a much larger, more scattered residual.
        This class deliberately has no Camera/pose dependency itself, so it
        only carries the raw (ts, px) pairs -- it doesn't compute the
        residual."""
        ray_origin = np.asarray(ray_origin, dtype=np.float64).reshape(3)
        ray_dir = np.asarray(ray_dir, dtype=np.float64).reshape(3)
        norm = np.linalg.norm(ray_dir)
        if norm == 0:
            return
        ray_dir = ray_dir / norm

        n_steps = max(1, int(max_range_m / step_m))
        depths = (np.arange(1, n_steps + 1) * step_m)[:, None]
        pts = ray_origin[None, :] + depths * ray_dir[None, :]
        voxel_idx = np.floor(pts / self.voxel_size_m).astype(np.int64)
        unique_voxels = np.unique(voxel_idx, axis=0)

        for row in unique_voxels:
            vox = (int(row[0]), int(row[1]), int(row[2]))
            entry = self._voxels.get(vox)
            if entry is None:
                entry = {"count": 0, "origins": [], "timestamps": [], "radii": [], "brightnesses": [],
                         "blob_px": []}
                self._voxels[vox] = entry
            entry["count"] += 1
            entry["origins"].append(ray_origin)
            entry["timestamps"].append(frame_ts_ns)
            entry["radii"].append(blob_radius_px)
            entry["brightnesses"].append(blob_brightness)
            entry["blob_px"].append(None if blob_px is None else np.asarray(blob_px, dtype=np.float64).reshape(2))

    def _spread_stats(self, vox: _VoxelKey, origins: np.ndarray) -> Tuple[float, float]:
        bbox_diag = float(np.linalg.norm(origins.max(axis=0) - origins.min(axis=0)))
        center = self._voxel_center(vox)
        bearings = center[None, :] - origins
        norms = np.linalg.norm(bearings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        bearings = bearings / norms
        cos_matrix = np.clip(bearings @ bearings.T, -1.0, 1.0)
        max_angle_deg = float(np.degrees(np.arccos(cos_matrix.min())))
        return bbox_diag, max_angle_deg

    def _passes_spread_checks(self, vox: _VoxelKey, origins: np.ndarray) -> bool:
        bbox_diag, max_angle_deg = self._spread_stats(vox, origins)
        return bbox_diag >= self.min_positional_spread_m and max_angle_deg >= self.min_angular_spread_deg

    def _merge_clusters(self, voxel_keys) -> List[List[_VoxelKey]]:
        """Union confirmed voxels into clusters via a distance graph on
        their centers (see `cluster_merge_distance_m` in __init__), not
        grid adjacency -- O(n^2) in the number of CONFIRMED voxels, which
        is small (single/low double digits per recording) so this is cheap."""
        keys = list(voxel_keys)
        centers = {k: self._voxel_center(k) for k in keys}
        remaining = set(keys)
        clusters = []
        while remaining:
            start = remaining.pop()
            stack = [start]
            cluster = [start]
            while stack:
                cur_center = centers[stack.pop()]
                for other in list(remaining):
                    if np.linalg.norm(centers[other] - cur_center) <= self.cluster_merge_distance_m:
                        remaining.remove(other)
                        stack.append(other)
                        cluster.append(other)
            clusters.append(cluster)
        return clusters

    def finalize(self, camera_calib_id: str = "",
                 created_from_recordings: Optional[List[str]] = None,
                 return_debug: bool = False):
        """`return_debug=True` additionally returns a list (one dict per
        confirmed cluster, same order as the returned map's arrays) with
        `n_voxels`, `n_votes_total`, `ts_min`/`ts_max` (the frame_ts_ns range
        that contributed votes to this cluster) and `n_unique_frames` (how
        many distinct frames those votes came from -- a cluster whose votes
        cluster in a narrow ts range and/or come from very few unique frames
        despite passing the positional/angular spread gates is a red flag
        for near-duplicate-frame contamination during a near-static stretch,
        not genuine independent multi-viewpoint confirmation). Diagnostic
        only -- not persisted into the StaticLightMap JSON."""
        confirmed: Dict[_VoxelKey, dict] = {}
        for vox, data in self._voxels.items():
            if data["count"] < self.min_votes:
                continue
            origins = np.array(data["origins"])
            if not self._passes_spread_checks(vox, origins):
                continue
            if self.min_confirmed_median_radius_px > 0.0:
                r = [v for v in data["radii"] if v is not None]
                if r and float(np.median(r)) < self.min_confirmed_median_radius_px:
                    continue
            if self.min_confirmed_median_brightness > 0.0:
                b = [v for v in data["brightnesses"] if v is not None]
                if b and float(np.median(b)) < self.min_confirmed_median_brightness:
                    continue
            confirmed[vox] = data

        clusters = self._merge_clusters(confirmed.keys())

        positions, radii, votes = [], [], []
        debug_info = []
        for cluster in clusters:
            voxel_centers = np.array([self._voxel_center(v) for v in cluster])
            weights = np.array([confirmed[v]["count"] for v in cluster], dtype=np.float64)
            centroid = np.average(voxel_centers, axis=0, weights=weights)
            radius = float(np.max(np.linalg.norm(voxel_centers - centroid, axis=1))) + self.radius_margin_m
            positions.append(centroid)
            radii.append(radius)
            votes.append(int(weights.sum()))
            if return_debug:
                ts_all = [t for v in cluster for t in confirmed[v]["timestamps"] if t is not None]
                r_all = [v for vk in cluster for v in confirmed[vk]["radii"] if v is not None]
                b_all = [v for vk in cluster for v in confirmed[vk]["brightnesses"] if v is not None]
                spreads = [self._spread_stats(vk, np.array(confirmed[vk]["origins"])) for vk in cluster]
                min_bbox_diag = min(s[0] for s in spreads)
                min_angle_deg = min(s[1] for s in spreads)
                ts_px_pairs = [(t, px) for vk in cluster
                               for t, px in zip(confirmed[vk]["timestamps"], confirmed[vk]["blob_px"])
                               if t is not None and px is not None]
                debug_info.append({
                    "min_bbox_diag_m": min_bbox_diag,
                    "min_angle_deg": min_angle_deg,
                    "ts_px_pairs": ts_px_pairs,
                    "n_voxels": len(cluster),
                    "n_votes_total": int(weights.sum()),
                    "ts_min": min(ts_all) if ts_all else None,
                    "ts_max": max(ts_all) if ts_all else None,
                    "n_unique_frames": len(set(ts_all)),
                    "median_radius_px": float(np.median(r_all)) if r_all else None,
                    "median_brightness": float(np.median(b_all)) if b_all else None,
                    "max_radius_px": float(np.max(r_all)) if r_all else None,
                    "max_brightness": float(np.max(b_all)) if b_all else None,
                    "p90_brightness": float(np.percentile(b_all, 90)) if b_all else None,
                })

        result = StaticLightMap(
            voxel_size_m=self.voxel_size_m,
            voxel_room_positions=(np.array(positions, dtype=np.float64).reshape(-1, 3)
                                   if positions else np.zeros((0, 3))),
            voxel_radii_m=np.array(radii, dtype=np.float64),
            vote_counts=np.array(votes, dtype=np.int64),
            camera_calib_id=camera_calib_id,
            created_from_recordings=list(created_from_recordings or []),
        )
        return (result, debug_info) if return_debug else result
