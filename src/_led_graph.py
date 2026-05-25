import numpy as np
from itertools import combinations
from typing import List, Tuple

from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from src.debug_config import is_deep


def _build_led_neighbor_lists(positions: np.ndarray, normals: np.ndarray, k: int = 8) -> List[np.ndarray]:
    """
    For each LED (anchor): among LEDs whose normal is within 90° of the anchor's normal
    (dot product >= 0), return up to k nearest by Euclidean distance.

    Normal filter first — only LEDs facing roughly the same direction are candidates,
    ensuring they can be simultaneously visible.  Spatial sort second — closest
    normal-compatible LEDs form the tightest, most discriminative hypotheses.
    """
    n = len(positions)
    k_act = min(k, n - 1)
    dists = cdist(positions, positions)   # (N, N)
    dots  = normals @ normals.T           # (N, N) pairwise normal cosines

    result = []
    for i in range(n):
        valid = dots[i] >= 0.0
        valid[i] = False
        candidates = np.where(valid)[0]
        if len(candidates) == 0:
            result.append(np.array([], dtype=int))
            continue
        order = np.argsort(dists[i, candidates])
        result.append(candidates[order[:k_act]])
    return result


def _build_led_neighbor_lists_edge(
    positions: np.ndarray,
    normals: np.ndarray,
    is_inner: np.ndarray,
    z_rel: np.ndarray,
    k: int = 8,
) -> List[np.ndarray]:
    """
    Alternative neighbourhood for grazing-angle views (~30° to the frustum base plane)
    where both inner and outer LEDs are simultaneously visible.

    For each anchor LED the k neighbours are filled with a strict split:
      - n_same  = min(k // 2, 2): at most 2 nearest same-type LEDs (outer→outer or
                                   inner→inner) with dot >= 0, sorted by distance.
      - n_cross = k - n_same     : cross-type LEDs (outer→inner or inner→outer)
                                   with dot >= 0, sorted by normal similarity descending.

    The two halves are interleaved with cross-type first:
      cross[0], same[0], cross[1], same[1], …
    so rank 0 is always the best cross-type match (the grazing-view target), rank 1 the
    nearest same-type, and deeper ranks continue alternating.  Depth-2 triple (0,1)
    therefore pairs the best cross LED with the nearest same LED.

    debug_led_ids: if provided, print the chosen neighbours for each listed LED id.
    """
    n       = len(positions)
    n_same  = min(k // 2, 2)
    n_cross = k - n_same

    dists = cdist(positions, positions)   # (N,N) Euclidean distances
    dots  = normals @ normals.T           # (N,N) pairwise normal cosines

    result = []
    for i in range(n):
        # ── same-type: nearest by distance, dot >= 0, capped at 2 ────────────────
        same_valid = (is_inner == is_inner[i]) & (dots[i] >= 0.0)
        same_valid[i] = False
        same_cands = np.where(same_valid)[0]
        if len(same_cands):
            same_nbrs = same_cands[np.argsort(dists[i, same_cands])[:n_same]]
        else:
            same_nbrs = np.array([], dtype=int)

        # ── cross-type: most normal-similar, dot >= 0 ────────────────────────────
        cross_valid = (is_inner != is_inner[i]) & (dots[i] >= 0.0)
        cross_cands = np.where(cross_valid)[0]
        if len(cross_cands):
            cross_nbrs = cross_cands[np.argsort(-dots[i, cross_cands])[:n_cross]]
        else:
            cross_nbrs = np.array([], dtype=int)

        # ── interleave: cross[0], same[0], cross[1], same[1], … ─────────────────
        nbrs = []
        for slot in range(max(len(same_nbrs), len(cross_nbrs))):
            if slot < len(cross_nbrs):
                nbrs.append(cross_nbrs[slot])
            if slot < len(same_nbrs):
                nbrs.append(same_nbrs[slot])
        nbrs = np.array(nbrs, dtype=int)
        result.append(nbrs)

        if is_deep():
            kind = "inner" if is_inner[i] else "outer"
            lines = [f"LED {i:2d} ({kind}, z_rel={z_rel[i]:+.5f})  →  neighbours:"]
            for rank, j in enumerate(nbrs):
                jkind = "inner" if is_inner[j] else "outer"
                src   = "same " if is_inner[j] == is_inner[i] else "cross"
                lines.append(f"  rank {rank}: LED {j:2d} ({jkind}/{src}, "
                              f"z_rel={z_rel[j]:+.5f}, dot={dots[i,j]:.4f}, "
                              f"dist={dists[i,j]*1000:.1f} mm)")
            # logger.debug("\n".join(lines))

    return result


def _build_blob_neighbor_lists(blobs: np.ndarray, k: int) -> List[np.ndarray]:
    """
    For each blob: indices of up to k nearest blob neighbours, excluding self.
    """
    n = len(blobs)
    if n <= 1:
        return [np.array([], dtype=int) for _ in range(n)]
    tree  = KDTree(blobs)
    k_act = min(k, n - 1)
    _, idx = tree.query(blobs, k=k_act + 1)
    return [row[1:] for row in idx]


def _precompute_led_quads(
    positions: np.ndarray, led_nbr: List[np.ndarray], k: int = 8
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Enumerate LED triples for P3P. Each (anchor, l1, l2) ordered triple appears once per
    anchor — i.e., the same three LEDs may appear under different anchors because each
    anchor defines a distinct P3P geometry (different gate pool, different depth rank).
    Duplicate bijection-level P3P calls are deduplicated later in brute_match.

    Returns
    -------
    triple_idx   : (N, 3) int32      — [anchor, l1, l2]; all three used for P3P
    triple_depth : (N,) int32        — max neighbour rank used (1-based); shallow/deep split
    triple_gates : List[np.ndarray]  — for each triple, remaining neighbour LED indices (gate pool)
    """
    idx_rows:   List[Tuple]       = []
    depth_rows: List[int]         = []
    gate_rows:  List[np.ndarray]  = []
    seen_per_anchor: set          = set()   # dedup only within the same anchor's C(k,2)
    n = len(positions)
    for anchor in range(n):
        nbrs   = led_nbr[anchor][:k]
        nb_len = len(nbrs)
        if nb_len < 2:
            continue
        for i1, i2 in combinations(range(nb_len), 2):
            l1, l2 = int(nbrs[i1]), int(nbrs[i2])
            key = (anchor, min(l1, l2), max(l1, l2))
            if key in seen_per_anchor:
                continue
            seen_per_anchor.add(key)
            depth  = i2 + 1
            gates  = np.array(
                [int(nbrs[j]) for j in range(nb_len) if j != i1 and j != i2],
                dtype=np.int32,
            )
            idx_rows.append((anchor, l1, l2))
            depth_rows.append(depth)
            gate_rows.append(gates)
    if not idx_rows:
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            [],
        )
    return (
        np.array(idx_rows,   dtype=np.int32),
        np.array(depth_rows, dtype=np.int32),
        gate_rows,
    )
