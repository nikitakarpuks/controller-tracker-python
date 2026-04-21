# `_matching.py` — Controller Pose Estimation via LED Matching

This module solves the core computer-vision problem of the tracker: given a set of 2D blob detections from a camera, figure out the 6-DoF pose (position + orientation) of the controller. It does this by matching detected image blobs to known 3D LED positions on the controller model.

---

## Background and Problem Statement

The controller has a "ring" of IR LEDs at known 3D positions. A camera sees their 2D projections as blobs. The goal is to recover the rotation `R` and translation `t` (camera-to-controller transform) that explains which blob corresponds to which LED.

This is a **Perspective-n-Point (PnP)** problem, but with an unknown correspondence — you don't know upfront which blob is which LED. The module solves both the correspondence and the pose jointly.

---

## High-Level Architecture

Two public functions handle the matching, used in different situations:

```
proximity_match()  ←  used every frame when a good prior pose is available
brute_match()      ←  used on first acquisition or after tracking loss
```

All other functions are internal helpers.

---

## Internal Helpers

### `_ransac_pnp(obj_pts, img_pts, K, dc, ...)`

The core pose solver used everywhere. Based on OpenHMD/Monado's approach:

1. **Undistort** image points to normalised camera coordinates (removes distortion and K).
2. Pass an **identity K + zero distortion** to OpenCV's `solvePnPRansac` — this is valid because the points are already in normalised space.
3. Uses `SOLVEPNP_SQPNP` as the minimal solver inside RANSAC (closed-form, more numerically stable than iterative LM inside RANSAC).
4. Converts the pixel reprojection threshold to normalised units as `reprojection_px / fx`.

Returns `(ok, rvec, tvec, inlier_indices)`. Requires at least 4 point pairs.

---

### `_visible_mask(R, tvec, positions, normals, ...)`

Determines which LEDs are physically visible to the camera given a pose hypothesis. Two checks:

**Check 1 — Depth:** LED must be in front of the camera (`z > 0.01` in camera space). Anything behind the image plane is impossible to see.

**Check 2 — Emission cone:** Each LED emits into a 90° half-angle cone along its surface normal. The camera must lie inside this cone. Tested via dot product of the LED's normal (in camera space) with the view direction: `dot < -0.01`.

**Check 3 — Frustum occlusion (inner LEDs only):** Inner LEDs face toward the ring center and can be blocked by the outer conical wall of the controller body. The outer wall is modelled as a truncated cone (frustum). For each inner LED the module casts a ray from the camera to the LED and checks if it intersects the frustum wall at a point strictly between the two (`t ∈ (0,1)`) and within the frustum's axial extent. Intersection is found by solving a quadratic (or a linear fallback when the ray is nearly tangent to the cone).

---

### `_compute_frustum_geometry(positions, normals)`

Called once at tracker initialisation to precompute all geometry needed for the occlusion test. Steps:

1. **Ring axis** — currently hardcoded as `[0, 0, -1]` (SVD-based fitting is commented out).
2. **Radial directions** — project each LED position onto the ring plane, normalise → outward unit vector per LED.
3. **Inner/outer classification** — LEDs whose normal opposes the radial direction are `inner` (facing inward); others are `outer`.
4. **Axis orientation** — flip `ring_axis` so it always points toward the larger-radius base of the frustum.
5. **Frustum cone fit** — least-squares linear fit through the outer LED radial distances vs. axial positions gives `R_fc` (radius at centroid) and `frustum_slope` (dR/dz).
6. **Axial bounds** — `z_frustum_top/bot` are computed from the outer LED z-extents plus a small margin.

Returns a tuple of geometry values stored on the tracker object as `self._*` fields.

---

### `_build_led_neighbor_lists(positions, normals, k=8)`

For each LED, finds up to `k` nearest neighbours (by Euclidean distance) among LEDs whose normal is within 90° of the anchor's normal (dot ≥ 0). The normal filter ensures neighbours can be simultaneously visible — used for standard (non-grazing) views.

### `_build_led_neighbor_lists_edge(positions, normals, is_inner, z_rel, k=8)`

Variant for grazing-angle views (~30° to the frustum base plane) where inner and outer LEDs are both visible at the same time. Each anchor's `k` neighbour slots are split:

- **Up to 2 same-type** neighbours (outer→outer or inner→inner), nearest by distance.
- **Remaining slots: cross-type** neighbours (outer→inner or inner→outer), ranked by normal similarity.

The two groups are interleaved `[cross₀, same₀, cross₁, same₁, …]` so the depth-2 triple always pairs the best cross-type LED with the nearest same-type LED.

### `_build_blob_neighbor_lists(blobs, k)`

For each detected blob, finds the `k` nearest blob neighbours using a KD-tree. Used in `brute_match` to enumerate blob pairs as potential correspondences.

### `_precompute_led_quads(positions, led_nbr, k=8)`

Enumerates all unique LED triples `(anchor, l1, l2)` for use in P3P, with deduplication. For each triple also stores:
- `depth` — the max neighbour rank used (controls the depth-tier search order).
- `gates` — the remaining neighbour LEDs (not in the triple) used as a consistency check.

Deduplication ensures each unique unordered set of 3 LEDs appears exactly once, avoiding redundant P3P calls.

### `_gate_any_point(...)` and `_gate_fourth_point(...)`

Fast consistency checks used to reject bad pose hypotheses before running expensive operations:

- **`_gate_any_point`**: reprojects any gate LED with a simple pinhole projection (no distortion, avoids `cv2.projectPoints` overhead) and returns `True` if *any* gate LED lands near *any* gate blob within `thresh_sq` pixels.
- **`_gate_fourth_point`**: single-point version of the same check.

---

## `proximity_match` — Frame-to-Frame Tracking

Used every frame when a predicted pose is available from a motion model or the previous frame.

### Path 1: Assignment-locked nearest-neighbour

When the previous frame's LED–blob assignment is provided:

1. Project each previously-matched LED using the predicted pose.
2. Snap each projected point to its nearest blob within `max_distance_px`.
3. Run RANSAC PnP on the locked pairs to reject any that drifted to the wrong blob.
4. Re-check visibility with the *refined* pose (not the predicted one) — drops inner LEDs that became occluded.
5. Return if ≥ 3 final inliers with mean reprojection error below threshold.

This is O(N_matches) and deliberately avoids running a full Hungarian expansion, which would introduce spurious pairings that corrupt the prior for the next frame.

### Path 2: Full Hungarian (fallback)

When no prior assignment exists or Path 1 fails:

1. Compute visible LEDs from the predicted pose using `_visible_mask`.
2. Project all visible LEDs.
3. Run Hungarian assignment (`linear_sum_assignment`) between blobs and projections.
4. Filter matches by distance threshold, run RANSAC PnP, return result.

---

## `brute_match` — Pose Acquisition (Exhaustive Search)

Used when no reliable prior pose exists. Implements OpenHMD's `correspondence_search` strategy.

### Loop structure

```
for each LED triple (anchor, l1, l2):          # precomputed, deduplicated
    for each blob as anchor:
        for each pair (b1, b2) from blob neighbours:
            for each ordering (b1,b2) and (b2,b1):
                1. P3P  →  up to 4 pose hypotheses
                2. depth check  (0.05 – 15 m)
                3. gate check   (any gate LED near any gate blob)
                4. visibility mask + Hungarian inlier count
                5. RANSAC PnP refinement
                6. visibility recheck with refined pose
                7. visibility coverage check  (≥ 50% of visible LEDs matched)
                8. update best solution
```

Each unique `(LED triple, blob triple)` bijection is evaluated exactly once.

### Progressive depth-tier search

`depth_tiers` is a tuple of `(led_max, blob_max[, neighbourhood_type])` specs. The search explores LED and blob neighbourhood depth progressively:

- **`led_max`**: only LED triples with depth ≤ led_max are eligible (shallower = closer neighbours = tighter hypotheses).
- **`blob_max`**: only blob pairs with index `i2 < blob_max` are used.
- **`neighbourhood_type`**: `'standard'` uses `_led_triple_idx`; `'edge'` uses `_led_triple_idx_edge` (for grazing views).

Each tier processes only the *new* combinations not covered by prior tiers. The search exits early as soon as a **strong match** is found: ≥ `strong_match_inliers` inliers and mean error ≤ `strong_match_error_px`.

Default tier progression:
```
(led≤2, blob≤3)          # tightest: 2 nearest LED neighbours, 3 nearest blobs
(led≤2, blob≤4)          # expand blob reach
(led≤2, blob≤4, edge)    # same blob reach, grazing-view LED neighbourhood
(led≤3, blob≤5)          # deeper LED neighbours
(led≤3, blob≤5, edge)    # grazing version
(led≤4, blob≤6)          # widest search
```

### Solution ranking

Among all valid hypotheses, the best is chosen by:
1. More inliers + lower per-inlier error.
2. Significantly more inliers (≥+2), even if slightly worse error.
3. Same inliers, lower total error.
4. Same inliers, nearly equal error → prefer solution closer to `pose_prior` orientation.

---

## Return value (both functions)

A dict with:
```python
{
    "rvec":       np.ndarray,        # (3,1) Rodrigues rotation vector
    "tvec":       np.ndarray,        # (3,) or (3,1) translation vector
    "error":      float,             # mean reprojection error in pixels
    "assignment": List[(blob_idx, led_idx)],  # final LED–blob pairs
    "method":     str,               # "proximity_locked" | "proximity" | "p3p_systematic"
}
```
Returns `None` if no valid pose was found.

---

## Data flow summary

```
Camera blobs (2D)
    │
    ├─ proximity_match ──► predicted pose + prior assignment
    │       │                       │
    │       │                   locked nearest-neighbour snap
    │       │                       │
    │       └───────────────────────┴──► RANSAC PnP ──► refined pose
    │
    └─ brute_match ──► P3P over (LED triple × blob triple)
                            │
                        gate check ──► reject bad hypotheses early
                            │
                        Hungarian ──► count inliers
                            │
                        RANSAC PnP ──► refine + recheck visibility
                            │
                        best solution across all tiers
```
