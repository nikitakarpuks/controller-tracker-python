# Controller Tracker — Parameter Reference

See `algorithm_overview.md` for the step-by-step diagrams.

---

## Blob Detection (`blob_detection` in config.yml)

Two-threshold approach: a low threshold casts a wide net to capture faint LED halos; a higher threshold then rejects blobs that never reach a meaningful brightness.

| param | what it controls | config value |
|---|---|---|
| `min_threshold` | pixels ≥ this value are included in a blob | 5 |
| `required_threshold` | blob must contain at least one pixel at or above this level; purely dim regions are dropped | 12 |
| `min_area` | minimum blob area in px² | 4 |
| `max_area` | maximum blob area in px² | 300 |
| `max_wh` | maximum bounding-box width **or** height in px; rejects smeared / non-LED shapes | 35 |

**Merged-blob splitting** (enabled with `split_merged: true`)

Two LEDs that are very close together can appear as one large blob. The splitter finds local intensity maxima inside each blob and, if exactly two survive NMS, splits the blob by a Voronoi partition around those peaks.

| param | what it controls | config value |
|---|---|---|
| `split_merged` | enable the splitting step | true |
| `min_split_dist` | NMS suppression radius in px — maxima closer than this are merged into the stronger one | 5.0 |
| `split_valley_ratio` | `saddle_min / lower_peak` must be **below** this to allow a split; prevents false splits in single bright blobs with a saddle artifact | 0.6 |

---

## Frustum Geometry (`geometry` in config.yml)

Computed once at startup from the controller's LED positions and normals. The outer LEDs are fit to a truncated cone (frustum) used to detect occlusion of inner LEDs by the controller body wall.

| param | what it controls | config value |
|---|---|---|
| `z_frustum_top_padding` | metres added above the topmost outer LED axial position to extend the frustum ceiling | 0.0045 |
| `z_frustum_bot_padding` | metres subtracted below the bottommost outer LED axial position to extend the frustum floor | 0.0055 |
| `wall_thickness` | frustum wall thickness in metres; `R_inner = R_outer − wall_thickness` | 0.007 |

---

## Prior-constrained Match (`matching` in config.yml)

Activated when only 2–3 blobs are visible and a prior pose is available. Rotation is fixed from the predicted pose; only translation is solved.

**P2P mode (3 blobs):** snaps 3 pairs, builds an overdetermined 4×3 linear system from 2 pairs (undistorted normalised coordinates), solves (tx, ty, tz) via least-squares, validates with the 3rd pair.

**P1P mode (2 blobs):** additionally fixes tz from the prior (single-camera depth cannot be recovered from one correspondence alone), solves (tx, ty) analytically from 1 pair, validates with the 2nd. Depth is reliable only for slow radial motion.

All pairs must reproject within `reprojection_threshold` — too few matches to average away a wrong correspondence.

Snap params (`led_radius_mm`, `proximity_snap_factor`, `blob_size_*`) are shared with proximity_match. The `reprojection_threshold` is shared with brute_match.

---

## Proximity Matching (`matching` in config.yml)

Used every frame when a prior pose exists. Projects previous LED positions with the predicted pose and snaps current blobs to them in two passes.

### Snap radius

The snap radius is depth-scaled: `snap_px = focal × (led_radius_mm / 1000) / depth × proximity_snap_factor`. A blob is only a candidate if it is within this radius **and** its pixel size is consistent with the expected LED size at that depth.

| param | what it controls | config value |
|---|---|---|
| `led_radius_mm` | physical LED emitter radius used to compute the expected blob size in pixels | 3.0 |
| `proximity_snap_factor` | multiplier on the expected blob size to get the snap radius | 4.0 |
| `blob_size_min_factor` | blob radius < expected × this → rejected as sub-pixel noise | 0.2 |
| `blob_size_max_factor` | blob radius > expected × this → rejected as merged blob or noise | 4.0 |
| `blob_size_score_weight` | px penalty per px of size mismatch added to the argmin score | 0.5 |
| `proximity_argmin_max_dist_px` | hard distance cap for the greedy pass (ID-path uses the full depth-scaled snap radius) | 5.0 |
| `blob_tracking_snap_px` | max pixel distance to carry a LED ID from the previous frame's blob to the current one | 25.0 |

### Hungarian expansion

Triggered when the snap passes lock fewer pairs than expected, to catch newly-visible LEDs not in the prior assignment.

| param | what it controls | config value |
|---|---|---|
| `proximity_expansion_threshold` | if `locked / model_visible` < this, run Hungarian expansion | 0.7 |
| `proximity_expansion_px` | max reprojection distance to accept a Hungarian-expanded pair | 8.0 |
| `proximity_vis_drop_threshold` | also expand if the number of visible LEDs dropped by more than this from the prior frame | 2 |

### Fallback and pose-jump guard

| param | what it controls | config value |
|---|---|---|
| `use_proximity_match` | set to false to skip proximity and always use brute-force | true |
| `proximity_max_pair_error_px` | if any inlier pair's reprojection error exceeds this, proximity is considered degraded and brute-force runs as a fallback | 1.5 |
| `pose_jump_pos_thresh_m` | per-axis translation jump limit [x, y, z] in metres; candidate rejected if exceeded | [0.25, 0.25, 0.30] |
| `pose_jump_rot_thresh_deg` | per-axis rotation jump limit [rx, ry, rz] in degrees | [30, 30, 30] |

---

## Brute-force Matching (`matching` in config.yml)

### Search space

| param | what it controls | config value |
|---|---|---|
| `depth_tiers` | ordered list of `[led_max, blob_max]` (or `[led_max, blob_max, "edge"]`) specs; controls progressive deepening — see below | see config |
| `led_facing_angle_deg` | LED emission half-angle; LEDs whose normal points more than this away from the camera are culled before any matching | 85.0 |

Each tier `[led_max, blob_max]` controls how far into the neighbourhood graph the search reaches:
- `led_max` — only LED triples whose neighbourhood depth ≤ this are eligible (smaller = tighter, nearer-neighbour hypotheses tried first)
- `blob_max` — only blob pairs within the k=`blob_max` nearest blob neighbours are used
- `"edge"` variant — uses a mixed inner/outer LED neighbourhood for grazing-angle views where both LED ring sides are visible

Each tier evaluates only the *new* combinations not already covered by earlier tiers. Early exit as soon as a strong match is found at the end of any LED triple.

Default tier progression:

| tier | led_max | blob_max | neighbourhood | purpose |
|---|---|---|---|---|
| 0 | 2 | 3 | standard | tightest — nearest LED neighbours + 3 nearest blobs; fastest to exit on a strong match |
| 1 | 2 | 4 | standard | same LED depth, wider blob reach |
| 2 | 2 | 4 | edge | same reach, switch to grazing-view LED pairs (inner + outer LEDs mixed) |
| 3 | 3 | 5 | standard | deeper LED neighbourhood |
| 4 | 3 | 5 | edge | grazing version of tier 3 |
| 5 | 4 | 6 | standard | widest search; last resort |

### Visibility mask — occlusion checks

Applied twice per hypothesis — once before Hungarian assignment (step 4) to limit matching to physically plausible LEDs, and again after RANSAC (step 6) with the refined pose to drop inliers that became occluded under the corrected pose.

Three checks determine whether an LED is visible from the camera:

1. **Depth** — LED must be in front of the camera (z > 0.01 m in camera space). Anything behind the image plane is impossible to see.
2. **Emission cone** — each LED emits into a cone along its surface normal. The camera must lie inside this cone: `dot(normal_in_camera_space, view_direction) < 0`. The `led_facing_angle_deg` param controls the cone half-angle.
3. **Frustum occlusion** (inner LEDs only) — inner LEDs face inward and can be physically blocked by the controller body wall. The wall is modelled as a truncated cone (frustum); a ray from the camera to each inner LED is tested against it.

**Normals also affect the coverage check (step 7).** Each visible LED is weighted by cos(θ), where θ is the angle between the LED normal and the view direction. Grazing LEDs (θ → 90°) contribute less to both numerator and denominator, so they don't unfairly fail the coverage gate when they are simply too dim to detect reliably.

### Validation thresholds

| param | what it controls | config value |
|---|---|---|
| `p4_threshold_px` | gate check radius (px) — a gate LED must project within this of at least one gate blob | 2.0 |
| `hungarian_threshold_px` | loose pre-RANSAC distance filter on Hungarian pairs; intentionally wide because P3P poses are noisy | 5.0 |
| `reprojection_threshold` | RANSAC inlier threshold (px); all errors in the final assignment are ≤ this | 2.5 |
| `min_inliers` | minimum inlier blob–LED pairs to accept a pose | 4 |
| `min_inlier_fraction` | alternative to `min_inliers`: fraction of detected blobs that must be matched (overrides `min_inliers` if set) | null |
| `min_vis_coverage` | minimum weighted fraction of visible LEDs that must be matched; weights are cos(angle) so grazing LEDs count less | 0.7 |

### Early exit

| param | what it controls | config value |
|---|---|---|
| `strong_match_inliers` | inlier count that qualifies as a "strong" match (triggers early exit when combined with error threshold) | 7 |
| `strong_match_error_px` | max mean reprojection error (px) for a strong match | 1.5 |
| `rng_seed` | seed for LED triple shuffling within each tier; ensures reproducibility | 42 |

---

**Distance-based outlier filter** (params currently hardcoded, not exposed in config)

After detection, each blob's mean distance to its `k` nearest neighbours is compared to the median of all such means. Blobs that are spatially isolated (mean > `outlier_factor × median`) are dropped as noise.

| param | what it controls | hardcoded default |
|---|---|---|
| `neighbor_k` | number of nearest neighbours used | 3 |
| `outlier_factor` | isolation threshold relative to median | 3.0 |