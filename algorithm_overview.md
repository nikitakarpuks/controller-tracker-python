# Algorithm Overview

```mermaid
flowchart TD
    A[Load config\nYAML + camera intrinsics + controller JSON] --> B[Setup\nCamera · ControllerModel · TrackingSystem]
    B --> C[Prepare visualization geometry\nLED positions · normals · 3D model]
    C --> D([For each frame])

    D --> E[Blob detection\nfind LED centroids · contours · radii in image]
    E --> F[TrackingSystem.update\nassign blobs to LEDs · estimate pose]
    F --> G{Pose found?}

    G -- yes --> H[Store pose & assignment\nlog reprojection error]
    G -- no  --> I[Mark tracking lost\ncopy frame to debug folder]

    H --> D
    I --> D

    D -- all frames done --> J{Any valid pose?}
    J -- no  --> K([Error: no poses in sequence])
    J -- yes --> L[Visualize with Rerun\nanimate poses · blobs · 3D model overlay]
```

---

## Blob Detection

```mermaid
flowchart TD
    A[Input: grayscale image] --> B[Threshold at min_threshold\nbinary mask]
    B --> C[Find contours\nconnected components]
    C --> D[Filter contours\narea · bounding box size · required_threshold]
    D --> E[Intensity-weighted centroid per blob\n1-based greysum]
    E --> F{split_merged?}

    F -- yes --> G[Find local maxima per blob\nNMS with min_split_dist]
    G --> H{Exactly 2 maxima\nand valley ratio ok?}
    H -- yes --> I[Voronoi split → 2 centroids]
    H -- no  --> J[Keep blob as-is]
    I --> K
    J --> K

    F -- no --> K[Distance outlier filter\nkNN mean vs. median]
    K --> L[Compute equivalent radius per blob\nradius = sqrt area / pi]
    L --> M[Return: centroids · contours · radii]
```

---

## Tracker Initialization (`SingleViewTracker.__init__`)

Run once at startup from the fixed controller model. All results are cached for the lifetime of the tracker.

```mermaid
flowchart TD
    A[Controller model\nLED positions + normals] --> B[Fit frustum geometry\nring axis · inner/outer LED classification\ncone radius + slope · axial bounds]
    A --> C[Build LED neighbor lists\nstandard: same-side neighbours\nedge: mixed inner/outer for grazing views]
    C --> D[Precompute LED triples\ndeduplicated anchor + l1 + l2 with gate LEDs\nfor each neighbourhood type]
    B --> E[SingleViewTracker ready\ngeometry + LED graph cached]
    D --> E
```

**Tracker state** — maintained across frames, cleared on tracking loss:

| field | what it holds | cleared on loss? |
|---|---|---|
| `prev_pose` | rvec + tvec from the last accepted frame | yes |
| `prev_prev_pose` | frame before that; used for velocity extrapolation | yes |
| `prev_assignment` | LED–blob pairs from the last accepted frame | yes |
| `last_good_pose` | last accepted pose; **kept across loss** for re-acquisition plausibility | no |
| `prev_blob_positions` | blob pixel positions from previous frame | yes |
| `prev_blob_led_ids` | LED ID carried by each previous blob; feeds the ID fast path in proximity_match | yes |

**Constant-velocity pose prediction** — when two consecutive prior poses exist, the next pose is extrapolated before passing to any solver:
- Translation: `t_{n+1} = 2·t_n − t_{n−1}`
- Rotation: `R_{n+1} = (R_n · R_{n−1}ᵀ) · R_n` (apply the same delta rotation again)

---

## Per-frame Tracking State Machine (`track`)

Decides which solver to call based on available state.

```mermaid
flowchart TD
    A[New frame: blobs + radii] --> B{prev_pose?}

    B -- yes --> C[Predict pose\nconstant-velocity extrapolation]
    C --> D[Carry blob LED IDs\nnearest-neighbour from previous frame]
    D --> E{n_blobs ≥ 3?}
    E -- yes --> F[proximity_match]
    E -- no  --> G
    F --> G{Proximity ok\nand max_pair_error ≤ threshold?}
    G -- yes         --> J
    G -- no + n≥4   --> H[brute_match\nwith predicted pose as prior]
    H --> J{Best candidate}

    B -- no --> I{n_blobs ≥ 4?}
    I -- yes --> K[brute_match\ncold start]
    I -- no  --> LOST([Tracking lost])
    K --> J

    J -- none + 2-3 blobs + prev_pose --> L[prior_constrained_match\nfix R · solve t only\nP2P for 3 blobs · P1P for 2 blobs]
    L --> M
    J --> M{Pose jump\ntoo large vs prev_pose?}
    M -- yes --> N[brute recovery attempt]
    N --> O
    M -- no  --> O{error < 5 px?}
    O -- yes --> P[Accept · store pose · assignment · blob IDs]
    O -- no  --> LOST
```

---

## Brute-force Matching (`brute_match`)

Used on first acquisition or after tracking loss — no prior pose available.

```mermaid
flowchart TD
    A[Setup\nundistort blobs once · build blob kNN lists] --> B

    B([For each tier\nled_max · blob_max · neighbourhood])
    B --> C[Select eligible LED triples\ndepth ≤ led_max · has new blob pairs this tier]
    C --> D([For each LED triple × blob anchor × blob pair × ordering])

    D --> E[P3P\nup to 4 pose hypotheses from 3 point pairs]
    E --> F{Depth in\n0.05 – 15 m?}
    F -- no  --> D
    F -- yes --> G{Gate check\nany gate LED projects near any gate blob?}
    G -- no  --> D
    G -- yes --> H[Visibility mask\ncull occluded and back-facing LEDs]
    H --> I[Hungarian assignment\nblobs to visible LEDs]
    I --> J{Enough inliers\nbefore RANSAC?}
    J -- no  --> D
    J -- yes --> K[RANSAC PnP\nrefine pose · reject outlier pairs]
    K --> L{RANSAC ok\n+ enough inliers?}
    L -- no  --> D
    L -- yes --> M[Visibility recheck\ndrop inliers occluded under refined pose]
    M --> N[Post-RANSAC blob recovery\ngreedy nearest-neighbour for missed blobs]
    N --> O{Coverage check\nweighted matched / visible ≥ min_vis_coverage?}
    O -- no  --> D
    O -- yes --> P[Update best solution\nrank by inliers · error · prior distance]
    P --> Q{Strong match?\ninliers ≥ strong_match_inliers\nerror ≤ strong_match_error_px}
    Q -- yes --> R([Return best solution])
    Q -- no  --> D

    D -- tier exhausted --> B
    B -- all tiers done --> R
```


---

## Proximity Match (`proximity_match`)

Fast path used every frame when a prior pose is available. Projects previous LEDs forward with the predicted pose, then snaps current blobs to them.

```mermaid
flowchart TD
    A[Project prior LEDs\nwith predicted pose] --> B
    B[Compute visibility mask\nwith predicted pose] --> C

    C[Snap pass 1 — ID fast path\nblobs carrying a known LED ID matched directly\nsubject to size + distance checks] --> D

    D[Snap pass 2 — greedy blob→LED\neach unmatched blob → nearest unclaimed LED\nscore = dist + size_weight × size_err] --> E

    E{Coverage low?\nlocked / visible < expansion_threshold\nor visible LED count dropped?}
    E -- yes --> F[Hungarian expansion\nassign free blobs to newly-visible LEDs]
    E -- no  --> G
    F --> G[RANSAC PnP\nrefine pose on all locked pairs]
    G --> H{RANSAC ok\n+ inliers ≥ min_inliers?}
    H -- no  --> I([Return None])
    H -- yes --> J[Return pose · assignment]
```

---

## Prior-constrained Match (`prior_constrained_match`)

Fallback when only 2–3 blobs are visible — too few for P3P or RANSAC. Fixes rotation from the predicted pose as a hard constraint and collapses the problem to translation-only.

```mermaid
flowchart TD
    A[Snap prior LEDs to nearest blobs\ndepth-scaled radius · size filter · argmin score] --> B{Enough pairs\nsnapped?}
    B -- no  --> NONE([Return None])
    B -- yes --> C{n_blobs?}

    C -- 3 → P2P --> D[Undistort 2 hypothesis blobs\nnormalised image coords]
    D --> E[Build 4×3 linear system\none row-pair per correspondence\nfix R · solve tx ty tz via least-squares]
    E --> F[Depth sanity check\n0.05 – 15 m]
    F -- fail --> NONE
    F -- ok  --> G[Validate all 3 pairs\nreprojection error ≤ threshold?]
    G -- fail --> NONE
    G -- ok  --> H([Return pose · assignment])

    C -- 2 → P1P --> I[Undistort 1 hypothesis blob\nnormalised image coords]
    I --> J[Fix tz from predicted pose\nsolve tx ty analytically]
    J --> F
```