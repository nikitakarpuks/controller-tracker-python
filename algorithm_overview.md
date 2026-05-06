# Algorithm Overview

```mermaid
flowchart TD
    A["<font color='#e07a00'>Load config</font><br/>YAML + camera intrinsics + controller JSON"] --> B["<font color='#e07a00'>Setup</font><br/>Camera · ControllerModel · TrackingSystem"]
    B --> C["<font color='#e07a00'>Prepare visualization geometry</font><br/>LED positions · normals · 3D model"]
    C --> D([For each frame])

    D --> E["<font color='#e07a00'>Blob detection</font><br/>find LED centroids · contours · radii in image"]
    E --> F["<font color='#e07a00'>TrackingSystem.update</font><br/>assign blobs to LEDs · estimate pose"]
    F --> G{Pose found?}

    G -- yes --> H["<font color='#e07a00'>Store pose &amp; assignment</font><br/>log reprojection error"]
    G -- no  --> I["<font color='#e07a00'>Mark tracking lost</font><br/>copy frame to debug folder"]

    H --> D
    I --> D

    D -- all frames done --> J{Any valid pose?}
    J -- no  --> K([Error: no poses in sequence])
    J -- yes --> L["<font color='#e07a00'>Visualize with Rerun</font><br/>animate poses · blobs · 3D model overlay"]
```

---

## Blob Detection

```mermaid
flowchart TD
    A[Input: grayscale image] --> B["<font color='#e07a00'>Threshold at min_threshold</font><br/>binary mask"]
    B --> C["<font color='#e07a00'>Find contours</font><br/>connected components"]
    C --> D["<font color='#e07a00'>Filter contours</font><br/>area · bounding box size · required_threshold"]
    D --> E["<font color='#e07a00'>Intensity-weighted centroid per blob</font><br/>1-based greysum"]

    E --> G["<font color='#e07a00'>Find local maxima per blob</font><br/>NMS with min_split_dist"]
    G --> H{"<font color='#e07a00'>Merged blob recovery</font><br/>Exactly 2 maximas for a \nsingle blob and valley ratio ok?"}
    H -- yes --> I[Voronoi split → 2 centroids]
    H -- no  --> J[Keep blob as-is]
    I --> K["<font color='#e07a00'>Distance outlier filter</font><br/>kNN mean vs. factor * median"]
    J --> K
    K --> L["<font color='#e07a00'>Compute equivalent radius per blob</font><br/>radius = sqrt area / pi"]
    L --> M[Return: centroids · contours · radii]
```

---

## Tracker Initialization (`SingleViewTracker.__init__`)

Run once at startup from the fixed controller model. All results are cached for the lifetime of the tracker.

```mermaid
flowchart TD
    A["<font color='#e07a00'>Controller model</font><br/>LED positions + normals"] --> B["<font color='#e07a00'>Fit frustum geometry</font><br/>ring axis · inner/outer LED classification<br/>cone radius + slope · axial bounds"]
    A --> C["<font color='#e07a00'>Build LED neighbor lists</font><br/>standard: same-side neighbours<br/>edge: mixed inner/outer for grazing views"]
    C --> D["<font color='#e07a00'>Precompute LED triples</font><br/>deduplicated anchor + support LED 1 and 2 (l1, l2)\n with gate LEDs<br/>for each neighbourhood type"]
    B --> E["<font color='#e07a00'>SingleViewTracker ready</font><br/>geometry + LED graph cached"]
    D --> E
```

## Per-frame Tracking State Machine (`track`)

Decides which solver to call based on available state.

```mermaid
flowchart TD
    A[New frame: blobs + radii] --> B{prev_pose?}

    B -- yes --> C["<font color='#e07a00'>Predict pose</font><br/>constant-velocity extrapolation"]
    C --> D["<font color='#e07a00'>Carry blob LED IDs</font><br/>nearest-neighbour from previous frame"]
    D --> E{n_blobs ≥ 3?}
    E -- yes --> F[proximity_match]
    E -- no  --> G
    F --> G{"<font color='#e07a00'>Proximity ok</font><br/>and max_pair_error ≤ threshold?"}
    G -- yes         --> J
    G -- no + n≥4   --> H["<font color='#e07a00'>brute_match</font><br/>with predicted pose as prior"]
    H --> J{Best candidate}

    B -- no --> I{n_blobs ≥ 4?}
    I -- yes --> K["<font color='#e07a00'>brute_match</font><br/>cold start"]
    I -- no  --> LOST([Tracking lost])
    K --> J

    J -- none + 2-3 blobs + prev_pose --> L["<font color='#e07a00'>prior_constrained_match</font><br/>fix R · solve t only<br/>P2P for 3 blobs · P1P for 2 blobs"]
    L --> M
    J --> M{"<font color='#e07a00'>Pose jump</font><br/>too large vs prev_pose?"}
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
    A["<font color='#e07a00'>Setup</font><br/>undistort blobs once · build blob kNN lists"] --> B

    B(["<font color='#e07a00'>For each tier</font><br/>led_max · blob_max · neighbourhood"])
    B --> C["<font color='#e07a00'>Select eligible LED triples</font><br/>depth ≤ led_max · has new blob pairs this tier"]
    C --> D([For each LED triple × blob anchor × blob pair × ordering])

    D --> E["<font color='#e07a00'>P3P</font><br/>up to 4 pose hypotheses from 3 point pairs"]
    E --> F{"<font color='#e07a00'>Depth in</font><br/>0.05 – 15 m?"}
    F -- no  --> D
    F -- yes --> G{"<font color='#e07a00'>Gate check</font><br/>any gate LED projects near any gate blob?"}
    G -- no  --> D
    G -- yes --> H["<font color='#e07a00'>Visibility mask</font><br/>filter out occluded and back-facing LEDs"]
    H --> I["<font color='#e07a00'>Hungarian assignment</font><br/>blobs to visible LEDs"]
    I --> J{"<font color='#e07a00'>Enough inliers</font><br/>before RANSAC?"}
    J -- no  --> D
    J -- yes --> K["<font color='#e07a00'>RANSAC PnP</font><br/>refine pose · reject outlier pairs"]
    K --> L{"<font color='#e07a00'>RANSAC ok</font><br/>+ enough inliers?"}
    L -- no  --> D
    L -- yes --> M["<font color='#e07a00'>Visibility recheck</font><br/>drop inliers occluded under refined pose"]
    M --> N["<font color='#e07a00'>Post-RANSAC blob recovery</font><br/>greedy nearest-neighbour for missed blobs"]
    N --> O{"<font color='#e07a00'>Coverage check</font><br/>weighted matched / visible ≥ min_vis_coverage?"}
    O -- no  --> D
    O -- yes --> P["<font color='#e07a00'>Update best solution</font><br/>rank by inliers · error · prior distance"]
    P --> Q{"<font color='#e07a00'>Strong match?</font><br/>inliers ≥ strong_match_inliers<br/>error ≤ strong_match_error_px"}
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
    A["<font color='#e07a00'>Project prior LEDs</font><br/>with predicted pose"] --> B
    B["<font color='#e07a00'>Compute visibility mask</font><br/>with predicted pose"] --> C

    C["<font color='#e07a00'>Snap pass 1 — ID fast path</font><br/>blobs carrying a known LED ID matched directly<br/>subject to size + distance checks"] --> D

    D["<font color='#e07a00'>Snap pass 2 — greedy blob→LED</font><br/>each unmatched blob → nearest unclaimed LED<br/>score = dist + size_weight × size_err"] --> E

    E{"<font color='#e07a00'>Coverage low?</font><br/>locked / visible < expansion_threshold<br/>or visible LED count dropped?"}
    E -- yes --> F["<font color='#e07a00'>Hungarian expansion</font><br/>assign free blobs to newly-visible LEDs"]
    E -- no  --> G
    F --> G["<font color='#e07a00'>RANSAC PnP</font><br/>refine pose on all locked pairs"]
    G --> H{"<font color='#e07a00'>RANSAC ok</font><br/>+ inliers ≥ min_inliers?"}
    H -- no  --> I([Return None])
    H -- yes --> J[Return pose · assignment]
```

---

## Prior-constrained Match (`prior_constrained_match`)

Fallback when only 2–3 blobs are visible — too few for P3P or RANSAC. Fixes rotation from the predicted pose as a hard constraint and collapses the problem to translation-only.

```mermaid
flowchart TD
    A["<font color='#e07a00'>Snap prior LEDs to nearest blobs</font><br/>depth-scaled radius · size filter · argmin score"] --> B{"<font color='#e07a00'>Enough pairs</font><br/>snapped?"}
    B -- no  --> NONE([Return None])
    B -- yes --> C{n_blobs?}

    C -- 3 → P2P --> D["<font color='#e07a00'>Undistort 2 hypothesis blobs</font><br/>normalised image coords"]
    D --> E["<font color='#e07a00'>Build 4×3 linear system</font><br/>one row-pair per correspondence<br/>fix R · solve tx ty tz via least-squares"]
    E --> F["<font color='#e07a00'>Depth sanity check</font><br/>0.05 – 15 m"]
    F -- fail --> NONE
    F -- ok  --> G["<font color='#e07a00'>Validate all 3 pairs</font><br/>reprojection error ≤ threshold?"]
    G -- fail --> NONE
    G -- ok  --> H([Return pose · assignment])

    C -- 2 → P1P --> I["<font color='#e07a00'>Undistort 1 hypothesis blob</font><br/>normalised image coords"]
    I --> J["<font color='#e07a00'>Fix tz from predicted pose</font><br/>solve tx ty analytically"]
    J --> F
```
