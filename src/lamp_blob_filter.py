"""Stateless, single-frame policy layer that decides which detected
lamp-fixture-like blob candidates are safe to remove from a cold-path blob
set before it reaches brute-force controller matching. No cross-frame state,
no I/O, no camera/controller identity: this module only ever sees one
frame's BlobResult.

Algorithm (rewritten 2026-09-08 -- see below for why the original approach
was replaced): a deterministic, GLOBAL line finder over the combined
kept+dim/context point set (src/blob_detector.py builds this combined set;
see its own comments). For every pair of candidate points spaced
[spacing_min_px, spacing_max_px] apart, fit the infinite line through them,
find every OTHER candidate point within max_line_residual_px of that line
(not just immediate chain neighbors), then keep only the largest contiguous
run (sorted along the line) whose consecutive gaps all stay within
[spacing_min_px, spacing_max_px] -- this is what lets a single real physical
gap of up to spacing_max_px in a lamp row (confirmed on real recordings, see
below) get bridged, while a stray unrelated point far off to one side still
can't extend the line indefinitely. The best (most populated) run across all
seed pairs is accepted if it has >= min_points members; its points are
removed from the pool and the search repeats (up to max_lines times, since
the fixture is typically two nearly-parallel rows, though each row is
independently sufficient -- no pairing requirement).

Two guards, both conjunctive per candidate (either failing spares the WHOLE
candidate -- deliberately biased toward never removing a real controller LED
over removing every real lamp blob):

  - brightness: every member blob's peak pixel value must stay at/below
    max_brightness. Production's real per-frame floor already guarantees
    every blob reaching this filter has brightness >= required_threshold
    (min_threshold * required_threshold_factor, e.g. 7*2.0=14) -- real LEDs
    typically saturate far brighter than that; a genuinely bright blob swept
    into a spurious line match should never be removed.
  - straightness + point count + spacing, all enforced together during line
    finding itself (see below for why this combination, not straightness
    alone, is what actually makes this safe).

Why this replaced the original find_lines/find_fixture_candidates-based
design (src/lamp_structure_detector.py, still used unchanged by the
unrelated offline 3D room-mapping tools in find_lamp_structures.py /
lamp_lattice_fit.py -- this module no longer depends on it):

That greedy nearest-neighbor chain-grower turned out to be fragile exactly
where it mattered most. On real recordings (cam3, frame 2 of this project's
walk_medium recording), the two parallel fixture rows sit close enough
together (~6px apart) that a trailing row-A point's nearest neighbor was
often a row-B point, not its own row-A neighbor -- the chain-grower would
seed a short, useless diagonal "chain" across the two rows, which grew to
only 2 points and got silently discarded, leaving both the real trailing
row-A points AND whatever they should have chained with un-recognized. It
also could not bridge a real physical gap: this project's own lamp1 fixture
has a genuine ~20px gap partway along its row (confirmed via direct
connected-component inspection), well past any spacing_max_px tight enough
to avoid the cross-row confusion above. Retuning spacing_max_px alone could
not fix this without making the cross-row confusion worse.

The deeper problem: chain-growing decides connectivity from LOCAL,
pairwise, greedy decisions, so it's fragile to exactly the kind of close
parallel-line geometry this fixture has. The new approach instead evaluates
every candidate line GLOBALLY (which points are near this line, not just
which point is nearest to the chain's current end), which is what makes
bridging a real gap safe: a stray point 20px away only joins the line if it
is ALSO within max_line_residual_px of the fitted line, not merely within
spacing_max_px of some other point.

Why point count + spacing, not just a tight residual, is the real safety
margin (this is the important change from the original design): earlier
calibration treated max_line_residual_px as the primary safety lever,
tightened to 0.2-0.25px to stay below the ~0.272px bow a real controller-
ring arc can produce at long synthetic radii (6-point arcs, 15-300px
radius sweep) -- but that made the filter nearly useless in practice (only
3/40 of cam3's first 40 frames got fully cleared at 0.2px) because real
lamp-row noise (~0.10-0.41px on real hardware) directly overlaps that same
range; no single tight residual threshold can separate them.

Direct empirical simulation (not guesswork) of every controller-ring pose
resolves this cleanly: projecting all 32 real LEDs (data/controllers/*.json
positions, filtered to front-facing via led_facing_angle_deg) through this
project's own real camera model, across both (a) 100,000 random orientation/
distance samples and (b) a deliberate systematic sweep of ~2500 edge-on/
tangential views (the ring plane forced to contain the view axis -- the
flattest, most lamp-row-like projection possible) at every roll angle and a
range of realistic distances (0.15-1.5m) -- searching every contiguous
subset of the visible, in-frame LEDs for ANY run that mimics real lamp-row
spacing (consecutive gaps all within [4px, 25px], matching spacing_max_px):
  n=4 points: achievable, but only at residual >= 6.6px
  n=5 points: achievable, but only at residual >= 8.8px
  n=6 points: achievable, but only at residual >= 11.96px
  n=7 points: achievable, but only at residual >= 14.7px
  n=8+ points: NEVER achieved with lamp-like spacing, at ANY residual, in
               either sweep
A real controller ring, viewed at ANY angle or distance, naturally produces
either a few widely-spaced points (a shallow arc segment) or many points
bunched very tightly near the silhouette edge (~3px spacing, tighter than
spacing_min_px) -- never a long run of evenly, lamp-spaced (4-25px) points.
This is a structural, physical fact about projecting points off a ~5cm-radius
ring, not a coincidence of this one calibration. Real lamp rows, by
contrast, are confirmed (this project's own recordings) to show 14-15 points
per row spanning 100-150px, with spacing dominated by 5-9px steps and at
most one or two gaps up to ~20-25px -- though any single frame can show
considerably fewer of a row's real elements than that (down to 6-7), simply
from ordinary frame-to-frame brightness/detection noise.

So: min_points=6 and spacing bounds of [4px, 25px] together already provide
a safety margin of several PIXELS (not fractions of a pixel) against every
controller pose tested -- letting max_line_residual_px itself be set far
more generously (real lamp noise reaches ~0.41px; 2.0px is used here, ~5x
that, with enormous room left below the smallest controller-achievable
residual at min_points, 11.96px at n=6 -- a 6x margin over the 2.0px guard).
min_points=8 was tried first and found needlessly conservative: real, single
frames of this project's own recordings repeatedly show a genuine lamp row
at only 6-7 detected points (one point short of 8 purely because a single
non-collinear outlier displaced it from the run, not because the physical
row has fewer elements) -- letting those go un-recognized reopens exactly
the "filter barely does anything" problem this whole design exists to avoid,
for no safety benefit (n=6 and n=7 are both still many pixels away from
being controller-achievable). This is what actually fixes that problem
without reopening the "might remove a real LED" risk -- the risk profile is
now dominated by the count+spacing requirement, which is categorically
different from (not just a looser version of) the old residual-only guard.

See tests/test_lamp_blob_filter.py for the regression tests encoding both
the controller-pose simulation's boundary (no controller configuration
should ever satisfy min_points+spacing+residual together) and real-lamp-like
synthetic rows (must be removed, including across a large physical gap).
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.blob_detector import BlobResult


@dataclass
class LampLine:
    blob_indices: List[int]
    residual_px: float


@dataclass
class FixtureCandidate:
    """`lines` always holds exactly one LampLine -- kept as a list (not a
    single field) so the brightness/straightness guard functions below stay
    written generically over "every line in the candidate", matching how
    src/blob_detector.py's visualization code reads `.blob_indices`."""
    lines: List[LampLine]
    blob_indices: List[int]
    centroid_px: np.ndarray


@dataclass
class LampFilterResult:
    keep_mask: np.ndarray                                  # (N,) bool, True = keep (matches BlobResult.filter's convention)
    rejected_candidates: List[FixtureCandidate] = field(default_factory=list)   # actually removed
    guarded_candidates: List[FixtureCandidate] = field(default_factory=list)    # structurally lamp-like, guard-vetoed (kept)
    skipped_pool_too_large: int = 0   # 0 = ran normally; else the pool size that triggered max_pool_size (see detect_lamp_blobs)


def _find_dominant_lines(centroids: np.ndarray, area_ok: np.ndarray,
                          spacing_min_px: float, spacing_max_px: float,
                          max_line_residual_px: float, min_points: int,
                          max_lines: int) -> List[LampLine]:
    """Deterministic, global line finder -- see this module's docstring for
    the full design rationale. Not a chain-grower: every seed pair's line is
    scored by ALL points near it, not just its immediate neighbors, which is
    what lets a real physical gap (up to spacing_max_px) get bridged safely.

    Performance note: the original implementation evaluated each seed pair
    in a pure-Python loop, one numpy call at a time -- fine for a handful of
    points, but its cost scales with the number of seed pairs (up to
    O(n^2)), and on real, denser recordings this measured up to ~250ms for a
    single frame (profiled 2026-09-08). The residual/along-line projection
    for EVERY seed pair against EVERY pool point is now computed in one
    batched (S, N) numpy operation (S = seed pairs, N = pool size) instead
    of S separate small numpy calls; only seeds that already have enough
    inliers (a cheap vectorized filter) reach the remaining sort + gap-run
    step.

    SECOND perf pass (2026-09-09): that remaining step was STILL a sequential
    per-seed Python loop (one np.argsort + np.diff + a manual gap scan per
    candidate seed) -- fine when packbits deduplication collapses S down to a
    handful of truly distinct seeds, but on a real frame with min_threshold
    lowered (letting far more faint/noise points through, e.g. a ceiling
    fixture becoming much more visible), packbits dedup barely helps: with
    ~500-700 mostly-noisy points, most seed pairs produce SIMILAR but not
    EXACTLY duplicate inlier sets, leaving thousands of "distinct" seeds post-
    dedup -- measured 3000-4700 remaining, each paying its own Python-loop
    overhead, 800ms-3s per cold-detect call on a real cam3 frame (n_pool up
    to 669). Now batched: the longest-contiguous-valid-gap-run search for
    EVERY remaining candidate seed is computed together via one column-wise
    vectorized pass (O(N) numpy ops over ALL seeds at once, not O(S) Python-
    level calls) -- same math, same seed population, same gap/spacing rules,
    just evaluated as one batch instead of thousands of tiny sequential ones.
    Only the single WINNING seed's exact run (start/end indices, not just its
    length) is then re-derived via the original sequential per-seed logic, so
    the actual points returned are computed by the exact same code path as
    before, just for one seed instead of thousands. Verified byte-identical
    output (same lines, same blob_indices, same residual_px) against the
    pre-rewrite implementation across 10 real calls captured from this
    project's own cam3 frames (both the ~500-700-point dense case and the
    smaller pass-2 pools) -- see tests/test_lamp_blob_filter.py's own
    PerfRewriteEquivalenceTests."""
    pool = np.where(area_ok)[0]
    lines: List[LampLine] = []

    for _ in range(max_lines):
        if len(pool) < min_points:
            break
        pts = centroids[pool]
        n = len(pool)
        diffs = pts[:, None, :] - pts[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        seed_i, seed_j = np.where((dists >= spacing_min_px) & (dists <= spacing_max_px))
        seed_mask = seed_i < seed_j
        seed_i, seed_j = seed_i[seed_mask], seed_j[seed_mask]

        best_run = None  # local indices into `pool`
        if len(seed_i):
            directions = pts[seed_j] - pts[seed_i]                       # (S, 2)
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            normals = np.stack([-directions[:, 1], directions[:, 0]], axis=1)  # (S, 2)

            rel = pts[None, :, :] - pts[seed_i][:, None, :]              # (S, N, 2)
            resid = np.abs(np.einsum('snd,sd->sn', rel, normals))        # (S, N)
            along = np.einsum('snd,sd->sn', rel, directions)             # (S, N)

            inlier_mask = resid <= max_line_residual_px                  # (S, N)
            inlier_counts = inlier_mask.sum(axis=1)
            candidate_seeds = np.where(inlier_counts >= min_points)[0]

            # Any two points sampled from the SAME real line tend to recover
            # nearly the same inlier set -- for a real N-point line, most of
            # the O(N^2) seed pairs on it are redundant, each independently
            # paying the sort+gap-run cost below for what is often the exact
            # same result. Collapsing to one representative per distinct
            # inlier set is what actually fixes this: profiled on a real
            # dense frame, this cut ~7000 redundant per-seed iterations down
            # to a handful. np.unique directly on the (M, N) boolean rows
            # was tried first and made things WORSE (sorting N-wide rows is
            # itself not cheap) -- packing each row into ceil(N/8) bytes
            # first, then deduping THOSE, is the actual win.
            if len(candidate_seeds) > 1:
                packed = np.packbits(inlier_mask[candidate_seeds], axis=1)
                _, unique_idx = np.unique(packed, axis=0, return_index=True)
                candidate_seeds = candidate_seeds[unique_idx]

            # Remaining duplicates are near- but not exactly identical (one
            # boundary point differs), so the packbits dedup above can't
            # collapse them -- on a dense/noisy real pool this can still
            # leave thousands of "distinct" seeds, MANY of which genuinely
            # tie on final run length (several independent seed pairs
            # rediscovering the same physical line through the noise).
            # Sort by raw inlier count descending BEFORE batching -- not for
            # correctness of the max itself, but so a tie in final run length
            # resolves the same way the original sequential loop would (it
            # only overwrote best_run on a STRICT length increase, so among
            # ties the FIRST seed in this same descending-raw-count order
            # always won; matching that order here, then taking argmax below
            # -- which also returns the first occurrence on a tie -- makes
            # the winning seed, and therefore the exact points returned,
            # identical to the pre-batching implementation instead of merely
            # "an equally valid" alternative pick).
            count_order = np.argsort(-inlier_counts[candidate_seeds])
            candidate_seeds = candidate_seeds[count_order]

            # Find the best run length for EVERY remaining seed in one
            # batched pass (see this function's own docstring, "SECOND perf
            # pass") instead of a sequential Python loop over each one.
            along_sub = along[candidate_seeds]                       # (S2, N)
            inlier_sub = inlier_mask[candidate_seeds]                 # (S2, N)
            # Push non-inliers to a sentinel far past any real pixel
            # coordinate so they sort to the end and never form a spurious
            # "good gap" with each other (sentinel-sentinel gap is exactly 0,
            # which correctly fails the gap >= spacing_min_px check as long
            # as spacing_min_px > 0) or with a real inlier (sentinel-real gap
            # is enormous, correctly failing gap <= spacing_max_px).
            _SENTINEL = 1.0e9
            along_masked = np.where(inlier_sub, along_sub, _SENTINEL)
            sort_idx = np.argsort(along_masked, axis=1)               # (S2, N) local pool indices
            sorted_along_sub = np.take_along_axis(along_masked, sort_idx, axis=1)
            gaps_sub = np.diff(sorted_along_sub, axis=1)              # (S2, N-1)
            good_gap = (gaps_sub >= spacing_min_px) & (gaps_sub <= spacing_max_px)

            # Longest run of consecutive True per row (per seed), via a
            # column-wise cumulative-reset pass -- O(N) vectorized steps over
            # ALL seeds at once, equivalent to (but far cheaper than) finding
            # each seed's longest valid-gap run independently.
            run_len = np.zeros(good_gap.shape[0], dtype=np.int64)
            _cur = np.zeros(good_gap.shape[0], dtype=np.int64)
            for col in range(good_gap.shape[1]):
                _cur = np.where(good_gap[:, col], _cur + 1, 0)
                np.maximum(run_len, _cur, out=run_len)
            best_point_count = run_len + 1  # a lone inlier (0 valid gaps) is still a run of 1

            ok = best_point_count >= min_points
            if np.any(ok):
                ok_idx = np.flatnonzero(ok)
                s = candidate_seeds[ok_idx[np.argmax(best_point_count[ok_idx])]]
                # Re-derive the exact run (start/end indices, not just its
                # length) for this one winning seed only, via the original
                # sequential logic -- keeps the actual extracted points
                # byte-identical to the pre-batching implementation.
                inlier_local = np.where(inlier_mask[s])[0]
                along_vals = along[s, inlier_local]
                order = np.argsort(along_vals)
                sorted_local = inlier_local[order]
                sorted_along = along_vals[order]
                gaps = np.diff(sorted_along)

                best_this_seed = (0, 1) if len(sorted_local) else (0, 0)
                cur_start = 0
                for gi, gap in enumerate(gaps):
                    if gap < spacing_min_px or gap > spacing_max_px:
                        if (gi + 1 - cur_start) > (best_this_seed[1] - best_this_seed[0]):
                            best_this_seed = (cur_start, gi + 1)
                        cur_start = gi + 1
                if (len(sorted_local) - cur_start) > (best_this_seed[1] - best_this_seed[0]):
                    best_this_seed = (cur_start, len(sorted_local))

                best_run = sorted_local[best_this_seed[0]:best_this_seed[1]]

        if best_run is None:
            break

        global_indices = [int(pool[k]) for k in best_run]
        final_pts = centroids[global_indices]
        centered = final_pts - final_pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centered)
        direction = vt[0]
        normal = np.array([-direction[1], direction[0]])
        residual_px = float(np.max(np.abs(centered @ normal)))

        lines.append(LampLine(blob_indices=global_indices, residual_px=residual_px))
        pool = np.setdiff1d(pool, np.asarray(global_indices, dtype=pool.dtype), assume_unique=True)

    return lines


def _brightness_ok(candidate: FixtureCandidate, blobs: BlobResult, max_brightness: float) -> bool:
    return bool(np.all(blobs.brightnesses[candidate.blob_indices] <= max_brightness))


def restrict_dim_context_to_kept_neighborhood(
    dim_centroids: list, dim_contours: list, dim_max_pixels: list,
    kept_centroids: np.ndarray, radius_px: float,
    dim_extent_bounds: list = None,
) -> tuple:
    """Keep only "dim" context blobs (see detect_lamp_blobs' own docstring on
    why they exist at all: recognizing a real fixture row that's mostly dim
    but has SOME brighter, KEPT members nearby) within radius_px of at least
    one KEPT blob. A dim point with no kept blob anywhere near it was never
    going to matter: only kept-indexed points can ever actually be removed
    (see this function's caller in src/blob_detector.py -- kept_keep_mask
    only reads indices < n_kept), and the other thing a found line produces,
    a pass-2 exclusion zone, still exists to protect an already-recognized
    REAL structure, not an isolated patch of dim noise with no known
    detection anywhere near it.

    Found 2026-09-09 investigating a real performance blowup: lowering
    min_threshold let far more faint pixels through as "dim" blobs --
    one real cam3 frame had 8 kept blobs but 429 dim ones, the vast
    majority scattered across the WHOLE frame with no spatial relationship
    to any actual detection. _find_dominant_lines' cost grows roughly
    cubically with pool size (see its own docstring), so feeding it 437
    mostly-irrelevant points instead of a spatially-restricted handful
    turned a sub-millisecond search into multiple seconds.

    radius_px default (lamp_blob_filter.dim_context_radius_px, 150.0)
    matches this project's own documented real-fixture-row span (100-150px,
    see this module's docstring) -- generous enough that no genuine same-row
    dim member should ever be excluded just because the nearest kept member
    happens to sit at the row's opposite end.

    Returns (dim_centroids, dim_contours, dim_max_pixels) restricted to the
    surviving indices in their original relative order, or the inputs
    unchanged if there are no dim points or no kept anchor at all (an
    anchor-less dim point's relevance can't be decided by this rule --
    existing behavior preserved for that edge case). If `dim_extent_bounds`
    is given (src/blob_detector.py's own per-entry "real full extent of a
    split-peak's original blob" list, see its own comment), it's filtered
    in sync and returned as a 4th element; omitted entirely (3-tuple, this
    function's original contract) when not given."""
    if not dim_centroids or len(kept_centroids) == 0:
        if dim_extent_bounds is not None:
            return dim_centroids, dim_contours, dim_max_pixels, dim_extent_bounds
        return dim_centroids, dim_contours, dim_max_pixels
    dim_arr = np.asarray(dim_centroids, dtype=np.float64)
    kept_arr = np.asarray(kept_centroids, dtype=np.float64)
    # (D, K) pairwise distances -- D (dim count) is exactly the population
    # this function exists to shrink, and K (kept count) is normally small,
    # so this itself stays cheap (the O(pool^3)-ish cost lives entirely in
    # _find_dominant_lines downstream, not here).
    d2 = ((dim_arr[:, None, :] - kept_arr[None, :, :]) ** 2).sum(axis=2)
    near = np.sqrt(d2.min(axis=1)) <= radius_px
    keep_idx = np.where(near)[0]
    if dim_extent_bounds is not None:
        return ([dim_centroids[i] for i in keep_idx],
                [dim_contours[i] for i in keep_idx],
                [dim_max_pixels[i] for i in keep_idx],
                [dim_extent_bounds[i] for i in keep_idx])
    return ([dim_centroids[i] for i in keep_idx],
            [dim_contours[i] for i in keep_idx],
            [dim_max_pixels[i] for i in keep_idx])


def detect_lamp_blobs(blobs: BlobResult, cfg: dict) -> LampFilterResult:
    """Stateless, single-frame lamp-fixture blob filter. Pure function: no
    I/O, no cross-frame state, no camera/controller identity -- callers own
    all of that. Never raises on an empty BlobResult.

    cfg is the `lamp_blob_filter` sub-block of blob_detection config. See
    this module's docstring for what each tunable means and how its default
    was derived."""
    n = len(blobs)
    keep_mask = np.ones(n, dtype=bool)
    if n == 0:
        return LampFilterResult(keep_mask=keep_mask)

    area_min = float(cfg.get("area_min", 0.4))
    area_max = float(cfg.get("area_max", 15.0))
    spacing_min_px = float(cfg.get("spacing_min_px", 4.0))
    spacing_max_px = float(cfg.get("spacing_max_px", 25.0))
    min_points = int(cfg.get("min_points", 6))
    max_lines = int(cfg.get("max_lines", 20))
    max_brightness = float(cfg.get("max_brightness", 60.0))
    max_line_residual_px = float(cfg.get("max_line_residual_px", 2.0))

    areas = np.array(
        [float(np.pi * r * r) for r in blobs.radii],
        dtype=np.float64,
    ) if len(blobs.radii) else np.empty(0, dtype=np.float64)
    area_ok = (areas >= area_min) & (areas <= area_max)

    # Circuit breaker, not a tuning knob: _find_dominant_lines' cost grows
    # roughly cubically with pool size (every seed pair tested against every
    # pool point, then a batched per-seed run search over all surviving
    # seeds -- see that function's own docstring) because a real lamp row
    # legitimately needs ALL points tested against every candidate line, not
    # just spatially nearby ones. This project's own recordings normally put
    # this pool at 6-15 points (a real single-row/frame lamp population,
    # confirmed on real cam3 frames) -- max_pool_size=150 leaves generous
    # headroom above that (measured ~55ms at 150 points on a real dense
    # frame) while decisively catching the degenerate case: min_threshold
    # lowered enough to flood this pool with noise (measured 400-700 points,
    # 0.8-3.3s per call, entirely outside the population this filter's own
    # empirical validation -- 100,000 synthetic controller-ring poses plus a
    # systematic edge-on sweep -- was ever run against). Skipping in that
    # case is the SAFE failure mode (matches this filter's own stated bias:
    # never wrongly removing a real LED beats always catching every lamp
    # blob) -- an un-filtered lamp blob still has to survive brute-force
    # matching's own geometric consistency checks downstream, it doesn't
    # silently corrupt tracking.
    #
    # Applied PER SPATIALLY-CONNECTED CLUSTER, not globally over the whole
    # candidate pool -- found 2026-09-10 on a real cam0 frame: the frame's
    # pool was 299 points, but 237 of those were one unrelated dense blob
    # elsewhere in the frame (nowhere near lamp-row spacing); the OLD global
    # check (n_pool=299 > 150) skipped the ENTIRE frame's line search,
    # silently letting a completely clean, isolated 8-point lamp row in the
    # frame's opposite corner go unrecognized. A real lamp row's own points
    # are, by construction, chain-connected within spacing_max_px of each
    # other (that's what lets _find_dominant_lines bridge a real physical
    # gap at all), so clustering on that same radius can never split a
    # genuine row across two clusters -- it only ever separates points that
    # could never have been mistaken for the same row in the first place.
    # Two unrelated structures within spacing_max_px of each other still land
    # in one cluster together (harmless: _find_dominant_lines already has to
    # pick out the actual line from within whatever pool it's given, cluster
    # or not), but a small, cleanly isolated fixture is no longer held
    # hostage by noise volume somewhere else in the same frame. Building the
    # radius graph itself is a cheap KDTree query, not the expensive part --
    # the cubic cost this breaker actually guards against lives entirely
    # inside _find_dominant_lines, and giving it many small per-cluster pools
    # instead of one huge combined one only ever makes that cheaper.
    max_pool_size = int(cfg.get("max_pool_size", 150))
    pool_all = np.where(area_ok)[0]
    centroids64 = blobs.centroids.astype(np.float64)

    if len(pool_all) == 0:
        labels = np.empty(0, dtype=np.int64)
        n_clusters = 0
    elif len(pool_all) == 1:
        labels = np.zeros(1, dtype=np.int64)
        n_clusters = 1
    else:
        tree = cKDTree(centroids64[pool_all])
        pairs = tree.query_pairs(r=spacing_max_px, output_type="ndarray")
        n_local = len(pool_all)
        if len(pairs):
            graph = coo_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])),
                                shape=(n_local, n_local))
            n_clusters, labels = connected_components(graph, directed=False)
        else:
            n_clusters, labels = n_local, np.arange(n_local)

    found_lines: List[LampLine] = []
    n_pool_skipped = 0
    for comp in range(n_clusters):
        cluster_local = pool_all[labels == comp]
        if len(cluster_local) < min_points:
            continue  # can never satisfy min_points -- cheap skip, not a safety decision
        if max_pool_size > 0 and len(cluster_local) > max_pool_size:
            n_pool_skipped += len(cluster_local)
            continue
        cluster_mask = np.zeros(n, dtype=bool)
        cluster_mask[cluster_local] = True
        found_lines.extend(_find_dominant_lines(
            centroids64, cluster_mask,
            spacing_min_px, spacing_max_px, max_line_residual_px, min_points, max_lines,
        ))

    rejected, guarded = [], []
    for line in found_lines:
        candidate = FixtureCandidate(
            lines=[line], blob_indices=line.blob_indices,
            centroid_px=blobs.centroids[line.blob_indices].mean(axis=0),
        )
        if _brightness_ok(candidate, blobs, max_brightness):
            keep_mask[candidate.blob_indices] = False
            rejected.append(candidate)
        else:
            guarded.append(candidate)

    return LampFilterResult(keep_mask=keep_mask, rejected_candidates=rejected,
                             guarded_candidates=guarded, skipped_pool_too_large=n_pool_skipped)
