"""
Persistent process pool for parallel pose search — cheap per-camera search (proximity
+ prior_constrained) and per-tier brute-force recovery rounds.

Uses the 'spawn' start method, not 'fork'. Fork was tried first and deadlocks in this
codebase: main.py calls rerun's rr.init() before TrackingSystem is constructed, and
rerun's SDK spins up native background threads at init time — forking a process that
already has other threads running risks inheriting a lock (malloc arena, logging, or
inside rerun's own runtime) in a permanently-held state in the child, since the thread
that would release it doesn't exist post-fork. This is a well-known fork+threads
footgun, not specific to this code. 'spawn' starts a clean interpreter per worker with
no inherited thread/lock state, at the (one-time, at pool-startup) cost of each worker
reconstructing its own PoseSearcher registry instead of inheriting it via
copy-on-write.

Workers hold a resident registry of PoseSearcher instances (read-only/stateless after
construction — see PoseSearcher's own docstring), built once per worker from
picklable construction specs sent via the pool's initializer. Only small per-frame
arguments (blobs, priors, BruteSearchState) cross the process boundary on each task
call — never a live CameraTracker, whose mutable tracking state (prev_pose,
pose_history, ...) a worker has no way to keep in sync with anyway.
"""
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Optional, Tuple

import cv2
from loguru import logger

from src.pose_search import PoseSearcher, BruteSearchState

# Main-process-side registry of construction specs, sent to every worker's
# initializer via initargs. Populate with register_pose_searcher_spec() for every
# camera tracker before create_pool().
_BUILD_SPECS: Dict[Tuple[str, int], tuple] = {}

# Worker-side registry, built once per worker process by _pool_initializer.
_POSE_SEARCHERS: Dict[Tuple[str, int], PoseSearcher] = {}


def register_pose_searcher_spec(key: Tuple[str, int], camera, model,
                                 geometry_cfg: Optional[dict], matching_cfg: Optional[dict]) -> None:
    """Register the (picklable) construction args for one camera's PoseSearcher.
    Call for every camera tracker before create_pool() — each worker builds its own
    PoseSearcher from these specs once, at pool startup."""
    _BUILD_SPECS[key] = (camera, model, geometry_cfg, matching_cfg)


def _pool_initializer(build_specs: Dict[Tuple[str, int], tuple], debug_cfg: Optional[dict]) -> None:
    # Avoid OpenCV's own internal threading fighting the process-level parallelism —
    # each worker already IS the unit of parallelism, it shouldn't also fan out.
    cv2.setNumThreads(1)
    for key, (camera, model, geometry_cfg, matching_cfg) in build_specs.items():
        _POSE_SEARCHERS[key] = PoseSearcher(camera, model, geometry_cfg, matching_cfg)

    # A spawned worker starts with fresh module globals — debug_config's mode/
    # verbose flags and loguru's sink/format (both configured once in main.py's
    # main(), which this worker never runs) need to be replicated here, or
    # proximity_search/brute_search_tier's debug logging silently reverts to
    # defaults (wrong format, wrong verbosity, targeted LED/blob debugging inert)
    # even though the actual matching results are unaffected (verified: every
    # debug_config-gated branch in pose_search.py is logging-only, never a
    # decision).
    if debug_cfg is not None:
        from src import debug_config
        from src.debug_config import DebugMode
        from loguru import logger
        debug_config.configure(**debug_cfg)
        logger.remove()
        if debug_cfg['mode'] == DebugMode.SEQUENTIAL:
            logger.add(sys.stderr, level="INFO",
                       format="<green>{time:HH:mm:ss}</green> | {message}")
        else:
            logger.add(sys.stderr, level="DEBUG",
                       format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}")


def create_pool(max_workers: int, debug_cfg: Optional[dict] = None) -> ProcessPoolExecutor:
    """Create the persistent worker pool. Call once at startup, after every camera's
    spec has been registered via register_pose_searcher_spec(). Uses 'spawn' — see
    module docstring for why fork isn't safe here.

    debug_cfg: pass src.debug_config.get_config() from the main process so workers
    replicate the same debug_config state and loguru format (see _pool_initializer)."""
    ctx = multiprocessing.get_context("spawn")
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx,
                                initializer=_pool_initializer, initargs=(_BUILD_SPECS, debug_cfg))


def _warmup_task() -> bool:
    """No-op task. Submitting one per worker right after pool creation forces every
    worker to actually spawn now (spawn's interpreter boot, re-importing numpy/
    scipy/cv2, running _pool_initializer's PoseSearcher construction) instead of
    lazily on the first real frame — ProcessPoolExecutor only starts a worker when
    there's a task for it, so without this the entire one-time startup cost lands on
    whichever frame happens to submit the first task."""
    return True


def warmup_pool(pool: ProcessPoolExecutor, n_workers: int, timeout: Optional[float] = None) -> None:
    """Force every worker in `pool` to spawn and finish its initializer now, rather
    than lazily on the first real task. Call once, right after create_pool()."""
    futures = [pool.submit(_warmup_task) for _ in range(n_workers)]
    for fut in futures:
        fut.result(timeout=timeout)


def run_cheap_search(
    key: Tuple[str, int],
    matching_cfg: dict,
    prior: dict,
    blobs, blob_radii=None, blob_brightnesses=None,
    blob_mask=None, occluders_per_cam=None,
):
    """Top-level, picklable worker task: proximity + prior_constrained only, via
    cheap_search_core (see src/controller.py) — a pure function of the resident
    PoseSearcher and the explicit prior-state bundle, no live CameraTracker needed."""
    # Per-call compute-time logging was removed — the pool round-trip time
    # (submit->result, covers this plus IPC overhead) is already rolled into
    # ControllerTracker.update()'s one-line-per-controller "update timing"
    # summary (src/controller.py), so a separate per-worker line here was
    # pure duplicate noise, one line per camera per frame.
    from src.controller import cheap_search_core
    pose_searcher = _POSE_SEARCHERS[key]
    result = cheap_search_core(
        pose_searcher, prior, matching_cfg,
        blobs, blob_radii, blob_brightnesses,
        other_cameras_blobs=None, blob_mask=blob_mask,
        occluders_per_cam=occluders_per_cam,
    )
    return result


def run_brute_tier(key: Tuple[str, int], state: BruteSearchState, tier_idx: int) -> BruteSearchState:
    """Top-level, picklable worker task: run exactly one depth-tier of a brute-force
    recovery attempt, mutating and returning `state` (round-tripped through pickling,
    since ProcessPoolExecutor tasks aren't pinned to a particular worker)."""
    pose_searcher = _POSE_SEARCHERS[key]
    pose_searcher.brute_search_tier(state, tier_idx)
    return state
