"""
Persistent process pool for parallel pose search — cheap per-camera search (proximity
+ prior_constrained) and per-tier brute-force recovery rounds — and for parallel blob
detection across cameras (both the warm per-LED local-search path and the cold
full-image 2-pass path).

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
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Optional, Tuple

import cv2

from src.pose_search import PoseSearcher, BruteSearchState

# Main-process-side registry of construction specs, sent to every worker's
# initializer via initargs. Populate with register_pose_searcher_spec() for every
# camera tracker before create_pool().
_BUILD_SPECS: Dict[Tuple[str, int], tuple] = {}

# Same idea for blob detection: {cam_idx: blob_detection_cfg}, populated via
# register_blob_detector_spec() before create_pool().
_BLOB_BUILD_SPECS: Dict[int, dict] = {}

# Worker-side registries, built once per worker process by _pool_initializer.
_POSE_SEARCHERS: Dict[Tuple[str, int], PoseSearcher] = {}
_BLOB_DETECTORS: Dict[int, "BlobDetector"] = {}


def register_pose_searcher_spec(key: Tuple[str, int], camera, model,
                                 geometry_cfg: Optional[dict], matching_cfg: Optional[dict]) -> None:
    """Register the (picklable) construction args for one camera's PoseSearcher.
    Call for every camera tracker before create_pool() — each worker builds its own
    PoseSearcher from these specs once, at pool startup."""
    _BUILD_SPECS[key] = (camera, model, geometry_cfg, matching_cfg)


def register_blob_detector_spec(cam_idx: int, blob_detection_cfg: dict) -> None:
    """Register the (picklable) construction args for one camera's BlobDetector.
    Call for every camera before create_pool() — each worker builds its own
    BlobDetector from these specs once, at pool startup."""
    _BLOB_BUILD_SPECS[cam_idx] = blob_detection_cfg


def _pool_initializer(build_specs: Dict[Tuple[str, int], tuple],
                       blob_build_specs: Dict[int, dict],
                       debug_cfg: Optional[dict]) -> None:
    # Avoid OpenCV's own internal threading fighting the process-level parallelism —
    # each worker already IS the unit of parallelism, it shouldn't also fan out.
    cv2.setNumThreads(1)
    for key, (camera, model, geometry_cfg, matching_cfg) in build_specs.items():
        _POSE_SEARCHERS[key] = PoseSearcher(camera, model, geometry_cfg, matching_cfg)

    from src.blob_detector import BlobDetector
    for cam_idx, blob_detection_cfg in blob_build_specs.items():
        _BLOB_DETECTORS[cam_idx] = BlobDetector(cam_idx, blob_detection_cfg)

    # A spawned worker starts with fresh module globals — debug_config's
    # continuous-frames/verbose/log-category flags and loguru's sink (both
    # configured once in main.py's main(), which this worker never runs) need
    # to be replicated here, or proximity_search/brute_search_tier's debug
    # logging silently reverts to defaults (wrong verbosity, targeted LED/blob
    # debugging inert) even though the actual matching results are unaffected
    # (verified: every debug_config-gated branch in pose_search.py is
    # logging-only, never a decision).
    if debug_cfg is not None:
        from src import debug_config
        debug_config.configure(**debug_cfg)
        debug_config.setup_logging()


def create_pool(max_workers: int, debug_cfg: Optional[dict] = None) -> ProcessPoolExecutor:
    """Create the persistent worker pool. Call once at startup, after every camera's
    specs have been registered via register_pose_searcher_spec() and
    register_blob_detector_spec() — initargs captures both registries' contents at
    this call's time, not lazily. Uses 'spawn' — see module docstring for why fork
    isn't safe here.

    debug_cfg: pass src.debug_config.get_config() from the main process so workers
    replicate the same debug_config state and loguru format (see _pool_initializer)."""
    ctx = multiprocessing.get_context("spawn")
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx,
                                initializer=_pool_initializer,
                                initargs=(_BUILD_SPECS, _BLOB_BUILD_SPECS, debug_cfg))


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
    blobs, frame_ts_ns: int, blob_radii=None, blob_brightnesses=None,
    blob_mask=None, occluders_per_cam=None,
):
    """Top-level, picklable worker task: proximity + prior_constrained only, via
    cheap_search_core (see src/controller.py) — a pure function of the resident
    PoseSearcher and the explicit prior-state bundle, no live CameraTracker needed.

    frame_ts_ns: the current frame's real capture timestamp (nanoseconds) —
    forwarded to cheap_search_core for time-aware pose extrapolation."""
    # Per-call compute-time logging was removed — the pool round-trip time
    # (submit->result, covers this plus IPC overhead) is already rolled into
    # ControllerTracker.update()'s one-line-per-controller "update timing"
    # summary (src/controller.py), so a separate per-worker line here was
    # pure duplicate noise, one line per camera per frame.
    from src.controller import cheap_search_core
    pose_searcher = _POSE_SEARCHERS[key]
    result = cheap_search_core(
        pose_searcher, prior, matching_cfg,
        blobs, frame_ts_ns, blob_radii, blob_brightnesses,
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


def run_blob_detect(
    cam_idx: int,
    ctrl_label: str,
    image,
    predicted_leds=None,
    local_search_radius_px: float = 0.0,
    threshold_scale: float = 1.0,
    velocity_px: float = 0.0,
    visualize: bool = False,
    img_path=None,
    frame_name: Optional[str] = None,
    memory_in: Optional[dict] = None,
):
    """Top-level, picklable worker task: run BlobDetector.detect() for one camera
    against the resident (per-camera) BlobDetector.

    memory_in/return value's 3rd element: BlobDetector's only cross-call mutable
    state (`_memory`, cold-path EMA-blended detection thresholds) is round-tripped
    explicitly rather than left as worker-resident state, same reasoning as
    BruteSearchState above — a call for a given (camera, controller) pair isn't
    pinned to the same worker each time, so continuity has to travel with the data.
    The caller keys its own memory cache by (cam_idx, ctrl_name) rather than just
    cam_idx, so two controllers cold-starting on the same camera in the same frame
    no longer clobber each other's EMA state (a pre-existing quirk when this memory
    lived solely on a single shared per-camera BlobDetector instance).

    Returns (BlobResult, canvases_dict, memory_out).
    """
    bd = _BLOB_DETECTORS[cam_idx]
    bd._memory = memory_in if memory_in is not None else {}
    result, canvases = bd.detect(
        image,
        ctrl_label=ctrl_label,
        predicted_leds=predicted_leds,
        local_search_radius_px=local_search_radius_px,
        threshold_scale=threshold_scale,
        velocity_px=velocity_px,
        visualize=visualize,
        img_path=img_path,
        frame_name=frame_name,
    )
    return result, canvases, bd._memory
