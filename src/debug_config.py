import sys
from typing import Dict, List, Optional, Tuple

from loguru import logger


# Log category flags — see config.yml's debug: section for what each one covers.
# Every category defaults to False: with every one left off, nothing but the tqdm
# bar prints (warnings/errors are the one exception — those always print, see
# _log_filter).
_LOG_CATEGORIES = (
    "startup", "frame_summary", "timings", "blob_detection", "blob_diag",
    "matching_decisions", "batch_orchestration", "pose_fusion", "occlusion",
    "proximity_match", "hypothesis_testing", "ransac", "self_calibration",
)

# True = input is a real continuous capture (frame N+1 is ~one exposure after
# frame N) → the pose-continuity jump check on re-acquisition applies. False =
# input is a curated/isolated set of problem frames with no temporal relation to
# each other (see config.yml's assume_continuous_frames) → that check is skipped.
# The only remaining behavioral (non-logging) flag in this module.
_continuous_frames: bool           = True
_debug_led_ids:  Optional[List[int]] = None   # target LED triple [anchor, l1, l2]
_debug_blob_ids: Optional[List[int]] = None   # target blob triple [b_anchor, b1, b2]
_log_all_triples: bool               = False  # log every P3P hypothesis triple, not just the target one
_log_all_proximity_hyps: bool        = False  # log every proximity-match candidate hypothesis tried
_log_best:       bool                = True   # log each time a new best solution is found
_log_categories: Dict[str, bool]     = {c: False for c in _LOG_CATEGORIES}


def configure(
    continuous_frames: bool = True,
    debug_led_ids:  Optional[List[int]] = None,
    debug_blob_ids: Optional[List[int]] = None,
    log_all_triples: bool = False,
    log_all_proximity_hyps: bool = False,
    log_best:       bool = True,
    log_categories: Optional[Dict[str, bool]] = None,
) -> None:
    global _continuous_frames, _debug_led_ids, _debug_blob_ids, _log_all_triples, \
        _log_all_proximity_hyps, _log_best, _log_categories
    _continuous_frames = continuous_frames
    _debug_led_ids      = debug_led_ids
    _debug_blob_ids     = debug_blob_ids
    _log_all_triples    = log_all_triples
    _log_all_proximity_hyps = log_all_proximity_hyps
    _log_best           = log_best
    if log_categories is not None:
        _log_categories = {c: bool(log_categories.get(c, False)) for c in _LOG_CATEGORIES}


def is_continuous_sequence() -> bool:
    """True if input frames should be treated as a real temporal sequence — gates
    the pose-continuity jump check on re-acquisition (controller.py). False for a
    curated/isolated set of problem frames, where consecutive frames aren't
    actually temporally adjacent and that check would misfire."""
    return _continuous_frames


def log_all_triples() -> bool:
    """True → log every P3P hypothesis triple in brute_match (ignores LED/blob filter)."""
    return _log_all_triples


def log_all_proximity_hyps() -> bool:
    """True → log every candidate hypothesis tried in proximity_match's None-branching
    search (each combo's score), not just the level/cap/early-stop/final-accept summary
    lines that log_proximity_match alone shows."""
    return _log_all_proximity_hyps


def log_best() -> bool:
    """True → log each new best solution update in brute_match."""
    return _log_best


def get_debug_triple() -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """Return (debug_led_ids, debug_blob_ids) for targeted triple logging in brute_match."""
    return _debug_led_ids, _debug_blob_ids


def log_enabled(category: str) -> bool:
    """True if this log category's console output is currently enabled (see config.yml
    debug: section). Call sites bind their category via logger.bind(cat=category) and
    rely on the filter installed by setup_logging() — this getter exists for the few
    spots (e.g. the shared per-hypothesis `dbg` trigger in brute_search_tier) that need
    to branch on category state directly rather than just emitting a bound log call."""
    return _log_categories.get(category, False)


def get_config() -> dict:
    """Return the current configuration as kwargs for configure() — used to propagate
    this process's debug_config state into a spawned worker process, which starts
    with fresh (default) module globals otherwise."""
    return {
        'continuous_frames': _continuous_frames,
        'debug_led_ids':     _debug_led_ids,
        'debug_blob_ids':    _debug_blob_ids,
        'log_all_triples':   _log_all_triples,
        'log_all_proximity_hyps': _log_all_proximity_hyps,
        'log_best':          _log_best,
        'log_categories':    dict(_log_categories),
    }


def _log_filter(record) -> bool:
    # Warnings/errors always print — they signal genuine anomalies, not routine
    # verbosity, so no category can suppress them.
    if record["level"].no >= logger.level("WARNING").no:
        return True
    cat = record["extra"].get("cat")
    if cat == "_gated":
        # PoseSearcher._dbg's caller already resolved category-enabled + triple
        # match into the `active` bool before ever calling _dbg — nothing left
        # for this filter to check.
        return True
    if cat is None:
        return False
    return _log_categories.get(cat, False)


def setup_logging() -> None:
    """(Re)configure loguru's sink. Always emits at TRACE level — per-category flags
    (see _log_filter) decide what actually gets through, not the sink's level, so the
    debug: config controls verbosity independent of is_continuous_sequence() (which
    only affects tracking behavior). Call after configure() so the filter sees current
    category state immediately. Safe to call again (e.g. in a spawned worker) —
    logger.remove() clears any prior sink first."""
    logger.remove()
    fmt = "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}"
    logger.add(sys.stderr, level="TRACE", format=fmt, filter=_log_filter)
