from enum import Enum
from typing import List, Optional, Tuple


class DebugMode(Enum):
    SEQUENTIAL = "sequential"   # large sequential repo: minimal logs, copy slow/lost frames
    DEEP       = "deep"         # isolated debug frames: verbose matching logs, no prior pose


_mode:           DebugMode          = DebugMode.SEQUENTIAL
_debug_led_ids:  Optional[List[int]] = None   # target LED triple [anchor, l1, l2]
_debug_blob_ids: Optional[List[int]] = None   # target blob triple [b_anchor, b1, b2]
_verbose_all:    bool                = False  # log every P3P hypothesis, not just the target triple
_log_best:       bool                = True   # log each time a new best solution is found


def configure(
    mode: DebugMode,
    debug_led_ids:  Optional[List[int]] = None,
    debug_blob_ids: Optional[List[int]] = None,
    verbose_all:    bool = False,
    log_best:       bool = True,
) -> None:
    global _mode, _debug_led_ids, _debug_blob_ids, _verbose_all, _log_best
    _mode           = mode
    _debug_led_ids  = debug_led_ids
    _debug_blob_ids = debug_blob_ids
    _verbose_all    = verbose_all
    _log_best       = log_best


def is_deep() -> bool:
    return _mode == DebugMode.DEEP


def is_verbose_all() -> bool:
    """True → log every P3P hypothesis in brute_match (ignores LED/blob filter)."""
    return _verbose_all


def log_best() -> bool:
    """True → log each new best solution update in brute_match."""
    return _log_best


def get_debug_triple() -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """Return (debug_led_ids, debug_blob_ids) for targeted triple logging in brute_match."""
    return _debug_led_ids, _debug_blob_ids
