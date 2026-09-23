"""Per-frame survey blob-set computation -- Phase 3 (map-building time only)
of the mocap-based static-light (ceiling-lamp) exclusion feature.

Every frame is processed independently: no cross-frame blob identity is
tracked or needed here, which is the direct fix for the earlier-diagnosed
problem that individual lamp LEDs flicker between detected/dim and can't be
reliably followed frame to frame. `detect_survey` on its own is stateless
(see src/blob_detector.py's BlobDetector.detect_survey); the "removed
claimed-LED" step below is likewise recomputed fresh every call.
"""
import numpy as np

from src.blob_detector import BlobDetector, BlobResult


def compute_survey_blob_set(detector: BlobDetector, image: np.ndarray,
                             blob_detection_cfg: dict, static_light_cfg: dict,
                             claimed_positions_px) -> BlobResult:
    """One permissive full-image survey detection call, minus any blob within
    `claimed_blob_exclusion_px` of a position in `claimed_positions_px`
    (this frame's warm controllers' claimed LED pixels, from whatever
    source the caller has -- e.g. a replayed led_detections_csv row set, or
    TrackingSystem.get_predicted_led_projections_per_camera's proj_hints).

    pixel_threshold (the segmentation floor that decides which pixels are
    even connected into candidate blobs) is deliberately left EQUAL to
    production's own `min_threshold`, never lowered -- confirmed on real
    data (a low-light "walk_medium" recording, cam3) that lowering it
    instead collapses the whole frame into one giant connected blob once the
    threshold drops anywhere near the ambient noise floor, which gets
    rejected as too-large and yields ZERO usable blobs, the opposite of
    "permissive". Only `required_threshold` (the "at least one pixel must
    reach this peak" brightness gate) is relaxed, via
    `static_light_cfg["survey_required_threshold_factor"]` -- this is what
    actually recovers blobs production's dim-rejection would drop, without
    touching segmentation/connectivity at all.
    """
    min_threshold_base = float(blob_detection_cfg.get("min_threshold", 7))
    pixel_threshold = min_threshold_base
    required_factor = float(static_light_cfg.get("survey_required_threshold_factor", 1.0))
    required_threshold = required_factor * min_threshold_base
    min_area = float(static_light_cfg.get("dim_blob_min_area_px", 2.0))
    claimed_exclusion_px = float(static_light_cfg.get("claimed_blob_exclusion_px", 6.0))

    blobs = detector.detect_survey(image, pixel_threshold, required_threshold, min_area)
    if len(blobs) == 0:
        return blobs

    claimed = np.asarray(claimed_positions_px, dtype=np.float64).reshape(-1, 2) \
        if len(claimed_positions_px) else np.zeros((0, 2))
    if len(claimed) == 0:
        return blobs

    dists = np.linalg.norm(
        blobs.centroids[:, None, :].astype(np.float64) - claimed[None, :, :], axis=2)
    keep = dists.min(axis=1) > claimed_exclusion_px
    return blobs.filter(keep)
