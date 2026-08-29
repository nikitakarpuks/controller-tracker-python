"""
Instrumented real-pipeline run against real data (no source files modified,
config.yml untouched -- everything below is monkeypatched in-process):

  1. Tallies the brute-search funnel (tier_p3p_calls -> gate -> vis ->
     hungarian -> n_reached_ransac), already logged by main.py/pose_search.py
     at DEBUG level, so we know the real ratio between "P3P calls" (the
     bottleneck previously discussed) and calls that actually reach
     _ransac_pnp (the only call the SQPNP->AP3P flag experiment touches).

  2. For every real _ransac_pnp call made during the run, additionally solves
     the identical (obj_pts, img_pts, rvec_init, tvec_init) with
     SOLVEPNP_AP3P instead of the production SOLVEPNP_SQPNP, purely for
     comparison -- the wrapper always returns the original SQPNP result to
     the caller unchanged, so production control flow (and therefore the
     funnel counts from #1) is exactly what a real run would produce, not
     perturbed by the comparison.

Forces matching.parallel_search_enabled=False for this run only (in-memory
config override) so pose search runs in-process -- required for the
monkeypatch in #2 to see every call; parallel_search normally dispatches
brute-force tiers to a worker pool, which would bypass it.

Run: python benchmarks/run_instrumented.py [n_frames]
"""
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 150
DATA_ROOT_OVERRIDE = sys.argv[2] if len(sys.argv) > 2 else None

import src.load_config as load_config_mod

_orig_load_yaml = load_config_mod.load_yaml_config


def _patched_load_yaml(path):
    cfg = _orig_load_yaml(path)
    if str(path).endswith("config.yml"):
        cfg["matching"]["parallel_search_enabled"] = False
        cfg["debug"]["assume_continuous_frames"] = False
        cfg["debug"]["split_to_folders"] = False
        cfg["debug"]["calibration_csv"] = None
        cfg["visualization"]["enabled"] = False
        if DATA_ROOT_OVERRIDE:
            cfg["data"]["root"] = str(ROOT / DATA_ROOT_OVERRIDE)
    return cfg


load_config_mod.load_yaml_config = _patched_load_yaml

import main as main_mod

main_mod.load_yaml_config = _patched_load_yaml

_orig_get_data = main_mod.get_data
main_mod.get_data = lambda cfg: itertools.islice(_orig_get_data(cfg), N_FRAMES)
main_mod.count_images = lambda cfg: N_FRAMES

# ---------------------------------------------------------------------------
# Capture every real _ransac_pnp call made by src.pose_search.
# ---------------------------------------------------------------------------
import src.pose_search as pose_search_mod
from src._pnp import _ransac_pnp as _orig_ransac_pnp

_K_IDENTITY = np.eye(3, dtype=np.float64)
_DC_ZERO = np.zeros(4, dtype=np.float64)

comparisons = []


def _ransac_pnp_compare(obj_pts, img_pts, K, dc, rvec_init=None, tvec_init=None,
                         reprojection_px=2.0, iterations=100, confidence=0.99,
                         is_fisheye=False):
    result = _orig_ransac_pnp(obj_pts, img_pts, K, dc, rvec_init, tvec_init,
                               reprojection_px, iterations, confidence, is_fisheye)
    ok, rvec, tvec, inliers = result

    entry = {"n_pts": len(obj_pts)}
    try:
        fx = float(K[0, 0])
        inp = img_pts.astype(np.float32).reshape(-1, 1, 2)
        if is_fisheye:
            pts_norm = cv2.fisheye.undistortPoints(
                inp.astype(np.float64), K.astype(np.float64), dc).reshape(-1, 2)
        else:
            pts_norm = cv2.undistortPoints(inp, K, dc).reshape(-1, 2)
        use_guess = rvec_init is not None and tvec_init is not None
        r0 = np.asarray(rvec_init, dtype=np.float64).reshape(3, 1) if use_guess else None
        t0 = np.asarray(tvec_init, dtype=np.float64).reshape(3, 1) if use_guess else None
        ret2, rvec2, tvec2, inl2 = cv2.solvePnPRansac(
            obj_pts.astype(np.float64), pts_norm.astype(np.float64), _K_IDENTITY, _DC_ZERO,
            r0, t0, useExtrinsicGuess=use_guess, iterationsCount=iterations,
            reprojectionError=reprojection_px / fx, confidence=confidence,
            flags=cv2.SOLVEPNP_AP3P,
        )
        ok2 = bool(ret2 and inl2 is not None and len(inl2) >= 4)
    except cv2.error:
        ok2, rvec2, tvec2, inl2 = False, None, None, None

    entry["ok_sqpnp"] = bool(ok)
    entry["ok_ap3p"] = ok2
    if ok and ok2:
        R1, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        R2, _ = cv2.Rodrigues(np.asarray(rvec2, dtype=np.float64))
        cos_ang = np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)
        entry["rot_diff_deg"] = float(np.degrees(np.arccos(cos_ang)))
        entry["trans_diff_m"] = float(np.linalg.norm(
            np.asarray(tvec, dtype=np.float64).reshape(3) - np.asarray(tvec2, dtype=np.float64).reshape(3)))
        entry["n_inliers_sqpnp"] = int(len(inliers))
        entry["n_inliers_ap3p"] = int(len(inl2))
    comparisons.append(entry)
    return result


pose_search_mod._ransac_pnp = _ransac_pnp_compare

# ---------------------------------------------------------------------------
# Run the real pipeline
# ---------------------------------------------------------------------------
main_mod.main()

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("\n\n=== _ransac_pnp call comparison: SQPNP (production, unchanged) vs AP3P, identical inputs ===")
print(f"total _ransac_pnp calls this run: {len(comparisons)}")
if comparisons:
    both_ok = [c for c in comparisons if c["ok_sqpnp"] and c["ok_ap3p"]]
    only_sqpnp = [c for c in comparisons if c["ok_sqpnp"] and not c["ok_ap3p"]]
    only_ap3p = [c for c in comparisons if c["ok_ap3p"] and not c["ok_sqpnp"]]
    neither = [c for c in comparisons if not c["ok_sqpnp"] and not c["ok_ap3p"]]
    print(f"  both succeeded        : {len(both_ok)}")
    print(f"  SQPNP ok, AP3P failed : {len(only_sqpnp)}")
    print(f"  AP3P ok, SQPNP failed : {len(only_ap3p)}")
    print(f"  both failed           : {len(neither)}")
    if both_ok:
        rot = np.array([c["rot_diff_deg"] for c in both_ok])
        trans = np.array([c["trans_diff_m"] for c in both_ok]) * 1000.0
        inl_diff = np.array([c["n_inliers_ap3p"] - c["n_inliers_sqpnp"] for c in both_ok])
        print(f"  rotation diff (deg)   : mean={rot.mean():.4f}  median={np.median(rot):.4f}  max={rot.max():.4f}")
        print(f"  translation diff (mm) : mean={trans.mean():.4f}  median={np.median(trans):.4f}  max={trans.max():.4f}")
        print(f"  inlier count diff (AP3P-SQPNP): mean={inl_diff.mean():.3f}  min={inl_diff.min()}  max={inl_diff.max()}")
