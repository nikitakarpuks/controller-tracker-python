"""
Microbenchmark for the two hot spots flagged by profiling: RANSAC PnP calls
(_ransac_pnp, i.e. cv2.solvePnPRansac) and per-ROI blob detection
(_detect_blobs).

Both functions already bottom out in compiled OpenCV C++. The question this
script answers is not "is C++ faster than Python at this math" (it won't be,
it's the same code) but: is the wall time dominated by

  (a) actual RANSAC/labeling compute inside cv2 -- a C++ port of the same
      algorithm buys nothing here, only a cheaper algorithm or fewer calls
      would, or
  (b) fixed per-call Python<->cv2 dispatch/marshaling overhead -- this
      argues for batching many small calls into one native call, independent
      of language, since it's paid once per call regardless of problem size.

Uses real calibration (data/cameras/calibration_mateo_together.json), a real
controller LED model, a real captured frame, and the real blob_detection cfg
block from config/config.yml, so the numbers reflect production-sized inputs
rather than made-up ones.

Run: python benchmarks/bench_p3p_blob.py
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src._pnp import _ransac_pnp
from src.blob_detector import _detect_blobs
from src.load_config import load_yaml_config

RNG = np.random.default_rng(0)


def bench(fn, reps=300, warmup=20):
    """Mean wall time per call, in microseconds."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e6


# ---------------------------------------------------------------------------
# RANSAC PnP
# ---------------------------------------------------------------------------

def load_camera():
    calib = json.load(open(ROOT / "data/cameras/calibration_mateo_together.json"))
    intr = calib["value0"]["intrinsics"][0]["intrinsics"]
    K = np.array([[intr["fx"], 0, intr["cx"]],
                  [0, intr["fy"], intr["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    dc = np.array([intr["k1"], intr["k2"], intr["k3"], intr["k4"]], dtype=np.float64)
    return K, dc


def load_led_positions():
    cfg = json.load(open(ROOT / "data/controllers/right_controller_A85K6081930636R.json"))
    leds = cfg["CalibrationInformation"]["ControllerLeds"]
    return np.array([l["Position"] for l in leds], dtype=np.float32)


def make_correspondences(n_pts, K, dc, led_positions, noise_px=0.4):
    """Project a random plausible controller pose through the real fisheye
    model to get correspondences that behave like real detected blobs
    (mostly inliers, a bit of pixel noise) rather than pure noise."""
    idx = RNG.choice(len(led_positions), size=n_pts, replace=False)
    obj_pts = led_positions[idx]

    rvec = RNG.uniform(-0.3, 0.3, size=3)
    tvec = np.array([RNG.uniform(-0.1, 0.1),
                      RNG.uniform(-0.1, 0.1),
                      RNG.uniform(0.3, 1.0)])

    pts, _ = cv2.fisheye.projectPoints(
        obj_pts.astype(np.float64).reshape(-1, 1, 3), rvec, tvec, K, dc,
    )
    pts = pts.reshape(-1, 2) + RNG.normal(0, noise_px, size=(n_pts, 2))
    return obj_pts, pts.astype(np.float32)


def bench_solve_p3p():
    """The actual top-of-funnel call: cv2.solveP3P on exactly 3 correspondences,
    at identity K / zero distortion (pose_search.py pre-undistorts blobs once
    per attempt), mirroring pose_search.py:2040-2046 exactly. This is what
    tier_p3p_calls counts -- called once per (anchor, b1, b2) triple bijection
    tried in the brute search, i.e. the dominant call count, unlike
    _ransac_pnp below which only runs on survivors of the full gate/visibility/
    Hungarian funnel."""
    K, dc = load_camera()
    led_positions = load_led_positions()
    idx = RNG.choice(len(led_positions), size=3, replace=False)
    world_pts = led_positions[idx].reshape(3, 1, 3)

    rvec = RNG.uniform(-0.3, 0.3, size=3)
    tvec = np.array([RNG.uniform(-0.1, 0.1), RNG.uniform(-0.1, 0.1), RNG.uniform(0.3, 1.0)])
    img_pts_dist, _ = cv2.fisheye.projectPoints(
        world_pts.astype(np.float64), rvec, tvec, K, dc,
    )
    img_norm = cv2.fisheye.undistortPoints(img_pts_dist, K, dc).reshape(3, 1, 2).astype(np.float32)
    K_id_f32, dc0_f32 = np.eye(3, dtype=np.float32), np.zeros(4, dtype=np.float32)

    print("\n=== cv2.solveP3P (the actual tier_p3p_calls hot path, 3 points) ===")

    def solve(flag):
        return cv2.solveP3P(world_pts.astype(np.float32), img_norm, K_id_f32, dc0_f32, flags=flag)

    us_p3p = bench(lambda: solve(cv2.SOLVEPNP_P3P), reps=1000)
    us_ap3p = bench(lambda: solve(cv2.SOLVEPNP_AP3P), reps=1000)
    print(f"  SOLVEPNP_P3P  (current) : {us_p3p:>7.2f} us/call")
    print(f"  SOLVEPNP_AP3P            : {us_ap3p:>7.2f} us/call")
    print(f"  difference               : {us_p3p - us_ap3p:>7.2f} us/call ({(us_p3p-us_ap3p)/us_p3p*100:.1f}%)")


def bench_p3p():
    K, dc = load_camera()
    led_positions = load_led_positions()

    print("\n=== RANSAC PnP (_ransac_pnp, SOLVEPNP_SQPNP, fisheye) ===")
    print(f"{'n_pts':>6}  {'us/call':>10}")
    for n_pts in (4, 6, 8, 10, 14, 20):
        obj_pts, img_pts = make_correspondences(n_pts, K, dc, led_positions)
        us = bench(lambda o=obj_pts, i=img_pts: _ransac_pnp(o, i, K, dc, is_fisheye=True))
        print(f"{n_pts:>6}  {us:>10.1f}")

    # Split the call into its two internal stages (undistort vs solve) to see
    # where the time actually goes.
    print("\n--- stage split, n_pts=8 ---")
    obj_pts, img_pts = make_correspondences(8, K, dc, led_positions)
    inp = img_pts.astype(np.float32).reshape(-1, 1, 2)
    K_id, dc0 = np.eye(3), np.zeros(4)

    def undistort_only():
        return cv2.fisheye.undistortPoints(inp.astype(np.float64), K, dc).reshape(-1, 2)

    pts_norm = undistort_only()

    def solve_only():
        return cv2.solvePnPRansac(
            obj_pts.astype(np.float64), pts_norm.astype(np.float64), K_id, dc0,
            useExtrinsicGuess=False, iterationsCount=100,
            reprojectionError=2.5 / K[0, 0], confidence=0.99,
            flags=cv2.SOLVEPNP_SQPNP,
        )

    print(f"  undistortPoints only : {bench(undistort_only):>8.1f} us/call")
    print(f"  solvePnPRansac only  : {bench(solve_only):>8.1f} us/call")
    full = bench(lambda: _ransac_pnp(obj_pts, img_pts, K, dc, is_fisheye=True))
    print(f"  full _ransac_pnp     : {full:>8.1f} us/call")

    # SQPNP (current, general non-minimal solver) vs AP3P (closed-form
    # minimal solver) across the same n_pts sweep -- a same-language,
    # zero-C++ experiment. n_pts=4 alone is degenerate (the minimal sample
    # *is* the whole input, so RANSAC barely samples) so this checks whether
    # solver choice matters once real combinatorial RANSAC sampling kicks in
    # at higher n_pts.
    print("\n--- SQPNP vs AP3P flag, across n_pts (mean of 8 random trials each) ---")
    print(f"{'n_pts':>6}  {'SQPNP us':>10}  {'AP3P us':>10}  {'reduction':>10}")
    all_sqpnp, all_ap3p = [], []
    for n_pts in (4, 6, 8, 10, 14, 20):
        trial_sqpnp, trial_ap3p = [], []
        for _ in range(8):
            obj_n, img_n = make_correspondences(n_pts, K, dc, led_positions)
            inp_n = img_n.astype(np.float32).reshape(-1, 1, 2)
            pts_norm_n = cv2.fisheye.undistortPoints(inp_n.astype(np.float64), K, dc).reshape(-1, 2)

            def solve_flag(flag, o=obj_n, p=pts_norm_n):
                return cv2.solvePnPRansac(
                    o.astype(np.float64), p.astype(np.float64), K_id, dc0,
                    useExtrinsicGuess=False, iterationsCount=100,
                    reprojectionError=2.5 / K[0, 0], confidence=0.99, flags=flag,
                )

            trial_sqpnp.append(bench(lambda: solve_flag(cv2.SOLVEPNP_SQPNP), reps=60, warmup=5))
            trial_ap3p.append(bench(lambda: solve_flag(cv2.SOLVEPNP_AP3P), reps=60, warmup=5))
        us_sqpnp, us_ap3p = float(np.mean(trial_sqpnp)), float(np.mean(trial_ap3p))
        all_sqpnp.append(us_sqpnp); all_ap3p.append(us_ap3p)
        print(f"{n_pts:>6}  {us_sqpnp:>10.1f}  {us_ap3p:>10.1f}  {(us_sqpnp-us_ap3p)/us_sqpnp*100:>9.1f}%")

    # n_pts=4 is a special case: OpenCV forces the minimal-set solver to P3P
    # regardless of the requested flag when input count == 4 (see
    # solvePnPRansac docstring), so it isn't a fair sample of the flag's effect.
    sq_no4, ap_no4 = all_sqpnp[1:], all_ap3p[1:]
    avg_sq, avg_ap = float(np.mean(sq_no4)), float(np.mean(ap_no4))
    print(f"\n  average over n_pts=6..20 (excludes forced-P3P n=4 case):")
    print(f"    SQPNP {avg_sq:.1f} us/call  ->  AP3P {avg_ap:.1f} us/call "
          f"  ({avg_sq - avg_ap:.1f} us/call saved, {(avg_sq-avg_ap)/avg_sq*100:.1f}% reduction)")


# ---------------------------------------------------------------------------
# Blob detection
# ---------------------------------------------------------------------------

def load_frame(rel_path):
    img = cv2.imread(str(ROOT / rel_path), cv2.IMREAD_GRAYSCALE)
    # 4 cameras stitched horizontally (640px each); row 0 is the
    # exposure/gain technical row, not image data.
    cam0 = img[1:, 0:640]
    return np.ascontiguousarray(cam0)


FRAMES = {
    "clean (basic moves)":
        "data/datasets/single_ctrl/1_r_ctrl_all_basic_moves/84056661988912.png",
    "noisy (in front of window)":
        "data/datasets/threshold_search/in_front_of_window/30627412308130.png",
}


def bench_blobs():
    cfg = load_yaml_config(str(ROOT / "config/config.yml"))["blob_detection"]

    pixel_thr = cfg["min_threshold"]
    req_thr = int(pixel_thr * cfg.get("required_threshold_factor", 2.0))

    print("\n=== Blob detection (_detect_blobs) ===")

    frame = None
    for label, path in FRAMES.items():
        f = load_frame(path)
        n_components = cv2.connectedComponentsWithStats(
            cv2.threshold(f, pixel_thr, 255, cv2.THRESH_BINARY)[1], connectivity=8,
        )[0] - 1
        us_full = bench(lambda f=f: _detect_blobs(f, pixel_thr, req_thr, cfg), reps=100)
        print(f"  cold, full frame, {label:<28} ({n_components:>5} raw components): {us_full:>9.1f} us/call")
        if frame is None:
            frame = f  # use the "clean" frame for the per-LED ROI sweep below

    # Per-LED local ROI, as used by the warm 'fit' path in _detect_blobs_local:
    # a small circular crop around one predicted LED position.
    # search_radius_px default is 8px -> 17x17; wider radii cover the
    # depth_k-scaled case.
    print("\n  warm-path style per-LED ROI calls:")
    for radius in (8, 15, 25):
        size = radius * 2 + 1
        cy, cx = frame.shape[0] // 2, frame.shape[1] // 2
        roi = frame[cy - radius:cy + radius + 1, cx - radius:cx + radius + 1].copy()

        us_roi = bench(lambda r=roi: _detect_blobs(r, pixel_thr, req_thr, cfg, warm_mode=True), reps=300)
        print(f"    {size}x{size} ROI (r={radius:>2}px): {us_roi:>8.1f} us/call", end="")
        for n_leds in (8, 16, 32):
            print(f"   | x{n_leds:>2} LEDs = {us_roi * n_leds / 1000:.2f} ms/frame", end="")
        print(f"   (vs {us_full / 1000:.2f} ms for 1 cold full-frame call)")


if __name__ == "__main__":
    bench_solve_p3p()
    bench_p3p()
    bench_blobs()
