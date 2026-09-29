#!/usr/bin/env python3
"""run_drift_test.py -- held-out test of a linear mocap<->vision clock-drift correction, for ONE
recording. Written fresh (not a modification of any earlier scratch script) per the user's request
to re-run static_medium and static_hard from scratch and save a separate CSV per recording.

What it does, per controller:
  1. Loads STRONG vision poses from a freshly-generated vision_pose.csv (full-recording main.py run,
     current pipeline: mocap_vision_offset_ns already applied, current bridge already loaded by main.py
     for the mocap_gt export -- but this script re-derives the mocap comparison independently of that
     export, straight from the recording's own mocap_filtered/ + mocap_calib files, the same way
     compare_vision_mocap.py does).
  2. Splits the strong frames into folds (first-half/second-half in both directions, plus 4-block CV).
  3. For each fold, fits TWO models on the TRAIN frames only, each with its OWN bridge refit (robust,
     1/err^2-weighted chordal mean + iterative 3-MAD trim) under that model's own per-frame shift:
       - "constant": one extra shift a (ms) added to the recording's already-applied vision offset
       - "linear":   a + slope * (t - t_ref)/60, t_ref = train-set mean time, slope in ms/min
     then scores both on the TEST frames (bridge and shift model frozen from training).
  4. Writes one row per (controller, split, fold, model) to <out_csv> -- the per-fold summary -- plus
     a second CSV of the fitted shift-vs-time curve sampled every strong frame (for plotting).

Usage:
    python3 analysis/drift_test/run_drift_test.py \
        --recording-dir /home/nikitakarpuks/Downloads/recordings-aug26/euroc_recording_.../mav0/.. \
        --vision-pose-csv analysis/drift_test/static_medium_run/vision_pose.csv \
        --config analysis/drift_test/static_medium_run/config.yml \
        --name static_medium \
        --out-dir analysis/drift_test
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.load_config import load_yaml_config
from src.mocap_data import (load_mocap_csv, load_mocap_fine_offset_ns, load_T_imu_marker,
                             load_mocap_bridge, load_vision_offset_ns, relative_pose, DeviceMocap,
                             DRIFT_CHECK_VARIANT)
from src.transformations import Transform

CONTROLLERS = ("left_controller", "right_controller")
_MOCAP_DISK_NAMES = {"headset": "headset", "left_controller": "ctrlleft", "right_controller": "ctrlright"}

# Same 180-about-X seed compare_vision_mocap.py uses for its chordal-mean fit -- not claimed exact,
# just a real-rotation starting point in the right neighborhood.
_ROTATION_SEED = np.diag([1.0, -1.0, -1.0])

# Extra shift search grid, milliseconds, ON TOP OF the recording's own mocap_vision_offset_ns.
GRID_MS = np.arange(-10.0, 10.01, 0.5)


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def load_device_mocap(recording_root: Path, device_key: str, config: dict, vision_offset_ns: float) -> DeviceMocap:
    device_dir = recording_root / "mocap_filtered" / _MOCAP_DISK_NAMES[device_key]
    if device_key == "headset":
        calib_path = config["cameras"]["mocap_calib_path"]
    else:
        calib_path = config["controllers"][device_key]["mocap_calib_path"]
    t, position, quat_xyzw = load_mocap_csv(device_dir / "data.csv")
    fine_offset_ns = load_mocap_fine_offset_ns(device_dir / "drift_check" / DRIFT_CHECK_VARIANT / "drift_check.json")
    T_imu_marker = load_T_imu_marker(calib_path)
    max_gap_ns = float(config.get("mocap", {}).get("max_interp_gap_ms", 30.0)) * 1e6
    return DeviceMocap(t, position, quat_xyzw, fine_offset_ns, T_imu_marker,
                        max_interp_gap_ns=max_gap_ns, vision_offset_ns=vision_offset_ns)


def with_extra_shift(dev: DeviceMocap, extra_ms: float) -> DeviceMocap:
    return DeviceMocap(dev.t_ns, dev.position, dev.quat_xyzw, dev.fine_offset_ns, dev.T_imu_marker,
                        max_interp_gap_ns=dev.max_interp_gap_ns, vision_offset_ns=dev.vision_offset_ns + extra_ms * 1e6)


def load_strong_vision_poses(vision_pose_csv: Path, ctrl_name: str, strong_inliers: float, strong_error_px: float):
    """{timestamp_ns: Transform}, {timestamp_ns: error_px} -- vision_pose_csv's own layout
    (ts,ctrl,qx,qy,qz,qw,px,py,pz,confidence,error_px,n_inliers), filtered to strong frames."""
    poses, errors = {}, {}
    with open(vision_pose_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[1] != ctrl_name:
                continue
            error_px = float(row[10])
            n_inliers = float(row[11])
            if n_inliers < strong_inliers or error_px > strong_error_px:
                continue
            ts_ns = int(row[0])
            qx, qy, qz, qw = (float(x) for x in row[2:6])
            px, py, pz = (float(x) for x in row[6:9])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            poses[ts_ns] = Transform(R, np.array([px, py, pz]))
            errors[ts_ns] = error_px
    return poses, errors


def strong_thresholds(config: dict, ctrl_name: str) -> tuple:
    base = config["fusion_heuristic"]
    override = config["fusion_heuristic"].get("per_controller", {}).get(ctrl_name, {})
    return (float(override.get("vision_weight_strong_inliers", base.get("vision_weight_strong_inliers", 8))),
            float(override.get("vision_weight_strong_error_px", base.get("vision_weight_strong_error_px", 0.15))))


def raw_seed_residuals(poses: dict, headset_mocap: DeviceMocap, ctrl_mocap: DeviceMocap):
    """Per-frame residual Transform of T_vision.compose(seed) against the LIBRARY's own
    relative_pose(headset, ctrl, t) (src.mocap_data.relative_pose -- not reimplemented here, to
    avoid a convention bug), for every (timestamp, shift) combination on GRID_MS -- precomputed
    once so the fold-fitting loop below never re-touches mocap interpolation. This is exactly
    compare_vision_mocap.py's own seed-residual formula (residual_stats with bridge=seed). Returns
    kept timestamps (only ones with mocap coverage at EVERY grid shift) and a (n_frames, n_grid)
    list-of-lists of residual Transforms."""
    seed = Transform(_ROTATION_SEED, np.zeros(3))
    kept_ts, rows = [], []
    for ts_ns, T_vision in poses.items():
        row = []
        ok = True
        for shift_ms in GRID_MS:
            mocap_rel = relative_pose(headset_mocap, with_extra_shift(ctrl_mocap, shift_ms), ts_ns)
            if mocap_rel is None:
                ok = False
                break
            row.append(T_vision.compose(seed).inverse().compose(mocap_rel))
        if ok:
            kept_ts.append(ts_ns)
            rows.append(row)
    return kept_ts, rows


def fit_constant_correction(residuals, weights):
    Rs = Rotation.from_matrix(np.stack([r.R for r in residuals]))
    R_mean = Rs.mean(weights=weights).as_matrix()
    t_mean = np.average(np.stack([r.t for r in residuals]), axis=0, weights=weights)
    return Transform(R_mean, t_mean)


def robust_bridge(rows, idxs, weights, iters=4, k=3.0):
    """idxs: list of (frame_i, grid_j) pairs -- rows[frame_i][grid_j] is that frame's residual under
    the shift model being evaluated. Iteratively re-weighted chordal mean with 3-MAD trimming."""
    seed = Transform(_ROTATION_SEED, np.zeros(3))
    res = [rows[i][j] for i, j in idxs]
    w = weights[[i for i, _ in idxs]]
    keep = np.ones(len(res), dtype=bool)
    mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
    corr = None
    for _ in range(iters):
        corr = fit_constant_correction([r for r, m in zip(res, keep) if m], w[keep])
        errs = [corr.inverse().compose(r) for r in res]
        et = np.array([np.linalg.norm(e.t) for e in errs]) * 1000.0
        er = np.array([rotation_angle_deg(e.R) for e in errs])
        keep = (et < np.median(et) + k * mad(et)) & (er < np.median(er) + k * mad(er))
    return seed.compose(corr)


def score(rows, idxs, bridge):
    seed = Transform(_ROTATION_SEED, np.zeros(3))
    correction = seed.inverse().compose(bridge)
    rt, rr = [], []
    for i, j in idxs:
        e = correction.inverse().compose(rows[i][j])
        rt.append(np.linalg.norm(e.t) * 1000.0)
        rr.append(rotation_angle_deg(e.R))
    return np.array(rt), np.array(rr)


def gi(shift_ms):
    return int(np.clip(np.round((shift_ms - GRID_MS[0]) / 0.5), 0, len(GRID_MS) - 1))


def fit_models(rows, t_s, weights, train):
    """Grid-search (coarse then refined) the best constant and best linear (a, slope) on TRAIN
    frames only, cost = median(trans_mm + 3*rot_deg), bridge refit fresh at every candidate."""
    t_ref = t_s[train].mean()

    def eval_model(a, slope):
        idxs = [(i, gi(a + slope * (t_s[i] - t_ref) / 60.0)) for i in train]
        bridge = robust_bridge(rows, idxs, weights)
        rt, rr = score(rows, idxs, bridge)
        return float(np.median(rt + 3 * rr)), bridge

    best_a = min(np.arange(-6.0, 6.01, 1.0), key=lambda a: eval_model(a, 0.0)[0])
    best_a = min(np.arange(best_a - 1.0, best_a + 1.01, 0.5), key=lambda a: eval_model(a, 0.0)[0])
    coarse = [((a, s), eval_model(a, s)[0]) for a in np.arange(-4.0, 4.01, 1.0) for s in np.arange(-2.0, 8.01, 1.0)]
    (a0, s0), _ = min(coarse, key=lambda x: x[1])
    fine = [((a, s), eval_model(a, s)[0]) for a in np.arange(a0 - 1.0, a0 + 1.01, 0.5)
            for s in np.arange(s0 - 1.0, s0 + 1.01, 0.5)]
    (best_a_lin, best_slope), _ = min(fine, key=lambda x: x[1])

    _, bridge_zero = eval_model(0.0, 0.0)
    _, bridge_const = eval_model(best_a, 0.0)
    _, bridge_lin = eval_model(best_a_lin, best_slope)
    return {
        "zero":   (lambda tt: 0.0, bridge_zero, {"a_ms": 0.0, "slope_ms_per_min": 0.0}),
        "const":  (lambda tt, a=best_a: a, bridge_const, {"a_ms": best_a, "slope_ms_per_min": 0.0}),
        "linear": (lambda tt, a=best_a_lin, s=best_slope, tr=t_ref: a + s * (tt - tr) / 60.0,
                   bridge_lin, {"a_ms": best_a_lin, "slope_ms_per_min": best_slope}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording-dir", required=True, help="recording root (parent of mav0/ and mocap_filtered/)")
    ap.add_argument("--vision-pose-csv", required=True)
    ap.add_argument("--config", required=True, help="the per-run config.yml used to produce vision-pose-csv")
    ap.add_argument("--name", required=True, help="short recording name, e.g. static_medium")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    config = load_yaml_config(args.config)
    recording_root = Path(args.recording_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    headset_mocap = load_device_mocap(recording_root, "headset", config, vision_offset_ns=0.0)

    summary_rows = []
    curve_rows = []

    for ctrl in CONTROLLERS:
        si, se = strong_thresholds(config, ctrl)
        poses, errors = load_strong_vision_poses(Path(args.vision_pose_csv), ctrl, si, se)
        ctrl_vision_offset_ns = load_vision_offset_ns(config["controllers"][ctrl])
        ctrl_mocap = load_device_mocap(recording_root, ctrl, config, vision_offset_ns=ctrl_vision_offset_ns)

        kept_ts, rows = raw_seed_residuals(poses, headset_mocap, ctrl_mocap)
        if len(kept_ts) < 40:
            print(f"[{args.name}/{ctrl}] only {len(kept_ts)} strong+covered frames -- skipping (need >=40)")
            continue
        order = np.argsort(kept_ts)
        kept_ts = np.array(kept_ts)[order]
        rows = [rows[i] for i in order]
        t0 = kept_ts[0]
        t_s = (kept_ts - t0) / 1e9
        weights = np.array([1.0 / max(errors[int(ts)], 0.05) ** 2 for ts in kept_ts])
        n = len(kept_ts)
        print(f"[{args.name}/{ctrl}] n={n} strong+mocap-covered frames, span {t_s[-1]:.1f}s "
              f"(recording's own vision offset {ctrl_vision_offset_ns/1e6:.2f}ms)")

        idx_all = np.arange(n)
        folds_by_split = {
            "first_half_train": [(idx_all[:n // 2], idx_all[n // 2:])],
            "second_half_train": [(idx_all[n // 2:], idx_all[:n // 2])],
            "4block_cv": [(np.concatenate([b for j, b in enumerate(np.array_split(idx_all, 4)) if j != k]),
                           np.array_split(idx_all, 4)[k]) for k in range(4)],
        }

        for split_name, folds in folds_by_split.items():
            for fold_idx, (train, test) in enumerate(folds):
                models = fit_models(rows, t_s, weights, train)
                for model_name, (shift_fn, bridge, params) in models.items():
                    test_idxs = [(i, gi(shift_fn(t_s[i]))) for i in test]
                    rt, rr = score(rows, test_idxs, bridge)
                    summary_rows.append({
                        "recording": args.name, "controller": ctrl, "split": split_name, "fold": fold_idx,
                        "model": model_name, "n_train": len(train), "n_test": len(test),
                        "fitted_a_ms": round(params["a_ms"], 3),
                        "fitted_slope_ms_per_min": round(params["slope_ms_per_min"], 3),
                        "test_trans_mean_mm": round(float(rt.mean()), 3),
                        "test_trans_median_mm": round(float(np.median(rt)), 3),
                        "test_trans_p90_mm": round(float(np.percentile(rt, 90)), 3),
                        "test_rot_mean_deg": round(float(rr.mean()), 3),
                        "test_rot_median_deg": round(float(np.median(rr)), 3),
                        "test_rot_p90_deg": round(float(np.percentile(rr, 90)), 3),
                    })
                    print(f"  [{split_name} fold{fold_idx}] {model_name:6s} a={params['a_ms']:+.2f}ms "
                          f"slope={params['slope_ms_per_min']:+.2f}ms/min -> test trans mean "
                          f"{rt.mean():.2f}mm rot mean {rr.mean():.2f}deg (n_test={len(test)})")

        # Best single linear model fit on ALL frames of this recording (not held out) -- for the
        # curve CSV: per-frame best extra shift under the best linear model's own bridge, for plotting.
        full_models = fit_models(rows, t_s, weights, idx_all)
        shift_fn, bridge, params = full_models["linear"]
        for i in idx_all:
            j = gi(shift_fn(t_s[i]))
            e = Transform(_ROTATION_SEED, np.zeros(3)).inverse().compose(bridge).inverse().compose(rows[i][j])
            curve_rows.append({
                "recording": args.name, "controller": ctrl, "timestamp_ns": int(kept_ts[i]),
                "t_since_first_strong_s": round(float(t_s[i]), 3),
                "fitted_shift_ms": round(float(shift_fn(t_s[i])), 3),
                "trans_mm_at_fitted_shift": round(float(np.linalg.norm(
                    Transform(_ROTATION_SEED, np.zeros(3)).inverse().compose(bridge).inverse()
                    .compose(rows[i][j]).t) * 1000.0), 3),
            })

    summary_csv = out_dir / f"drift_test_{args.name}.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nWrote {len(summary_rows)} rows -> {summary_csv}")

    curve_csv = out_dir / f"drift_curve_{args.name}.csv"
    with open(curve_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        w.writeheader()
        w.writerows(curve_rows)
    print(f"Wrote {len(curve_rows)} rows -> {curve_csv}")


if __name__ == "__main__":
    main()
