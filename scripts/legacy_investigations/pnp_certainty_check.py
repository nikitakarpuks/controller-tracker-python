#!/usr/bin/env python3
"""
pnp_certainty_check.py -- Step 2 of the IMU+vision fusion plan: the vision/PnP
term's "certainty weighting" -- does the per-frame PnP solve's own quality
signal (reprojection error, inlier count) actually predict how far vision's
solved pose is from the TRUE (held-out mocap) pose?

Reproj-error weighting alone barely correlates with true accuracy (found this
session via compare_vision_mocap.py's residual_stats + pose_csv's
reproj_err_px). This script checks inlier count's correlation the same way --
the other half of the "PnP certainty" idea, untested until now -- and combines
both into one weighting function, checked for smoothness (no degenerate/
exploding values).

Needs inlier_count per accepted frame, which the ORIGINAL pose_log_2765frames.csv
never recorded (only reproj_err_px). main.py's pose_csv writer now also writes
inlier_count = len(sol["assignment"]) (the primary camera's own post-RANSAC
correspondence count -- same solve reproj_err_px comes from, so both signals
are about the SAME primary-camera PnP fit's certainty). Re-running main.py
over the full 2765-frame recording (debug.pose_csv retargeted to a new file,
visualization disabled for speed) reproduced every qx/qy/qz/qw/px/py/pz/
reproj_err_px value bit-for-bit against the original pose_log_2765frames.csv
(deterministic pipeline, same rng_seed) -- data/pose_log_2765frames_inliers.csv
is that same run with the new column added, not a different dataset.

Weighting:
  w_reproj(e)      = 1 / max(e, MIN_REPROJ_ERR_PX)^2         (compare_vision_
                      mocap.py's existing inverse-variance weighting)
  confidence(n)    = clip((n - 3) / 6, 0, 1)                 (config.yml's own
                      proximity_redundancy_ref=6 confidence_factor convention
                      from src/pose_search.py -- a 3-point fit has zero
                      redundancy/trust, 9+ points is fully trusted)
  w_combined(e, n) = w_reproj(e) * confidence(n)

Test: plot w_combined vs. reproj_err_px and vs. inlier_count separately,
confirm w_combined is finite/bounded (no degenerate/exploding values), and
compare |correlation| of raw reproj_err_px vs. true residual against raw
inlier_count vs. true residual (Pearson, on both trans_mm and rot_deg).

Usage: python pnp_certainty_check.py [path/to/config.yml] [output_dir]
Requires data/pose_log_2765frames_inliers.csv (or override via argv) and the
same real (out-of-repo) mocap files as compare_vision_mocap.py.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from compare_vision_mocap import load_device_mocap, load_or_fit_bridge, residual_stats, _MIN_REPROJ_ERR_PX
from src.load_config import load_yaml_config
from src.transformations import Transform

_POSE_CSV_WITH_INLIERS = "./data/pose_log_2765frames_inliers.csv"
_REDUNDANCY_REF = 6.0   # matches config.yml matching.proximity_redundancy_ref
_MIN_FIT_INLIERS = 3.0  # matches src/pose_search.py's confidence_factor floor (3-point minimal fit)


def load_pose_csv_with_inliers(path):
    """(poses, reproj_err_px, inlier_count) -- all {ctrl_name: {timestamp_ns: value}}
    -- from the inlier_count-extended pose_csv (see module docstring)."""
    poses, errors, inliers = {}, {}, {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert "inlier_count" in header, f"{path} has no inlier_count column -- rerun main.py with the Step 2 pose_csv change"
        for row in reader:
            ts_ns = int(row[0])
            ctrl_name = row[1]
            qx, qy, qz, qw = (float(x) for x in row[2:6])
            px, py, pz = (float(x) for x in row[6:9])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([px, py, pz])
            poses.setdefault(ctrl_name, {})[ts_ns] = Transform(R, t)
            errors.setdefault(ctrl_name, {})[ts_ns] = float(row[9])
            inliers.setdefault(ctrl_name, {})[ts_ns] = int(row[10])
    return poses, errors, inliers


def combined_weight(reproj_err_px: np.ndarray, inlier_count: np.ndarray):
    w_reproj = 1.0 / np.maximum(reproj_err_px, _MIN_REPROJ_ERR_PX) ** 2
    confidence = np.clip((inlier_count - _MIN_FIT_INLIERS) / _REDUNDANCY_REF, 0.0, 1.0)
    return w_reproj * confidence, w_reproj, confidence


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def plot_weight_scatter(ctrl_name: str, err_px, inlier_count, w_combined, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(err_px, w_combined, s=6, alpha=0.35, color="steelblue")
    axes[0].set_xlabel("reproj_err_px")
    axes[0].set_ylabel("w_combined")
    axes[0].set_title("weight vs. reprojection error")

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.15, 0.15, size=len(inlier_count))
    axes[1].scatter(inlier_count + jitter, w_combined, s=6, alpha=0.35, color="darkorange")
    axes[1].set_xlabel("inlier_count (jittered)")
    axes[1].set_ylabel("w_combined")
    axes[1].set_title("weight vs. inlier count")

    fig.suptitle(f"{ctrl_name}: combined PnP-certainty weight (n={len(err_px)})")
    fig.tight_layout()
    fig.savefig(out_dir / f"pnp_certainty_{ctrl_name}_weights.png", dpi=130)
    plt.close(fig)


def plot_residual_vs_signal(ctrl_name: str, err_px, inlier_count, trans_mm, rot_deg, out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].scatter(err_px, trans_mm, s=6, alpha=0.35, color="steelblue")
    axes[0, 0].set_xlabel("reproj_err_px")
    axes[0, 0].set_ylabel("true trans residual (mm)")
    axes[0, 0].set_title(f"r={pearson(err_px, trans_mm):+.3f}")

    axes[0, 1].scatter(inlier_count, trans_mm, s=6, alpha=0.35, color="darkorange")
    axes[0, 1].set_xlabel("inlier_count")
    axes[0, 1].set_ylabel("true trans residual (mm)")
    axes[0, 1].set_title(f"r={pearson(inlier_count, trans_mm):+.3f}")

    axes[1, 0].scatter(err_px, rot_deg, s=6, alpha=0.35, color="steelblue")
    axes[1, 0].set_xlabel("reproj_err_px")
    axes[1, 0].set_ylabel("true rot residual (deg)")
    axes[1, 0].set_title(f"r={pearson(err_px, rot_deg):+.3f}")

    axes[1, 1].scatter(inlier_count, rot_deg, s=6, alpha=0.35, color="darkorange")
    axes[1, 1].set_xlabel("inlier_count")
    axes[1, 1].set_ylabel("true rot residual (deg)")
    axes[1, 1].set_title(f"r={pearson(inlier_count, rot_deg):+.3f}")

    fig.suptitle(f"{ctrl_name}: does the PnP certainty signal predict true (mocap) residual?")
    fig.tight_layout()
    fig.savefig(out_dir / f"pnp_certainty_{ctrl_name}_residual_correlation.png", dpi=130)
    plt.close(fig)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yml"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("visualization/step2")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml_config(config_path)

    pose_csv_path = Path(_POSE_CSV_WITH_INLIERS)
    if not pose_csv_path.exists():
        raise SystemExit(f"{pose_csv_path} not found -- rerun main.py with debug.pose_csv pointed there "
                          f"(inlier_count column added to the pose_csv writer for Step 2)")

    poses, errors, inlier_counts = load_pose_csv_with_inliers(pose_csv_path)
    headset_mocap = load_device_mocap("headset")

    for ctrl_name in ("left_controller", "right_controller"):
        if ctrl_name not in poses:
            continue
        ctrl_cfg = config["controllers"][ctrl_name]
        ctrl_mocap = load_device_mocap(ctrl_name)
        bridge = load_or_fit_bridge(ctrl_name, ctrl_cfg, poses[ctrl_name], headset_mocap, ctrl_mocap,
                                     errors.get(ctrl_name))

        rot_deg, trans_mm, n_skipped, _residuals, kept_ts = residual_stats(
            poses[ctrl_name], headset_mocap, ctrl_mocap, bridge)
        err_px = np.array([errors[ctrl_name][ts] for ts in kept_ts])
        n_inliers = np.array([inlier_counts[ctrl_name][ts] for ts in kept_ts], dtype=np.float64)

        w_combined, w_reproj, confidence = combined_weight(err_px, n_inliers)

        print(f"[{ctrl_name}] {len(kept_ts)} frames ({n_skipped} skipped, no mocap coverage)")
        print(f"    w_combined: min={w_combined.min():.6g} max={w_combined.max():.6g} "
              f"finite={np.all(np.isfinite(w_combined))} nonneg={np.all(w_combined >= 0)}")
        print(f"    correlation with true trans residual (mm): reproj_err_px r={pearson(err_px, trans_mm):+.3f}  "
              f"inlier_count r={pearson(n_inliers, trans_mm):+.3f}")
        print(f"    correlation with true rot residual (deg):  reproj_err_px r={pearson(err_px, rot_deg):+.3f}  "
              f"inlier_count r={pearson(n_inliers, rot_deg):+.3f}")

        plot_weight_scatter(ctrl_name, err_px, n_inliers, w_combined, out_dir)
        plot_residual_vs_signal(ctrl_name, err_px, n_inliers, trans_mm, rot_deg, out_dir)
        print(f"[{ctrl_name}] saved plots to {out_dir}/pnp_certainty_{ctrl_name}_{{weights,residual_correlation}}.png\n")


if __name__ == "__main__":
    main()
