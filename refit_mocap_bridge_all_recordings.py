#!/usr/bin/env python3
"""Checks the current mocap-controller bridge (data/mocap_calib/controller_
{left,right}_mocap_bridge_basalt01.json, fit from static_dark only) against
ALL 8 recordings under recordings-aug26, using STRONG vision poses only
(n_inliers >= vision_weight_strong_inliers, error_px <= vision_weight_strong_
error_px -- each controller's own real config.yml threshold), then refits
pooling strong poses across every recording and compares residuals.

Reuses evaluate_2026-09-16/<name>/vision_pose.csv (RAW vision, unaffected by
fusion.enabled -- the correct "pure vision" input the bridge fit needs, see
compare_vision_mocap.py's own module docstring) from the batch run already
done -- no need to re-run main.py.

Usage: python3 refit_mocap_bridge_all_recordings.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_vision_mocap import (fit_constant_correction, residual_stats,  # noqa: E402
                                   rotation_angle_deg, _ROTATION_SEED, save_bridge)
from src.load_config import load_yaml_config  # noqa: E402
from src.mocap_data import load_mocap_csv, load_mocap_fine_offset_ns, load_T_imu_marker, \
                            load_mocap_bridge, DeviceMocap, relative_pose, DRIFT_CHECK_VARIANT, \
                            load_vision_offset_ns, load_vision_drift_params  # noqa: E402
from src.transformations import Transform  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
RECORDINGS_ROOT = Path("/home/nikitakarpuks/Downloads/recordings-aug26")
EVAL_DIR = REPO_ROOT / "visualization" / "evaluate_2026-09-16"
CONFIG = load_yaml_config(str(REPO_ROOT / "config" / "config.yml"))

CONTROLLERS = ("left_controller", "right_controller")
_MOCAP_DISK_NAMES = {"headset": "headset", "left_controller": "ctrlleft", "right_controller": "ctrlright"}
_MOCAP_CALIB_FILES = {
    "headset": CONFIG["cameras"]["mocap_calib_path"],
    "left_controller": CONFIG["controllers"]["left_controller"]["mocap_calib_path"],
    "right_controller": CONFIG["controllers"]["right_controller"]["mocap_calib_path"],
}

RECORDINGS = sorted(p for p in RECORDINGS_ROOT.iterdir()
                     if p.is_dir() and p.name.startswith("euroc_recording_"))


def rec_short_name(rec_dir: Path) -> str:
    parts = rec_dir.name.split("_", 3)
    return parts[3] if len(parts) == 4 else rec_dir.name


def load_device_mocap(rec_dir: Path, device_key: str) -> DeviceMocap:
    device_dir = rec_dir / "mocap_filtered" / _MOCAP_DISK_NAMES[device_key]
    calib_path = _MOCAP_CALIB_FILES[device_key]
    t, position, quat_xyzw = load_mocap_csv(device_dir / "data.csv")
    fine_offset_ns = load_mocap_fine_offset_ns(device_dir / "drift_check" / DRIFT_CHECK_VARIANT / "drift_check.json")
    T_imu_marker = load_T_imu_marker(calib_path)
    dev_cfg = CONFIG["controllers"].get(device_key) if device_key != "headset" else None
    drift_offset_ns, drift_rate_ns_per_ns = load_vision_drift_params(dev_cfg)
    return DeviceMocap(t, position, quat_xyzw, fine_offset_ns, T_imu_marker,
                       vision_offset_ns=load_vision_offset_ns(dev_cfg),
                       drift_offset_ns=drift_offset_ns, drift_rate_ns_per_ns=drift_rate_ns_per_ns)


def load_strong_vision_poses(vision_pose_csv: Path, ctrl_name: str,
                              strong_inliers: float, strong_error_px: float):
    """{timestamp_ns: Transform} + {timestamp_ns: error_px}, filtered to
    n_inliers >= strong_inliers and error_px <= strong_error_px -- vision_
    pose_csv's own layout (ts,ctrl,qx,qy,qz,qw,px,py,pz,confidence,error_px,
    n_inliers), NOT pose_csv's (different column order -- see module
    docstring)."""
    poses, errors = {}, {}
    if not vision_pose_csv.exists():
        return poses, errors
    with open(vision_pose_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
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


def strong_thresholds(ctrl_name: str) -> tuple:
    base = CONFIG["fusion_heuristic"]
    override = CONFIG["fusion_heuristic"].get("per_controller", {}).get(ctrl_name, {})
    strong_inliers = float(override.get("vision_weight_strong_inliers", base.get("vision_weight_strong_inliers", 8)))
    strong_error_px = float(override.get("vision_weight_strong_error_px", base.get("vision_weight_strong_error_px", 0.15)))
    return strong_inliers, strong_error_px


def main():
    for ctrl_name in CONTROLLERS:
        strong_inliers, strong_error_px = strong_thresholds(ctrl_name)
        print(f"\n{'='*70}\n{ctrl_name}  (strong: n_inliers>={strong_inliers:.0f}, error_px<={strong_error_px:.2f})\n{'='*70}")

        current_bridge_path = Path(CONFIG["controllers"][ctrl_name]["mocap_bridge_path"])
        current_bridge = load_mocap_bridge(str(current_bridge_path)) if current_bridge_path.exists() else None

        per_rec_poses = {}   # name -> (poses dict, errors dict)
        per_rec_mocap = {}   # name -> (headset_mocap, ctrl_mocap)
        for rec_dir in RECORDINGS:
            name = rec_short_name(rec_dir)
            vision_csv = EVAL_DIR / name / "vision_pose.csv"
            poses, errors = load_strong_vision_poses(vision_csv, ctrl_name, strong_inliers, strong_error_px)
            if not poses:
                print(f"  [{name}] no strong poses -- skipping")
                continue
            headset_mocap = load_device_mocap(rec_dir, "headset")
            ctrl_mocap = load_device_mocap(rec_dir, ctrl_name)
            per_rec_poses[name] = (poses, errors)
            per_rec_mocap[name] = (headset_mocap, ctrl_mocap)

        # ── Report CURRENT bridge's residual per recording, strong poses only ──
        print(f"\n-- current bridge ({current_bridge_path.name}) residual on STRONG poses --")
        all_rot, all_trans = [], []
        for name, (poses, errors) in per_rec_poses.items():
            headset_mocap, ctrl_mocap = per_rec_mocap[name]
            rot_deg, trans_mm, n_skipped, _, _ = residual_stats(poses, headset_mocap, ctrl_mocap, current_bridge)
            if len(rot_deg) == 0:
                print(f"  [{name}] 0 frames with mocap coverage (of {len(poses)} strong)")
                continue
            print(f"  [{name}] n={len(rot_deg)} (skipped {n_skipped})  "
                  f"rot {rot_deg.mean():.3f}±{rot_deg.std():.3f}° (max {rot_deg.max():.2f})  "
                  f"trans {trans_mm.mean():.3f}±{trans_mm.std():.3f}mm (max {trans_mm.max():.2f})")
            all_rot.append(rot_deg)
            all_trans.append(trans_mm)
        if all_rot:
            all_rot_c = np.concatenate(all_rot)
            all_trans_c = np.concatenate(all_trans)
            print(f"  POOLED (all recordings): n={len(all_rot_c)}  "
                  f"rot {all_rot_c.mean():.3f}±{all_rot_c.std():.3f}°  "
                  f"trans {all_trans_c.mean():.3f}±{all_trans_c.std():.3f}mm")

        # ── Refit pooling STRONG poses from every recording ──────────────────
        pooled_poses, pooled_errors = {}, {}
        pooled_headset_ts_map = {}  # ts collisions across recordings are impossible (device clock per-recording), but keep separate per-rec pairs for residual_stats
        # residual_stats needs one consistent (headset_mocap, ctrl_mocap) pair --
        # can't pool across recordings directly (different mocap trajectories).
        # Fit via seed-residual pooling instead: compute each recording's own
        # seed-residual Transforms/weights, then fit_constant_correction over
        # the UNION.
        seed = Transform(_ROTATION_SEED, np.zeros(3))
        all_residuals, all_weights, total_n, total_skipped = [], [], 0, 0
        for name, (poses, errors) in per_rec_poses.items():
            headset_mocap, ctrl_mocap = per_rec_mocap[name]
            _, _, n_skipped, seed_residuals, kept_ts = residual_stats(poses, headset_mocap, ctrl_mocap, seed)
            total_skipped += n_skipped
            total_n += len(seed_residuals)
            if not seed_residuals:
                continue
            err = np.array([errors.get(ts, float("nan")) for ts in kept_ts])
            err = np.nan_to_num(err, nan=np.nanmean(err) if not np.all(np.isnan(err)) else 0.1)
            w = 1.0 / np.maximum(err, 0.05) ** 2
            all_residuals.extend(seed_residuals)
            all_weights.append(w)
        if not all_residuals:
            print("  no strong+mocap-covered frames anywhere -- cannot refit")
            continue
        weights = np.concatenate(all_weights)
        correction = fit_constant_correction(all_residuals, weights)
        refit_bridge = seed.compose(correction)

        print(f"\n-- REFIT bridge (pooled {total_n} strong frames across {len(per_rec_poses)} recordings, "
              f"{total_skipped} skipped) residual on STRONG poses --")
        all_rot2, all_trans2 = [], []
        for name, (poses, errors) in per_rec_poses.items():
            headset_mocap, ctrl_mocap = per_rec_mocap[name]
            rot_deg, trans_mm, n_skipped, _, _ = residual_stats(poses, headset_mocap, ctrl_mocap, refit_bridge)
            if len(rot_deg) == 0:
                continue
            print(f"  [{name}] n={len(rot_deg)}  "
                  f"rot {rot_deg.mean():.3f}±{rot_deg.std():.3f}° (max {rot_deg.max():.2f})  "
                  f"trans {trans_mm.mean():.3f}±{trans_mm.std():.3f}mm (max {trans_mm.max():.2f})")
            all_rot2.append(rot_deg)
            all_trans2.append(trans_mm)
        all_rot2_c = np.concatenate(all_rot2)
        all_trans2_c = np.concatenate(all_trans2)
        print(f"  POOLED (all recordings): n={len(all_rot2_c)}  "
              f"rot {all_rot2_c.mean():.3f}±{all_rot2_c.std():.3f}°  "
              f"trans {all_trans2_c.mean():.3f}±{all_trans2_c.std():.3f}mm")

        # ── Leave-one-recording-out cross-validation (refit on N-1, test on the
        # held-out one) -- checks the refit generalizes rather than overfitting
        # the pooled sample. ──────────────────────────────────────────────────
        print(f"\n-- leave-one-recording-out cross-validation --")
        loo_rot, loo_trans = [], []
        for held_out in per_rec_poses:
            train_residuals, train_weights = [], []
            for name, (poses, errors) in per_rec_poses.items():
                if name == held_out:
                    continue
                headset_mocap, ctrl_mocap = per_rec_mocap[name]
                _, _, _, seed_residuals, kept_ts = residual_stats(poses, headset_mocap, ctrl_mocap, seed)
                if not seed_residuals:
                    continue
                err = np.array([errors.get(ts, float("nan")) for ts in kept_ts])
                err = np.nan_to_num(err, nan=np.nanmean(err) if not np.all(np.isnan(err)) else 0.1)
                train_residuals.extend(seed_residuals)
                train_weights.append(1.0 / np.maximum(err, 0.05) ** 2)
            if not train_residuals:
                continue
            loo_bridge = seed.compose(fit_constant_correction(train_residuals, np.concatenate(train_weights)))
            poses, errors = per_rec_poses[held_out]
            headset_mocap, ctrl_mocap = per_rec_mocap[held_out]
            rot_deg, trans_mm, _, _, _ = residual_stats(poses, headset_mocap, ctrl_mocap, loo_bridge)
            if len(rot_deg) == 0:
                continue
            print(f"  held out [{held_out}]: rot {rot_deg.mean():.3f}° trans {trans_mm.mean():.3f}mm "
                  f"(trained on the other {len(per_rec_poses)-1})")
            loo_rot.append(rot_deg.mean())
            loo_trans.append(trans_mm.mean())
        if loo_rot:
            print(f"  LOO-CV mean across held-out recordings: rot {np.mean(loo_rot):.3f}°  "
                  f"trans {np.mean(loo_trans):.3f}mm")

        print(f"\n-- SUMMARY [{ctrl_name}] --")
        print(f"  current (static_dark-only) bridge, pooled-strong residual: "
              f"rot {all_rot_c.mean():.3f}° trans {all_trans_c.mean():.3f}mm")
        print(f"  refit (all-recordings-pooled) bridge, pooled-strong residual: "
              f"rot {all_rot2_c.mean():.3f}° trans {all_trans2_c.mean():.3f}mm")
        if loo_rot:
            print(f"  refit LOO-CV (honest out-of-sample) residual: "
                  f"rot {np.mean(loo_rot):.3f}° trans {np.mean(loo_trans):.3f}mm")

        improved = (all_rot2_c.mean() < all_rot_c.mean()) and (all_trans2_c.mean() < all_trans_c.mean())
        print(f"  refit improves both rot AND trans vs current: {improved}")

        globals()[f"_result_{ctrl_name}"] = {
            "current_bridge_path": current_bridge_path,
            "refit_bridge": refit_bridge,
            "refit_rot_deg": all_rot2_c, "refit_trans_mm": all_trans2_c,
            "n_frames": total_n, "improved": improved,
        }


if __name__ == "__main__":
    main()
