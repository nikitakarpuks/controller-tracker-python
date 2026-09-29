#!/usr/bin/env python3
"""evaluate_mocap.py -- compare the tracker's reported output against mocap
ground truth for a whole recording and compute error metrics.

Reads three CSVs per controller (all keyed by the same vision/IMU-clock
timestamp_ns -- no re-interpolation needed here, main.py already did it):
  - debug.pose_csv         -- FUSED/reported T_world_ctrl, one row per frame
                               that reached _commit_fused_solution (accepted
                               OR fusion-rejected -- see main.py)
  - debug.vision_pose_csv  -- RAW vision-only T_world_ctrl, same coverage
  - debug.mocap_log_dir/<ctrl>_mocap_gt.csv -- mocap ground truth
                               (T_headsetImu_ctrlImu), one row per PROCESSED
                               frame with mocap coverage, independent of
                               whether tracking succeeded that frame

and each controller's mocap_bridge_path (LED-reference-frame -> mocap
accelerometer-IMU-frame, see src/mocap_data.load_mocap_bridge and
compare_vision_mocap.py's module docstring for why this bridge exists and how
it was fit). A vision/fused pose composed with this bridge lands in the same
frame as the mocap ground truth, so the per-frame error is just a plain
Transform residual.

Usage: python3 evaluate_mocap.py [config/config_eval.yml]
Outputs (under the config's own debug.pose_csv directory, i.e. data/eval/ for
the shipped eval config):
  - metrics_summary.json      -- per-controller/per-source summary stats
  - per_frame_errors_<ctrl>.csv -- per-frame pos/rot error, both sources, for
                                    the plotting step (visualize_mocap_comparison.py)
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.load_config import load_yaml_config
from src.mocap_data import load_mocap_bridge
from src.transformations import Transform

CONTROLLERS = ("left_controller", "right_controller")


def load_pose_csv(path: Path) -> dict:
    """{ctrl_name: {timestamp_ns: Transform}} from a pose_csv/vision_pose_csv-
    shaped file (timestamp_ns, ctrl_name, qx,qy,qz,qw, px,py,pz, ...)."""
    out: dict = {}
    if not path.exists():
        return out
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            ts_ns = int(row[0])
            ctrl_name = row[1]
            qx, qy, qz, qw = (float(x) for x in row[2:6])
            px, py, pz = (float(x) for x in row[6:9])
            if not np.isfinite([qx, qy, qz, qw, px, py, pz]).all():
                continue  # some early accepted frames log an all-NaN quaternion
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            out.setdefault(ctrl_name, {})[ts_ns] = Transform(R, np.array([px, py, pz]))
    return out


def load_mocap_gt_csv(path: Path) -> dict:
    """{timestamp_ns: Transform} from a <ctrl>_mocap_gt.csv (see main.py's
    mocap_log_dir writer): #timestamp_ns, p_x,p_y,p_z, q_w,q_x,q_y,q_z --
    note the w-first quaternion order, unlike pose_csv's qx,qy,qz,qw."""
    out = {}
    if not path.exists():
        return out
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            ts_ns = int(row[0])
            px, py, pz = (float(x) for x in row[1:4])
            qw, qx, qy, qz = (float(x) for x in row[4:8])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            out[ts_ns] = Transform(R, np.array([px, py, pz]))
    return out


def rotation_angle_deg(R: np.ndarray) -> float:
    cos_angle = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _stats(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {"n": 0}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
    }


def longest_gap(sorted_ts_ns: list, tracked_ts_set: set) -> dict:
    """Longest consecutive run of ground-truth frames with NO tracked pose
    (a real "tracking lost" streak), in both frame count and elapsed seconds."""
    best_frames, best_s = 0, 0.0
    cur_frames, cur_start_ts = 0, None
    for ts in sorted_ts_ns:
        if ts not in tracked_ts_set:
            if cur_frames == 0:
                cur_start_ts = ts
            cur_frames += 1
            cur_s = (ts - cur_start_ts) / 1e9
            if cur_frames > best_frames:
                best_frames = cur_frames
                best_s = cur_s
        else:
            cur_frames = 0
    return {"frames": best_frames, "seconds": round(best_s, 3)}


def evaluate_controller(ctrl_name: str, fused_poses: dict, vision_poses: dict,
                         gt_poses: dict, bridge: Transform) -> dict:
    gt_ts_sorted = sorted(gt_poses.keys())
    result = {"ctrl_name": ctrl_name, "n_gt_frames": len(gt_ts_sorted)}

    for source_name, poses in (("fused", fused_poses), ("vision", vision_poses)):
        pos_err_mm, rot_err_deg, kept_ts = [], [], []
        for ts_ns, T_gt in gt_poses.items():
            T_est = poses.get(ts_ns)
            if T_est is None:
                continue
            residual = T_est.compose(bridge).inverse().compose(T_gt)
            pos_err_mm.append(float(np.linalg.norm(residual.t)) * 1000.0)
            rot_err_deg.append(rotation_angle_deg(residual.R))
            kept_ts.append(ts_ns)

        tracked_ts_set = set(kept_ts)
        n_tracked = len(kept_ts)
        result[source_name] = {
            "n_tracked_frames": n_tracked,
            "coverage_pct": round(100.0 * n_tracked / len(gt_ts_sorted), 2) if gt_ts_sorted else 0.0,
            "pos_err_mm": _stats(np.array(pos_err_mm)),
            "rot_err_deg": _stats(np.array(rot_err_deg)),
            "longest_lost_streak": longest_gap(gt_ts_sorted, tracked_ts_set),
        }

    return result


def write_per_frame_csv(out_path: Path, ctrl_name: str, fused_poses: dict, vision_poses: dict,
                         gt_poses: dict, bridge: Transform) -> None:
    """One row per mocap-covered frame: ts, elapsed_s, mocap position, and
    (when tracked) fused/vision position + per-source pos/rot error -- the
    single input visualize_mocap_comparison.py needs for every plot."""
    gt_ts_sorted = sorted(gt_poses.keys())
    t0 = gt_ts_sorted[0] if gt_ts_sorted else 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ns", "elapsed_s",
                    "mocap_x", "mocap_y", "mocap_z",
                    "fused_x", "fused_y", "fused_z", "fused_pos_err_mm", "fused_rot_err_deg",
                    "vision_x", "vision_y", "vision_z", "vision_pos_err_mm", "vision_rot_err_deg"])
        for ts_ns in gt_ts_sorted:
            T_gt = gt_poses[ts_ns]
            row = [ts_ns, f"{(ts_ns - t0) / 1e9:.4f}", *[f"{v:.6f}" for v in T_gt.t]]
            for poses in (fused_poses, vision_poses):
                T_est = poses.get(ts_ns)
                if T_est is None:
                    row += ["", "", "", "", ""]
                    continue
                T_bridged = T_est.compose(bridge)
                residual = T_bridged.inverse().compose(T_gt)
                pos_err_mm = float(np.linalg.norm(residual.t)) * 1000.0
                rot_err_deg = rotation_angle_deg(residual.R)
                row += [f"{v:.6f}" for v in T_bridged.t] + [f"{pos_err_mm:.4f}", f"{rot_err_deg:.4f}"]
            w.writerow(row)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config_eval.yml"
    config = load_yaml_config(config_path)
    debug_cfg = config.get("debug", {})

    pose_csv_path = debug_cfg.get("pose_csv")
    vision_pose_csv_path = debug_cfg.get("vision_pose_csv")
    mocap_log_dir = debug_cfg.get("mocap_log_dir")
    if not pose_csv_path or not mocap_log_dir:
        raise SystemExit("debug.pose_csv and debug.mocap_log_dir must both be set -- "
                          "run main.py with this config first")

    out_dir = Path(pose_csv_path).parent
    fused_all  = load_pose_csv(Path(pose_csv_path))
    vision_all = load_pose_csv(Path(vision_pose_csv_path)) if vision_pose_csv_path else {}

    summary = {}
    for ctrl_name in CONTROLLERS:
        gt_path = Path(mocap_log_dir) / f"{ctrl_name}_mocap_gt.csv"
        gt_poses = load_mocap_gt_csv(gt_path)
        if not gt_poses:
            print(f"[{ctrl_name}] no mocap ground-truth CSV found at {gt_path} -- skipping")
            continue

        bridge_path = config["controllers"][ctrl_name].get("mocap_bridge_path")
        if not bridge_path or not Path(bridge_path).exists():
            print(f"[{ctrl_name}] no mocap_bridge_path / file missing ({bridge_path}) -- skipping")
            continue
        bridge = load_mocap_bridge(bridge_path)

        fused_poses  = fused_all.get(ctrl_name, {})
        vision_poses = vision_all.get(ctrl_name, {})
        result = evaluate_controller(ctrl_name, fused_poses, vision_poses, gt_poses, bridge)
        summary[ctrl_name] = result

        per_frame_path = out_dir / f"per_frame_errors_{ctrl_name}.csv"
        write_per_frame_csv(per_frame_path, ctrl_name, fused_poses, vision_poses, gt_poses, bridge)

        f, v = result["fused"], result["vision"]
        print(f"\n[{ctrl_name}] {result['n_gt_frames']} mocap-covered frames")
        print(f"  fused : coverage={f['coverage_pct']}%  "
              f"pos mm  mean={f['pos_err_mm'].get('mean', float('nan')):.2f} "
              f"median={f['pos_err_mm'].get('median', float('nan')):.2f} "
              f"p90={f['pos_err_mm'].get('p90', float('nan')):.2f} "
              f"max={f['pos_err_mm'].get('max', float('nan')):.2f}  |  "
              f"rot deg  mean={f['rot_err_deg'].get('mean', float('nan')):.2f} "
              f"p90={f['rot_err_deg'].get('p90', float('nan')):.2f}  |  "
              f"longest lost streak={f['longest_lost_streak']['frames']} frames "
              f"({f['longest_lost_streak']['seconds']}s)")
        print(f"  vision: coverage={v['coverage_pct']}%  "
              f"pos mm  mean={v['pos_err_mm'].get('mean', float('nan')):.2f} "
              f"median={v['pos_err_mm'].get('median', float('nan')):.2f} "
              f"p90={v['pos_err_mm'].get('p90', float('nan')):.2f} "
              f"max={v['pos_err_mm'].get('max', float('nan')):.2f}  |  "
              f"rot deg  mean={v['rot_err_deg'].get('mean', float('nan')):.2f} "
              f"p90={v['rot_err_deg'].get('p90', float('nan')):.2f}  |  "
              f"longest lost streak={v['longest_lost_streak']['frames']} frames "
              f"({v['longest_lost_streak']['seconds']}s)")

    summary_path = out_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {summary_path}")
    print(f"Per-frame errors → {out_dir}/per_frame_errors_<ctrl_name>.csv")


if __name__ == "__main__":
    main()
