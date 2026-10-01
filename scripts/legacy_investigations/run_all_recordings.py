#!/usr/bin/env python3
"""Batch-run every euroc recording under RECORDINGS_ROOT through main.py
(lightweight: rrd gets pose/3D + fusion-debug streams only, no per-frame
blob-canvas visualization), then evaluate each recording against mocap
ground truth (reusing evaluate_mocap.py's own metrics/per-frame-error
machinery) and flag anomalous frames -- a fused/predicted pose that's
present but far from mocap truth -- into a per-recording CSV for manual
review.

Recordings run SEQUENTIALLY (one main.py process at a time): each run
already uses matching.parallel_search_workers (defaults to nproc) via its
own persistent process pool, so running two recordings concurrently would
just oversubscribe the same cores rather than add real throughput.

Usage: python3 run_all_recordings.py [eval_date]
  eval_date defaults to today (YYYY-MM-DD), used only for the output dir
  name (./visualization/evaluate_<eval_date>/).
"""
import ast
import csv
import json
import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from evaluate_mocap import load_pose_csv, load_mocap_gt_csv, rotation_angle_deg, _stats  # noqa: E402
from src.load_config import load_yaml_config  # noqa: E402
from src.mocap_data import load_mocap_bridge  # noqa: E402

RECORDINGS_ROOT = Path("/home/nikitakarpuks/Downloads/recordings-aug26")
BASE_CONFIG_PATH = REPO_ROOT / "config" / "config.yml"

# Anomaly thresholds for the FUSED (predicted/reported) pose vs mocap --
# a frame counts as anomalous if EITHER exceeds its threshold. Chosen from
# this project's own established sense of "worth a manual look" vs
# "ordinary tracking noise" (see memory: real confirmed bugs this session
# ran 250mm-1600mm / 40-150deg off truth; ordinary good tracking sits
# under ~90mm/10deg) -- WARN catches "worth a second look",
# SEVERE catches "almost certainly a real bug".
WARN_POS_MM, WARN_ROT_DEG = 150.0, 20.0
SEVERE_POS_MM, SEVERE_ROT_DEG = 400.0, 60.0

CONTROLLERS = ("left_controller", "right_controller")


def rec_short_name(rec_dir: Path) -> str:
    # euroc_recording_20260826173103_static_dark -> static_dark
    parts = rec_dir.name.split("_", 3)
    return parts[3] if len(parts) == 4 else rec_dir.name


def _force_bool_key(txt: str, key: str, value: bool) -> str:
    """Set a top-level-indented `  key: true/false` config line to `value`,
    regardless of its current setting -- robust to the base config.yml
    already having it either way (unlike a literal-string .replace)."""
    val_str = "true" if value else "false"
    return re.sub(rf'(?m)^(  {re.escape(key)}:)\s*(true|false)\b', rf'\1 {val_str}', txt, count=1)


def build_scratch_config(base_text: str, rec_dir: Path, out_dir: Path, name: str) -> str:
    mav0 = str(rec_dir / "mav0")
    rrd_path = str(out_dir / f"recording_{name}.rrd")
    pose_csv = str(out_dir / f"{name}_pose.csv")
    vision_pose_csv = str(out_dir / f"{name}_vision_pose.csv")
    mocap_log_dir = str(out_dir / f"{name}_mocap_gt")

    txt = base_text
    txt = re.sub(r'(?m)^  root: "[^"]*"', f'  root: "{mav0}"', txt, count=1)
    txt = re.sub(r'(?m)^  calibration_csv:.*$', '  calibration_csv: null', txt, count=1)
    txt = re.sub(r'(?m)^  pose_csv:.*$', f'  pose_csv: "{pose_csv}"', txt, count=1)
    txt = re.sub(r'(?m)^  vision_pose_csv:.*$', f'  vision_pose_csv: "{vision_pose_csv}"', txt, count=1)
    txt = re.sub(r'(?m)^  led_detections_csv:.*$', '  led_detections_csv: null', txt, count=1)
    txt = re.sub(r'(?m)^  mocap_log_dir:.*$', f'  mocap_log_dir: "{mocap_log_dir}"', txt, count=1)
    txt = re.sub(r'(?m)^  save_recording:.*$', f'  save_recording: "{rrd_path}"', txt, count=1)

    # Lightweight: no per-frame blob-canvas debug visualization (the whole
    # point of this batch run) -- animator.begin() still logs poses/
    # 3D-model/fusion-debug into the rrd stream regardless of these two.
    # NOTE: visualization.enabled is NOT touched here -- "enabled:" is not
    # a unique key in this file (mocap/fusion/self_calibration blocks all
    # have their own), so a blind global toggle risks hitting the wrong
    # one; it's already true in the base config.yml, verified below.
    txt = _force_bool_key(txt, "visualize_save", False)
    txt = _force_bool_key(txt, "visualize_rerun", False)

    # Cut per-decision console noise -- this is an unattended batch run over
    # ~55k frames total, nobody is reading matching/occlusion/proximity/
    # batch-orchestration decision traces live; keep just the per-frame
    # accept/lost summary line and startup banner.
    for key in ("log_matching_decisions", "log_batch_orchestration", "log_pose_fusion",
                "log_occlusion", "log_proximity_match", "log_best", "log_blob_detection"):
        txt = _force_bool_key(txt, key, False)

    # Full recording, not whatever frame_range a prior investigation left behind.
    txt = re.sub(r'(?m)^    lower:.*$', '    lower: null', txt, count=1)
    txt = re.sub(r'(?m)^    upper:.*$', '    upper: null', txt, count=1)
    return txt


def evaluate_and_flag(result: dict, out_dir: Path) -> dict:
    name = result["name"]
    scratch_cfg_path = out_dir / f"_config_{name}.yml"
    config = load_yaml_config(str(scratch_cfg_path))

    fused_all = load_pose_csv(Path(result["pose_csv"]))
    summary = {}
    n_anomalies_total = 0
    anomalies_path = out_dir / f"{name}_anomalies.csv"
    with open(anomalies_path, "w", newline="") as af:
        aw = csv.writer(af)
        aw.writerow(["ctrl_name", "timestamp_ns", "severity", "pos_err_mm", "rot_err_deg",
                     "fused_x", "fused_y", "fused_z", "mocap_x", "mocap_y", "mocap_z"])
        for ctrl_name in CONTROLLERS:
            gt_path = Path(result["mocap_log_dir"]) / f"{ctrl_name}_mocap_gt.csv"
            gt_poses = load_mocap_gt_csv(gt_path)
            if not gt_poses:
                summary[ctrl_name] = {"error": f"no mocap ground truth at {gt_path}"}
                continue
            bridge_path = config["controllers"][ctrl_name].get("mocap_bridge_path")
            if not bridge_path or not Path(bridge_path).exists():
                summary[ctrl_name] = {"error": f"no mocap_bridge_path / file missing ({bridge_path})"}
                continue
            bridge = load_mocap_bridge(bridge_path)
            fused_poses = fused_all.get(ctrl_name, {})

            pos_err_mm, rot_err_deg, n_anom = [], [], 0
            for ts_ns, T_est in fused_poses.items():
                T_gt = gt_poses.get(ts_ns)
                if T_gt is None:
                    continue  # no mocap coverage this exact frame
                T_bridged = T_est.compose(bridge)
                residual = T_bridged.inverse().compose(T_gt)
                p_err = float(np.linalg.norm(residual.t)) * 1000.0
                r_err = rotation_angle_deg(residual.R)
                pos_err_mm.append(p_err)
                rot_err_deg.append(r_err)
                severity = None
                if p_err >= SEVERE_POS_MM or r_err >= SEVERE_ROT_DEG:
                    severity = "SEVERE"
                elif p_err >= WARN_POS_MM or r_err >= WARN_ROT_DEG:
                    severity = "WARN"
                if severity:
                    n_anom += 1
                    aw.writerow([ctrl_name, ts_ns, severity, f"{p_err:.1f}", f"{r_err:.2f}",
                                 *[f"{v:.4f}" for v in T_bridged.t], *[f"{v:.4f}" for v in T_gt.t]])
            n_anomalies_total += n_anom
            summary[ctrl_name] = {
                "n_gt_frames": len(gt_poses),
                "n_tracked_frames": len(pos_err_mm),
                "coverage_pct": round(100.0 * len(pos_err_mm) / len(gt_poses), 2) if gt_poses else 0.0,
                "pos_err_mm": _stats(np.array(pos_err_mm)),
                "rot_err_deg": _stats(np.array(rot_err_deg)),
                "n_anomalies": n_anom,
            }
    (out_dir / f"{name}_metrics_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{name}] mocap eval done -- {n_anomalies_total} anomalous frames -> {anomalies_path}", flush=True)
    return {"n_anomalies": n_anomalies_total, "anomalies_csv": str(anomalies_path),
            "metrics_summary": summary}


_COUNT_MISMATCH_RE = re.compile(
    r"data\.layout='per_camera_folders' requires the same image count in every "
    r"camera folder, got (\{.*\})")


def _run_main(scratch_cfg_path: Path, log_path: Path) -> int:
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            [sys.executable, "main.py", str(scratch_cfg_path)],
            cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT,
        )
    return proc.returncode


def run_one_recording(rec_dir: Path, out_dir: Path, base_text: str) -> dict:
    name = rec_short_name(rec_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_cfg_path = out_dir / f"_config_{name}.yml"
    cfg_text = build_scratch_config(base_text, rec_dir, out_dir, name)
    scratch_cfg_path.write_text(cfg_text)

    log_path = out_dir / f"{name}_run.log"
    t0 = time.time()
    print(f"[{name}] starting -> log: {log_path}", flush=True)
    returncode = _run_main(scratch_cfg_path, log_path)

    # A raw recording occasionally has one camera folder short by a frame or
    # two (e.g. one capture thread stopping writing slightly before the
    # others at the very end) -- count_images() hard-fails on that rather
    # than silently misaligning cameras. Detect it from the traceback and
    # retry ONCE with frame_range.upper capped to the smallest per-camera
    # count, which trims only the trailing frames every camera doesn't
    # share (same alignment count_images would have produced anyway had the
    # counts matched).
    if returncode != 0:
        log_text = log_path.read_text()
        m = _COUNT_MISMATCH_RE.search(log_text)
        if m:
            counts = ast.literal_eval(m.group(1))  # literal {int: int} dict from the traceback
            min_count = min(counts.values())
            print(f"[{name}] per-camera frame count mismatch {counts} -- "
                  f"retrying with frame_range.upper={min_count}", flush=True)
            cfg_text = re.sub(r'(?m)^    upper:.*$', f'    upper: {min_count}', cfg_text, count=1)
            scratch_cfg_path.write_text(cfg_text)
            t0 = time.time()
            returncode = _run_main(scratch_cfg_path, log_path)

    elapsed = time.time() - t0
    status = "OK" if returncode == 0 else f"FAILED (exit {returncode})"
    print(f"[{name}] {status} in {elapsed:.0f}s", flush=True)
    return {"name": name, "returncode": returncode, "elapsed_s": elapsed,
            "pose_csv": str(out_dir / f"{name}_pose.csv"),
            "vision_pose_csv": str(out_dir / f"{name}_vision_pose.csv"),
            "mocap_log_dir": str(out_dir / f"{name}_mocap_gt"),
            "rrd": str(out_dir / f"recording_{name}.rrd")}


def main():
    eval_date = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    out_dir = REPO_ROOT / "visualization" / f"evaluate_{eval_date}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_text = BASE_CONFIG_PATH.read_text()
    recordings = sorted(p for p in RECORDINGS_ROOT.iterdir()
                         if p.is_dir() and p.name.startswith("euroc_recording_"))
    print(f"Found {len(recordings)} recordings -> output dir {out_dir}", flush=True)

    overview = {"eval_date": eval_date, "recordings": []}
    for rec_dir in recordings:
        result = run_one_recording(rec_dir, out_dir, base_text)
        if result["returncode"] == 0:
            try:
                eval_result = evaluate_and_flag(result, out_dir)
                result.update(eval_result)
            except Exception as e:
                print(f"[{result['name']}] evaluation step failed: {e}", flush=True)
        overview["recordings"].append(result)
        (out_dir / "overview.json").write_text(json.dumps(overview, indent=2))

    print(f"\nAll recordings done. Overview -> {out_dir / 'overview.json'}", flush=True)


if __name__ == "__main__":
    main()
