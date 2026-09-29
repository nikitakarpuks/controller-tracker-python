#!/usr/bin/env python3
"""Recompute per-frame mocap errors for a batch-evaluation directory
(visualization/evaluate_<date>/<recording>/{pose.csv,vision_pose.csv,mocap_gt/,config.yml})
by reusing evaluate_mocap.py's own validated path (mocap bridge composed before
comparing; residual = (T_est @ bridge)^-1 @ T_gt). The batch harness only kept
summary statistics, not per-frame errors, so this regenerates them.

This is the ONLY script in TUM-THESIS/scripts that imports repo code; it is
read-only w.r.t. the repo (writes only under --out) and must be run from
anywhere (it chdirs to the repo root, where config-relative bridge paths resolve).

Usage: python3 TUM-THESIS/scripts/compute_per_frame_errors.py \
           [--src visualization/evaluate_2026-09-16] [--out TUM-THESIS/figures/data/batch_0916]
Cross-check: the recomputed n_tracked / median errors are compared against the
batch's own metrics_summary.json and any mismatch is printed.
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from evaluate_mocap import (CONTROLLERS, evaluate_controller,  # noqa: E402
                            load_mocap_gt_csv, load_pose_csv, write_per_frame_csv)
from src.load_config import load_yaml_config  # noqa: E402
from src.mocap_data import load_mocap_bridge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="visualization/evaluate_2026-09-16")
    ap.add_argument("--out", default="TUM-THESIS/figures/data/batch_0916")
    args = ap.parse_args()
    src, out = REPO / args.src, REPO / args.out

    for rec_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        name = rec_dir.name
        if not (rec_dir / "pose.csv").exists():
            continue
        cfg = load_yaml_config(str(rec_dir / "config.yml"))
        fused = load_pose_csv(rec_dir / "pose.csv")
        vision = load_pose_csv(rec_dir / "vision_pose.csv")
        o = out / name
        o.mkdir(parents=True, exist_ok=True)
        summary = {}
        for ctrl in CONTROLLERS:
            gt = load_mocap_gt_csv(rec_dir / "mocap_gt" / f"{ctrl}_mocap_gt.csv")
            if not gt:
                print(f"[{name}/{ctrl}] no mocap gt -- skipped")
                continue
            bridge = load_mocap_bridge(cfg["controllers"][ctrl]["mocap_bridge_path"])
            res = evaluate_controller(ctrl, fused.get(ctrl, {}), vision.get(ctrl, {}), gt, bridge)
            summary[ctrl] = res
            write_per_frame_csv(o / f"per_frame_errors_{ctrl}.csv", ctrl,
                                fused.get(ctrl, {}), vision.get(ctrl, {}), gt, bridge)
            # cross-check against the batch's own summary (same code path -> must match)
            ref_path = rec_dir / "metrics_summary.json"
            if ref_path.exists():
                ref = json.load(open(ref_path)).get(ctrl, {})
                rp = ref.get("pos_err_mm", {})
                ok = (ref.get("n_tracked_frames") == res["fused"]["n_tracked_frames"]
                      and abs(rp.get("median", -1) - res["fused"]["pos_err_mm"]["median"]) < 1e-6)
                print(f"[{name}/{ctrl}] gt={res['n_gt_frames']} tracked={res['fused']['n_tracked_frames']} "
                      f"cross-check vs batch summary: {'OK' if ok else 'MISMATCH'}")
        json.dump(summary, open(o / "metrics_summary_recomputed.json", "w"), indent=2)


if __name__ == "__main__":
    main()
