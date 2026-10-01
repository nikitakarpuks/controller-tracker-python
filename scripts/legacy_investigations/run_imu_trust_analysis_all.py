#!/usr/bin/env python3
"""Runs imu_trust_analysis.analyze across all 8 recordings x 2 controllers in
parallel (ProcessPoolExecutor, up to nproc workers), merges into one big CSV.

Usage: python3 run_imu_trust_analysis_all.py <out_dir>
"""
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imu_trust_analysis import analyze, RECORDINGS_ROOT

CONTROLLERS = ("left_controller", "right_controller")


def _run_one(rec_name, ctrl_name):
    rec_dir = RECORDINGS_ROOT / rec_name
    rows = analyze(rec_dir, ctrl_name)
    return rec_name, ctrl_name, rows


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    recordings = sorted(p.name for p in RECORDINGS_ROOT.iterdir()
                         if p.is_dir() and p.name.startswith("euroc_recording_"))

    combos = [(rec, ctrl) for rec in recordings for ctrl in CONTROLLERS]
    print(f"{len(combos)} (recording, controller) combos, {os.cpu_count()} cpus", flush=True)

    merged_path = out_dir / "imu_trust_all.csv"
    with open(merged_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recording", "ctrl_name", "anchor_ts_ns", "dt_s",
                     "peak_gyro_dps", "peak_accel_mps2", "speed_mps", "rot_err_deg", "pos_err_mm"])

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
            futures = {ex.submit(_run_one, rec, ctrl): (rec, ctrl) for rec, ctrl in combos}
            for fut in as_completed(futures):
                rec, ctrl = futures[fut]
                try:
                    rec_name, ctrl_name, rows = fut.result()
                except Exception as e:
                    print(f"[{rec}/{ctrl}] FAILED: {e}", flush=True)
                    continue
                for row in rows:
                    w.writerow([rec_name, ctrl_name, *row])
                f.flush()
                print(f"[{rec_name}/{ctrl_name}] {len(rows)} rows", flush=True)

    print(f"\nMerged -> {merged_path}", flush=True)


if __name__ == "__main__":
    main()
