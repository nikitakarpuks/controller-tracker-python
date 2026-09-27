#!/usr/bin/env python3
"""Runs trust_sweep.analyze for the given variants over all 8 recordings x 2 controllers (parallel), merging to
out/<variant>/imu_trust_all.csv. Usage: python3 run_variants.py old A B C C0 CLED OLDLED"""
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trust_sweep import analyze, VARIANTS, RECORDINGS_ROOT, HEADER  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
CONTROLLERS = ("left_controller", "right_controller")


def _one(variant, rec, ctrl):
    return variant, rec, ctrl, analyze(RECORDINGS_ROOT / rec, ctrl, VARIANTS[variant])


def main():
    variants = sys.argv[1:]
    recs = sorted(p.name for p in RECORDINGS_ROOT.iterdir() if p.is_dir() and p.name.startswith("euroc_recording_"))
    workers = 5
    for v in variants:
        t0 = time.time()
        (OUT / v).mkdir(parents=True, exist_ok=True)
        merged = OUT / v / "imu_trust_all.csv"
        n = 0
        with open(merged, "w", newline="") as f, ProcessPoolExecutor(max_workers=workers) as ex:
            w = csv.writer(f); w.writerow(HEADER)
            futs = [ex.submit(_one, v, r, c) for r in recs for c in CONTROLLERS]
            for fut in as_completed(futs):
                try:
                    _, rec, ctrl, rows = fut.result()
                except Exception as e:
                    print(f"[{v}] FAILED: {e}", flush=True); continue
                for row in rows:
                    w.writerow([rec, ctrl, *row])
                n += len(rows)
        print(f"[{v}] {n} rows -> {merged}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
