#!/usr/bin/env python3
"""Bins imu_trust_all.csv (from run_imu_trust_analysis_all.py) by motion-speed
bucket and dt, reports error-vs-dt curves and the trust-duration (max dt
before error crosses a tolerance) per bucket -- for BOTH gyro-only rotation
dead-reckoning and gyro+accel position dead-reckoning.

Usage: python3 synthesize_imu_trust.py <imu_trust_all.csv>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GYRO_BUCKETS = [(0, 100, "calm(<100dps)"), (100, 500, "slow(100-500)"),
                (500, 1500, "moderate(500-1500)"), (1500, 3000, "fast(1500-3000)"),
                (3000, 1e9, "extreme(>3000)")]
ACCEL_BUCKETS = [(0, 2, "calm(<2m/s2)"), (2, 8, "slow(2-8)"),
                  (8, 20, "moderate(8-20)"), (20, 40, "fast(20-40)"),
                  (40, 1e9, "extreme(>40)")]

ROT_TOLERANCES_DEG = [5.0, 10.0, 20.0]
POS_TOLERANCES_MM = [20.0, 50.0, 100.0]


def bucketize(series, buckets):
    labels = np.full(len(series), "", dtype=object)
    for lo, hi, name in buckets:
        mask = (series >= lo) & (series < hi)
        labels[mask.to_numpy()] = name
    return labels


def trust_duration(df_bucket, dt_col, err_col, tolerance, agg="median"):
    """Max dt at which the aggregate (median or p90) error is still <=
    tolerance, walking dt from smallest to largest and stopping at the first
    violation (so a later dt with an anomalously-low error, e.g. sparse-
    sample noise, doesn't falsely extend the window past a real earlier
    breach)."""
    dts = sorted(df_bucket[dt_col].unique())
    best = 0.0
    for dt in dts:
        sub = df_bucket[df_bucket[dt_col] == dt][err_col]
        if len(sub) < 5:
            continue
        val = sub.median() if agg == "median" else sub.quantile(0.9)
        if val <= tolerance:
            best = dt
        else:
            break
    return best


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "imu_trust_all.csv"
    df = pd.read_csv(path)
    print(f"loaded {len(df)} rows from {path}\n")

    df["gyro_bucket"] = bucketize(df["peak_gyro_dps"], GYRO_BUCKETS)
    df["accel_bucket"] = bucketize(df["peak_accel_mps2"], ACCEL_BUCKETS)

    print("="*100)
    print("GYRO-ONLY ROTATION DEAD-RECKONING -- median/p90 rot_err_deg vs dt, by peak_gyro_dps bucket")
    print("="*100)
    for lo, hi, name in GYRO_BUCKETS:
        sub = df[df["gyro_bucket"] == name]
        if len(sub) < 20:
            print(f"\n[{name}] n={len(sub)} -- too few samples, skipping")
            continue
        print(f"\n[{name}] n={len(sub)}")
        print(f"  {'dt(s)':>7} {'n':>6} {'median_deg':>11} {'p90_deg':>9} {'max_deg':>9}")
        for dt in sorted(sub["dt_s"].unique()):
            s = sub[sub["dt_s"] == dt]["rot_err_deg"]
            print(f"  {dt:7.3f} {len(s):6d} {s.median():11.3f} {s.quantile(0.9):9.3f} {s.max():9.2f}")
        print(f"  trust duration (median <= tolerance):")
        for tol in ROT_TOLERANCES_DEG:
            td = trust_duration(sub, "dt_s", "rot_err_deg", tol, agg="median")
            print(f"    <= {tol:5.1f} deg: {td*1000:.0f} ms")
        print(f"  trust duration (p90 <= tolerance):")
        for tol in ROT_TOLERANCES_DEG:
            td = trust_duration(sub, "dt_s", "rot_err_deg", tol, agg="p90")
            print(f"    <= {tol:5.1f} deg: {td*1000:.0f} ms")

    print("\n" + "="*100)
    print("GYRO+ACCEL POSITION DEAD-RECKONING -- median/p90 pos_err_mm vs dt, by peak_accel_mps2 bucket")
    print("="*100)
    for lo, hi, name in ACCEL_BUCKETS:
        sub = df[df["accel_bucket"] == name]
        if len(sub) < 20:
            print(f"\n[{name}] n={len(sub)} -- too few samples, skipping")
            continue
        print(f"\n[{name}] n={len(sub)}")
        print(f"  {'dt(s)':>7} {'n':>6} {'median_mm':>10} {'p90_mm':>8} {'max_mm':>9}")
        for dt in sorted(sub["dt_s"].unique()):
            s = sub[sub["dt_s"] == dt]["pos_err_mm"]
            print(f"  {dt:7.3f} {len(s):6d} {s.median():10.2f} {s.quantile(0.9):8.2f} {s.max():9.1f}")
        print(f"  trust duration (median <= tolerance):")
        for tol in POS_TOLERANCES_MM:
            td = trust_duration(sub, "dt_s", "pos_err_mm", tol, agg="median")
            print(f"    <= {tol:5.1f} mm: {td*1000:.0f} ms")
        print(f"  trust duration (p90 <= tolerance):")
        for tol in POS_TOLERANCES_MM:
            td = trust_duration(sub, "dt_s", "pos_err_mm", tol, agg="p90")
            print(f"    <= {tol:5.1f} mm: {td*1000:.0f} ms")

    print("\n" + "="*100)
    print("CORRELATION CHECK: does trust duration shrink monotonically with speed?")
    print("="*100)
    print("\nGYRO trust duration (median<=10deg) vs peak_gyro_dps bucket:")
    for lo, hi, name in GYRO_BUCKETS:
        sub = df[df["gyro_bucket"] == name]
        if len(sub) < 20:
            continue
        td = trust_duration(sub, "dt_s", "rot_err_deg", 10.0, agg="median")
        print(f"  {name:22} -> {td*1000:.0f} ms")
    print("\nIMU (accel) trust duration (median<=50mm) vs peak_accel_mps2 bucket:")
    for lo, hi, name in ACCEL_BUCKETS:
        sub = df[df["accel_bucket"] == name]
        if len(sub) < 20:
            continue
        td = trust_duration(sub, "dt_s", "pos_err_mm", 50.0, agg="median")
        print(f"  {name:22} -> {td*1000:.0f} ms")


if __name__ == "__main__":
    main()
