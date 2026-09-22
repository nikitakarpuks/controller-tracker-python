#!/usr/bin/env python3
"""fit_deploy_drift.py -- final fit of the DEPLOYABLE linear drift model (one intercept + one rate
per controller, to ship in config.yml / src/mocap_data.py), pivoted at each recording's own device
mocap-track start (ctrl_mocap.t_ns[0]) so the model is a pure function of (query_ts_ns - t_ns[0])
and needs no extra per-recording bookkeeping at runtime.

Uses the SAME validated machinery as run_drift_test.py (raw_seed_residuals / robust_bridge / score,
imported directly -- not reimplemented) but fits ONE model per controller pooled across the 3
recordings that have fresh, current-pipeline (post vision_offset_ns fix) vision poses: static_dark,
static_medium, static_hard. The other 5 recordings-aug26 sets only have pre-fix batch vision poses
and are deliberately excluded from this deployment fit (they were fine for validating that drift
exists, in the earlier held-out test, but re-deriving a constant to SHIP from stale-pipeline vision
would be circular).

Also prints each recording's OWN separate fit (not pooled) so the pooled choice can be checked for
consistency before shipping it.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_drift_test import (load_device_mocap, load_strong_vision_poses, strong_thresholds,
                             raw_seed_residuals, robust_bridge, score, gi, GRID_MS)
from src.load_config import load_yaml_config
from src.mocap_data import load_vision_offset_ns

RECORDINGS = [
    dict(name="static_dark", recording_dir="/home/nikitakarpuks/Downloads/recordings-aug26/euroc_recording_20260826173103_static_dark",
         vision_pose_csv="data/vision_pose_log.csv", config="config/config.yml"),
    dict(name="static_medium", recording_dir="/home/nikitakarpuks/Downloads/recordings-aug26/euroc_recording_20260826174213_static_medium",
         vision_pose_csv="analysis/drift_test/static_medium_run/vision_pose.csv", config="analysis/drift_test/static_medium_run/config.yml"),
    dict(name="static_hard", recording_dir="/home/nikitakarpuks/Downloads/recordings-aug26/euroc_recording_20260826174510_static_hard",
         vision_pose_csv="analysis/drift_test/static_hard_run/vision_pose.csv", config="analysis/drift_test/static_hard_run/config.yml"),
]
CONTROLLERS = ("left_controller", "right_controller")


def eval_model(rows, t_pivot_s, a_ms, slope_ms_per_min, idx_all):
    idxs = [(i, gi(a_ms + slope_ms_per_min * t_pivot_s[i] / 60.0)) for i in idx_all]
    bridge = robust_bridge(rows, idxs, np.ones(len(rows)))
    rt, rr = score(rows, idxs, bridge)
    return float(np.median(rt + 3 * rr)), bridge, rt, rr


def fit_one(rows, t_pivot_s, idx_all):
    """Grid-search (a_ms, slope) anchored at t_pivot_s=0 (the device mocap track's own start) --
    NOT at the data's own mean time, so the fitted a_ms is directly the intercept to ship."""
    def cost(a, s):
        return eval_model(rows, t_pivot_s, a, s, idx_all)[0]
    best_a = min(np.arange(-6.0, 6.01, 1.0), key=lambda a: cost(a, 0.0))
    best_a = min(np.arange(best_a - 1.0, best_a + 1.01, 0.5), key=lambda a: cost(a, 0.0))
    coarse = [((a, s), cost(a, s)) for a in np.arange(best_a - 4.0, best_a + 4.01, 1.0) for s in np.arange(-1.0, 6.01, 1.0)]
    (a0, s0), _ = min(coarse, key=lambda x: x[1])
    fine = [((a, s), cost(a, s)) for a in np.arange(a0 - 1.0, a0 + 1.01, 0.5) for s in np.arange(s0 - 1.0, s0 + 1.01, 0.5)]
    (a_best, s_best), _ = min(fine, key=lambda x: x[1])
    return a_best, s_best


def main():
    out_csv = Path("analysis/drift_test/deploy_drift_fit.csv")
    out_rows = []
    for ctrl in CONTROLLERS:
        print(f"\n===== {ctrl} =====")
        pooled_rows, pooled_t = [], []
        per_rec = {}
        for rec in RECORDINGS:
            config = load_yaml_config(rec["config"])
            recording_root = Path(rec["recording_dir"])
            hm = load_device_mocap(recording_root, "headset", config, vision_offset_ns=0.0)
            si, se = strong_thresholds(config, ctrl)
            poses, errors = load_strong_vision_poses(Path(rec["vision_pose_csv"]), ctrl, si, se)
            voff = load_vision_offset_ns(config["controllers"][ctrl])
            cm = load_device_mocap(recording_root, ctrl, config, vision_offset_ns=voff)
            kept_ts, rows = raw_seed_residuals(poses, hm, cm)
            if len(kept_ts) < 40:
                print(f"  [{rec['name']}] only {len(kept_ts)} frames -- skipping")
                continue
            kept_ts = np.array(kept_ts)
            t_pivot_s = (kept_ts - cm.t_ns[0]) / 1e9  # anchored at THIS device's mocap-track start
            idx_all = np.arange(len(kept_ts))
            a, s = fit_one(rows, t_pivot_s, idx_all)
            cost0, bridge0, rt0, rr0 = eval_model(rows, t_pivot_s, 0.0, 0.0, idx_all)   # current pipeline
            cost1, bridge1, rt1, rr1 = eval_model(rows, t_pivot_s, a, s, idx_all)       # fitted drift model
            print(f"  [{rec['name']}] n={len(kept_ts)}  pivot=t_ns[0] (mocap track start, "
                  f"{t_pivot_s.min():.1f}..{t_pivot_s.max():.1f}s since it)")
            print(f"    fitted: a={a:+.2f}ms slope={s:+.2f}ms/min  |  zero-model trans {rt0.mean():.2f}mm rot {rr0.mean():.2f}deg"
                  f"  ->  fitted-model trans {rt1.mean():.2f}mm rot {rr1.mean():.2f}deg")
            per_rec[rec["name"]] = dict(a=a, s=s, n=len(kept_ts))
            out_rows.append(dict(controller=ctrl, recording=rec["name"], n=len(kept_ts),
                                  fitted_a_ms=round(a, 3), fitted_slope_ms_per_min=round(s, 3),
                                  zero_trans_mean_mm=round(float(rt0.mean()), 3), zero_rot_mean_deg=round(float(rr0.mean()), 3),
                                  fitted_trans_mean_mm=round(float(rt1.mean()), 3), fitted_rot_mean_deg=round(float(rr1.mean()), 3)))
            pooled_rows.append((rows, t_pivot_s, idx_all, len(kept_ts)))

        # Pooled fit: search (a, slope) minimizing the N-weighted sum of each recording's own median cost
        # (each recording keeps its OWN pivot=t_ns[0], so a single (a, slope) pair is genuinely shared).
        def pooled_cost(a, s):
            tot, n_tot = 0.0, 0
            for rows, t_pivot_s, idx_all, n in pooled_rows:
                c, _, _, _ = eval_model(rows, t_pivot_s, a, s, idx_all)
                tot += c * n
                n_tot += n
            return tot / n_tot

        best_a = min(np.arange(-4.0, 4.01, 0.5), key=lambda a: pooled_cost(a, 0.0))
        coarse = [((a, s), pooled_cost(a, s)) for a in np.arange(best_a - 2.0, best_a + 2.01, 0.5) for s in np.arange(0.0, 5.01, 0.5)]
        (a_p, s_p), _ = min(coarse, key=lambda x: x[1])
        print(f"\n  POOLED (n-weighted over {len(pooled_rows)} recordings): a={a_p:+.2f}ms  slope={s_p:+.2f}ms/min")
        for rows, t_pivot_s, idx_all, n in pooled_rows:
            c0, _, rt0, rr0 = eval_model(rows, t_pivot_s, 0.0, 0.0, idx_all)
            c1, _, rt1, rr1 = eval_model(rows, t_pivot_s, a_p, s_p, idx_all)
            print(f"    applied to n={n}: zero trans {rt0.mean():.2f}mm rot {rr0.mean():.2f}deg  ->  "
                  f"pooled-model trans {rt1.mean():.2f}mm rot {rr1.mean():.2f}deg")
        out_rows.append(dict(controller=ctrl, recording="POOLED", n=sum(r[3] for r in pooled_rows),
                              fitted_a_ms=round(a_p, 3), fitted_slope_ms_per_min=round(s_p, 3),
                              zero_trans_mean_mm="", zero_rot_mean_deg="", fitted_trans_mean_mm="", fitted_rot_mean_deg=""))

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {len(out_rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
