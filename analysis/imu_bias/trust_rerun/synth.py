#!/usr/bin/env python3
"""Aggregates each variant's imu_trust_all.csv with the THESIS's own aggregate()/crossing() (imported from
TUM-THESIS/scripts/make_imu_trust_figure.py, not reimplemented), writes out/<v>/aggregates.csv + crossings.csv +
summary.json, gates the 'old' variant against the thesis's stored aggregates, and prints old-vs-new tables and the
comparison with the shipped coast-budget formulas. Usage: python3 synth.py <variant> [<variant> ...]"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
THESIS = Path("/home/nikitakarpuks/PyCharmProjects/controller-tracker-python/TUM-THESIS")
spec = importlib.util.spec_from_file_location("thesis_fig", THESIS / "scripts" / "make_imu_trust_figure.py")
tf = importlib.util.module_from_spec(spec); spec.loader.exec_module(tf)

SHOW_DT = [0.011, 0.035, 0.1, 0.3, 1.0]


def summarize(raw):
    n_anchor = raw.groupby(["recording", "ctrl_name", "anchor_ts_ns"]).ngroups
    s = {"samples": int(len(raw)), "anchors_recording_controller": int(n_anchor)}
    low = raw[raw.peak_gyro_dps < 500]
    for dt in (0.1, 0.2):
        x = low[low.dt_s == dt]
        s[f"gyro_lt500_dt{int(dt*1000)}ms"] = {"median_rot_deg": float(x.rot_err_deg.median()),
                                                "median_pos_mm": float(x.pos_err_mm.median()),
                                                "frac_rot_lt10deg": float((x.rot_err_deg < 10).mean())}
    return s


def run_variant(v):
    raw = pd.read_csv(OUT / v / "imu_trust_all.csv")
    agg, cross = tf.aggregate(raw)
    agg.to_csv(OUT / v / "aggregates.csv", index=False)
    cross.to_csv(OUT / v / "crossings.csv", index=False)
    summ = summarize(raw)
    json.dump(summ, open(OUT / v / "summary.json", "w"), indent=1)
    return raw, agg, cross, summ


def gate_old_vs_thesis(agg_old):
    th = pd.read_csv(THESIS / "figures" / "data" / "imu_trust" / "aggregates.csv")
    m = th.merge(agg_old, on=["kind", "set", "lo", "hi", "dt_s"], suffixes=("_th", "_new"))
    m["ratio"] = m["median_new"] / m["median_th"]
    print(f"\nGATE old-variant vs thesis aggregates: {len(m)} matched (kind,set,bin,dt) cells of {len(th)} thesis cells")
    print(f"  median-error ratio new/thesis: median {m.ratio.median():.3f}  p10 {m.ratio.quantile(.1):.3f}  p90 {m.ratio.quantile(.9):.3f}  "
          f"min {m.ratio.min():.3f} max {m.ratio.max():.3f}")
    for kind in ("gyro", "accel"):
        s = m[(m.kind == kind) & (m.set == "main")]
        print(f"  {kind} main bins: ratio median {s.ratio.median():.3f} (n cells {len(s)}); sample-count ratio median {(s.n_new/s.n_th).median():.3f}")
    print(f"  total thesis samples 214951 | old rerun samples (from summary) see below")
    return m


def table(agg, kind, title, dts=SHOW_DT):
    s = agg[(agg.kind == kind) & (agg.set == "main")]
    print(f"\n  {title}  (median error; n at 11ms in brackets)")
    hdr = "   bin".ljust(22) + "".join(f"{int(d*1000):>9d}ms" for d in dts)
    print(hdr)
    for (lo, hi), g in s.groupby(["lo", "hi"]):
        name = tf.label(kind, lo, hi).replace("$^\\circ$", "deg").replace("$^2$", "2")
        n11 = g[g.dt_s == 0.011].n.iloc[0] if (g.dt_s == 0.011).any() else 0
        vals = []
        for d in dts:
            r = g[g.dt_s == d]
            vals.append(f"{r['median'].iloc[0]:>11.2f}" if len(r) else " " * 9 + "--")
        print(f"   {name:<14s}[{n11:5d}]".ljust(22) + "".join(vals))


def crossing_table(cross, kind):
    c = cross[cross.kind == kind].sort_values("peak_median")
    print(f"   {'bin':<14s}{'n':>7s}{'peak_med':>10s}{'med@11ms':>10s}{'cross1(ms)':>12s}{'cross2(ms)':>12s}")
    for _, r in c.iterrows():
        f = lambda x: "inf" if not np.isfinite(x) else f"{x*1e3:.1f}"
        print(f"   {tf.label(kind, r.lo, r.hi).replace('$^\\circ$','deg').replace('$^2$','2'):<14s}{int(r.n):>7d}{r.peak_median:>10.1f}{r.median_at_shortest:>10.2f}{f(r.cross_primary_s):>12s}{f(r.cross_secondary_s):>12s}")


def budget_check(cross, kind, label):
    """Compare shipped coast-budget formula with the empirical crossings at each fine bin's peak median."""
    c = cross[cross.kind == kind].sort_values("peak_median")
    fx = tf.rot_budget if kind == "gyro" else tf.pos_budget
    tol1, tol2 = ("10deg", "20deg") if kind == "gyro" else ("100mm", "50mm")
    print(f"\n  [{label}] {kind}: shipped budget vs empirical trust duration (ms). budget<=cross means still conservative")
    print(f"   {'peak_med':>9s}{'budget':>9s}{'cross_' + tol1:>13s}{'cross_' + tol2:>13s}{'b/c1':>7s}{'b/c2':>7s}")
    n_ok1 = n_ok2 = n = 0
    rows = []
    for _, r in c.iterrows():
        b = float(fx(r.peak_median)) * 1e3
        c1 = r.cross_primary_s * 1e3; c2 = r.cross_secondary_s * 1e3
        ok1 = (not np.isfinite(c1)) or b <= c1 + 1e-9
        ok2 = (not np.isfinite(c2)) or b <= c2 + 1e-9
        n += 1; n_ok1 += ok1; n_ok2 += ok2
        f = lambda x: "inf" if not np.isfinite(x) else f"{x:.1f}"
        rat = lambda bb, cc: "  --" if (not np.isfinite(cc) or cc <= 0) else f"{bb/cc:.2f}"
        print(f"   {r.peak_median:>9.1f}{b:>9.1f}{f(c1):>13s}{f(c2):>13s}{rat(b, c1):>7s}{rat(b, c2):>7s}{'  <-- budget exceeds crossing' if not (ok1 and ok2) else ''}")
        rows.append((r.peak_median, b, c1, c2))
    print(f"   bins with budget <= {tol1} crossing: {n_ok1}/{n}; <= {tol2} crossing: {n_ok2}/{n}")
    return rows


def main():
    variants = sys.argv[1:]
    res = {}
    for v in variants:
        raw, agg, cross, summ = run_variant(v)
        res[v] = (raw, agg, cross, summ)
        print("=" * 110); print(f"VARIANT {v}: {summ['samples']} samples, {summ['anchors_recording_controller']} anchors")
        print(f"  gyro<500dps: dt100ms median rot {summ['gyro_lt500_dt100ms']['median_rot_deg']:.2f} deg, median pos {summ['gyro_lt500_dt100ms']['median_pos_mm']:.1f} mm, frac<10deg {summ['gyro_lt500_dt100ms']['frac_rot_lt10deg']:.3f}"
              f" | dt200ms {summ['gyro_lt500_dt200ms']['median_rot_deg']:.2f} deg, {summ['gyro_lt500_dt200ms']['median_pos_mm']:.1f} mm, {summ['gyro_lt500_dt200ms']['frac_rot_lt10deg']:.3f}")
        table(agg, "gyro", "ROTATION error (deg) by peak gyro rate")
        table(agg, "accel", "POSITION error (mm) by peak dynamic accel")
        print("\n  ROTATION crossings (fine bins)"); crossing_table(cross, "gyro")
        print("\n  POSITION crossings (fine bins)"); crossing_table(cross, "accel")
        budget_check(cross, "gyro", v); budget_check(cross, "accel", v)
    if "old" in res:
        gate_old_vs_thesis(res["old"][1])
        th = json.load(open(THESIS / "figures" / "data" / "imu_trust" / "summary.json"))
        print("\nthesis summary.json:", {k: th[k] for k in ("samples", "anchors_recording_controller", "gyro_lt500_dt100ms", "gyro_lt500_dt200ms")})


if __name__ == "__main__":
    main()
