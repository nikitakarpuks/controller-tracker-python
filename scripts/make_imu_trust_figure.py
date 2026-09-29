#!/usr/bin/env python3
"""Figure for thesis section 'How Long Can the IMU Be Trusted?' (sec:coast-sweep).

Input: the raw output of the author's own sweep (repo root, untracked analysis scripts
imu_trust_analysis.py + run_imu_trust_analysis_all.py: blind dead-reckoning with
src.imu_data.predict_headset_relative_pose from a mocap-truth start at anchors every
0.1 s, 12 gap lengths 11 ms - 1 s, headset ego-motion from mocap, MOCAP_ROOM_G_WORLD,
compared to mocap truth at the end of the gap). Its CSV columns:
  recording, ctrl_name, anchor_ts_ns, dt_s, peak_gyro_dps, peak_accel_mps2, speed_mps,
  rot_err_deg, pos_err_mm

Usage:
  python3 make_imu_trust_figure.py <imu_trust_all.csv>   # recompute aggregates + figure
  python3 make_imu_trust_figure.py                       # figure from stored aggregates
Writes (TUM-THESIS/figures/): imu_trust_sweep.pdf, data/imu_trust/aggregates.csv,
  data/imu_trust/crossings.csv, data/imu_trust/summary.json

Budget curves are the committed (HEAD ec3f849) config values: base 0.066 s
(fusion_heuristic.degenerate_fallback_max_s); rotation axis: coast_trust_rot_shrink_s_per_dps
= 0.00018, coast_trust_rot_min_budget_s = 0.003, calm extension to 0.15 s below 100 deg/s;
position axis: coast_trust_shrink_s_per_mps2 = 0.001145, coast_trust_min_budget_s = 0.035
(both calm floors 0). Formula = src.imu_data.effective_coast_budget_s.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

THESIS = Path(__file__).resolve().parents[1]
FIG = THESIS / "figures"
DATA = FIG / "data" / "imu_trust"

BASE_S, ROT_SHRINK, ROT_MIN, EXT_CEIL, EXT_MAXDPS = 0.066, 0.00018, 0.003, 0.15, 100.0
POS_SHRINK, POS_MIN = 0.001145, 0.035
ROT_TOL, ROT_TOL2 = 1.0, 2.0
POS_TOL, POS_TOL2 = 100.0, 50.0

GYRO_BINS = [(0, 100), (100, 250), (250, 500), (500, 1000), (1000, 1e9)]
ACC_BINS = [(0, 5), (5, 20), (20, 40), (40, 1e9)]
FINE_GYRO = [0, 50, 100, 150, 200, 250, 350, 450, 550, 750, 1000, 1500, 1e9]
FINE_ACC = [0, 2, 5, 10, 20, 30, 40, 60, 1e9]

BLUE, VERM, GREEN, ORANGE, SKY, PINK, YELLOW, BLACK = (
    "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#F0E442", "#000000")
SEQ = [SKY, GREEN, ORANGE, VERM, "#7a0019"]

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["Linux Libertine O", "Libertinus Serif", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 6.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.4, "savefig.bbox": "tight",
})


def rot_budget(p):
    p = np.asarray(p, float)
    shrink = ROT_SHRINK * p
    ramp = np.clip(1.0 - p / EXT_MAXDPS, 0.0, None)
    extend = max(0.0, EXT_CEIL - BASE_S) * ramp
    return np.maximum(ROT_MIN, BASE_S + extend - shrink)


def pos_budget(a):
    return np.maximum(POS_MIN, BASE_S - POS_SHRINK * np.asarray(a, float))


def crossing(dts, med, tol):
    """Interpolated (log-dt) gap length where the per-dt median first exceeds tol.
    0 if already above at the shortest gap; inf if never exceeded."""
    prev = None
    for dt, v in zip(dts, med):
        if v > tol:
            if prev is None:
                return 0.0
            d0, v0 = prev
            f = (tol - v0) / (v - v0)
            return float(np.exp(np.log(d0) + f * (np.log(dt) - np.log(d0))))
        prev = (dt, v)
    return float("inf")


def aggregate(raw: pd.DataFrame):
    dts = sorted(raw.dt_s.unique())
    rows, cross = [], []
    for kind, col, err, bins_main, bins_fine in (
            ("gyro", "peak_gyro_dps", "rot_err_deg", GYRO_BINS, list(zip(FINE_GYRO[:-1], FINE_GYRO[1:]))),
            ("accel", "peak_accel_mps2", "pos_err_mm", ACC_BINS, list(zip(FINE_ACC[:-1], FINE_ACC[1:])))):
        for set_name, bins in (("main", bins_main), ("fine", bins_fine)):
            for lo, hi in bins:
                sub = raw[(raw[col] >= lo) & (raw[col] < hi)]
                if len(sub) < 200:
                    continue
                meds = []
                for dt in dts:
                    s = sub[sub.dt_s == dt][err]
                    if len(s) < 20:
                        continue
                    q25, med, q75 = np.percentile(s, [25, 50, 75])
                    rows.append(dict(kind=kind, set=set_name, lo=lo, hi=hi, dt_s=dt, n=len(s),
                                     q25=q25, median=med, q75=q75, peak_median=float(sub[col].median())))
                    meds.append((dt, med))
                if set_name == "fine" and meds:
                    dd, mm = zip(*meds)
                    t1, t2 = (ROT_TOL, ROT_TOL2) if kind == "gyro" else (POS_TOL, POS_TOL2)
                    cross.append(dict(kind=kind, lo=lo, hi=hi, n=len(sub),
                                      peak_median=float(sub[col].median()),
                                      median_at_shortest=mm[0],
                                      cross_primary_s=crossing(dd, mm, t1),
                                      cross_secondary_s=crossing(dd, mm, t2)))
    return pd.DataFrame(rows), pd.DataFrame(cross)


def label(kind, lo, hi):
    u = r"$^\circ$/s" if kind == "gyro" else r"m/s$^2$"
    if lo == 0:
        return f"<{hi:g} {u}"
    if hi >= 1e8:
        return f">{lo:g} {u}"
    return f"{lo:g}–{hi:g} {u}"


def plot(agg: pd.DataFrame, cross: pd.DataFrame):
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.9))

    for panel, kind, title, ylab, tol, tolab, tolstyle in (
            (a, "gyro", "(a) Rotation error of gyroscope dead-reckoning", "rotation error [$^\\circ$]", 5.0, r"$5^\circ$", "--"),
            (b, "accel", "(b) Position error of accelerometer dead-reckoning", "position error [mm]", POS_TOL, "100 mm", ":")):
        sub_all = agg[(agg.kind == kind) & (agg.set == "main")]
        for k, ((lo, hi), col) in enumerate(zip(sorted(set(zip(sub_all.lo, sub_all.hi))), SEQ)):
            s = sub_all[(sub_all.lo == lo) & (sub_all.hi == hi)].sort_values("dt_s")
            panel.plot(s.dt_s * 1e3, s["median"], color=col, marker="o", ms=2.2, label=label(kind, lo, hi))
            panel.fill_between(s.dt_s * 1e3, s.q25, s.q75, color=col, alpha=0.12, lw=0)
        panel.axhline(tol, color=BLACK, ls=tolstyle, lw=0.9)
        panel.text(1050, tol * 1.08, tolab, ha="right", va="bottom", fontsize=7)
        panel.set_xscale("log"); panel.set_yscale("log")
        panel.set_xlabel("gap length [ms]"); panel.set_ylabel(ylab)
        panel.set_title(title, loc="left")
        panel.set_xlim(9, 1200)
        panel.legend(title="peak gyro rate in gap" if kind == "gyro" else "peak dynamic accel. in gap",
                     title_fontsize=6.5, loc="upper left", frameon=True, framealpha=0.9,
                     edgecolor="none", facecolor="white")
    a.set_ylim(0.1, 11)
    b.set_ylim(0.5, 30000)
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    return fig


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        raw = pd.read_csv(sys.argv[1])
        agg, cross = aggregate(raw)
        agg.to_csv(DATA / "aggregates.csv", index=False)
        cross.to_csv(DATA / "crossings.csv", index=False)
        n_anchor = raw.groupby(["recording", "ctrl_name", "anchor_ts_ns"]).ngroups
        summ = {
            "samples": int(len(raw)), "anchors_recording_controller": int(n_anchor),
            "recordings": int(raw.recording.nunique()), "gap_lengths_s": [float(x) for x in sorted(raw.dt_s.unique())],
            "peak_gyro_dps_max": float(raw.peak_gyro_dps.max()), "peak_accel_mps2_max": float(raw.peak_accel_mps2.max()),
        }
        low = raw[raw.peak_gyro_dps < 500]
        for dt in (0.1, 0.2):
            s = low[low.dt_s == dt]
            summ[f"gyro_lt500_dt{int(dt*1000)}ms"] = {
                "median_rot_deg": float(s.rot_err_deg.median()), "median_pos_mm": float(s.pos_err_mm.median()),
                "frac_rot_lt10deg": float((s.rot_err_deg < 10).mean())}
        json.dump(summ, open(DATA / "summary.json", "w"), indent=1)
    else:
        agg = pd.read_csv(DATA / "aggregates.csv")
        cross = pd.read_csv(DATA / "crossings.csv")
    fig = plot(agg, cross)
    fig.savefig(FIG / "imu_trust_sweep.pdf")
    print("wrote", FIG / "imu_trust_sweep.pdf")


if __name__ == "__main__":
    main()
