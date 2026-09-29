#!/usr/bin/env python3
"""Evaluation-chapter figures, tables and numbers from per-frame mocap errors.

Reads (no repo imports; numpy/pandas/matplotlib only):
  TUM-THESIS/figures/data/batch_0916/<rec>/per_frame_errors_{left,right}_controller.csv
      (written by scripts/compute_per_frame_errors.py from the 2026-09-16 batch run;
       error = (T_est @ mocap_bridge)^-1 @ T_mocap, i.e. bridge composed before comparing)
  TUM-THESIS/figures/data/old_0905/{static_dark,walk_dark}/per_frame_errors_*.csv
      (preliminary 2026-09-05 runs, kept for the old-vs-new comparison)
  optionally visualization/evaluate_2026-09-16/<rec>/{pose,vision_pose}.csv (read-only)
      for the gross-error classification (inlier counts / confidence).
Writes:
  TUM-THESIS/figures/*.pdf, figures/eval_numbers.json, figures/tables/*.tex

Definitions (also stored in eval_numbers.json):
  frame            one mocap-covered processed camera frame (ground-truth frame)
  tracked          a reported pose exists for that frame (coverage = tracked / GT frames)
  gross error      tracked frame with position error > 150 mm OR rotation error > 20 deg
                   (the repo's own WARN thresholds; also the trough of the pooled error
                   distribution: only 0.09 % of tracked frames lie between 75 and 200 mm)
  longest lost     evaluate_mocap.py's definition: longest run of consecutive GT frames
                   without a pose; seconds = first-to-last lost frame timestamp span
  blind time       (histogram only) time from the last tracked frame before a lost run to the
                   first tracked frame after it
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

THESIS = Path(__file__).resolve().parents[1]
REPO = THESIS.parent
FIG = THESIS / "figures"
BATCH = FIG / "data" / "batch_0916"
OLD = FIG / "data" / "old_0905"
BATCH_SRC = REPO / "visualization" / "evaluate_2026-09-16"
TABLES = FIG / "tables"

POS_OUT_MM, ROT_OUT_DEG = 150.0, 20.0
SWAP_RADIUS_MM = 100.0
POST_GAP_FRAMES = 30
RECS = ["static_dark", "static_easy", "static_medium", "static_hard",
        "walk_dark", "walk_easy", "walk_medium", "walk_hard"]
CTRLS = ["left", "right"]

# Okabe-Ito colour-blind-safe palette
BLUE, VERM, GREEN, ORANGE, SKY, PINK, YELLOW, BLACK = (
    "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#F0E442", "#000000")
CTRL_COLOR = {"left": BLUE, "right": VERM}
REC_COLOR = {"dark": BLACK, "easy": SKY, "medium": ORANGE, "hard": VERM}

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["Linux Libertine O", "Libertinus Serif", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.4, "savefig.bbox": "tight",
})


# ----------------------------------------------------------------------------- data
def load(root: Path, rec: str, ctrl: str):
    p = root / rec / f"per_frame_errors_{ctrl}_controller.csv"
    return pd.read_csv(p) if p.exists() else None


def stats(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "max": float(x.max()),
            "rmse": float(np.sqrt(np.mean(x ** 2)))}


def lost_streaks(ts_ns, tracked):
    ts_ns = np.asarray(ts_ns)
    streaks, i, n = [], 0, len(tracked)
    t0 = ts_ns[0] if n else 0
    while i < n:
        if tracked[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and not tracked[j + 1]:
            j += 1
        before = ts_ns[i - 1] if i > 0 else None
        after = ts_ns[j + 1] if j + 1 < n else None
        streaks.append({
            "frames": j - i + 1, "span_s": (ts_ns[j] - ts_ns[i]) / 1e9,
            "blind_s": (after - before) / 1e9 if (before is not None and after is not None) else None,
            "t_from_s": ((before if before is not None else ts_ns[i]) - t0) / 1e9,
            "t_to_s": ((after if after is not None else ts_ns[j]) - t0) / 1e9})
        i = j + 1
    return streaks


def summarize(df: pd.DataFrame) -> dict:
    out = {"n_gt_frames": int(len(df))}
    ts = df["timestamp_ns"].values
    for src in ("fused", "vision"):
        ep_all, er_all = df[f"{src}_pos_err_mm"].values, df[f"{src}_rot_err_deg"].values
        trk = ~np.isnan(ep_all)
        ep, er = ep_all[trk], er_all[trk]
        gross = (ep > POS_OUT_MM) | (er > ROT_OUT_DEG)
        streaks = lost_streaks(ts, trk)
        longest = max(streaks, key=lambda s: s["frames"]) if streaks else {"frames": 0, "span_s": 0.0}
        out[src] = {
            "n_tracked": int(trk.sum()), "coverage_pct": float(100 * trk.mean()) if len(trk) else None,
            "pos_mm": stats(ep), "rot_deg": stats(er),
            "gross_n": int(gross.sum()), "gross_pct": float(100 * gross.mean()) if len(gross) else None,
            "pos_mm_nongross": stats(ep[~gross]), "rot_deg_nongross": stats(er[~gross]),
            "n_lost_streaks": len(streaks),
            "longest_lost_frames": int(longest["frames"]), "longest_lost_s": float(longest["span_s"])}
    both = ~np.isnan(df["fused_pos_err_mm"].values) & ~np.isnan(df["vision_pos_err_mm"].values)
    paired = {"n_both": int(both.sum())}
    for name, f_col, v_col in (("pos_mm", "fused_pos_err_mm", "vision_pos_err_mm"),
                               ("rot_deg", "fused_rot_err_deg", "vision_rot_err_deg")):
        d = (df[v_col].values - df[f_col].values)[both]     # >0: fused better
        eps = 1e-3
        diff = d[np.abs(d) > eps]
        paired[name] = {
            "frac_fused_better": float((d > eps).mean()) if d.size else None,
            "frac_fused_worse": float((d < -eps).mean()) if d.size else None,
            "frac_identical": float((np.abs(d) <= eps).mean()) if d.size else None,
            "median_improvement_on_differing": float(np.median(diff)) if diff.size else None,
            "mean_improvement": float(d.mean()) if d.size else None}
    out["paired_fused_vs_vision"] = paired
    return out


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


# ----------------------------------------------------------------------- gross errors
def classify_gross(frames):
    """Why are gross-error frames wrong? frames: {(rec, ctrl): DataFrame}."""
    rows, pooled_in, pooled_out = [], [], []
    for rec in RECS:
        if (rec, "left") not in frames or (rec, "right") not in frames:
            continue
        pose = vis = None
        if (BATCH_SRC / rec / "pose.csv").exists():
            pose = pd.read_csv(BATCH_SRC / rec / "pose.csv")
            vis = pd.read_csv(BATCH_SRC / rec / "vision_pose.csv")
        for c, o in (("left", "right"), ("right", "left")):
            d = frames[(rec, c)].copy()
            other = frames[(rec, o)].set_index("timestamp_ns")[["mocap_x", "mocap_y", "mocap_z"]]
            other.columns = ["o_x", "o_y", "o_z"]
            d = d.join(other, on="timestamp_ns")
            trk = d["fused_pos_err_mm"].notna().values
            since, cnt = [], 10 ** 9
            for t in trk:
                cnt = 0 if not t else cnt + 1
                since.append(cnt)
            d["since_gap"] = since
            d["gross"] = trk & ((d.fused_pos_err_mm > POS_OUT_MM) | (d.fused_rot_err_deg > ROT_OUT_DEG))
            d["dist_other_mm"] = 1000 * np.sqrt((d.fused_x - d.o_x) ** 2 + (d.fused_y - d.o_y) ** 2
                                                + (d.fused_z - d.o_z) ** 2)
            if pose is not None:
                v = vis[vis.ctrl_name == f"{c}_controller"].drop_duplicates("timestamp_ns", keep="last") \
                    .set_index("timestamp_ns")
                d = d.join(v[["confidence", "error_px", "n_inliers"]], on="timestamp_ns")
            g = d[d.gross]
            row = {"rec": rec, "ctrl": c, "n_tracked": int(trk.sum()), "n_gross": int(len(g)),
                   "near_other_controller": int((g.dist_other_mm < SWAP_RADIUS_MM).sum()),
                   "vision_also_gross": int(((g.vision_pos_err_mm > POS_OUT_MM)
                                             | (g.vision_rot_err_deg > ROT_OUT_DEG)).sum()),
                   "within_30_frames_after_lost": int((g.since_gap <= POST_GAP_FRAMES).sum())}
            if "n_inliers" in d:
                row["looks_strong_n_inliers>=8_err<0.5px"] = int(((g.n_inliers >= 8) & (g.error_px < 0.5)).sum())
                pooled_out.append(g[["n_inliers", "confidence", "error_px"]])
                pooled_in.append(d[d.fused_pos_err_mm.notna() & ~d.gross][["n_inliers", "confidence", "error_px"]])
            rows.append(row)
    tot = {k: int(sum(r.get(k, 0) for r in rows)) for k in rows[0] if k not in ("rec", "ctrl")}
    res = {"per_recording": rows, "pooled": tot,
           "definitions": {"near_other_controller": f"fused position within {SWAP_RADIUS_MM} mm of the OTHER "
                                                     "controller's mocap position at the same frame (identity swap)",
                           "within_30_frames_after_lost": "<=30 GT frames since the last un-tracked GT frame"}}
    if pooled_out:
        po, pi = pd.concat(pooled_out), pd.concat(pooled_in)
        res["pooled"]["vision_quality_median"] = {
            "gross": {c: float(po[c].median()) for c in po}, "non_gross": {c: float(pi[c].median()) for c in pi}}
    return res


# ------------------------------------------------------------------------------ plots
def cdf_xy(x):
    x = np.sort(np.asarray(x, dtype=float))
    return x, np.arange(1, len(x) + 1) / len(x)


def fig_cdf_static_dark(frames, path):
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.5))
    for ax, key, xlab, thr, lo, hi in ((axes[0], "pos_err_mm", "Position error [mm]", POS_OUT_MM, 0.3, 3000),
                                       (axes[1], "rot_err_deg", "Rotation error [deg]", ROT_OUT_DEG, 0.05, 200)):
        for c in CTRLS:
            df = frames[("static_dark", c)]
            for src, ls in (("fused", "-"), ("vision", "--")):
                x, y = cdf_xy(df[f"{src}_{key}"].dropna())
                ax.plot(x, y, ls=ls, color=CTRL_COLOR[c], label=f"{c} / {src}")
        ax.axvline(thr, color="k", ls=":", lw=0.7)
        ax.set_xscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(0, 1.005)
        ax.set_xlabel(xlab); ax.set_ylabel("cumulative fraction")
    axes[1].legend(loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_cdf_all(frames, path):
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 3.0))
    for ax, key, xlab, thr, lo, hi in ((axes[0], "fused_pos_err_mm", "Position error [mm]", POS_OUT_MM, 0.5, 3000),
                                       (axes[1], "fused_rot_err_deg", "Rotation error [deg]", ROT_OUT_DEG, 0.1, 200)):
        for rec in RECS:
            if (rec, "left") not in frames:
                continue
            x = pd.concat([frames[(rec, c)][key].dropna() for c in CTRLS])
            x, y = cdf_xy(x)
            kind = rec.split("_")[1]
            ax.plot(x, y, color=REC_COLOR[kind], ls="-" if rec.startswith("static") else "--", label=rec.replace("_", " "))
        ax.axvline(thr, color="k", ls=":", lw=0.7)
        ax.set_xscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(0, 1.005)
        ax.set_xlabel(xlab); ax.set_ylabel("cumulative fraction")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, columnspacing=1.2)
    fig.tight_layout(rect=[0, 0.13, 1, 1]); fig.savefig(path); plt.close(fig)


def fig_error_time(df, path, title):
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 3.6), sharex=True)
    t = df["elapsed_s"].values
    pos, rot = df["fused_pos_err_mm"].values, df["fused_rot_err_deg"].values
    trk = ~np.isnan(pos)
    gross = trk & ((pos > POS_OUT_MM) | (rot > ROT_OUT_DEG))
    streaks = lost_streaks(df["timestamp_ns"].values, trk)
    for ax, y, thr, lab, lo, hi in ((axes[0], pos, POS_OUT_MM, "Position error [mm]", 0.3, 3000),
                                    (axes[1], rot, ROT_OUT_DEG, "Rotation error [deg]", 0.05, 200)):
        for s in streaks:
            ax.axvspan(s["t_from_s"], s["t_to_s"], color="0.82", lw=0, zorder=0)
        ax.plot(t[trk & ~gross], y[trk & ~gross], ".", ms=1.3, color=BLUE, zorder=2)
        ax.plot(t[gross], y[gross], ".", ms=2.6, color=VERM, zorder=3)
        ax.axhline(thr, color="k", ls=":", lw=0.7)
        ax.set_yscale("log"); ax.set_ylim(lo, hi); ax.set_ylabel(lab)
    axes[1].set_xlabel("time [s]"); axes[0].set_title(title, loc="left")
    fig.legend(handles=[Line2D([], [], marker=".", ls="", color=BLUE, label="tracked"),
                        Line2D([], [], marker=".", ls="", color=VERM, label="gross error (dotted lines)"),
                        Patch(color="0.82", label="no pose (lost)")],
               loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=[0, 0.06, 1, 1]); fig.savefig(path); plt.close(fig)


def fig_lost_hist(frames, path):
    groups = {"static": [], "walk": []}
    for rec in RECS:
        for c in CTRLS:
            if (rec, c) not in frames:
                continue
            df = frames[(rec, c)]
            for s in lost_streaks(df["timestamp_ns"].values, df["fused_pos_err_mm"].notna().values):
                if s["blind_s"] is not None:
                    groups[rec.split("_")[0]].append(s["blind_s"])
    bins = np.logspace(np.log10(0.028), np.log10(8), 17)
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    for name, col in (("static", BLUE), ("walk", VERM)):
        v = np.array(groups[name])
        ax.hist(v, bins=bins, histtype="stepfilled", alpha=0.45, color=col, lw=0)
        ax.hist(v, bins=bins, histtype="step", color=col, lw=1.0,
                label=f"{name}  (n={len(v)}, median {np.median(v):.2f} s, max {v.max():.1f} s)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.7, 2000); ax.set_xlabel("blind time of a lost streak [s]")
    ax.set_ylabel("number of streaks"); ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return {k: {"n": len(v), "median_s": float(np.median(v)), "p90_s": float(np.percentile(v, 90)),
                "max_s": float(np.max(v))} for k, v in ((k, np.array(v)) for k, v in groups.items())}


# ----------------------------------------------------------------------------- tables
def f1(x, d=1):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def tex_rec(rec):
    return r"\texttt{" + rec.replace("_", r"\_") + "}"


def table_rec(S, rec, path):
    lines = [r"\begin{tabular}{@{}llrrrrrrrrr@{}}", r"\toprule",
             r" & & Cov. & \multicolumn{3}{c}{Position error [mm]} & \multicolumn{3}{c}{Rotation error [$^\circ$]}"
             r" & Gross & Longest \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
             r"Ctrl. & Source & [\%] & med. & p95 & p99 & med. & p95 & p99 & err.\ [\%] & lost [s] \\", r"\midrule"]
    for ci, c in enumerate(CTRLS):
        for si, src in enumerate(("fused", "vision")):
            s = S[(rec, c)][src]
            lines.append(" & ".join([c if si == 0 else "", src, f1(s["coverage_pct"]),
                                     f1(s["pos_mm"]["median"]), f1(s["pos_mm"]["p95"]), f1(s["pos_mm"]["p99"]),
                                     f1(s["rot_deg"]["median"]), f1(s["rot_deg"]["p95"]), f1(s["rot_deg"]["p99"]),
                                     f1(s["gross_pct"], 2), f1(s["longest_lost_s"], 2)]) + r" \\")
        if ci == 0:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def table_all(S, path):
    lines = [r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
             r"Recording & Ctrl. & Cov. [\%] & \multicolumn{2}{c}{Pos.\ err.\ [mm]} & \multicolumn{2}{c}{Rot.\ err.\ [$^\circ$]}"
             r" & Gross [\%] & Longest lost [s] & GT frames \\",
             r" & & & med. & p95 & med. & p95 & & & \\", r"\midrule"]
    for ri, rec in enumerate(RECS):
        if (rec, "left") not in S:
            continue
        for ci, c in enumerate(CTRLS):
            s = S[(rec, c)]["fused"]
            lines.append(" & ".join([tex_rec(rec) if ci == 0 else "", c, f1(s["coverage_pct"]),
                                     f1(s["pos_mm"]["median"]), f1(s["pos_mm"]["p95"]),
                                     f1(s["rot_deg"]["median"]), f1(s["rot_deg"]["p95"]),
                                     f1(s["gross_pct"], 2), f1(s["longest_lost_s"], 2),
                                     str(S[(rec, c)]["n_gt_frames"])]) + r" \\")
        if ri < len(RECS) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def table_old_new(S_old, S_new, path):
    lines = [r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
             r"Run & Ctrl. & Cov. [\%] & \multicolumn{4}{c}{Position error [mm]} & Rot. med. [$^\circ$] & Gross [\%] \\",
             r"\cmidrule(lr){4-7}", r" & & & med. & mean & p95 & max & & \\", r"\midrule"]
    for label, S in (("2026-09-05", S_old), ("2026-09-16", S_new)):
        for ci, c in enumerate(CTRLS):
            s = S[("static_dark", c)]["fused"]
            lines.append(" & ".join([label if ci == 0 else "", c, f1(s["coverage_pct"]), f1(s["pos_mm"]["median"]),
                                     f1(s["pos_mm"]["mean"]), f1(s["pos_mm"]["p95"]), f1(s["pos_mm"]["max"], 0),
                                     f1(s["rot_deg"]["median"], 2), f1(s["gross_pct"], 2)]) + r" \\")
        if label == "2026-09-05":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------------------- main
def main():
    FIG.mkdir(exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    frames = {(r, c): d for r in RECS for c in CTRLS if (d := load(BATCH, r, c)) is not None}
    old = {(r, c): d for r in ("static_dark", "walk_dark") for c in CTRLS if (d := load(OLD, r, c)) is not None}
    if not frames:
        sys.exit("no per-frame CSVs in figures/data/batch_0916 -- run compute_per_frame_errors.py first")

    S = {k: summarize(v) for k, v in frames.items()}
    S_old = {k: summarize(v) for k, v in old.items()}

    fig_cdf_static_dark(frames, FIG / "eval_cdf_static_dark.pdf")
    fig_cdf_all(frames, FIG / "eval_cdf_all_recordings.pdf")
    fig_error_time(frames[("static_dark", "left")], FIG / "eval_error_time_static_dark_left.pdf",
                   "static_dark, left controller (fused)")
    fig_error_time(frames[("walk_hard", "right")], FIG / "eval_error_time_walk_hard_right.pdf",
                   "walk_hard, right controller (fused)")
    hist = fig_lost_hist(frames, FIG / "eval_lost_streak_hist.pdf")

    table_rec(S, "static_dark", TABLES / "static_dark_summary.tex")
    table_rec(S, "walk_dark", TABLES / "walk_dark_summary.tex")
    table_all(S, TABLES / "all_recordings_summary.tex")
    if ("static_dark", "left") in S_old:
        table_old_new(S_old, S, TABLES / "static_dark_old_vs_new.tex")

    numbers = {
        "definitions": {
            "error": "per GT frame: residual = (T_est @ mocap_bridge)^-1 @ T_mocap (bridge composed before comparing); "
                     "position = |t| in mm, rotation = geodesic angle in deg",
            "coverage": "tracked GT frames / GT frames (GT frames = processed frames with valid mocap; mocap gaps excluded)",
            "tracked": "a reported pose exists at that frame (fused: accepted-or-fusion-rejected commit; vision: raw solve)",
            "fused_vs_vision": "fused = pose_csv (after fusion filter, One Euro OFF in the batch); vision = raw vision solve",
            "gross_error": f"tracked and (pos > {POS_OUT_MM} mm or rot > {ROT_OUT_DEG} deg)",
            "longest_lost": "evaluate_mocap.longest_gap: max run of consecutive GT frames without pose; seconds = first..last lost frame",
            "blind_time": "histogram only: last tracked frame before a lost run -> first tracked frame after it"},
        "provenance": {
            "batch_0916": "visualization/evaluate_2026-09-16 (run_all_recordings.py); config snapshot == config.yml at commit "
                          "cc52c0e (only visualization keys differ); pre-dates 855537b and ec3f849 (One Euro was OFF; "
                          "coast-budget, cold-reacquire rot-veto and swap fixes not included)",
            "old_0905": "data/eval (2026-09-05, before commit f145c9d); walk_dark = 28.5 s subsequence only",
            "bridge_caveat": "mocap bridge JSONs were fitted on static_dark itself -> static_dark position/rotation "
                             "errors are in-sample for the bridge; other recordings are out-of-sample"},
        "batch_0916": {f"{r}/{c}": S[(r, c)] for (r, c) in S},
        "old_0905": {f"{r}/{c}": S_old[(r, c)] for (r, c) in S_old},
        "lost_streak_histogram": hist,
        "gross_error_classification": classify_gross(frames)}
    (FIG / "eval_numbers.json").write_text(json.dumps(clean(numbers), indent=1))
    print("wrote figures, tables and eval_numbers.json under", FIG)


if __name__ == "__main__":
    main()
