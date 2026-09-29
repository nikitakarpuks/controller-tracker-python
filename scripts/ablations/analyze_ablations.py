#!/usr/bin/env python3
"""Tables and figure for the ablation study, with block-bootstrap confidence intervals.

Reads run_ablations.py output (<runs>/<ablation>/<recording>/{pose.csv,vision_pose.csv,mocap_gt/,config.yml,DONE.json})
and computes per-frame errors by REUSING evaluate_mocap.py's validated path (mocap bridge composed before comparing:
residual = (T_est @ bridge)^-1 @ T_mocap; position = |t| in mm, rotation = geodesic angle in deg). Each series is
cross-checked against evaluate_mocap.evaluate_controller (n_tracked, median) before use.

Metrics (per controller, pooled by concatenating frames): coverage, median/p95 position error, median/p95 rotation error,
gross-error rate (>150 mm or >20 deg among tracked frames), longest lost streak.
Confidence intervals: percentile bootstrap over 10-second blocks of the ground-truth timeline, resampled WITH THE SAME BLOCK
DRAWS for every ablation (so differences to the reference are paired). Pooled statistics resample blocks within every
recording, then pool both controllers and all recordings. Longest lost streak: point value is exact; its CI uses the
block-wise longest streak (streaks spanning a block boundary are cut) and is therefore a lower-bound style interval.

Usage: python analyze_ablations.py --runs <dir> [--ref full] [--source fused|vision] [--boot 500] [--out TUM-THESIS/figures]
       python analyze_ablations.py --selftest   # reproduce the 2026-09-16 batch summary through the same code path
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
LIVE = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LIVE))
from ablations import ABLATIONS  # noqa: E402
from evaluate_mocap import (load_pose_csv, load_mocap_gt_csv, rotation_angle_deg,  # noqa: E402
                            evaluate_controller)
from src.mocap_data import load_mocap_bridge  # noqa: E402

CONTROLLERS = ("left_controller", "right_controller")
POS_OUT_MM, ROT_OUT_DEG = 150.0, 20.0
BLOCK_S = 10.0
plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "font.size": 8, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#888888"


# ------------------------------------------------------------------------------------------ loading
def _bridge(cfg, ctrl):
    p = Path(cfg["controllers"][ctrl]["mocap_bridge_path"])
    return load_mocap_bridge(str(p if p.is_absolute() else LIVE / p))


def load_series(job_dir, source="fused", window=None):
    """{ctrl: dict(ts, t, tracked, pos, rot)} for one job; None if unusable."""
    job_dir = Path(job_dir)
    cfg = yaml.safe_load((job_dir / "config.yml").read_text())
    done = json.loads((job_dir / "DONE.json").read_text())
    if done.get("returncode") != 0:
        return None
    poses_all = load_pose_csv(job_dir / ("pose.csv" if source == "fused" else "vision_pose.csv"))
    lo, hi = done.get("eval_from_ts"), done.get("eval_to_ts")
    if window:
        lo, hi = window
    out = {}
    for ctrl in CONTROLLERS:
        gt = load_mocap_gt_csv(job_dir / "mocap_gt" / f"{ctrl}_mocap_gt.csv")
        if lo is not None:
            gt = {t: v for t, v in gt.items() if lo <= t <= hi}
        if not gt:
            continue
        bridge = _bridge(cfg, ctrl)
        poses = {t: p for t, p in poses_all.get(ctrl, {}).items() if t in gt}
        ts = np.array(sorted(gt))
        pos = np.full(len(ts), np.nan)
        rot = np.full(len(ts), np.nan)
        for i, t in enumerate(ts):
            T = poses.get(int(t))
            if T is None:
                continue
            res = T.compose(bridge).inverse().compose(gt[int(t)])
            pos[i] = float(np.linalg.norm(res.t)) * 1000.0
            rot[i] = rotation_angle_deg(res.R)
        # cross-check against the validated evaluation path (same filtered inputs)
        ref = evaluate_controller(ctrl, poses if source == "fused" else {}, poses if source == "vision" else {}, gt, bridge)[source]
        tracked = ~np.isnan(pos)
        assert ref["n_tracked_frames"] == int(tracked.sum()), (job_dir, ctrl, ref["n_tracked_frames"], tracked.sum())
        if tracked.any():
            assert abs(ref["pos_err_mm"]["median"] - np.median(pos[tracked])) < 1e-6
        out[ctrl] = {"ts": ts, "t": (ts - ts[0]) / 1e9, "tracked": tracked, "pos": pos, "rot": rot}
    return out or None


# ------------------------------------------------------------------------------------------ metrics
def _streak_seconds(t, tracked):
    best, start, cnt = 0.0, None, 0
    for ti, tr in zip(t, tracked):
        if tr:
            cnt, start = 0, None
        else:
            if start is None:
                start = ti
            cnt += 1
            best = max(best, ti - start)
    return best


def stats_from(tracked, pos, rot, streak_s):
    n = len(tracked)
    tr = tracked
    p, r = pos[tr], rot[tr]
    if p.size == 0:
        return dict(cov=0.0, pos_med=np.nan, pos_p95=np.nan, rot_med=np.nan, rot_p95=np.nan, gross=np.nan, streak=streak_s)
    return dict(cov=100.0 * tr.sum() / n, pos_med=np.median(p), pos_p95=np.percentile(p, 95),
                rot_med=np.median(r), rot_p95=np.percentile(r, 95),
                gross=100.0 * np.mean((p > POS_OUT_MM) | (r > ROT_OUT_DEG)), streak=streak_s)


METRICS = ["cov", "pos_med", "pos_p95", "rot_med", "rot_p95", "gross", "streak"]


class BlockedSeries:
    """One series (rec x ctrl x ablation) cut into 10-s blocks of the GT timeline."""

    def __init__(self, s):
        self.s = s
        b = np.floor(s["t"] / BLOCK_S).astype(int)
        self.block_of = b
        self.nblocks = int(b.max()) + 1
        self.idx = [np.where(b == k)[0] for k in range(self.nblocks)]
        self.streak = np.array([_streak_seconds(s["t"][i], s["tracked"][i]) if len(i) else 0.0 for i in self.idx])

    def gather(self, blocks):
        sel = np.concatenate([self.idx[k] for k in blocks]) if len(blocks) else np.array([], int)
        return sel


def pooled_stats(parts, draw):
    """parts: [(rec_index, BlockedSeries)], draw: {rec_index: block ids} -> metrics dict."""
    tr, pos, rot, streak = [], [], [], 0.0
    for ri, bs in parts:
        blocks = draw[ri] if draw is not None else range(bs.nblocks)
        sel = bs.gather(list(blocks))
        s = bs.s
        tr.append(s["tracked"][sel]); pos.append(s["pos"][sel]); rot.append(s["rot"][sel])
        streak = max(streak, float(bs.streak[list(blocks)].max()) if len(blocks) else 0.0)
    tr, pos, rot = np.concatenate(tr), np.concatenate(pos), np.concatenate(rot)
    return stats_from(tr, pos, rot, streak)


def exact_streak(parts):
    return max(_streak_seconds(bs.s["t"], bs.s["tracked"]) for _, bs in parts)


# ------------------------------------------------------------------------------------------ main analysis
def analyze(runs, ref, source, boot, seed):
    runs = Path(runs)
    abls = [d.name for d in sorted(runs.iterdir()) if d.is_dir()]
    order = [a for a in ABLATIONS if a in abls] + [a for a in abls if a not in ABLATIONS]
    recs = sorted({r.name for a in abls for r in (runs / a).iterdir() if r.is_dir()})
    data = {}                       # (ablation, rec) -> {ctrl: BlockedSeries}
    for a in order:
        for r in recs:
            jd = runs / a / r
            if not (jd / "DONE.json").exists():
                continue
            s = load_series(jd, source)
            if s:
                data[(a, r)] = {c: BlockedSeries(v) for c, v in s.items()}
    # the reference defines which (rec, ctrl) have ground truth; other ablations must match its GT timeline
    have = [(a, r) for (a, r) in data]
    ref_recs = sorted({r for (a, r) in have if a == ref})
    if not ref_recs:
        raise SystemExit(f"reference ablation '{ref}' has no finished job under {runs}")
    ridx = {r: i for i, r in enumerate(ref_recs)}
    for (a, r), cs in list(data.items()):
        for c, bs in cs.items():
            g = data[(ref, r)].get(c) if (ref, r) in data else None
            if g is None or len(g.s["ts"]) != len(bs.s["ts"]) or not np.array_equal(g.s["ts"], bs.s["ts"]):
                print(f"warn: GT frame set of {a}/{r}/{c} differs from reference -> dropped", file=sys.stderr)
                del cs[c]
    # common draws (paired across ablations)
    nblocks = {r: max(bs.nblocks for bs in data[(ref, r)].values()) for r in ref_recs}
    draws = []
    for b in range(boot):
        draws.append({ridx[r]: np.random.default_rng([seed, b, ridx[r]]).integers(0, nblocks[r], nblocks[r]) for r in ref_recs})
    total_blocks = int(sum(nblocks.values()))
    if total_blocks < 10:
        print(f"warn: only {total_blocks} ten-second blocks in total -> bootstrap CIs are not meaningful and are omitted "
              f"(run full recordings)", file=sys.stderr)
    res = {"meta": {"ref": ref, "source": source, "boot": boot, "block_s": BLOCK_S, "recordings": ref_recs,
                    "total_blocks": total_blocks, "ci_valid": total_blocks >= 10,
                    "gross_thresholds": [POS_OUT_MM, ROT_OUT_DEG]}, "pooled": {}, "per_recording": {}}
    for a in order:
        parts = [(ridx[r], bs) for r in ref_recs if (a, r) in data for bs in data[(a, r)].values()]
        if not parts:
            continue
        complete = len({p[0] for p in parts}) == len(ref_recs)
        point = pooled_stats(parts, None)
        point["streak"] = exact_streak(parts)
        vals = {m: np.array([pooled_stats(parts, d)[m] for d in draws]) for m in METRICS}
        res["pooled"][a] = {"complete": complete, "point": point,
                            "ci": {m: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for m, v in vals.items()},
                            "_boot": vals}
        for r in ref_recs:
            pr = [(ridx[r], bs) for bs in data.get((a, r), {}).values()]
            if pr:
                pt = pooled_stats(pr, None); pt["streak"] = exact_streak(pr)
                dv = {m: np.array([pooled_stats(pr, {ridx[r]: d[ridx[r]]})[m] for d in draws]) for m in METRICS}
                res["per_recording"].setdefault(a, {})[r] = {
                    "point": pt, "ci": {m: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for m, v in dv.items()},
                    "_boot": dv}
    # paired deltas vs reference
    for a in res["pooled"]:
        bo, br = res["pooled"][a]["_boot"], res["pooled"][ref]["_boot"]
        res["pooled"][a]["delta"] = {m: {"point": float(res["pooled"][a]["point"][m] - res["pooled"][ref]["point"][m]),
                                         "ci": [float(np.nanpercentile(bo[m] - br[m], 2.5)), float(np.nanpercentile(bo[m] - br[m], 97.5))]}
                                     for m in METRICS}
        for r, pr in res["per_recording"].get(a, {}).items():
            rr = res["per_recording"].get(ref, {}).get(r)
            if rr:
                pr["delta"] = {m: float(pr["point"][m] - rr["point"][m]) for m in METRICS}
    return res, order, ref_recs


# ------------------------------------------------------------------------------------------ outputs
LABEL = {"full": "full", "imu_off": "no IMU, no fusion", "fusion_off": "no fusion filter (IMU kept)",
         "imu_off_filter_on": "no IMU, filter kept", "no_one_euro": "no One Euro", "no_swap_detect": "no swap detection",
         "no_rot_veto": "no rotation veto", "no_lamp_filter": "no lamp filter", "no_lamp_memory": "no lamp memory",
         "no_twopass": "no second pass", "no_streak_rescue": "no streak rescue", "no_proximity": "no proximity match",
         "no_edge_taper": "no edge taper", "ladder_0_brute_only": "L0 brute force only", "ladder_1_warm": "L1 + warm path",
         "ladder_2_lamp": "L2 + lamp filter", "monado_like_brute": "Monado-style brute force"}
FMT = {"cov": "{:.1f}", "pos_med": "{:.1f}", "pos_p95": "{:.1f}", "rot_med": "{:.2f}", "rot_p95": "{:.1f}", "gross": "{:.2f}", "streak": "{:.1f}"}


def _cell(pt, ci, m, valid=True):
    f = FMT[m]
    if not valid:
        return f.format(pt).replace("nan", "--")
    return (f.format(pt) + r" {\scriptsize[" + f.format(ci[0]) + ", " + f.format(ci[1]) + "]}").replace("nan", "--")


def write_tables(res, order, out):
    out.mkdir(parents=True, exist_ok=True)
    head = r"""\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Configuration & Cov.\ [\%] & Pos.\ med.\ [mm] & Pos.\ p95 [mm] & Rot.\ med.\ [$^\circ$] & Gross [\%] & Longest lost [s]$^\dagger$ \\
\midrule
"""
    rows = []
    for a in order:
        if a not in res["pooled"]:
            continue
        p, ci = res["pooled"][a]["point"], res["pooled"][a]["ci"]
        star = "" if res["pooled"][a]["complete"] else r"$^\ddagger$"
        rows.append(LABEL.get(a, a).replace("_", r"\_") + star + " & " + " & ".join(
            _cell(p[m], ci[m], m, res["meta"]["ci_valid"]) for m in ["cov", "pos_med", "pos_p95", "rot_med", "gross", "streak"]) + r" \\")
    (out / "ablation_pooled.tex").write_text(head + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")
    # deltas
    head2 = r"""\begin{tabular}{@{}lrrr@{}}
\toprule
Configuration & $\Delta$ coverage [pp] & $\Delta$ pos.\ median [mm] & $\Delta$ gross rate [pp] \\
\midrule
"""
    rows2 = []
    for a in order:
        if a not in res["pooled"] or a == res["meta"]["ref"]:
            continue
        d = res["pooled"][a]["delta"]

        def dc(m, f):
            pt, (lo, hi) = d[m]["point"], d[m]["ci"]
            if not res["meta"]["ci_valid"]:
                return f.format(pt)
            sig = r"$^\ast$" if (lo > 0 or hi < 0) else ""
            return (f.format(pt) + sig + r" {\scriptsize[" + f.format(lo) + ", " + f.format(hi) + "]}").replace("nan", "--")
        rows2.append(LABEL.get(a, a).replace("_", r"\_") + " & " + " & ".join([dc("cov", "{:+.1f}"), dc("pos_med", "{:+.1f}"), dc("gross", "{:+.2f}")]) + r" \\")
    (out / "ablation_delta.tex").write_text(head2 + "\n".join(rows2) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")
    # per recording tables
    recs = res["meta"]["recordings"]
    for m, name in [("cov", "coverage"), ("pos_med", "posmed"), ("gross", "gross")]:
        h = r"\begin{tabular}{@{}l" + "r" * len(recs) + r"@{}}" + "\n" + r"\toprule" + "\nConfiguration & " + " & ".join(
            r"\texttt{" + r.replace("_", r"\_") + "}" for r in recs) + r" \\" + "\n" + r"\midrule" + "\n"
        rr = []
        for a in order:
            if a in res["per_recording"]:
                rr.append(LABEL.get(a, a).replace("_", r"\_") + " & " + " & ".join(
                    FMT[m].format(res["per_recording"][a][r]["point"][m]) if r in res["per_recording"][a] else "--" for r in recs) + r" \\")
        (out / f"ablation_per_recording_{name}.tex").write_text(h + "\n".join(rr) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")


def write_figure(res, order, out):
    abl = [a for a in order if a in res["pooled"] and a != res["meta"]["ref"]]
    if not abl:
        return
    panels = [("cov", "coverage [pp]"), ("pos_med", "median position error [mm]"), ("gross", "gross-error rate [pp]")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 0.28 * len(abl) + 1.0), sharey=True)
    y = np.arange(len(abl))[::-1]
    for ax, (m, lab) in zip(axes, panels):
        for yi, a in zip(y, abl):
            d = res["pooled"][a]["delta"][m]
            lo, hi = d["ci"]
            sig = (lo > 0 or hi < 0) and res["meta"]["ci_valid"]
            if res["meta"]["ci_valid"]:
                ax.plot([lo, hi], [yi, yi], color=ORANGE if sig else GRAY, lw=1.4, solid_capstyle="butt")
            ax.plot(d["point"], yi, "o", color=ORANGE if sig else GRAY, ms=3.5)
            for r, pr in res["per_recording"].get(a, {}).items():
                if "delta" in pr and not np.isnan(pr["delta"][m]):
                    ax.plot(pr["delta"][m], yi, ".", color=BLUE, ms=2.2, alpha=0.55)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel(r"$\Delta$ " + lab + " vs. " + res["meta"]["ref"])
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABEL.get(a, a) for a in abl])
    fig.text(0.995, 0.005, ("line: paired 95% block-bootstrap CI (orange: excludes 0); " if res["meta"]["ci_valid"] else "CI omitted (too few blocks); ") + "blue dots: single recordings", ha="right", va="bottom", fontsize=6)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out / "ablation_paired.pdf")
    plt.close(fig)


def selftest():
    """Reproduce the 2026-09-16 batch summary (static_dark) through this script's loader."""
    batch = LIVE / "visualization" / "evaluate_2026-09-16" / "static_dark"
    cfg = yaml.safe_load((batch / "config.yml").read_text())
    ref = json.loads((batch / "metrics_summary.json").read_text())
    poses = load_pose_csv(batch / "pose.csv")
    for ctrl in CONTROLLERS:
        gt = load_mocap_gt_csv(batch / "mocap_gt" / f"{ctrl}_mocap_gt.csv")
        bridge = _bridge(cfg, ctrl)
        pos = []
        for t, T in poses[ctrl].items():
            if t in gt:
                pos.append(float(np.linalg.norm(T.compose(bridge).inverse().compose(gt[t]).t)) * 1000.0)
        print(ctrl, "n=%d median=%.4f mm | batch summary n=%d median=%.4f mm" % (
            len(pos), np.median(pos), ref[ctrl]["n_tracked_frames"], ref[ctrl]["pos_err_mm"]["median"]))
        assert len(pos) == ref[ctrl]["n_tracked_frames"] and abs(np.median(pos) - ref[ctrl]["pos_err_mm"]["median"]) < 1e-6
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs")
    ap.add_argument("--ref", default="full")
    ap.add_argument("--source", choices=["fused", "vision"], default="fused")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--out", default=str(LIVE / "TUM-THESIS" / "figures"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    res, order, recs = analyze(a.runs, a.ref, a.source, a.boot, a.seed)
    out = Path(a.out)
    write_tables(res, order, out / "tables")
    write_figure(res, order, out)
    clean = json.loads(json.dumps(res, default=lambda o: None if isinstance(o, np.ndarray) else float(o)))
    for a_ in clean["pooled"].values():
        a_.pop("_boot", None)
    for d in clean["per_recording"].values():
        for v in d.values():
            v.pop("_boot", None)
    prov = Path(a.runs) / "PROVENANCE.json"
    clean["meta"]["provenance"] = json.loads(prov.read_text()) if prov.exists() else None
    (out / "ablation_results.json").write_text(json.dumps(clean, indent=1))
    print("wrote tables/ablation_*.tex, ablation_paired.pdf, ablation_results.json ->", out)


if __name__ == "__main__":
    main()
