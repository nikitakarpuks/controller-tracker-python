#!/usr/bin/env python3
"""Build the xrtslam (evo ATE/RTE) results tables of thesis chapter 6.

Reads <recordings-root>/euroc_recording_*/tracking_results/results.json for the
eight recordings (missing ones are shown as '--'), and writes

  figures/tables/xrtslam_summary.tex   per recording and controller
  figures/tables/xrtslam_pooled.tex    n-weighted pooled over the available recordings
  figures/xrtslam_numbers.json         all numbers (mm) plus provenance
  figures/xrtslam/<recording>/*.png    downscaled copies of the ATE/RTE plots

Provenance goes to the JSON only (never to the LaTeX): mtime of results.json and of
the xrtslam-metrics run/target CSVs behind it, the time of the last commit touching
src/ or config/, a 'stale' flag, and newer run variants found on disk.

Usage: python3 scripts/xrtslam_tables.py [--recordings-root DIR] [--out-dir DIR]
"""
import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

THESIS = Path(__file__).resolve().parents[1]
REPO = THESIS.parent
XRT = Path("/home/nikitakarpuks/PyCharmProjects/xrtslam-metrics/ctrl_eval")

ORDER = [
    ("static_dark", "20260826173103"), ("static_easy", "20260826173932"),
    ("static_medium", "20260826174213"), ("static_hard", "20260826174510"),
    ("walk_dark", "20260826173350"), ("walk_easy", "20260826175226"),
    ("walk_medium", "20260826175510"), ("walk_hard", "20260826180039"),
]
SIDES = ("left", "right")

# Status stated by the peer session (2026-09-19); recorded in the JSON only.
PEER_STATUS = {
    "static_dark": "old run, pre-dates all later fixes and the mocap regeneration: stale, do not quote",
    "walk_dark": "old run, pre-dates all later fixes and the mocap regeneration: stale, do not quote",
    "static_easy": "results.json computed before constrained-fallback / weak-solo / gravity / brute-tier "
                   "changes and a blob_detector.py change: stale, regenerate before quoting",
    "walk_easy": "results.json computed before constrained-fallback / weak-solo / gravity / brute-tier "
                 "changes: stale, regenerate before quoting",
    "static_medium": "full run with current code exists, no tracking_results yet",
    "walk_medium": "full run in progress",
    "static_hard": "no trustworthy mocap-metrics run",
    "walk_hard": "no trustworthy mocap-metrics run",
}
PEER_STALE = {"static_dark", "walk_dark", "static_easy", "walk_easy"}


def iso(ts):
    return dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds") if ts else None


def last_code_commit_ts():
    out = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%ct", "HEAD", "--", "src", "config"],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out else None


def mm(x):
    return None if x is None else 1000.0 * x


def load_recording(root: Path, name: str, stamp: str, code_ts):
    d = root / f"euroc_recording_{stamp}_{name}"
    res_path = d / "tracking_results" / "results.json"
    rec = {"recording": name, "dir": str(d), "available": res_path.exists(),
           "peer_status": PEER_STATUS[name]}
    if not res_path.exists():
        rec["stale"] = None
        return rec
    raw = json.loads(res_path.read_text())
    m = res_path.stat().st_mtime
    rec["results_mtime"] = iso(m)
    rec["last_code_commit"] = iso(code_ts)
    stale_rule = bool(code_ts and m < code_ts)
    rec["stale_by_mtime_rule"] = stale_rule
    rec["stale_by_peer_status"] = name in PEER_STALE
    rec["stale"] = stale_rule or name in PEER_STALE
    for side in SIDES:
        s = raw[side]
        ate, rte, cov = s["ate"], s["rte"], s["coverage"]
        n_ate = ate["sse"] / ate["rmse"] ** 2 if ate["rmse"] else None   # sse = n * rmse^2
        n_rte = rte["sse"] / rte["rmse"] ** 2 if rte["rmse"] else None
        entry = {"coverage": cov,
                 "ate_mm": {k: mm(v) for k, v in ate.items() if k != "sse"},
                 "rte_mm": {k: mm(v) for k, v in rte.items() if k != "sse"},
                 "ate_sse_m2": ate["sse"], "rte_sse_m2": rte["sse"],
                 "n_ate_pairs": n_ate, "n_rte_pairs": n_rte}
        run_csv = XRT / "runs" / "OurPipeline" / f"{name}_{side}" / "tracking.csv"
        gt_csv = XRT / "targets" / f"{name}_{side}" / "gt.csv"
        entry["run_tracking_csv_mtime"] = iso(run_csv.stat().st_mtime) if run_csv.exists() else None
        entry["target_gt_csv_mtime"] = iso(gt_csv.stat().st_mtime) if gt_csv.exists() else None
        variants = sorted(p.name for p in (XRT / "runs" / "OurPipeline").glob(f"{name}_v*_{side}")
                          if p.stat().st_mtime > m)
        entry["newer_run_variants_on_disk"] = variants
        rec[side] = entry
    return rec


def fmt(x, nd=1):
    return "--" if x is None else f"{x:.{nd}f}"


def summary_table(recs):
    rows = []
    for rec in recs:
        name = rec["recording"].replace("_", r"\_")
        for i, side in enumerate(SIDES):
            first = rf"\texttt{{{name}}}" if i == 0 else ""
            if not rec["available"]:
                vals = ["--"] * 6
            else:
                e = rec[side]
                vals = [fmt(e["coverage"]["tracked_pct"]), fmt(e["coverage"]["both_pct"]),
                        fmt(e["ate_mm"]["rmse"]), fmt(e["ate_mm"]["median"]),
                        fmt(e["rte_mm"]["rmse"]), fmt(e["rte_mm"]["median"])]
            rows.append(f"{first} & {side} & " + " & ".join(vals) + r" \\")
        rows.append(r"\addlinespace")
    rows.pop()
    head = (r"\begin{tabular}{@{}llrrrrrr@{}}" "\n" r"\toprule" "\n"
            r"Recording & Ctrl. & Tracked & Tracked \& & \multicolumn{2}{c}{ATE [mm]} & "
            r"\multicolumn{2}{c}{RTE [mm]} \\" "\n"
            r" & & [\%] & measurable [\%] & RMSE & median & RMSE & median \\" "\n"
            r"\cmidrule(lr){5-6}\cmidrule(l){7-8}" "\n" r"\midrule")
    return head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" "\n" r"\end{tabular}" + "\n"


def pooled(recs):
    avail = [r for r in recs if r["available"]]
    out = {"recordings_included": len(avail), "recordings_total": len(recs), "rows": {}}
    for label, sides in (("left", ("left",)), ("right", ("right",)), ("both", SIDES)):
        tot = trk = both = 0.0
        n_a = sse_a = sum_a = n_r = sse_r = sum_r = 0.0
        for r in avail:
            for s in sides:
                e = r[s]
                t = e["coverage"]["total_frames"]
                tot += t
                trk += t * e["coverage"]["tracked_pct"] / 100
                both += t * e["coverage"]["both_pct"] / 100
                n_a += e["n_ate_pairs"]; sse_a += e["ate_sse_m2"]; sum_a += e["ate_mm"]["mean"] * e["n_ate_pairs"]
                n_r += e["n_rte_pairs"]; sse_r += e["rte_sse_m2"]; sum_r += e["rte_mm"]["mean"] * e["n_rte_pairs"]
        out["rows"][label] = {
            "tracked_pct": 100 * trk / tot if tot else None, "both_pct": 100 * both / tot if tot else None,
            "ate_rmse_mm": 1000 * (sse_a / n_a) ** 0.5 if n_a else None, "ate_mean_mm": sum_a / n_a if n_a else None,
            "rte_rmse_mm": 1000 * (sse_r / n_r) ** 0.5 if n_r else None, "rte_mean_mm": sum_r / n_r if n_r else None,
            "n_ate_pairs": n_a, "n_rte_pairs": n_r, "camera_frames": tot}
    return out


def pooled_table(p):
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             rf"\multicolumn{{7}}{{@{{}}l@{{}}}}{{Pooled over {p['recordings_included']} of {p['recordings_total']} recordings, "
             r"n-weighted; both controllers of each recording.} \\ \midrule",
             r"Controller & Tracked & Tracked \& & \multicolumn{2}{c}{ATE [mm]} & \multicolumn{2}{c}{RTE [mm]} \\",
             r" & [\%] & measurable [\%] & RMSE & mean & RMSE & mean \\",
             r"\cmidrule(lr){4-5}\cmidrule(l){6-7}", r"\midrule"]
    for label in ("left", "right", "both"):
        r = p["rows"][label]
        lines.append(f"{label} & {fmt(r['tracked_pct'])} & {fmt(r['both_pct'])} & {fmt(r['ate_rmse_mm'])} & "
                     f"{fmt(r['ate_mean_mm'])} & {fmt(r['rte_rmse_mm'])} & {fmt(r['rte_mean_mm'])} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def copy_plots(recs, out_dir):
    try:
        from PIL import Image
    except ImportError:
        return []
    done = []
    for rec in recs:
        if not rec["available"]:
            continue
        src_dir = Path(rec["dir"]) / "tracking_results"
        dst_dir = out_dir / "xrtslam" / rec["recording"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        for png in sorted(src_dir.glob("*.png")):
            im = Image.open(png)
            w = 720
            im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
            im.save(dst_dir / png.name, optimize=True)
            done.append(str((dst_dir / png.name).relative_to(out_dir)))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings-root", default="/home/nikitakarpuks/Downloads/recordings-aug26")
    ap.add_argument("--out-dir", default=str(THESIS / "figures"))
    a = ap.parse_args()
    out = Path(a.out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    code_ts = last_code_commit_ts()
    recs = [load_recording(Path(a.recordings_root), n, s, code_ts) for n, s in ORDER]
    p = pooled(recs)
    (out / "tables" / "xrtslam_summary.tex").write_text(summary_table(recs))
    (out / "tables" / "xrtslam_pooled.tex").write_text(pooled_table(p))
    plots = copy_plots(recs, out)
    (out / "xrtslam_numbers.json").write_text(json.dumps(
        {"generated": iso(dt.datetime.now().timestamp()), "last_code_commit": iso(code_ts),
         "units": "ATE/RTE in mm, coverage in % of camera frames", "recordings": recs, "pooled": p,
         "plots": plots}, indent=1))
    print(f"recordings available: {p['recordings_included']}/{p['recordings_total']}; plots copied: {len(plots)}")
    for r in recs:
        print(f"  {r['recording']:14s} available={r['available']} stale={r['stale']}")


if __name__ == "__main__":
    main()
