#!/usr/bin/env python3
"""trim_recording.py -- write a TRIMMED COPY of one euroc_recording_* folder (the source is never modified).

Why: a recording whose capture stalled at the end (e.g. static_hard: Monado crashed in the last ~3 s, so camera
frames arrive late/bursty and their timestamps are up to ~240 ms later than the exposure) should not be published or
scored as-is. This script cuts every stream at one common time so the published dataset has no dangling data.

Cut rule
  * cameras (mav0/cam*/): rows/images with timestamp <= T_cut are kept.
  * IMUs (mav0/imu*/), cleaned mocap (mocap_filtered/<dev>/data.csv, *-refit.csv, data-recovery-report.csv) and the raw
    Motive export (<take>.csv): rows up to T_cut + margin (default 0.5 s) are kept, so integration, time alignment and
    the mocap interpolation at the last frame still have data on both sides.
  * T_cut is either --last-frame-ts (ns) or --last-frame-index N = the N-th (0-based, inclusive) valid timestamp of the
    reference camera (default cam4, the same frame list the tracker's frame_range indexes).
  * Rows with an impossible timestamp (>= 9e18, e.g. the corrupt last row of static_hard's cam4 data.csv) are dropped.

Mocap time base: the cleaned mocap data.csv timestamps and the refit/recovery-report/raw Motive "Time" column are tied by
  ts_ns = anchor_ns + Time_s * 1e9. The anchor is found by matching the first data.csv row to its refit row
  (Refit_Pos*), so no assumption about the anchor is baked in.

Left as they were (copied unchanged, flagged in TRIM_NOTES.md): mocap_filtered/<dev>/drift_check/* and the *.png / *.txt
quality reports and calibration json -- they were computed on the FULL recording. mocap_filtered.bak_* folders are skipped
unless --include-backups.

Safety: refuses to write into an existing output folder; snapshots the source (relative path, size, mtime) before and
after and aborts if anything changed; verifies every stream afterwards; --dry-run prints the plan and counts only.

usage:
  python3 trim_recording.py SRC_DIR --last-frame-index 5858 [--margin-s 0.5] [--out DST_DIR] [--ref-cam cam4]
                            [--hardlink] [--include-backups] [--dry-run]
"""
import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

BAD_TS = 9e18


def _read_ts_rows(path):
    """(header_lines, [(ts_ns or None, raw_line), ...]) of an EuRoC-style csv whose first column is a ns timestamp."""
    header, rows = [], []
    with open(path, newline="") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                header.append(line)
                continue
            try:
                ts = int(line.split(",", 1)[0])
            except ValueError:
                ts = None
            rows.append((ts, line))
    return header, rows


def _eol(path):
    with open(path, "rb") as f:
        return "\r\n" if b"\r\n" in f.read(65536) else "\n"


def _snapshot(root: Path):
    return {str(p.relative_to(root)): (p.stat().st_size, int(p.stat().st_mtime_ns))
            for p in sorted(root.rglob("*")) if p.is_file()}


def _mocap_anchor_ns(dev_dir: Path):
    """anchor_ns such that ts_ns = anchor_ns + Time_s*1e9, from data.csv vs the refit csv (None if unavailable)."""
    refits = list(dev_dir.glob("*-refit.csv"))
    dfile = dev_dir / "data.csv"
    if not refits or not dfile.exists():
        return None
    _, drows = _read_ts_rows(dfile)
    drows = [(t, l) for t, l in drows if t is not None]
    if not drows:
        return None
    with open(refits[0], newline="") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        need = ["Time", "Refit_PosX", "Refit_PosY", "Refit_PosZ"]
        if not all(c in cols for c in need):
            return None
        refit = [(float(r["Time"]), tuple(float(r[c]) if r[c] != "" else float("nan") for c in need[1:])) for r in rd]
    t0, l0 = drows[0]
    p0 = tuple(float(x) for x in l0.split(",")[1:4])
    for time_s, pos in refit:
        if all(abs(a - b) < 2e-6 for a, b in zip(pos, p0)):
            return t0 - int(round(time_s * 1e9))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--last-frame-index", type=int)
    g.add_argument("--last-frame-ts", type=int)
    ap.add_argument("--margin-s", type=float, default=0.5)
    ap.add_argument("--ref-cam", default="cam4")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hardlink", action="store_true", help="hardlink images instead of copying (same filesystem only)")
    ap.add_argument("--include-backups", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = Path(a.src).resolve()
    mav0 = src / "mav0"
    if not mav0.is_dir():
        sys.exit(f"{src} has no mav0/")
    out = Path(a.out).resolve() if a.out else src.with_name(src.name + "_trimmed")
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to write into non-empty {out}")

    # ---- cut times
    ref_csv = mav0 / a.ref_cam / "data.csv"
    _, ref_rows = _read_ts_rows(ref_csv)
    ref_ts = sorted(t for t, _ in ref_rows if t is not None and t < BAD_TS)
    if a.last_frame_ts is not None:
        t_cut = a.last_frame_ts
    else:
        if not (0 <= a.last_frame_index < len(ref_ts)):
            sys.exit(f"--last-frame-index {a.last_frame_index} out of range (0..{len(ref_ts) - 1})")
        t_cut = ref_ts[a.last_frame_index]
    t_end = t_cut + int(round(a.margin_s * 1e9))
    print(f"source      : {src}\noutput      : {out}\nreference   : {a.ref_cam} ({len(ref_ts)} valid frames)")
    print(f"T_cut  (cameras)            : {t_cut}  (frame index {ref_ts.index(t_cut) if t_cut in ref_ts else 'n/a'}, "
          f"{(t_cut - ref_ts[0]) / 1e9:.3f} s after the first frame)")
    print(f"T_end  (IMU/mocap, +{a.margin_s:g} s): {t_end}")

    before = _snapshot(src)
    plan = {"cameras": {}, "imus": {}, "mocap": {}, "raw_motive": {}, "copied_unchanged": [], "skipped": []}

    # ---- cameras / IMUs
    cam_keep = {}
    for d in sorted(p for p in mav0.iterdir() if p.is_dir()):
        dcsv = d / "data.csv"
        if not dcsv.exists():
            plan["copied_unchanged"].append(str(d.relative_to(src)))
            continue
        header, rows = _read_ts_rows(dcsv)
        is_cam = (d / "data").is_dir()
        limit = t_cut if is_cam else t_end
        keep = [(t, l) for t, l in rows if t is not None and t < BAD_TS and t <= limit]
        dropped = len(rows) - len(keep)
        bad = sum(1 for t, _ in rows if t is None or t >= BAD_TS)
        info = dict(kept=len(keep), dropped=dropped, invalid_timestamp_rows=bad,
                    first=keep[0][0] if keep else None, last=keep[-1][0] if keep else None, limit=limit)
        (plan["cameras"] if is_cam else plan["imus"])[d.name] = info
        cam_keep[d.name] = (header, keep, is_cam)

    # ---- mocap
    mocap_root = src / "mocap_filtered"
    mocap_dirs = []
    if mocap_root.is_dir():
        mocap_dirs = [p for p in sorted(mocap_root.iterdir()) if p.is_dir()]
    for p in sorted(src.glob("mocap_filtered.bak*")):
        if a.include_backups:
            mocap_dirs.append(p)
        else:
            plan["skipped"].append(p.name)
    mocap_plan = {}
    for dev in mocap_dirs:
        anchor = _mocap_anchor_ns(dev)
        if anchor is None:
            plan["mocap"][str(dev.relative_to(src))] = "NO ANCHOR (could not tie data.csv to the refit csv) -- will be copied UNTRIMMED"
        mocap_plan[dev] = anchor

    print("\nplan:")
    for k in ("cameras", "imus"):
        for name, i in plan[k].items():
            print(f"  {k[:-1]:6s} {name:5s}: keep {i['kept']:6d} rows, drop {i['dropped']:4d} (invalid timestamps {i['invalid_timestamp_rows']}), "
                  f"last kept {i['last']} <= {i['limit']}")
    for dev, anchor in mocap_plan.items():
        print(f"  mocap  {dev.relative_to(src)}: anchor_ns={anchor}")
    if plan["skipped"]:
        print("  skipped (not published):", ", ".join(plan["skipped"]))
    if a.dry_run:
        print("\n--dry-run: nothing written")
        return

    # ---- write
    out.mkdir(parents=True, exist_ok=True)
    notes = {"source": str(src), "T_cut_ns": t_cut, "T_end_ns": t_end, "margin_s": a.margin_s, "ref_cam": a.ref_cam,
             "streams": {}, "mocap": {}}
    for name, (header, keep, is_cam) in cam_keep.items():
        ddir = out / "mav0" / name
        ddir.mkdir(parents=True, exist_ok=True)
        with open(ddir / "data.csv", "w", newline="") as f:
            f.writelines(header)
            f.writelines(l for _, l in keep)
        if is_cam:
            (ddir / "data").mkdir(exist_ok=True)
            for _, line in keep:
                fname = line.strip().split(",")[1] if "," in line else None
                s = mav0 / name / "data" / fname
                if s.exists():
                    (os.link if a.hardlink else shutil.copy2)(s, ddir / "data" / fname)
                else:
                    print(f"  WARNING missing image {s}")
        # anything else inside the stream folder (sensor.yaml ...) is copied as is
        for extra in (mav0 / name).iterdir():
            if extra.name not in ("data", "data.csv"):
                shutil.copy2(extra, ddir / extra.name) if extra.is_file() else shutil.copytree(extra, ddir / extra.name)
        notes["streams"][name] = (plan["cameras"].get(name) or plan["imus"].get(name))
    for d in plan["copied_unchanged"]:
        s = src / d
        shutil.copytree(s, out / d) if s.is_dir() else shutil.copy2(s, out / d)

    for dev, anchor in mocap_plan.items():
        rel = dev.relative_to(src)
        odev = out / rel
        odev.mkdir(parents=True, exist_ok=True)
        cutoff_time_s = None if anchor is None else (t_end - anchor) / 1e9
        stats = {"anchor_ns": anchor, "cutoff_time_s": cutoff_time_s}
        for f in sorted(dev.iterdir()):
            o = odev / f.name
            if f.is_dir():
                shutil.copytree(f, o)                      # drift_check/*: computed on the full recording, kept as is
            elif anchor is None:
                shutil.copy2(f, o)
            elif f.name == "data.csv":
                header, rows = _read_ts_rows(f)
                keep = [(t, l) for t, l in rows if t is not None and t <= t_end]
                with open(o, "w", newline="") as w:
                    w.writelines(header); w.writelines(l for _, l in keep)
                stats["data.csv"] = dict(kept=len(keep), dropped=len(rows) - len(keep), last=keep[-1][0] if keep else None)
            elif f.name.endswith("-refit.csv"):
                with open(f, newline="") as r, open(o, "w", newline="") as w:
                    rd = csv.reader(r); wr = csv.writer(w, lineterminator=_eol(f)); hdr = next(rd); wr.writerow(hdr)
                    ti = hdr.index("Time"); n = k = 0
                    for row in rd:
                        n += 1
                        if float(row[ti]) <= cutoff_time_s:
                            wr.writerow(row); k += 1
                stats[f.name] = dict(kept=k, dropped=n - k)
            elif f.name == "data-recovery-report.csv":
                with open(f, newline="") as r, open(o, "w", newline="") as w:
                    rd = csv.reader(r); wr = csv.writer(w, lineterminator=_eol(f)); hdr = next(rd); wr.writerow(hdr); n = k = 0
                    for row in rd:
                        n += 1
                        if float(row[0]) <= cutoff_time_s:
                            wr.writerow(row); k += 1
                stats[f.name] = dict(kept=k, dropped=n - k)
            else:
                shutil.copy2(f, o)
        notes["mocap"][str(rel)] = stats

    # raw Motive export(s): top-level *.csv whose header starts with "Format Version"
    for f in sorted(src.glob("*.csv")):
        with open(f, newline="") as r:
            first = next(csv.reader(r), [])
        if not (first and first[0] == "Format Version"):
            continue
        anchors = [x for x in mocap_plan.values() if x is not None]
        if not anchors:
            shutil.copy2(f, out / f.name); continue
        anchor = min(anchors)   # devices have slightly different anchors; the smallest keeps the most raw rows (never cuts data a device needs)
        cutoff_time_s = (t_end - anchor) / 1e9
        eol = _eol(f)
        with open(f, newline="") as r:
            lines = r.read().split(eol)
        hdr_end = next(i for i, l in enumerate(lines) if l.startswith("Frame,Time"))   # 'Frame,Time,X,Y,...' row
        data = [l for l in lines[hdr_end + 1:] if l.strip()]
        keep = [l for l in data if float(l.split(",", 2)[1]) <= cutoff_time_s]
        meta = next(csv.reader([lines[0]]))
        for key in ("Total Frames in Take", "Total Exported Frames"):
            if key in meta:
                meta[meta.index(key) + 1] = str(len(keep))
        with open(out / f.name, "w", newline="") as w:
            w.write(",".join(meta) + eol)
            w.write(eol.join(lines[1:hdr_end + 1]) + eol + eol.join(keep) + eol)
        notes["raw_motive"] = {f.name: dict(kept=len(keep), dropped=len(data) - len(keep), cutoff_time_s=cutoff_time_s)}

    # any other top-level file/dir that is not a recognised part (README, calib ...) is copied unchanged
    handled = {"mav0", "mocap_filtered"} | {p.name for p in src.glob("mocap_filtered.bak*")} | \
              {f.name for f in src.glob("*.csv")}
    for p in sorted(src.iterdir()):
        if p.name not in handled:
            shutil.copytree(p, out / p.name) if p.is_dir() else shutil.copy2(p, out / p.name)

    txt = (f"# Trimmed copy of {src.name}\n\n"
           f"The full recording ended with a capture stall (camera frames delivered late / in bursts, timestamps up to ~240 ms "
           f"later than the exposure). Every stream was cut at one common time:\n\n"
           f"- cameras: timestamp <= {t_cut} ns\n- IMUs, mocap, raw Motive export: timestamp <= {t_end} ns ({a.margin_s:g} s margin)\n\n"
           f"Not recomputed on the trimmed data (kept as produced from the full recording): mocap_filtered/*/drift_check/*, "
           f"the mocap quality reports (*.png/*.txt) and calibration files.\n"
           f"Rows with impossible timestamps (>= 9e18) were removed.\n\nDetails: TRIM_INFO.json\n")
    (out / "TRIM_NOTES.md").write_text(txt)
    (out / "TRIM_INFO.json").write_text(json.dumps(notes, indent=2))

    # ---- verify
    problems = []
    for name, (_, keep, is_cam) in cam_keep.items():
        limit = t_cut if is_cam else t_end
        _, rows = _read_ts_rows(out / "mav0" / name / "data.csv")
        ts = [t for t, _ in rows]
        if any(t is None or t > limit for t in ts):
            problems.append(f"{name}: timestamp beyond {limit}")
        if is_cam:
            imgs = {p.name for p in (out / "mav0" / name / "data").iterdir()}
            want = {l.strip().split(",")[1] for _, l in keep}
            if imgs != want:
                problems.append(f"{name}: image files {len(imgs)} != rows {len(want)}")
    ref_out = [t for t, _ in _read_ts_rows(out / "mav0" / a.ref_cam / "data.csv")[1] if t is not None]
    if not ref_out or max(ref_out) != t_cut:
        problems.append(f"{a.ref_cam} last timestamp is not T_cut")
    after = _snapshot(src)
    if before != after:
        problems.append("SOURCE CHANGED DURING THE RUN")
    print("\nverification:", "OK -- all streams end at or before their limit, images match rows, source untouched"
          if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
