#!/usr/bin/env python3
"""Per-frame wall-time measurement of the tracker, using ONLY the pipeline's own log lines
(no instrumentation of src/): each processed frame logs, per controller,
  [<image stem>.png]  [<ctrl>]  X ms[blob] + Y ms[pose] = Z ms ...
with a millisecond wall-clock prefix. Per-frame wall time is the difference between the first log timestamps of two
CONSECUTIVE image indices (frames without any controller line are skipped, not averaged in).

  python timing.py run    --snapshot S --out O [--recordings walk_medium] [--frames 3000:3200] [--warmup 100]
                          [--workers 1,2,4,6] [--sequential]
  python timing.py report --out O

Runs are strictly sequential (one main.py at a time, nice 10) so that worker-count effects are not confounded by
concurrent jobs; the load average of the machine is recorded with each run because other sessions may share the CPU.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LINE = re.compile(r"(\d\d:\d\d:\d\d\.\d{3}) \| INFO\s+\| \[(\d+)\.png\]\s+\[(\w+)\]\s+([\d.]+)ms\[blob\] \+ ([\d.]+)ms\[pose\] = ([\d.]+)ms")


def _t(s):
    d = datetime.strptime(s, "%H:%M:%S.%f")
    return d.hour * 3600 + d.minute * 60 + d.second + d.microsecond / 1e6


def parse_log(log, stems, eval_from_ts):
    idx = {s: i for i, s in enumerate(stems)}
    first, per_ctrl = {}, []
    for line in Path(log).read_text(errors="replace").splitlines():
        m = LINE.search(line)
        if not m:
            continue
        ts, stem = _t(m.group(1)), int(m.group(2))
        if stem < eval_from_ts:
            continue
        first.setdefault(stem, ts)
        per_ctrl.append((float(m.group(4)), float(m.group(5)), float(m.group(6))))
    ordered = sorted(first, key=lambda s: idx[s])
    walls = [first[b] - first[a] for a, b in zip(ordered, ordered[1:]) if idx[b] == idx[a] + 1]
    pc = np.array(per_ctrl) if per_ctrl else np.zeros((0, 3))
    return walls, pc


def report(out):
    out = Path(out)
    rows = []
    for d in sorted(out.iterdir()):
        for jd in sorted(d.glob("*/*")) if d.is_dir() else []:
            done = jd / "DONE.json"
            if not done.exists():
                continue
            dj = json.loads(done.read_text())
            import yaml
            cfg = yaml.safe_load((jd / "config.yml").read_text())
            root = Path(cfg["data"]["root"])
            cam = root / cfg["data"]["camera_folder_pattern"].format(idx=int(cfg["data"].get("controller_cam_start_index", 0))) / cfg["data"].get("images_subdir", "")
            stems = sorted(int(p.stem) for p in cam.glob("*.png"))
            walls, pc = parse_log(jd / "run.log", stems, dj.get("eval_from_ts") or 0)
            meta = json.loads((jd / "timing_meta.json").read_text()) if (jd / "timing_meta.json").exists() else {}
            w = np.array(walls) * 1000
            rows.append({"config": d.name, "n_frames": len(w), "wall_ms_median": float(np.median(w)), "wall_ms_mean": float(np.mean(w)),
                         "wall_ms_p95": float(np.percentile(w, 95)), "ctrl_ms_median": float(np.median(pc[:, 2])) if len(pc) else None,
                         "ctrl_ms_p95": float(np.percentile(pc[:, 2], 95)) if len(pc) else None,
                         "blob_ms_median": float(np.median(pc[:, 0])) if len(pc) else None,
                         "pose_ms_median": float(np.median(pc[:, 1])) if len(pc) else None,
                         "job_elapsed_s": dj["elapsed_s"], "load_avg_start": meta.get("load_avg_start")})
    base = next((r for r in rows if r["config"] == "w1"), None)
    print(f"{'config':10s} {'n':>4s} {'wall med':>9s} {'wall mean':>10s} {'wall p95':>9s} {'ctrl med':>9s} {'ctrl p95':>9s} {'speedup*':>9s} {'job s':>7s} {'load0':>6s}")
    for r in rows:
        sp = base["wall_ms_mean"] / r["wall_ms_mean"] if base else float("nan")
        r["speedup_vs_w1_mean"] = sp
        print(f"{r['config']:10s} {r['n_frames']:4d} {r['wall_ms_median']:9.1f} {r['wall_ms_mean']:10.1f} {r['wall_ms_p95']:9.1f} "
              f"{(r['ctrl_ms_median'] or 0):9.1f} {(r['ctrl_ms_p95'] or 0):9.1f} {sp:9.2f} {r['job_elapsed_s']:7.0f} {(r['load_avg_start'] or 0):6.2f}")
    print("* speedup = mean wall per frame of w1 / mean wall per frame of this config; ms values in milliseconds")
    (out / "timing_summary.json").write_text(json.dumps(rows, indent=1))


def run(a):
    out = Path(a.out)
    configs = [(f"w{w}", ["--workers", str(w)]) for w in a.workers.split(",")]
    if a.sequential:
        configs.append(("seq", ["--sequential", "--workers", "1"]))
    for name, extra in configs:
        jd = out / name
        if (jd / "full" / a.recordings / "DONE.json").exists():
            print("skip", name)
            continue
        load = os.getloadavg()[0]
        cmd = [sys.executable, str(HERE / "run_ablations.py"), "--snapshot", a.snapshot, "--out", str(jd), "--ablations", "full",
               "--recordings", a.recordings, "--frames", a.frames, "--warmup", str(a.warmup), "--jobs", "1"] + extra
        print(">>", name, f"(load avg at start {load:.2f})", flush=True)
        subprocess.run(cmd, check=True)
        (jd / "full" / a.recordings / "timing_meta.json").write_text(json.dumps({"load_avg_start": load}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--snapshot", required=True); r.add_argument("--out", required=True)
    r.add_argument("--recordings", default="walk_medium"); r.add_argument("--frames", default="3000:3200")
    r.add_argument("--warmup", type=int, default=100); r.add_argument("--workers", default="1,2,4,6")
    r.add_argument("--sequential", action="store_true")
    p = sub.add_parser("report"); p.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a.out)


if __name__ == "__main__":
    main()
