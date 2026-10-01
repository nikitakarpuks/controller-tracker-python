#!/usr/bin/env python3
"""One-off cleanup: sort ./visualization/evaluate_<date>/'s flat pile of
per-recording files into one subfolder per recording, with the redundant
<name>_ prefix dropped inside each (recording_<name>.rrd -> <name>/recording.rrd,
etc.), and patches overview.json's absolute paths to match.

Usage: python3 reorganize_evaluate_dir.py <eval_date>
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

RENAME_MAP = {
    "recording_{name}.rrd": "recording.rrd",
    "{name}_pose.csv": "pose.csv",
    "{name}_vision_pose.csv": "vision_pose.csv",
    "{name}_mocap_gt": "mocap_gt",
    "{name}_anomalies.csv": "anomalies.csv",
    "{name}_metrics_summary.json": "metrics_summary.json",
    "{name}_run.log": "run.log",
    "_config_{name}.yml": "config.yml",
}


def main():
    eval_date = sys.argv[1]
    out_dir = REPO_ROOT / "visualization" / f"evaluate_{eval_date}"
    overview_path = out_dir / "overview.json"
    overview = json.loads(overview_path.read_text())

    names = [r["name"] for r in overview["recordings"]]
    for name in names:
        rec_dir = out_dir / name
        rec_dir.mkdir(exist_ok=True)
        for pattern, new_name in RENAME_MAP.items():
            src = out_dir / pattern.format(name=name)
            if src.exists():
                src.rename(rec_dir / new_name)
            else:
                print(f"  [{name}] missing (skipped): {src.name}")

    # Patch overview.json's absolute paths to the new subfolder locations.
    field_to_filename = {
        "pose_csv": "pose.csv",
        "vision_pose_csv": "vision_pose.csv",
        "mocap_log_dir": "mocap_gt",
        "rrd": "recording.rrd",
        "anomalies_csv": "anomalies.csv",
    }
    for r in overview["recordings"]:
        name = r["name"]
        rec_dir = out_dir / name
        for field, filename in field_to_filename.items():
            if field in r:
                r[field] = str(rec_dir / filename)
    overview_path.write_text(json.dumps(overview, indent=2))

    print(f"Reorganized {len(names)} recordings under {out_dir}")
    print(f"Top level now: {sorted(p.name for p in out_dir.iterdir())}")


if __name__ == "__main__":
    main()
