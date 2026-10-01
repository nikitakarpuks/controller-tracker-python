#!/usr/bin/env python3
"""make_eval_config.py -- generate a full-recording evaluation config for one
of the recordings-aug26 datasets, from the CURRENT config/config.yml (i.e.
whatever pipeline tuning is live right now).

Only data.root actually changes between recordings -- every mocap_calib_path
and mocap_bridge_path in config.yml is rig/controller-hardware-level, not
per-recording (see config.yml's own comments), and each recording's
mocap_filtered/<device>/drift_check/.../drift_check.json already carries that
recording's own IMU<->mocap fine time offset -- src/mocap_data.py reads it
automatically from data.root's parent, no manual timing step needed per
recording (confirmed: static_dark's headset offset is 138.9ms, walk_dark's is
55.6ms, both already sitting in their own drift_check.json).

Usage:
  python3 make_eval_config.py walk_dark
  python3 make_eval_config.py euroc_recording_20260826173350_walk_dark
  python3 make_eval_config.py /full/path/to/some_other_recording/mav0

Writes config/config_eval_<tag>.yml, with:
  - data.root -> the resolved recording's mav0
  - debug.pose_csv / vision_pose_csv / mocap_log_dir -> data/eval/<tag>/...
  - debug.led_detections_csv / calibration_csv / algorithm_log_dir -> null
  - visualization.enabled -> false, debug.log_* -> false (except log_startup)
    (full-recording runs would otherwise be slow and produce huge logs)
Then run, in order:
  python3 main.py config/config_eval_<tag>.yml
  python3 evaluate_mocap.py config/config_eval_<tag>.yml
  python3 visualization/build_mocap_dashboard.py data/eval/<tag> visualization/mocap_dashboard_<tag>.html
"""
import re
import sys
from pathlib import Path

import yaml

_RECORDINGS_ROOT = Path("/home/nikitakarpuks/Downloads/recordings-aug26")
_TAG_RE = re.compile(r"^euroc_recording_\d+_(.+)$")


def resolve_recording(arg: str) -> tuple:
    """(mav0_path, tag) from a short tag ("walk_dark"), a recording folder
    name, or a full path to a recording dir or its mav0 subdir."""
    p = Path(arg)
    if p.is_absolute() or arg.startswith("."):
        mav0 = p if p.name == "mav0" else p / "mav0"
        if not mav0.exists():
            raise SystemExit(f"no mav0/ under {p}")
        m = _TAG_RE.match(mav0.parent.name)
        tag = m.group(1) if m else mav0.parent.name
        return mav0, tag

    # short tag or bare recording folder name -- look it up under recordings-aug26/
    candidates = [d for d in _RECORDINGS_ROOT.iterdir() if d.is_dir() and d.name.startswith("euroc_recording_")]
    for d in candidates:
        m = _TAG_RE.match(d.name)
        tag = m.group(1) if m else d.name
        if arg in (d.name, tag):
            return d / "mav0", tag
    raise SystemExit(f"couldn't find a recording matching '{arg}' under {_RECORDINGS_ROOT} -- "
                      f"available tags: {sorted(_TAG_RE.match(d.name).group(1) for d in candidates if _TAG_RE.match(d.name))}")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    mav0_path, tag = resolve_recording(sys.argv[1])

    with open("config/config.yml") as f:
        cfg = yaml.safe_load(f)

    cfg["data"]["root"] = str(mav0_path)

    out_dir = f"./data/eval/{tag}"
    cfg["debug"]["pose_csv"] = f"{out_dir}/pose_log_full.csv"
    cfg["debug"]["vision_pose_csv"] = f"{out_dir}/vision_pose_log_full.csv"
    cfg["debug"]["mocap_log_dir"] = f"{out_dir}/mocap_gt"
    cfg["debug"]["led_detections_csv"] = None
    cfg["debug"]["calibration_csv"] = None
    cfg["debug"]["algorithm_log_dir"] = None
    for k in list(cfg["debug"].keys()):
        if k.startswith("log_"):
            cfg["debug"][k] = False
    cfg["debug"]["log_startup"] = True

    cfg["visualization"]["enabled"] = False

    out_path = Path(f"config/config_eval_{tag}.yml")
    with open(out_path, "w") as f:
        f.write(f"# AUTO-GENERATED eval config for recording tag '{tag}' -- see make_eval_config.py.\n")
        f.write(f"# Regenerate whenever config.yml's tuned pipeline settings change.\n")
        f.write(f"# data.root -> {mav0_path}\n")
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    print(f"Wrote {out_path}")
    print(f"  data.root = {mav0_path}")
    print(f"  outputs under {out_dir}/")
    print(f"\nNext:")
    print(f"  python3 main.py {out_path}")
    print(f"  python3 evaluate_mocap.py {out_path}")
    print(f"  python3 visualization/build_mocap_dashboard.py {out_dir} visualization/mocap_dashboard_{tag}.html")


if __name__ == "__main__":
    main()
