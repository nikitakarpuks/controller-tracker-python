#!/usr/bin/env python3
"""One-off follow-up: walk_easy failed in the main run_all_recordings.py
batch (cam7 short one trailing frame vs cam4-6 -- count_images() hard-fails
on that, see run_all_recordings.py's now-added auto-retry). Waits for the
main driver process to finish (so it doesn't oversubscribe the same cores),
then reruns walk_easy with the fix and patches its entry into overview.json.

Usage: python3 fixup_walk_easy.py <driver_pid> <eval_date>
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all_recordings import (REPO_ROOT, RECORDINGS_ROOT, BASE_CONFIG_PATH,  # noqa: E402
                                 run_one_recording, evaluate_and_flag)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main():
    driver_pid = int(sys.argv[1])
    eval_date = sys.argv[2]
    out_dir = REPO_ROOT / "visualization" / f"evaluate_{eval_date}"

    print(f"waiting for driver pid={driver_pid} to finish...", flush=True)
    while pid_alive(driver_pid):
        time.sleep(15)
    print("driver finished, rerunning walk_easy", flush=True)

    rec_dir = RECORDINGS_ROOT / "euroc_recording_20260826175226_walk_easy"
    base_text = BASE_CONFIG_PATH.read_text()
    result = run_one_recording(rec_dir, out_dir, base_text)
    if result["returncode"] == 0:
        eval_result = evaluate_and_flag(result, out_dir)
        result.update(eval_result)
    else:
        print(f"walk_easy STILL failing (exit {result['returncode']}) -- see "
              f"{out_dir / 'walk_easy_run.log'}", flush=True)

    overview_path = out_dir / "overview.json"
    overview = json.loads(overview_path.read_text())
    overview["recordings"] = [r for r in overview["recordings"] if r["name"] != "walk_easy"]
    overview["recordings"].append(result)
    overview_path.write_text(json.dumps(overview, indent=2))
    print("overview.json patched with corrected walk_easy entry", flush=True)


if __name__ == "__main__":
    main()
