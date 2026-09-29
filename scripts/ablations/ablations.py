#!/usr/bin/env python3
"""Ablation definitions and config generator for the thesis's final evaluation.

Every ablation is a set of (key path -> value) edits on the committed config.yml of a
code snapshot; `generate` writes one config per ablation and VERIFIES (by parsing both
YAMLs) that exactly the intended keys differ. Ablations that need a code change (no clean
config switch) are listed with the exact code location and are NOT generated.

Usage:
  python ablations.py list
  python ablations.py generate --base <snapshot>/config/config.yml --out <dir> [--names a,b]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yaml_patch import set_value, diff_keys  # noqa: E402

FH = ["fusion_heuristic"]
BD = ["blob_detection"]
LF = BD + ["lamp_blob_filter"]
MT = ["matching"]

# name -> (description, {key path tuple: value})
ABLATIONS = {
    "full": ("shipped configuration of the snapshot", {}),
    # ---- IMU / fusion ---------------------------------------------------------------
    "imu_off": (
        "pure vision: no IMU loaded (no gyro-predicted rotation for search, no accel/gyro-aware "
        "jump thresholds), no fusion filter (no gates, coast, buffer, One Euro)",
        {("imu", "enabled"): False, ("fusion", "enabled"): False}),
    "fusion_off": (
        "fusion filter bypassed (raw vision reported: no gates/pushback/coast/weak buffer/One Euro), "
        "but the IMU is still loaded: gyro-predicted rotation for search + IMU-aware vision-side jump gate remain",
        {("fusion", "enabled"): False}),
    "imu_off_filter_on": (
        "IMU not loaded but filter constructed: every warm frame takes the filter's FAIL-OPEN branch; "
        "bootstrap weak-candidate buffer, sibling checks and One Euro remain",
        {("imu", "enabled"): False}),
    "no_one_euro": ("reported-pose smoothing off",
                    {tuple(FH + ["one_euro_enabled"]): False}),
    "no_swap_detect": ("cold identity-swap detection off (swapped < 0 x direct never true)",
                       {tuple(MT + ["cold_swap_margin"]): 0.0}),
    "no_rot_veto": ("rot_pred_implausible veto off (threshold above any possible angle); also disables "
                    "the speed-aware hard reject past imu_decay_frames, which reuses the same flag",
                    {tuple(FH + ["cold_reacquire_rot_veto_thresh_deg"]): 361.0}),
    # ---- detection ------------------------------------------------------------------
    "no_lamp_filter": ("lamp-fixture filter off (lamp region memory is then a no-op: it is only created "
                       "from strictly recognized rows)",
                       {tuple(LF + ["enabled"]): False}),
    "no_lamp_memory": ("lamp region memory off, line-finder kept",
                       {tuple(LF + ["static_lamp_mask", "enabled"]): False}),
    "no_twopass": ("second detection pass off (pass2_threshold_factor 0)",
                   {tuple(BD + ["pass2_threshold_factor"]): 0}),
    "no_streak_rescue": ("motion-streak rescue off (max_streak_elongation 0: no component qualifies)",
                         {tuple(BD + ["max_streak_elongation"]): 0.0}),
    # ---- search ---------------------------------------------------------------------
    "no_proximity": ("proximity warm matching off: warm frames with >=4 blobs fall through to brute force "
                     "(constrained search for 2-3 blobs remains: no switch)",
                     {tuple(MT + ["use_proximity_match"]): False}),
    "no_edge_taper": ("edge-of-frame confidence off: taper weight 1 everywhere, no threshold widening",
                      {tuple(MT + ["edge_confidence_floor"]): 1.0,
                       tuple(MT + ["edge_reproj_widen_max"]): 1.0}),
    # ---- baseline ladder (cumulative; imu_off doubles as rung 3, full as rung 4) ----------
    "ladder_0_brute_only": (
        "rung 0: brute force only (no proximity), no lamp filter, no second pass, no IMU, no fusion",
        {("imu", "enabled"): False, ("fusion", "enabled"): False,
         tuple(MT + ["use_proximity_match"]): False, tuple(LF + ["enabled"]): False,
         tuple(BD + ["pass2_threshold_factor"]): 0}),
    "ladder_1_warm": (
        "rung 1: + proximity warm matching",
        {("imu", "enabled"): False, ("fusion", "enabled"): False,
         tuple(LF + ["enabled"]): False, tuple(BD + ["pass2_threshold_factor"]): 0}),
    "ladder_2_lamp": (
        "rung 2: + lamp filter and region memory",
        {("imu", "enabled"): False, ("fusion", "enabled"): False,
         tuple(BD + ["pass2_threshold_factor"]): 0}),
    # rung 3 = imu_off (+ two-pass), rung 4 = full (+ IMU and fusion)
    "monado_like_brute": (
        "approximation of a Monado/OpenHMD-style tracker: ladder_0 with the upstream strong-match rule "
        "(7 LEDs, 1.5 px). NOT identical to Monado: constrained search, warm blob detection, joint "
        "multi-camera refinement, coverage fallback, edge taper remain (no switches)",
        {("imu", "enabled"): False, ("fusion", "enabled"): False,
         tuple(MT + ["use_proximity_match"]): False, tuple(LF + ["enabled"]): False,
         tuple(BD + ["pass2_threshold_factor"]): 0,
         tuple(MT + ["strong_match_inliers"]): 7, tuple(MT + ["strong_match_error_px"]): 1.5}),
}

# name -> (description, exact code location in the HEAD snapshot, what to change)
NEEDS_CODE = {
    "no_weak_buffer": (
        "weak-candidate buffer off (trust every reacquisition immediately)",
        "src/pose_fusion_heuristic.py:2025 `weak = (coverage_fallback or n_inliers <= weak_inliers or ...)` "
        "in _try_cold_reacquire (also reached from the bootstrap branch, try_update ~line 1178)",
        "add a config flag (e.g. fusion_heuristic.cold_weak_buffer_enabled) making `weak` False; "
        "vision_weight_weak_inliers is NOT a clean switch (it also drives the vision-weight ramps and "
        "coverage_fallback / contested / swap_suspected / rot_pred still force weak)"),
    "no_constrained": (
        "constrained (P2P/P1P) search off",
        "src/controller.py:562-563 in cheap_search_core: `if solution is None and prev_assignment is not None "
        "and 2 <= n_available <= 3:` (the only call site of constrained_search)",
        "add a config flag (e.g. matching.use_constrained_search) to that condition"),
    "no_warm_detection": (
        "full-frame cold blob detection on every frame (no predicted-LED-neighborhood crop)",
        "main.py: where per-camera kwargs['predicted_leds'] are built (has_prior = predicted_leds is not None, "
        "src/blob_detector.py:2446-2479 warm_strategy 'fit'|'hybrid' has no 'off' value)",
        "pass predicted_leds=None for every camera when a flag is set"),
    "log_imu_only_frames": (
        "log the reported (IMU-coasted) pose on frames WITHOUT a vision candidate, to credit IMU bridging",
        "main.py:1301-1430 (pose_csv rows are written only for frames that produced a solution; the display "
        "coast pose of _report_if_still_usable, src/pose_fusion_heuristic.py:1947, is never logged for "
        "vision-less frames)",
        "write a second CSV (or rows flagged imu_only) from the filter's reported_R/reported_p on every frame "
        "while the display budget allows"),
    "no_coverage_fallback": (
        "brute-force coverage fallback off",
        "src/pose_search.py:~2700-2740 (state.fallback_solution with coverage_fallback=True)",
        "guard the fallback branch with a flag; min_vis_coverage 0 is NOT equivalent (removes the gate)"),
    "no_weak_solo": (
        "weak solo accept guard off",
        "src/controller.py:178 (_weak_solo_accept_cids) called at ~2531/2540 and in TrackingSystem.update_warm_batch",
        "flag in _weak_solo_accept_cids returning an empty set; weak_solo_blob_utilization_floor 0 alone "
        "still leaves the strong_match_inliers criterion"),
}


def generate_text(base_text, name):
    if name in NEEDS_CODE:
        raise SystemExit(f"'{name}' needs a code change (no clean config switch): {NEEDS_CODE[name][1]}")
    desc, edits = ABLATIONS[name]
    text = base_text
    for path, val in edits.items():
        text = set_value(text, list(path), val)
    changed = {tuple(k) for k in diff_keys(base_text, text)}
    if changed != {tuple(k) for k in edits}:
        raise AssertionError(f"{name}: intended keys {sorted(edits)} != changed keys {sorted(changed)}")
    return text


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    g = sub.add_parser("generate")
    g.add_argument("--base", required=True, help="snapshot config/config.yml")
    g.add_argument("--out", required=True)
    g.add_argument("--names", default="all")
    a = ap.parse_args()
    if a.cmd == "list":
        print("CONFIG-ONLY ablations:")
        for n, (d, e) in ABLATIONS.items():
            print(f"  {n:22s} {d}\n{'':24s}edits: " + (", ".join(f"{'/'.join(k)}={v}" for k, v in e.items()) or "-"))
        print("\nNEED CODE CHANGE (not generated):")
        for n, (d, loc, fix) in NEEDS_CODE.items():
            print(f"  {n:22s} {d}\n{'':24s}where: {loc}\n{'':24s}change: {fix}")
        return
    base_text = Path(a.base).read_text()
    names = list(ABLATIONS) if a.names == "all" else a.names.split(",")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for n in names:
        txt = generate_text(base_text, n)
        (out / f"{n}.yml").write_text(txt)
        print(f"wrote {out / (n + '.yml')}  (verified: only {len(ABLATIONS[n][1])} key(s) differ)")


if __name__ == "__main__":
    main()
