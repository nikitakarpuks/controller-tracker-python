# Ablation and runtime infrastructure (thesis, table `tab:eval-plan`)

All scripts run from a **read-only snapshot of a commit**, never from the live tree (another session edits it).
Nothing here modifies `src/`, `config/`, `data/` or `master.tex`. Everything below was smoke-tested on committed HEAD ec3f849.

## One-command reproduction (after a commit is frozen)
```
S=/path/to/snap; O=/path/to/ablation_runs
scripts/ablations/make_snapshot.sh $S <frozen-commit>                 # code snapshot + COMMIT file
python scripts/ablations/run_ablations.py --snapshot $S --out $O --ablations all --recordings all --jobs 3
python scripts/ablations/analyze_ablations.py --runs $O --boot 1000    # -> figures/tables/ablation_*.tex, figures/ablation_paired.pdf, figures/ablation_results.json
```
* `--ablations a,b`, `--recordings static_dark,walk_hard`, `--frames LO:HI --warmup N` (subsequence; warm-up frames are run but excluded from metrics), `--dry-run`, `--force`, `--workers N`, `--viz on|off`.
* Resumable (a job with `DONE.json` returncode 0 is skipped), max 3 concurrent jobs, `nice -n 10`, one retry on the per-camera frame-count mismatch (walk_easy).
* Inputs missing from a git snapshot (untracked bridge/calibration files) are read from the live `data/` tree by absolute path; their SHA-256 hashes are stored in `<out>/PROVENANCE.json` together with the commit.
* `analyze_ablations.py --selftest` reproduces the 2026-09-16 batch numbers exactly through the same code path (bridge composed, `evaluate_mocap.py` reused; each series is cross-checked against `evaluate_controller`).
* Do NOT mix these absolute-residual metrics with the xrtslam-metrics ATE/RTE (Umeyama-aligned) numbers in one table.
* Statistics: percentile bootstrap over 10-s blocks, identical block draws for every ablation (paired deltas). Longest lost streak: exact point value, CI from block-wise streaks (lower-bound style).

## Which ablation is a config switch, which needs code (HEAD ec3f849)
`python scripts/ablations/ablations.py list` prints this; `generate` writes verified configs (asserts that exactly the intended keys differ).

| name | switch | note |
|---|---|---|
| `full` | - | |
| `imu_off` | `imu.enabled=false`, `fusion.enabled=false` | pure vision (also = ladder rung 3) |
| `fusion_off` | `fusion.enabled=false` | filter bypassed, IMU still gives gyro-predicted rotation + IMU-aware vision jump gate |
| `imu_off_filter_on` | `imu.enabled=false` | filter constructed, FAIL-OPEN every warm frame; bootstrap buffer + One Euro remain |
| `no_one_euro` | `fusion_heuristic.one_euro_enabled` | |
| `no_swap_detect` | `matching.cold_swap_margin=0` | |
| `no_rot_veto` | `fusion_heuristic.cold_reacquire_rot_veto_thresh_deg=361` | also disables the past-imu_decay hard reject (same flag) |
| `no_lamp_filter` / `no_lamp_memory` | `blob_detection.lamp_blob_filter.enabled` / `.static_lamp_mask.enabled` | memory is a no-op without the filter |
| `no_twopass` | `blob_detection.pass2_threshold_factor=0` | |
| `no_streak_rescue` | `blob_detection.max_streak_elongation=0` | |
| `no_proximity` | `matching.use_proximity_match=false` | constrained search (2-3 blobs) still on |
| `no_edge_taper` | `matching.edge_confidence_floor=1`, `edge_reproj_widen_max=1` | |
| `ladder_0_brute_only`, `ladder_1_warm`, `ladder_2_lamp` | cumulative; rung 3 = `imu_off`, rung 4 = `full` | |
| `monado_like_brute` | ladder_0 + `strong_match_inliers=7`, `strong_match_error_px=1.5` | an approximation only |
| **needs code** `no_weak_buffer` | `src/pose_fusion_heuristic.py:2025` (`weak = ...` in `_try_cold_reacquire`; bootstrap route ~1178) | `vision_weight_weak_inliers` is not a clean switch |
| **needs code** `no_constrained` | `src/controller.py:562-563` (`2 <= n_available <= 3`) | only call site |
| **needs code** `no_warm_detection` | `main.py` where `predicted_leds` is built (`blob_detector.py:2446-2479` has no 'off') | |
| **needs code** `log_imu_only_frames` | `main.py:1301-1430`; display coast `pose_fusion_heuristic.py:1947` | needed to credit IMU bridging |
| **needs code** `no_coverage_fallback`, `no_weak_solo` | `pose_search.py:~2700-2740`; `controller.py:178` | |

## Smoke-test results (walk_medium frames 3000:3200, +100 warm-up, HEAD snapshot, shared machine, load 1.3-3.7, n=1 per cell)
Per-frame wall time from the pipeline's own log lines (`timing.py`), 199 frames:

| config | median ms | mean ms | p95 ms | speed-up (mean) |
|---|---|---|---|---|
| 1 worker | 49 | 88 | 342 | 1.00 |
| 2 workers | 40 | 71 | 220 | 1.24 |
| 4 workers | 33 | 58 | 184 | 1.51 |
| 6 workers | 43 | 70 | 194 | 1.25 |
| `parallel_search_enabled=false` | 47 | 83 | 285 | 1.05 |

Job time of one 300-frame job (6 workers, solo): full 41 s, no_one_euro 34, no_twopass 36, ladder_1 94, no_lamp_filter 105, no_proximity 132, imu_off 135, ladder_0 254 (6.2x). Three concurrent jobs with 2 workers each ran a cheap job in 34-39 s (= solo speed).

## Full-run cost estimate
Batch of 2026-09-16 (viz on, 6 workers): 9753 s = 2.71 h for 8 recordings; viz off is ~20 % faster (41 s vs 51 s) => ~2.2 h per ablation for `full`. Multipliers from the smoke test (unmeasured ones assumed): 17 config-only ablations sum to ~39 x => ~85 h sequential-equivalent; with 3 concurrent jobs (near-linear here) ~28-35 h wall. A minimal thesis set (full, imu_off, fusion_off, no_swap_detect, no_rot_veto, no_lamp_filter, no_lamp_memory, ladder_0/1/2 = ~23 x) ~50 h => ~17 h wall. Ratios come from one mid-recording segment; hard recordings may differ.

## Surprises
1. `parallel_search_enabled=false` is NOT result-equivalent to the process pool (4 fewer tracked rows, 114/563 common rows differ by >0.01 mm, max 5.9 mm). Worker count and job concurrency, in contrast, give byte-identical poses. Never use `--sequential` for evaluation.
2. `visualization.enabled` / `pose_fusion_debug` on vs off gives byte-identical `pose.csv` and `vision_pose.csv` (so the runner disables them: no 1 GB .rrd per job).
3. Removing the lamp filter or the warm path slows the run 2.6-6x (candidate-blob explosion in brute force), so ablation cost is dominated by those.
4. Worker scaling is weak (1 -> 4 workers: 1.5x; 6 is not better than 4 on this shared machine), hence 3 jobs x 2 workers is the efficient layout.
5. Config-relative paths (`./data/...`) in a git snapshot miss untracked files (bridges); the runner absolutizes them.
