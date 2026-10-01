# visualization/

Output of full evaluation runs: one subfolder per run, each holding a Rerun recording plus
per-recording metrics/plots for all 8 evaluation recordings. This directory is large (Rerun
recordings dominate, several hundred MB per recording) and **not git-tracked**; treat every run
folder as regenerable in principle, but check the table below before deleting one, since several
are loaded by exact path from committed scripts or cited directly in the thesis text.

| Folder | What it is | Still needed because |
|---|---|---|
| `evaluate_2026-09-27_full/` | The full pipeline, final configuration | **The** final run. Loaded by `scripts/coverage_decomposition.py`, `TUM-THESIS/scripts/fig_proximity_matching.py`, `TUM-THESIS/scripts/make_eval_figures_0927.py`. |
| `evaluate_2026-09-27_visiononly/` | Same recordings, `fusion.enabled: false` | The vision-only ablation cited throughout the thesis's "Does the IMU Help?" section and its tables. |
| `evaluate_2026-09-27_safeguardsoff/` | Same recordings, swap-detection + rotation-veto disabled | Cited directly in the thesis's reacquisition-safeguard paragraph (574 reacquisition events). |
| `evaluate_2026-09-26/` | Numerically identical to `_full` (only a display-only mesh-handedness fix differs — verified byte-identical `metrics_summary` across all 8 recordings) | `TUM-THESIS/scripts/final_run_tables.py` still hardcodes this exact folder name. **Do not delete without first repointing that script at `_full` and verifying it reproduces the committed `.tex` tables byte-for-byte.** |
| `controller_calibration_for_basalt/` | IMU calibration workspace | Cited by name in `src/imu_data.py`'s module docstring and `main.py` as the evidence trail for the shipped IMU axis-convention fix. |
| `thesis_update_2026_09_27.md`, `thesis_update_ch6_metrics_2026_09_27.md`, `ch6_metrics_2026-09-27.json` | Narrative notes + data behind specific numbers in the abstract, RQ2, and §6 | Most authoritative source for those numbers; not reproduced anywhere else. |
| `report_2026-09-24/` | A dated investigation (minus its own `_tmp/` scratch subfolder, removed) | `tools/rec_analysis.py`'s methodology is reused by the committed `ch6_metrics.py`. |
| `step3/` | Live default output dir for several IMU diagnostic scripts in `scripts/legacy_investigations/` | Refills on the next diagnostic run; safe to clear anytime. |
| `static_medium_swap_slam_cams/` | One-off frame dump from debugging an identity-swap bug (since fixed and shipped) | No code/thesis reference found; likely safe to delete, kept out of caution. |

If you're trying to reclaim disk space: the one real win here is `evaluate_2026-09-26/`
(several GB, numerically redundant with `_full`), gated on the `final_run_tables.py` repoint
above. Everything else is either actively cited or already small.
