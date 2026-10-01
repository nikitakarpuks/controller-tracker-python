# Legacy investigation scripts

One-off debugging/investigation scripts written during development, moved here from the
repo root during a 2026-10 cleanup pass. None of these are imported by `src/`, `scripts/`,
`tests/`, or any committed reproducibility script, and none are referenced by filename
anywhere in `TUM-THESIS/`. They are kept (not deleted) because they document *how* specific
bugs were found and fixed — useful context if a similar issue resurfaces — not because they
are meant to be run again as-is.

Most assume the repo root is on `sys.path` and a specific recording is available locally
(often via a hardcoded absolute path under `~/Downloads/...`), so treat them as read-only
reference material rather than a maintained toolchain. The actual reproducibility scripts for
the thesis's figures and tables live in `scripts/` (top level) and `scripts/ablations/`, with
their own READMEs.

Two scripts stand slightly apart from the rest:
- `imu_trust_analysis.py` / `run_imu_trust_analysis_all.py` — the original (now superseded by
  stored aggregates) way to regenerate the coast-budget sweep behind
  `scripts/make_imu_trust_figure.py`'s input data from scratch.

The rest are devlog-style one-shot checks (axis-convention audits, lamp-filter exploration,
pose-fusion jump-detection validation, etc.) corresponding to entries in the thesis's own
development-log appendix.
