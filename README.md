# Controller Tracker

A correspondence-free, vision-led 6-DoF pose tracker for VR/XR hand-held controllers that
carry a ring of LEDs, tracked from a headset's own fisheye cameras. The controller's IMU
supports the vision pipeline (prediction, implausibility gating, short-gap coasting) but never
drives it. Written for a Master's thesis; evaluated against OptiTrack motion-capture ground
truth on an HP Reverb G2 rig.

This README is the map of the repo and the quickest path to running the pipeline. Deeper
technical references are linked from the relevant section below rather than duplicated here.

## Repo map

| Path | What's there |
|---|---|
| `main.py` | Entry point. `python main.py [config_path]` (default `config/config.yml`). |
| `src/` | The tracking pipeline itself: blob detection, correspondence search (brute-force / proximity / constrained), pose-fusion filter, camera/controller models, mocap loading. |
| `config/` | `config.yml` (the live default) plus named variants used for specific evaluation runs (`config_eval*.yml`, `config_sweep_*.yml`, `config_smf_*.yml`, `config_final_static_medium.yml`, `config_static_light_probe.yml`). |
| `data/` | Calibration inputs (camera intrinsics, controller LED/IMU models, mocap bridge transforms) and historical development recordings. See [Data](#data) below — the live pipeline reads recordings from *outside* this directory by default. |
| `tests/` | `pytest` unit/regression suite (44 files) covering the pipeline modules in `src/`. |
| `scripts/` | The maintained reproducibility toolchain: evaluation-metric scripts, the ablation-study harness (`scripts/ablations/`, its own README), and figure/table generators. |
| `scripts/legacy_investigations/` | ~40 one-off debugging scripts from development, kept for context on how specific bugs were found and fixed. Not maintained, not imported by anything — see its own README before trying to run one. |
| `benchmarks/` | Microbenchmarks for the P3P/blob-matching hot path. |
| `analysis/` | Larger standalone research investigations (IMU bias characterization, drift fitting) that fed specific thesis findings; each subfolder documents its own methodology and conclusions. |
| `visualization/` | Output of full evaluation runs (Rerun recordings, per-recording metrics, plots). Large — see the note in that directory if you're regenerating thesis numbers, since several specific run folders are cited by exact name in committed scripts. |
| `docs/` | [`docs/parameters.md`](docs/parameters.md) (config parameter reference) and [`docs/algorithm_overview.md`](docs/algorithm_overview.md) (pipeline flowcharts). |

## Environment

Python 3.12. Install pinned dependencies with:

```bash
pip install -r requirements.txt
```

Core pipeline: NumPy, SciPy, OpenCV, PyYAML. Evaluation scripts additionally use pandas and
Matplotlib. The live visualizer uses the [Rerun](https://rerun.io) SDK. The pipeline is a
research prototype, not real-time.

## Running the pipeline

```bash
python main.py config/config.yml
```

Before your first run, open `config/config.yml` and update `data.root` to point at your own
recording — it currently holds a path from the machine it was developed on
(`/home/.../Downloads/recordings-aug26/.../mav0`), not a path that exists on a fresh checkout.
The comments throughout `config.yml` explain every other field; `docs/parameters.md` documents
the matching/detection parameters in more depth.

## Tests

```bash
pytest tests/
```

## Data

`data/` holds calibration inputs the pipeline actually loads by path from `config.yml`:
camera intrinsics (`data/cameras/`), per-controller LED position/IMU models
(`data/controllers/`), and the controller↔mocap bridge transform (`data/mocap_calib/`).

`data/datasets/` holds **earlier development-phase recordings**, not the recording the default
`config.yml` points at (that one lives outside the repo, per the path you set above). They're
kept because several hardcoded tuning constants in `config.yml` cite a specific recording from
here as their empirical source — useful provenance, not a live dependency.

The Reverb G2 controller's 3D model (`data/controllers/reverbg2.glb`, plus any `.stp`/`.step`
CAD file) is a licensed/mentor-provided asset and is **git-ignored on purpose** — see the
comments in `.gitignore`. It must stay local only; never commit it.

## Reproducing thesis figures/tables

The scripts under `scripts/` (and the ablation harness in `scripts/ablations/`, which has its
own detailed README covering one-command reproduction, cost estimates, and which ablations
need a code change vs. a config flag) are the maintained path from a recorded run to the
numbers and figures cited in the thesis. `scripts/legacy_investigations/` is explicitly not
part of that path.
