# CANVAS: Competency-Aware Navigation Assessment for Pedestrian Trajectory Forecasting

CANVAS evaluates pedestrian-trajectory predictors by running a unicycle ego-robot through
dataset-driven, open-loop replays of recorded crowds (ETH/UCY + the in-house
SNU-ASRI lobby dataset) and computing per-frame **competency indices** derived from
adaptive conformal prediction.

The pipeline is deliberately lightweight: no physics engine, no ROS. A single call to
`examples/simulation_mpc_lqt.py` runs a complete predict-and-control experiment and
produces per-frame competency indices, cost breakdowns, empirical miscoverage curves,
and (optionally) rendered frames.

---

## Table of Contents
1. [Installation](#installation)
2. [Dataset / Model Assets](#dataset--model-assets)
3. [Usage Examples](#usage-examples)
4. [Visualization Guide](#visualization-guide)
5. [Project Structure](#project-structure)
6. [Extending CANVAS](#extending-canvas)
7. [Troubleshooting](#troubleshooting)
8. [License](#license)

---

## Installation

### 1. Clone the repository with submodules

The MPPI controller depends on a git submodule (`pytorch_mppi`), so the recursive clone
is mandatory:

```bash
git clone --recurse-submodules https://github.com/CORE-SNU/CANVAS.git
cd CANVAS
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 2. Create the conda environment

The reference environment (`conformal`) is fully specified in `environment_full.yml`
(Python 3.13, PyTorch 2.6, NumPy 2.1):

```bash
conda env create -f environment_full.yml
conda activate conformal
```

### 3. Install the MPPI submodule (editable)

The submodule at `canvas/controllers/pytorch_mppi` must be installed in editable mode
so that `KernelMPPI` can import it:

```bash
pip install -e canvas/controllers/pytorch_mppi
```

This is a fork of [pytorch-mppi][pytorch-mppi-link]; PyTorch must already be installed
(the conda env above takes care of this).
### 4. Install the canvas submodule (editable)
The submodule at `canvas` must also be installed in editable mode
so that any file can import it:

```bash
pip install -e .
```
### 5. (Optional) Headless / no-display machines

If you are on a server without an X display, export:

```bash
export QT_QPA_PLATFORM=offscreen
```

---

## Dataset / Model Assets

Datasets (`.npy` arrays) and pretrained weights (Trajectron++ etc.) are **not**
checked into the repository. From the repo root run:

```bash
bash assets/download_assets.sh
```

This uses `gdown` to populate:

```
assets/
  datasets/       # ETH/UCY and SNU-ASRI trajectory tensors
  videos/         # raw scene videos used by the visualizer
  models/         # pretrained Trajectron++ checkpoints
```

For scene rendering you also need **per-frame background images** under
`assets/frames/<scene>/<t>.png`. The ETH/UCY folders can be parsed from the downloaded
videos; see `assets/video_parser.py`. The SNU-ASRI scenes use the single static image
`assets/snu-asri.png`.

### Hardcoded paths to edit

Several scripts still contain absolute paths that must be adapted to your machine:

- `default_path_to_frames` argument inside `examples/simulation_mpc_lqt.py`,
  `examples/simulation_mppi_lqt.py`, and files under `tests/`.
- Pretrained model paths inside the `Predictors` constructor
  (`canvas/predictors/predictor.py`) are interpreted relative to the CWD. Run all
  scripts from the repo root, and place model files at the literal expected paths.

---

## Usage Examples

All runnable experiment drivers live under `examples/`. The canonical, up-to-date
driver that wires every component together is
**`examples/simulation_mpc_lqt.py`** — use it as the reference when building your own
script.

### Quickstart (canonical example)

```bash
python examples/simulation_mpc_lqt.py \
    --dataset zara1 \
    --predictor traj \
    --predictor_base linear \
    --visualize
```

What this invocation does, per frame:

1. Selects the `zara1` scene from `RegisteredDatasets`.
2. Runs **Trajectron++** (`traj`) as the primary predictor and the **linear**
   forecaster as the baseline denominator for ratio-based scoring.
3. Feeds both predictions into `BaseMPC` (IPOPT-refined grid MPC).
4. Updates three **conformalized competency indices** driven by a
   `LinearQuantileTracker`: `PD` (positional displacement), `AD` (action divergence),
   and `PR` (planning regret).
5. Computes hindsight analogues against the ground-truth future for comparative
   evaluation.
6. Optionally renders per-frame PDFs into `./viz_mpc_<dataset>/`.

### Output artifacts

| File | Contents |
| --- | --- |
| `viz_mpc_<dataset>/NNN.pdf` | Per-frame scene render with predictions, open-loop plan, CI overlay (only when `--visualize`). |
| `indices.pdf` | Competency-index trajectories (predicted vs. hindsight) over the run. |
| `scores.pdf` | Raw (un-conformalized) score histories. |
| `coverage.pdf` | Empirical miscoverage curves vs. target `1-alpha`. |

### Command-line flags (`simulation_mpc_lqt.py`)

| Flag | Default | Description |
| --- | --- | --- |
| `--dataset` | `zara1` | One of `eth`, `hotel`, `zara1`, `zara2`, `univ`, `snu-asri`. Each scene has a pre-configured init pose / goal / `t_begin` / `t_end` inside the script. |
| `--predictor` | `traj` | Primary predictor. See table below. |
| `--predictor_base` | `linear` | Baseline predictor for ratio-based score normalization. |
| `--num_iter` | `1` | Number of repeated runs. |
| `--visualize` | off | Render per-frame PDFs. |
| `--video_fps` | `2.5` | FPS for video composition (used by downstream tooling). |

Supported predictor names:

| Name | Model |
| --- | --- |
| `linear` | Constant-velocity baseline (always the `--predictor_base`). |
| `traj` | [Trajectron++][trajectronpp-link] |
| `eigen` | [EigenTrajectory][eigentraj-link] |
| `koopcast` | [KoopCast][koopcast-link] |
| `socialvae` | [Social-VAE][socialvae-link] |
| `STGCNN` / `socialstgcnn` | [Social-STGCNN][socialstgcnn-link] |
| `gp` | Gaussian-process baseline |
| `pytorch` | Generic torch wrapper |

### The two drivers

`examples/` ships only two scripts, both using the same pipeline and differing only in
the controller:

| Script | Controller | Notes |
| --- | --- | --- |
| `simulation_mpc_lqt.py` | `BaseMPC` (grid search + IPOPT refinement) | Canonical driver. Flags documented above. |
| `simulation_mppi_lqt.py` | `VanillaMPPI` (sampling-based, GPU-batched) | Same CLI flags; use when MPC is too slow or you want stochastic control. |

Both scripts share the same score functions (PD, AD, PR), the same
`LinearQuantileTracker`-driven conformal indices, and the same hindsight analogues.
Pick whichever controller fits your experiment.

### Sanity-check scripts

Stand-alone scripts under `tests/` exercise individual subsystems. They are plain
Python, not `pytest` modules — run them directly:

```bash
python tests/test_registered_datasets.py   # check dataset loading
python tests/test_env_visualization.py     # check Environment.render()
python tests/test_truncated_history.py     # check history windowing
python tests/test_mppi.py                  # check the MPPI controller
```

---

## Visualization Guide

### Per-frame rendering

Set `--visualize` in `examples/simulation_mpc_lqt.py` (or the equivalent flag in any
other driver) to write one PDF per frame into `./viz_mpc_<dataset>/`. Rendering is
performed by `Environment.render()` in `canvas/envs/env.py`, which:

1. Loads the background image `assets/frames/<dataset_label>/<t>.png`
   (or `assets/snu-asri.png` for SNU-ASRI).
2. Applies the per-scene homography `assets/homographies/<dataset>.txt` to warp world
   coordinates into pixel coordinates.
3. Overlays pedestrian histories, the ego-robot sprite (`assets/robot.png`), static
   obstacle regions, predicted trajectories, and the MPC open-loop plan.
4. Colors each pedestrian by a supplied per-frame **competency value** when the `c=`
   / `hc=` kwargs are passed.

The kwargs consumed by `render()` in `simulation_mpc_lqt.py` are:

```python
fig, ax = env.render(
    c=indices.get_history(name='PR'),      # predicted CI trajectory
    hc=h_indices.get_history(name='PR'),   # hindsight CI trajectory
    open_loop=X,                           # model-driven MPC plan
    open_loop_gt=X_gt,                     # oracle MPC plan under GT future
    open_loop_base=X_base,                 # baseline MPC plan
)
```

### Aggregate plots

After the main loop, `simulation_mpc_lqt.py` emits three summary figures:

- **`indices.pdf`** — predicted vs. hindsight competency index for each registered
  score.
- **`scores.pdf`** — raw score histories (hindsight PD shown by default).
- **`coverage.pdf`** — cumulative empirical miscoverage for each score, comparing
  the conformalized index to a moving-average reference. The horizontal dashed line
  marks the target `1 - alpha`.

### Assembling a thumbnail / gallery

The helper at `tools/make_thumbnail.py` runs the canonical example on a list of scenes
and composes a grid image:

```bash
python tools/make_thumbnail.py \
    --predictor traj \
    --scenes zara1 zara2 hotel eth univ snu-asri \
    --path_to_frames /abs/path/to/assets/frames \
    --output thumbnail.png
```

### Building a video from PDF frames

The per-frame PDFs can be rasterized and concatenated with any standard toolchain —
for example, `ImageMagick` to convert PDFs to PNGs and `ffmpeg` to stitch them into an
MP4. Baking predictions and CI overlays directly onto a video requires a user-supplied
pipeline on top of the `Environment.render()` output.

### Custom static geometry

Static obstacles are stored in `assets/geometries/<dataset>.json` as a list of
axis-aligned regions, loaded into a `Geometry` object (`canvas/envs/env_utils.py`) that
supports signed-distance queries. To edit, modify the JSON and re-run the driver — no
code changes are needed.

---

## Project Structure

```
CANVAS/
|-- README.md
|-- LICENSE
|-- environment_full.yml              # conda environment spec (name: conformal)
|-- appendix_repository_architecture.txt
|
|-- canvas/                           # core package
|   |-- __init__.py                   # intentionally empty — import from submodules
|   |-- datasets/
|   |   |-- dataset.py                # Dataset (NaN interpolation, scene/future getters)
|   |   |-- dataset_loader.py         # DatasetSpec, static_regions, background extent
|   |   `-- __init__.py               # RegisteredDatasets, PATHS_REGISTERED
|   |-- envs/
|   |   |-- env.py                    # Environment (unicycle ego + replay of dataset)
|   |   |-- env_utils.py              # Geometry, Rectangle, signed-distance queries
|   |   `-- visualization_utils.py    # homography helpers
|   |-- predictors/
|   |   |-- predictor.py              # Predictors factory (string-dispatched)
|   |   |-- wrapper_predictor.py      # BasePredictors abstract interface
|   |   |-- linear_predictor.py       # Linear constant-velocity baseline
|   |   |-- gp_predictor.py           # Gaussian-process forecaster
|   |   |-- koopcast_predictor.py     # KoopCast wrapper
|   |   |-- trajectron_predictor.py   # Trajectron++ wrapper
|   |   |-- eigen/                    # EigenTrajectory code
|   |   |-- trajectron/               # Trajectron++ code
|   |   |-- koopcast/                 # KoopCast code
|   |   |-- Social_STGCNN/            # Social-STGCNN code
|   |   `-- SocialVAE/                # Social-VAE code
|   |-- controllers/
|   |   |-- controller.py             # controllers factory (mpc, mppi)
|   |   |-- mpc.py                    # BaseMPC (grid search + optional NLP refine)
|   |   |-- mppi.py                   # KernelMPPI (RBF-kernel MPPI)
|   |   |-- optim_solver.py           # IPOPT-based NLP used by BaseMPC
|   |   |-- grid_solver.py            # grid rollout utilities
|   |   |-- sampling_based_mpc.py     # sampling MPC (legacy)
|   |   |-- conformal_controller.py   # conformal MPC (legacy)
|   |   |-- ecp_mpc.py                # egocentric conformal MPC (legacy)
|   |   `-- pytorch_mppi/             # submodule: forked pytorch-mppi
|   |-- conformal_predictors/
|   |   |-- aci.py                    # DelayedACI (Dixit et al. 2023)
|   |   |-- lqt.py                    # LinearQuantileTracker (Areces et al. 2025)
|   |   |-- scores.py                 # PD / AD / PR score functions
|   |   |-- hindsight_scores.py       # hindsight analogues
|   |   `-- cp_utils.py               # shared helpers (quantile ops, buffers)
|   |-- competency_indices/
|   |   `-- core.py                   # MovingAverage / Conformalized / Hindsight CI
|   |-- detection/                    # pedestrian-detection helpers (peripheral)
|   `-- assets/                       # package-internal sprites / overlay helpers
|
|-- examples/                         # experiment drivers
|   |-- simulation_mpc_lqt.py         # BaseMPC  + LQT + PD/AD/PR (canonical)
|   `-- simulation_mppi_lqt.py        # VanillaMPPI + LQT + PD/AD/PR
|
|-- tests/                            # stand-alone sanity scripts (no pytest)
|   |-- test_registered_datasets.py
|   |-- test_env_visualization.py
|   |-- test_truncated_history.py
|   `-- test_mppi.py
|
|-- tools/
|   `-- make_thumbnail.py             # multi-scene thumbnail generator
|
`-- assets/                           # downloaded + committed assets
    |-- download_assets.sh            # gdown helper for datasets + weights
    |-- datasets/                     # .npy trajectory tensors (downloaded)
    |-- models/                       # pretrained model weights (downloaded)
    |-- videos/                       # raw scene videos (downloaded)
    |-- frames/<scene>/<t>.png        # per-frame background images (user-supplied)
    |-- geometries/<scene>.json       # static obstacle regions
    |-- homographies/<scene>.txt      # world <-> pixel homographies
    |-- snu-asri.png                  # static background for SNU-ASRI scenes
    |-- robot.png                     # ego-robot sprite
    `-- video_parser.py               # utility to parse MP4 -> per-frame PNG
```

### Pipeline at a glance

Per-frame loop (in `examples/simulation_mpc_lqt.py`):

1. `obs = env._get_obs()` yields ego pose + pedestrian histories.
2. `predictor(obs['non-ego'])` and the **baseline** predictor both produce
   `Dict[int, ndarray(N, 2)]`. A linear baseline is always run alongside for ratio-based
   scoring.
3. `indices.update(obs)` buffers observations; `forward()` computes scores, updates the
   conformal module, and emits `CI_t = 1 / (1 + s_t)`.
4. `controller(obs, prediction)` returns `(u, info)` where
   `info = {X, U, cost_to_go}`.
5. `env.step(u)` advances the unicycle model one step.

### Data contract

Every component communicates through a single type:

```
Dict[int, np.ndarray]     # keyed by pedestrian id
```

- Observed histories: shape `(H, 2)`.
- Predicted or ground-truth futures: shape `(N, 2)`.

Do **not** introduce adapter layers — uphold the dict contract.

---

## Extending CANVAS

All extension points are documented in `appendix_repository_architecture.txt` with
code templates. The most common ones:

### Add a predictor

Subclass `BasePredictors` and implement `__call__`:

```python
from canvas.predictors.wrapper_predictor import BasePredictors

class MyPredictor(BasePredictors):
    def __init__(self, prediction_len=12, history_len=8, dt=0.1, device='cpu', **kw):
        super().__init__(prediction_len, history_len, dt, device)
        # load weights / build model

    def __call__(self, tracking_result):
        out = {}
        for obj_id, h in tracking_result.items():      # h : (H, 2)
            out[obj_id] = self._forecast(h)            # (N, 2)
        return out
```

Then register the name in `canvas/predictors/predictor.py::Predictors.__init__`,
including per-dataset model paths if applicable.

### Add a score function

Subclass `ScoreFunction` in `canvas/conformal_predictors/scores.py` and return an
`(E_model, E_baseline)` pair; the base class handles ratio normalization and clipping.
Register it with `CompetencyIndex.register(score, cp, name)`.

### Add a dataset

1. Drop a `.npy` of shape `(T, n_agents, 2)` under `assets/datasets/...`.
2. Register it in `canvas/datasets/__init__.py::PATHS_REGISTERED`
   (`dt=0.4` for ETH/UCY style, `dt=0.1` for SNU-ASRI style).
3. Add a `DatasetSpec` entry in `canvas/datasets/dataset_loader.py`
   (background image, extent, `static_regions`).
4. Create `assets/geometries/<name>.json` and `assets/homographies/<name>.txt`.
5. Add a per-dataset model-path branch in each predictor you want to use.

---

## Troubleshooting

**Qt platform plugin errors on a headless machine:**
```bash
export QT_QPA_PLATFORM=offscreen
```

**`ModuleNotFoundError: canvas`:**
Run scripts from the repository root (`python examples/simulation_mpc_lqt.py`, not
from inside `examples/`). The scripts rely on the CWD being at the repo root.

**`ImportError: cannot import name 'Predictors' from 'canvas'`:**
`canvas/__init__.py` is intentionally empty. Import from submodules instead:
```python
from canvas.predictors import Predictors
from canvas.datasets import RegisteredDatasets, get_dataset_spec
from canvas.envs.env import Environment
```

**Pretrained weights not found:**
The predictor factory reads weight paths relative to the CWD. Verify that you are
running from the repo root and that `bash assets/download_assets.sh` completed
without errors. Weights that are not covered by the download script (e.g.
Social-STGCNN, Social-VAE, KoopCast, EigenTrajectory) must be placed at the literal
paths the factory expects; see each branch in `canvas/predictors/predictor.py`.

**Missing per-frame images:**
`Environment.render()` expects `assets/frames/<dataset>/<t>.png`. Parse these from the
downloaded videos using `assets/video_parser.py`, or set `--visualize` off.

**`DelayedACI.update` prints every frame:**
`_alpha_t` is emitted as a debug trace. Strip the print in `aci.py` if you need quiet
logs.

---

## License

See `LICENSE` for terms.

---

[trajectronpp-link]: https://github.com/StanfordASL/Trajectron-plus-plus
[eigentraj-link]: https://github.com/InhwanBae/EigenTrajectory
[socialstgcnn-link]: https://github.com/abduallahmohamed/Social-STGCNN
[socialvae-link]: https://github.com/xupei0610/SocialVAE
[koopcast-link]: https://github.com/Koopcast/Koopcast
[pytorch-mppi-link]: https://github.com/UM-ARM-Lab/pytorch_mppi
