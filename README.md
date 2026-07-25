# NDProp: Neural Decision-Propagation for Answer Set Programming

This repository contains the implementation for the paper **Neural
Decision-Propagation for Answer Set Programming**. It includes the source code,
ASP programs, and shell scripts used to run the paper experiments.

The release covers three experiment families:

- Random ASP solving on Negative Two Literal (N2L) and 3-LP programs
- Neuro-symbolic tasks including MNIST Addition (one and two digits), Add 2x2,
  Apply 2x2, and Membership (three and five numbers)
- Visual 4x4 Sudoku with learned digit recognition CNNs

## Setup

Python 3.12 is the tested interpreter version.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Additionally, you have to install `pytorch`, as well as the `torch-geometric` and `torch-scatter` packages that correspond to your system and CUDA version. See the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) and the [PyTorch GeIometric installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for guidance.

The `clingo` Python package is sufficient for the default setup. You can also
install a standalone `clingo` executable on `PATH`, or pass `--clingo-path` to
scripts that expose that option.

## Data and Assets

Tracked release inputs:

- `analysis_cases/`: small ASP test programs for analysis.
- `indexes/`: reproducible splits for neuro-symbolic tasks.
- `assets/operator_images/hasy/`: small operator-image asset for `apply2x2`;
  see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and licensing.
- `src/examples/**/programs/`: ASP and ASPIF programs for each task.

Regeneration commands:

```bash
python src/generate_indexes.py --mnist --indexes-root indexes --mnist-root data
python src/generate_mnist_sudoku.py --out-dir data/mnist_sudoku --mnist-root data --train-count 10000 --test-count 1000
```

## Experiment Scripts

Each experiment family has one public entrypoint:

- `scripts/random_asp/main.sh`
- `scripts/neuro_symbolic_tasks/main.sh`
- `scripts/visual_sudoku/main.sh`

Use the main scripts rather than invoking their helpers directly. Each main
script lists its commands, and every command has focused help:

```bash
bash scripts/random_asp/main.sh --help
bash scripts/random_asp/main.sh train --help
bash scripts/neuro_symbolic_tasks/main.sh task --help
bash scripts/visual_sudoku/main.sh generate --help
```

The top-level help shows the normal experiment controls. Command-specific help
also documents path overrides, model settings, and multi-device controls.
Helper scripts remain available for automation, but they are implementation
details of these entrypoints.

### Random ASP

Generate benchmark splits:

```bash
bash scripts/random_asp/main.sh generate --family n2l
bash scripts/random_asp/main.sh generate --family 3lp
```

This creates repo-local `benchmarks/n2l` or `benchmarks/3lp` directories with
`train_easy|medium|hard`, `val_easy|medium|hard`, and `test_easy|medium|hard`
splits.

| Difficulty | Atom range | Training | Validation | Test |
| --- | --- | ---: | ---: | ---: |
| Easy | 5-10 | 1000 | 100 | 100 |
| Medium | 11-50 | 1000 | 100 | 100 |
| Hard | 51-100 | 1000 | 100 | 100 |

Existing complete splits are reused.

Train NDProp:

```bash
bash scripts/random_asp/main.sh train --family n2l --difficulty medium --device cuda:0
bash scripts/random_asp/main.sh train --family 3lp --difficulty medium --device cuda:0
```

The training defaults follow the paper setup: 1000 epochs, 10 training
iterations, hidden dimension 32, logical dimension 32, learning rate `1e-3`,
Godel t-norm, and validation on `val_<difficulty>`.

Run the full random ASP pipeline for both families and difficulties:

```bash
bash scripts/random_asp/main.sh all --device cuda:0
```

The `all` command verifies/generates repo-local benchmarks, trains NDProp, runs
difficulty and iteration analysis, runs the random and RDProp baselines, and
writes the grouped method-comparison plots. By default it runs both N2L and
3-LP, training NDProp on both `easy` and `medium` training splits.

Run baselines and analysis:

```bash
bash scripts/random_asp/main.sh baseline --family n2l --restarts 1 10 100
bash scripts/random_asp/main.sh analyze --family n2l --run-dir runs/random_asp/n2l/medium/<run_dir>
bash scripts/random_asp/main.sh iteration --family n2l --run-dir runs/random_asp/n2l/medium/<run_dir>
```

#### Analyze the bundled ASP cases with an existing random ASP run

```bash
bash scripts/random_asp/main.sh analysis-cases --run-dir <run_dir> --device cuda:0
```

### Neuro-Symbolic Tasks

Run one small task:

```bash
bash scripts/neuro_symbolic_tasks/main.sh task --dataset addition --device cuda:0 --epochs 1 --train-limit 32 --eval-limit 16
```

Run and summarize all six neuro-symbolic tasks:

```bash
bash scripts/neuro_symbolic_tasks/main.sh run
bash scripts/neuro_symbolic_tasks/main.sh summarize
```

`scripts/neuro_symbolic_tasks/main.sh all` runs the six tasks and then summarizes
them. `summarize` rebuilds `results.csv` from the newest `run_*` directory in
each task/seed/repeat directory, so stale rows from older runs are excluded.
Task-specific dataset files, ASP programs, and model defaults are selected
automatically. Use `task --help` only when an experiment needs an explicit
override.

### Visual Sudoku

Run the complete pipeline with:

```bash
bash scripts/visual_sudoku/main.sh
```

The command checks `data/mnist_sudoku` before building the model. If any
required tensor is missing, it downloads the selected image dataset as needed
and generates the complete 4x4 Visual Sudoku dataset automatically. The
individual `pretrain`, `train`, and `baseline` commands perform the same check.
Use the explicit `generate` command when you want custom generation settings
or forced regeneration.

The default run is expensive: it trains five label fractions over three seeds.
Use flags to select a smaller experiment grid or assign a device. For example:

```bash
bash scripts/visual_sudoku/main.sh all --pcts 1.0 --seeds 1 --device cuda:0
```

This experiment writes digit checkpoints under `weights/` and run outputs under
`runs_nesy/visual_sudoku`. The full pipeline finishes by aggregating NDProp,
Clingo, and neural-baseline cell/grid accuracy across seeds, with shaded ±1
sample-standard-deviation bands. Rebuild only the analysis artifacts with:

```bash
bash scripts/visual_sudoku/main.sh analysis
```

## Repository Layout

```text
analysis_cases/                Curated ASP programs for analysis.
assets/                        Small bundled assets needed by released tasks.
indexes/                       Tracked experiment split files.
scripts/random_asp/            Random ASP main script and helpers.
scripts/neuro_symbolic_tasks/  Neuro-symbolic task entrypoint, runner, and defaults.
scripts/visual_sudoku/         Visual Sudoku main script.
src/                           Python source code and data generators.
src/examples/                  Paper task implementations.
src/ndprop/                    ASP parsing, graph construction, and NDProp model.
src/nesy/                      Shared neuro-symbolic runner components.
src/experiment_utils/          CSV aggregation and plotting helpers.
```
