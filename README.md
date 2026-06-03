# NDProp: Neural Decision-Propagation for Answer Set Programming

This repository contains the public release code for the IJCAI-ECAI 2026 paper
**Neural Decision-Propagation for Answer Set Programming**.

The repo covers the following experiments from the paper:

1. Random ASP solving on N2L and 3-LP programs.
2. Neuro-symbolic MNIST reasoning on `addition`, `addition2`, `add2x2`,
   `apply2x2`, `member3`, and `member5`.
3. Visual 4x4 Sudoku with incomplete or imperfect digit predictions.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install `clingo` on `PATH`, or pass `--clingo-path` to scripts that expose it.

PyTorch Geometric wheels depend on your PyTorch and CUDA versions. If
`pip install -r requirements.txt` cannot resolve `torch-geometric` or
`torch-scatter`, install the matching wheels from the official PyTorch
Geometric instructions.

## Data Preparation

Generated and downloaded data live under `data/` by default. Run outputs live
under `runs/`, `runs_nesy/`, and `baselines/`. These paths are ignored by git.

### MNIST Reasoning Indexes

The tracked files in `indexes/` provide reproducible splits for the MNIST
reasoning tasks. Regenerate them with:

```bash
python src/generate_indexes.py --mnist --indexes-root indexes --mnist-root data
```

### Visual Sudoku Tensors

Generate the MNIST Sudoku tensors used by the visual Sudoku experiment:

```bash
python src/generate_mnist_sudoku.py \
  --output-root data/mnist_sudoku \
  --mnist-root data \
  --train-size 10000 \
  --test-size 1000 \
  --clue-fraction 0.5
```

The generator writes `train_features_img.pt`, `train_labels.pt`,
`test_features_img.pt`, and `test_labels.pt`. Blank cells are rendered with
MNIST zero images; nonblank cells use MNIST digits `1` through `4`.

## Experiments

### 1. Random ASP Solving

Generate benchmark splits:

```bash
python src/generate_random_asp.py --family n2l
python src/generate_random_asp.py --family 3lp
```

Train NDProp:

```bash
scripts/run_random_asp_experiments.sh --family n2l --device cuda:0
scripts/run_random_asp_experiments.sh --family 3lp --device cuda:0
```

Run the RDProp restart baselines used for Table 2:

```bash
scripts/run_random_asp_rdprop_baseline.sh --family n2l --restarts 1 10 100
scripts/run_random_asp_rdprop_baseline.sh --family 3lp --restarts 1 10 100
```

Analyze a trained run by difficulty split:

```bash
scripts/analyze_random_asp_results.sh \
  --family n2l \
  --run-dir runs/random_asp/n2l/<run_dir>
```

### 2. Neuro-Symbolic MNIST Reasoning

Run one task:

```bash
scripts/run_mnist_task.sh \
  --dataset addition \
  --device cpu \
  --epochs 1 \
  --train-limit 32 \
  --eval-limit 16
```

Run the six-tasks:

```bash
scripts/run_mnist_reasoning_experiments.sh
```

Summarize the results for the six-tasks:

```bash
scripts/summarize_mnist_reasoning_results.sh
```

### 3. Visual Sudoku

Run the visual Sudoku task over digit-label fractions:

```bash
scripts/run_visual_sudoku_experiments.sh
```

## Analysis Utilities

Run the ASP analysis cases:

```bash
scripts/analyze_asp_cases.sh --cases-dir analysis_cases
```

Measure random ASP iteration scaling, if needed:

```bash
scripts/analyze_random_asp_iteration_scaling.sh --family n2l
```

## Repository Layout

```text
analysis_cases/        Curated ASP programs for analysis and baselines.
indexes/               Tracked MNIST reasoning split files.
scripts/               Shell entrypoints for paper experiments.
src/                   Python source code and data generators.
src/examples/          Paper task implementations.
src/fuzzy_tp/          ASP parsing, graph construction, and propagation code.
src/nesy/              Shared neuro-symbolic runner components.
```
