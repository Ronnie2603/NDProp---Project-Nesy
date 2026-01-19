# NDProp: Neural Decision–Propagation for Answer Set Programming (IJCAI 2026 Submission)

This repository contains the implementation accompanying our IJCAI 2026 submission
"Neural Decision–Propagation for Answer Set Programming." The submission is blind;
please do not add identifying information in this README.

## Overview

The paper introduces Decision–Propagation (DProp), a novel computational
characterization of stable model semantics for normal logic programs. DProp
alternates between falsity decisions and truth propagation, and we prove that
successful DProp computations correspond exactly to stable models. Unlike
traditional ASP solvers, stability is enforced as an invariant of the
decision–propagation dynamics themselves, rather than through loop formulas or
external solver mechanisms.

Building on this foundation, we propose Neural Decision–Propagation (NDProp), a
fully differentiable extension of DProp. NDProp replaces discrete decisions
with neural decision modules and uses fuzzy logic operators for propagation,
while provably recovering classical stable model semantics in the binary limit.
This enables end-to-end learning of reasoning dynamics without invoking an
external ASP solver.

We evaluate NDProp on:
- Random ASP solving, to assess its ability to learn solving dynamics.
- Neuro-symbolic MNIST benchmarks, to compare against ASP-based baselines.
- Fuzzy reasoning tasks such as visual Sudoku.

Results show that NDProp improves accuracy and inference speed while preserving
clear semantic ties to stable model reasoning.

## Repository Structure

- `src/`: Core implementation, models, and example tasks.
- `scripts/`: Experiment and analysis entry points (shell and Python).
- `data/`: Datasets and cached assets (not all data are included by default).
- `runs/`, `runs_nesy/`: Experiment outputs and logs.
- `weights/`: Model checkpoints.

## Setup

Use a Python environment with common ML dependencies (PyTorch, torchvision,
NumPy, etc.). Install any missing dependencies according to your environment
preferences (e.g., `pip` or `conda`). Some scripts expect data in `data/`.

## Running Experiments

The `scripts/` directory contains the primary entry points. Typical workflows:

- Random ASP experiments: `scripts/experiments.sh`
- MNIST Sudoku: `scripts/mnist_sudoku_experiment.sh`
- Baselines: `scripts/experiment_dprop_baseline.sh`,
  `scripts/experiment_random_baseline.sh`
- Analysis: `scripts/analyze.sh` and other `scripts/analyze_*.sh`

Adjust paths or environment variables as needed for your local setup.

## Notes

- The submission is blind; please keep this repository anonymized.
- Output artifacts will be written into `runs/` and `runs_nesy/` by default.

## Citation

A citation entry will be added after the review process.
