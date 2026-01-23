# NDProp: Neural Decision–Propagation for Answer Set Programming (IJCAI 2026 Submission)

This repository contains the implementation accompanying our IJCAI 2026 submission
"Neural Decision–Propagation for Answer Set Programming."

## Repository Structure

- `src/`: Core implementation, models, and example tasks.
- `scripts/`: Experiment and analysis entry points (shell and Python).
- `data/`: Datasets
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