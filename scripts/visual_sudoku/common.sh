#!/usr/bin/env bash

# Shared configuration for the Visual Sudoku commands. This file is sourced by
# the runnable scripts and is not a standalone command-line entrypoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

DEVICE="${DEVICE:-}"
read -r -a CUDA_DEVICES <<< "${CUDA_DEVICES:-0 1 2 3}"
read -r -a SEEDS <<< "${SEEDS:-1 2 3}"
read -r -a PCTS <<< "${PCTS:-1.0 0.5 0.1 0.05 0.01}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1000}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-3}"
ITERS="${ITERS:-3}"
HIDDEN_DIM="${HIDDEN_DIM:-8}"
LOGICAL_DIM="${LOGICAL_DIM:-2}"
T_NORM="${T_NORM:-godel}"
DECISION_NORMALIZATION="${DECISION_NORMALIZATION:-program}"
BATCH_SIZE="${BATCH_SIZE:-64}"

MNIST_ROOT="${MNIST_ROOT:-${PROJECT_ROOT}/data}"
SUDOKU_ROOT="${SUDOKU_ROOT:-${PROJECT_ROOT}/data/mnist_sudoku}"
PROGRAM_PATH="${PROGRAM_PATH:-${PROJECT_ROOT}/src/examples/mnist_sudoku/programs/mnist_sudoku_4x4.aspif}"
IMAGE_DATASET="${IMAGE_DATASET:-mnist}"
BASELINE_MODE="${BASELINE_MODE:-both}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/runs_nesy/visual_sudoku}"
WEIGHT_ROOT="${WEIGHT_ROOT:-${PROJECT_ROOT}/weights}"

pct_tag() {
  local pct="$1"
  printf '%s\n' "${pct//./p}"
}

weight_path_for() {
  local pct="$1"
  local seed="$2"
  printf '%s/mnist_sudoku_digit_%s_seed_%s.pt\n' "${WEIGHT_ROOT}" "$(pct_tag "${pct}")" "${seed}"
}

device_arg() {
  if [[ -n "${DEVICE:-}" ]]; then
    printf '%s\n' "${DEVICE}"
  else
    printf 'cuda:%s\n' "${CUDA_ID:-0}"
  fi
}
