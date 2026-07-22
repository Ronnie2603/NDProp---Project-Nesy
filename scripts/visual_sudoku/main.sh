#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [command] [options]

Commands:
  generate  Generate the 4x4 visual Sudoku tensors.
  pretrain  Pretrain the digit net for one PCT/SEED pair.
  train     Train NDProp for one PCT/SEED pair.
  baseline  Run visual Sudoku baselines for one PCT/SEED pair.
  analysis  Aggregate methods across seeds and plot mean ± standard deviation.
  all       Generate data, run experiments over PCTS x SEEDS, then analyze them.

Options for the all command:
Experiment grid:
      --pcts VALUE ...      Digit-label fractions (default: ${PCTS[*]})
      --seeds N ...         Random seeds (default: ${SEEDS[*]})
      --device DEVICE       Use one fixed device for every job
      --cuda-devices ID ... CUDA device IDs used for round-robin assignment
                             when --device is not supplied (default: ${CUDA_DEVICES[*]})
      --parallel N          Number of concurrent jobs (default: ${PARALLEL:-1})

Paths:
      --mnist-root PATH     Image-dataset root (default: ${MNIST_ROOT})
      --sudoku-root PATH    Generated tensor root (default: ${SUDOKU_ROOT})
      --log-root PATH       Run output root (default: ${LOG_ROOT})
      --weight-root PATH    Digit-checkpoint root (default: ${WEIGHT_ROOT})
      --program-path PATH   ASP/ASPIF program (default: ${PROGRAM_PATH})

Advanced experiment settings:
      --pretrain-epochs N   Digit-pretraining epochs (default: ${PRETRAIN_EPOCHS})
      --epochs N            NDProp training epochs (default: ${EPOCHS})
      --lr VALUE            Learning rate (default: ${LR})
      --iters N             NDProp iterations (default: ${ITERS})
      --hidden-dim N        Decision-controller hidden size (default: ${HIDDEN_DIM})
      --logical-dim N       Logical feature channels (default: ${LOGICAL_DIM})
      --tnorm NAME          product|lukasiewicz|godel (default: ${T_NORM})
      --decision-normalization NAME
                            none|program (default: ${DECISION_NORMALIZATION})
      --batch-size N        Batch size (default: ${BATCH_SIZE})
      --image-dataset NAME  mnist|fashion_mnist|cifar10 (default: ${IMAGE_DATASET})
      --baseline-mode MODE  both|clingo|nn (default: ${BASELINE_MODE})

Pipeline control:
      --skip-generate   Do not run the generation check
      --force-generate  Regenerate tensors even if they already exist
  -h, --help            Show this message and exit

The default command is "all". Use '<command> --help' for command-specific help.
EOF
}

run_one() {
  local pct="$1"
  local seed="$2"
  local cuda_id="$3"
  local job_device="${DEVICE:-cuda:${cuda_id}}"
  bash "${SCRIPT_DIR}/pretrain.sh" --pct "${pct}" --seed "${seed}" --device "${job_device}"
  bash "${SCRIPT_DIR}/train.sh" --pct "${pct}" --seed "${seed}" --device "${job_device}"
  bash "${SCRIPT_DIR}/baseline.sh" --pct "${pct}" --seed "${seed}" --device "${job_device}"
}

run_all() {
  local skip_generate=false
  local force_generate=false
  local parallel="${PARALLEL:-1}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --pcts)
        shift
        PCTS=()
        while [[ $# -gt 0 && "$1" != --* ]]; do PCTS+=("$1"); shift; done
        ;;
      --seeds)
        shift
        SEEDS=()
        while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done
        ;;
      --cuda-devices)
        shift
        CUDA_DEVICES=()
        while [[ $# -gt 0 && "$1" != --* ]]; do CUDA_DEVICES+=("$1"); shift; done
        ;;
      --skip-generate) skip_generate=true; shift ;;
      --force-generate) force_generate=true; shift ;;
      --parallel) parallel="$2"; shift 2 ;;
      --device) DEVICE="$2"; shift 2 ;;
      --mnist-root) MNIST_ROOT="$2"; shift 2 ;;
      --sudoku-root) SUDOKU_ROOT="$2"; shift 2 ;;
      --log-root) LOG_ROOT="$2"; shift 2 ;;
      --weight-root) WEIGHT_ROOT="$2"; shift 2 ;;
      --program-path) PROGRAM_PATH="$2"; shift 2 ;;
      --pretrain-epochs) PRETRAIN_EPOCHS="$2"; shift 2 ;;
      --epochs) EPOCHS="$2"; shift 2 ;;
      --lr) LR="$2"; shift 2 ;;
      --iters) ITERS="$2"; shift 2 ;;
      --hidden-dim) HIDDEN_DIM="$2"; shift 2 ;;
      --logical-dim) LOGICAL_DIM="$2"; shift 2 ;;
      --tnorm) T_NORM="$2"; shift 2 ;;
      --decision-normalization) DECISION_NORMALIZATION="$2"; shift 2 ;;
      --batch-size) BATCH_SIZE="$2"; shift 2 ;;
      --image-dataset) IMAGE_DATASET="$2"; shift 2 ;;
      --baseline-mode) BASELINE_MODE="$2"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown all option: $1" >&2; usage; exit 1 ;;
    esac
  done

  if [[ ${#PCTS[@]} -eq 0 || ${#SEEDS[@]} -eq 0 ]]; then
    echo "--pcts and --seeds must each contain at least one value." >&2
    exit 1
  fi
  if [[ -z "${DEVICE}" && ${#CUDA_DEVICES[@]} -eq 0 ]]; then
    echo "--cuda-devices must contain at least one ID when --device is not set." >&2
    exit 1
  fi

  export DEVICE PRETRAIN_EPOCHS EPOCHS LR ITERS HIDDEN_DIM LOGICAL_DIM
  export T_NORM DECISION_NORMALIZATION BATCH_SIZE MNIST_ROOT SUDOKU_ROOT
  export PROGRAM_PATH IMAGE_DATASET BASELINE_MODE LOG_ROOT WEIGHT_ROOT
  if [[ "${skip_generate}" != true ]]; then
    if [[ "${force_generate}" == true ]]; then
      bash "${SCRIPT_DIR}/generate.sh" --force
    else
      bash "${SCRIPT_DIR}/generate.sh"
    fi
  fi

  if [[ "${parallel}" -le 1 ]]; then
    for pct in "${PCTS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        run_one "${pct}" "${seed}" "${CUDA_DEVICES[0]}"
      done
    done
    bash "${SCRIPT_DIR}/analysis.sh" \
      --log-root "${LOG_ROOT}" --pcts "${PCTS[@]}" --seeds "${SEEDS[@]}"
    return
  fi

  local job_count=0
  local job_index=0
  for pct in "${PCTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      local cuda_id="${CUDA_DEVICES[$((job_index % ${#CUDA_DEVICES[@]}))]}"
      run_one "${pct}" "${seed}" "${cuda_id}" &
      job_count=$((job_count + 1))
      job_index=$((job_index + 1))
      if [[ "${job_count}" -ge "${parallel}" ]]; then
        wait -n
        job_count=$((job_count - 1))
      fi
    done
  done
  wait
  bash "${SCRIPT_DIR}/analysis.sh" \
    --log-root "${LOG_ROOT}" --pcts "${PCTS[@]}" --seeds "${SEEDS[@]}"
}

cmd="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi

case "${cmd}" in
  generate) exec bash "${SCRIPT_DIR}/generate.sh" "$@" ;;
  pretrain) exec bash "${SCRIPT_DIR}/pretrain.sh" "$@" ;;
  train) exec bash "${SCRIPT_DIR}/train.sh" "$@" ;;
  baseline) exec bash "${SCRIPT_DIR}/baseline.sh" "$@" ;;
  analysis) exec bash "${SCRIPT_DIR}/analysis.sh" "$@" ;;
  all) run_all "$@" ;;
  -h|--help|help) usage ;;
  *) echo "Unknown command: ${cmd}" >&2; usage; exit 1 ;;
esac
