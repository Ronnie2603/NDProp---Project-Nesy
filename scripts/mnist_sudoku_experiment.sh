#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

DEVICE="${DEVICE:-cuda:0}"
CUDA_DEVICES=(${CUDA_DEVICES:-0 1 2 3})
SEEDS=(${SEEDS:-1 2 3 })
PCTS=(${PCTS:-1.0 0.5 0.1 0.05 0.01})
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1000}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-3}"
ITERS="${ITERS:-3}"
HIDDEN_DIM="${HIDDEN_DIM:-8}"
LOGICAL_DIM="${LOGICAL_DIM:-2}"
T_NORM="${T_NORM:-godel}"
MNIST_ROOT="${MNIST_ROOT:-${PROJECT_ROOT}/data}"
SUDOKU_ROOT="${SUDOKU_ROOT:-${PROJECT_ROOT}/../Sudoku/sudoku4}"
PROGRAM_PATH="${PROGRAM_PATH:-${PROJECT_ROOT}/src/examples/mnist_sudoku/programs/mnist_sudoku_4x4.aspif}"
IMAGE_DATASET="${IMAGE_DATASET:-mnist}"
BASELINE_MODE="${BASELINE_MODE:-both}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/runs_nesy/mnist_sudoku_exp2}"
PARALLEL="${PARALLEL:-1}"

run_all() {
  local pct="$1"
  local seed="$2"
  local cuda_id="$3"
  local pct_tag="${pct//./p}"
  local weight_path="${PROJECT_ROOT}/weights/mnist_sudoku_digit_${pct_tag}_seed_${seed}.pt"
  local device="cuda:${cuda_id}"

  echo "=== Pretrain digit net: seed=${seed} pct=${pct} ==="
  "${PYTHON_BIN}" "${PROJECT_ROOT}/src/nesy.py" \
    --dataset mnist_sudoku \
    --device "${device}" \
    --seed "${seed}" \
    --epochs 0 \
    --lr "${LR}" \
    --iters "${ITERS}" \
    --logical-dimension "${LOGICAL_DIM}" \
    --tnorm "${T_NORM}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --program-path "${PROGRAM_PATH}" \
    --use-init-hidden \
    --mnist-sudoku-digit-pretrain-epochs "${PRETRAIN_EPOCHS}" \
    --mnist-sudoku-digit-pretrain-fraction "${pct}" \
    --mnist-sudoku-digit-pretrain-path "${weight_path}" \
    --mnist-sudoku-image-dataset "${IMAGE_DATASET}" \
    --mnist-root "${MNIST_ROOT}" \
    --mnist-sudoku-root "${SUDOKU_ROOT}" \
    --log-root "${LOG_ROOT}/pretrain_pct_${pct_tag}_seed_${seed}"\
    --batch-size 64

  echo "=== Train NDProp: seed=${seed} pct=${pct} ==="
  "${PYTHON_BIN}" "${PROJECT_ROOT}/src/nesy.py" \
    --dataset mnist_sudoku \
    --device "${device}" \
    --seed "${seed}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --iters "${ITERS}" \
    --logical-dimension "${LOGICAL_DIM}" \
    --tnorm "${T_NORM}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --program-path "${PROGRAM_PATH}" \
    --use-init-hidden \
    --mnist-sudoku-digit-pretrain-epochs 0 \
    --mnist-sudoku-digit-pretrain-path "${weight_path}" \
    --mnist-sudoku-image-dataset "${IMAGE_DATASET}" \
    --mnist-root "${MNIST_ROOT}" \
    --mnist-sudoku-root "${SUDOKU_ROOT}" \
    --log-root "${LOG_ROOT}/train_pct_${pct_tag}_seed_${seed}" \
    --batch-size 64

  echo "=== Baselines: seed=${seed} pct=${pct} ==="
  "${PYTHON_BIN}" "${PROJECT_ROOT}/src/nesy.py" \
    --dataset mnist_sudoku \
    --device "${device}" \
    --seed "${seed}" \
    --epochs 0 \
    --lr "${LR}" \
    --iters "${ITERS}" \
    --logical-dimension "${LOGICAL_DIM}" \
    --tnorm "${T_NORM}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --program-path "${PROGRAM_PATH}" \
    --use-init-hidden \
    --mnist-sudoku-run-baselines \
    --mnist-sudoku-baseline-mode "${BASELINE_MODE}" \
    --mnist-sudoku-digit-pretrain-epochs 0 \
    --mnist-sudoku-digit-pretrain-path "${weight_path}" \
    --mnist-sudoku-image-dataset "${IMAGE_DATASET}" \
    --mnist-root "${MNIST_ROOT}" \
    --mnist-sudoku-root "${SUDOKU_ROOT}" \
    --log-root "${LOG_ROOT}/baseline_pct_${pct_tag}_seed_${seed}" \
    --batch-size 64
}

if [[ "${PARALLEL}" -le 1 ]]; then
  for pct in "${PCTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_all "${pct}" "${seed}" "${CUDA_DEVICES[0]}"
    done
  done
  exit 0
fi

job_count=0
job_index=0
for pct in "${PCTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    cuda_id="${CUDA_DEVICES[$((job_index % ${#CUDA_DEVICES[@]}))]}"
    run_all "${pct}" "${seed}" "${cuda_id}" &
    job_count=$((job_count + 1))
    job_index=$((job_index + 1))
    if [[ "${job_count}" -ge "${PARALLEL}" ]]; then
      wait -n
      job_count=$((job_count - 1))
    fi
  done
done
wait
