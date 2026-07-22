#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

PCT="${PCT:-1.0}"
SEED="${SEED:-1}"
WEIGHT_PATH="${WEIGHT_PATH:-}"
PASSTHROUGH=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- nesy-options]

Pretrains the Visual Sudoku digit classifier for one PCT/SEED pair.

Options:
      --pct VALUE         Digit-label fraction (default: ${PCT})
      --seed N            Random seed (default: ${SEED})
      --device DEVICE     Compute device (default: cuda:0)
      --weight-path PATH  Checkpoint path derived from pct/seed by default
      --pretrain-epochs N Digit-pretraining epochs (default: ${PRETRAIN_EPOCHS})
  -h, --help              Show this message and exit

Arguments after '--', as well as unrecognized backend options, are appended to
src/nesy.py. They can override advanced model settings such as epochs, learning
rate, iterations, dimensions, paths, and batch size.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pct) PCT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --weight-path) WEIGHT_PATH="$2"; shift 2 ;;
    --pretrain-epochs) PRETRAIN_EPOCHS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; PASSTHROUGH+=("$@"); break ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

PCT_TAG="$(pct_tag "${PCT}")"
WEIGHT_PATH="${WEIGHT_PATH:-$(weight_path_for "${PCT}" "${SEED}")}"
DEVICE_ARG="$(device_arg)"

mkdir -p "$(dirname "${WEIGHT_PATH}")"

echo "=== Visual Sudoku digit pretrain: seed=${SEED} pct=${PCT} device=${DEVICE_ARG} ==="
"${PYTHON_BIN}" "${PROJECT_ROOT}/src/nesy.py" \
  --dataset mnist_sudoku \
  --device "${DEVICE_ARG}" \
  --seed "${SEED}" \
  --epochs 0 \
  --lr "${LR}" \
  --iters "${ITERS}" \
  --logical-dimension "${LOGICAL_DIM}" \
  --tnorm "${T_NORM}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --decision-normalization "${DECISION_NORMALIZATION}" \
  --program-path "${PROGRAM_PATH}" \
  --use-init-hidden \
  --mnist-sudoku-digit-pretrain-epochs "${PRETRAIN_EPOCHS}" \
  --mnist-sudoku-digit-pretrain-fraction "${PCT}" \
  --mnist-sudoku-digit-pretrain-path "${WEIGHT_PATH}" \
  --mnist-sudoku-image-dataset "${IMAGE_DATASET}" \
  --mnist-root "${MNIST_ROOT}" \
  --mnist-sudoku-root "${SUDOKU_ROOT}" \
  --log-root "${LOG_ROOT}/pretrain_pct_${PCT_TAG}_seed_${SEED}" \
  --batch-size "${BATCH_SIZE}" \
  "${PASSTHROUGH[@]}"
