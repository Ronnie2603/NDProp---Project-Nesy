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

Trains NDProp for one Visual Sudoku PCT/SEED pair using a pretrained digit
classifier.

Options:
      --pct VALUE         Digit-label fraction (default: ${PCT})
      --seed N            Random seed (default: ${SEED})
      --device DEVICE     Compute device (default: cuda:0)
      --weight-path PATH  Pretrained checkpoint derived from pct/seed by default
  -h, --help              Show this message and exit

Arguments after '--', as well as unrecognized backend options, are appended to
src/nesy.py and may override advanced model settings.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pct) PCT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --weight-path) WEIGHT_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; PASSTHROUGH+=("$@"); break ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

PCT_TAG="$(pct_tag "${PCT}")"
WEIGHT_PATH="${WEIGHT_PATH:-$(weight_path_for "${PCT}" "${SEED}")}"
DEVICE_ARG="$(device_arg)"

echo "=== Visual Sudoku NDProp train: seed=${SEED} pct=${PCT} device=${DEVICE_ARG} ==="
"${PYTHON_BIN}" "${PROJECT_ROOT}/src/nesy.py" \
  --dataset mnist_sudoku \
  --device "${DEVICE_ARG}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --iters "${ITERS}" \
  --logical-dimension "${LOGICAL_DIM}" \
  --tnorm "${T_NORM}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --decision-normalization "${DECISION_NORMALIZATION}" \
  --program-path "${PROGRAM_PATH}" \
  --use-init-hidden \
  --mnist-sudoku-digit-pretrain-epochs 0 \
  --mnist-sudoku-digit-pretrain-path "${WEIGHT_PATH}" \
  --mnist-sudoku-image-dataset "${IMAGE_DATASET}" \
  --mnist-root "${MNIST_ROOT}" \
  --mnist-sudoku-root "${SUDOKU_ROOT}" \
  --log-root "${LOG_ROOT}/train_pct_${PCT_TAG}_seed_${SEED}" \
  --batch-size "${BATCH_SIZE}" \
  "${PASSTHROUGH[@]}"
