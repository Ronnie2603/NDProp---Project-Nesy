#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

TRAIN_COUNT="${TRAIN_COUNT:-10000}"
TEST_COUNT="${TEST_COUNT:-1000}"
BASE="${BASE:-2}"
CLUES="${CLUES:-4}"
SUDOKU_SEED="${SUDOKU_SEED:-0}"
MAX_TRIES="${MAX_TRIES:-1000000}"
FORCE_GENERATE="${FORCE_GENERATE:-0}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Generates the tensors consumed by the Visual Sudoku experiment. Existing
complete tensors are reused unless generation is forced.

Options:
      --out-dir PATH       Tensor output directory (default: ${SUDOKU_ROOT})
      --train-count N      Training puzzles (default: ${TRAIN_COUNT})
      --test-count N       Test puzzles (default: ${TEST_COUNT})
      --base N             Sudoku base (default: ${BASE})
      --clues N            Minimum starting clues (default: ${CLUES})
      --mnist-root PATH    Image-dataset root (default: ${MNIST_ROOT})
      --image-dataset NAME mnist|fashion_mnist|cifar10 (default: ${IMAGE_DATASET})
      --seed N             Generator seed (default: ${SUDOKU_SEED})
      --max-tries N        Maximum generation attempts (default: ${MAX_TRIES})
      --force              Regenerate even when all expected files exist
  -h, --help               Show this message and exit
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) SUDOKU_ROOT="$2"; shift 2 ;;
    --train-count) TRAIN_COUNT="$2"; shift 2 ;;
    --test-count) TEST_COUNT="$2"; shift 2 ;;
    --base) BASE="$2"; shift 2 ;;
    --clues) CLUES="$2"; shift 2 ;;
    --mnist-root) MNIST_ROOT="$2"; shift 2 ;;
    --image-dataset) IMAGE_DATASET="$2"; shift 2 ;;
    --seed) SUDOKU_SEED="$2"; shift 2 ;;
    --max-tries) MAX_TRIES="$2"; shift 2 ;;
    --force) FORCE_GENERATE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

required_files=(
  train_features.pt
  train_features_img.pt
  train_labels.pt
  test_features.pt
  test_features_img.pt
  test_labels.pt
  perm.pt
)

if [[ "${FORCE_GENERATE}" != "1" && "${FORCE_GENERATE}" != "true" ]]; then
  complete=true
  for file in "${required_files[@]}"; do
    if [[ ! -f "${SUDOKU_ROOT}/${file}" ]]; then
      complete=false
      break
    fi
  done
  if [[ "${complete}" == true ]]; then
    echo "Visual Sudoku tensors already exist in ${SUDOKU_ROOT}; skipping generation."
    echo "Use --force or FORCE_GENERATE=1 to regenerate."
    exit 0
  fi
fi

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/src/generate_mnist_sudoku.py" \
  --out-dir "${SUDOKU_ROOT}" \
  --train-count "${TRAIN_COUNT}" \
  --test-count "${TEST_COUNT}" \
  --base "${BASE}" \
  --clues "${CLUES}" \
  --mnist-root "${MNIST_ROOT}" \
  --image-dataset "${IMAGE_DATASET}" \
  --seed "${SUDOKU_SEED}" \
  --max-tries "${MAX_TRIES}"
