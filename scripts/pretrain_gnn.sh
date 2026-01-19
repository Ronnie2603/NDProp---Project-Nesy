#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DEVICE="cpu"
BATCH_SIZE=64
EPOCHS=50
LR=1e-3
TRAIN_SIZE=4096
EVAL_SIZE=1024
LOG_INTERVAL=100
NUM_WORKERS=0
LOGIT_BIAS=3.0
TEMPERATURE=1.0
ITERATIONS=30
GL_WEIGHT=10.0
BIN_WEIGHT=1.0
RUN_ROOT=""
GNNCHECKPOINT=""
PROGRAM_PATH=""
CLINGO_PATH=""
RESAMPLE=""

usage() {
  cat <<EOF_USAGE
Usage: $(basename "$0") [options] [-- extra-args]

Pretrain the StableModelGNN on synthetic addition inputs (src/pretrain_gnn.py).

Options:
  -d, --device DEVICE         Compute device (default: ${DEVICE})
  -b, --batch-size N          Batch size (default: ${BATCH_SIZE})
  -e, --epochs N              Training epochs (default: ${EPOCHS})
  -l, --lr VALUE              Learning rate (default: ${LR})
      --train-size N          Synthetic training samples per epoch (default: ${TRAIN_SIZE})
      --eval-size N           Synthetic evaluation samples (default: ${EVAL_SIZE})
      --log-interval N        Logging interval in batches (default: ${LOG_INTERVAL})
      --num-workers N         DataLoader workers (default: ${NUM_WORKERS})
      --logit-bias VALUE      Bias added to ground-truth digit logit (default: ${LOGIT_BIAS})
      --temperature VALUE     Sampling temperature (default: ${TEMPERATURE})
      --iters N               TP iterations (default: ${ITERATIONS})
      --gl-weight VALUE       Weight for GL loss (default: ${GL_WEIGHT})
      --bin-weight VALUE      Weight for binarisation loss (default: ${BIN_WEIGHT})
      --log-root PATH         Output root for run directories (default: runs_nesy under project root)
      --gnn-checkpoint PATH   Initialise from checkpoint/run directory
      --program-path PATH     Override addition program path
      --clingo-path PATH      Override clingo executable
      --resample              Regenerate synthetic data every epoch
  -h, --help                  Show this message and exit

Any arguments after '--' are forwarded to src/pretrain_gnn.py.
EOF_USAGE
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--device) DEVICE="$2"; shift 2 ;;
    -b|--batch-size) BATCH_SIZE="$2"; shift 2 ;;
    -e|--epochs) EPOCHS="$2"; shift 2 ;;
    -l|--lr) LR="$2"; shift 2 ;;
    --train-size) TRAIN_SIZE="$2"; shift 2 ;;
    --eval-size) EVAL_SIZE="$2"; shift 2 ;;
    --log-interval) LOG_INTERVAL="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --logit-bias) LOGIT_BIAS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --iters) ITERATIONS="$2"; shift 2 ;;
    --gl-weight) GL_WEIGHT="$2"; shift 2 ;;
    --bin-weight) BIN_WEIGHT="$2"; shift 2 ;;
    --log-root) RUN_ROOT="$2"; shift 2 ;;
    --gnn-checkpoint) GNNCHECKPOINT="$2"; shift 2 ;;
    --program-path) PROGRAM_PATH="$2"; shift 2 ;;
    --clingo-path) CLINGO_PATH="$2"; shift 2 ;;
    --resample) RESAMPLE="--resample"; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=($@); break ;;
    -*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

CMD=(
  "${PYTHON_BIN}"
  "${PROJECT_ROOT}/src/pretrain_gnn.py"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --train-size "${TRAIN_SIZE}"
  --eval-size "${EVAL_SIZE}"
  --log-interval "${LOG_INTERVAL}"
  --num-workers "${NUM_WORKERS}"
  --logit-bias "${LOGIT_BIAS}"
  --temperature "${TEMPERATURE}"
  --iters "${ITERATIONS}"
  --gl-weight "${GL_WEIGHT}"
  --bin-weight "${BIN_WEIGHT}"
)

if [[ -n "${RUN_ROOT}" ]]; then CMD+=(--log-root "${RUN_ROOT}"); fi
if [[ -n "${GNNCHECKPOINT}" ]]; then CMD+=(--gnn-checkpoint "${GNNCHECKPOINT}"); fi
if [[ -n "${PROGRAM_PATH}" ]]; then CMD+=(--program-path "${PROGRAM_PATH}"); fi
if [[ -n "${CLINGO_PATH}" ]]; then CMD+=(--clingo-path "${CLINGO_PATH}"); fi
if [[ -n "${RESAMPLE}" ]]; then CMD+=("${RESAMPLE}"); fi

CMD+=("${EXTRA_ARGS[@]}")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
