#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

source "${SCRIPT_DIR}/defaults.sh"

DATASET="addition"
DEVICE="${DEVICE:-cuda:2}"
BATCH_SIZE=""
EPOCHS=10
ITERS=2
SEED=2
MNIST_ROOT="${PROJECT_ROOT}/data"
LOG_INTERVAL=100
NUM_WORKERS=0
TRAIN_LIMIT=""
EVAL_LIMIT=""
TRAIN_FILE=""
EVAL_FILE=""
PROGRAM_PATH=""
CLINGO_PATH=""
PAIRS_PER_SAMPLE=2
LOG_ROOT="${LOG_ROOT:-}"
OP_ROOT=""
OP_TRAIN_SUBDIR="train_apply2x2"
OP_EVAL_SUBDIR="test_apply2x2"
INFERENCE_ONLY=false
INFERENCE_WARMUP=""
INFERENCE_REPEATS=""
EPOCHS_SET=false
ITERS_SET=false
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- extra-args]

Runs one neuro-symbolic task.

Task and training:
  -s, --dataset NAME         addition|addition2|add2x2|apply2x2|member3|member5
  -d, --device DEVICE        Compute device (default: ${DEVICE})
  -b, --batch-size N         Batch size (default: task-specific)
  -e, --epochs N             Training epochs (default: task-specific)
  -i, --iters N              NDProp iterations (default: task-specific)
      --seed N               RNG seed (default: ${SEED})
      --log-root PATH        Output root (default: PROJECT_ROOT/runs_nesy/neuro_symbolic_tasks/DATASET)

Data and programs:
  -m, --mnist-root PATH      MNIST root (default: ${MNIST_ROOT})
      --train-file PATH      Override the task's training split
      --eval-file PATH       Override the task's evaluation split
      --train-limit N        Limit training examples
      --eval-limit N         Limit evaluation examples
      --program-path PATH    Override the task's ASP/ASPIF program
      --clingo-path PATH     clingo executable

Task-specific data:
      --pairs-per-sample N   Addition random-pair count (default: ${PAIRS_PER_SAMPLE})
      --op-root PATH         apply2x2 operator-image root
      --op-train-subdir S    apply2x2 training subdirectory
      --op-eval-subdir S     apply2x2 evaluation subdirectory
Runtime and measurement:
      --num-workers N        DataLoader workers (default: ${NUM_WORKERS})
      --log-interval N       Logging interval (default: ${LOG_INTERVAL})
      --inference-only       Measure single-sample inference and exit
      --inference-warmup N   Warmup runs before timing
      --inference-repeats N  Number of timed runs
  -h, --help                 Show this message and exit

Advanced arguments after '--' are passed unchanged to src/nesy.py.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--dataset) DATASET="$2"; shift 2 ;;
    -d|--device) DEVICE="$2"; shift 2 ;;
    -b|--batch-size) BATCH_SIZE="$2"; shift 2 ;;
    -e|--epochs) EPOCHS="$2"; EPOCHS_SET=true; shift 2 ;;
    -i|--iters) ITERS="$2"; ITERS_SET=true; shift 2 ;;
    -m|--mnist-root) MNIST_ROOT="$2"; shift 2 ;;
    --train-file) TRAIN_FILE="$2"; shift 2 ;;
    --eval-file) EVAL_FILE="$2"; shift 2 ;;
    --train-limit) TRAIN_LIMIT="$2"; shift 2 ;;
    --eval-limit) EVAL_LIMIT="$2"; shift 2 ;;
    --program-path) PROGRAM_PATH="$2"; shift 2 ;;
    --clingo-path) CLINGO_PATH="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --log-interval) LOG_INTERVAL="$2"; shift 2 ;;
    --pairs-per-sample) PAIRS_PER_SAMPLE="$2"; shift 2 ;;
    --op-root) OP_ROOT="$2"; shift 2 ;;
    --op-train-subdir) OP_TRAIN_SUBDIR="$2"; shift 2 ;;
    --op-eval-subdir) OP_EVAL_SUBDIR="$2"; shift 2 ;;
    --inference-only) INFERENCE_ONLY=true; shift ;;
    --inference-warmup) INFERENCE_WARMUP="$2"; shift 2 ;;
    --inference-repeats) INFERENCE_REPEATS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

neuro_symbolic_task_init_defaults
if [[ -z "${DEFAULT_PROG[${DATASET}]+x}" ]]; then
  echo "Unknown dataset: ${DATASET}" >&2
  exit 1
fi
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/runs_nesy/neuro_symbolic_tasks/${DATASET}}"

if [[ "${ITERS_SET}" != true ]]; then ITERS="${DEFAULT_ITERS[${DATASET}]}"; fi
if [[ "${EPOCHS_SET}" != true ]]; then EPOCHS="${DEFAULT_EPOCHS[${DATASET}]}"; fi
if [[ -z "${BATCH_SIZE}" ]]; then BATCH_SIZE="${DEFAULT_BATCH[${DATASET}]}"; fi
LR="${DEFAULT_LR[${DATASET}]}"
T_NORM="${DEFAULT_TNORM[${DATASET}]}"
HIDDEN_DIM="${DEFAULT_HIDDEN[${DATASET}]}"
LOGICAL_DIM="${DEFAULT_LOGICAL[${DATASET}]}"
USE_MNIST_TASK="${USE_MNIST[${DATASET}]}"
use_train="${TRAIN_FILE:-${DEFAULT_TRAIN[${DATASET}]}}"
use_eval="${EVAL_FILE:-${DEFAULT_EVAL[${DATASET}]}}"
use_limit="${TRAIN_LIMIT:-${DEFAULT_LIMIT[${DATASET}]}}"
use_program="${PROGRAM_PATH:-${DEFAULT_PROG[${DATASET}]}}"

CMD=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/src/nesy.py"
  --dataset "${DATASET}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --log-interval "${LOG_INTERVAL}"
  --num-workers "${NUM_WORKERS}"
  --iters "${ITERS}"
  --logical-dimension "${LOGICAL_DIM}"
  --tnorm "${T_NORM}"
  --hidden-dim "${HIDDEN_DIM}"
)

[[ "${USE_MNIST_TASK}" == true ]] && CMD+=(--mnist-root "${MNIST_ROOT}")
CMD+=(--log-root "${LOG_ROOT}")
[[ -n "${use_train}" ]] && CMD+=(--train-pairs "${use_train}")
[[ -n "${use_eval}" ]] && CMD+=(--eval-pairs "${use_eval}")
[[ -n "${use_program}" ]] && CMD+=(--program-path "${use_program}")
[[ -n "${CLINGO_PATH}" ]] && CMD+=(--clingo-path "${CLINGO_PATH}")

if [[ "${USE_MNIST_TASK}" == true || "${DATASET}" == "apply2x2" ]]; then
  [[ -n "${use_limit}" ]] && CMD+=(--train-limit "${use_limit}")
  [[ -n "${EVAL_LIMIT}" ]] && CMD+=(--eval-limit "${EVAL_LIMIT}")
fi

[[ "${DATASET}" == "addition" ]] && CMD+=(--pairs-per-sample "${PAIRS_PER_SAMPLE}")
[[ "${INFERENCE_ONLY}" == true ]] && CMD+=(--inference-only)
[[ -n "${INFERENCE_WARMUP}" ]] && CMD+=(--inference-warmup "${INFERENCE_WARMUP}")
[[ -n "${INFERENCE_REPEATS}" ]] && CMD+=(--inference-repeats "${INFERENCE_REPEATS}")

if [[ "${DATASET}" == "apply2x2" ]]; then
  OP_ROOT="${OP_ROOT:-${PROJECT_ROOT}/assets/operator_images/hasy}"
  CMD+=(--op-root "${OP_ROOT}")
  CMD+=(--op-train-subdir "${OP_TRAIN_SUBDIR}")
  CMD+=(--op-eval-subdir "${OP_EVAL_SUBDIR}")
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
