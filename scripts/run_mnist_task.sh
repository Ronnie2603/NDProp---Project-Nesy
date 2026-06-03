#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET="addition" # addition | addition2 | add2x2 | apply2x2 | member3 | member5 | mnist_sudoku
DEVICE="cuda:2"
BATCH_SIZE=""
EPOCHS=10
LR=1e-3
ITERS=2
EPOCHS_SET=false
LR_SET=false
ITERS_SET=false
LOGICAL_DIM=1
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
T_NORM="godel"  # lukasiewicz | godel | product
LOG_ROOT=""
HIDDEN_DIM=8
OP_ROOT=""
OP_TRAIN_SUBDIR="train_apply2x2"
OP_EVAL_SUBDIR="test_apply2x2"
MNIST_SUDOKU_ROOT=""
MNIST_SUDOKU_DIGIT_PATH=""
INFERENCE_ONLY=false
INFERENCE_WARMUP=""
INFERENCE_REPEATS=""

DATASET_LIST="addition|addition2|add2x2|apply2x2|member3|member5|mnist_sudoku"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- extra-args]

Runs one neuro-symbolic MNIST reasoning task from the paper experiments.

Options:
  -s, --dataset NAME       Dataset to run (${DATASET_LIST}) (default: ${DATASET})
  -d, --device DEVICE      Compute device (default: ${DEVICE})
  -b, --batch-size N       Batch size (default: dataset-specific)
  -e, --epochs N           Number of epochs (default: ${EPOCHS})
  -l, --lr VALUE           Learning rate (default: ${LR})
  -i, --iters N            GNN iterations (default: ${ITERS})
  -m, --mnist-root PATH    MNIST root directory (default: ${MNIST_ROOT})
      --train-file PATH    Train pairs file (task-specific; optional, falls back to random sampling)
      --eval-file PATH     Eval pairs file (task-specific; optional, falls back to random sampling)
      --train-limit N      Limit number of training examples
      --eval-limit N       Limit number of eval examples
      --program-path PATH  ASP/aspif program path (defaults to dataset built-in)
      --clingo-path PATH   clingo executable
      --pairs-per-sample N Number of random partner pairs per digit (addition only, default: ${PAIRS_PER_SAMPLE})
      --op-root PATH       Operator dataset root (apply2x2 only; default: assets/operator_images/hasy)
      --op-train-subdir S  Operator train subdir (apply2x2 only)
      --op-eval-subdir S   Operator eval subdir (apply2x2 only)
      --mnist-sudoku-root PATH       MNIST Sudoku dataset root
      --mnist-sudoku-digit-path PATH Pretrained digit CNN weights for MNIST Sudoku
      --inference-only      Measure single-sample inference time and exit
      --inference-warmup N  Warmup runs before timing inference
      --inference-repeats N Timed repeats for single-sample inference
      --seed N             RNG seed (default: ${SEED})
      --log-root PATH      Root directory for run logs/checkpoints (default: auto under runs_nesy/<dataset>)
      --log-interval N     Logging interval in batches (default: ${LOG_INTERVAL})
      --num-workers N      DataLoader workers (default: ${NUM_WORKERS})
  -h, --help               Show this message and exit

Any arguments after '--' are passed directly to src/nesy.py.
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--dataset) DATASET="$2"; shift 2 ;;
    -d|--device) DEVICE="$2"; shift 2 ;;
    -b|--batch-size) BATCH_SIZE="$2"; shift 2 ;;
    -e|--epochs) EPOCHS="$2"; EPOCHS_SET=true; shift 2 ;;
    -l|--lr) LR="$2"; LR_SET=true; shift 2 ;;
    -i|--iters) ITERS="$2"; ITERS_SET=true; shift 2 ;;
    -m|--mnist-root) MNIST_ROOT="$2"; shift 2 ;;
    --train-file) TRAIN_FILE="$2"; shift 2 ;;
    --eval-file) EVAL_FILE="$2"; shift 2 ;;
    --train-limit) TRAIN_LIMIT="$2"; shift 2 ;;
    --eval-limit) EVAL_LIMIT="$2"; shift 2 ;;
    --program-path) PROGRAM_PATH="$2"; shift 2 ;;
    --clingo-path) CLINGO_PATH="$2"; shift 2 ;;
    --log-interval) LOG_INTERVAL="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --logical-dim) LOGICAL_DIM="$2"; shift 2 ;;
    --pairs-per-sample) PAIRS_PER_SAMPLE="$2"; shift 2 ;;
    --op-root) OP_ROOT="$2"; shift 2 ;;
    --op-train-subdir) OP_TRAIN_SUBDIR="$2"; shift 2 ;;
    --op-eval-subdir) OP_EVAL_SUBDIR="$2"; shift 2 ;;
    --mnist-sudoku-root) MNIST_SUDOKU_ROOT="$2"; shift 2 ;;
    --mnist-sudoku-digit-path) MNIST_SUDOKU_DIGIT_PATH="$2"; shift 2 ;;
    --inference-only) INFERENCE_ONLY=true; shift ;;
    --inference-warmup) INFERENCE_WARMUP="$2"; shift 2 ;;
    --inference-repeats) INFERENCE_REPEATS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    *) shift ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

declare -A DEFAULT_TRAIN
declare -A DEFAULT_EVAL
declare -A DEFAULT_LIMIT
declare -A DEFAULT_PROG
declare -A DEFAULT_ITERS
declare -A DEFAULT_EPOCHS
declare -A DEFAULT_TNORM
declare -A DEFAULT_HIDDEN
declare -A DEFAULT_LOGICAL
declare -A DEFAULT_BATCH
declare -A DEFAULT_LR
declare -A USE_MNIST

DEFAULT_TRAIN[addition]="${PROJECT_ROOT}/indexes/mnistAdd_train.txt"
DEFAULT_EVAL[addition]="${PROJECT_ROOT}/indexes/mnistAdd_test.txt"
DEFAULT_LIMIT[addition]=-1
DEFAULT_PROG[addition]="${PROJECT_ROOT}/src/examples/addition/programs/addition_simple.aspif"
DEFAULT_ITERS[addition]=1
DEFAULT_EPOCHS[addition]=10
DEFAULT_TNORM[addition]="product"
DEFAULT_HIDDEN[addition]=8
DEFAULT_LOGICAL[addition]="${LOGICAL_DIM}"
DEFAULT_BATCH[addition]=64
DEFAULT_LR[addition]="${LR}"
USE_MNIST[addition]=true

DEFAULT_TRAIN[addition2]="${PROJECT_ROOT}/indexes/mnistAdd2_train.txt"
DEFAULT_EVAL[addition2]="${PROJECT_ROOT}/indexes/mnistAdd2_test.txt"
DEFAULT_LIMIT[addition2]=-1
DEFAULT_PROG[addition2]="${PROJECT_ROOT}/src/examples/addition2/programs/addition2_simple.aspif"
DEFAULT_ITERS[addition2]=1
DEFAULT_EPOCHS[addition2]=10
DEFAULT_TNORM[addition2]="product"
DEFAULT_HIDDEN[addition2]=8
DEFAULT_LOGICAL[addition2]="${LOGICAL_DIM}"
DEFAULT_BATCH[addition2]=64
DEFAULT_LR[addition2]="${LR}"
USE_MNIST[addition2]=true


DEFAULT_TRAIN[add2x2]="${PROJECT_ROOT}/indexes/add2x2_train.txt"
DEFAULT_EVAL[add2x2]="${PROJECT_ROOT}/indexes/add2x2_test.txt"
DEFAULT_LIMIT[add2x2]=-1
DEFAULT_PROG[add2x2]="${PROJECT_ROOT}/src/examples/add2x2/programs/add2x2_simple.aspif"
DEFAULT_ITERS[add2x2]=1
DEFAULT_EPOCHS[add2x2]=10
DEFAULT_TNORM[add2x2]="product"
DEFAULT_HIDDEN[add2x2]="8"
DEFAULT_LOGICAL[add2x2]="${LOGICAL_DIM}"
DEFAULT_BATCH[add2x2]=64
DEFAULT_LR[add2x2]="${LR}"
USE_MNIST[add2x2]=true

DEFAULT_TRAIN[apply2x2]="${PROJECT_ROOT}/indexes/apply2x2_train_10000.txt"
DEFAULT_EVAL[apply2x2]="${PROJECT_ROOT}/indexes/apply2x2_test.txt"
DEFAULT_LIMIT[apply2x2]=-1
DEFAULT_PROG[apply2x2]="${PROJECT_ROOT}/src/examples/apply2x2/programs/apply2x2.aspif"
DEFAULT_ITERS[apply2x2]=1
DEFAULT_EPOCHS[apply2x2]=10
DEFAULT_TNORM[apply2x2]="product"
DEFAULT_HIDDEN[apply2x2]="8"
DEFAULT_LOGICAL[apply2x2]="${LOGICAL_DIM}"
DEFAULT_BATCH[apply2x2]=64
DEFAULT_LR[apply2x2]="${LR}"
USE_MNIST[apply2x2]=false

DEFAULT_TRAIN[member3]="${PROJECT_ROOT}/indexes/member3_train_10000.txt"
DEFAULT_EVAL[member3]="${PROJECT_ROOT}/indexes/member3_test.txt"
DEFAULT_LIMIT[member3]=-1
DEFAULT_PROG[member3]="${PROJECT_ROOT}/src/examples/member3/programs/member3_simple.aspif"
DEFAULT_ITERS[member3]=1
DEFAULT_EPOCHS[member3]=10
DEFAULT_TNORM[member3]="product"
DEFAULT_HIDDEN[member3]="8"
DEFAULT_LOGICAL[member3]="${LOGICAL_DIM}"
DEFAULT_BATCH[member3]=64
DEFAULT_LR[member3]="${LR}"
USE_MNIST[member3]=true

DEFAULT_TRAIN[member5]="${PROJECT_ROOT}/indexes/member5_train_10000.txt"
DEFAULT_EVAL[member5]="${PROJECT_ROOT}/indexes/member5_test.txt"
DEFAULT_LIMIT[member5]=-1
DEFAULT_PROG[member5]="${PROJECT_ROOT}/src/examples/member5/programs/member5_simple.aspif"
DEFAULT_ITERS[member5]=1
DEFAULT_EPOCHS[member5]=10
DEFAULT_TNORM[member5]="product"
DEFAULT_HIDDEN[member5]="8"
DEFAULT_LOGICAL[member5]="${LOGICAL_DIM}"
DEFAULT_BATCH[member5]=64
DEFAULT_LR[member5]="${LR}"
USE_MNIST[member5]=true

DEFAULT_TRAIN[mnist_sudoku]=""
DEFAULT_EVAL[mnist_sudoku]=""
DEFAULT_LIMIT[mnist_sudoku]=""
DEFAULT_PROG[mnist_sudoku]="${PROJECT_ROOT}/src/examples/mnist_sudoku/programs/mnist_sudoku_4x4_2.aspif"
DEFAULT_ITERS[mnist_sudoku]=3
DEFAULT_EPOCHS[mnist_sudoku]=100
DEFAULT_TNORM[mnist_sudoku]="godel"
DEFAULT_HIDDEN[mnist_sudoku]=8
DEFAULT_LOGICAL[mnist_sudoku]=2
DEFAULT_BATCH[mnist_sudoku]=64
DEFAULT_LR[mnist_sudoku]=0.001
USE_MNIST[mnist_sudoku]=false
if [[ -z "${MNIST_SUDOKU_ROOT}" ]]; then
  MNIST_SUDOKU_ROOT="${PROJECT_ROOT}/data/mnist_sudoku"
fi

if [[ -z "${DEFAULT_PROG[${DATASET}]+x}" ]]; then
  echo "Unknown dataset: ${DATASET}" >&2
  exit 1
fi

DEFAULT_TRAIN="${DEFAULT_TRAIN[${DATASET}]}"
DEFAULT_EVAL="${DEFAULT_EVAL[${DATASET}]}"
DEFAULT_LIMIT="${DEFAULT_LIMIT[${DATASET}]}"
DEFAULT_PROG="${DEFAULT_PROG[${DATASET}]}"
if [[ "${ITERS_SET}" != true ]]; then
  ITERS="${DEFAULT_ITERS[${DATASET}]}"
fi
if [[ "${EPOCHS_SET}" != true ]]; then
  EPOCHS="${DEFAULT_EPOCHS[${DATASET}]}"
fi
T_NORM="${DEFAULT_TNORM[${DATASET}]}"
HIDDEN_DIM="${DEFAULT_HIDDEN[${DATASET}]}"
LOGICAL_DIM="${DEFAULT_LOGICAL[${DATASET}]}"
if [[ "${LR_SET}" != true ]]; then
  LR="${DEFAULT_LR[${DATASET}]}"
fi
USE_MNIST="${USE_MNIST[${DATASET}]}"

if [[ -z "${BATCH_SIZE}" ]]; then
  BATCH_SIZE="${DEFAULT_BATCH[${DATASET}]}"
fi

CMD=(
  "${PYTHON_BIN}"
  "${PROJECT_ROOT}/src/nesy.py"
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

if [[ "${USE_MNIST}" == true ]]; then
  CMD+=(--mnist-root "${MNIST_ROOT}")
fi

if [[ -n "${LOG_ROOT}" ]]; then
  CMD+=(--log-root "${LOG_ROOT}")
fi

use_train="${TRAIN_FILE:-$DEFAULT_TRAIN}"
use_eval="${EVAL_FILE:-$DEFAULT_EVAL}"
use_limit="${TRAIN_LIMIT:-$DEFAULT_LIMIT}"

if [[ -n "${use_train}" ]]; then
  CMD+=(--train-pairs "${use_train}")
fi
if [[ -n "${use_eval}" ]]; then
  CMD+=(--eval-pairs "${use_eval}")
fi
if [[ "${USE_MNIST}" == true || "${DATASET}" == "apply2x2" ]]; then
  if [[ -n "${use_limit}" ]]; then
    CMD+=(--train-limit "${use_limit}")
  fi
  if [[ -n "${EVAL_LIMIT}" ]]; then
    CMD+=(--eval-limit "${EVAL_LIMIT}")
  fi
fi
if [[ "${DATASET}" == "addition" ]]; then
  CMD+=(--pairs-per-sample "${PAIRS_PER_SAMPLE}")
fi
if [[ "${INFERENCE_ONLY}" == true ]]; then
  CMD+=(--inference-only)
fi
if [[ -n "${INFERENCE_WARMUP}" ]]; then
  CMD+=(--inference-warmup "${INFERENCE_WARMUP}")
fi
if [[ -n "${INFERENCE_REPEATS}" ]]; then
  CMD+=(--inference-repeats "${INFERENCE_REPEATS}")
fi

if [[ -n "${PROGRAM_PATH}" ]]; then
  CMD+=(--program-path "${PROGRAM_PATH}")
elif [[ -n "${DEFAULT_PROG}" ]]; then
  CMD+=(--program-path "${DEFAULT_PROG}")
fi
if [[ -n "${CLINGO_PATH}" ]]; then
  CMD+=(--clingo-path "${CLINGO_PATH}")
fi
if [[ "${DATASET}" == "apply2x2" ]]; then
  if [[ -z "${OP_ROOT}" ]]; then
    OP_ROOT="${PROJECT_ROOT}/assets/operator_images/hasy"
  fi
  if [[ -n "${OP_ROOT}" ]]; then
    CMD+=(--op-root "${OP_ROOT}")
  fi
  CMD+=(--op-train-subdir "${OP_TRAIN_SUBDIR}")
  CMD+=(--op-eval-subdir "${OP_EVAL_SUBDIR}")
fi

if [[ "${DATASET}" == "mnist_sudoku" ]]; then
  PCT=0.01
  CMD+=(--use-init-hidden --mnist-sudoku-digit-pretrain-path "./weights/mnist_sudoku_digit_1p0_seed_2.pt" --mnist-sudoku-digit-pretrain-fraction $PCT --mnist-sudoku-image-dataset mnist)
  CMD+=(--mnist-root "${MNIST_ROOT}")
  if [[ -n "${MNIST_SUDOKU_ROOT}" ]]; then
    CMD+=(--mnist-sudoku-root "${MNIST_SUDOKU_ROOT}")
  fi
  if [[ -n "${MNIST_SUDOKU_DIGIT_PATH}" ]]; then
    CMD+=(--mnist-sudoku-digit-path "${MNIST_SUDOKU_DIGIT_PATH}")
  fi
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
