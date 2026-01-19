#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET="addition" # addition | addition2 | add2x2 | apply2x2 | member3 | member5 | awa | cubs | mnist_sudoku | hmcnn
DEVICE="cuda:2"
BATCH_SIZE=""
EPOCHS=10
LR=1e-3
ITERS=2
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
OP_TRAIN_SUBDIR=""
OP_EVAL_SUBDIR=""
HMCNN_DATASET=""
HMCNN_EVAL_SPLIT="val"
HMCNN_PRETRAIN_EPOCHS=0
HMCNN_PRETRAIN_LR=1e-3
HMCNN_PRETRAIN_PATH=""
HMCNN_HIDDEN_DIM=256
HMCNN_NUM_LAYERS=2
HMCNN_DROPOUT=0.2
HMCNN_NONLIN="relu"
CUBS_PRETRAIN_EPOCHS=0
CUBS_PRETRAIN_LR=1e-3
CUBS_PRETRAIN_PATH=""
MNIST_SUDOKU_ROOT=""
MNIST_SUDOKU_DIGIT_PATH=""

DATASET_LIST="addition|addition2|add2x2|apply2x2|member3|member5|awa|cubs|mnist_sudoku|hmcnn"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- extra-args]

Runs the NeSy training pipeline (src/nesy.py) for either dataset.

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
      --op-root PATH       Operator dataset root (apply2x2 only)
      --op-train-subdir S  Operator train subdir (apply2x2 only)
      --op-eval-subdir S   Operator eval subdir (apply2x2 only)
      --mnist-sudoku-root PATH       MNIST Sudoku dataset root
      --mnist-sudoku-digit-path PATH Pretrained digit CNN weights for MNIST Sudoku
      --cubs-pretrain-epochs N  CUBS attr mapper pretrain epochs (default: ${CUBS_PRETRAIN_EPOCHS})
      --cubs-pretrain-lr VALUE  CUBS attr mapper pretrain LR (default: ${CUBS_PRETRAIN_LR})
      --cubs-pretrain-path PATH CUBS attr mapper weights path
      --hmcnn-dataset NAME HMCNN dataset name (hmcnn only)
      --hmcnn-eval-split S HMCNN eval split (val|test, default: ${HMCNN_EVAL_SPLIT})
      --hmcnn-pretrain-epochs N  HMCNN pretrain epochs (default: ${HMCNN_PRETRAIN_EPOCHS})
      --hmcnn-pretrain-lr VALUE  HMCNN pretrain LR (default: ${HMCNN_PRETRAIN_LR})
      --hmcnn-pretrain-path PATH HMCNN pretrain weights path
      --hmcnn-hidden-dim N  HMCNN hidden dim (default: ${HMCNN_HIDDEN_DIM})
      --hmcnn-num-layers N  HMCNN num layers (default: ${HMCNN_NUM_LAYERS})
      --hmcnn-dropout P     HMCNN dropout (default: ${HMCNN_DROPOUT})
      --hmcnn-nonlin NAME   HMCNN nonlinearity (relu|tanh, default: ${HMCNN_NONLIN})
      --seed N             RNG seed (default: ${SEED})
      --log-root PATH      Root directory for run logs/checkpoints (default: auto under ./runs/<dataset>)
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
    -e|--epochs) EPOCHS="$2"; shift 2 ;;
    -l|--lr) LR="$2"; shift 2 ;;
    -i|--iters) ITERS="$2"; shift 2 ;;
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
    --cubs-pretrain-epochs) CUBS_PRETRAIN_EPOCHS="$2"; shift 2 ;;
    --cubs-pretrain-lr) CUBS_PRETRAIN_LR="$2"; shift 2 ;;
    --cubs-pretrain-path) CUBS_PRETRAIN_PATH="$2"; shift 2 ;;
    --hmcnn-dataset) HMCNN_DATASET="$2"; shift 2 ;;
    --hmcnn-eval-split) HMCNN_EVAL_SPLIT="$2"; shift 2 ;;
    --hmcnn-pretrain-epochs) HMCNN_PRETRAIN_EPOCHS="$2"; shift 2 ;;
    --hmcnn-pretrain-lr) HMCNN_PRETRAIN_LR="$2"; shift 2 ;;
    --hmcnn-pretrain-path) HMCNN_PRETRAIN_PATH="$2"; shift 2 ;;
    --hmcnn-hidden-dim) HMCNN_HIDDEN_DIM="$2"; shift 2 ;;
    --hmcnn-num-layers) HMCNN_NUM_LAYERS="$2"; shift 2 ;;
    --hmcnn-dropout) HMCNN_DROPOUT="$2"; shift 2 ;;
    --hmcnn-nonlin) HMCNN_NONLIN="$2"; shift 2 ;;
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

DEFAULT_TRAIN[awa]=""
DEFAULT_EVAL[awa]=""
DEFAULT_LIMIT[awa]=""
DEFAULT_PROG[awa]="${PROJECT_ROOT}/src/examples/awa/programs/awa_simple.aspif"
DEFAULT_ITERS[awa]=2
DEFAULT_EPOCHS[awa]=100
DEFAULT_TNORM[awa]="godel"
DEFAULT_HIDDEN[awa]=16
DEFAULT_LOGICAL[awa]=1
DEFAULT_BATCH[awa]=64
DEFAULT_LR[awa]=0.001
USE_MNIST[awa]=false

DEFAULT_TRAIN[cubs]=""
DEFAULT_EVAL[cubs]=""
DEFAULT_LIMIT[cubs]=""
DEFAULT_PROG[cubs]="${PROJECT_ROOT}/src/examples/cubs/programs/cubs2.aspif"
DEFAULT_ITERS[cubs]=2
DEFAULT_EPOCHS[cubs]=100
DEFAULT_TNORM[cubs]="godel"
DEFAULT_HIDDEN[cubs]=16
DEFAULT_LOGICAL[cubs]=1
DEFAULT_BATCH[cubs]=64
DEFAULT_LR[cubs]=0.001
USE_MNIST[cubs]=false
CUBS_PRETRAIN_EPOCHS=-1
CUBS_PRETRAIN_PATH="${PROJECT_ROOT}/weights/cubs_attr_map_resnet.pt"





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
MNIST_SUDOKU_ROOT="/home/moriyama/ASP/Sudoku/sudoku4"








DEFAULT_TRAIN[hmcnn]=""
DEFAULT_EVAL[hmcnn]=""
DEFAULT_LIMIT[hmcnn]=""
DEFAULT_PROG[hmcnn]=""
DEFAULT_ITERS[hmcnn]=3
DEFAULT_EPOCHS[hmcnn]=50
DEFAULT_TNORM[hmcnn]="product"
DEFAULT_HIDDEN[hmcnn]=64
DEFAULT_LOGICAL[hmcnn]=1
DEFAULT_BATCH[hmcnn]=64
DEFAULT_LR[hmcnn]=1e-3
USE_MNIST[hmcnn]=false

if [[ -z "${DEFAULT_PROG[${DATASET}]+x}" ]]; then
  echo "Unknown dataset: ${DATASET}" >&2
  exit 1
fi

DEFAULT_TRAIN="${DEFAULT_TRAIN[${DATASET}]}"
DEFAULT_EVAL="${DEFAULT_EVAL[${DATASET}]}"
DEFAULT_LIMIT="${DEFAULT_LIMIT[${DATASET}]}"
DEFAULT_PROG="${DEFAULT_PROG[${DATASET}]}"
ITERS="${DEFAULT_ITERS[${DATASET}]}"
EPOCHS="${DEFAULT_EPOCHS[${DATASET}]}"
T_NORM="${DEFAULT_TNORM[${DATASET}]}"
HIDDEN_DIM="${DEFAULT_HIDDEN[${DATASET}]}"
LOGICAL_DIM="${DEFAULT_LOGICAL[${DATASET}]}"
LR="${DEFAULT_LR[${DATASET}]}"
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
  # --use-init-hidden
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

if [[ -n "${PROGRAM_PATH}" ]]; then
  CMD+=(--program-path "${PROGRAM_PATH}")
elif [[ -n "${DEFAULT_PROG}" ]]; then
  CMD+=(--program-path "${DEFAULT_PROG}")
fi
if [[ -n "${CLINGO_PATH}" ]]; then
  CMD+=(--clingo-path "${CLINGO_PATH}")
fi
if [[ -n "${OP_ROOT}" ]]; then
  CMD+=(--op-root "${OP_ROOT}")
fi
if [[ -n "${OP_TRAIN_SUBDIR}" ]]; then
  CMD+=(--op-train-subdir "${OP_TRAIN_SUBDIR}")
fi
if [[ -n "${OP_EVAL_SUBDIR}" ]]; then
  CMD+=(--op-eval-subdir "${OP_EVAL_SUBDIR}")
fi

if [[ "${DATASET}" == "awa" ]]; then
  CMD+=(--awa-attr-pretrain-path "./weights/awa_attr_map.pt" --awa-train-fraction 0.1)
  CMD+=(--use-init-hidden)
  # CMD+=(--awa-class-split-dataset)
  # CMD+=(--awa-test-classes /home/moriyama/ASP/AwA/Animals_with_Attributes2/test_classes.txt)
  # CMD+=(--awa-train-classes /home/moriyama/ASP/AwA/Animals_with_Attributes2/train_classes.txt)
  # CMD+=(--awa-attr-pretrain-epochs 50)
  # CMD+=(--awa-run-baselines --awa-baseline-mode "both" --awa-baseline-epochs 50)
fi
if [[ "${DATASET}" == "mnist_sudoku" ]]; then
  PCT=0.01
  CMD+=(--use-init-hidden --mnist-sudoku-digit-pretrain-path "./weights/mnist_sudoku_digit_1p0_seed_2.pt" --mnist-sudoku-digit-pretrain-fraction $PCT --mnist-sudoku-image-dataset mnist)
  # CMD+=(--mnist-sudoku-run-baselines --mnist-sudoku-baseline-mode clingo)
  # CMD+=(--mnist-sudoku-digit-pretrain-epochs 1000)
  CMD+=(--mnist-root "${MNIST_ROOT}")
  if [[ -n "${MNIST_SUDOKU_ROOT}" ]]; then
    CMD+=(--mnist-sudoku-root "${MNIST_SUDOKU_ROOT}")
  fi
  if [[ -n "${MNIST_SUDOKU_DIGIT_PATH}" ]]; then
    CMD+=(--mnist-sudoku-digit-path "${MNIST_SUDOKU_DIGIT_PATH}")
  fi
fi
if [[ "${DATASET}" == "cubs" ]]; then
  CMD+=(--cubs-use-split)
  CMD+=(--cubs-attr-pretrain-epochs "${CUBS_PRETRAIN_EPOCHS}")
  CMD+=(--cubs-attr-pretrain-lr "${CUBS_PRETRAIN_LR}")
  CMD+=(--use-init-hidden)
  # CMD+=(--cubs-run-baselines)
  if [[ -n "${CUBS_PRETRAIN_PATH}" ]]; then
    CMD+=(--cubs-attr-pretrain-path "${CUBS_PRETRAIN_PATH}")
  fi
fi
if [[ "${DATASET}" == "hmcnn" ]]; then
  if [[ -z "${HMCNN_DATASET}" ]]; then
    echo "--hmcnn-dataset is required when dataset=hmcnn" >&2
    exit 1
  fi
  CMD+=(--hmcnn-dataset "${HMCNN_DATASET}")
  CMD+=(--hmcnn-eval-split "${HMCNN_EVAL_SPLIT}")
  CMD+=(--hmcnn-pretrain-epochs "${HMCNN_PRETRAIN_EPOCHS}")
  CMD+=(--hmcnn-pretrain-lr "${HMCNN_PRETRAIN_LR}")
  if [[ -n "${HMCNN_PRETRAIN_PATH}" ]]; then
    CMD+=(--hmcnn-pretrain-path "${HMCNN_PRETRAIN_PATH}")
  fi
  CMD+=(--hmcnn-hidden-dim "${HMCNN_HIDDEN_DIM}")
  CMD+=(--hmcnn-num-layers "${HMCNN_NUM_LAYERS}")
  CMD+=(--hmcnn-dropout "${HMCNN_DROPOUT}")
  CMD+=(--hmcnn-nonlin "${HMCNN_NONLIN}")
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
