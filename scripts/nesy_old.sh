#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET="addition" # addition | addition2 | addition3 | add2x2 | member3 | awa | celeba | predcls
DEVICE="cuda:2"
BATCH_SIZE=64
EPOCHS=10
LR=1e-3
ITERS=2
LOGICAL_DIM=1
SEED=7
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
CELEBA_ROOT="/home/moriyama/ASP/celeba/2"
CELEBA_ATTR_FILE=""
CELEBA_PARTITION_FILE=""
CELEBA_IMAGE_DIR=""
PREDCLS_DATA=""
PREDCLS_MAX_OBJECTS=10
PREDCLS_RELATIONS="left,right,above,below,overlap"
PREDCLS_TRAIN_SIZE=512
PREDCLS_EVAL_SIZE=128
PREDCLS_TRAIN_FRACTION=0.8
PREDCLS_SPLIT_SEED=0
PREDCLS_LABEL_FRACTION=1.0
PREDCLS_STABILITY_WEIGHT=0.1
PREDCLS_TRANSITIVE=false
PREDCLS_TRANSITIVE_REL="left"
PREDCLS_RUN_BASELINES=false
PREDCLS_BASELINE_SEED=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- extra-args]

Runs the NeSy training pipeline (src/nesy.py) for either dataset.

Options:
  -s, --dataset NAME       Dataset to run (addition | addition2 | addition3 | add2x2 | member3 | awa | celeba | predcls) (default: ${DATASET})
  -d, --device DEVICE      Compute device (default: ${DEVICE})
  -b, --batch-size N       Batch size (default: ${BATCH_SIZE})
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
      --seed N             RNG seed (default: ${SEED})
      --log-root PATH      Root directory for run logs/checkpoints (default: auto under ./runs/<dataset>)
      --log-interval N     Logging interval in batches (default: ${LOG_INTERVAL})
      --num-workers N      DataLoader workers (default: ${NUM_WORKERS})
      --celeba-root PATH   Root for CelebA data (default: ${CELEBA_ROOT})
      --celeba-attr-file PATH  CelebA attribute CSV (default: list_attr_celeba.csv under root)
      --celeba-partition-file PATH CelebA split CSV (default: list_eval_partition.csv under root)
      --celeba-image-dir PATH    CelebA image directory (default: img_align_celeba under root)
      --predcls-data PATH    PredCls data file (.pt or .json)
      --predcls-max-objects N Max objects per image (default: ${PREDCLS_MAX_OBJECTS})
      --predcls-relations CSV Comma-separated relations (default: ${PREDCLS_RELATIONS})
      --predcls-train-size N Synthetic train size (default: ${PREDCLS_TRAIN_SIZE})
      --predcls-eval-size N  Synthetic eval size (default: ${PREDCLS_EVAL_SIZE})
      --predcls-train-fraction F Train fraction when splitting a file (default: ${PREDCLS_TRAIN_FRACTION})
      --predcls-split-seed N Split seed (default: ${PREDCLS_SPLIT_SEED})
      --predcls-label-fraction F Fraction of labeled pairs (default: ${PREDCLS_LABEL_FRACTION})
      --predcls-stability-weight F Stability loss weight (default: ${PREDCLS_STABILITY_WEIGHT})
      --predcls-transitive  Enable transitivity rule
      --predcls-transitive-relation NAME Relation name for transitivity (default: ${PREDCLS_TRANSITIVE_REL})
      --predcls-run-baselines Compute raw/random/learned baselines
      --predcls-baseline-seed N Seed for random baseline (default: ${PREDCLS_BASELINE_SEED})
  -h, --help               Show this message and exit

Any arguments after '--' are passed directly to src/nesy.py.
EOF
}

POSITIONAL=()
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
    --seed) SEED="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --celeba-root) CELEBA_ROOT="$2"; shift 2 ;;
    --celeba-attr-file) CELEBA_ATTR_FILE="$2"; shift 2 ;;
    --celeba-partition-file) CELEBA_PARTITION_FILE="$2"; shift 2 ;;
    --celeba-image-dir) CELEBA_IMAGE_DIR="$2"; shift 2 ;;
    --predcls-data) PREDCLS_DATA="$2"; shift 2 ;;
    --predcls-max-objects) PREDCLS_MAX_OBJECTS="$2"; shift 2 ;;
    --predcls-relations) PREDCLS_RELATIONS="$2"; shift 2 ;;
    --predcls-train-size) PREDCLS_TRAIN_SIZE="$2"; shift 2 ;;
    --predcls-eval-size) PREDCLS_EVAL_SIZE="$2"; shift 2 ;;
    --predcls-train-fraction) PREDCLS_TRAIN_FRACTION="$2"; shift 2 ;;
    --predcls-split-seed) PREDCLS_SPLIT_SEED="$2"; shift 2 ;;
    --predcls-label-fraction) PREDCLS_LABEL_FRACTION="$2"; shift 2 ;;
    --predcls-stability-weight) PREDCLS_STABILITY_WEIGHT="$2"; shift 2 ;;
    --predcls-transitive) PREDCLS_TRANSITIVE=true; shift ;;
    --predcls-transitive-relation) PREDCLS_TRANSITIVE_REL="$2"; shift 2 ;;
    --predcls-run-baselines) PREDCLS_RUN_BASELINES=true; shift ;;
    --predcls-baseline-seed) PREDCLS_BASELINE_SEED="$2"; shift 2 ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

USE_MNIST=true
case "${DATASET}" in
  addition)
    DEFAULT_TRAIN="${PROJECT_ROOT}/../NeurASP/examples/mnistAdd/data/train_data.txt"
    DEFAULT_EVAL="${PROJECT_ROOT}/../NeurASP/examples/mnistAdd/data/test_data.txt"
    DEFAULT_LIMIT=30000
    DEFAULT_PROG="${PROJECT_ROOT}/src/nesy/programs/addition_simple.aspif"
    ITERS=2
    EPOCHS=20
    T_NORM="product"
    ;;
  addition2)
    DEFAULT_TRAIN="${PROJECT_ROOT}/../NeurASP/examples/data/mnistAdd2_train.txt"
    DEFAULT_EVAL="${PROJECT_ROOT}/../NeurASP/examples/data/mnistAdd2_test.txt"
    DEFAULT_LIMIT=-1
    DEFAULT_PROG="${PROJECT_ROOT}/src/nesy/programs/addition2_simple.aspif"
    # DEFAULT_PROG="${PROJECT_ROOT}/src/nesy/programs/addition2.lp"
    ITERS=3
    HIDDEN_DIM=8
    EPOCHS=50
    T_NORM="product"
    ;;
  addition3)
    DEFAULT_TRAIN="${PROJECT_ROOT}/../NeurASP/examples/data/mnistAdd3_train.txt"
    DEFAULT_EVAL="${DEFAULT_TRAIN}"
    DEFAULT_LIMIT=-1
    DEFAULT_PROG="${PROJECT_ROOT}/src/nesy/programs/addition3.lp"
    ITERS=3
    EPOCHS=50
    ;;
  add2x2)
    DEFAULT_TRAIN="${PROJECT_ROOT}/../NeurASP/examples/data/add2x2_train.txt"
    DEFAULT_EVAL="${PROJECT_ROOT}/../NeurASP/examples/data/add2x2_test.txt"
    DEFAULT_LIMIT=-1
    DEFAULT_PROG="${PROJECT_ROOT}/src/nesy/programs/add2x2_simple.aspif"
    ITERS=3
    EPOCHS=50
    T_NORM="product"
    ;;
  member3)
    DEFAULT_TRAIN="${PROJECT_ROOT}/../NeurASP/examples/data/member3_train.txt"
    DEFAULT_EVAL=""
    DEFAULT_LIMIT=3000
    DEFAULT_PROG="${PROJECT_ROOT}/src/nesy/programs/member3.lp"
    ITERS=2
    EPOCHS=10
    T_NORM="product"
    ;;
  awa)
    USE_MNIST=false
    DEFAULT_TRAIN=""
    DEFAULT_EVAL=""
    DEFAULT_LIMIT=""
    DEFAULT_PROG="${PROJECT_ROOT}/src/nesy/programs/awa_simple.aspif"
    # DEFAULT_PROG="${PROJECT_ROOT}/src/nesy/programs/awa_small.aspif"
    ITERS=5
    EPOCHS=50
    HIDDEN_DIM=8
    LOGICAL_DIM=1
    # LR=1e-4
    T_NORM="product"
    ;;
  celeba)
    USE_MNIST=false
    DEFAULT_TRAIN=""
    DEFAULT_EVAL=""
    DEFAULT_LIMIT=""
    DEFAULT_PROG="${PROJECT_ROOT}/src/nesy/programs/celeba.aspif"
    ITERS=2
    EPOCHS=10
    T_NORM="product"
    ;;
  predcls)
    USE_MNIST=false
    DEFAULT_TRAIN=""
    DEFAULT_EVAL=""
    DEFAULT_LIMIT=""
    DEFAULT_PROG="${PROJECT_ROOT}/src/nesy/programs/predcls_rel.lp"
    ITERS=5
    EPOCHS=20
    T_NORM="product"
    ;;
  *)
    echo "Unknown dataset: ${DATASET}" >&2
    exit 1
    ;;
esac

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
  --use-init-hidden
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
if [[ "${USE_MNIST}" == true ]]; then
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

if [[ "${USE_MNIST}" == true ]]; then
  if [[ -n "${PROGRAM_PATH}" ]]; then
    CMD+=(--program-path "${PROGRAM_PATH}")
  elif [[ -n "${DEFAULT_PROG}" ]]; then
    CMD+=(--program-path "${DEFAULT_PROG}")
  fi
  if [[ -n "${CLINGO_PATH}" ]]; then
    CMD+=(--clingo-path "${CLINGO_PATH}")
  fi
fi

if [[ "${DATASET}" == "awa" ]]; then
  CMD+=(--awa-attr-pretrain-path "./weights/attr_mapper_0.8.pt" --awa-train-fraction 0.8)
  CMD+=(--program-path "${DEFAULT_PROG}")
  # CMD+=(--awa-attr-pretrain-epochs 50)
  CMD+=(--awa-run-baselines --awa-baseline-mode "both" --awa-baseline-epochs 50)
fi

if [[ "${DATASET}" == "celeba" ]]; then
  if [[ -n "${PROGRAM_PATH}" ]]; then
    CMD+=(--program-path "${PROGRAM_PATH}")
  elif [[ -n "${DEFAULT_PROG}" ]]; then
    CMD+=(--program-path "${DEFAULT_PROG}")
  fi
  if [[ -n "${CLINGO_PATH}" ]]; then
    CMD+=(--clingo-path "${CLINGO_PATH}")
  fi
  CMD+=(--celeba-root "${CELEBA_ROOT}")
  if [[ -n "${CELEBA_ATTR_FILE}" ]]; then
    CMD+=(--celeba-attr-file "${CELEBA_ATTR_FILE}")
  fi
  if [[ -n "${CELEBA_PARTITION_FILE}" ]]; then
    CMD+=(--celeba-partition-file "${CELEBA_PARTITION_FILE}")
  fi
  if [[ -n "${CELEBA_IMAGE_DIR}" ]]; then
    CMD+=(--celeba-image-dir "${CELEBA_IMAGE_DIR}")
  fi
  if [[ -n "${EVAL_LIMIT}" ]]; then
    CMD+=(--eval-limit "${EVAL_LIMIT}")
  fi
  if [[ -n "${TRAIN_LIMIT}" ]]; then
    CMD+=(--train-limit "${TRAIN_LIMIT}")
  fi
  CMD+=(--celeba-pretrain-path "./weights/attr_mapper_celeba.pt")
  # CMD+=(--celeba-pretrain-epochs 2)
fi

if [[ "${DATASET}" == "predcls" ]]; then
  if [[ -n "${PROGRAM_PATH}" ]]; then
    CMD+=(--program-path "${PROGRAM_PATH}")
  elif [[ -n "${DEFAULT_PROG}" ]]; then
    CMD+=(--program-path "${DEFAULT_PROG}")
  fi
  if [[ -n "${CLINGO_PATH}" ]]; then
    CMD+=(--clingo-path "${CLINGO_PATH}")
  fi
  if [[ -n "${PREDCLS_DATA}" ]]; then
    CMD+=(--predcls-data "${PREDCLS_DATA}")
  fi
  CMD+=(
    --predcls-max-objects "${PREDCLS_MAX_OBJECTS}"
    --predcls-relations "${PREDCLS_RELATIONS}"
    --predcls-train-size "${PREDCLS_TRAIN_SIZE}"
    --predcls-eval-size "${PREDCLS_EVAL_SIZE}"
    --predcls-train-fraction "${PREDCLS_TRAIN_FRACTION}"
    --predcls-split-seed "${PREDCLS_SPLIT_SEED}"
    --predcls-label-fraction "${PREDCLS_LABEL_FRACTION}"
    --predcls-stability-weight "${PREDCLS_STABILITY_WEIGHT}"
    --predcls-transitive-relation "${PREDCLS_TRANSITIVE_REL}"
    --predcls-baseline-seed "${PREDCLS_BASELINE_SEED}"
  )
  if [[ "${PREDCLS_TRANSITIVE}" == true ]]; then
    CMD+=(--predcls-transitive)
  fi
  if [[ "${PREDCLS_RUN_BASELINES}" == true ]]; then
    CMD+=(--predcls-run-baselines)
  fi
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
