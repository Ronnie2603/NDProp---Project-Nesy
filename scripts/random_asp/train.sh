#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

FAMILY="n2l"
DIFFICULTY="medium"
BENCHMARK_ROOT=""
DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MODEL_ITERS="${MODEL_ITERS:-10}"
ATOM_HIDDEN="${ATOM_HIDDEN:-32}"
LOGICAL_DIMENSION="${LOGICAL_DIMENSION:-32}"
TNORM="${TNORM:-godel}"
TNORM_WFS="${TNORM_WFS:-godel}"
LR="${LR:-1e-3}"
SEED="${SEED:-7}"
LOG_ROOT=""
CLINGO_PATH=""
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- extra-args]

Trains NDProp on a random ASP benchmark family.

Experiment:
      --family NAME          Random ASP family: n2l or 3lp (default: ${FAMILY})
      --difficulty NAME      easy|medium|hard training split (default: ${DIFFICULTY})
      --benchmark-root PATH  Root containing train_*/val_*/test_* splits
      --device DEVICE        Compute device (default: ${DEVICE})

Optimization and model:
      --epochs N             Training epochs (default: ${EPOCHS})
      --batch-size N         Batch size (default: ${BATCH_SIZE})
      --model-iters N        Decision-propagation iterations (default: ${MODEL_ITERS})
      --atom-hidden N        Decision controller hidden size (default: ${ATOM_HIDDEN})
      --logical-dimension N  Logical feature channels (default: ${LOGICAL_DIMENSION})
      --tnorm NAME           product|lukasiewicz|godel (default: ${TNORM})
      --tnorm-wfs NAME       product|lukasiewicz|godel (default: ${TNORM_WFS})
      --lr VALUE             Learning rate (default: ${LR})
      --seed N               RNG seed (default: ${SEED})

Output and tools:
      --log-root PATH        Output root
      --clingo-path PATH     clingo executable
  -h, --help                 Show this message and exit

Advanced arguments after '--' are passed unchanged to src/main.py.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --family) FAMILY="$2"; shift 2 ;;
    --difficulty) DIFFICULTY="$2"; shift 2 ;;
    --benchmark-root) BENCHMARK_ROOT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --model-iters) MODEL_ITERS="$2"; shift 2 ;;
    --atom-hidden) ATOM_HIDDEN="$2"; shift 2 ;;
    --logical-dimension) LOGICAL_DIMENSION="$2"; shift 2 ;;
    --tnorm) TNORM="$2"; shift 2 ;;
    --tnorm-wfs) TNORM_WFS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --clingo-path) CLINGO_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

case "${FAMILY}" in
  n2l) DEFAULT_BENCHMARK_ROOT="${PROJECT_ROOT}/benchmarks/n2l" ;;
  3lp) DEFAULT_BENCHMARK_ROOT="${PROJECT_ROOT}/benchmarks/3lp" ;;
  *) echo "Unknown --family '${FAMILY}' (expected n2l or 3lp)." >&2; exit 1 ;;
esac
case "${DIFFICULTY}" in
  easy|medium|hard) ;;
  *) echo "Unknown --difficulty '${DIFFICULTY}' (expected easy, medium, or hard)." >&2; exit 1 ;;
esac

BENCHMARK_ROOT="${BENCHMARK_ROOT:-${DEFAULT_BENCHMARK_ROOT}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/runs/random_asp/${FAMILY}/${DIFFICULTY}}"
TRAIN_DIR="${BENCHMARK_ROOT}/train_${DIFFICULTY}"
EVAL_DIR="${BENCHMARK_ROOT}/val_${DIFFICULTY}"

if [[ ! -d "${TRAIN_DIR}" || ! -d "${EVAL_DIR}" ]]; then
  echo "Expected benchmark splits under ${BENCHMARK_ROOT}." >&2
  echo "Generate them with: ${PYTHON_BIN} src/generate_random_asp.py --family ${FAMILY}" >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/src/main.py"
  --device "${DEVICE}"
  --seed "${SEED}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --batch-size "${BATCH_SIZE}"
  --model-iters "${MODEL_ITERS}"
  --atom-hidden "${ATOM_HIDDEN}"
  --logical-dimension "${LOGICAL_DIMENSION}"
  --tnorm "${TNORM}"
  --tnorm-wfs "${TNORM_WFS}"
  --train-dir "${TRAIN_DIR}"
  --eval-dir "${EVAL_DIR}"
  --run-dir "${LOG_ROOT}/run_$(date +%Y%m%d_%H%M%S)"
)
if [[ -n "${CLINGO_PATH}" ]]; then
  CMD+=(--clingo-path "${CLINGO_PATH}")
fi
CMD+=("${EXTRA_ARGS[@]}")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
