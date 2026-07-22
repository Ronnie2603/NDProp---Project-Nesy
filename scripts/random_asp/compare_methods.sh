#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ndprop_matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/ndprop_cache}"

FAMILY="both"
OUTPUT=""
RESTARTS=(1 10 100)
TRAIN_DIFFICULTIES=(easy medium)
TRAIN_DIFFICULTIES_SET=false
RANDOM_ROOTS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Plots NDProp, random baseline, and RDProp accuracy by test difficulty.

Options:
      --family NAME          Random ASP family: n2l, 3lp, or both (default: ${FAMILY})
      --output PATH          Output PNG path
      --restarts N ...       RDProp restart counts (default: ${RESTARTS[*]})
      --train-difficulty D   NDProp training difficulty to include; can repeat
                             (default: ${TRAIN_DIFFICULTIES[*]})
      --random-root PATH     Random-baseline output root; can repeat
  -h, --help                 Show this message and exit
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --family) FAMILY="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --restarts)
      shift
      RESTARTS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        RESTARTS+=("$1")
        shift
      done
      ;;
    --train-difficulty)
      if [[ "${TRAIN_DIFFICULTIES_SET}" != true ]]; then
        TRAIN_DIFFICULTIES=()
        TRAIN_DIFFICULTIES_SET=true
      fi
      TRAIN_DIFFICULTIES+=("$2")
      shift 2
      ;;
    --random-root)
      RANDOM_ROOTS+=("$2")
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

run_family() {
  local family="$1"
  CMD=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/src/experiment_utils/plot_random_asp_methods.py"
    --family "${family}"
    --restarts "${RESTARTS[@]}"
  )

  if [[ -n "${OUTPUT}" ]]; then
    CMD+=(--output "${OUTPUT}")
  fi

  for difficulty in "${TRAIN_DIFFICULTIES[@]}"; do
    CMD+=(--train-difficulty "${difficulty}")
  done

  for root in "${RANDOM_ROOTS[@]}"; do
    CMD+=(--random-root "${root}")
  done

  "${CMD[@]}"
}

case "${FAMILY}" in
  both)
    if [[ -n "${OUTPUT}" ]]; then
      echo "--output requires --family n2l or --family 3lp." >&2
      exit 1
    fi
    run_family n2l
    run_family 3lp
    ;;
  n2l|3lp)
    run_family "${FAMILY}"
    ;;
  *)
    echo "Unknown --family '${FAMILY}' (expected n2l, 3lp, or both)." >&2
    exit 1
    ;;
esac
