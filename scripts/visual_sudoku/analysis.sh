#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

OUTPUT=""
NDPROP_EPOCH="best"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Aggregates Visual Sudoku NDProp and baseline results across configured seeds.
Writes a two-panel accuracy plot with ±1 sample-standard-deviation shading,
a summary CSV, and a raw-samples CSV with source provenance.

Options:
      --log-root PATH        Run input root (default: ${LOG_ROOT})
      --pcts VALUE ...       Label fractions (default: ${PCTS[*]})
      --seeds N ...          Seeds (default: ${SEEDS[*]})
      --output PATH          Output PNG (default: LOG_ROOT/visual_sudoku_analysis.png)
      --ndprop-epoch MODE    best or last (default: ${NDPROP_EPOCH})
  -h, --help                 Show this message and exit
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --pcts)
      shift
      PCTS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do PCTS+=("$1"); shift; done
      ;;
    --seeds)
      shift
      SEEDS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done
      ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --ndprop-epoch) NDPROP_EPOCH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ${#PCTS[@]} -eq 0 || ${#SEEDS[@]} -eq 0 ]]; then
  echo "--pcts and --seeds must each contain at least one value." >&2
  exit 1
fi
OUTPUT="${OUTPUT:-${LOG_ROOT}/visual_sudoku_analysis.png}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/src/experiment_utils/plot_visual_sudoku.py" \
  --log-root "${LOG_ROOT}" \
  --output "${OUTPUT}" \
  --pcts "${PCTS[@]}" \
  --seeds "${SEEDS[@]}" \
  --ndprop-epoch "${NDPROP_EPOCH}"
