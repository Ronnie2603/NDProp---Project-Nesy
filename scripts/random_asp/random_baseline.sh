#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

FAMILY="both"
BENCHMARK_ROOT=""
CASES_DIRS=()
OUTPUT_ROOT="${PROJECT_ROOT}/baselines/random_asp_random"
SAMPLES="100"
INCLUDE_PROB="0.5"
SEED="7"
CLINGO_PATH=""
TIMESTAMP=""
DIFFICULTIES=(test_easy test_medium test_hard)

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs the random candidate-set baseline for random ASP benchmark splits.

Options:
      --family NAME          Random ASP family: n2l, 3lp, or both (default: ${FAMILY})
      --benchmark-root PATH  Root containing test_easy/test_medium/test_hard
      --cases-dir PATH       Explicit cases directory; can be supplied multiple times
      --output-root PATH     Output root (default: ${OUTPUT_ROOT})
      --samples N            Random samples per program (default: ${SAMPLES})
      --include-prob P       Atom inclusion probability (default: ${INCLUDE_PROB})
      --seed N               RNG seed (default: ${SEED})
      --clingo-path PATH     clingo executable
      --timestamp TAG        Output timestamp/tag prefix
  -h, --help                 Show this message and exit
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --family) FAMILY="$2"; shift 2 ;;
    --benchmark-root) BENCHMARK_ROOT="$2"; shift 2 ;;
    --cases-dir) CASES_DIRS+=("$2"); shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    --include-prob) INCLUDE_PROB="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --clingo-path) CLINGO_PATH="$2"; shift 2 ;;
    --timestamp) TIMESTAMP="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}"

run_family() {
  local family="$1"
  local benchmark_root="${BENCHMARK_ROOT}"
  local timestamp_base
  local family_cases=()

  case "${family}" in
    n2l) benchmark_root="${benchmark_root:-${PROJECT_ROOT}/benchmarks/n2l}" ;;
    3lp) benchmark_root="${benchmark_root:-${PROJECT_ROOT}/benchmarks/3lp}" ;;
    *) echo "Unknown --family '${family}' (expected n2l, 3lp, or both)." >&2; exit 1 ;;
  esac

  if [[ ${#CASES_DIRS[@]} -eq 0 ]]; then
    for difficulty in "${DIFFICULTIES[@]}"; do
      family_cases+=("${benchmark_root}/${difficulty}")
    done
  else
    family_cases=("${CASES_DIRS[@]}")
  fi

  timestamp_base="${TIMESTAMP:-random_${family}_$(date +%Y%m%d_%H%M%S)}"

  for cases_dir in "${family_cases[@]}"; do
    if [[ ! -d "${cases_dir}" ]]; then
      echo "Cases directory does not exist: ${cases_dir}" >&2
      exit 1
    fi
    label="$(basename "${cases_dir}")"
    output_dir="${OUTPUT_ROOT}/${timestamp_base}_${label}"
    echo "=== Random baseline family=${family} split=${label} ==="
    CMD=(
      "${PYTHON_BIN}" "${PROJECT_ROOT}/src/random_stable_baseline.py"
      --cases-dir "${cases_dir}"
      --samples "${SAMPLES}"
      --include-prob "${INCLUDE_PROB}"
      --seed "${SEED}"
      --output-dir "${output_dir}"
    )
    if [[ -n "${CLINGO_PATH}" ]]; then
      CMD+=(--clingo-path "${CLINGO_PATH}")
    fi
    "${CMD[@]}"
  done
}

case "${FAMILY}" in
  both)
    if [[ -n "${BENCHMARK_ROOT}" || ${#CASES_DIRS[@]} -gt 0 ]]; then
      echo "--benchmark-root and --cases-dir require --family n2l or --family 3lp." >&2
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
