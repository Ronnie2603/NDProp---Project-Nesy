#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

FAMILY="both"
BENCHMARK_ROOT=""
CASES_DIRS=()
OUTPUT_ROOT="${PROJECT_ROOT}/baselines/random_asp_rdprop"
RESTARTS=(1 10 100)
UNIQUE_TRIALS="0"
UNIQUE_MAX_ATTEMPTS="10000"
SEED="7"
CLINGO_PATH=""
DLV_PATH=""
TIMESTAMP=""
DIFFICULTIES=(test_easy test_medium test_hard)

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs the RDProp random-decision baseline for the random ASP paper experiments.

Options:
      --family NAME          Random ASP family: n2l, 3lp, or both (default: ${FAMILY})
      --benchmark-root PATH  Root containing test_easy/test_medium/test_hard
      --cases-dir PATH       Explicit cases directory; can be supplied multiple times
      --output-root PATH     Output root (default: ${OUTPUT_ROOT})
      --restarts N ...       Restart counts to evaluate (default: ${RESTARTS[*]})
      --unique-trials        Require unique decision sequences per restart count
      --unique-max-attempts N
      --seed N               RNG seed (default: ${SEED})
      --clingo-path PATH     clingo executable
      --dlv-path PATH        DLV executable for WFS cross-checks
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
    --restarts)
      shift
      RESTARTS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        RESTARTS+=("$1")
        shift
      done
      ;;
    --unique-trials) UNIQUE_TRIALS="1"; shift ;;
    --unique-max-attempts) UNIQUE_MAX_ATTEMPTS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --clingo-path) CLINGO_PATH="$2"; shift 2 ;;
    --dlv-path) DLV_PATH="$2"; shift 2 ;;
    --timestamp) TIMESTAMP="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ${#RESTARTS[@]} -eq 0 ]]; then
  echo "--restarts must contain at least one value." >&2
  exit 1
fi

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

  timestamp_base="${TIMESTAMP:-rdprop_${family}_$(date +%Y%m%d_%H%M%S)}"

  for restarts in "${RESTARTS[@]}"; do
    for cases_dir in "${family_cases[@]}"; do
      if [[ ! -d "${cases_dir}" ]]; then
        echo "Cases directory does not exist: ${cases_dir}" >&2
        exit 1
      fi
      label="$(basename "${cases_dir}")"
      output_dir="${OUTPUT_ROOT}/${timestamp_base}_r${restarts}_${label}"
      echo "=== RDProp family=${family} restarts=${restarts} split=${label} ==="
      CMD=(
        "${PYTHON_BIN}" "${PROJECT_ROOT}/src/dprop_baseline.py"
        --cases-dir "${cases_dir}"
        --trials "${restarts}"
        --seed "${SEED}"
        --output-dir "${output_dir}"
      )
      if [[ "${UNIQUE_TRIALS}" == "1" ]]; then
        CMD+=(--unique-trials --unique-max-attempts "${UNIQUE_MAX_ATTEMPTS}")
      fi
      if [[ -n "${CLINGO_PATH}" ]]; then
        CMD+=(--clingo-path "${CLINGO_PATH}")
      fi
      if [[ -n "${DLV_PATH}" ]]; then
        CMD+=(--dlv-path "${DLV_PATH}")
      fi
      "${CMD[@]}"
    done
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
