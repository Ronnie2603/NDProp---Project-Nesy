#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [command] [options]

Commands:
  generate   Generate random ASP benchmark splits inside benchmarks/.
  train      Train NDProp on one benchmark family.
  random     Run the random candidate-set baseline.
  baseline   Run the RDProp random-decision baseline.
  analyze    Analyze a trained NDProp run by difficulty.
  iteration  Plot accuracy by propagation iteration.
  compare    Plot NDProp, random baseline, and RDProp by difficulty.
  analysis-cases
             Analyze bundled or custom ASP cases with a trained NDProp run.
  all        For each selected family: generate/verify splits, train NDProp,
             run random/RDProp baselines, analysis, and comparison plots.

The default command is "all". Use '<command> --help' for command-specific
options. Options for the all command are:

Experiment selection:
      --family n2l|3lp|both  Families to run (default: both)
      --difficulty NAME      Training difficulty; repeatable
                             (default: easy and medium)
      --device DEVICE        Compute device (default: cpu)
      --clingo-path PATH     clingo executable

Baselines and optional analysis:
      --restarts N ...       Baseline restart counts (default: 1 10 100)
      --analysis-cases       Analyze analysis_cases/ after each trained run
      --analysis-cases-dir PATH
                             Curated ASP cases directory (default: analysis_cases)
      --analysis-threshold X Prediction threshold for those cases (default: 0.5)

Pipeline control:
      --skip-generate        Skip benchmark generation and verification
      --keep-results         Preserve existing comparison CSV/PNG artifacts
  -h, --help                 Show this message and exit
EOF
}

analysis_cases_usage() {
  cat <<EOF
Usage: $(basename "$0") analysis-cases --run-dir PATH [options]

Analyzes bundled or custom ASP cases with a trained NDProp checkpoint.

Options:
      --run-dir PATH       Directory containing config.json and best.pt (required)
      --cases-dir PATH     Directory containing .lp programs
      --device DEVICE      Compute device (default: cpu)
      --threshold VALUE    Prediction threshold (default: 0.5)
      --model-iters N      Override model iterations (wrapper default: 50)
      --clingo-path PATH   clingo executable
      --timestamp TAG      Output tag
      --no-visualize       Skip plots and visual artifacts
  -h, --help               Show this message and exit
EOF
}

run_all() {
  local family_arg="both"
  local difficulties=(easy medium)
  local device="${DEVICE:-cpu}"
  local clingo_path=""
  local skip_generate=false
  local keep_results=false
  local run_analysis_cases=false
  local analysis_cases_dir="${PROJECT_ROOT}/analysis_cases"
  local analysis_threshold="0.5"
  local restarts=(1 10 100)

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --family) family_arg="$2"; shift 2 ;;
      --difficulty)
        difficulties=()
        while [[ $# -gt 1 && "$2" != --* ]]; do
          difficulties+=("$2")
          shift 2
          if [[ $# -eq 0 || "$1" != "--difficulty" ]]; then
            break
          fi
        done
        ;;
      --device) device="$2"; shift 2 ;;
      --clingo-path) clingo_path="$2"; shift 2 ;;
      --analysis-cases) run_analysis_cases=true; shift ;;
      --analysis-cases-dir) analysis_cases_dir="$2"; shift 2 ;;
      --analysis-threshold) analysis_threshold="$2"; shift 2 ;;
      --skip-generate) skip_generate=true; shift ;;
      --keep-results) keep_results=true; shift ;;
      --restarts)
        shift
        restarts=()
        while [[ $# -gt 0 && "$1" != --* ]]; do
          restarts+=("$1")
          shift
        done
        ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown all option: $1" >&2; usage; exit 1 ;;
    esac
  done

  local families=()
  case "${family_arg}" in
    both) families=(n2l 3lp) ;;
    n2l|3lp) families=("${family_arg}") ;;
    *) echo "Unknown --family '${family_arg}' (expected n2l, 3lp, or both)." >&2; exit 1 ;;
  esac
  if [[ ${#difficulties[@]} -eq 0 ]]; then
    echo "--difficulty must contain at least one value." >&2
    exit 1
  fi
  for difficulty in "${difficulties[@]}"; do
    case "${difficulty}" in
      easy|medium|hard) ;;
      *) echo "Unknown --difficulty '${difficulty}' (expected easy, medium, or hard)." >&2; exit 1 ;;
    esac
  done
  if [[ ${#restarts[@]} -eq 0 ]]; then
    echo "--restarts must contain at least one value." >&2
    exit 1
  fi

  if [[ "${keep_results}" != true ]]; then
    for family in "${families[@]}"; do
      comparison_root="${PROJECT_ROOT}/runs/random_asp/${family}"
      comparison_base="${comparison_root}/random_asp_${family}_method_comparison"
      if [[ -f "${comparison_base}.csv" || -f "${comparison_base}.png" ]]; then
        echo "Resetting random ASP comparison artifacts for ${family}: ${comparison_base}.{csv,png}"
        rm -f "${comparison_base}.csv" "${comparison_base}.png"
      fi
    done
  fi

  local clingo_args=()
  if [[ -n "${clingo_path}" ]]; then
    clingo_args=(--clingo-path "${clingo_path}")
  fi

  default_benchmark_root() {
    case "$1" in
      n2l) printf '%s\n' "${PROJECT_ROOT}/benchmarks/n2l" ;;
      3lp) printf '%s\n' "${PROJECT_ROOT}/benchmarks/3lp" ;;
      *) return 1 ;;
    esac
  }

  for family in "${families[@]}"; do
    local benchmark_root
    benchmark_root="$(default_benchmark_root "${family}")"
    echo "=== Random ASP family=${family}: prepare benchmarks ==="
    if [[ "${skip_generate}" != true ]]; then
      "${PYTHON_BIN}" "${PROJECT_ROOT}/src/generate_random_asp.py" \
        --family "${family}" \
        "${clingo_args[@]}"
    fi
    if [[ ! -d "${benchmark_root}" ]]; then
      echo "Expected benchmark root does not exist: ${benchmark_root}" >&2
      if [[ "${skip_generate}" == true ]]; then
        echo "Remove --skip-generate or generate the benchmark splits first." >&2
      fi
      exit 1
    fi

    for difficulty in "${difficulties[@]}"; do
      echo "=== Random ASP family=${family} difficulty=${difficulty}: train NDProp ==="
      bash "${SCRIPT_DIR}/train.sh" \
        --family "${family}" \
        --difficulty "${difficulty}" \
        --benchmark-root "${benchmark_root}" \
        --device "${device}" \
        "${clingo_args[@]}"

      local run_root="${PROJECT_ROOT}/runs/random_asp/${family}/${difficulty}"
      local run_dir
      run_dir="$(find "${run_root}" -maxdepth 1 -type d -name 'run_*' -print 2>/dev/null | sort -r | head -n1 || true)"
      if [[ -z "${run_dir}" ]]; then
        echo "Failed to locate trained run directory under ${run_root}." >&2
        exit 1
      fi

      echo "=== Random ASP family=${family} difficulty=${difficulty}: difficulty analysis ==="
      bash "${SCRIPT_DIR}/analyze.sh" \
        --family "${family}" \
        --benchmark-root "${benchmark_root}" \
        --run-dir "${run_dir}" \
        --device "${device}" \
        "${clingo_args[@]}"

      echo "=== Random ASP family=${family} difficulty=${difficulty}: iteration analysis ==="
      bash "${SCRIPT_DIR}/iteration_scaling.sh" \
        --family "${family}" \
        --benchmark-root "${benchmark_root}" \
        --run-dir "${run_dir}" \
        --device "${device}" \
        "${clingo_args[@]}"

      if [[ "${run_analysis_cases}" == true ]]; then
        echo "=== Random ASP family=${family} difficulty=${difficulty}: curated analysis cases ==="
        "${PYTHON_BIN}" "${PROJECT_ROOT}/src/analyze.py" \
          --run-dir "${run_dir}" \
          --device "${device}" \
          --cases-dir "${analysis_cases_dir}" \
          --threshold "${analysis_threshold}" \
          --model-iters 50 \
          "${clingo_args[@]}"
      fi
    done

    echo "=== Random ASP family=${family}: random baseline ==="
    bash "${SCRIPT_DIR}/random_baseline.sh" \
      --family "${family}" \
      --benchmark-root "${benchmark_root}" \
      "${clingo_args[@]}"

    echo "=== Random ASP family=${family}: RDProp baseline ==="
    bash "${SCRIPT_DIR}/rdprop_baseline.sh" \
      --family "${family}" \
      --benchmark-root "${benchmark_root}" \
      --restarts "${restarts[@]}" \
      "${clingo_args[@]}"

    echo "=== Random ASP family=${family}: method comparison ==="
    bash "${SCRIPT_DIR}/compare_methods.sh" \
      --family "${family}" \
      --restarts "${restarts[@]}"
  done
}

cmd="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi

case "${cmd}" in
  generate)
    exec "${PYTHON_BIN}" "${PROJECT_ROOT}/src/generate_random_asp.py" "$@"
    ;;
  train)
    exec bash "${SCRIPT_DIR}/train.sh" "$@"
    ;;
  random)
    exec bash "${SCRIPT_DIR}/random_baseline.sh" "$@"
    ;;
  baseline)
    exec bash "${SCRIPT_DIR}/rdprop_baseline.sh" "$@"
    ;;
  analyze)
    exec bash "${SCRIPT_DIR}/analyze.sh" "$@"
    ;;
  iteration)
    exec bash "${SCRIPT_DIR}/iteration_scaling.sh" "$@"
    ;;
  compare)
    exec bash "${SCRIPT_DIR}/compare_methods.sh" "$@"
    ;;
  analysis-cases|cases)
    if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
      analysis_cases_usage
      exit 0
    fi
    exec "${PYTHON_BIN}" "${PROJECT_ROOT}/src/analyze.py" \
      --cases-dir "${PROJECT_ROOT}/analysis_cases" \
      --model-iters 50 \
      "$@"
    ;;
  all)
    run_all "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage
    exit 1
    ;;
esac
