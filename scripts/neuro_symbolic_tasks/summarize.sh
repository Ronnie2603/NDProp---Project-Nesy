#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

TASKS=(apply2x2 addition2 add2x2 member3 member5 addition)
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/runs_nesy/neuro_symbolic_tasks}"
RESULTS_CSV="${RESULTS_CSV:-}"
SUMMARY_CSV="${SUMMARY_CSV:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--help]

Rebuilds results.csv from the most recent run in each task/seed/repeat
directory, then summarizes those results.

Options:
      --tasks NAME ...    Tasks to include (default: ${TASKS[*]})
      --log-root PATH     Run input root (default: ${LOG_ROOT})
      --results-csv PATH  Rebuilt results CSV (default: LOG_ROOT/results.csv)
      --summary-csv PATH  Summary CSV (default: LOG_ROOT/summary.csv)
  -h, --help  Show this message and exit.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)
      shift
      TASKS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do TASKS+=("$1"); shift; done
      ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --results-csv) RESULTS_CSV="$2"; shift 2 ;;
    --summary-csv) SUMMARY_CSV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ${#TASKS[@]} -eq 0 ]]; then
  echo "--tasks must contain at least one task." >&2
  exit 1
fi

RESULTS_CSV="${RESULTS_CSV:-${LOG_ROOT}/results.csv}"
SUMMARY_CSV="${SUMMARY_CSV:-${LOG_ROOT}/summary.csv}"

mkdir -p "$(dirname "${RESULTS_CSV}")"
results_tmp="$(mktemp "${RESULTS_CSV}.tmp.XXXXXX")"
trap 'rm -f -- "${results_tmp}"' EXIT

echo "task,repeat,seed,run_dir,total_train_time,eval_time_last_epoch,mnist_test_acc,inference_acc,metric_name" > "${results_tmp}"

run_count=0
for task in "${TASKS[@]}"; do
  for task_log_root in "${LOG_ROOT}/${task}"/seed_*_rep_*; do
    [[ -d "${task_log_root}" ]] || continue

    run_label="$(basename "${task_log_root}")"
    if [[ ! "${run_label}" =~ ^seed_(.+)_rep_(.+)$ ]]; then
      continue
    fi
    seed="${BASH_REMATCH[1]}"
    repeat="${BASH_REMATCH[2]}"
    run_dir="$(ls -td -- "${task_log_root}"/run_* 2>/dev/null | head -n1 || true)"
    [[ -n "${run_dir}" ]] || continue

    "${PYTHON_BIN}" "${PROJECT_ROOT}/src/experiment_utils/neuro_symbolic_task_result_row.py" \
      --task "${task}" \
      --run-dir "${run_dir}" \
      --repeat "${repeat}" \
      --seed "${seed}" >> "${results_tmp}"
    run_count=$((run_count + 1))
  done
done

if [[ "${run_count}" -eq 0 ]]; then
  echo "No neuro-symbolic task runs found under ${LOG_ROOT}" >&2
  exit 1
fi

mv -- "${results_tmp}" "${RESULTS_CSV}"
trap - EXIT
echo "Rebuilt ${RESULTS_CSV} from ${run_count} latest runs"

"${PYTHON_BIN}" "${PROJECT_ROOT}/src/experiment_utils/summarize_neuro_symbolic_tasks.py" \
  --results-csv "${RESULTS_CSV}" \
  --summary-csv "${SUMMARY_CSV}"
