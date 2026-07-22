#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

TASKS=(addition addition2 add2x2 apply2x2 member3 member5)
REPEATS=3
BASE_SEED=7
DEVICE="${DEVICE:-cuda:3}"
CUDA_DEVICES=(${CUDA_DEVICES:-0 1 2 3})
PARALLEL=(${PARALLEL:-1})
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/runs_nesy/neuro_symbolic_tasks}"
RESULTS_CSV="${RESULTS_CSV:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs the six neuro-symbolic task experiments from the paper.

Experiment matrix:
      --tasks NAME ...       Tasks (default: ${TASKS[*]})
      --repeats N            Repeats per task (default: ${REPEATS})
      --base-seed N          First repeat seed (default: ${BASE_SEED})
      --device DEVICE        Device used when running sequentially
                             (default: ${DEVICE})
      --cuda-devices ID ...  CUDA IDs used for parallel jobs
                             (default: ${CUDA_DEVICES[*]})
      --parallel N           Concurrent jobs (default: ${PARALLEL})

Output:
      --log-root PATH        Run output root (default: ${LOG_ROOT})
      --results-csv PATH     Per-run results CSV (default: LOG_ROOT/results.csv)
      --keep-results  Append this run to an existing results.csv instead of
                      resetting it first.
  -h, --help                 Show this message and exit.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)
      shift
      TASKS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do TASKS+=("$1"); shift; done
      ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --base-seed) BASE_SEED="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --cuda-devices)
      shift
      CUDA_DEVICES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do CUDA_DEVICES+=("$1"); shift; done
      ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --results-csv) RESULTS_CSV="$2"; shift 2 ;;
    --keep-results)
      KEEP_RESULTS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ${#TASKS[@]} -eq 0 ]]; then
  echo "--tasks must contain at least one task." >&2
  exit 1
fi
if [[ "${PARALLEL}" -gt 1 && ${#CUDA_DEVICES[@]} -eq 0 ]]; then
  echo "--cuda-devices must contain at least one ID for parallel runs." >&2
  exit 1
fi
RESULTS_CSV="${RESULTS_CSV:-${LOG_ROOT}/results.csv}"
RESULTS_LOCK="${RESULTS_CSV}.lock"

mkdir -p "${LOG_ROOT}" "$(dirname "${RESULTS_CSV}")"
# A results file represents one invocation of this script. Reset it here so a
# subsequent summarize command cannot mix the latest batch with earlier runs.
if [[ "${KEEP_RESULTS:-0}" != "1" && "${KEEP_RESULTS:-0}" != "true" ]]; then
  echo "task,repeat,seed,run_dir,total_train_time,eval_time_last_epoch,mnist_test_acc,inference_acc,metric_name" > "${RESULTS_CSV}"
elif [[ ! -f "${RESULTS_CSV}" ]]; then
  echo "task,repeat,seed,run_dir,total_train_time,eval_time_last_epoch,mnist_test_acc,inference_acc,metric_name" > "${RESULTS_CSV}"
fi

append_result() {
  local line="$1"
  if command -v flock >/dev/null 2>&1; then
    (
      flock -x 200
      printf '%s\n' "${line}" >> "${RESULTS_CSV}"
    ) 200>"${RESULTS_LOCK}"
  else
    printf '%s\n' "${line}" >> "${RESULTS_CSV}"
  fi
}

run_one() {
  local task="$1"
  local rep="$2"
  local seed="$3"
  local device="$4"
  local task_log_root="${LOG_ROOT}/${task}/seed_${seed}_rep_${rep}"
  local run_dir
  local line

  mkdir -p "${task_log_root}"
  bash "${SCRIPT_DIR}/task.sh" \
    --dataset "${task}" \
    --log-root "${task_log_root}" \
    --seed "${seed}" \
    --device "${device}"

  run_dir=$(ls -td "${task_log_root}"/run_* | head -n1)
  if [[ -z "${run_dir}" ]]; then
    echo "Failed to locate run directory under ${task_log_root}" >&2
    exit 1
  fi

  line="$(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/src/experiment_utils/neuro_symbolic_task_result_row.py" \
      --task "${task}" \
      --run-dir "${run_dir}" \
      --repeat "${rep}" \
      --seed "${seed}"
  )"
  append_result "${line}"
}

if [[ "${PARALLEL}" -le 1 ]]; then
  for task in "${TASKS[@]}"; do
    for rep in $(seq 1 "${REPEATS}"); do
      seed=$((BASE_SEED + rep - 1))
      run_one "${task}" "${rep}" "${seed}" "${DEVICE}"
    done
  done
  exit 0
fi

job_count=0
job_index=0
for task in "${TASKS[@]}"; do
  for rep in $(seq 1 "${REPEATS}"); do
    seed=$((BASE_SEED + rep - 1))
    cuda_id="${CUDA_DEVICES[$((job_index % ${#CUDA_DEVICES[@]}))]}"
    run_one "${task}" "${rep}" "${seed}" "cuda:${cuda_id}" &
    job_count=$((job_count + 1))
    job_index=$((job_index + 1))
    if [[ "${job_count}" -ge "${PARALLEL}" ]]; then
      wait -n
      job_count=$((job_count - 1))
    fi
  done
done
wait
