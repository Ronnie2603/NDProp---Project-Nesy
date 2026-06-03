#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

TASKS=(apply2x2 addition2 add2x2 member3 member5 addition)
REPEATS=3
BASE_SEED=7
DEVICE="${DEVICE:-cuda:3}"
CUDA_DEVICES=(${CUDA_DEVICES:-0 1 2 3})
PARALLEL="${PARALLEL:-4}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/runs_nesy/mnist_reasoning}"
RESULTS_CSV="${LOG_ROOT}/results.csv"
RESULTS_LOCK="${RESULTS_CSV}.lock"

usage() {
  cat <<EOF
Usage: $(basename "$0")

Runs the six neuro-symbolic MNIST reasoning experiments from the paper.

Configuration is controlled through environment variables:
  DEVICE=${DEVICE}
  CUDA_DEVICES="${CUDA_DEVICES[*]}"
  PARALLEL=${PARALLEL}
  LOG_ROOT=${LOG_ROOT}
  PYTHON_BIN=${PYTHON_BIN}
EOF
}

if [[ $# -eq 1 && "$1" == "--help" ]]; then
  usage
  exit 0
elif [[ $# -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  usage
  exit 1
fi

mkdir -p "${LOG_ROOT}"
if [[ ! -f "${RESULTS_CSV}" ]]; then
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
  "${PROJECT_ROOT}/scripts/run_mnist_task.sh" \
    --dataset "${task}" \
    --log-root "${task_log_root}" \
    --seed "${seed}" \
    --device "${device}"

  run_dir=$(ls -td "${task_log_root}"/run_* | head -n1)
  if [[ -z "${run_dir}" ]]; then
    echo "Failed to locate run directory under ${task_log_root}" >&2
    exit 1
  fi

  line=$(TASK="${task}" RUN_DIR="${run_dir}" REPEAT="${rep}" SEED="${seed}" "${PYTHON_BIN}" - <<'PY'
import csv
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
task = os.environ["TASK"]

metric_map = {
    "addition": "sum_acc",
    "addition2": "sum_acc",
    "add2x2": "total_acc",
    "member3": "member_acc",
    "member5": "member_acc",
    "apply2x2": "total_acc",
}
metric_name = metric_map[task]

def read_rows(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)

def last_epoch_value(rows, column):
    last_epoch = None
    last_value = None
    for row in rows:
        epoch = row.get("epoch", "")
        if epoch.isdigit():
            ep = int(epoch)
            val = row.get(column, "")
            if val != "":
                last_epoch = ep
                last_value = float(val)
    return last_value, last_epoch

def best_epoch_value(rows, column):
    best_epoch = None
    best_value = None
    for row in rows:
        epoch = row.get("epoch", "")
        if epoch.isdigit():
            val = row.get(column, "")
            if val != "":
                value = float(val)
                if best_value is None or value > best_value:
                    best_value = value
                    best_epoch = int(epoch)
    return best_value, best_epoch

def value_at_epoch(rows, column, epoch):
    if epoch is None:
        return None
    for row in rows:
        if row.get("epoch") == str(epoch):
            val = row.get(column, "")
            return float(val) if val != "" else None
    return None

def total_train_time(rows):
    for row in rows:
        if row.get("epoch") == "total":
            val = row.get("train", "")
            return float(val) if val != "" else None
    total = 0.0
    for row in rows:
        epoch = row.get("epoch", "")
        if epoch.isdigit():
            val = row.get("train", "")
            if val != "":
                total += float(val)
    return total if total > 0 else None

# Training/eval time
time_rows = read_rows(run_dir / "time.csv")
total_train = total_train_time(time_rows)
eval_time, _ = last_epoch_value(time_rows, "eval")

# MNIST digit accuracy
if task in ("addition", "addition2", "add2x2", "member3", "member5"):
    mnist_metric = "mnist_test_acc"
else:
    mnist_metric = "op_test_acc"
mnist_rows = read_rows(run_dir / f"{mnist_metric}.csv")
mnist_best, mnist_best_epoch = best_epoch_value(mnist_rows, "eval")

# Task-specific inference accuracy
task_rows = read_rows(run_dir / f"{metric_name}.csv")
inference_acc = value_at_epoch(task_rows, "eval", mnist_best_epoch)

def fmt(value):
    return "" if value is None else f"{value:.6f}"

fields = [
    task,
    os.environ.get("REPEAT", ""),
    os.environ.get("SEED", ""),
    str(run_dir),
    fmt(total_train),
    fmt(eval_time),
    fmt(mnist_best),
    fmt(inference_acc),
    metric_name,
]
print(",".join(fields))
PY
)
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
