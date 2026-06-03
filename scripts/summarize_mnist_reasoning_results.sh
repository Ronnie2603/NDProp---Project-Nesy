#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/runs_nesy/mnist_reasoning}"
RESULTS_CSV="${RESULTS_CSV:-${LOG_ROOT}/results.csv}"
SUMMARY_CSV="${SUMMARY_CSV:-${LOG_ROOT}/summary.csv}"

usage() {
  cat <<EOF
Usage: $(basename "$0")

Summarizes run_mnist_reasoning_experiments.sh results.

Configuration is controlled through environment variables:
  RESULTS_CSV=${RESULTS_CSV}
  SUMMARY_CSV=${SUMMARY_CSV}
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

if [[ ! -f "${RESULTS_CSV}" ]]; then
  echo "Missing results file: ${RESULTS_CSV}" >&2
  exit 1
fi

RESULTS_CSV="${RESULTS_CSV}" SUMMARY_CSV="${SUMMARY_CSV}" "${PYTHON_BIN}" - <<'PY'
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

results_csv = Path(os.environ["RESULTS_CSV"])
summary_csv = Path(os.environ["SUMMARY_CSV"])

def to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None

fields = [
    "total_train_time",
    "eval_time_last_epoch",
    "mnist_test_acc",
    "inference_acc",
]

data = defaultdict(lambda: {key: [] for key in fields})

with results_csv.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        task = row.get("task", "")
        if not task:
            continue
        for key in fields:
            value = to_float(row.get(key, ""))
            if value is not None:
                data[task][key].append(value)

def mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)

with summary_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["task", "metric", "mean", "std", "n"])
    for task in sorted(data.keys()):
        for key in fields:
            values = data[task][key]
            mean, std = mean_std(values)
            writer.writerow(
                [
                    task,
                    key,
                    "" if mean is None else f"{mean:.6f}",
                    "" if std is None else f"{std:.6f}",
                    len(values),
                ]
            )

print(f"Wrote {summary_csv}")
PY
