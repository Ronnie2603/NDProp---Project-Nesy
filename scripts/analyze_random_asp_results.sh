#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOP_LEVEL_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

FAMILY="n2l"
BENCHMARK_ROOT=""
RUN_DIR=""
DEVICE="cpu"
THRESHOLD="0.5"
CLINGO_PATH=""
TIMESTAMP=""
MODEL_ITERS="50"
CASES_DIRS=()
DIFFICULTIES=(test_easy test_medium test_hard)

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Analyzes an NDProp checkpoint on random ASP benchmark splits.

Options:
      --family NAME          Random ASP family: n2l or 3lp (default: ${FAMILY})
      --benchmark-root PATH  Root containing test_easy/test_medium/test_hard
      --cases-dir PATH       Explicit cases directory; can be supplied multiple times
      --run-dir PATH         NDProp run directory; defaults to latest runs/random_asp/*/run_*
      --device DEVICE        Compute device (default: ${DEVICE})
      --threshold VALUE      Prediction threshold (default: ${THRESHOLD})
      --model-iters N        Model iterations for analysis (default: ${MODEL_ITERS})
      --clingo-path PATH     clingo executable
      --timestamp TAG        Analysis timestamp/tag prefix
  -h, --help                 Show this message and exit
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --family) FAMILY="$2"; shift 2 ;;
    --benchmark-root) BENCHMARK_ROOT="$2"; shift 2 ;;
    --cases-dir) CASES_DIRS+=("$2"); shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --model-iters) MODEL_ITERS="$2"; shift 2 ;;
    --clingo-path) CLINGO_PATH="$2"; shift 2 ;;
    --timestamp) TIMESTAMP="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

case "${FAMILY}" in
  n2l) DEFAULT_BENCHMARK_ROOT="${TOP_LEVEL_ROOT}/benchmarks/n2l" ;;
  3lp) DEFAULT_BENCHMARK_ROOT="${TOP_LEVEL_ROOT}/benchmarks/3lp" ;;
  *) echo "Unknown --family '${FAMILY}' (expected n2l or 3lp)." >&2; exit 1 ;;
esac

BENCHMARK_ROOT="${BENCHMARK_ROOT:-${DEFAULT_BENCHMARK_ROOT}}"
if [[ ${#CASES_DIRS[@]} -eq 0 ]]; then
  for difficulty in "${DIFFICULTIES[@]}"; do
    CASES_DIRS+=("${BENCHMARK_ROOT}/${difficulty}")
  done
fi

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(ls -td "${PROJECT_ROOT}"/runs/random_asp/"${FAMILY}"/run_* 2>/dev/null | head -n1 || true)"
  if [[ -z "${RUN_DIR}" ]]; then
    echo "Failed to locate a run directory. Provide --run-dir." >&2
    exit 1
  fi
  echo "Auto-selected run directory: ${RUN_DIR}"
fi

TIMESTAMP_BASE="${TIMESTAMP:-random_asp_${FAMILY}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_DIR}/analysis"

SUMMARY_ARGS=()
REPORT_LINES=()

for cases_dir in "${CASES_DIRS[@]}"; do
  if [[ ! -d "${cases_dir}" ]]; then
    echo "Cases directory does not exist: ${cases_dir}" >&2
    exit 1
  fi
  label="$(basename "${cases_dir}")"
  timestamp="${TIMESTAMP_BASE}_${label}"
  echo "=== Analyze family=${FAMILY} split=${label} ==="
  CMD=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/src/analyze.py"
    --run-dir "${RUN_DIR}"
    --device "${DEVICE}"
    --cases-dir "${cases_dir}"
    --threshold "${THRESHOLD}"
    --model-iters "${MODEL_ITERS}"
    --timestamp "${timestamp}"
    --no-visualize
  )
  if [[ -n "${CLINGO_PATH}" ]]; then
    CMD+=(--clingo-path "${CLINGO_PATH}")
  fi
  "${CMD[@]}"

  eval_path="${RUN_DIR}/analysis/${timestamp}/eval.json"
  if [[ ! -f "${eval_path}" ]]; then
    echo "Evaluation summary not found at ${eval_path}" >&2
    continue
  fi

  read -r success total accuracy <<<"$("${PYTHON_BIN}" - "${eval_path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
values = list(data.values())
total = len(values)
success = sum(1 for value in values if value)
acc = (success / total * 100.0) if total else 0.0
print(success, total, f"{acc:.2f}")
PY
)"
  SUMMARY_ARGS+=("${label}" "${success}" "${total}")
  REPORT_LINES+=("${label}: ${success}/${total} cases (${accuracy}% exact)")
done

SUMMARY_PLOT="${RUN_DIR}/analysis/${TIMESTAMP_BASE}_difficulty_accuracy.png"
if [[ ${#SUMMARY_ARGS[@]} -gt 0 ]]; then
  PLOT_PATH="$("${PYTHON_BIN}" - "${SUMMARY_PLOT}" "${SUMMARY_ARGS[@]}" <<'PY'
import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402

plot_path = Path(sys.argv[1]).resolve()
triples = sys.argv[2:]
labels, pretty_labels, accuracies, pairs = [], [], [], []
for idx in range(0, len(triples), 3):
    label = triples[idx]
    success = int(triples[idx + 1])
    total = int(triples[idx + 2])
    acc = success / total if total else 0.0
    pretty = label[len("test_"):] if label.startswith("test_") else label
    labels.append(label)
    pretty_labels.append(pretty.capitalize())
    accuracies.append(acc)
    pairs.append((success, total))

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(pretty_labels, accuracies, color="#4c72b0", alpha=0.85)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Exact-match accuracy")
ax.set_xlabel("Difficulty")
ax.set_title("NDProp random ASP accuracy by difficulty")
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
for bar, (success, total), acc in zip(bars, pairs, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc*100:.1f}%\n({success}/{total})", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
plot_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(plot_path, dpi=150)
print(plot_path)
PY
)"
else
  PLOT_PATH=""
fi

echo "=== Difficulty summary ==="
if [[ ${#REPORT_LINES[@]} -eq 0 ]]; then
  echo "No evaluation summaries were generated."
else
  for line in "${REPORT_LINES[@]}"; do
    echo " - ${line}"
  done
  if [[ -n "${PLOT_PATH}" ]]; then
    echo "Summary plot saved to: ${PLOT_PATH}"
  fi
fi
