#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOP_LEVEL_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
RUN_DIR=""
DEVICE="cpu"
THRESHOLD="0.5"
CLINGO_PATH=""
TIMESTAMP=""
MODEL_ITERS="50"
BENCHMARK_ROOT="$TOP_LEVEL_ROOT/benchmarks/random_data/datasets_1sm"
declare -a CASES_DIRS=()
declare -a DEFAULT_DIFFICULTIES=("test_easy" "test_medium" "test_hard")

usage() {
  cat <<'EOF'
Usage: analyze_rand.sh [--run-dir PATH] [--device DEVICE] [--cases-dir PATH]...
                       [--benchmark-root PATH] [--threshold VALUE]
                       [--clingo-path PATH] [--timestamp TAG]
                       [--model-iters COUNT]

Runs decision-based GNN analysis on the random benchmark splits (easy/medium/hard by
default), prints their exact-match accuracy and renders a summary plot at
RUN_DIR/analysis/random_difficulty_accuracy.png. Pass --cases-dir multiple times
to target custom directories instead of the default benchmark root layout.
EOF
}

if [[ $# -eq 1 && "$1" == "--help" ]]; then
  usage
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --cases-dir)
      CASES_DIRS+=("$2")
      shift 2
      ;;
    --benchmark-root)
      BENCHMARK_ROOT="$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    --clingo-path)
      CLINGO_PATH="$2"
      shift 2
      ;;
    --timestamp)
      TIMESTAMP="$2"
      shift 2
      ;;
    --model-iters)
      MODEL_ITERS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  if [[ -d "$REPO_ROOT/runs" ]]; then
    RUN_DIR="$(ls -td "$REPO_ROOT"/runs/run_* 2>/dev/null | head -n1 || true)"
  fi
  if [[ -z "$RUN_DIR" ]]; then
    echo "Failed to locate a run directory. Provide one with --run-dir." >&2
    exit 1
  fi
  echo "Auto-selected run directory: $RUN_DIR"
fi

if [[ ${#CASES_DIRS[@]} -eq 0 ]]; then
  for difficulty in "${DEFAULT_DIFFICULTIES[@]}"; do
    CASES_DIRS+=("$BENCHMARK_ROOT/$difficulty")
  done
fi

if [[ -z "$TIMESTAMP" ]]; then
  TIMESTAMP_BASE="rand_$(date +%Y%m%d_%H%M%S)_$RANDOM"
else
  TIMESTAMP_BASE="$TIMESTAMP"
fi

declare -a SUMMARY_ARGS=()
declare -a REPORT_LINES=()

mkdir -p "$RUN_DIR/analysis"

for cases_dir in "${CASES_DIRS[@]}"; do
  if [[ ! -d "$cases_dir" ]]; then
    echo "Cases directory does not exist: $cases_dir" >&2
    exit 1
  fi
  label="$(basename "$cases_dir")"
  timestamp="${TIMESTAMP_BASE}_${label}"
  echo -e "\n===== Running analysis for ${label} (${cases_dir}) ====="
  CMD=(python "$REPO_ROOT/src/analyze.py"
       --run-dir "$RUN_DIR"
       --device "$DEVICE"
       --cases-dir "$cases_dir"
       --threshold "$THRESHOLD"
       --model-iters "$MODEL_ITERS"
       --timestamp "$timestamp"
       --no-visualize)
  if [[ -n "$CLINGO_PATH" ]]; then
    CMD+=(--clingo-path "$CLINGO_PATH")
  fi
  "${CMD[@]}"

  eval_path="$RUN_DIR/analysis/$timestamp/eval.json"
  if [[ ! -f "$eval_path" ]]; then
    echo "Evaluation summary not found at $eval_path" >&2
    continue
  fi

  read -r success total accuracy <<<"$(python - "$eval_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8") as handle:
    data = json.load(handle)
values = list(data.values())
total = len(values)
success = sum(1 for value in values if value)
acc = (success / total * 100.0) if total else 0.0
print(success, total, f"{acc:.2f}")
PY
"$eval_path"
)"

  SUMMARY_ARGS+=("$label" "$success" "$total")
  REPORT_LINES+=("$label: $success/$total cases (${accuracy}% exact)")
done

SUMMARY_PLOT="$RUN_DIR/analysis/random_difficulty_accuracy.png"
if [[ ${#SUMMARY_ARGS[@]} -gt 0 ]]; then
  PLOT_PATH="$(python - "$SUMMARY_PLOT" "${SUMMARY_ARGS[@]}" <<'PY'
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402

plot_path = Path(sys.argv[1]).resolve()
triples = sys.argv[2:]
if len(triples) % 3 != 0:
    raise SystemExit("Expected triples of label,success,total for summary plot.")

labels = []
pretty_labels = []
accuracies = []
pairs = []

for idx in range(0, len(triples), 3):
    label = triples[idx]
    success = int(triples[idx + 1])
    total = int(triples[idx + 2])
    acc = success / total if total else 0.0
    labels.append(label)
    pretty = label
    if pretty.startswith("test_"):
        pretty = pretty[len("test_") :]
    pretty_labels.append(pretty.capitalize())
    accuracies.append(acc)
    pairs.append((success, total))

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(pretty_labels, accuracies, color="#4c72b0", alpha=0.85)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Exact-match accuracy")
ax.set_xlabel("Difficulty")
ax.set_title("StableModelGNN random test accuracy by difficulty")
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

for bar, (success, total), acc in zip(bars, pairs, accuracies):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        acc + 0.02,
        f"{acc*100:.1f}%\\n({success}/{total})",
        ha="center",
        va="bottom",
        fontsize=9,
    )

fig.tight_layout()
plot_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(plot_path, dpi=150)
print(plot_path)
PY
)"
else
  PLOT_PATH=""
fi

echo -e "\n===== Difficulty summary ====="
if [[ ${#REPORT_LINES[@]} -eq 0 ]]; then
  echo "No evaluation summaries were generated."
else
  for line in "${REPORT_LINES[@]}"; do
    echo " - $line"
  done
  if [[ -n "$PLOT_PATH" ]]; then
    echo "Summary plot saved to: $PLOT_PATH"
  fi
fi
