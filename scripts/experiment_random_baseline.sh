#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOP_LEVEL_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
CASES_DIRS=()
# BENCHMARK_ROOT="$TOP_LEVEL_ROOT/benchmarks/random_klp/3-lp"
BENCHMARK_ROOT="$TOP_LEVEL_ROOT/benchmarks/random_data/datasets_2sm"
OUTPUT_ROOT="${REPO_ROOT}/baselines/random_stable_n2l"
SAMPLES="100"
INCLUDE_PROB="0.5"
SEED="7"
CLINGO_PATH=""
TIMESTAMP=""
declare -a DEFAULT_DIFFICULTIES=("test_easy" "test_medium" "test_hard")

usage() {
  cat <<'EOF'
Usage: experiment_random_baseline.sh [--cases-dir PATH]...
                                     [--benchmark-root PATH]
                                     [--output-root PATH]
                                     [--samples N]
                                     [--include-prob P]
                                     [--seed N]
                                     [--clingo-path PATH]
                                     [--timestamp TAG]

Runs the random stable-model baseline on benchmark splits. The baseline samples
random candidate sets and checks if they match any stable model.
EOF
}

if [[ $# -eq 1 && "$1" == "--help" ]]; then
  usage
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cases-dir)
      CASES_DIRS+=("$2")
      shift 2
      ;;
    --benchmark-root)
      BENCHMARK_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --samples)
      SAMPLES="$2"
      shift 2
      ;;
    --include-prob)
      INCLUDE_PROB="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
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
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ${#CASES_DIRS[@]} -eq 0 ]]; then
  for difficulty in "${DEFAULT_DIFFICULTIES[@]}"; do
    CASES_DIRS+=("$BENCHMARK_ROOT/$difficulty")
  done
fi

if [[ -z "$TIMESTAMP" ]]; then
  TIMESTAMP_BASE="rand_baseline_$(date +%Y%m%d_%H%M%S)_$RANDOM"
else
  TIMESTAMP_BASE="$TIMESTAMP"
fi

mkdir -p "$OUTPUT_ROOT"

declare -a SUMMARY_ARGS=()
declare -a REPORT_LINES=()

for cases_dir in "${CASES_DIRS[@]}"; do
  if [[ ! -d "$cases_dir" ]]; then
    echo "Cases directory does not exist: $cases_dir" >&2
    exit 1
  fi
  label="$(basename "$cases_dir")"
  output_dir="$OUTPUT_ROOT/${TIMESTAMP_BASE}_${label}"

  echo -e "\n===== Running baseline for ${label} (${cases_dir}) ====="
  CMD=(python "$REPO_ROOT/src/random_stable_baseline.py"
       --cases-dir "$cases_dir"
       --samples "$SAMPLES"
       --include-prob "$INCLUDE_PROB"
       --seed "$SEED"
       --output-dir "$output_dir")
  if [[ -n "$CLINGO_PATH" ]]; then
    CMD+=(--clingo-path "$CLINGO_PATH")
  fi
  "${CMD[@]}"

  eval_path="$output_dir/eval.json"
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
  REPORT_LINES+=("$label: $success/$total cases (${accuracy}% success)")
done

SUMMARY_PLOT="$OUTPUT_ROOT/${TIMESTAMP_BASE}_accuracy.png"
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
bars = ax.bar(pretty_labels, accuracies, color="#55a868", alpha=0.85)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Random baseline success rate")
ax.set_xlabel("Difficulty")
ax.set_title("Random stable-model baseline accuracy")
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

echo -e "\n===== Baseline summary ====="
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
