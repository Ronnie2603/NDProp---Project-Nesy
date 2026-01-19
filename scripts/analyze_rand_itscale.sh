#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOP_LEVEL_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
RUN_DIR=""
DEVICE="cpu"
THRESHOLD="0.5"
CLINGO_PATH=""
TIMESTAMP=""
BENCHMARK_ROOT="$TOP_LEVEL_ROOT/benchmarks/random_klp/3-lp"
CASES_DIRS=()
MODEL_ITERS="20"
declare -a DEFAULT_DIFFICULTIES=("test_easy" "test_medium" "test_hard")

usage() {
  cat <<'EOF'
Usage: experiment_dprop_baseline.sh [--run-dir PATH] [--device DEVICE] [--cases-dir PATH]...
                                    [--benchmark-root PATH] [--threshold VALUE]
                                    [--clingo-path PATH] [--timestamp TAG]
                                    [--model-iters COUNT]

Runs a single analyze.py pass per split and derives per-iteration exact-match
accuracy from the model's per-step probabilities.
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
  if [[ -d "$REPO_ROOT/runs_nesy" ]]; then
    RUN_DIR="$(ls -td "$REPO_ROOT"/runs_nesy/* 2>/dev/null | head -n1 || true)"
  elif [[ -d "$REPO_ROOT/runs" ]]; then
    RUN_DIR="$(ls -td "$REPO_ROOT"/runs/* 2>/dev/null | head -n1 || true)"
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
  TIMESTAMP_BASE="dprop_iter_$(date +%Y%m%d_%H%M%S)_$RANDOM"
else
  TIMESTAMP_BASE="$TIMESTAMP"
fi

declare -a MANIFESTS=()
declare -a LABELS=()
declare -a REPORT_LINES=()

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

  manifest_path="$RUN_DIR/analysis/$timestamp/manifest.json"
  if [[ ! -f "$manifest_path" ]]; then
    echo "Manifest not found at $manifest_path" >&2
    continue
  fi

  read -r success total accuracy iter_count <<<"$(python - "$manifest_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8") as handle:
    data = json.load(handle)
cases = data.get("cases", [])
iter_lists = [case.get("iter_exact_match", []) for case in cases if case.get("iter_exact_match")]
if not iter_lists:
    print(0, 0, "0.00", 0)
    raise SystemExit(0)
min_len = min(len(vals) for vals in iter_lists)
idx = min_len - 1
success = sum(1 for vals in iter_lists if vals[idx])
total = len(iter_lists)
acc = (success / total * 100.0) if total else 0.0
print(success, total, f"{acc:.2f}", min_len)
PY
)"

  MANIFESTS+=("$manifest_path")
  LABELS+=("$label")
  REPORT_LINES+=("$label: last_iter=$iter_count success=$success/$total (${accuracy}% exact)")
done

SUMMARY_PLOT="$RUN_DIR/analysis/${TIMESTAMP_BASE}_iteration_accuracy.png"
if [[ ${#MANIFESTS[@]} -gt 0 ]]; then
  PLOT_PATH="$(python - "$SUMMARY_PLOT" "${LABELS[@]}" -- "${MANIFESTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plot_path = Path(sys.argv[1]).resolve()
args = sys.argv[2:]
if "--" not in args:
    raise SystemExit("Expected -- separator between labels and manifests.")
sep = args.index("--")
labels = args[:sep]
manifests = args[sep + 1 :]
if len(labels) != len(manifests):
    raise SystemExit("Labels and manifests length mismatch.")

series = {}
for label, manifest in zip(labels, manifests):
    with Path(manifest).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    cases = data.get("cases", [])
    iter_lists = [case.get("iter_exact_match", []) for case in cases if case.get("iter_exact_match")]
    if not iter_lists:
        series[label] = []
        continue
    min_len = min(len(vals) for vals in iter_lists)
    accs = []
    for idx in range(min_len):
        total = len(iter_lists)
        success = sum(1 for vals in iter_lists if vals[idx])
        accs.append(success / total if total else 0.0)
    series[label] = accs

fig, ax = plt.subplots(figsize=(8, 4.5))
for label, accs in series.items():
    if not accs:
        continue
    xs = list(range(len(accs)))
    ax.plot(xs, accs, marker="o", linewidth=1.2, label=label)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Iteration")
ax.set_ylabel("Exact-match accuracy")
ax.set_title("Stable model match vs. iteration")
ax.grid(True, axis="y", alpha=0.3)
if series:
    ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
plot_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(plot_path, dpi=150)
print(plot_path)
PY
)"
else
  PLOT_PATH=""
fi

echo -e "\n===== Iteration summary ====="
if [[ ${#REPORT_LINES[@]} -eq 0 ]]; then
  echo "No analysis summaries were generated."
else
  for line in "${REPORT_LINES[@]}"; do
    echo " - $line"
  done
  if [[ -n "$PLOT_PATH" ]]; then
    echo "Iteration plot saved to: $PLOT_PATH"
  fi
fi
