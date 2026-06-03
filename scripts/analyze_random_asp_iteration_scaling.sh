#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOP_LEVEL_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

FAMILY="3lp"
BENCHMARK_ROOT=""
RUN_DIR=""
DEVICE="cpu"
THRESHOLD="0.5"
CLINGO_PATH=""
TIMESTAMP=""
MODEL_ITERS="20"
CASES_DIRS=()
DIFFICULTIES=(test_easy test_medium test_hard)

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Plots per-iteration exact-match accuracy for random ASP benchmark splits.

Options:
      --family NAME          Random ASP family: n2l or 3lp (default: ${FAMILY})
      --benchmark-root PATH  Root containing test_easy/test_medium/test_hard
      --cases-dir PATH       Explicit cases directory; can be supplied multiple times
      --run-dir PATH         NDProp run directory; defaults to latest runs/random_asp/<family>/run_*
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

TIMESTAMP_BASE="${TIMESTAMP:-random_asp_iter_${FAMILY}_$(date +%Y%m%d_%H%M%S)}"
MANIFESTS=()
LABELS=()
REPORT_LINES=()

for cases_dir in "${CASES_DIRS[@]}"; do
  if [[ ! -d "${cases_dir}" ]]; then
    echo "Cases directory does not exist: ${cases_dir}" >&2
    exit 1
  fi
  label="$(basename "${cases_dir}")"
  timestamp="${TIMESTAMP_BASE}_${label}"
  echo "=== Iteration analysis family=${FAMILY} split=${label} ==="
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

  manifest_path="${RUN_DIR}/analysis/${timestamp}/manifest.json"
  if [[ ! -f "${manifest_path}" ]]; then
    echo "Manifest not found at ${manifest_path}" >&2
    continue
  fi
  MANIFESTS+=("${manifest_path}")
  LABELS+=("${label}")
done

SUMMARY_PLOT="${RUN_DIR}/analysis/${TIMESTAMP_BASE}_iteration_accuracy.png"
if [[ ${#MANIFESTS[@]} -gt 0 ]]; then
  PLOT_PATH="$("${PYTHON_BIN}" - "${SUMMARY_PLOT}" "${LABELS[@]}" -- "${MANIFESTS[@]}" <<'PY'
import json
import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plot_path = Path(sys.argv[1]).resolve()
args = sys.argv[2:]
sep = args.index("--")
labels = args[:sep]
manifests = args[sep + 1:]

fig, ax = plt.subplots(figsize=(8, 4.5))
for label, manifest in zip(labels, manifests):
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    iter_lists = [case.get("iter_exact_match", []) for case in data.get("cases", []) if case.get("iter_exact_match")]
    if not iter_lists:
        continue
    min_len = min(len(vals) for vals in iter_lists)
    accs = [sum(1 for vals in iter_lists if vals[idx]) / len(iter_lists) for idx in range(min_len)]
    ax.plot(range(len(accs)), accs, marker="o", linewidth=1.2, label=label)

ax.set_ylim(0, 1.05)
ax.set_xlabel("Iteration")
ax.set_ylabel("Exact-match accuracy")
ax.set_title("NDProp random ASP accuracy vs. iteration")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
plot_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(plot_path, dpi=150)
print(plot_path)
PY
)"
  echo "Iteration plot saved to: ${PLOT_PATH}"
else
  echo "No manifests were generated."
fi
