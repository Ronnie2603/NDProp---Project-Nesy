#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR=""
DEVICE="cpu"
CASES_DIR="/home/moriyama/ASP/benchmarks/erdos_renyi/datasets/test"
THRESHOLD="0.5"
CLINGO_PATH=""
TIMESTAMP=""

usage() {
  cat <<'EOF'
Usage: analyze.sh [--run-dir PATH] [--device DEVICE] [--cases-dir PATH] [--threshold VALUE]
                  [--clingo-path PATH] [--timestamp TAG]

Runs the decision-based GNN analyzer on curated logic programs. When --run-dir is
omitted the most recent run under runs/ is selected automatically.
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
      CASES_DIR="$2"
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
# CASES_DIR="$RUN_DIR/eval_programs.lp"

CMD=(python "$REPO_ROOT/src/analyze.py" --run-dir "$RUN_DIR" --device "$DEVICE" --cases-dir "$CASES_DIR" --threshold "$THRESHOLD" --model-iters 50 --no-visualize)

if [[ -n "$CLINGO_PATH" ]]; then
  CMD+=(--clingo-path "$CLINGO_PATH")
fi

if [[ -n "$TIMESTAMP" ]]; then
  CMD+=(--timestamp "$TIMESTAMP")
fi

exec "${CMD[@]}"
