#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") [command] [options]

Commands:
  task       Run one neuro-symbolic task.
  run        Run all six neuro-symbolic tasks.
  summarize  Rebuild results.csv from the latest runs, then summarize it.
  all        Run the six tasks, then summarize them.

The default command is "run". Use '<command> --help' for command-specific
options. The task command controls one experiment; run and all control the task
matrix, devices, concurrency, and output paths.

Note: "all --keep-results" preserves results during the run phase, but the
following summarize phase rebuilds results.csv from the latest run directories.
EOF
}

cmd="${1:-run}"
if [[ $# -gt 0 ]]; then shift; fi

case "${cmd}" in
  task) exec bash "${SCRIPT_DIR}/task.sh" "$@" ;;
  run) exec bash "${SCRIPT_DIR}/run_all.sh" "$@" ;;
  summarize) exec bash "${SCRIPT_DIR}/summarize.sh" "$@" ;;
  all)
    if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
      exec bash "${SCRIPT_DIR}/run_all.sh" --help
    fi
    summarize_args=()
    scan_args=("$@")
    scan_index=0
    while [[ ${scan_index} -lt ${#scan_args[@]} ]]; do
      case "${scan_args[${scan_index}]}" in
        --tasks)
          summarize_args+=(--tasks)
          scan_index=$((scan_index + 1))
          while [[ ${scan_index} -lt ${#scan_args[@]} && "${scan_args[${scan_index}]}" != --* ]]; do
            summarize_args+=("${scan_args[${scan_index}]}")
            scan_index=$((scan_index + 1))
          done
          ;;
        --log-root|--results-csv)
          summarize_args+=("${scan_args[${scan_index}]}" "${scan_args[$((scan_index + 1))]}")
          scan_index=$((scan_index + 2))
          ;;
        *) scan_index=$((scan_index + 1)) ;;
      esac
    done
    bash "${SCRIPT_DIR}/run_all.sh" "$@"
    exec bash "${SCRIPT_DIR}/summarize.sh" "${summarize_args[@]}"
    ;;
  -h|--help|help) usage ;;
  *) echo "Unknown command: ${cmd}" >&2; usage; exit 1 ;;
esac
