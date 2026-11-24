#!/usr/bin/env bash
# NDProp 3-coloring SAT experiment pipeline, mirroring scripts/random_asp/main.sh.
# Does not edit repository files. Uses the repository's existing Python entrypoints
# and plotting utilities, with a graph-coloring-specific method-comparison report.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ndprop_matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/ndprop_cache}"

COMMAND="${1:-all}"
if (($#)); then shift; fi
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${PROJECT_ROOT}/../graph_coloring_sat_grounded}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/runs/graph_coloring}"
DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MODEL_ITERS="${MODEL_ITERS:-10}"
TEST_ITERS="${TEST_ITERS:-50}"
ATOM_HIDDEN="${ATOM_HIDDEN:-32}"
LOGICAL_DIMENSION="${LOGICAL_DIMENSION:-32}"
TNORM="${TNORM:-godel}"
TNORM_WFS="${TNORM_WFS:-godel}"
LR="${LR:-1e-3}"
SEED="${SEED:-7}"
THRESHOLD="${THRESHOLD:-0.5}"
RANDOM_SAMPLES="${RANDOM_SAMPLES:-100}"
INCLUDE_PROB="${INCLUDE_PROB:-0.5}"
CLINGO_PATH=""
DLV_PATH=""
RUN_DIR=""
TRAIN_DIFFICULTIES=(easy medium)
TEST_DIFFICULTIES=(easy medium hard)
RESTARTS=(1 10 100)
SET_TRAIN=false
SET_TEST=false
SKIP_BASELINES=false
KEEP_RESULTS=false
VISUALIZE=false
EXTRA_TRAIN_ARGS=()

fail() { echo "ERROR: $*" >&2; exit 1; }
usage() { cat <<'EOF'
Usage: bash scripts/graph_coloring/main.sh [command] [options] [-- extra-main.py-args]
Commands:
  all        Train, analyze by difficulty, iteration scaling, random/RDProp,
             comparative CSV + PNG (default)
  train      Train one model per selected training difficulty
  analyze    Difficulty accuracy and (optional) per-case visualization
  iteration  Accuracy-vs-iteration plot for each trained model
  random     Random candidate-set baseline
  baseline   RDProp baseline (restarts: 1, 10, 100 by default)
  compare    Comparison table CSV + grouped-bar PNG
Options:
  --difficulty easy|medium|hard       Repeatable training split (default easy+medium)
  --test-difficulty easy|medium|hard  Repeatable test split (default all three)
  --device DEVICE                    Default cpu
  --benchmark-root PATH              Default ../graph_coloring_sat_grounded
  --log-root PATH                    Default runs/graph_coloring
  --run-dir PATH                     Only for one training difficulty
  --epochs N                         Default 1000
  --batch-size N                     Default 64
  --model-iters N                    Training iterations, default 10
  --test-iters N                     Evaluation iterations, default 50
  --atom-hidden N                    Default 32
  --logical-dimension N             Default 32
  --tnorm NAME                       Default godel
  --tnorm-wfs NAME                   Default godel
  --lr FLOAT                         Default 1e-3
  --seed N                           Default 7
  --threshold FLOAT                  Default 0.5
  --random-samples N                 Default 100
  --include-prob FLOAT               Default 0.5 (same as original Random ASP)
  --restarts N ...                   Default 1 10 100
  --clingo-path PATH
  --dlv-path PATH                    Optional WFS cross-check for RDProp
  --visualize                        Enable per-program analysis plots (off by default)
  --skip-baselines                   Only with 'all'
  --keep-results                     Preserve existing comparison CSV/PNG
  --                               Pass remaining arguments to src/main.py (train/all)
No --family: one task, SAT graph 3-coloring. No benchmark regeneration.
No repository files are modified.
EOF
}
if [[ "$COMMAND" == help || "$COMMAND" == --help || "$COMMAND" == -h ]]; then usage; exit 0; fi
while (($#)); do
  case "$1" in
    --difficulty) (($#>=2)) || fail 'Missing difficulty'; if [[ "$SET_TRAIN" == false ]]; then TRAIN_DIFFICULTIES=(); SET_TRAIN=true; fi; TRAIN_DIFFICULTIES+=("$2"); shift 2 ;;
    --test-difficulty) (($#>=2)) || fail 'Missing test difficulty'; if [[ "$SET_TEST" == false ]]; then TEST_DIFFICULTIES=(); SET_TEST=true; fi; TEST_DIFFICULTIES+=("$2"); shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --benchmark-root) BENCHMARK_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --model-iters) MODEL_ITERS="$2"; shift 2 ;;
    --test-iters) TEST_ITERS="$2"; shift 2 ;;
    --atom-hidden) ATOM_HIDDEN="$2"; shift 2 ;;
    --logical-dimension) LOGICAL_DIMENSION="$2"; shift 2 ;;
    --tnorm) TNORM="$2"; shift 2 ;;
    --tnorm-wfs) TNORM_WFS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    --random-samples) RANDOM_SAMPLES="$2"; shift 2 ;;
    --include-prob) INCLUDE_PROB="$2"; shift 2 ;;
    --clingo-path) CLINGO_PATH="$2"; shift 2 ;;
    --dlv-path) DLV_PATH="$2"; shift 2 ;;
    --visualize) VISUALIZE=true; shift ;;
    --skip-baselines) SKIP_BASELINES=true; shift ;;
    --keep-results) KEEP_RESULTS=true; shift ;;
    --restarts) shift; RESTARTS=(); while (($#)) && [[ "$1" != --* ]]; do RESTARTS+=("$1"); shift; done ;;
    --) shift; EXTRA_TRAIN_ARGS=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done
case "$COMMAND" in all|train|analyze|iteration|random|baseline|compare) ;; *) fail "Unknown command: $COMMAND" ;; esac
((${#TRAIN_DIFFICULTIES[@]})) || fail 'No training difficulties'
((${#TEST_DIFFICULTIES[@]})) || fail 'No test difficulties'
((${#RESTARTS[@]})) || fail 'No RDProp restarts'
for d in "${TRAIN_DIFFICULTIES[@]}" "${TEST_DIFFICULTIES[@]}"; do case "$d" in easy|medium|hard) ;; *) fail "Invalid difficulty: $d" ;; esac; done
for n in "$EPOCHS" "$BATCH_SIZE" "$MODEL_ITERS" "$TEST_ITERS" "$ATOM_HIDDEN" "$LOGICAL_DIMENSION" "$SEED" "$RANDOM_SAMPLES" "${RESTARTS[@]}"; do [[ "$n" =~ ^[0-9]+$ ]] || fail "Expected nonnegative integer: $n"; done
[[ -z "$RUN_DIR" || ${#TRAIN_DIFFICULTIES[@]} -eq 1 ]] || fail '--run-dir requires one --difficulty'
[[ -z "$RUN_DIR" || "$COMMAND" != all && "$COMMAND" != train ]] || fail '--run-dir is only for analyze/iteration/compare'

absolute() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s/%s\n' "$PWD" "$1" ;; esac; }
BENCHMARK_ROOT="$(absolute "$BENCHMARK_ROOT")"
LOG_ROOT="$(absolute "$LOG_ROOT")"
[[ -z "$RUN_DIR" ]] || RUN_DIR="$(absolute "$RUN_DIR")"

split_dir() {
  local split="$1" d="$2"
  if [[ -d "$BENCHMARK_ROOT/${split}_${d}" ]]; then printf '%s\n' "$BENCHMARK_ROOT/${split}_${d}"
  elif [[ -d "$BENCHMARK_ROOT/$split/$d" ]]; then printf '%s\n' "$BENCHMARK_ROOT/$split/$d"
  else fail "Missing split ${split}_${d} or $split/$d in $BENCHMARK_ROOT"; fi
}
check_split() { local dir; dir="$(split_dir "$1" "$2")"; compgen -G "$dir/*.lp" >/dev/null || fail "No .lp files: $dir"; printf '%s\n' "$dir"; }
latest_run() {
  local d="$1" item
  [[ -d "$LOG_ROOT/$d" ]] || fail "Missing training runs in $LOG_ROOT/$d"
  while IFS= read -r item; do
    if [[ -f "$item/config.json" && -f "$item/best.pt" ]]; then printf '%s\n' "$item"; return; fi
  done < <(find "$LOG_ROOT/$d" -mindepth 1 -maxdepth 1 -type d -name 'run_*' | LC_ALL=C sort -r)
  fail "No completed checkpoints in $LOG_ROOT/$d"
}
run_for() { if [[ -n "$RUN_DIR" ]]; then [[ -f "$RUN_DIR/best.pt" && -f "$RUN_DIR/config.json" ]] || fail "Invalid --run-dir: $RUN_DIR"; echo "$RUN_DIR"; else latest_run "$1"; fi; }
clingo_args=(); [[ -z "$CLINGO_PATH" ]] || clingo_args=(--clingo-path "$CLINGO_PATH")
new_tag() { printf 'graph_coloring_%s_%s_%s' "$1" "$(date +%Y%m%d_%H%M%S)" "$BASHPID"; }

train_one() {
  local d="$1" train_dir eval_dir out
  train_dir="$(check_split train "$d")"; eval_dir="$(check_split val "$d")"
  mkdir -p "$LOG_ROOT/$d"
  out="$LOG_ROOT/$d/run_$(date +%Y%m%d_%H%M%S)_${BASHPID}"
  local cmd=("$PYTHON_BIN" "$PROJECT_ROOT/src/main.py"
    --device "$DEVICE" --seed "$SEED" --epochs "$EPOCHS" --lr "$LR"
    --batch-size "$BATCH_SIZE" --model-iters "$MODEL_ITERS"
    --atom-hidden "$ATOM_HIDDEN" --logical-dimension "$LOGICAL_DIMENSION"
    --tnorm "$TNORM" --tnorm-wfs "$TNORM_WFS"
    --train-dir "$train_dir" --eval-dir "$eval_dir" --run-dir "$out"
    "${clingo_args[@]}" "${EXTRA_TRAIN_ARGS[@]}")
  echo "=== NDProp train=$d ==="; printf 'Running: '; printf '%q ' "${cmd[@]}"; echo
  (cd "$PROJECT_ROOT" && "${cmd[@]}")
  [[ -f "$out/config.json" && -f "$out/best.pt" ]] || fail "Training finished without checkpoint: $out"
}

# Same summary helper + plot as scripts/random_asp/analyze.sh.
analyze_one() {
  local d="$1" run="$2" tag dir label p stats success total accuracy
  tag="$(new_tag "analysis_train_${d}")"
  local summary=() reports=()
  for label in "${TEST_DIFFICULTIES[@]}"; do
    dir="$(check_split test "$label")"
    local stamp="${tag}_test_${label}"
    local cmd=("$PYTHON_BIN" "$PROJECT_ROOT/src/analyze.py" --run-dir "$run"
      --device "$DEVICE" --cases-dir "$dir" --threshold "$THRESHOLD"
      --model-iters "$TEST_ITERS" --timestamp "$stamp" "${clingo_args[@]}")
    [[ "$VISUALIZE" == true ]] || cmd+=(--no-visualize)
    echo "=== Difficulty analysis train=$d test=$label ==="
    (cd "$PROJECT_ROOT" && "${cmd[@]}")
    p="$run/analysis/$stamp/eval.json"
    [[ -f "$p" ]] || fail "Missing eval.json: $p"
    stats="$("$PYTHON_BIN" "$PROJECT_ROOT/src/experiment_utils/random_asp_eval_summary.py" "$p")"
    read -r success total accuracy <<< "$stats"
    summary+=("test_${label}" "$success" "$total")
    reports+=("test_${label}: $success/$total (${accuracy}%)")
  done
  local plot="$run/analysis/${tag}_difficulty_accuracy.png"
  "$PYTHON_BIN" "$PROJECT_ROOT/src/experiment_utils/plot_random_asp_difficulty.py" "$plot" "${summary[@]}"
  printf ' - %s\n' "${reports[@]}"
}

# Same per-iteration manifest plotter as scripts/random_asp/iteration_scaling.sh.
iteration_one() {
  local d="$1" run="$2" tag label dir stamp manifest
  tag="$(new_tag "iteration_train_${d}")"
  local args=()
  for label in "${TEST_DIFFICULTIES[@]}"; do
    dir="$(check_split test "$label")"; stamp="${tag}_test_${label}"
    local cmd=("$PYTHON_BIN" "$PROJECT_ROOT/src/analyze.py" --run-dir "$run"
      --device "$DEVICE" --cases-dir "$dir" --threshold "$THRESHOLD"
      --model-iters "$TEST_ITERS" --timestamp "$stamp" "${clingo_args[@]}")
    [[ "$VISUALIZE" == true ]] || cmd+=(--no-visualize)
    echo "=== Iteration analysis train=$d test=$label ==="
    (cd "$PROJECT_ROOT" && "${cmd[@]}")
    manifest="$run/analysis/$stamp/manifest.json"
    [[ -f "$manifest" ]] || fail "Missing manifest.json: $manifest"
    args+=(--label "test_${label}" --manifest "$manifest")
  done
  "$PYTHON_BIN" "$PROJECT_ROOT/src/experiment_utils/plot_random_asp_iterations.py" \
    "$run/analysis/${tag}_iteration_accuracy.png" "${args[@]}"
}

baseline_one() {
  local kind="$1" label dir n tag output
  tag="$(new_tag "${kind}")"
  for label in "${TEST_DIFFICULTIES[@]}"; do
    dir="$(check_split test "$label")"
    if [[ "$kind" == random ]]; then
      output="$LOG_ROOT/baselines/random/${tag}_test_${label}"
      echo "=== Random baseline test=$label ==="
      (cd "$PROJECT_ROOT" && "$PYTHON_BIN" "$PROJECT_ROOT/src/random_stable_baseline.py" \
        --cases-dir "$dir" --samples "$RANDOM_SAMPLES" --include-prob "$INCLUDE_PROB" \
        --seed "$SEED" --output-dir "$output" "${clingo_args[@]}")
    else
      for n in "${RESTARTS[@]}"; do
        output="$LOG_ROOT/baselines/rdprop/${tag}_r${n}_test_${label}"
        local args=(); [[ -z "$DLV_PATH" ]] || args=(--dlv-path "$DLV_PATH")
        echo "=== RDProp-$n test=$label ==="
        (cd "$PROJECT_ROOT" && "$PYTHON_BIN" "$PROJECT_ROOT/src/dprop_baseline.py" \
          --cases-dir "$dir" --trials "$n" --seed "$SEED" --output-dir "$output" \
          "${clingo_args[@]}" "${args[@]}")
      done
    fi
  done
}

# Use exact same accuracy semantics as the original plot_random_asp_methods.py.
# Its CLI only supports n2l/3lp, so gather graph-coloring paths here and draw
# the corresponding grouped-bar chart in one in-memory Python snippet.
compare_results() {
  mkdir -p "$LOG_ROOT"
  local target="$LOG_ROOT/graph_coloring_method_comparison" d run
  local run_pairs=()
  for d in "${TRAIN_DIFFICULTIES[@]}"; do run="$(run_for "$d")"; run_pairs+=("$d=$run"); done
  if [[ "$KEEP_RESULTS" != true ]]; then rm -f "$target.csv" "$target.png"; fi
  "$PYTHON_BIN" - "$target" "$TEST_ITERS" "${#TEST_DIFFICULTIES[@]}" "${TEST_DIFFICULTIES[@]}" "${#RESTARTS[@]}" "${RESTARTS[@]}" "${#run_pairs[@]}" "${run_pairs[@]}" <<'PY'
import csv, json, math, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

args=iter(sys.argv[1:])
base=Path(next(args)); model_iters=int(next(args))
tests=[next(args) for _ in range(int(next(args)))]
restarts=[next(args) for _ in range(int(next(args)))]
runs=dict(next(args).split('=', 1) for _ in range(int(next(args))))
root=base.parent

def summary(path):
    try: x=json.loads(path.read_text())
    except (OSError, ValueError): return None
    if x.get('total'):
        return float(x['successes']) / float(x['total'])
    a=x.get('accuracy')
    return None if a is None else (float(a)/100 if float(a)>1 else float(a))

def newest(paths):
    xs=[p for p in paths if p.is_file()]
    return max(xs,key=lambda p:(p.stat().st_mtime,str(p))) if xs else None

def eval_acc(path):
    x=json.loads(path.read_text()); vs=list(x.values())
    return sum(bool(v) for v in vs)/len(vs) if vs else None

def matching_eval(run, test):
    p=Path(run)/'analysis'
    xs=[]
    for f in p.glob(f'graph_coloring_analysis_train_*_test_{test}/eval.json'):
        m=f.parent/'manifest.json'
        try: it=int(json.loads(m.read_text()).get('model_iters'))
        except (OSError, ValueError, TypeError): continue
        if it==model_iters: xs.append(f)
    return newest(xs)

series={}
for difficulty, run in runs.items():
    series[f'NDProp train {difficulty}']={t:(eval_acc(f) if (f:=matching_eval(run,t)) else None) for t in tests}
series['Random']={t:(summary(f) if (f:=newest((root/'baselines/random').glob(f'*_test_{t}/summary.json'))) else None) for t in tests}
for n in restarts:
    series[f'RDProp-{n}']={t:(summary(f) if (f:=newest((root/'baselines/rdprop').glob(f'*_r{n}_test_{t}/summary.json'))) else None) for t in tests}
base.parent.mkdir(parents=True,exist_ok=True)
with base.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f);w.writerow(['method','difficulty','accuracy'])
    for method, values in series.items():
        for t in tests: w.writerow([method,t.capitalize(),'' if values[t] is None else f'{values[t]:.6f}'])
fig,ax=plt.subplots(figsize=(max(10,1.6*len(series)+4),5.2))
width=0.84/max(1,len(series))
styles=['#6baed6','#2171b5','#08306b','#74c476','#fdd0a2','#f16913','#a63603']
for j,(method, vals) in enumerate(series.items()):
    xs=[i-0.42+width/2+j*width for i in range(len(tests))]
    heights=[vals[t] or 0 for t in tests]
    bars=ax.bar(xs,heights,width=width*0.92,label=method,color=styles[j%len(styles)],
                hatch='//' if method.startswith('NDProp') else ('xx' if method=='Random' else ''))
    for bar,t in zip(bars,tests):
        v=vals[t]
        ax.text(bar.get_x()+bar.get_width()/2,0.015 if v is None else min(v+0.015,1.03),
                'n/a' if v is None else f'{v*100:.0f}',ha='center',va='bottom',fontsize=7,rotation=90)
ax.set_xticks(range(len(tests)),[t.capitalize() for t in tests])
ax.set_ylim(0,1.08);ax.set_xlabel('Test difficulty');ax.set_ylabel('Accuracy')
ax.set_title('Graph 3-coloring SAT: NDProp vs baselines')
ax.yaxis.set_major_formatter(PercentFormatter(1.0));ax.grid(axis='y',alpha=.25)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=min(3,len(series)))
fig.tight_layout(); fig.savefig(base.with_suffix('.png'),dpi=150);plt.close(fig)
print(f'Comparison CSV: {base.with_suffix(".csv")}')
print(f'Comparison PNG: {base.with_suffix(".png")}')
for method, vals in series.items():
    print(method+': '+', '.join(f'{t}={"n/a" if vals[t] is None else f"{100*vals[t]:.2f}%"}' for t in tests))
PY
}

case "$COMMAND" in
  train) for d in "${TRAIN_DIFFICULTIES[@]}"; do train_one "$d"; done ;;
  analyze) for d in "${TRAIN_DIFFICULTIES[@]}"; do analyze_one "$d" "$(run_for "$d")"; done ;;
  iteration) for d in "${TRAIN_DIFFICULTIES[@]}"; do iteration_one "$d" "$(run_for "$d")"; done ;;
  random) baseline_one random ;;
  baseline) baseline_one rdprop ;;
  compare) compare_results ;;
  all)
    for d in "${TRAIN_DIFFICULTIES[@]}"; do
      train_one "$d"
      run="$(latest_run "$d")"
      analyze_one "$d" "$run"
      iteration_one "$d" "$run"
    done
    if [[ "$SKIP_BASELINES" == false ]]; then baseline_one random; baseline_one rdprop; fi
    compare_results ;;
esac
