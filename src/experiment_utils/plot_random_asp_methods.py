from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ndprop_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/ndprop_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402


DIFFICULTIES = ("test_easy", "test_medium", "test_hard")
PRETTY_DIFFICULTIES = {
    "test_easy": "Easy",
    "test_medium": "Medium",
    "test_hard": "Hard",
}
PRETTY_FAMILIES = {
    "n2l": "N2L",
    "3lp": "3-LP",
}
PRETTY_METHODS = {
    "NDProp train easy": "NDProp (Trained on Easy dataset)",
    "NDProp train medium": "NDProp (Trained on Medium dataset)",
    "NDProp train hard": "NDProp (Trained on Hard dataset)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot random ASP accuracy across NDProp and baselines.")
    parser.add_argument("--family", choices=("n2l", "3lp"), default="n2l")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--baseline-root", type=Path, default=None)
    parser.add_argument("--random-root", type=Path, action="append", default=[])
    parser.add_argument("--train-difficulty", action="append", choices=("easy", "medium", "hard"), default=[])
    parser.add_argument("--restarts", nargs="+", default=["1", "10", "100"])
    parser.add_argument("--model-iters", type=int, default=50, help="Required NDProp analysis iteration count.")
    return parser.parse_args()


def read_eval_accuracy(path: Path) -> Optional[float]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    values = list(data.values())
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def read_summary_accuracy(path: Path) -> Optional[float]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    successes = data.get("successes")
    total = data.get("total")
    if successes is not None and total:
        return float(successes) / float(total)
    accuracy = data.get("accuracy")
    if accuracy is None:
        return None
    accuracy = float(accuracy)
    return accuracy / 100.0 if accuracy > 1.0 else accuracy


def latest_path(paths: Iterable[Path]) -> Optional[Path]:
    existing = [path for path in paths if path.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda path: (path.stat().st_mtime, str(path)))


def latest_run(run_root: Path) -> Optional[Path]:
    if not run_root.exists():
        return None
    runs = [
        path
        for path in run_root.iterdir()
        if path.is_dir() and path.name.startswith("run_") and (path / "config.json").is_file()
    ]
    if not runs:
        return None
    return max(runs, key=lambda path: (path.stat().st_mtime, str(path)))


def analysis_model_iters(eval_path: Path) -> Optional[int]:
    manifest_path = eval_path.parent / "manifest.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    value = manifest.get("model_iters")
    return int(value) if value is not None else None


def latest_ndprop_eval(run_dir: Path, family: str, split: str, model_iters: int) -> Optional[Path]:
    analysis_dir = run_dir / "analysis"
    if not analysis_dir.exists():
        return None
    candidates = []
    for eval_path in analysis_dir.glob("*/eval.json"):
        name = eval_path.parent.name
        if not name.endswith(f"_{split}"):
            continue
        if name.startswith(f"random_asp_{family}_"):
            if analysis_model_iters(eval_path) == model_iters:
                candidates.append(eval_path)
    return latest_path(candidates)


def collect_ndprop(
    runs_root: Path,
    family: str,
    train_difficulties: Iterable[str],
    model_iters: int,
) -> Dict[str, Dict[str, Optional[float]]]:
    series: Dict[str, Dict[str, Optional[float]]] = {}
    for train_difficulty in train_difficulties:
        label = f"NDProp train {train_difficulty}"
        values = {split: None for split in DIFFICULTIES}
        run_dir = latest_run(runs_root / family / train_difficulty)
        if run_dir is not None:
            for split in DIFFICULTIES:
                eval_path = latest_ndprop_eval(run_dir, family, split, model_iters)
                if eval_path is not None:
                    values[split] = read_eval_accuracy(eval_path)
        series[label] = values
    return series


def collect_random(family: str, random_roots: Iterable[Path]) -> Dict[str, Optional[float]]:
    values = {split: None for split in DIFFICULTIES}
    roots = [root for root in random_roots if root.exists()]
    if not roots:
        return values
    for split in DIFFICULTIES:
        candidates = []
        for root in roots:
            candidates.extend(root.glob(f"random_{family}_*_{split}/summary.json"))
        summary_path = latest_path(candidates)
        if summary_path is not None:
            values[split] = read_summary_accuracy(summary_path)
    return values


def collect_rdprop(
    baseline_root: Path,
    family: str,
    restarts: Iterable[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    series: Dict[str, Dict[str, Optional[float]]] = {}
    for restart in restarts:
        label = f"RDProp-{restart}"
        values = {split: None for split in DIFFICULTIES}
        for split in DIFFICULTIES:
            pattern = f"rdprop_{family}_*_r{restart}_{split}/summary.json"
            summary_path = latest_path(baseline_root.glob(pattern))
            if summary_path is not None:
                values[split] = read_summary_accuracy(summary_path)
        series[label] = values
    return series


def write_csv(csv_path: Path, series: Dict[str, Dict[str, Optional[float]]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "difficulty", "accuracy"])
        for method, values in series.items():
            for split in DIFFICULTIES:
                value = values.get(split)
                writer.writerow([method, PRETTY_DIFFICULTIES[split], "" if value is None else f"{value:.6f}"])


def method_style(method: str, method_idx: int) -> Dict[str, object]:
    if method.startswith("NDProp"):
        ndprop_colors = {
            "NDProp train easy": "#6baed6",
            "NDProp train medium": "#2171b5",
            "NDProp train hard": "#08306b",
        }
        return {
            "color": ndprop_colors.get(method, "#2171b5"),
            "hatch": "//",
            "edgecolor": "#1f4e79",
            "linewidth": 0.8,
            "alpha": 0.9,
        }
    if method.startswith("RDProp"):
        rdprop_colors = {
            "RDProp-1": "#fdd0a2",
            "RDProp-10": "#f16913",
            "RDProp-100": "#a63603",
        }
        return {
            "color": rdprop_colors.get(method, "#f16913"),
            "hatch": "",
            "edgecolor": "#7f2704",
            "linewidth": 0.6,
            "alpha": 0.9,
        }
    if method == "Random":
        return {
            "color": "#74c476",
            "hatch": "xx",
            "edgecolor": "#238b45",
            "linewidth": 0.8,
            "alpha": 0.85,
        }
    fallback_colors = ["#9ecae1", "#fdae6b", "#a1d99b", "#bcbddc"]
    return {
        "color": fallback_colors[method_idx % len(fallback_colors)],
        "hatch": "",
        "edgecolor": "#444444",
        "linewidth": 0.6,
        "alpha": 0.88,
    }


def pretty_method(method: str) -> str:
    return PRETTY_METHODS.get(method, method)


def plot(plot_path: Path, family: str, series: Dict[str, Dict[str, Optional[float]]]) -> None:
    methods = list(series.keys())
    x_positions = list(range(len(DIFFICULTIES)))
    group_width = 0.84
    bar_width = group_width / max(1, len(methods))

    fig_width = max(10.0, 1.6 * len(methods) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))

    for method_idx, method in enumerate(methods):
        style = method_style(method, method_idx)
        offset = -group_width / 2 + bar_width / 2 + method_idx * bar_width
        heights = []
        missing_indices = []
        for split_idx, split in enumerate(DIFFICULTIES):
            value = series[method].get(split)
            if value is None or math.isnan(value):
                heights.append(0.0)
                missing_indices.append(split_idx)
            else:
                heights.append(value)
        bars = ax.bar(
            [x + offset for x in x_positions],
            heights,
            width=bar_width * 0.92,
            label=pretty_method(method),
            color=style["color"],
            hatch=str(style["hatch"]),
            edgecolor=style["edgecolor"],
            linewidth=float(style["linewidth"]),
            alpha=float(style["alpha"]),
        )
        for split_idx, bar in enumerate(bars):
            value = series[method].get(DIFFICULTIES[split_idx])
            if split_idx in missing_indices:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.015,
                    "n/a",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                    color="#555555",
                )
            elif value is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(value + 0.015, 1.03),
                    f"{value * 100:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([PRETTY_DIFFICULTIES[split] for split in DIFFICULTIES])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Test difficulty")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Random ASP comparison ({PRETTY_FAMILIES.get(family, family)})")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=min(3, len(methods)))
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def default_random_roots(project_root: Path, family: str) -> list[Path]:
    return [project_root / "baselines" / "random_asp_random"]


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    runs_root = (args.runs_root or project_root / "runs" / "random_asp").resolve()
    baseline_root = (args.baseline_root or project_root / "baselines" / "random_asp_rdprop").resolve()
    train_difficulties = args.train_difficulty or ["easy", "medium"]
    output = args.output or (
        project_root / "runs" / "random_asp" / args.family / f"random_asp_{args.family}_method_comparison.png"
    )

    series: Dict[str, Dict[str, Optional[float]]] = {}
    series.update(collect_ndprop(runs_root, args.family, train_difficulties, args.model_iters))
    random_roots = [root.resolve() for root in args.random_root] or default_random_roots(project_root, args.family)
    series["Random"] = collect_random(args.family, random_roots)
    series.update(collect_rdprop(baseline_root, args.family, args.restarts))

    plot(output.resolve(), args.family, series)
    write_csv(output.resolve().with_suffix(".csv"), series)
    print(output.resolve())


if __name__ == "__main__":
    main()
