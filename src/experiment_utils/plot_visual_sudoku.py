from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ndprop_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/ndprop_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402


METRICS = ("grid_acc", "cell_acc")
METRIC_TITLES = {"grid_acc": "Exact-grid accuracy", "cell_acc": "Cell accuracy"}
METHODS = ("NDProp", "Clingo", "NN baseline", "RDProp")
METHOD_STYLES = {
    "NDProp": {"color": "#2171b5", "marker": "o", "linestyle": "-"},
    "Clingo": {"color": "#31a354", "marker": "^", "linestyle": ":"},
    "NN baseline": {"color": "#756bb1", "marker": "D", "linestyle": "-."},
    "RDProp": {"color": "#e6550d", "marker": "s", "linestyle": "--"},
}


@dataclass(frozen=True)
class Observation:
    metric: str
    method: str
    pct: float
    pct_label: str
    seed: str
    value: float
    source: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and plot Visual Sudoku accuracy across seeds.")
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pcts", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument(
        "--ndprop-epoch",
        choices=("best", "last"),
        default="best",
        help="Use the best or final NDProp evaluation epoch (default: best).",
    )
    return parser.parse_args()


def pct_tag(value: str) -> str:
    return value.replace(".", "p")


def latest_with(parent: Path, filename: str) -> Optional[Path]:
    if not parent.exists():
        return None
    candidates = [path for path in parent.glob("run_*") if path.is_dir() and (path / filename).is_file()]
    return max(candidates, key=lambda path: path.name) if candidates else None


def read_ndprop_metric(path: Path, selection: str) -> Optional[float]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return None
    if selection == "best":
        for row in rows:
            if row.get("epoch") == "best" and row.get("eval", "") != "":
                return float(row["eval"])
    numeric = [row for row in rows if row.get("epoch", "").isdigit() and row.get("eval", "") != ""]
    return float(numeric[-1]["eval"]) if numeric else None


def read_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def add_observation(
    observations: list[Observation],
    *,
    metric: str,
    method: str,
    pct: float,
    pct_label: str,
    seed: str,
    value: object,
    source: Path,
) -> None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(parsed):
        observations.append(Observation(metric, method, pct, pct_label, seed, parsed, source))


def collect(args: argparse.Namespace) -> list[Observation]:
    observations: list[Observation] = []
    baseline_keys = {
        "grid_acc": {
            "Clingo": "clingo_grid_acc",
            "NN baseline": "nn_eval_grid_acc",
            "RDProp": "rdprop_grid_acc",
        },
        "cell_acc": {
            "Clingo": "clingo_cell_acc",
            "NN baseline": "nn_eval_cell_acc",
            "RDProp": "rdprop_cell_acc",
        },
    }

    for pct_label in args.pcts:
        pct = float(pct_label)
        tag = pct_tag(pct_label)
        for seed in args.seeds:
            train_parent = args.log_root / f"train_pct_{tag}_seed_{seed}"
            train_run = latest_with(train_parent, "grid_acc.csv")
            if train_run is not None:
                for metric in METRICS:
                    source = train_run / f"{metric}.csv"
                    value = read_ndprop_metric(source, args.ndprop_epoch)
                    add_observation(
                        observations,
                        metric=metric,
                        method="NDProp",
                        pct=pct,
                        pct_label=pct_label,
                        seed=seed,
                        value=value,
                        source=source,
                    )

            baseline_parent = args.log_root / f"baseline_pct_{tag}_seed_{seed}"
            baseline_run = latest_with(baseline_parent, "mnist_sudoku_baselines.json")
            if baseline_run is None:
                continue
            source = baseline_run / "mnist_sudoku_baselines.json"
            data = read_json(source)
            for metric, method_keys in baseline_keys.items():
                for method, key in method_keys.items():
                    add_observation(
                        observations,
                        metric=metric,
                        method=method,
                        pct=pct,
                        pct_label=pct_label,
                        seed=seed,
                        value=data.get(key),
                        source=source,
                    )
    return observations


def grouped_values(observations: list[Observation]) -> dict[tuple[str, str, float], list[float]]:
    grouped: dict[tuple[str, str, float], list[float]] = {}
    for item in observations:
        grouped.setdefault((item.metric, item.method, item.pct), []).append(item.value)
    return grouped


def summarize(values: list[float]) -> tuple[float, Optional[float], int]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    return mean, std, len(values)


def write_raw(path: Path, observations: list[Observation]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "method", "pretrain_fraction", "seed", "value", "source"])
        for item in sorted(observations, key=lambda x: (x.metric, x.method, x.pct, x.seed)):
            writer.writerow([item.metric, item.method, item.pct_label, item.seed, f"{item.value:.8f}", item.source])


def write_summary(
    path: Path,
    grouped: dict[tuple[str, str, float], list[float]],
    pct_labels: dict[float, str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "method", "pretrain_fraction", "mean", "std", "n"])
        for (metric, method, pct), values in sorted(grouped.items()):
            mean, std, n = summarize(values)
            writer.writerow(
                [metric, method, pct_labels[pct], f"{mean:.8f}", "" if std is None else f"{std:.8f}", n]
            )


def plot(
    path: Path,
    grouped: dict[tuple[str, str, float], list[float]],
    pcts: list[float],
    pct_labels: dict[float, str],
    ndprop_epoch: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, metric in zip(axes, METRICS):
        for method in METHODS:
            if not any((metric, method, pct) in grouped for pct in pcts):
                continue
            means = []
            lower = []
            upper = []
            has_std = False
            for pct in pcts:
                values = grouped.get((metric, method, pct), [])
                if not values:
                    means.append(math.nan)
                    lower.append(math.nan)
                    upper.append(math.nan)
                    continue
                mean, std, _ = summarize(values)
                means.append(mean)
                if std is None:
                    lower.append(mean)
                    upper.append(mean)
                else:
                    has_std = True
                    lower.append(max(0.0, mean - std))
                    upper.append(min(1.0, mean + std))
            style = METHOD_STYLES[method]
            ax.plot(
                pcts,
                means,
                label=method,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0,
            )
            if has_std:
                ax.fill_between(pcts, lower, upper, color=style["color"], alpha=0.18, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks(pcts)
        ax.set_xticklabels([pct_labels[pct] for pct in pcts])
        ax.set_xlabel("Digit-backbone pretraining fraction")
        ax.set_title(METRIC_TITLES[metric])
        ax.grid(alpha=0.22)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.025)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Visual Sudoku: mean ± 1 sample standard deviation across seeds\nNDProp epoch selection: {ndprop_epoch}")
    fig.tight_layout(rect=(0, 0.10, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.log_root = args.log_root.resolve()
    output = (args.output or args.log_root / "visual_sudoku_analysis.png").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    observations = collect(args)
    if not observations:
        raise SystemExit(f"No Visual Sudoku metrics found under {args.log_root}.")
    grouped = grouped_values(observations)
    pct_labels = {float(label): label for label in args.pcts}
    pcts = sorted(pct_labels)

    plot(output, grouped, pcts, pct_labels, args.ndprop_epoch)
    write_summary(output.with_suffix(".csv"), grouped, pct_labels)
    write_raw(output.with_name(f"{output.stem}_samples.csv"), observations)
    print(output)


if __name__ == "__main__":
    main()
