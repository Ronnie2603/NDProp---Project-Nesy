from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot random ASP exact-match accuracy by difficulty.")
    parser.add_argument("plot_path", type=Path)
    parser.add_argument("summary", nargs="+", help="Triples: label success total.")
    args = parser.parse_args()

    if len(args.summary) % 3 != 0:
        raise SystemExit("Expected summary arguments in triples: label success total.")

    pretty_labels = []
    accuracies = []
    pairs = []
    for idx in range(0, len(args.summary), 3):
        label = args.summary[idx]
        success = int(args.summary[idx + 1])
        total = int(args.summary[idx + 2])
        accuracy = success / total if total else 0.0
        pretty = label[len("test_") :] if label.startswith("test_") else label
        pretty_labels.append(pretty.capitalize())
        accuracies.append(accuracy)
        pairs.append((success, total))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(pretty_labels, accuracies, color="#4c72b0", alpha=0.85)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Exact-match accuracy")
    ax.set_xlabel("Difficulty")
    ax.set_title("NDProp random ASP accuracy by difficulty")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    for bar, (success, total), accuracy in zip(bars, pairs, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 0.02,
            f"{accuracy * 100:.1f}%\n({success}/{total})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    plot_path = args.plot_path.resolve()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    print(plot_path)


if __name__ == "__main__":
    main()
