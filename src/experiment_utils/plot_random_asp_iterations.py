from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot random ASP exact-match accuracy by iteration.")
    parser.add_argument("plot_path", type=Path)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    args = parser.parse_args()

    if len(args.label) != len(args.manifest):
        raise SystemExit("--label and --manifest must be supplied the same number of times.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, manifest in zip(args.label, args.manifest):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        iter_lists = [
            case.get("iter_exact_match", [])
            for case in data.get("cases", [])
            if case.get("iter_exact_match")
        ]
        if not iter_lists:
            continue
        min_len = min(len(values) for values in iter_lists)
        accuracies = [
            sum(1 for values in iter_lists if values[idx]) / len(iter_lists)
            for idx in range(min_len)
        ]
        ax.plot(range(len(accuracies)), accuracies, marker="o", linewidth=1.2, label=label)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Exact-match accuracy")
    ax.set_title("NDProp random ASP accuracy vs. iteration")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    plot_path = args.plot_path.resolve()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    print(plot_path)


if __name__ == "__main__":
    main()
