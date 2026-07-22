from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


FIELDS = [
    "total_train_time",
    "eval_time_last_epoch",
    "mnist_test_acc",
    "inference_acc",
]


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize neuro-symbolic task results.csv.")
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    args = parser.parse_args()

    data = defaultdict(lambda: {field: [] for field in FIELDS})
    with args.results_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = row.get("task", "")
            if not task:
                continue
            for field in FIELDS:
                value = to_float(row.get(field, ""))
                if value is not None:
                    data[task][field].append(value)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "metric", "mean", "std", "n"])
        for task in sorted(data.keys()):
            for field in FIELDS:
                values = data[task][field]
                mean, std = mean_std(values)
                writer.writerow(
                    [
                        task,
                        field,
                        "" if mean is None else f"{mean:.6f}",
                        "" if std is None else f"{std:.6f}",
                        len(values),
                    ]
                )

    print(f"Wrote {args.summary_csv}")


if __name__ == "__main__":
    main()
