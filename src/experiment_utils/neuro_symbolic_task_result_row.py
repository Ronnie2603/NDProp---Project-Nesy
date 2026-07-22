from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRIC_MAP = {
    "addition": "sum_acc",
    "addition2": "sum_acc",
    "add2x2": "total_acc",
    "member3": "member_acc",
    "member5": "member_acc",
    "apply2x2": "total_acc",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def last_epoch_value(rows: list[dict[str, str]], column: str) -> tuple[float | None, int | None]:
    last_epoch = None
    last_value = None
    for row in rows:
        epoch = row.get("epoch", "")
        if epoch.isdigit():
            value = row.get(column, "")
            if value != "":
                last_epoch = int(epoch)
                last_value = float(value)
    return last_value, last_epoch


def best_epoch_value(
    rows: list[dict[str, str]],
    column: str,
    *,
    ignore_last_epoch: bool = False,
) -> tuple[float | None, int | None]:
    best_epoch = None
    best_value = None
    epoch_rows = [row for row in rows if row.get("epoch", "").isdigit()]
    if ignore_last_epoch and len(epoch_rows) > 1:
        epoch_rows = epoch_rows[:-1]
    for row in epoch_rows:
        value = row.get(column, "")
        if value != "":
            candidate = float(value)
            if best_value is None or candidate > best_value:
                best_value = candidate
                best_epoch = int(row["epoch"])
    return best_value, best_epoch


def value_at_epoch(rows: list[dict[str, str]], column: str, epoch: int | None) -> float | None:
    if epoch is None:
        return None
    for row in rows:
        if row.get("epoch") == str(epoch):
            value = row.get(column, "")
            return float(value) if value != "" else None
    return None


def total_train_time(rows: list[dict[str, str]]) -> float | None:
    for row in rows:
        if row.get("epoch") == "total":
            value = row.get("train", "")
            return float(value) if value != "" else None
    total = 0.0
    for row in rows:
        epoch = row.get("epoch", "")
        if epoch.isdigit():
            value = row.get("train", "")
            if value != "":
                total += float(value)
    return total if total > 0 else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one neuro-symbolic task results.csv row.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repeat", required=True)
    parser.add_argument("--seed", required=True)
    args = parser.parse_args()

    metric_name = METRIC_MAP[args.task]
    time_rows = read_rows(args.run_dir / "time.csv")
    total_train = total_train_time(time_rows)

    digit_metric = "mnist_test_acc" if args.task != "apply2x2" else "op_test_acc"
    digit_best, digit_best_epoch = best_epoch_value(
        read_rows(args.run_dir / f"{digit_metric}.csv"),
        "eval",
        ignore_last_epoch=True,
    )
    eval_time = value_at_epoch(time_rows, "eval", digit_best_epoch)
    inference_acc = value_at_epoch(read_rows(args.run_dir / f"{metric_name}.csv"), "eval", digit_best_epoch)

    fields = [
        args.task,
        args.repeat,
        args.seed,
        str(args.run_dir),
        fmt(total_train),
        fmt(eval_time),
        fmt(digit_best),
        fmt(inference_acc),
        metric_name,
    ]
    print(",".join(fields))


if __name__ == "__main__":
    main()
