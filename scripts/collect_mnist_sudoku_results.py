#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple


ROOT = Path("/home/moriyama/ASP/GNN-ASP/runs_nesy/mnist_sudoku_exp2")


def _parse_parent(name: str) -> Tuple[str, str, str]:
    # Expected: <kind>_pct_<pct>_seed_<seed>
    kind = "unknown"
    pct = ""
    seed = ""
    parts = name.split("_")
    if len(parts) >= 5 and parts[1] == "pct" and parts[3] == "seed":
        kind = parts[0]
        pct = parts[2].replace("p", ".")
        seed = parts[4]
    return kind, pct, seed


def _read_best_csv(path: Path) -> Tuple[str, str]:
    if not path.exists():
        return "", ""
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return "", ""
    header = rows[0]
    best = None
    for row in rows[1:]:
        if row and row[0] == "best":
            best = row
    if best is None:
        best = rows[-1]
    out = {}
    for idx, key in enumerate(header):
        if idx < len(best):
            out[key] = best[idx]
    return out.get("train", ""), out.get("eval", "")


def _read_baselines(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {f"baseline_{k}": str(v) for k, v in data.items()}


def _read_pretrain_log(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    accs = [float(m.group(1)) for m in re.finditer(r"acc=([0-9.]+)", text)]
    out: Dict[str, str] = {}
    if accs:
        out["pretrain_acc_best"] = f"{max(accs):.6f}"
        out["pretrain_acc_last"] = f"{accs[-1]:.6f}"
    return out


def _iter_runs(root: Path) -> Iterable[Path]:
    for parent in sorted(root.glob("*_pct_*_seed_*")):
        for run_dir in sorted(parent.glob("run_*")):
            if run_dir.is_dir():
                yield run_dir


def main() -> None:
    rows = []
    all_keys = set()
    for run_dir in _iter_runs(ROOT):
        parent = run_dir.parent
        kind, pct, seed = _parse_parent(parent.name)
        record: Dict[str, str] = {
            "run_dir": str(run_dir),
            "kind": kind,
            "pct": pct,
            "seed": seed,
        }

        args_path = run_dir / "args.json"
        if args_path.exists():
            args = json.loads(args_path.read_text(encoding="utf-8"))
            for k, v in args.items():
                record[f"arg_{k}"] = str(v)

        if kind == "train":
            train_cell, eval_cell = _read_best_csv(run_dir / "cell_acc.csv")
            train_grid, eval_grid = _read_best_csv(run_dir / "grid_acc.csv")
            train_loss, eval_loss = _read_best_csv(run_dir / "loss.csv")
            train_bad, eval_bad = _read_best_csv(run_dir / "bad_bin_loss.csv")
            record.update(
                {
                    "train_cell_acc_best": train_cell,
                    "eval_cell_acc_best": eval_cell,
                    "train_grid_acc_best": train_grid,
                    "eval_grid_acc_best": eval_grid,
                    "train_loss_best": train_loss,
                    "eval_loss_best": eval_loss,
                    "train_bad_bin_loss_best": train_bad,
                    "eval_bad_bin_loss_best": eval_bad,
                }
            )
        elif kind == "baseline":
            record.update(_read_baselines(run_dir / "mnist_sudoku_baselines.json"))
        elif kind == "pretrain":
            record.update(_read_pretrain_log(run_dir / "train.log"))

        rows.append(record)
        all_keys.update(record.keys())

    if not rows:
        raise SystemExit(f"No runs found under {ROOT}.")

    base_order = ["run_dir", "kind", "pct", "seed"]
    keys = base_order + sorted(k for k in all_keys if k not in base_order)
    out_path = ROOT / "mnist_sudoku_experiment_summary.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
