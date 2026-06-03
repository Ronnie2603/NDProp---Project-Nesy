#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_SIZES = {
    "mnistAdd_train": 30000,
    "mnistAdd_test": 1000,
    "mnistAdd2_train": 10000,
    "mnistAdd2_test": 1000,
    "add2x2_train": 10000,
    "add2x2_test": 1000,
    "member3_train": 10000,
    "member3_test": 1000,
    "member5_train": 10000,
    "member5_test": 1000,
    "apply2x2_train": 10000,
    "apply2x2_test": 1000,
}


def _mnist_labels(root: Path, *, train: bool) -> Sequence[int]:
    from torchvision import datasets

    dataset = datasets.MNIST(str(root), train=train, download=True)
    return [int(label) for label in dataset.targets]


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def _sample_indices(rng: random.Random, labels: Sequence[int], n: int) -> list[int]:
    return [rng.randrange(0, len(labels)) for _ in range(n)]


def generate_addition(path: Path, labels: Sequence[int], *, size: int, seed: int) -> None:
    rng = random.Random(seed)
    lines = []
    for _ in range(size):
        i, j = _sample_indices(rng, labels, 2)
        lines.append(f"{i} {j} {labels[i] + labels[j]}")
    _write_lines(path, lines)


def generate_addition2(path: Path, labels: Sequence[int], *, size: int, seed: int) -> None:
    rng = random.Random(seed)
    lines = []
    for _ in range(size):
        i1, i2, i3, i4 = _sample_indices(rng, labels, 4)
        total = (labels[i1] * 10 + labels[i2]) + (labels[i3] * 10 + labels[i4])
        lines.append(f"{i1} {i2} {i3} {i4} {total}")
    _write_lines(path, lines)


def generate_add2x2(path: Path, labels: Sequence[int], *, size: int, seed: int) -> None:
    rng = random.Random(seed)
    lines = []
    for _ in range(size):
        i1, i2, i3, i4 = _sample_indices(rng, labels, 4)
        d1, d2, d3, d4 = labels[i1], labels[i2], labels[i3], labels[i4]
        lines.append(f"{i1} {i2} {i3} {i4} {d1 + d2} {d3 + d4} {d1 + d3} {d2 + d4}")
    _write_lines(path, lines)


def generate_member(path: Path, labels: Sequence[int], *, digits: int, size: int, seed: int) -> None:
    rng = random.Random(seed)
    lines = []
    for _ in range(size):
        idxs = _sample_indices(rng, labels, digits)
        query = rng.randrange(0, 10)
        label = 1 if query in [labels[i] for i in idxs] else 0
        lines.append(" ".join(str(x) for x in [*idxs, query, label]))
    _write_lines(path, lines)


def _operator_label(labels: Sequence[int], sample_idx: int) -> int:
    label = int(labels[sample_idx])
    if label not in (0, 1, 2):
        raise ValueError(
            "apply2x2 expects operator image classes encoded as labels 0, 1, 2 "
            "for plus, minus, multiply. Check the ImageFolder class order."
        )
    return label


def _apply(d1: int, op1: int, d2: int, op2: int, d3: int) -> int:
    if op1 == 0:
        left = d1 + d2
    elif op1 == 1:
        left = d1 - d2
    else:
        left = d1 * d2
    if op2 == 0:
        return left + d3
    if op2 == 1:
        return left - d3
    return left * d3


def generate_apply2x2(
    path: Path,
    *,
    op_root: Path,
    op_subdir: str,
    size: int,
    seed: int,
) -> None:
    from torchvision.datasets import ImageFolder

    op_path = op_root / op_subdir
    if not op_path.exists():
        raise FileNotFoundError(
            f"Operator image directory '{op_path}' not found. Provide --op-root and "
            "--apply-train-subdir/--apply-eval-subdir, or skip --apply2x2."
        )
    ops = ImageFolder(str(op_path))
    if len(ops) == 0:
        raise ValueError(f"Operator image directory '{op_path}' is empty.")

    rng = random.Random(seed)
    lines = []
    for _ in range(size):
        d1 = rng.randrange(0, 11)
        d2 = rng.randrange(0, 11)
        d3 = rng.randrange(0, 11)
        op_ids = [rng.randrange(0, len(ops)) for _ in range(4)]
        op_labels = [_operator_label(ops.targets, idx) for idx in op_ids]
        r1 = _apply(d1, op_labels[0], d2, op_labels[1], d3)
        r2 = _apply(d1, op_labels[2], d2, op_labels[3], d3)
        c1 = _apply(d1, op_labels[0], d2, op_labels[2], d3)
        c2 = _apply(d1, op_labels[1], d2, op_labels[3], d3)
        lines.append(
            f"{d1} {d2} {d3} {op_ids[0]} {op_ids[1]} {op_ids[2]} {op_ids[3]} {r1} {r2} {c1} {c2}"
        )
    _write_lines(path, lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate NDProp paper-experiment index files.")
    parser.add_argument("--indexes-root", type=Path, default=Path("indexes"), help="Output directory for index files.")
    parser.add_argument("--mnist-root", type=Path, default=Path("data"), help="MNIST download/cache root.")
    parser.add_argument("--seed", type=int, default=7, help="Base RNG seed.")
    parser.add_argument("--all", action="store_true", help="Generate all supported index files.")
    parser.add_argument("--mnist", action="store_true", help="Generate MNIST-backed task indexes.")
    parser.add_argument("--apply2x2", action="store_true", help="Generate apply2x2 pair files from provided operator images.")
    parser.add_argument("--op-root", type=Path, default=Path("assets/operator_images/hasy"), help="Operator image root for apply2x2.")
    parser.add_argument("--apply-train-subdir", default="train_apply2x2", help="Operator image train subdirectory.")
    parser.add_argument("--apply-eval-subdir", default="test_apply2x2", help="Operator image eval subdirectory.")
    parser.add_argument("--train-size", type=int, default=None, help="Override train size for generated files.")
    parser.add_argument("--eval-size", type=int, default=None, help="Override eval size for generated files.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.all and not args.mnist and not args.apply2x2:
        raise SystemExit("Choose at least one of --all, --mnist, or --apply2x2.")

    generate_mnist = args.all or args.mnist
    generate_apply = args.all or args.apply2x2

    if generate_mnist:
        train_labels = _mnist_labels(args.mnist_root, train=True)
        test_labels = _mnist_labels(args.mnist_root, train=False)
        train_size = args.train_size
        eval_size = args.eval_size

        generate_addition(
            args.indexes_root / "mnistAdd_train.txt",
            train_labels,
            size=train_size or DEFAULT_SIZES["mnistAdd_train"],
            seed=args.seed,
        )
        generate_addition(
            args.indexes_root / "mnistAdd_test.txt",
            test_labels,
            size=eval_size or DEFAULT_SIZES["mnistAdd_test"],
            seed=args.seed + 1,
        )
        generate_addition2(
            args.indexes_root / "mnistAdd2_train.txt",
            train_labels,
            size=train_size or DEFAULT_SIZES["mnistAdd2_train"],
            seed=args.seed + 2,
        )
        generate_addition2(
            args.indexes_root / "mnistAdd2_test.txt",
            test_labels,
            size=eval_size or DEFAULT_SIZES["mnistAdd2_test"],
            seed=args.seed + 3,
        )
        generate_add2x2(
            args.indexes_root / "add2x2_train.txt",
            train_labels,
            size=train_size or DEFAULT_SIZES["add2x2_train"],
            seed=args.seed + 4,
        )
        generate_add2x2(
            args.indexes_root / "add2x2_test.txt",
            test_labels,
            size=eval_size or DEFAULT_SIZES["add2x2_test"],
            seed=args.seed + 5,
        )
        generate_member(
            args.indexes_root / "member3_train.txt",
            train_labels,
            digits=3,
            size=train_size or DEFAULT_SIZES["member3_train"],
            seed=args.seed + 6,
        )
        generate_member(
            args.indexes_root / "member3_test.txt",
            test_labels,
            digits=3,
            size=eval_size or DEFAULT_SIZES["member3_test"],
            seed=args.seed + 7,
        )
        generate_member(
            args.indexes_root / "member5_train.txt",
            train_labels,
            digits=5,
            size=train_size or DEFAULT_SIZES["member5_train"],
            seed=args.seed + 8,
        )
        generate_member(
            args.indexes_root / "member5_test.txt",
            test_labels,
            digits=5,
            size=eval_size or DEFAULT_SIZES["member5_test"],
            seed=args.seed + 9,
        )

    if generate_apply:
        train_size = args.train_size or DEFAULT_SIZES["apply2x2_train"]
        eval_size = args.eval_size or DEFAULT_SIZES["apply2x2_test"]
        generate_apply2x2(
            args.indexes_root / "apply2x2_train_10000.txt",
            op_root=args.op_root,
            op_subdir=args.apply_train_subdir,
            size=train_size,
            seed=args.seed + 10,
        )
        generate_apply2x2(
            args.indexes_root / "apply2x2_test.txt",
            op_root=args.op_root,
            op_subdir=args.apply_eval_subdir,
            size=eval_size,
            seed=args.seed + 11,
        )


if __name__ == "__main__":
    main()
