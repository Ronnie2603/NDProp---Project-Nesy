#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torchvision import datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _base_solution(base: int) -> list[list[int]]:
    side = base * base
    return [[((base * (r % base) + r // base + c) % side) + 1 for c in range(side)] for r in range(side)]


def _shuffled_solution(base: int, rng: random.Random) -> list[list[int]]:
    side = base * base
    pattern = _base_solution(base)

    def _shuffle_groups() -> list[int]:
        groups = list(range(base))
        rng.shuffle(groups)
        result: list[int] = []
        for group in groups:
            inner = list(range(base))
            rng.shuffle(inner)
            result.extend(group * base + idx for idx in inner)
        return result

    rows = _shuffle_groups()
    cols = _shuffle_groups()
    symbols = list(range(1, side + 1))
    rng.shuffle(symbols)
    remap = {old: new for old, new in zip(range(1, side + 1), symbols)}
    return [[remap[pattern[r][c]] for c in cols] for r in rows]


def _digit_indices(dataset: datasets.MNIST, max_digit: int) -> dict[int, list[int]]:
    targets = dataset.targets
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)
    indices: dict[int, list[int]] = {}
    for digit in range(max_digit + 1):
        digit_indices = torch.nonzero(targets == digit, as_tuple=False).view(-1).tolist()
        if not digit_indices:
            raise ValueError(f"MNIST split has no images for digit {digit}.")
        indices[digit] = [int(idx) for idx in digit_indices]
    return indices


def _sample_image(
    *,
    data: torch.Tensor,
    digit_to_indices: dict[int, list[int]],
    digit: int,
    rng: random.Random,
) -> torch.Tensor:
    idx = rng.choice(digit_to_indices[digit])
    return data[idx].float().div(255.0)


def _generate_split(
    *,
    root: Path,
    mnist_root: Path,
    train: bool,
    size: int,
    base: int,
    clue_fraction: float,
    seed: int,
) -> tuple[Path, Path]:
    if not 0.0 <= clue_fraction <= 1.0:
        raise ValueError("--clue-fraction must be in [0, 1].")

    rng = random.Random(seed)
    side = base * base
    mnist = datasets.MNIST(str(mnist_root), train=train, download=True)
    digit_to_indices = _digit_indices(mnist, side)
    data = mnist.data

    images = torch.empty((size, side, side, 28, 28), dtype=torch.float32)
    labels = torch.empty((size, side, side), dtype=torch.long)
    for sample_idx in range(size):
        solution = _shuffled_solution(base, rng)
        for r in range(side):
            for c in range(side):
                value = solution[r][c]
                labels[sample_idx, r, c] = value - 1
                image_digit = value if rng.random() < clue_fraction else 0
                images[sample_idx, r, c] = _sample_image(
                    data=data,
                    digit_to_indices=digit_to_indices,
                    digit=image_digit,
                    rng=rng,
                )

    root.mkdir(parents=True, exist_ok=True)
    split_name = "train" if train else "test"
    features_path = root / f"{split_name}_features_img.pt"
    labels_path = root / f"{split_name}_labels.pt"
    torch.save(images, features_path)
    torch.save(labels, labels_path)
    return features_path, labels_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 4x4 MNIST Sudoku tensors for NDProp.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "mnist_sudoku")
    parser.add_argument("--mnist-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--base", type=int, default=2, help="Sudoku base; 2 produces the paper 4x4 task.")
    parser.add_argument(
        "--clue-fraction",
        type=float,
        default=0.5,
        help="Fraction of cells rendered as their solution digit; remaining cells use MNIST zero as blank.",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.base != 2:
        raise SystemExit("Only --base 2 is currently supported by the released MNIST Sudoku task.")
    if args.train_size <= 0 or args.test_size <= 0:
        raise SystemExit("--train-size and --test-size must be positive.")

    train_features, train_labels = _generate_split(
        root=args.output_root,
        mnist_root=args.mnist_root,
        train=True,
        size=args.train_size,
        base=args.base,
        clue_fraction=args.clue_fraction,
        seed=args.seed,
    )
    test_features, test_labels = _generate_split(
        root=args.output_root,
        mnist_root=args.mnist_root,
        train=False,
        size=args.test_size,
        base=args.base,
        clue_fraction=args.clue_fraction,
        seed=args.seed + 1,
    )
    metadata = {
        "base": args.base,
        "side": args.base * args.base,
        "clue_fraction": args.clue_fraction,
        "seed": args.seed,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "files": [
            train_features.name,
            train_labels.name,
            test_features.name,
            test_labels.name,
        ],
    }
    metadata_path = args.output_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {train_features}")
    print(f"Wrote {train_labels}")
    print(f"Wrote {test_features}")
    print(f"Wrote {test_labels}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
