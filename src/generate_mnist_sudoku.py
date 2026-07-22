#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
from typing import List

import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DATA_FILES = (
    "train_features.pt",
    "train_features_img.pt",
    "train_labels.pt",
    "test_features.pt",
    "test_features_img.pt",
    "test_labels.pt",
    "perm.pt",
)


def dataset_is_complete(out_dir: Path) -> bool:
    return all((out_dir / filename).is_file() for filename in REQUIRED_DATA_FILES)


def _pattern(r: int, c: int, base: int, side: int) -> int:
    return (base * (r % base) + r // base + c) % side


def _shuffle(rng: np.random.Generator, seq: List[int]) -> List[int]:
    seq = list(seq)
    rng.shuffle(seq)
    return seq


def _make_solution(rng: np.random.Generator, base: int) -> np.ndarray:
    side = base * base
    rows = [g * base + r for g in _shuffle(rng, list(range(base))) for r in _shuffle(rng, list(range(base)))]
    cols = [g * base + c for g in _shuffle(rng, list(range(base))) for c in _shuffle(rng, list(range(base)))]
    nums = _shuffle(rng, list(range(1, side + 1)))
    board = np.array([[nums[_pattern(r, c, base, side)] for c in cols] for r in rows], dtype=np.int64)
    return board


def _mask_board(rng: np.random.Generator, board: np.ndarray, clues: int) -> np.ndarray:
    side = board.shape[0]
    total = side * side
    if clues <= 0 or clues > total:
        raise ValueError("clues must be in (0, side*side].")
    mask = np.zeros(total, dtype=bool)
    mask[:clues] = True
    rng.shuffle(mask)
    masked = board.copy().reshape(-1)
    masked[~mask] = 0
    return masked.reshape(side, side)


def _count_solutions(board: np.ndarray, base: int, limit: int = 2) -> int:
    side = base * base
    rows = [set() for _ in range(side)]
    cols = [set() for _ in range(side)]
    blocks = [set() for _ in range(side)]
    empties: List[tuple[int, int]] = []

    def block_index(r: int, c: int) -> int:
        return (r // base) * base + (c // base)

    for r in range(side):
        for c in range(side):
            val = int(board[r, c])
            if val == 0:
                empties.append((r, c))
            else:
                b = block_index(r, c)
                if val in rows[r] or val in cols[c] or val in blocks[b]:
                    return 0
                rows[r].add(val)
                cols[c].add(val)
                blocks[b].add(val)

    digits = list(range(1, side + 1))
    count = 0

    def search(pos: int) -> None:
        nonlocal count
        if count >= limit:
            return
        if pos >= len(empties):
            count += 1
            return
        r, c = empties[pos]
        b = block_index(r, c)
        used = rows[r] | cols[c] | blocks[b]
        candidates = [d for d in digits if d not in used]
        for d in candidates:
            rows[r].add(d)
            cols[c].add(d)
            blocks[b].add(d)
            search(pos + 1)
            rows[r].remove(d)
            cols[c].remove(d)
            blocks[b].remove(d)
            if count >= limit:
                return

    search(0)
    return count


def _load_class_images(root: Path, digits: List[int], *, train: bool, dataset: str) -> List[torch.Tensor]:
    transform = transforms.ToTensor()
    if dataset == "mnist":
        ds = datasets.MNIST(str(root), train=train, download=True, transform=transform)
    elif dataset == "fashion_mnist":
        ds = datasets.FashionMNIST(str(root), train=train, download=True, transform=transform)
    elif dataset == "cifar10":
        ds = datasets.CIFAR10(str(root), train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    buckets: List[List[torch.Tensor]] = [[] for _ in digits]
    for img, label in ds:
        if int(label) in digits:
            idx = digits.index(int(label))
            buckets[idx].append(img)
    for idx, bucket in enumerate(buckets):
        if not bucket:
            raise ValueError(f"No MNIST images found for digit {digits[idx]}.")
    return [torch.stack(bucket, dim=0) for bucket in buckets]


def _sample_images(
    rng: np.random.Generator,
    digit_images: List[torch.Tensor],
    digits: List[int],
    board: np.ndarray,
) -> torch.Tensor:
    side = board.shape[0]
    channels, height, width = digit_images[0].shape[1:]
    out = torch.zeros((side, side, channels, height, width), dtype=torch.float32)
    for r in range(side):
        for c in range(side):
            val = int(board[r, c])
            if val == 0:
                digit_idx = 0
            else:
                digit_idx = digits.index(val) + 1
            choices = digit_images[digit_idx]
            pick = rng.integers(0, choices.size(0))
            out[r, c] = choices[pick]
    return out


def _one_hot(board: np.ndarray, digits: List[int]) -> torch.Tensor:
    side = board.shape[0]
    num_digits = len(digits)
    out = torch.zeros((side, side, num_digits), dtype=torch.float32)
    for r in range(side):
        for c in range(side):
            val = int(board[r, c])
            if val == 0:
                continue
            idx = digits.index(val)
            out[r, c, idx] = 1.0
    return out


def generate_dataset(
    *,
    out_dir: Path,
    train_count: int,
    test_count: int,
    base: int = 2,
    clues: int = 12,
    mnist_root: Path,
    seed: int = 0,
    max_tries: int = 100000,
    image_dataset: str = "mnist",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    digits = list(range(1, base * base + 1))
    digit_images_train = _load_class_images(mnist_root, [0] + digits, train=True, dataset=image_dataset)
    digit_images_test = _load_class_images(mnist_root, [0] + digits, train=False, dataset=image_dataset)

    total = train_count + test_count
    side = base * base
    train_features = torch.zeros((train_count, side, side, side), dtype=torch.float32)
    train_labels = torch.zeros((train_count, side, side, side), dtype=torch.float32)
    channels, height, width = digit_images_train[0].shape[1:]
    train_features_img = torch.zeros((train_count, side, side, channels, height, width), dtype=torch.float32)
    test_features = torch.zeros((test_count, side, side, side), dtype=torch.float32)
    test_labels = torch.zeros((test_count, side, side, side), dtype=torch.float32)
    test_features_img = torch.zeros((test_count, side, side, channels, height, width), dtype=torch.float32)

    filled = 0
    tries = 0
    with tqdm(total=total) as pbar:
        while filled < total:
            if tries >= max_tries:
                raise RuntimeError(f"Failed to generate {total} unique puzzles after {max_tries} attempts.")
            tries += 1
            solution = _make_solution(rng, base)
            puzzle = _mask_board(rng, solution, rng.integers(clues, 13))
            if _count_solutions(puzzle, base, limit=2) != 1:
                continue
            if filled < train_count:
                dst_labels = train_labels
                dst_features = train_features
                dst_images = train_features_img
                use_images = digit_images_train
                dst_idx = filled
            else:
                dst_labels = test_labels
                dst_features = test_features
                dst_images = test_features_img
                use_images = digit_images_test
                dst_idx = filled - train_count
            dst_labels[dst_idx] = _one_hot(solution, digits)
            dst_features[dst_idx] = _one_hot(puzzle, digits)
            dst_images[dst_idx] = _sample_images(rng, use_images, digits, puzzle)
            filled += 1
            pbar.update(1)

    perm = torch.arange(side * side, dtype=torch.long)

    torch.save(train_features, out_dir / "train_features.pt")
    torch.save(train_features_img, out_dir / "train_features_img.pt")
    torch.save(train_labels, out_dir / "train_labels.pt")
    torch.save(test_features, out_dir / "test_features.pt")
    torch.save(test_features_img, out_dir / "test_features_img.pt")
    torch.save(test_labels, out_dir / "test_labels.pt")
    torch.save(perm, out_dir / "perm.pt")


def ensure_dataset(
    *,
    out_dir: Path,
    mnist_root: Path,
    image_dataset: str = "mnist",
    train_count: int = 10000,
    test_count: int = 1000,
    base: int = 2,
    clues: int = 4,
    seed: int = 0,
    max_tries: int = 1000000,
) -> bool:
    """Generate a missing Visual Sudoku dataset once, including across parallel jobs."""
    if dataset_is_complete(out_dir):
        return False

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir.parent / f".{out_dir.name}.generate.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if dataset_is_complete(out_dir):
            return False

        missing = [name for name in REQUIRED_DATA_FILES if not (out_dir / name).is_file()]
        print(
            f"Visual Sudoku data is incomplete in {out_dir}; "
            f"generating it now (missing: {', '.join(missing)})."
        )
        generate_dataset(
            out_dir=out_dir,
            train_count=train_count,
            test_count=test_count,
            base=base,
            clues=clues,
            mnist_root=mnist_root,
            seed=seed,
            max_tries=max_tries,
            image_dataset=image_dataset,
        )
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "mnist_sudoku")
    parser.add_argument("--train-count", type=int, default=10000)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--base", type=int, default=2)
    parser.add_argument("--clues", type=int, default=4)
    parser.add_argument("--mnist-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--image-dataset", choices=["mnist", "fashion_mnist", "cifar10"], default="mnist")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tries", type=int, default=1000000)
    args = parser.parse_args()

    generate_dataset(
        out_dir=args.out_dir,
        train_count=args.train_count,
        test_count=args.test_count,
        base=args.base,
        clues=args.clues,
        mnist_root=args.mnist_root,
        seed=args.seed,
        max_tries=args.max_tries,
        image_dataset=args.image_dataset,
    )


if __name__ == "__main__":
    main()
