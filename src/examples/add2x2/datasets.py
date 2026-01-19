from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class Add2x2Dataset(Dataset):
    """
    Mirrors the NeurASP add2x2 dataset: each line has eight integers
    (i1 i2 i3 i4 r1 r2 c1 c2) where i* are MNIST indices and r/c are row/col sums.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        pairs_file: str | Path,
        limit: int = 3000,
        seed: int = 1,
        transform: Optional[transforms.Compose] = None,
        train: bool = True,
    ) -> None:
        self.root = str(root)
        self.pairs_file = Path(pairs_file)
        self.limit = limit
        self.rng = random.Random(seed)
        self.transform = transform or transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.mnist = datasets.MNIST(
            root=self.root,
            train=train,
            download=True,
            transform=self.transform,
        )
        if not self.pairs_file.exists():
            raise FileNotFoundError(f"Pairs file '{self.pairs_file}' not found.")
        self.examples: list[Tuple[int, int, int, int, int, int, int, int]] = []
        with self.pairs_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = [int(x) for x in line.strip().split()]
                if len(parts) != 8:
                    continue
                self.examples.append(tuple(parts))
        if self.limit is not None and self.limit >= 0 and self.limit < len(self.examples):
            self.rng.seed(seed)
            self.examples = [self.examples[i] for i in self.rng.sample(range(len(self.examples)), self.limit)]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        i1, i2, i3, i4, r1, r2, c1, c2 = self.examples[idx]
        m = len(self.mnist)
        i1, i2, i3, i4 = i1 % m, i2 % m, i3 % m, i4 % m
        imgs = [self.mnist[i1][0], self.mnist[i2][0], self.mnist[i3][0], self.mnist[i4][0]]
        digits = [self.mnist[i1][1], self.mnist[i2][1], self.mnist[i3][1], self.mnist[i4][1]]
        row0 = digits[0] + digits[1]
        row1 = digits[2] + digits[3]
        col0 = digits[0] + digits[2]
        col1 = digits[1] + digits[3]
        return {
            "imgs": torch.stack(imgs, dim=0),
            "sums": torch.tensor([row0, row1, col0, col1], dtype=torch.long),
            "digits": torch.tensor(digits, dtype=torch.long),
        }


def add2x2_collate(batch: Sequence[dict]) -> dict:
    imgs = torch.stack([item["imgs"] for item in batch], dim=0)
    sums = torch.stack([item["sums"] for item in batch], dim=0)
    out = {"imgs": imgs, "sums": sums}
    if "digits" in batch[0]:
        out["digits"] = torch.stack([item["digits"] for item in batch], dim=0)
    return out
