from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class Addition2Dataset(Dataset):
    """
    Four MNIST digits representing a 2-digit + 2-digit addition problem (mnistAdd2).
    Each line in the pairs file: i1 i2 i3 i4 sum where indices reference MNIST train split.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        pairs_file: str | Path,
        limit: Optional[int] = None,
        seed: int = 0,
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
        if not self.pairs_file.exists():
            raise FileNotFoundError(f"Pairs file '{self.pairs_file}' not found.")
        self.mnist = datasets.MNIST(
            root=self.root,
            train=train,
            download=True,
            transform=self.transform,
        )
        self.examples: list[Tuple[int, int, int, int, int]] = []
        with self.pairs_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = [int(x) for x in line.strip().split()]
                if len(parts) != 5:
                    continue
                self.examples.append(tuple(parts))
        if self.limit is not None and self.limit >= 0 and self.limit < len(self.examples):
            self.examples = [
                self.examples[idx] for idx in self.rng.sample(range(len(self.examples)), self.limit)
            ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        i1, i2, i3, i4, total = self.examples[idx]
        imgs = [self.mnist[i1][0], self.mnist[i2][0], self.mnist[i3][0], self.mnist[i4][0]]
        digits = [self.mnist[i1][1], self.mnist[i2][1], self.mnist[i3][1], self.mnist[i4][1]]
        return {
            "imgs": torch.stack(imgs, dim=0),
            "digits": torch.tensor(digits, dtype=torch.long),
            "sum": torch.tensor(total, dtype=torch.long),
        }


def addition2_collate(batch: Sequence[dict]) -> dict:
    imgs = torch.stack([item["imgs"] for item in batch], dim=0)
    digits = torch.stack([item["digits"] for item in batch], dim=0)
    sums = torch.stack([item["sum"] for item in batch], dim=0)
    return {"imgs": imgs, "digits": digits, "sum": sums}
