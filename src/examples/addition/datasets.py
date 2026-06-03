from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class AdditionDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        train: bool,
        *,
        pairs_per_sample: int = 1,
        pairs_file: Optional[str | Path] = None,
        seed: int = 0,
        limit: Optional[int] = None,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        if pairs_per_sample < 1:
            raise ValueError("pairs_per_sample must be at least 1.")
        self.root = str(root)
        self.train = train
        self.pairs_per_sample = pairs_per_sample
        self.pairs_file = Path(pairs_file) if pairs_file is not None else None
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
        if self.pairs_file is not None:
            self.indices = self._load_pairs_from_file(limit=limit)
        else:
            self.indices: list[Tuple[int, int]] = self._sample_pairs(limit=limit)

    def _load_pairs_from_file(self, *, limit: Optional[int]) -> list[Tuple[int, int, int]]:
        pairs: list[Tuple[int, int, int]] = []
        with self.pairs_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                pairs.append((int(parts[0]), int(parts[1]), int(parts[2])))
                if limit is not None and limit >= 0 and len(pairs) >= limit:
                    break
        return pairs

    def _sample_pairs(self, *, limit: Optional[int]) -> list[Tuple[int, int, int]]:
        population = list(range(len(self.mnist)))
        labels = self.mnist.targets
        pairs: list[Tuple[int, int, int]] = []
        for idx in population:
            for _ in range(self.pairs_per_sample):
                partner = self.rng.choice(population)
                total = int(labels[idx]) + int(labels[partner])
                pairs.append((idx, partner, total))
        self.rng.shuffle(pairs)
        if limit is not None and limit >= 0:
            return pairs[:limit]
        return pairs

    def resample(self, *, seed: Optional[int] = None, limit: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = random.Random(seed)
        if self.pairs_file is not None:
            self.indices = self._load_pairs_from_file(limit=limit)
        else:
            self.indices = self._sample_pairs(limit=limit)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        raw_pair = self.indices[idx]
        idx_a, idx_b = raw_pair[0], raw_pair[1]
        img_a, _ = self.mnist[idx_a]
        img_b, _ = self.mnist[idx_b]
        sum_label = raw_pair[2]

        return (
            img_a,
            img_b,
            torch.tensor(sum_label, dtype=torch.long),
        )


def addition_collate(batch: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> dict:
    imgs_a = torch.stack([item[0] for item in batch], dim=0)
    imgs_b = torch.stack([item[1] for item in batch], dim=0)
    sums = torch.stack([item[2] for item in batch], dim=0)
    return {
        "img_a": imgs_a,
        "img_b": imgs_b,
        "sum": sums
    }
