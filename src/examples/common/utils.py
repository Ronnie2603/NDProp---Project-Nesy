from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def default_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


def move_batch(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def use_random_eval(eval_pairs: Path | None, train_pairs: Path | None) -> bool:
    if eval_pairs is None or not eval_pairs.exists():
        print("Using random evaluation pairs.")
        return True
    if train_pairs is None or not train_pairs.exists():
        return False
    try:
        return eval_pairs.resolve() == train_pairs.resolve()
    except FileNotFoundError:
        return False


class RandomDigitTupleDataset(Dataset):
    """
    Simple MNIST-based generator for synthetic evaluation or fallback training splits.
    """

    def __init__(
        self,
        *,
        root: Path,
        num_digits: int,
        size: int,
        seed: int,
        label_mode: str,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.num_digits = num_digits
        self.size = size
        self.transform = transform or default_transform()
        # Synthetic data should always be drawn from the MNIST test split.
        self.mnist = datasets.MNIST(str(root), train=False, download=True, transform=self.transform)
        self.rng = random.Random(seed)
        self.indices = [
            [self.rng.randrange(0, len(self.mnist)) for _ in range(self.num_digits)]
            for _ in range(size)
        ]
        self.label_mode = label_mode
        self.query_digits = None
        if self.label_mode in ("member3", "member5"):
            self.query_digits = [self.rng.randrange(0, 10) for _ in range(size)]
            

    def __len__(self) -> int:
        return self.size

    def _addition2_sum(self, digits) -> int:
        return (digits[0] * 10 + digits[1]) + (digits[2] * 10 + digits[3])

    def _addition3_sum(self, digits) -> int:
        left = digits[0] * 100 + digits[1] * 10 + digits[2]
        right = digits[3] * 100 + digits[4] * 10 + digits[5]
        return left + right

    def _grid_sums(self, digits) -> torch.Tensor:
        row0 = digits[0] + digits[1]
        row1 = digits[2] + digits[3]
        col0 = digits[0] + digits[2]
        col1 = digits[1] + digits[3]
        return torch.tensor([row0, row1, col0, col1], dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict:
        mnist_indices = self.indices[idx]
        imgs = [self.mnist[i][0] for i in mnist_indices]
        digits = [int(self.mnist[i][1]) for i in mnist_indices]
        sample: Dict[str, torch.Tensor] = {"imgs": torch.stack(imgs, dim=0)}

        if self.label_mode == "addition2":
            sample["digits"] = torch.tensor(digits, dtype=torch.long)
            sample["sum"] = torch.tensor(self._addition2_sum(digits), dtype=torch.long)
        elif self.label_mode == "addition3":
            sample["digits"] = torch.tensor(digits, dtype=torch.long)
            sample["sum"] = torch.tensor(self._addition3_sum(digits), dtype=torch.long)
        elif self.label_mode == "add2x2":
            sample["digits"] = torch.tensor(digits, dtype=torch.long)
            sample["sums"] = self._grid_sums(digits)
        elif self.label_mode in ("member3", "member5"):
            if self.query_digits is None:
                raise ValueError(f"query_digits must be initialized for {self.label_mode} mode.")
            query = int(self.query_digits[idx])
            sample["digits"] = torch.tensor(digits, dtype=torch.long)
            sample["query"] = torch.tensor(query, dtype=torch.long)
            sample["member"] = torch.tensor(1 if query in digits else 0, dtype=torch.long)
        else:
            raise ValueError(f"Unknown label_mode '{self.label_mode}'")
        return sample
