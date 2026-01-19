from __future__ import annotations

import random
from pathlib import Path
import re
from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class Apply2x2Dataset(Dataset):
    """
    Mirrors the NeurASP apply2x2 dataset: each line has eleven integers
    (d1 d2 d3 o1 o2 o3 o4 r1 r2 c1 c2) where d* are digit values,
    o* are operator image indices, and r/c are row/col results.
    """

    def __init__(
        self,
        op_root: str | Path,
        *,
        pairs_file: str | Path,
        limit: int = 3000,
        seed: int = 1,
        transform: Optional[transforms.Compose] = None,
        train: bool = True,
        op_train_subdir: str = "train_apply2x2",
        op_eval_subdir: str = "test_apply2x2",
    ) -> None:
        self.pairs_file = Path(pairs_file)
        self.limit = limit
        self.rng = random.Random(seed)
        self.transform = transform or transforms.Compose(
            [transforms.Grayscale(1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        if not self.pairs_file.exists():
            raise FileNotFoundError(f"Pairs file '{self.pairs_file}' not found.")

        split = op_train_subdir if train else op_eval_subdir
        op_path = Path(op_root) / split
        if not op_path.exists():
            raise FileNotFoundError(f"Operator dataset '{op_path}' not found.")
        self.ops = datasets.ImageFolder(root=str(op_path), transform=self.transform)
        

        self.examples: list[Tuple[int, int, int, int, int, int, int, int, int, int, int]] = []
        with self.pairs_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = [int(x) for x in line.strip().split()]
                if len(parts) != 11:
                    continue
                self.examples.append(tuple(parts))
        if self.limit is not None and self.limit >= 0 and self.limit < len(self.examples):
            print(f"Applying limit: {self.limit} / {len(self.examples)} examples.")
            self.rng.seed(seed)
            candidates = self.examples
            if not train:
                op_ids = self._extract_op_ids()
                if op_ids:
                    filtered = [ex for ex in self.examples if all(op in op_ids for op in ex[3:7])]
                    if filtered:
                        candidates = filtered
            limit = min(self.limit, len(candidates))
            self.examples = [candidates[i] for i in self.rng.sample(range(len(candidates)), limit)]

    def _extract_op_ids(self) -> set[int]:
        ids: set[int] = set()
        for path, _ in self.ops.samples:
            stem = Path(path).stem
            match = re.search(r"\d+", stem)
            if match:
                ids.add(int(match.group(0)))
        return ids


    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        d1, d2, d3, o1, o2, o3, o4, r1, r2, c1, c2 = self.examples[idx]
        imgs = [self.ops[o1][0], self.ops[o2][0], self.ops[o3][0], self.ops[o4][0]]
        return {
            "imgs": torch.stack(imgs, dim=0),
            "sums": torch.tensor([r1, r2, c1, c2], dtype=torch.long),
            "digits": torch.tensor([d1, d2, d3], dtype=torch.long),
        }


def apply2x2_collate(batch: Sequence[dict]) -> dict:
    imgs = torch.stack([item["imgs"] for item in batch], dim=0)
    sums = torch.stack([item["sums"] for item in batch], dim=0)
    digits = torch.stack([item["digits"] for item in batch], dim=0)
    return {"imgs": imgs, "sums": sums, "digits": digits}



def generate_apply2x2_pairs(
    *,
    op_root: str | Path,
    op_eval_subdir: str,
    out_path: str | Path,
    size: int,
    seed: int,
) -> Path:
    op_path = Path(op_root) / op_eval_subdir
    if not op_path.exists():
        raise FileNotFoundError(f"Operator dataset '{op_path}' not found.")
    ops = datasets.ImageFolder(root=str(op_path))
    if len(ops) == 0:
        raise ValueError(f"No operator images found in '{op_path}'.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    m = len(ops)

    def _apply(d1: int, o1: int, d2: int, o2: int, d3: int) -> int:
        if o1 == 0:
            left = d1 + d2
        elif o1 == 1:
            left = d1 - d2
        else:
            left = d1 * d2
        if o2 == 0:
            return left + d3
        if o2 == 1:
            return left - d3
        return left * d3

    lines = []
    for _ in range(size):
        d1 = rng.randrange(0, 11)
        d2 = rng.randrange(0, 11)
        d3 = rng.randrange(0, 11)
        op_ids = [rng.randrange(0, m) for _ in range(4)]
        op_labels = [int(ops.samples[i][1]) for i in op_ids]
        r1 = _apply(d1, op_labels[0], d2, op_labels[1], d3)
        r2 = _apply(d1, op_labels[2], d2, op_labels[3], d3)
        c1 = _apply(d1, op_labels[0], d2, op_labels[2], d3)
        c2 = _apply(d1, op_labels[1], d2, op_labels[3], d3)
        lines.append(
            f"{d1} {d2} {d3} {op_ids[0]} {op_ids[1]} {op_ids[2]} {op_ids[3]} {r1} {r2} {c1} {c2}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path