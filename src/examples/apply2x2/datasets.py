from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

LOGGER = logging.getLogger(__name__)


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
        self.op_index_map = self._build_op_index_map(op_root=Path(op_root), train=train)
        self.examples = self._filter_available_examples()
        if not self.examples:
            raise ValueError(
                f"No apply2x2 examples in '{self.pairs_file}' match the operator images in '{op_path}'."
            )
        if self.limit is not None and self.limit >= 0 and self.limit < len(self.examples):
            LOGGER.info("Applying limit: %d / %d examples.", self.limit, len(self.examples))
            self.rng.seed(seed)
            candidates = self.examples
            limit = min(self.limit, len(candidates))
            self.examples = [candidates[i] for i in self.rng.sample(range(len(candidates)), limit)]

    def _build_op_index_map(self, *, op_root: Path, train: bool) -> dict[int, int] | None:
        if not self.examples:
            return None
        op_values = [op for example in self.examples for op in example[3:7]]
        op_min = min(op_values)
        op_max = max(op_values)
        if op_min < 0:
            raise ValueError(f"Found negative operator index in '{self.pairs_file}'.")
        if op_max < len(self.ops):
            return None

        op_index_map = self._build_map_from_original_train(op_root, op_values)
        if op_index_map is None:
            op_index_map = self._build_builtin_apply2x2_map(op_values)
        if op_index_map is not None:
            LOGGER.info(
                "Using apply2x2 operator index map for %s pairs.",
                "train" if train else "eval",
            )
            return op_index_map

        raise ValueError(
            f"Operator indices in '{self.pairs_file}' span {op_min}..{op_max}, "
            f"but '{self.ops.root}' contains only {len(self.ops)} images. "
            "These pairs appear to use original HASY train indices; provide the full original HASY train "
            "folder or the bundled apply2x2 operator subset with the expected class layout."
        )

    def _build_map_from_original_train(self, op_root: Path, op_values: list[int]) -> dict[int, int] | None:
        candidates = [
            op_root / "train",
            Path(__file__).resolve().parents[4] / "NeurASP" / "examples" / "data" / "hasy" / "train",
        ]
        full_train = next((path for path in candidates if path.exists()), None)
        if full_train is None:
            return None

        full_files = sorted(path for path in full_train.rglob("*") if path.is_file())
        local_by_stem = {Path(path).stem: idx for idx, (path, _) in enumerate(self.ops.samples)}
        index_map: dict[int, int] = {}
        for op_index in sorted(set(op_values)):
            if op_index >= len(full_files):
                return None
            local_index = local_by_stem.get(full_files[op_index].stem)
            if local_index is None:
                return None
            index_map[op_index] = local_index
        LOGGER.info("Mapped apply2x2 operator indices from original HASY folder '%s'.", full_train)
        return index_map

    def _build_builtin_apply2x2_map(self, op_values: list[int]) -> dict[int, int] | None:
        local_ranges: dict[str, tuple[int, int]] = {}
        for idx, (path, _) in enumerate(self.ops.samples):
            class_name = Path(path).parent.name
            start, _ = local_ranges.get(class_name, (idx, idx))
            local_ranges[class_name] = (start, idx)

        # Original NeurASP HASY train indices: plus=2709..2798, minus=2591..2708, times=2799..4307.
        expected_ranges = {
            "0plus": (2709, 2798),
            "1minus": (2591, 2708),
            "2times": (2799, 4307),
        }
        if not all(class_name in local_ranges for class_name in expected_ranges):
            return None

        index_map: dict[int, int] = {}
        for op_index in sorted(set(op_values)):
            for class_name, (global_start, global_end) in expected_ranges.items():
                if global_start <= op_index <= global_end:
                    local_start, local_end = local_ranges[class_name]
                    local_index = local_start + op_index - global_start
                    if local_index > local_end:
                        return None
                    index_map[op_index] = local_index
                    break
            else:
                return None
        return index_map

    def _local_op_index(self, op_index: int) -> int:
        if self.op_index_map is None:
            return op_index
        return self.op_index_map[op_index]

    def _filter_available_examples(self) -> list[Tuple[int, int, int, int, int, int, int, int, int, int, int]]:
        filtered = [
            example
            for example in self.examples
            if all(0 <= self._local_op_index(op) < len(self.ops) for op in example[3:7])
        ]
        if len(filtered) != len(self.examples):
            LOGGER.warning(
                "Dropped %d apply2x2 examples with unavailable operator images.",
                len(self.examples) - len(filtered),
            )
        return filtered

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        d1, d2, d3, o1, o2, o3, o4, r1, r2, c1, c2 = self.examples[idx]
        imgs = [
            self.ops[self._local_op_index(o1)][0],
            self.ops[self._local_op_index(o2)][0],
            self.ops[self._local_op_index(o3)][0],
            self.ops[self._local_op_index(o4)][0],
        ]
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
