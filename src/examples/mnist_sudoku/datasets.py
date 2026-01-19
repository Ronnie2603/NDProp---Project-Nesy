from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, Callable

import torch
from torch.utils.data import Dataset

SUDOKU_BASE = Path("/home/moriyama/ASP/Sudoku/sudoku")


def _load_tensor(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Sudoku data file '{path}' not found.")
    return torch.load(path, weights_only=True)


def _resolve_paths(root: Path, *, train: bool) -> tuple[Path, Path]:
    if train:
        features_img = root / "train_features_img.pt"
        labels = root / "train_labels.pt"
    else:
        features_img = root / "test_features_img.pt"
        labels = root / "test_labels.pt"
    if features_img.exists() and labels.exists():
        return features_img, labels
    return root / "features_img.pt", root / "labels.pt"


def _prepare_images(ximg: torch.Tensor) -> torch.Tensor:
    if ximg.dim() == 6:
        # (N, 9, 9, C, H, W) -> (N, 81, C, H, W)
        n, r, c, ch, h, w = ximg.shape
        return ximg.reshape(n, r * c, ch, h, w).float()
    if ximg.dim() == 5:
        # (N, 9, 9, H, W) -> (N, 81, 1, H, W)
        n, r, c, h, w = ximg.shape
        ximg = ximg.reshape(n, r * c, h, w)
        return ximg.unsqueeze(2).float()
    if ximg.dim() == 4:
        # (N, 81, H, W) -> (N, 81, 1, H, W)
        return ximg.unsqueeze(2).float()
    raise ValueError(f"Unexpected image tensor shape: {tuple(ximg.shape)}")


def _prepare_labels(y: torch.Tensor) -> torch.Tensor:
    if y.dim() >= 3 and y.size(-1) in (4, 9) and y.size(-1) == y.size(-2):
        # One-hot per cell: (N, 9, 9, 9) or (N, 81, 9)
        labels = y.argmax(dim=-1).view(y.size(0), -1).long()
        if labels.numel() > 0 and labels.min() >= 1:
            labels = labels - 1
        return labels
    if y.dim() == 3 and y.size(1) == 9 and y.size(2) == 9:
        labels = y.view(y.size(0), -1).long()
        if labels.numel() > 0 and labels.min() >= 1:
            labels = labels - 1
        return labels
    if y.dim() == 2 and y.size(1) == 81:
        labels = y.long()
        if labels.numel() > 0 and labels.min() >= 1:
            labels = labels - 1
        return labels
    raise ValueError(f"Unexpected label tensor shape: {tuple(y.shape)}")


class MNISTSudokuDataset(Dataset):
    """
    MNIST Sudoku dataset using pre-generated Sudoku boards and cell images.
    """

    def __init__(
        self,
        root: str | Path = SUDOKU_BASE,
        *,
        train: bool = True,
        train_fraction: float = 0.9,
        split_seed: int = 0,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        root = Path(root)
        ximg_path, labels_path = _resolve_paths(root, train=train)
        ximg = _load_tensor(ximg_path)
        labels = _load_tensor(labels_path)

        self.images = _prepare_images(ximg)
        self.labels = _prepare_labels(labels)
        if transform is not None:
            n, cells, ch, h, w = self.images.shape
            flat = self.images.view(n * cells, ch, h, w)
            flat = transform(flat)
            self.images = flat.view(n, cells, ch, h, w)

        if self.images.size(0) != self.labels.size(0):
            raise ValueError("Mismatch between image and label counts.")

        if ximg_path.name == "features_img.pt":
            total = self.images.size(0)
            if not 0.0 < train_fraction < 1.0:
                raise ValueError("train_fraction must be in (0, 1).")
            gen = torch.Generator().manual_seed(split_seed)
            perm = torch.randperm(total, generator=gen)
            split = int(total * train_fraction)
            if split <= 0 or split >= total:
                raise ValueError("train_fraction produces empty split.")

            if train:
                idx = perm[:split]
            else:
                idx = perm[split:]
            self.images = self.images[idx]
            self.labels = self.labels[idx]

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, idx: int) -> dict:
        return {"images": self.images[idx], "labels": self.labels[idx]}


def mnist_sudoku_collate(batch: Sequence[dict]) -> dict:
    images = torch.stack([item["images"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    return {"images": images, "labels": labels}
