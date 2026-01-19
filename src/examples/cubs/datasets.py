from __future__ import annotations

from pathlib import Path
from typing import Sequence, List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

CUB_BASE = Path("/home/moriyama/ASP/Cubs/CUB_200_2011")
CUB_DEFAULT_ROOT = CUB_BASE
CUB_CLASSES_PATH = CUB_BASE / "classes.txt"
CUB_IMAGE_LABELS_PATH = CUB_BASE / "image_class_labels.txt"
CUB_IMAGES_PATH = CUB_BASE / "images.txt"
CUB_SPLIT_PATH = CUB_BASE / "train_test_split.txt"
CUB_CLASS_ATTR_PATH = CUB_BASE / "attributes" / "class_attribute_labels_continuous.txt"
CUB_IMAGE_ATTR_PATH = CUB_BASE / "attributes" / "image_attribute_labels.txt"


def _read_class_names(path: Path) -> List[str]:
    names: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            names.append(parts[-1])
    return names


def _num_images(images_path: Path) -> int:
    with images_path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _load_image_labels(path: Path, *, num_images: int) -> torch.Tensor:
    labels = np.zeros((num_images,), dtype=np.int64)
    data = np.loadtxt(path, dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    image_ids = data[:, 0] - 1
    class_ids = data[:, 1] - 1
    labels[image_ids] = class_ids
    return torch.from_numpy(labels)


def _load_split(path: Path, *, num_images: int) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    split = np.zeros((num_images,), dtype=np.int64)
    image_ids = data[:, 0] - 1
    split_flags = data[:, 1]
    split[image_ids] = split_flags
    return split


def _load_class_attr_matrix(path: Path) -> torch.Tensor:
    mat = np.loadtxt(path, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"Class attribute matrix at {path} must be 2D, got shape {mat.shape}.")
    mat = mat / 100.0
    return torch.from_numpy(mat).float()


def _load_image_attr_matrix(path: Path, *, num_images: int, num_attrs: int) -> torch.Tensor:
    data = np.loadtxt(path, dtype=np.float32, usecols=(0, 1, 2))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    image_ids = data[:, 0].astype(np.int64) - 1
    attr_ids = data[:, 1].astype(np.int64) - 1
    values = data[:, 2].astype(np.float32)

    sums = np.zeros((num_images, num_attrs), dtype=np.float32)
    counts = np.zeros((num_images, num_attrs), dtype=np.float32)
    np.add.at(sums, (image_ids, attr_ids), values)
    np.add.at(counts, (image_ids, attr_ids), 1.0)
    counts[counts == 0] = 1.0
    avg = sums / counts
    return torch.from_numpy((avg >= 0.5).astype(np.float32))


def _load_image_paths(images_path: Path, *, root: Path) -> List[Path]:
    paths: List[Path] = []
    with images_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            paths.append(root / "images" / parts[1])
    return paths


class CUBSDataset(Dataset):
    """
    CUB-200-2011 dataset using per-image attribute labels as features.
    """

    def __init__(
        self,
        root: str | Path = CUB_DEFAULT_ROOT,
        *,
        classes_file: str | Path | None = None,
        image_labels_file: str | Path | None = None,
        images_file: str | Path | None = None,
        split_file: str | Path | None = None,
        class_attr_file: str | Path | None = None,
        image_attr_file: str | Path | None = None,
        split: str | None = None,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        root = Path(root)
        classes_path = Path(classes_file) if classes_file is not None else root / "classes.txt"
        image_labels_path = Path(image_labels_file) if image_labels_file is not None else root / "image_class_labels.txt"
        images_path = Path(images_file) if images_file is not None else root / "images.txt"
        split_path = Path(split_file) if split_file is not None else root / "train_test_split.txt"
        class_attr_path = (
            Path(class_attr_file) if class_attr_file is not None else root / "attributes" / "class_attribute_labels_continuous.txt"
        )
        image_attr_path = (
            Path(image_attr_file) if image_attr_file is not None else root / "attributes" / "image_attribute_labels.txt"
        )

        for path in (classes_path, image_labels_path, images_path, split_path, class_attr_path, image_attr_path):
            if not path.exists():
                raise FileNotFoundError(f"Required CUBS file '{path}' not found.")

        _ = _read_class_names(classes_path)
        num_images = _num_images(images_path)
        labels = _load_image_labels(image_labels_path, num_images=num_images)
        class_attr = _load_class_attr_matrix(class_attr_path)
        image_attr = _load_image_attr_matrix(image_attr_path, num_images=num_images, num_attrs=class_attr.size(1))
        split_flags = _load_split(split_path, num_images=num_images)
        image_paths = _load_image_paths(images_path, root=root)
        if len(image_paths) != num_images:
            raise ValueError("images.txt does not match expected image count.")

        self.transform = transform or transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        if split is not None:
            split_key = split.lower()
            if split_key not in {"train", "test"}:
                raise ValueError("split must be 'train', 'test', or None.")
            mask = split_flags == (1 if split_key == "train" else 0)
            indices = np.nonzero(mask)[0]
            self.image_paths = [image_paths[idx] for idx in indices.tolist()]
            self.labels = labels[indices]
            self.image_attrs = image_attr[indices]
        else:
            self.image_paths = image_paths
            self.labels = labels
            self.image_attrs = image_attr

        self.attr_by_class = class_attr

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        label = self.labels[idx]
        img = Image.open(self.image_paths[idx]).convert("RGB")
        feats = self.transform(img)
        attrs = self.image_attrs[idx]
        class_attrs = self.attr_by_class[label.long()]
        return {"features": feats, "label": label, "attrs": attrs, "class_attrs": class_attrs}


def cubs_collate(batch: Sequence[dict]) -> dict:
    feats = torch.stack([item["features"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    out = {"features": feats, "label": labels}
    if "attrs" in batch[0]:
        out["attrs"] = torch.stack([item["attrs"] for item in batch], dim=0)
    if "class_attrs" in batch[0]:
        out["class_attrs"] = torch.stack([item["class_attrs"] for item in batch], dim=0)
    return out
