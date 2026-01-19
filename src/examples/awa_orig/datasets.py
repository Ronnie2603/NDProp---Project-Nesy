from __future__ import annotations

from pathlib import Path
from typing import Sequence, List

import numpy as np
import torch
from torch.utils.data import Dataset

AWA_BASE = Path("/home/moriyama/ASP/AwA/Animals_with_Attributes2")
AWA_DEFAULT_ROOT = AWA_BASE / "Features" / "ResNet101"
AWA_CLASSES_PATH = AWA_BASE / "classes.txt"
AWA_PREDICATE_MATRIX_PATH = AWA_BASE / "predicate-matrix-binary.txt"


def _read_awa_classes(path: Path) -> List[str]:
    class_names: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            class_names.append(parts[-1].replace("+", "_"))
    return class_names


def _load_attr_matrix(matrix_path: Path, *, class_count: int) -> torch.Tensor:
    mat = np.loadtxt(matrix_path, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"Predicate matrix at {matrix_path} must be 2D, got shape {mat.shape}.")
    if mat.shape[0] == class_count:
        mat = mat.T
    elif mat.shape[1] != class_count:
        raise ValueError(f"Predicate matrix shape {mat.shape} incompatible with {class_count} classes.")
    return torch.from_numpy(mat.T).float()


class AwADataset(Dataset):
    """
    AwA features-only dataset using precomputed ResNet101 embeddings.
    """

    def __init__(
        self,
        root: str | Path = AWA_DEFAULT_ROOT,
        *,
        features_file: str | Path | None = None,
        labels_file: str | Path | None = None,
        classes_file: str | Path | None = None,
        predicate_matrix_file: str | Path | None = None,
    ) -> None:
        root = Path(root)
        feat_path = Path(features_file) if features_file is not None else root / "AwA2-features.txt"
        label_path = Path(labels_file) if labels_file is not None else root / "AwA2-labels.txt"
        classes_path = Path(classes_file) if classes_file is not None else AWA_CLASSES_PATH
        predicate_matrix_path = (
            Path(predicate_matrix_file) if predicate_matrix_file is not None else AWA_PREDICATE_MATRIX_PATH
        )
        if not feat_path.exists():
            raise FileNotFoundError(f"Features file '{feat_path}' not found.")
        if not label_path.exists():
            raise FileNotFoundError(f"Labels file '{label_path}' not found.")
        if not classes_path.exists():
            raise FileNotFoundError(f"Classes file '{classes_path}' not found.")
        if not predicate_matrix_path.exists():
            raise FileNotFoundError(f"Predicate matrix file '{predicate_matrix_path}' not found.")

        feats = np.loadtxt(feat_path, dtype=np.float32)
        labels = np.loadtxt(label_path, dtype=np.int64)
        if feats.shape[0] != labels.shape[0]:
            raise ValueError("Features and labels must have the same number of rows.")
        self.features = torch.from_numpy(feats)
        self.labels = torch.from_numpy(labels)
        class_names = _read_awa_classes(classes_path)
        self.attr_by_class = _load_attr_matrix(predicate_matrix_path, class_count=len(class_names))

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int) -> dict:
        label = self.labels[idx]
        if label.min() >= 1:
            label = label - 1
        attrs = self.attr_by_class[label.long()]
        return {"features": self.features[idx], "label": label, "attrs": attrs}


def awa_collate(batch: Sequence[dict]) -> dict:
    feats = torch.stack([item["features"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    out = {"features": feats, "label": labels}
    if "attrs" in batch[0]:
        out["attrs"] = torch.stack([item["attrs"] for item in batch], dim=0)
    return out
