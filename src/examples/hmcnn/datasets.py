from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

CHMCNN_ROOT = Path(__file__).resolve().parents[4] / "C-HMCNN"
if str(CHMCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(CHMCNN_ROOT))

os.environ.setdefault("DATA_FOLDER", str(CHMCNN_ROOT))

from utils import datasets as hmcnn_datasets  # type: ignore  # noqa: E402
from utils.parser import initialize_dataset, initialize_other_dataset  # type: ignore  # noqa: E402


class HMCNNDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, terms: list[str], adj: np.ndarray) -> None:
        self.features = features.float()
        self.labels = labels.float()
        self.terms = terms
        self.adj = adj

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int) -> dict:
        return {"features": self.features[idx], "labels": self.labels[idx]}


def load_hmcnn_splits(
    dataset_name: str,
    *,
    seed: int = 0,
) -> Tuple[HMCNNDataset, Optional[HMCNNDataset], HMCNNDataset]:
    if dataset_name not in hmcnn_datasets:
        raise ValueError(f"Unknown HMCNN dataset '{dataset_name}'.")

    is_others = "others" in dataset_name
    if is_others:
        train, test = initialize_other_dataset(dataset_name, hmcnn_datasets)
        train.X, valX, train.Y, valY = train_test_split(train.X, train.Y, test_size=0.30, random_state=seed)
        val = None
    else:
        train, val, test = initialize_dataset(dataset_name, hmcnn_datasets)

    if is_others:
        scaler = preprocessing.StandardScaler().fit(train.X)
        imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean").fit(train.X)
        train_X = imp_mean.transform(train.X)
        val_X = imp_mean.transform(valX)
        test_X = imp_mean.transform(test.X)
        train_X = scaler.transform(train_X)
        val_X = scaler.transform(val_X)
        test_X = scaler.transform(test_X)
    else:
        scaler = preprocessing.StandardScaler().fit(np.concatenate((train.X, val.X)))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean").fit(np.concatenate((train.X, val.X)))
        train_X = scaler.transform(imp_mean.transform(train.X))
        val_X = scaler.transform(imp_mean.transform(val.X))
        test_X = scaler.transform(imp_mean.transform(test.X))

    train_ds = HMCNNDataset(torch.tensor(train_X), torch.tensor(train.Y), train.terms, train.A)
    val_ds = None
    if is_others:
        val_ds = HMCNNDataset(torch.tensor(val_X), torch.tensor(valY), train.terms, train.A)
    else:
        val_ds = HMCNNDataset(torch.tensor(val_X), torch.tensor(val.Y), train.terms, train.A)
    test_ds = HMCNNDataset(torch.tensor(test_X), torch.tensor(test.Y), train.terms, train.A)
    return train_ds, val_ds, test_ds
