from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from examples.common import NeSyTask, gnn_from_args, move_batch
from nesy.reasoner import NeSyReasoner

from .datasets import HMCNNDataset, load_hmcnn_splits
from .model import HMCNNClassifier, HMCNNNeSyModel
from .program import build_hierarchy_program


class HMCNNTask(NeSyTask):
    name = "hmcnn"
    description = "HMCNN hierarchical classification with a NeSy reasoner."

    def __init__(self) -> None:
        self._cached: Dict[str, Tuple[HMCNNDataset, Optional[HMCNNDataset], HMCNNDataset]] = {}

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--hmcnn-dataset", type=str, required=True, help="HMCNN dataset name, e.g. cellcycle_FUN.")
        parser.add_argument(
            "--hmcnn-eval-split",
            type=str,
            default="val",
            choices=("val", "test"),
            help="Which split to use for evaluation.",
        )
        parser.add_argument(
            "--hmcnn-pretrain-epochs",
            type=int,
            default=0,
            help="Epochs to pretrain the HMCNN classifier (0 to skip).",
        )
        parser.add_argument(
            "--hmcnn-pretrain-lr",
            type=float,
            default=1e-3,
            help="Learning rate for HMCNN pretraining.",
        )
        parser.add_argument(
            "--hmcnn-pretrain-path",
            type=Path,
            default=None,
            help="Path to save/load pretrained HMCNN classifier weights.",
        )
        parser.add_argument("--hmcnn-hidden-dim", type=int, default=256, help="Hidden size for the HMCNN classifier.")
        parser.add_argument("--hmcnn-num-layers", type=int, default=2, help="Number of layers in the HMCNN classifier.")
        parser.add_argument("--hmcnn-dropout", type=float, default=0.2, help="Dropout for the HMCNN classifier.")
        parser.add_argument(
            "--hmcnn-nonlin",
            type=str,
            default="relu",
            choices=("relu", "tanh"),
            help="Activation function for the HMCNN classifier.",
        )

    def _load_splits(self, args: argparse.Namespace) -> Tuple[HMCNNDataset, Optional[HMCNNDataset], HMCNNDataset]:
        cache_key = f"{args.hmcnn_dataset}:{args.seed}"
        cached = self._cached.get(cache_key)
        if cached is not None:
            return cached
        splits = load_hmcnn_splits(args.hmcnn_dataset, seed=args.seed)
        self._cached[cache_key] = splits
        return splits

    def _pretrain_classifier(
        self,
        classifier: HMCNNClassifier,
        train_ds: HMCNNDataset,
        device: torch.device,
        *,
        lr: float,
        epochs: int,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        classifier.train()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        for _ in range(epochs):
            for batch in loader:
                feats = batch["features"].to(device)
                labels = batch["labels"].to(device)
                preds = classifier(feats)
                loss = loss_fn(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return classifier.state_dict()

    def build_model(self, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
        train_ds, val_ds, test_ds = self._load_splits(args)
        feature_dim = train_ds.features.size(1)
        class_dim = train_ds.labels.size(1)

        classifier = HMCNNClassifier(
            input_dim=feature_dim,
            output_dim=class_dim,
            hidden_dim=args.hmcnn_hidden_dim,
            num_layers=args.hmcnn_num_layers,
            dropout=args.hmcnn_dropout,
            non_lin=args.hmcnn_nonlin,
        ).to(device)

        pretrain_path = args.hmcnn_pretrain_path or (Path("runs_nesy") / "hmcnn" / f"{args.hmcnn_dataset}_pretrain.pt")
        freeze_classifier = False
        if args.hmcnn_pretrain_epochs > 0:
            state = self._pretrain_classifier(
                classifier,
                train_ds=train_ds,
                device=device,
                lr=args.hmcnn_pretrain_lr,
                epochs=args.hmcnn_pretrain_epochs,
                batch_size=args.batch_size,
            )
            pretrain_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, pretrain_path)
            freeze_classifier = True
        elif pretrain_path.exists():
            state = torch.load(pretrain_path, map_location=device, weights_only=True)
            classifier.load_state_dict(state)
            freeze_classifier = True
        if freeze_classifier:
            classifier.requires_grad_(False)

        program = build_hierarchy_program(train_ds.terms, train_ds.adj)
        gnn = gnn_from_args(args)
        reasoner = NeSyReasoner(program, gnn, init_hidden=args.use_init_hidden)
        model = HMCNNNeSyModel(classifier, reasoner, program.index_map)
        return model.to(device)

    def build_loaders(self, args: argparse.Namespace) -> Tuple[Dict[str, DataLoader], bool]:
        train_ds, val_ds, test_ds = self._load_splits(args)
        eval_ds = test_ds if args.hmcnn_eval_split == "test" or val_ds is None else val_ds
        loaders = {
            "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
            "eval": DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
        }
        return loaders, False

    def step(self, model: torch.nn.Module, batch: Dict, device: torch.device):
        batch = move_batch(batch, device)
        outputs = model(batch, return_tau_phi=False)
        class_probs = outputs["class_probs"]
        target = batch["labels"].float()
        loss = F.binary_cross_entropy(class_probs, target)
        loss = loss + (outputs["assignments"] * (1 - outputs["assignments"])).mean()
        preds = (class_probs >= 0.5).float()
        metrics = {
            "acc": (preds == target).float().mean(),
            "loss": loss.detach(),
        }
        return loss, metrics
