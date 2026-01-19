from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from examples.common import NeSyTask, gnn_from_args, move_batch
from nesy.reasoner import NeSyReasoner

from .datasets import AwADataset, awa_collate
from .model import AwANeSyModel
from .program import DEFAULT_DATA_DIR, build_program


class AwATask(NeSyTask):
    name = "awa"
    description = "Animals with Attributes using precomputed features (no CNN)."

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--awa-root",
            type=Path,
            default=Path("/home/moriyama/ASP/AwA/Animals_with_Attributes2/Features/ResNet101"),
            help="Root directory for AwA features.",
        )
        parser.add_argument("--awa-features", type=Path, default=None, help="Optional path to AwA2-features.txt.")
        parser.add_argument("--awa-labels", type=Path, default=None, help="Optional path to AwA2-labels.txt.")
        parser.add_argument(
            "--awa-classes",
            type=Path,
            default=DEFAULT_DATA_DIR / "classes.txt",
            help="Path to classes.txt.",
        )
        parser.add_argument("--program-path", type=Path, default=None, help="Optional unused program path placeholder.")
        parser.add_argument("--clingo-path", type=str, default=None, help="Optional clingo executable.")
        parser.add_argument("--awa-train-fraction", type=float, default=0.8, help="Fraction of AwA data used for training (rest for eval).")
        parser.add_argument("--awa-split-seed", type=int, default=0, help="Seed for AwA train/eval split.")
        parser.add_argument("--awa-attr-pretrain-epochs", type=int, default=0, help="Epochs to pretrain the AwA attribute mapper (0 to skip).")
        parser.add_argument("--awa-attr-pretrain-lr", type=float, default=1e-3, help="Learning rate for AwA attribute mapper pretraining.")
        parser.add_argument("--awa-attr-pretrain-path", type=Path, default=None, help="Path to save/load pretrained AwA attribute mapper weights.")
        parser.add_argument("--awa-run-baselines", action="store_true", help="Run AwA baselines on attribute mapper outputs.")
        parser.add_argument("--awa-baseline-hidden", type=int, default=64, help="Hidden size for AwA attribute classifier baseline.")
        parser.add_argument("--awa-baseline-epochs", type=int, default=20, help="Training epochs for AwA attribute classifier baseline.")
        parser.add_argument(
            "--awa-baseline-mode",
            choices=["both", "nearest", "classifier"],
            default="both",
            help="Which AwA baseline(s) to run: nearest prototype, classifier, or both.",
        )

    def _load_spec(self, args: argparse.Namespace):
        return build_program(classes_file=args.awa_classes, program_path=args.program_path)

    def _split_awa_dataset(self, args: argparse.Namespace):
        full_ds = AwADataset(root=args.awa_root, features_file=args.awa_features, labels_file=args.awa_labels)
        total = len(full_ds)
        if total < 2:
            raise ValueError("AwA dataset must contain at least 2 samples to split into train/eval.")
        train_len = int(total * args.awa_train_fraction)
        train_len = min(max(train_len, 1), total - 1)
        eval_len = total - train_len
        generator = torch.Generator().manual_seed(args.awa_split_seed)
        return random_split(full_ds, [train_len, eval_len], generator=generator)

    def _pretrain_attr_mapper(
        self,
        *,
        feature_dim: int,
        attr_dim: int,
        train_ds,
        device: torch.device,
        lr: float,
        epochs: int,
        batch_size: int,
    ) -> Dict:
        mapper = torch.nn.Linear(feature_dim, attr_dim).to(device)
        opt = torch.optim.Adam(mapper.parameters(), lr=lr)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=awa_collate)
        loss_fn = torch.nn.BCELoss()
        mapper.train()
        for _ in range(max(epochs, 0)):
            for batch in loader:
                feats = batch["features"].to(device)
                attrs = batch["attrs"].float().to(device)
                opt.zero_grad()
                preds = torch.sigmoid(mapper(feats))
                loss = loss_fn(preds, attrs)
                loss.backward()
                opt.step()
        return mapper.state_dict()

    def _extract_attr_matrix(self, dataset) -> torch.Tensor:
        base = dataset
        while hasattr(base, "dataset"):
            base = base.dataset
        if not hasattr(base, "attr_by_class"):
            raise ValueError("AwA dataset missing attribute matrix.")
        return base.attr_by_class

    def _collect_attr_outputs(self, model: torch.nn.Module, loader: DataLoader, device: torch.device):
        model.eval()
        preds: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                batch = move_batch(batch, device)
                feats = batch["features"]
                pred = torch.sigmoid(model.attr_mapper(feats))
                preds.append(pred.detach().cpu())
                labels.append(batch["label"].long().detach().cpu())
        if not preds:
            return torch.empty(0, model.attr_mapper.out_features), torch.empty(0, dtype=torch.long)
        return torch.cat(preds, dim=0), torch.cat(labels, dim=0)

    def _train_attr_classifier(
        self,
        train_attr: torch.Tensor,
        train_labels: torch.Tensor,
        eval_attr: torch.Tensor,
        eval_labels: torch.Tensor,
        *,
        hidden: int,
        epochs: int,
        device: torch.device,
        num_classes: int,
    ) -> float:
        if train_attr.numel() == 0:
            return float("nan")
        attr_dim = train_attr.size(1)
        clf = nn.Sequential(nn.Linear(attr_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_classes)).to(device)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        train_ds = TensorDataset(train_attr, train_labels)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        best_acc = 0.0
        for epoch in range(max(epochs, 0)):
            clf.train()
            print(f"Attr classifier training epoch {epoch+1}/{epochs}")
            sum_loss = 0
            for feats, labels in train_loader:
                feats = feats.to(device)
                labels = labels.to(device)
                opt.zero_grad()
                logits = clf(feats)
                loss = loss_fn(logits, labels)
                loss.backward()
                opt.step()
                sum_loss += loss.item() * feats.size(0)
            avg_loss = sum_loss / len(train_ds)
            print(f"  Avg train loss: {avg_loss:.4f}")
            clf.eval()
            with torch.no_grad():
                logits = clf(eval_attr.to(device))
                preds = torch.argmax(logits, dim=-1).cpu()
                acc = float((preds == eval_labels).float().mean())
                if acc > best_acc:
                    best_acc = acc
                print(best_acc)
        return best_acc

    def run_baselines(self, model: torch.nn.Module, loaders: Dict[str, DataLoader], device: torch.device, args: argparse.Namespace) -> Dict[str, float]:
        attr_matrix = self._extract_attr_matrix(loaders["train"].dataset).to(device)
        num_classes = attr_matrix.size(0)
        eval_attr, eval_labels = self._collect_attr_outputs(model, loaders["eval"], device)

        results: Dict[str, float] = {}

        if args.awa_baseline_mode in ("both", "nearest"):
            nearest_acc = float("nan")
            if eval_attr.numel() > 0:
                proto = attr_matrix.unsqueeze(0)
                preds = []
                batch = 512
                for start in range(0, eval_attr.size(0), batch):
                    chunk = eval_attr[start : start + batch].to(device)
                    dist = torch.norm(chunk.unsqueeze(1) - proto, dim=2)
                    preds.append(torch.argmin(dist, dim=1).cpu())
                pred_labels = torch.cat(preds, dim=0)
                nearest_acc = float((pred_labels == eval_labels).float().mean())
            results["nearest_proto_acc"] = nearest_acc

        if args.awa_baseline_mode in ("both", "classifier"):
            train_attr, train_labels = self._collect_attr_outputs(model, loaders["train"], device)
            nn_acc = self._train_attr_classifier(
                train_attr,
                train_labels,
                eval_attr,
                eval_labels,
                hidden=args.awa_baseline_hidden,
                epochs=args.awa_baseline_epochs,
                device=device,
                num_classes=num_classes,
            )
            results["attr_classifier_acc"] = nn_acc

        return results

    def build_model(self, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
        spec = self._load_spec(args)
        sample = AwADataset(root=args.awa_root, features_file=args.awa_features, labels_file=args.awa_labels)[0]
        feature_dim = sample["features"].numel()
        attr_dim = spec.index_map["attr"].numel()
        gnn = gnn_from_args(args)
        gnn.weak_negation_falsity = True
        reasoner = NeSyReasoner(spec, gnn, init_hidden=args.use_init_hidden)
        model = AwANeSyModel(feature_dim, attr_dim, spec.index_map["class"].numel(), reasoner, spec.index_map)
        pretrain_path = args.awa_attr_pretrain_path if args.awa_attr_pretrain_path is not None else (Path(args.awa_root) / "attr_mapper.pt")
        freeze_attr = False
        if args.awa_attr_pretrain_epochs > 0:
            train_ds, _ = self._split_awa_dataset(args)
            state = self._pretrain_attr_mapper(
                feature_dim=feature_dim,
                attr_dim=attr_dim,
                train_ds=train_ds,
                device=device,
                lr=args.awa_attr_pretrain_lr,
                epochs=args.awa_attr_pretrain_epochs,
                batch_size=args.batch_size,
            )
            pretrain_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, pretrain_path)
            freeze_attr = True
            model.attr_mapper.load_state_dict(state)
        elif pretrain_path is not None and pretrain_path.exists():
            state = torch.load(pretrain_path, map_location=device, weights_only=True)
            model.attr_mapper.load_state_dict(state)
            freeze_attr = True
            print(f"Loaded pretrained AwA attribute mapper from {pretrain_path}.")
        if freeze_attr:
            model.attr_mapper.requires_grad_(False)
        return model.to(device)

    def build_loaders(self, args: argparse.Namespace) -> tuple[Dict[str, DataLoader], bool]:
        train_ds, eval_ds = self._split_awa_dataset(args)
        loaders = {
            "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=awa_collate),
            "eval": DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=awa_collate),
        }
        return loaders, False

    def step(self, model: torch.nn.Module, batch: Dict, device: torch.device):
        batch = move_batch(batch, device)
        outputs = model(batch, return_tau_phi=False)
        class_probs = outputs["class_probs"]
        target = batch["label"].long()
        class_target = F.one_hot(target, num_classes=class_probs.size(1)).float()
        weight = torch.ones_like(class_target)
        weight[class_target == 1] = (class_probs.size(1) - 1.0)
        class_loss = F.binary_cross_entropy(class_probs, class_target, weight=weight) 

        loss = class_loss + (outputs["assignments"] * (1-outputs["assignments"])).mean()

        preds = torch.argmax(outputs["class_probs"], dim=-1)
        metrics = {
            "acc": (preds == batch["label"]).float().mean(),
            "loss": loss.detach(),
        }
        return loss, metrics
