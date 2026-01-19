from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from examples.common import NeSyTask, gnn_from_args, move_batch
from nesy.reasoner import NeSyReasoner

from .datasets import CUBSDataset, cubs_collate
from .model import CUBSNeSyModel
from .program import DEFAULT_DATA_DIR, build_program


class CUBSTask(NeSyTask):
    name = "cubs"
    description = "CUB-200-2011 using per-image attributes as features (no CNN)."

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--cubs-root",
            type=Path,
            default=Path("/home/moriyama/ASP/Cubs/CUB_200_2011"),
            help="Root directory for the CUB-200-2011 dataset.",
        )
        parser.add_argument("--cubs-classes", type=Path, default=DEFAULT_DATA_DIR / "classes.txt", help="Path to classes.txt.")
        parser.add_argument(
            "--cubs-image-labels",
            type=Path,
            default=DEFAULT_DATA_DIR / "image_class_labels.txt",
            help="Path to image_class_labels.txt.",
        )
        parser.add_argument(
            "--cubs-images",
            type=Path,
            default=DEFAULT_DATA_DIR / "images.txt",
            help="Path to images.txt.",
        )
        parser.add_argument(
            "--cubs-split",
            type=Path,
            default=DEFAULT_DATA_DIR / "train_test_split.txt",
            help="Path to train_test_split.txt.",
        )
        parser.add_argument(
            "--cubs-class-attrs",
            type=Path,
            default=DEFAULT_DATA_DIR / "attributes" / "class_attribute_labels_continuous.txt",
            help="Path to class_attribute_labels_continuous.txt.",
        )
        parser.add_argument(
            "--cubs-image-attrs",
            type=Path,
            default=DEFAULT_DATA_DIR / "attributes" / "image_attribute_labels.txt",
            help="Path to image_attribute_labels.txt.",
        )
        parser.add_argument("--program-path", type=Path, default=None, help="Optional program path.")
        parser.add_argument("--clingo-path", type=str, default=None, help="Optional clingo executable.")
        parser.add_argument("--cubs-use-split", action="store_true", help="Use the official train/test split.")
        parser.add_argument("--cubs-train-fraction", type=float, default=0.8, help="Fraction of CUB data used for training (rest for eval).")
        parser.add_argument("--cubs-split-seed", type=int, default=0, help="Seed for CUB train/eval split.")
        parser.add_argument("--cubs-attr-pretrain-epochs", type=int, default=0, help="Epochs to pretrain the CUB attribute mapper (0 to skip).")
        parser.add_argument("--cubs-attr-pretrain-lr", type=float, default=1e-3, help="Learning rate for CUB attribute mapper pretraining.")
        parser.add_argument("--cubs-attr-pretrain-path", type=Path, default=None, help="Path to save/load pretrained CUB attribute mapper weights.")
        parser.add_argument(
            "--cubs-attr-pretrain-split",
            type=float,
            default=0.9,
            help="Fraction of data used for attr mapper pretraining (rest for eval).",
        )
        parser.add_argument("--cubs-run-baselines", action="store_true", help="Run CUB baselines on attribute mapper outputs.")
        parser.add_argument("--cubs-baseline-hidden", type=int, default=64, help="Hidden size for CUB attribute classifier baseline.")
        parser.add_argument("--cubs-baseline-epochs", type=int, default=50, help="Training epochs for CUB attribute classifier baseline.")
        parser.add_argument(
            "--cubs-baseline-mode",
            choices=["both", "nearest", "classifier"],
            default="both",
            help="Which CUB baseline(s) to run: nearest prototype, classifier, or both.",
        )

    def _load_spec(self, args: argparse.Namespace):
        return build_program(classes_file=args.cubs_classes, program_path=args.program_path)

    def _split_cubs_dataset(self, args: argparse.Namespace, *, dataset: CUBSDataset | None = None):
        full_ds = dataset if dataset is not None else CUBSDataset(
            root=args.cubs_root,
            classes_file=args.cubs_classes,
            image_labels_file=args.cubs_image_labels,
            images_file=args.cubs_images,
            split_file=args.cubs_split,
            class_attr_file=args.cubs_class_attrs,
            image_attr_file=args.cubs_image_attrs,
            split=None,
        )
        total = len(full_ds)
        if total < 2:
            raise ValueError("CUB dataset must contain at least 2 samples to split into train/eval.")
        train_len = int(total * args.cubs_train_fraction)
        train_len = min(max(train_len, 1), total - 1)
        eval_len = total - train_len
        generator = torch.Generator().manual_seed(args.cubs_split_seed)
        return random_split(full_ds, [train_len, eval_len], generator=generator)

    def _pretrain_attr_mapper(
        self,
        *,
        model: torch.nn.Module,
        train_ds,
        eval_ds,
        device: torch.device,
        lr: float,
        epochs: int,
        batch_size: int,
    ) -> Dict:
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=cubs_collate)
        eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=cubs_collate)
        loss_fn = torch.nn.BCELoss()
        best_state = model.attr_mapper.state_dict()
        best_loss = float("inf")
        for epoch in range(max(epochs, 0)):
            model.train()
            running = 0.0
            count = 0
            correct = 0.0
            total = 0.0
            train_scores: list[torch.Tensor] = []
            train_targets: list[torch.Tensor] = []
            train_tp = 0.0
            train_fp = 0.0
            train_fn = 0.0
            for batch in tqdm(loader, desc=f"Attr pretrain {epoch+1}/{epochs}", leave=False):
                feats = batch["features"].to(device)
                attrs = batch["attrs"].float().to(device)
                opt.zero_grad()
                preds = torch.sigmoid(model.attr_mapper(model.backbone(feats)))
                loss = loss_fn(preds, attrs)
                loss.backward()
                opt.step()
                running += loss.item() * feats.size(0)
                count += feats.size(0)
                preds_bin = (preds >= 0.5).float()
                correct += (preds_bin == attrs).float().sum().item()
                total += attrs.numel()
                train_scores.append(preds.detach().cpu())
                train_targets.append(attrs.detach().cpu())
                train_tp += (preds_bin * attrs).sum().item()
                train_fp += (preds_bin * (1 - attrs)).sum().item()
                train_fn += ((1 - preds_bin) * attrs).sum().item()
            train_loss = running / max(count, 1)
            train_acc = correct / max(total, 1.0)
            model.eval()
            eval_running = 0.0
            eval_count = 0
            eval_correct = 0.0
            eval_total = 0.0
            eval_scores: list[torch.Tensor] = []
            eval_targets: list[torch.Tensor] = []
            eval_tp = 0.0
            eval_fp = 0.0
            eval_fn = 0.0
            with torch.no_grad():
                for batch in tqdm(eval_loader, desc=f"Attr eval {epoch+1}/{epochs}", leave=False):
                    feats = batch["features"].to(device)
                    attrs = batch["attrs"].float().to(device)
                    preds = torch.sigmoid(model.attr_mapper(model.backbone(feats)))
                    loss = loss_fn(preds, attrs)
                    eval_running += loss.item() * feats.size(0)
                    eval_count += feats.size(0)
                    preds_bin = (preds >= 0.5).float()
                    eval_correct += (preds_bin == attrs).float().sum().item()
                    eval_total += attrs.numel()
                    eval_scores.append(preds.detach().cpu())
                    eval_targets.append(attrs.detach().cpu())
                    eval_tp += (preds_bin * attrs).sum().item()
                    eval_fp += (preds_bin * (1 - attrs)).sum().item()
                    eval_fn += ((1 - preds_bin) * attrs).sum().item()
            eval_loss = eval_running / max(eval_count, 1)
            eval_acc = eval_correct / max(eval_total, 1.0)
            train_prec = train_tp / (train_tp + train_fp + 1e-8)
            train_rec = train_tp / (train_tp + train_fn + 1e-8)
            eval_prec = eval_tp / (eval_tp + eval_fp + 1e-8)
            eval_rec = eval_tp / (eval_tp + eval_fn + 1e-8)
            train_map = float("nan")
            eval_map = float("nan")
            if train_scores:
                train_map = self._compute_attr_map(torch.cat(train_scores, dim=0), torch.cat(train_targets, dim=0))
            if eval_scores:
                eval_map = self._compute_attr_map(torch.cat(eval_scores, dim=0), torch.cat(eval_targets, dim=0))
            print(
                f"Attr pretrain epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} train_prec={train_prec:.4f} train_rec={train_rec:.4f} "
                f"train_mAP={train_map:.4f} eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f} "
                f"eval_prec={eval_prec:.4f} eval_rec={eval_rec:.4f} eval_mAP={eval_map:.4f}"
            )
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.attr_mapper.state_dict().items()}
        return best_state

    def _extract_attr_matrix(self, dataset) -> torch.Tensor:
        base = dataset
        while hasattr(base, "dataset"):
            base = base.dataset
        if not hasattr(base, "attr_by_class"):
            raise ValueError("CUB dataset missing attribute matrix.")
        return base.attr_by_class

    def _collect_attr_outputs(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        *,
        include_targets: bool = False,
    ):
        model.eval()
        preds: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                batch = move_batch(batch, device)
                feats = batch["features"]
                pred = torch.sigmoid(model.attr_mapper(model.backbone(feats)))
                preds.append(pred.detach().cpu())
                labels.append(batch["label"].long().detach().cpu())
                if include_targets and "attrs" in batch:
                    targets.append(batch["attrs"].float().detach().cpu())
        if not preds:
            empty_attr = torch.empty(0, model.attr_dim)
            empty_labels = torch.empty(0, dtype=torch.long)
            if include_targets:
                return empty_attr, empty_labels, torch.empty(0, model.attr_dim)
            return empty_attr, empty_labels
        if include_targets:
            target_tensor = torch.cat(targets, dim=0) if targets else torch.empty(0, model.attr_dim)
            return torch.cat(preds, dim=0), torch.cat(labels, dim=0), target_tensor
        return torch.cat(preds, dim=0), torch.cat(labels, dim=0)

    def _train_attr_classifier(
        self,
        train_attr: torch.Tensor,
        train_labels: torch.Tensor,
        eval_attr: torch.Tensor,
        eval_labels: torch.Tensor,
        train_attr_targets: torch.Tensor,
        eval_attr_targets: torch.Tensor,
        *,
        hidden: int,
        epochs: int,
        device: torch.device,
        num_classes: int,
    ) -> float:
        if train_attr.numel() == 0:
            return float("nan")
        def _feat_stats(name: str, feats: torch.Tensor) -> None:
            mean = float(feats.mean())
            std = float(feats.std())
            frac_pos = float((feats >= 0.5).float().mean())
            print(f"{name} feats: mean={mean:.4f} std={std:.4f} >=0.5={frac_pos:.4f}")

        _feat_stats("train", train_attr)
        _feat_stats("eval", eval_attr)
        if train_attr_targets.numel() > 0:
            train_map = self._compute_attr_map(train_attr, train_attr_targets)
            train_prec = self._compute_attr_precision(train_attr, train_attr_targets)
            train_rec = self._compute_attr_recall(train_attr, train_attr_targets)
            print(f"train attr metrics: mAP={train_map:.4f} precision={train_prec:.4f} recall={train_rec:.4f}")
        if eval_attr_targets.numel() > 0:
            eval_map = self._compute_attr_map(eval_attr, eval_attr_targets)
            eval_prec = self._compute_attr_precision(eval_attr, eval_attr_targets)
            eval_rec = self._compute_attr_recall(eval_attr, eval_attr_targets)
            print(f"eval attr metrics: mAP={eval_map:.4f} precision={eval_prec:.4f} recall={eval_rec:.4f}")
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
        eval_attr, eval_labels, eval_targets = self._collect_attr_outputs(
            model, loaders["eval"], device, include_targets=True
        )

        results: Dict[str, float] = {}

        if args.cubs_baseline_mode in ("both", "nearest"):
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

        if args.cubs_baseline_mode in ("both", "classifier"):
            train_attr, train_labels, train_targets = self._collect_attr_outputs(
                model, loaders["train"], device, include_targets=True
            )
            nn_acc = self._train_attr_classifier(
                train_attr,
                train_labels,
                eval_attr,
                eval_labels,
                train_targets,
                eval_targets,
                hidden=args.cubs_baseline_hidden,
                epochs=args.cubs_baseline_epochs,
                device=device,
                num_classes=num_classes,
            )
            results["attr_classifier_acc"] = nn_acc
            if train_targets.numel() > 0:
                results["train_attr_map"] = self._compute_attr_map(train_attr, train_targets)
                results["train_attr_precision"] = self._compute_attr_precision(train_attr, train_targets)
                results["train_attr_recall"] = self._compute_attr_recall(train_attr, train_targets)
            if eval_targets.numel() > 0:
                results["eval_attr_map"] = self._compute_attr_map(eval_attr, eval_targets)
                results["eval_attr_precision"] = self._compute_attr_precision(eval_attr, eval_targets)
                results["eval_attr_recall"] = self._compute_attr_recall(eval_attr, eval_targets)

        return results

    def _compute_attr_map(self, scores: torch.Tensor, targets: torch.Tensor) -> float:
        scores = scores.detach().cpu()
        targets = targets.detach().cpu()
        if scores.numel() == 0:
            return float("nan")
        num_attrs = scores.size(1)
        aps = []
        for idx in range(num_attrs):
            s = scores[:, idx]
            t = targets[:, idx]
            if t.sum().item() == 0:
                continue
            order = torch.argsort(s, descending=True)
            t_sorted = t[order]
            tp = torch.cumsum(t_sorted, dim=0)
            denom = torch.arange(1, t_sorted.numel() + 1, dtype=torch.float32)
            prec = tp / denom
            ap = prec[t_sorted.bool()].mean().item()
            aps.append(ap)
        if not aps:
            return float("nan")
        return float(sum(aps) / len(aps))

    def _compute_attr_precision(self, scores: torch.Tensor, targets: torch.Tensor) -> float:
        preds = (scores >= 0.5).float()
        targets = targets.float()
        tp = (preds * targets).sum().item()
        fp = (preds * (1 - targets)).sum().item()
        return float(tp / (tp + fp + 1e-8))

    def _compute_attr_recall(self, scores: torch.Tensor, targets: torch.Tensor) -> float:
        preds = (scores >= 0.5).float()
        targets = targets.float()
        tp = (preds * targets).sum().item()
        fn = ((1 - preds) * targets).sum().item()
        return float(tp / (tp + fn + 1e-8))

    def build_model(self, args: argparse.Namespace, device: torch.device, build_program: bool) -> torch.nn.Module:
        spec = self._load_spec(args)
        attr_dim = spec.index_map["attr"].numel()
        full_ds = CUBSDataset(
            root=args.cubs_root,
            classes_file=args.cubs_classes,
            image_labels_file=args.cubs_image_labels,
            images_file=args.cubs_images,
            split_file=args.cubs_split,
            class_attr_file=args.cubs_class_attrs,
            image_attr_file=args.cubs_image_attrs,
            split=None,
        )
        sample = full_ds[0]
        feature_dim = sample["features"].numel()
        gnn = gnn_from_args(args)
        gnn.weak_negation_falsity = True
        reasoner = NeSyReasoner(spec, gnn, init_hidden=args.use_init_hidden)
        model = CUBSNeSyModel(feature_dim, attr_dim, spec.index_map["class"].numel() if spec is not None else 0, reasoner, spec.index_map if spec is not None else None)
        pretrain_path = args.cubs_attr_pretrain_path if args.cubs_attr_pretrain_path is not None else (Path(args.cubs_root) / "attr_mapper.pt")
        freeze_attr = False
        if args.cubs_attr_pretrain_epochs > 0:
            pretrain_full = CUBSDataset(
                root=args.cubs_root,
                classes_file=args.cubs_classes,
                image_labels_file=args.cubs_image_labels,
                images_file=args.cubs_images,
                split_file=args.cubs_split,
                class_attr_file=args.cubs_class_attrs,
                image_attr_file=args.cubs_image_attrs,
                split="train" if args.cubs_use_split else None,
            )
            if not 0.0 < args.cubs_attr_pretrain_split < 1.0:
                raise ValueError("--cubs-attr-pretrain-split must be in (0, 1).")
            pretrain_args = argparse.Namespace(
                cubs_train_fraction=args.cubs_attr_pretrain_split,
                cubs_split_seed=args.cubs_split_seed,
                cubs_root=args.cubs_root,
                cubs_classes=args.cubs_classes,
                cubs_image_labels=args.cubs_image_labels,
                cubs_images=args.cubs_images,
                cubs_split=args.cubs_split,
                cubs_class_attrs=args.cubs_class_attrs,
                cubs_image_attrs=args.cubs_image_attrs,
            )
            train_ds, eval_ds = self._split_cubs_dataset(pretrain_args, dataset=pretrain_full)
            state = self._pretrain_attr_mapper(
                model=model,
                train_ds=train_ds,
                eval_ds=eval_ds,
                device=device,
                lr=args.cubs_attr_pretrain_lr,
                epochs=args.cubs_attr_pretrain_epochs,
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
            print(f"Loaded pretrained CUB attribute mapper from {pretrain_path}.")
        if freeze_attr:
            model.attr_mapper.requires_grad_(False)
        return model.to(device)

    def build_loaders(self, args: argparse.Namespace) -> tuple[Dict[str, DataLoader], bool]:
        if args.cubs_use_split:
            train_ds = CUBSDataset(
                root=args.cubs_root,
                classes_file=args.cubs_classes,
                image_labels_file=args.cubs_image_labels,
                images_file=args.cubs_images,
                split_file=args.cubs_split,
                class_attr_file=args.cubs_class_attrs,
                image_attr_file=args.cubs_image_attrs,
                split="train",
            )
            eval_ds = CUBSDataset(
                root=args.cubs_root,
                classes_file=args.cubs_classes,
                image_labels_file=args.cubs_image_labels,
                images_file=args.cubs_images,
                split_file=args.cubs_split,
                class_attr_file=args.cubs_class_attrs,
                image_attr_file=args.cubs_image_attrs,
                split="test",
            )
        else:
            full_ds = CUBSDataset(
                root=args.cubs_root,
                classes_file=args.cubs_classes,
                image_labels_file=args.cubs_image_labels,
                images_file=args.cubs_images,
                split_file=args.cubs_split,
                class_attr_file=args.cubs_class_attrs,
                image_attr_file=args.cubs_image_attrs,
                split=None,
            )
            train_ds, eval_ds = self._split_cubs_dataset(args, dataset=full_ds)
        loaders = {
            "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=cubs_collate),
            "eval": DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=cubs_collate),
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

        loss = class_loss #+ attr_loss + (outputs["assignments"] * (1 - outputs["assignments"])).mean()

        preds = torch.argmax(outputs["class_probs"], dim=-1)
        attr_probs = outputs["attr_probs"]
        attr_target = batch["attrs"].float()
        attr_acc = ((attr_probs >= 0.5).float() == attr_target).float().mean()
        metrics = {
            "acc": (preds == batch["label"]).float().mean(),
            "loss": loss.detach(),
            "attr_acc": attr_acc,
        }
        return loss, metrics
