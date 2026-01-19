from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from examples.common import DigitNet, NeSyTask, RandomDigitTupleDataset, gnn_from_args, move_batch, use_random_eval
from nesy.reasoner import NeSyReasoner

from .datasets import Add2x2Dataset, add2x2_collate
from .model import Add2x2NeSyModel
from .program import build_program


class Add2x2Task(NeSyTask):
    name = "add2x2"
    description = "2x2 grid addition predicting row and column sums."

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--mnist-root", type=Path, default=Path("./data"), help="Root directory for MNIST.")
        parser.add_argument("--train-pairs", type=Path, default=None, help="Optional training pairs file.")
        parser.add_argument("--eval-pairs", type=Path, default=None, help="Optional eval pairs file.")
        parser.add_argument("--train-limit", type=int, default=3000, help="Cap training samples (-1 for all).")
        parser.add_argument("--eval-limit", type=int, default=1000, help="Cap eval samples (-1 for all).")
        parser.add_argument("--program-path", type=Path, default=None, help="Path to ASP program (lp/aspif).")
        parser.add_argument("--clingo-path", type=str, default=None, help="Optional clingo executable.")

    def build_model(self, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
        program = build_program(args.program_path, args.clingo_path)
        gnn = gnn_from_args(args)
        reasoner = NeSyReasoner(program, gnn, init_hidden=args.use_init_hidden)
        model = Add2x2NeSyModel(DigitNet(), reasoner, program.index_map)
        return model.to(device)

    def _maybe_dataset(self, path: Path | None, limit: int, train: bool, args: argparse.Namespace):
        if path is not None and path.exists():
            return Add2x2Dataset(
                root=args.mnist_root,
                pairs_file=path,
                limit=None if limit is None or limit < 0 else limit,
                seed=args.seed,
                train=train,
            )
        size = limit if limit is not None and limit > 0 else 2048
        return RandomDigitTupleDataset(
            root=args.mnist_root,
            num_digits=4,
            size=size,
            seed=args.seed if train else args.seed + 21,
            label_mode="add2x2",
        )

    def build_loaders(self, args: argparse.Namespace) -> tuple[Dict[str, DataLoader], bool]:
        train_ds = self._maybe_dataset(args.train_pairs, args.train_limit, True, args)
        eval_is_random = use_random_eval(args.eval_pairs, args.train_pairs)
        if eval_is_random:
            eval_size = args.eval_limit if args.eval_limit is not None and args.eval_limit > 0 else 2048
            eval_ds = RandomDigitTupleDataset(
                root=args.mnist_root,
                num_digits=4,
                size=eval_size,
                seed=args.seed + 21,
                label_mode="add2x2",
            )
        else:
            eval_ds = Add2x2Dataset(
                root=args.mnist_root,
                pairs_file=args.eval_pairs,
                limit=None if args.eval_limit is None or args.eval_limit < 0 else args.eval_limit,
                seed=args.seed + 21,
                train=False,
            )
        loaders = {
            "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=add2x2_collate),
            "eval": DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=add2x2_collate),
            "eval_sb": DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=add2x2_collate),
        }
        return loaders, eval_is_random

    def plot_samples(
        self,
        *,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        run_dir: Path,
        epoch: int,
    ) -> None:
        from .plotting import plot_samples

        plot_samples(batch=batch, outputs=outputs, run_dir=run_dir, epoch=epoch)

    def step(self, model: torch.nn.Module, batch: Dict, device: torch.device):
        batch = move_batch(batch, device)
        outputs = model(batch, return_tau_phi=False)
        target = batch["sums"]
        row_target = F.one_hot(target[:, :2].reshape(-1), num_classes=outputs["row_probs"].size(-1)).float()
        col_target = F.one_hot(target[:, 2:].reshape(-1), num_classes=outputs["col_probs"].size(-1)).float()
        
        row_weight = torch.ones_like(row_target)
        row_weight[row_target == 1] = outputs["row_probs"].size(-1) - 1.0
        
        col_weight = torch.ones_like(col_target)
        col_weight[col_target == 1] = outputs["col_probs"].size(-1) - 1.0
        
        row_loss = F.binary_cross_entropy(outputs["row_probs"].view(-1, outputs["row_probs"].size(-1)), row_target, weight=row_weight)
        col_loss = F.binary_cross_entropy(outputs["col_probs"].view(-1, outputs["col_probs"].size(-1)), col_target, weight=col_weight)
        loss = row_loss + col_loss
        
        row_pred = torch.argmax(outputs["row_probs"], dim=-1)
        col_pred = torch.argmax(outputs["col_probs"], dim=-1)
        metrics = {
            "total_acc": ((row_pred == target[:, :2]).all(dim=1) & (col_pred == target[:, 2:]).all(dim=1)).float().mean(),
            "loss": loss.detach(),
        }
        return loss, metrics
