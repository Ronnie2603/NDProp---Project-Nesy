from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from examples.common import (
    DigitNet,
    NeSyTask,
    RandomDigitTupleDataset,
    gnn_from_args,
    move_batch,
    use_random_eval,
)
from nesy.reasoner import NeSyReasoner

from .datasets import Addition2Dataset, addition2_collate
from .model import Addition2NeSyModel
from .program import build_program


class Addition2Task(NeSyTask):
    name = "addition2"
    description = "Four-digit addition producing a two-digit sum."

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--mnist-root", type=Path, default=Path("./data"), help="Root directory for MNIST.")
        parser.add_argument("--train-pairs", type=Path, default=None, help="Optional training pairs file.")
        parser.add_argument("--eval-pairs", type=Path, default=None, help="Optional eval pairs file.")
        parser.add_argument("--train-limit", type=int, default=-1, help="Cap training samples (-1 for all).")
        parser.add_argument("--eval-limit", type=int, default=1000, help="Cap eval samples (-1 for all).")
        parser.add_argument("--program-path", type=Path, default=None, help="Path to ASP program (lp/aspif).")
        parser.add_argument("--clingo-path", type=str, default=None, help="Optional clingo executable.")

    def build_model(self, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
        program = build_program(args.program_path, args.clingo_path)
        gnn = gnn_from_args(args)
        reasoner = NeSyReasoner(program, gnn, init_hidden=args.use_init_hidden)
        model = Addition2NeSyModel(DigitNet(), reasoner, program.index_map)
        return model.to(device)

    def _maybe_dataset(self, path: Path | None, limit: int, train: bool, args: argparse.Namespace):
        if path is not None and path.exists():
            return Addition2Dataset(
                root=args.mnist_root,
                pairs_file=path,
                limit=None if limit is None or limit < 0 else limit,
                seed=args.seed,
                train=train,
            )
        size = limit if limit is not None and limit > 0 else 1024
        return RandomDigitTupleDataset(
            root=args.mnist_root,
            num_digits=4,
            size=size,
            seed=args.seed if train else args.seed + 7,
            label_mode="addition2",
        )

    def build_loaders(self, args: argparse.Namespace) -> tuple[Dict[str, DataLoader], bool]:
        train_ds = self._maybe_dataset(args.train_pairs, args.train_limit, True, args)
        eval_is_random = use_random_eval(args.eval_pairs, args.train_pairs)
        if eval_is_random:
            eval_size = args.eval_limit if args.eval_limit is not None and args.eval_limit > 0 else 1024
            eval_ds = RandomDigitTupleDataset(
                root=args.mnist_root,
                num_digits=4,
                size=eval_size,
                seed=args.seed + 7,
                label_mode="addition2",
            )
        else:
            eval_ds = Addition2Dataset(
                root=args.mnist_root,
                pairs_file=args.eval_pairs,
                limit=None if args.eval_limit is None or args.eval_limit < 0 else args.eval_limit,
                seed=args.seed + 7,
                train=False,
            )
        loaders = {
            "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=addition2_collate),
            "eval": DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=addition2_collate),
            "eval_sb": DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=addition2_collate),
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
        target = F.one_hot(batch["sum"], num_classes=outputs["sum_probs"].size(1)).float()
        weight = torch.ones_like(target)
        weight[target == 1] = outputs["sum_probs"].size(1) - 1.0
        loss = F.binary_cross_entropy(outputs["sum_probs"], target, weight=weight)

        sum_pred = torch.argmax(outputs["sum_probs"], dim=-1)
        sum_acc = (sum_pred == batch["sum"]).float().mean()
        metrics = {"sum_acc": sum_acc, "loss": loss.detach()}
        return loss, metrics
