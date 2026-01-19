from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from examples.common import DigitNet, NeSyTask, default_transform, gnn_from_args, move_batch, use_random_eval
from nesy.reasoner import NeSyReasoner

from .datasets import AdditionDataset, addition_collate
from .model import AdditionNeSyModel
from .program import build_program


class AdditionTask(NeSyTask):
    name = "addition"
    description = "MNIST digit pair addition with single-digit sum."

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        project_root = Path(__file__).resolve().parents[2]
        neurasp_dir = project_root.parent / "NeurASP" / "examples" / "mnistAdd" / "data"
        parser.add_argument("--mnist-root", type=Path, default=Path("./data"), help="Root directory for MNIST.")
        parser.add_argument("--pairs-per-sample", type=int, default=2, help="Random pairs sampled per digit.")
        parser.add_argument("--train-limit", type=int, default=30000, help="Max training pairs to sample (-1 for all).")
        parser.add_argument("--eval-limit", type=int, default=1000, help="Max eval pairs to sample (-1 for all).")
        parser.add_argument("--train-pairs", type=Path, default=neurasp_dir / "train_data.txt", help="Optional NeurASP pairs file.")
        parser.add_argument("--eval-pairs", type=Path, default=neurasp_dir / "test_data.txt", help="Optional NeurASP eval pairs file.")
        parser.add_argument("--program-path", type=Path, default=None, help="Path to ASP program (lp/aspif).")
        parser.add_argument("--clingo-path", type=str, default=None, help="Optional clingo executable.")

    def build_model(self, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
        program = build_program(args.program_path, args.clingo_path)
        gnn = gnn_from_args(args)
        reasoner = NeSyReasoner(program, gnn, init_hidden=args.use_init_hidden)
        model = AdditionNeSyModel(DigitNet(), reasoner, program.index_map)
        return model.to(device)

    def build_loaders(self, args: argparse.Namespace) -> tuple[Dict[str, DataLoader], bool]:
        transform = default_transform()
        train_ds = AdditionDataset(
            root=args.mnist_root,
            train=True,
            pairs_per_sample=args.pairs_per_sample,
            pairs_file=args.train_pairs if args.train_pairs.exists() else None,
            limit=None if args.train_limit is None or args.train_limit < 0 else args.train_limit,
            transform=transform,
            seed=args.seed,
        )
        eval_is_random = use_random_eval(args.eval_pairs, args.train_pairs)
        eval_pairs = None if eval_is_random else args.eval_pairs
        eval_ds = AdditionDataset(
            root=args.mnist_root,
            train=False,
            pairs_per_sample=1,
            pairs_file=eval_pairs,
            limit=None if args.eval_limit is None or args.eval_limit < 0 else args.eval_limit,
            transform=transform,
            seed=args.seed + 1,
        )
        
        loaders = {
            "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=addition_collate),
            "eval": DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=addition_collate),
            "eval_sb": DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=addition_collate),
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

        # stability_loss = F.mse_loss(outputs["seed_a"], outputs["a_digit_probs"]) + F.mse_loss(outputs["seed_b"], outputs["b_digit_probs"])
        # stability_loss = F.cross_entropy(outputs["digit_logits_a"], torch.argmax(outputs["a_digit_probs"].detach(), dim=-1)) + F.cross_entropy(outputs["digit_logits_b"], torch.argmax(outputs["b_digit_probs"].detach(), dim=-1))
        # loss = loss + stability_loss

        sum_probs = outputs["sum_probs"]
        sum_pred = torch.argmax(sum_probs, dim=-1)
        sum_acc = (sum_pred == batch["sum"]).float().mean()
        metrics = {"sum_acc": sum_acc, "loss": loss.detach()}
        return loss, metrics
