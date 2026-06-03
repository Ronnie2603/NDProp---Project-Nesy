from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from examples.common import NeSyTask, gnn_from_args, move_batch, use_random_eval
from nesy.reasoner import NeSyReasoner

from .datasets import Apply2x2Dataset, apply2x2_collate, generate_apply2x2_pairs
from .model import Apply2x2NeSyModel, OpNet
from .program import MAX_RESULT, MIN_RESULT, build_program, extract_digit_triples, extract_value_maps


class Apply2x2Task(NeSyTask):
    name = "apply2x2"
    description = "2x2 grid operator application with provided digits and operator images."

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        indexes_dir = Path(__file__).resolve().parents[3] / "indexes"
        parser.add_argument("--train-pairs", type=Path, default=indexes_dir / "apply2x2_train_10000.txt", help="Training pairs file.")
        parser.add_argument("--eval-pairs", type=Path, default=indexes_dir / "apply2x2_test.txt", help="Eval pairs file.")
        parser.add_argument("--train-limit", type=int, default=3000, help="Cap training samples (-1 for all).")
        parser.add_argument("--eval-limit", type=int, default=1000, help="Cap eval samples (-1 for all).")
        parser.add_argument("--program-path", type=Path, default=None, help="Path to ASP program (lp/aspif).")
        parser.add_argument("--clingo-path", type=str, default=None, help="Optional clingo executable.")
        parser.add_argument("--op-root", type=Path, default=Path("./assets/operator_images/hasy"), help="Root directory for operator images.")
        parser.add_argument("--op-train-subdir", type=str, default="train_apply2x2", help="Operator train subdirectory.")
        parser.add_argument("--op-eval-subdir", type=str, default="test_apply2x2", help="Operator eval subdirectory.")

    def build_model(self, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
        program = build_program(args.program_path, args.clingo_path)
        gnn = gnn_from_args(args)
        reasoner = NeSyReasoner(program, gnn, init_hidden=args.use_init_hidden)
        model = Apply2x2NeSyModel(OpNet(), reasoner, program.index_map)
        digit_triples = extract_digit_triples(program.atom_names)
        model.digit_triples = digit_triples
        model.digits_lookup = {triple: idx for idx, triple in enumerate(digit_triples)}
        row_values, col_values = extract_value_maps(program.atom_names)
        model.row_values = torch.tensor(row_values, dtype=torch.long)
        model.col_values = torch.tensor(col_values, dtype=torch.long)
        row_lookup = torch.full((MAX_RESULT - MIN_RESULT + 1,), -1, dtype=torch.long)
        col_lookup = torch.full((MAX_RESULT - MIN_RESULT + 1,), -1, dtype=torch.long)
        for idx, val in enumerate(row_values):
            row_lookup[val - MIN_RESULT] = idx
        for idx, val in enumerate(col_values):
            col_lookup[val - MIN_RESULT] = idx
        model.row_lookup = row_lookup
        model.col_lookup = col_lookup
        return model.to(device)

    def _require_pairs(self, path: Path | None) -> Path:
        if path is None:
            raise ValueError("apply2x2 requires --train-pairs (and optionally --eval-pairs).")
        if not path.exists():
            raise FileNotFoundError(f"Pairs file '{path}' not found.")
        return path

    def build_loaders(self, args: argparse.Namespace) -> tuple[Dict[str, DataLoader], bool]:
        train_pairs = self._require_pairs(args.train_pairs)
        eval_pairs = args.eval_pairs
        eval_is_random = eval_pairs is None or not eval_pairs.exists()
        eval_limit = args.eval_limit
        if eval_is_random and (eval_limit is None or eval_limit < 0):
            eval_limit = 1000
        if eval_is_random:
            if eval_pairs is None:
                eval_pairs = Path("./indexes") / "apply2x2_eval_generated.txt"
            generate_apply2x2_pairs(
                op_root=args.op_root,
                op_eval_subdir=args.op_eval_subdir,
                out_path=eval_pairs,
                size=eval_limit,
                seed=args.seed + 21,
            )
        eval_pairs = self._require_pairs(eval_pairs)
        train_ds = Apply2x2Dataset(
            op_root=args.op_root,
            pairs_file=train_pairs,
            limit=None if args.train_limit is None or args.train_limit < 0 else args.train_limit,
            seed=args.seed,
            train=True,
            op_train_subdir=args.op_train_subdir,
            op_eval_subdir=args.op_eval_subdir,
        )
        eval_ds = Apply2x2Dataset(
            op_root=args.op_root,
            pairs_file=eval_pairs,
            limit=None if eval_limit is None or eval_limit < 0 else eval_limit,
            seed=args.seed + 21,
            train=False,
            op_train_subdir=args.op_train_subdir,
            op_eval_subdir=args.op_eval_subdir,
        )
        loaders = {
            "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=apply2x2_collate),
            "eval": DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=apply2x2_collate),
            "eval_sb": DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=apply2x2_collate),
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
        target_cpu = target.detach().cpu().long()
        if (target_cpu < MIN_RESULT).any() or (target_cpu > MAX_RESULT).any():
            print(batch)
            raise ValueError("Found target value outside apply2x2 result bounds.")
        row_lookup = model.row_lookup.cpu()
        col_lookup = model.col_lookup.cpu()
        row_target_idx = row_lookup[(target_cpu[:, :2] - MIN_RESULT)]
        col_target_idx = col_lookup[(target_cpu[:, 2:] - MIN_RESULT)]
        if (row_target_idx < 0).any() or (col_target_idx < 0).any():
            raise ValueError("Found target value not present in grounded apply2x2 program.")
        row_target_idx = row_target_idx.to(device)
        col_target_idx = col_target_idx.to(device)
        row_target = F.one_hot(row_target_idx.reshape(-1), num_classes=outputs["row_probs"].size(-1)).float()
        col_target = F.one_hot(col_target_idx.reshape(-1), num_classes=outputs["col_probs"].size(-1)).float()

        row_weight = torch.ones_like(row_target)
        row_weight[row_target == 1] = outputs["row_probs"].size(-1) - 1.0

        col_weight = torch.ones_like(col_target)
        col_weight[col_target == 1] = outputs["col_probs"].size(-1) - 1.0

        row_loss = F.binary_cross_entropy(outputs["row_probs"].view(-1, outputs["row_probs"].size(-1)), row_target, weight=row_weight)
        col_loss = F.binary_cross_entropy(outputs["col_probs"].view(-1, outputs["col_probs"].size(-1)), col_target, weight=col_weight)
        loss = row_loss + col_loss

        row_values = model.row_values.to(device)
        col_values = model.col_values.to(device)
        row_pred = row_values[torch.argmax(outputs["row_probs"], dim=-1)]
        col_pred = col_values[torch.argmax(outputs["col_probs"], dim=-1)]
        metrics = {
            "total_acc": ((row_pred == target[:, :2]).all(dim=1) & (col_pred == target[:, 2:]).all(dim=1)).float().mean(),
            "loss": loss.detach(),
        }
        return loss, metrics
