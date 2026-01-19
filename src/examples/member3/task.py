from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from examples.common import DigitNet, NeSyTask, RandomDigitTupleDataset, gnn_from_args, move_batch, use_random_eval
from nesy.reasoner import NeSyReasoner
from nesy.programs import QueryProgramSpec

from .datasets import Member3Dataset, member3_collate
from .model import Member3NeSyModel
from .program import build_programs_by_query


class Member3Task(NeSyTask):
    name = "member3"
    description = "Membership check over three MNIST digits."

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        project_root = Path(__file__).resolve().parents[2]
        neurasp_dir = project_root.parent / "NeurASP" / "examples" / "data"
        parser.add_argument("--mnist-root", type=Path, default=Path("./data"), help="Root directory for MNIST.")
        parser.add_argument("--train-pairs", type=Path, default=neurasp_dir / "member3_train.txt", help="Training pairs file.")
        parser.add_argument("--eval-pairs", type=Path, default=None, help="Optional eval pairs file.")
        parser.add_argument("--train-limit", type=int, default=3000, help="Cap training samples (-1 for all).")
        parser.add_argument("--eval-limit", type=int, default=1000, help="Cap eval samples (-1 for all).")
        parser.add_argument("--program-path", type=Path, default=None, help="Path to ASP program (lp/aspif).")
        parser.add_argument("--clingo-path", type=str, default=None, help="Optional clingo executable.")

    def build_model(self, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
        programs_by_query = build_programs_by_query(None, args.program_path, args.clingo_path)
        program = QueryProgramSpec(programs_by_query, default_query=0)
        gnn = gnn_from_args(args)
        reasoner = NeSyReasoner(program, gnn, init_hidden=args.use_init_hidden)
        index_maps_by_query = {key: spec.index_map for key, spec in programs_by_query.items()}
        model = Member3NeSyModel(DigitNet(), reasoner, programs_by_query[0].index_map, index_maps_by_query)
        return model.to(device)

    def _maybe_dataset(self, path: Path | None, limit: int, train: bool, args: argparse.Namespace):
        if path is not None and path.exists():
            return Member3Dataset(
                root=args.mnist_root,
                pairs_file=path,
                limit=None if limit is None or limit < 0 else limit,
                seed=args.seed,
                train=train,
            )
        size = limit if limit is not None and limit > 0 else 2048
        return RandomDigitTupleDataset(
            root=args.mnist_root,
            num_digits=3,
            size=size,
            seed=args.seed if train else args.seed + 13,
            label_mode="member3",
        )

    def build_loaders(self, args: argparse.Namespace) -> tuple[Dict[str, DataLoader], bool]:
        train_ds = self._maybe_dataset(args.train_pairs, args.train_limit, True, args)
        eval_is_random = use_random_eval(args.eval_pairs, args.train_pairs)
        if eval_is_random:
            eval_size = args.eval_limit if args.eval_limit is not None and args.eval_limit > 0 else 1024
            eval_ds = RandomDigitTupleDataset(
                root=args.mnist_root,
                num_digits=3,
                size=eval_size,
                seed=args.seed + 13,
                label_mode="member3",
            )
        else:
            eval_ds = Member3Dataset(
                root=args.mnist_root,
                pairs_file=args.eval_pairs,
                limit=None if args.eval_limit is None or args.eval_limit < 0 else args.eval_limit,
                seed=args.seed + 13,
                train=False,
            )
        loaders = {
            "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=member3_collate),
            "eval": DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=member3_collate),
            "eval_sb": DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=member3_collate),
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
        query = batch["query"]
        member0_probs = outputs["member0_probs"]
        member1_probs = outputs["member1_probs"]
        if member0_probs.size(1) == 1:
            member0 = member0_probs.squeeze(1)
            member1 = member1_probs.squeeze(1)
        else:
            member0 = member0_probs.gather(1, query.unsqueeze(1)).squeeze(1)
            member1 = member1_probs.gather(1, query.unsqueeze(1)).squeeze(1)
        member_probs = torch.stack([member0, member1], dim=-1)
        loss = F.binary_cross_entropy(member_probs, F.one_hot(batch["member"], num_classes=2).float())

        member_pred = torch.argmax(member_probs, dim=-1)
        member_acc = (member_pred == batch["member"]).float().mean()
        metrics =  {"member_acc": member_acc, "loss": loss.detach()}
        return loss, metrics
