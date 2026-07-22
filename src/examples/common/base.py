from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from ndprop import NDProp


class NeSyTask(ABC):
    name: str
    description: str

    @abstractmethod
    def add_args(self, parser: argparse.ArgumentParser) -> None: ...

    @abstractmethod
    def build_model(self, args: argparse.Namespace, device: torch.device) -> torch.nn.Module: ...

    @abstractmethod
    def build_loaders(self, args: argparse.Namespace) -> Tuple[Dict[str, DataLoader], bool]: ...

    @abstractmethod
    def step(self, model: torch.nn.Module, batch: Dict, device: torch.device) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]: ...

    def plot_samples(
        self,
        *,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        run_dir: Path,
        epoch: int,
    ) -> None:
        return


def gnn_from_args(args: argparse.Namespace) -> NDProp:
    return NDProp(
        num_iters=args.iters,
        tnorm=args.tnorm,
        tnorm_wfs=getattr(args, "tnorm_wfs", None),
        hidden_dim=args.hidden_dim,
        logical_dimension=args.logical_dimension,
        use_convergence=not args.disable_convergence,
        convergence_tol=args.convergence_tol,
        decision_normalization=getattr(args, "decision_normalization", "none"),
    )
