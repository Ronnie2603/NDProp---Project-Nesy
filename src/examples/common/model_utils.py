from __future__ import annotations

from typing import Dict, Tuple

import torch


def select(assignments: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if assignments.dim() == 2:
        return assignments.index_select(1, indices)
    return assignments.index_select(1, indices).squeeze(-1)


def unpack_outputs(outputs, return_iterates: bool):
    iterates = None
    if return_iterates:
        assignments, iterates = outputs
    else:
        assignments = outputs
    return assignments, iterates


def append_debug(
    results: Dict[str, torch.Tensor],
    iterates,
    include_tau_phi: bool,
    reasoner,
) -> None:
    if iterates is not None:
        results["iterates"] = iterates
    if include_tau_phi and getattr(reasoner, "last_tau_phi", None) is not None:
        tau_phi = reasoner.last_tau_phi
        if tau_phi is not None:
            results["tau"], results["phi"] = tau_phi
