from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from examples.common import append_debug, select, unpack_outputs
from nesy.reasoner import NeSyReasoner


class AdditionNeSyModel(nn.Module):
    def __init__(self, digit_net: nn.Module, reasoner: NeSyReasoner, index_map: Dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.digit_net = digit_net
        self.reasoner = reasoner
        self.index_map = {k: v.clone() for k, v in index_map.items()}

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        return_iterates: bool = False,
        return_tau_phi: bool = False,
    ) -> Dict[str, torch.Tensor]:
        logits_a = self.digit_net(batch["img_a"])
        logits_b = self.digit_net(batch["img_b"])
        p_a = torch.softmax(logits_a, dim=-1)
        p_b = torch.softmax(logits_b, dim=-1)
        evidence = {"digit0": p_a, "digit1": p_b}

        outputs = self.reasoner(
            evidence,
            return_iterates=return_iterates,
            return_tau_phi=return_tau_phi,
        )
        assignments, iterates = unpack_outputs(outputs, return_iterates)

        idx = {k: v.to(assignments.device) for k, v in self.index_map.items()}
        results: Dict[str, torch.Tensor] = {
            "assignments": assignments,
            "sum_probs": select(assignments, idx["sum"]),
            "a_digit_probs": select(assignments, idx["digit0"]),
            "b_digit_probs": select(assignments, idx["digit1"]),
            "seed_a": p_a,
            "seed_b": p_b,
            "digit_logits_a": logits_a,
            "digit_logits_b": logits_b,
        }
        append_debug(results, iterates, return_tau_phi, self.reasoner)
        return results
