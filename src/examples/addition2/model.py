from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from examples.common import append_debug, select, unpack_outputs
from nesy.reasoner import NeSyReasoner


class Addition2NeSyModel(nn.Module):
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
        imgs = batch["imgs"]
        if imgs.dim() != 5 or imgs.size(1) != 4:
            raise ValueError("imgs must be shaped [batch, 4, 1, 28, 28].")
        bsz = imgs.size(0)
        flat_imgs = imgs.view(-1, imgs.size(2), imgs.size(3), imgs.size(4))
        digit_logits = self.digit_net(flat_imgs)
        digit_probs = torch.softmax(digit_logits, dim=-1).view(bsz, 4, -1)

        evidence = {f"digit{pos}": digit_probs[:, pos, :] for pos in range(4)}
        outputs = self.reasoner(
            evidence,
            return_iterates=return_iterates,
            return_tau_phi=return_tau_phi,
        )
        assignments, iterates = unpack_outputs(outputs, return_iterates)

        idx = {k: v.to(assignments.device) for k, v in self.index_map.items()}
        digit_assignments = torch.stack([select(assignments, idx[f"digit{pos}"]) for pos in range(4)], dim=1)
        results: Dict[str, torch.Tensor] = {
            "assignments": assignments,
            "sum_probs": select(assignments, idx["sum"]),
            "digit_probs": digit_assignments,
            "seed_digits": digit_probs,
            "digit_logits": digit_logits.view(bsz, 4, -1),
        }
        append_debug(results, iterates, return_tau_phi, self.reasoner)
        return results
