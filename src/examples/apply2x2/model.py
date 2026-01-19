from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from examples.common import append_debug, select, unpack_outputs
from nesy.reasoner import NeSyReasoner


class OpNet(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 5),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class Apply2x2NeSyModel(nn.Module):
    def __init__(self, op_net: nn.Module, reasoner: NeSyReasoner, index_map: Dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.op_net = op_net
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
            raise ValueError("imgs must be shaped [batch, 4, 1, H, W].")
        bsz = imgs.size(0)
        flat_imgs = imgs.view(-1, imgs.size(2), imgs.size(3), imgs.size(4))
        op_logits = self.op_net(flat_imgs)
        op_probs = torch.softmax(op_logits, dim=-1).view(bsz, 4, -1)

        digits = batch["digits"].long()
        if not hasattr(self, "digits_lookup"):
            raise ValueError("digits_lookup missing on Apply2x2NeSyModel.")
        indices = []
        for triple in digits.detach().cpu().tolist():
            idx = self.digits_lookup.get(tuple(int(v) for v in triple), -1)
            if idx < 0:
                raise ValueError(f"Digit triple {triple} not found in grounded program.")
            indices.append(idx)
        digit_indices = torch.tensor(indices, device=digits.device)
        digit_probs = F.one_hot(digit_indices, num_classes=len(self.digit_triples)).float()

        evidence = {f"op{pos}": op_probs[:, pos, :] for pos in range(4)}
        evidence["digits"] = digit_probs
        outputs = self.reasoner(
            evidence,
            return_iterates=return_iterates,
            return_tau_phi=return_tau_phi,
        )
        assignments, iterates = unpack_outputs(outputs, return_iterates)

        idx = {k: v.to(assignments.device) for k, v in self.index_map.items()}
        rows = torch.stack([select(assignments, idx["row0"]), select(assignments, idx["row1"])], dim=1)
        cols = torch.stack([select(assignments, idx["col0"]), select(assignments, idx["col1"])], dim=1)
        results: Dict[str, torch.Tensor] = {
            "assignments": assignments,
            "row_probs": rows,
            "col_probs": cols,
            "seed_ops": op_probs,
            "seed_digits": digit_probs,
            "op_logits": op_logits.view(bsz, 4, -1),
        }
        if hasattr(self, "row_values"):
            results["row_values"] = self.row_values.to(assignments.device)
        if hasattr(self, "col_values"):
            results["col_values"] = self.col_values.to(assignments.device)
        append_debug(results, iterates, return_tau_phi, self.reasoner)
        return results
