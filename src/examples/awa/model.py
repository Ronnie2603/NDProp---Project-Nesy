from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from examples.common import append_debug, select, unpack_outputs
from nesy.reasoner import NeSyReasoner


class AwANeSyModel(nn.Module):
    def __init__(self, feature_dim: int, attr_dim: int, num_classes: int, reasoner: NeSyReasoner, index_map: Dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.attr_mapper = nn.Linear(feature_dim, attr_dim)
        self.reasoner = reasoner
        self.index_map = {k: v.clone() for k, v in index_map.items()}
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.attr_dim = attr_dim
        self.attr_fact_generator = nn.Sequential(nn.Linear(attr_dim, attr_dim*2), nn.ReLU(), nn.Linear(attr_dim*2, attr_dim), nn.Sigmoid())

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        return_iterates: bool = False,
        return_tau_phi: bool = False,
    ) -> Dict[str, torch.Tensor]:
        raw_feats = batch["features"]
        feats = torch.sigmoid(self.attr_mapper(raw_feats)).detach()
        feats = self.attr_fact_generator(feats)
        evidence = {"has": feats}
        outputs = self.reasoner(
            evidence,
            return_iterates=return_iterates,
            return_tau_phi=return_tau_phi,
        )
        assignments, iterates = unpack_outputs(outputs, return_iterates)
        idx = {k: v.to(assignments.device) for k, v in self.index_map.items()}
        class_probs = assignments.index_select(1, idx["class"])
        results: Dict[str, torch.Tensor] = {
            "assignments": assignments,
            "class_probs": class_probs,
            "seed_attr": feats,
        }
        append_debug(results, iterates, return_tau_phi, self.reasoner)
        return results
