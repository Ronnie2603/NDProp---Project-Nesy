from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from examples.common import append_debug, unpack_outputs
from nesy.reasoner import NeSyReasoner


class CUBSNeSyModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        attr_dim: int,
        num_classes: int,
        reasoner: NeSyReasoner,
        index_map: Dict[str, torch.Tensor],
    ) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.attr_mapper = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, attr_dim),
        )
        self.reasoner = reasoner
        self.index_map = {k: v.clone() for k, v in index_map.items()}
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.attr_dim = attr_dim

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        return_iterates: bool = False,
        return_tau_phi: bool = False,
    ) -> Dict[str, torch.Tensor]:
        raw_feats = batch["features"]
        feats = self.backbone(raw_feats)
        attr_logits = self.attr_mapper(feats)
        attr_probs = torch.sigmoid(attr_logits).detach()
        evidence = {"has": attr_probs}
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
            "attr_probs": attr_probs,
        }
        append_debug(results, iterates, return_tau_phi, self.reasoner)
        return results
