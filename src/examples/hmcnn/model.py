from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from examples.common import append_debug, unpack_outputs
from nesy.reasoner import NeSyReasoner


class HMCNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        non_lin: str,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        layers = []
        for idx in range(num_layers):
            if idx == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif idx == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
        self.drop = nn.Dropout(dropout)
        self.act = nn.Tanh() if non_lin == "tanh" else nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            if idx == len(self.layers) - 1:
                x = self.sigmoid(layer(x))
            else:
                x = self.act(layer(x))
                x = self.drop(x)
        return x


class HMCNNNeSyModel(nn.Module):
    def __init__(
        self,
        classifier: HMCNNClassifier,
        reasoner: NeSyReasoner,
        index_map: Dict[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.reasoner = reasoner
        self.index_map = {k: v.clone() for k, v in index_map.items()}

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        return_iterates: bool = False,
        return_tau_phi: bool = False,
    ) -> Dict[str, torch.Tensor]:
        feats = batch["features"]
        logits = self.classifier(feats)
        evidence = {"obs": logits.detach()}
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
            "seed_probs": logits,
        }
        append_debug(results, iterates, return_tau_phi, self.reasoner)
        return results
