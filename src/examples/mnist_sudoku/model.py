from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from examples.common import append_debug, unpack_outputs
from nesy.reasoner import NeSyReasoner


# class SudokuDigitNet(nn.Module):
#     def __init__(self, num_classes: int = 5, in_channels: int = 1) -> None:
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, 32, 3, 1, 1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(32, 64, 3, 1, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(64, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((4, 4)),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(128 * 4 * 4, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, num_classes),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.encoder(x)
#         x = x.view(x.size(0), -1)
#         return self.classifier(x)


class SudokuDigitNet(nn.Module):
    def __init__(self, num_classes: int = 1, in_channels: int = 1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class MNISTSudokuNeSyModel(nn.Module):
    def __init__(
        self,
        digit_net: nn.Module,
        reasoner: NeSyReasoner,
        index_map: Dict[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.digit_net = digit_net
        self.reasoner = reasoner
        self.index_map = {k: v.clone() for k, v in index_map.items()}
        num_vals = int(self.index_map["digit"].size(1))
        self.expected_classes = num_vals + 1
        self.hidden_gen: nn.Module | None = None
        hidden_dim = getattr(self.reasoner.model, "hidden_dim", None)
        if hidden_dim is not None and hidden_dim > 0:
            cells = int(self.index_map["digit"].size(0))
            input_dim = cells * self.expected_classes
            logical_dim = getattr(self.reasoner.model, "logical_dimension", 1)
            output_dim = hidden_dim * self.reasoner.program.num_atoms * logical_dim
            self.hidden_gen = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                # nn.LayerNorm(output_dim),
                nn.ReLU(),
            )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        return_iterates: bool = False,
        return_tau_phi: bool = False,
    ) -> Dict[str, torch.Tensor]:
        images = batch["images"]
        batch_size, cells, channels, height, width = images.shape
        num_vals = int(self.index_map["digit"].size(1))
        expected = getattr(self.digit_net.encoder[0], "in_channels", channels)
        if channels != expected:
            if channels == 1 and expected == 3:
                images = images.repeat(1, 1, 3, 1, 1)
                channels = 3
            elif channels == 3 and expected == 1:
                images = images[:, :, :1, :, :]
                channels = 1
            else:
                raise ValueError(f"Digit net expects {expected} channels but got {channels}.")
        flat = images.view(batch_size * cells, channels, height, width)
        with torch.no_grad():
            logits = self.digit_net(flat).detach()
        if logits.size(-1) < num_vals + 1:
            raise ValueError(f"Digit net output has {logits.size(-1)} classes but need {num_vals + 1}.")
        if logits.size(-1) > num_vals + 1:
            logits = logits[:, : num_vals + 1]
        probs = F.softmax(logits, dim=-1)
        probs_full = probs.view(batch_size, cells, probs.size(-1))
        if probs_full.size(-1) < self.expected_classes:
            pad = torch.zeros(
                (batch_size, cells, self.expected_classes - probs_full.size(-1)),
                device=probs_full.device,
                dtype=probs_full.dtype,
            )
            probs_full = torch.cat([pad, probs_full], dim=-1)
        elif probs_full.size(-1) > self.expected_classes:
            probs_full = probs_full[:, :, : self.expected_classes]
        blank_mask = torch.argmax(probs, dim=-1) == 0
        digit_evidence = probs[:, 1 : 1 + num_vals]
        digit_evidence[blank_mask] = 0.0
        digit_evidence = digit_evidence.view(batch_size, cells, num_vals)

        evidence = {"digit": digit_evidence.reshape(batch_size, -1)}
        initial_hidden = None
        if self.hidden_gen is not None:
            concat = probs_full.reshape(batch_size, -1)
            hidden_flat = self.hidden_gen(concat)
            logical_dim = getattr(self.reasoner.model, "logical_dimension", 1)
            hidden_flat = hidden_flat.view(batch_size, self.reasoner.program.num_atoms * logical_dim, -1)
            initial_hidden = hidden_flat.view(1, batch_size * self.reasoner.program.num_atoms * logical_dim, -1)
        outputs = self.reasoner(
            evidence,
            initial_hidden=initial_hidden,
            return_iterates=return_iterates,
            return_tau_phi=return_tau_phi,
        )
        assignments, iterates = unpack_outputs(outputs, return_iterates)

        idx = self.index_map["digit"].to(assignments.device)
        digit_atoms = assignments.index_select(1, idx.view(-1))
        digit_atoms = digit_atoms.view(batch_size, idx.size(0), idx.size(1))

        results: Dict[str, torch.Tensor] = {
            "assignments": assignments,
            "digit_probs": probs_full,
            "sudoku_probs": digit_atoms,
        }
        append_debug(results, iterates, return_tau_phi, self.reasoner)
        return results
