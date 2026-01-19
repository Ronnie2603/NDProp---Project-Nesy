from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from examples.common import append_debug, select, unpack_outputs
from nesy.reasoner import NeSyReasoner
from nesy.programs import QueryProgramSpec


class Member3NeSyModel(nn.Module):
    def __init__(
        self,
        digit_net: nn.Module,
        reasoner: NeSyReasoner,
        index_map: Dict[str, torch.Tensor],
        index_maps_by_query: Dict[int, Dict[str, torch.Tensor]] | None = None,
    ) -> None:
        super().__init__()
        self.digit_net = digit_net
        self.reasoner = reasoner
        self.index_map = {k: v.clone() for k, v in index_map.items()}
        self.index_maps_by_query = None
        if index_maps_by_query is not None:
            self.index_maps_by_query = {
                key: {k: v.clone() for k, v in value.items()} for key, value in index_maps_by_query.items()
            }

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        return_iterates: bool = False,
        return_tau_phi: bool = False,
    ) -> Dict[str, torch.Tensor]:
        imgs = batch["imgs"]
        if imgs.dim() != 5 or imgs.size(1) != 3:
            raise ValueError("imgs must be shaped [batch, 3, 1, 28, 28].")
        bsz = imgs.size(0)
        flat_imgs = imgs.view(-1, imgs.size(2), imgs.size(3), imgs.size(4))
        digit_logits = self.digit_net(flat_imgs)
        digit_probs = torch.softmax(digit_logits, dim=-1).view(bsz, 3, -1)

        evidence = {f"digit{pos}": digit_probs[:, pos, :] for pos in range(3)}
        query = batch["query"].to(digit_probs.device)
        if not isinstance(self.reasoner.program, QueryProgramSpec):
            check_probs = digit_probs.new_zeros((bsz, 10))
            check_probs.scatter_(1, query.view(-1, 1), 1.0)
            evidence["check"] = check_probs

        outputs = self.reasoner(
            evidence,
            queries=query,
            return_iterates=return_iterates,
            return_tau_phi=return_tau_phi,
        )
        assignments, iterates = unpack_outputs(outputs, return_iterates)

        if isinstance(self.reasoner.program, QueryProgramSpec):
            if self.index_maps_by_query is None:
                raise ValueError("index_maps_by_query is required for QueryProgramSpec.")

            query_list = [int(q) for q in query.detach().cpu().tolist()]

            def _stack_indices(key: str) -> torch.Tensor:
                idx_list = []
                for q in query_list:
                    index_map = self.index_maps_by_query.get(q)
                    if index_map is None:
                        raise ValueError(f"Missing index map for query {q}.")
                    idx_list.append(index_map[key].to(device=assignments.device))
                return torch.stack(idx_list, dim=0)

            def _gather(assignments_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
                if assignments_tensor.dim() == 2:
                    return assignments_tensor.gather(1, indices)
                expanded = indices.unsqueeze(-1).expand(-1, -1, assignments_tensor.size(-1))
                gathered = assignments_tensor.gather(1, expanded)
                if gathered.size(-1) == 1:
                    return gathered.squeeze(-1)
                return gathered

            digit_assignments = torch.stack(
                [_gather(assignments, _stack_indices(f"digit{pos}")) for pos in range(3)],
                dim=1,
            )
            member0_probs = _gather(assignments, _stack_indices("member0"))
            member1_probs = _gather(assignments, _stack_indices("member1"))
        else:
            idx = {k: v.to(assignments.device) for k, v in self.index_map.items()}
            digit_assignments = torch.stack([select(assignments, idx[f"digit{pos}"]) for pos in range(3)], dim=1)
            member0_probs = select(assignments, idx["member0"])
            member1_probs = select(assignments, idx["member1"])
        results: Dict[str, torch.Tensor] = {
            "assignments": assignments,
            "member0_probs": member0_probs,
            "member1_probs": member1_probs,
            "digit_probs": digit_assignments,
            "seed_digits": digit_probs,
            "digit_logits": digit_logits.view(bsz, 3, -1),
            "check_probs": check_probs if not isinstance(self.reasoner.program, QueryProgramSpec) else torch.tensor([]),
        }
        append_debug(results, iterates, return_tau_phi, self.reasoner)
        return results
