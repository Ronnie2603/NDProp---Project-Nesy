from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from nesy.programs.base import ProgramSpec


def _sanitize(name: str) -> str:
    cleaned = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            cleaned.append(ch)
        else:
            cleaned.append("_")
    token = "".join(cleaned)
    if token and token[0].isdigit():
        token = f"c_{token}"
    return token


def build_hierarchy_program(terms: list[str], adj: np.ndarray) -> ProgramSpec:
    atom_names = [f"class({_sanitize(term)})" for term in terms]
    num_atoms = len(atom_names)

    edge_pos = []
    head_edge = []
    rule_idx = 0
    for child_idx in range(adj.shape[0]):
        for parent_idx in range(adj.shape[1]):
            if child_idx == parent_idx:
                continue
            if adj[child_idx, parent_idx] != 0:
                edge_pos.append([child_idx, rule_idx])
                head_edge.append([rule_idx, parent_idx])
                rule_idx += 1

    if edge_pos:
        edge_pos_tensor = torch.tensor(edge_pos, dtype=torch.long).t().contiguous()
        head_edge_tensor = torch.tensor(head_edge, dtype=torch.long).t().contiguous()
    else:
        edge_pos_tensor = torch.empty(2, 0, dtype=torch.long)
        head_edge_tensor = torch.empty(2, 0, dtype=torch.long)
    edge_neg_tensor = torch.empty(2, 0, dtype=torch.long)

    fact_mask = torch.zeros(rule_idx, dtype=torch.bool)
    fact_values = torch.zeros(rule_idx, dtype=torch.float32)
    founded_mask = torch.zeros(num_atoms, dtype=torch.float32)

    class_indices = torch.arange(num_atoms, dtype=torch.long)
    index_map: Dict[str, torch.Tensor] = {"class": class_indices}

    input_rules = torch.arange(rule_idx, rule_idx + num_atoms, dtype=torch.long)
    input_fact_rules = {"obs": input_rules}
    input_head_edge = torch.stack([input_rules, class_indices], dim=0)
    head_edge_tensor = (
        torch.cat([head_edge_tensor, input_head_edge], dim=1) if head_edge_tensor.numel() > 0 else input_head_edge
    )
    fact_mask = torch.cat([fact_mask, torch.ones(num_atoms, dtype=torch.bool)], dim=0)
    fact_values = torch.cat([fact_values, torch.zeros(num_atoms, dtype=torch.float32)], dim=0)
    num_rules = rule_idx + num_atoms

    program_text = "\n".join(
        f"class({_sanitize(terms[parent])}) :- class({_sanitize(terms[child])})."
        for child in range(adj.shape[0])
        for parent in range(adj.shape[1])
        if child != parent and adj[child, parent] != 0
    )

    print(program_text)
    exit()

    return ProgramSpec(
        atom_names=atom_names,
        edge_pos=edge_pos_tensor,
        edge_neg=edge_neg_tensor,
        head_edge=head_edge_tensor,
        index_map=index_map,
        num_atoms=num_atoms,
        num_rules=num_rules,
        fact_mask=fact_mask,
        fact_values=fact_values,
        input_fact_rules=input_fact_rules,
        founded_mask=founded_mask,
        program_text=program_text,
        grounded_aspif="",
    )
