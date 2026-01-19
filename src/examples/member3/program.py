from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch

from nesy.programs import ProgramSpec, build_program_spec, load_program_graph

DEFAULT_PROGRAM_PATH = Path(__file__).resolve().parent / "programs" / "member3.lp"


def _build_index_map(atom_names: List[str]) -> Dict[str, torch.Tensor]:
    name_to_index = {name: idx for idx, name in enumerate(atom_names)}

    def _digit_indices(pos: int) -> torch.Tensor:
        indices: List[int] = []
        for value in range(10):
            atom = f"digit({pos},{value})"
            if atom not in name_to_index:
                raise ValueError(f"Atom '{atom}' not found in grounded program.")
            indices.append(name_to_index[atom])
        return torch.tensor(indices, dtype=torch.long)

    def _member_indices(label: int) -> torch.Tensor:
        indices: List[int] = []
        for value in range(10):
            atom = f"member({value},{label})"
            if atom not in name_to_index:
                raise ValueError(f"Atom '{atom}' not found in grounded program.")
            indices.append(name_to_index[atom])
        return torch.tensor(indices, dtype=torch.long)

    digit_map = {f"digit{pos}": _digit_indices(pos) for pos in range(3)}

    check_indices: List[int] = []
    for value in range(10):
        atom = f"check({value})"
        if atom not in name_to_index:
            raise ValueError(f"Atom '{atom}' not found in grounded program.")
        check_indices.append(name_to_index[atom])

    digit_map["check"] = torch.tensor(check_indices, dtype=torch.long)
    digit_map["member0"] = _member_indices(0)
    digit_map["member1"] = _member_indices(1)
    return digit_map


def build_program(
    program_path: Optional[Path] = None,
    clingo_path: Optional[str] = None,
) -> ProgramSpec:
    path = program_path if program_path is not None else DEFAULT_PROGRAM_PATH
    if not path.exists():
        raise FileNotFoundError(f"Program file '{path}' does not exist.")

    graph = load_program_graph(path, clingo_path)
    index_map = _build_index_map(graph.atom_names)
    input_atoms = {f"digit{pos}": index_map[f"digit{pos}"] for pos in range(3)}
    input_atoms["check"] = index_map["check"]
    return build_program_spec(
        program_path=path,
        clingo_path=clingo_path,
        index_map=index_map,
        input_fact_atoms=input_atoms,
        program_graph=graph,
    )


def build_programs_by_query(
    base: Optional[ProgramSpec] = None,
    program_path: Optional[Path] = None,
    clingo_path: Optional[str] = None,
) -> Dict[int, ProgramSpec]:
    base_spec = base or build_program(program_path, clingo_path)
    return {query: _prune_program_for_query(base_spec, query) for query in range(10)}


def _prune_program_for_query(spec: ProgramSpec, query: int) -> ProgramSpec:
    keep_rule = torch.zeros(spec.num_rules, dtype=torch.bool)
    head_edge = spec.head_edge
    if head_edge.numel() > 0:
        for idx in range(head_edge.size(1)):
            rule_idx = int(head_edge[0, idx].item())
            atom_idx = int(head_edge[1, idx].item())
            atom_name = spec.atom_names[atom_idx]
            if atom_name.startswith("digit("):
                keep_rule[rule_idx] = True
            elif atom_name.startswith(f"member({query},"):
                keep_rule[rule_idx] = True
            elif atom_name == f"check({query})":
                keep_rule[rule_idx] = True
    kept_indices = torch.nonzero(keep_rule, as_tuple=False).view(-1)
    if kept_indices.numel() == 0:
        raise ValueError(f"No rules kept for query {query}.")

    new_rule_idx = torch.full((spec.num_rules,), -1, dtype=torch.long)
    new_rule_idx[kept_indices] = torch.arange(kept_indices.numel(), dtype=torch.long)

    def _filter_edges(edge: torch.Tensor, rule_row: int) -> torch.Tensor:
        if edge.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long)
        mask = keep_rule[edge[rule_row]]
        if mask.numel() == 0 or not mask.any():
            return torch.empty((2, 0), dtype=torch.long)
        filtered = edge[:, mask].clone()
        filtered[rule_row] = new_rule_idx[filtered[rule_row]]
        return filtered

    def _drop_check_edges(edge: torch.Tensor, atom_row: int, check_idx: int) -> torch.Tensor:
        if edge.numel() == 0:
            return edge
        mask = edge[atom_row] != check_idx
        if mask.numel() == 0 or not mask.any():
            return torch.empty((2, 0), dtype=torch.long)
        return edge[:, mask].clone()

    edge_pos = _filter_edges(spec.edge_pos, 1)
    edge_neg = _filter_edges(spec.edge_neg, 1)
    head_edge = _filter_edges(spec.head_edge, 0)

    check_atom = f"check({query})"
    if check_atom in spec.atom_names:
        check_idx = spec.atom_names.index(check_atom)
        edge_pos = _drop_check_edges(edge_pos, 0, check_idx)
        edge_neg = _drop_check_edges(edge_neg, 0, check_idx)
        head_edge = _drop_check_edges(head_edge, 1, check_idx)

    fact_mask = spec.fact_mask[keep_rule].clone() if spec.fact_mask.numel() > 0 else torch.empty(0, dtype=torch.bool)
    fact_values = (
        spec.fact_values[keep_rule].clone() if spec.fact_values.numel() > 0 else torch.empty(0, dtype=torch.float32)
    )

    input_fact_rules: Dict[str, torch.Tensor] = {}
    for key, rules in spec.input_fact_rules.items():
        if rules.numel() == 0:
            input_fact_rules[key] = torch.empty(0, dtype=torch.long)
            continue
        mask = keep_rule[rules]
        if not mask.any():
            input_fact_rules[key] = torch.empty(0, dtype=torch.long)
            continue
        filtered = rules[mask]
        input_fact_rules[key] = new_rule_idx[filtered]
    input_fact_rules.pop("check", None)
    
    keep_atom = torch.zeros(spec.num_atoms, dtype=torch.bool)
    for idx, name in enumerate(spec.atom_names):
        if name.startswith("digit("):
            keep_atom[idx] = True
        elif name.startswith(f"member({query},"):
            keep_atom[idx] = True
        # elif name == f"check({query})":
        #     keep_atom[idx] = True
    for edge in (edge_pos, edge_neg):
        if edge.numel() > 0:
            keep_atom[edge[0]] = True
    if head_edge.numel() > 0:
        keep_atom[head_edge[1]] = True

    kept_atoms = torch.nonzero(keep_atom, as_tuple=False).view(-1)
    if kept_atoms.numel() == 0:
        raise ValueError(f"No atoms kept for query {query}.")
    new_atom_idx = torch.full((spec.num_atoms,), -1, dtype=torch.long)
    new_atom_idx[kept_atoms] = torch.arange(kept_atoms.numel(), dtype=torch.long)

    def _remap_atoms(edge: torch.Tensor, atom_row: int) -> torch.Tensor:
        if edge.numel() == 0:
            return edge
        remapped = edge.clone()
        remapped[atom_row] = new_atom_idx[remapped[atom_row]]
        return remapped

    edge_pos = _remap_atoms(edge_pos, 0)
    edge_neg = _remap_atoms(edge_neg, 0)
    head_edge = _remap_atoms(head_edge, 1)

    atom_names = [spec.atom_names[idx] for idx in kept_atoms.tolist()]
    founded_mask = spec.founded_mask[keep_atom].clone() if spec.founded_mask.numel() > 0 else torch.empty(0, dtype=torch.float32)
    index_map = _build_index_map_for_query(atom_names, query)

    return ProgramSpec(
        atom_names=atom_names,
        edge_pos=edge_pos,
        edge_neg=edge_neg,
        head_edge=head_edge,
        index_map=index_map,
        num_atoms=int(kept_atoms.numel()),
        num_rules=int(keep_rule.sum().item()),
        fact_mask=fact_mask,
        fact_values=fact_values,
        input_fact_rules=input_fact_rules,
        founded_mask=founded_mask,
        program_text=spec.program_text,
        grounded_aspif=spec.grounded_aspif,
    )


def _build_index_map_for_query(atom_names: List[str], query: int) -> Dict[str, torch.Tensor]:
    name_to_index = {name: idx for idx, name in enumerate(atom_names)}

    def _digit_indices(pos: int) -> torch.Tensor:
        indices: List[int] = []
        for value in range(10):
            atom = f"digit({pos},{value})"
            if atom not in name_to_index:
                raise ValueError(f"Atom '{atom}' not found in grounded program.")
            indices.append(name_to_index[atom])
        return torch.tensor(indices, dtype=torch.long)

    digit_map = {f"digit{pos}": _digit_indices(pos) for pos in range(3)}

    # check_atom = f"check({query})"
    # if check_atom not in name_to_index:
    #     raise ValueError(f"Atom '{check_atom}' not found in grounded program.")
    # digit_map["check"] = torch.tensor([name_to_index[check_atom]], dtype=torch.long)

    member0_atom = f"member({query},0)"
    member1_atom = f"member({query},1)"
    if member0_atom not in name_to_index or member1_atom not in name_to_index:
        raise ValueError(f"Member atoms for query {query} not found in grounded program.")
    digit_map["member0"] = torch.tensor([name_to_index[member0_atom]], dtype=torch.long)
    digit_map["member1"] = torch.tensor([name_to_index[member1_atom]], dtype=torch.long)
    return digit_map
