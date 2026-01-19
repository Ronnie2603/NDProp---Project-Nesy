from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
from collections import OrderedDict

import torch

from fuzzy_tp import ProgramGraph, ground_program_to_aspif, load_program_from_aspif, resolve_clingo_command


@dataclass(frozen=True)
class BatchProgram:
    edge_pos: torch.Tensor
    edge_neg: torch.Tensor
    head_edge: torch.Tensor
    atom_ptr: torch.Tensor
    rule_ptr: torch.Tensor
    num_atoms: int
    num_rules: int
    fact_mask: torch.Tensor
    fact_values: torch.Tensor
    input_fact_rules: Dict[str, torch.Tensor]
    founded_mask: torch.Tensor

    def materialize(self, device: torch.device) -> "BatchProgram":
        return BatchProgram(
            edge_pos=self.edge_pos.to(device),
            edge_neg=self.edge_neg.to(device),
            head_edge=self.head_edge.to(device),
            atom_ptr=self.atom_ptr.to(device),
            rule_ptr=self.rule_ptr.to(device),
            num_atoms=self.num_atoms,
            num_rules=self.num_rules,
            fact_mask=self.fact_mask.to(device),
            fact_values=self.fact_values.to(device),
            input_fact_rules={k: v.to(device) for k, v in self.input_fact_rules.items()},
            founded_mask=self.founded_mask.to(device),
        )

    def with_fact_values(self, fact_values: torch.Tensor) -> "BatchProgram":
        return BatchProgram(
            edge_pos=self.edge_pos,
            edge_neg=self.edge_neg,
            head_edge=self.head_edge,
            atom_ptr=self.atom_ptr,
            rule_ptr=self.rule_ptr,
            num_atoms=self.num_atoms,
            num_rules=self.num_rules,
            fact_mask=self.fact_mask,
            fact_values=fact_values,
            input_fact_rules=self.input_fact_rules,
            founded_mask=self.founded_mask,
        )


@dataclass
class ProgramSpec:
    atom_names: list[str]
    edge_pos: torch.Tensor
    edge_neg: torch.Tensor
    head_edge: torch.Tensor
    index_map: Dict[str, torch.Tensor]
    num_atoms: int
    num_rules: int
    fact_mask: torch.Tensor
    fact_values: torch.Tensor
    input_fact_rules: Dict[str, torch.Tensor]
    founded_mask: torch.Tensor
    program_text: str
    grounded_aspif: str
    _batch_cache: Dict[int, BatchProgram] = field(default_factory=dict, init=False, repr=False)
    _device_cache: Dict[tuple[int, str], BatchProgram] = field(default_factory=dict, init=False, repr=False)

    def expand(self, batch_size: int) -> BatchProgram:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        cached = self._batch_cache.get(batch_size)
        if cached is not None:
            return cached

        offsets_atom = torch.arange(batch_size, dtype=torch.long) * self.num_atoms
        offsets_rule = torch.arange(batch_size, dtype=torch.long) * self.num_rules

        def _tile_edges(edge: torch.Tensor, first_offsets: torch.Tensor, second_offsets: torch.Tensor) -> torch.Tensor:
            if edge.numel() == 0:
                return torch.empty((2, 0), dtype=torch.long)
            repeated = edge.repeat(1, batch_size)
            repeated[0] += first_offsets.repeat_interleave(edge.size(1))
            repeated[1] += second_offsets.repeat_interleave(edge.size(1))
            return repeated

        expanded_pos = _tile_edges(self.edge_pos, offsets_atom, offsets_rule)
        expanded_neg = _tile_edges(self.edge_neg, offsets_atom, offsets_rule)
        expanded_head = _tile_edges(self.head_edge, offsets_rule, offsets_atom)

        atom_ptr = torch.arange(batch_size + 1, dtype=torch.long) * self.num_atoms
        rule_ptr = torch.arange(batch_size + 1, dtype=torch.long) * self.num_rules

        expanded_fact_mask = self.fact_mask.repeat(batch_size) if self.fact_mask.numel() > 0 else torch.empty(0, dtype=torch.bool)
        expanded_fact_values = (
            self.fact_values.repeat(batch_size) if self.fact_values.numel() > 0 else torch.empty(0, dtype=torch.float32)
        )
        expanded_founded_mask = (
            self.founded_mask.repeat(batch_size) if self.founded_mask.numel() > 0 else torch.empty(0, dtype=torch.float32)
        )
        expanded_input_rules: Dict[str, torch.Tensor] = {}
        for key, indices in self.input_fact_rules.items():
            if indices.numel() == 0:
                expanded_input_rules[key] = torch.empty(0, dtype=torch.long)
                continue
            expanded_input_rules[key] = (indices.unsqueeze(0) + offsets_rule.unsqueeze(1)).reshape(-1)

        batch_program = BatchProgram(
            edge_pos=expanded_pos,
            edge_neg=expanded_neg,
            head_edge=expanded_head,
            atom_ptr=atom_ptr,
            rule_ptr=rule_ptr,
            num_atoms=self.num_atoms * batch_size,
            num_rules=self.num_rules * batch_size,
            fact_mask=expanded_fact_mask,
            fact_values=expanded_fact_values,
            input_fact_rules=expanded_input_rules,
            founded_mask=expanded_founded_mask,
        )
        self._batch_cache[batch_size] = batch_program
        return batch_program

    def get_batch(self, batch_size: int, device: torch.device | str) -> BatchProgram:
        key = (batch_size, str(device))
        cached = self._device_cache.get(key)
        if cached is not None:
            return cached
        base = self.expand(batch_size)
        materialised = base.materialize(torch.device(device))
        self._device_cache[key] = materialised
        return materialised


@dataclass
class QueryProgramSpec:
    programs_by_query: Dict[int, ProgramSpec]
    default_query: Optional[int] = None
    max_cache_size: int = 16
    num_atoms: int = field(init=False)
    num_rules: int = field(init=False)
    input_fact_rules: Dict[str, torch.Tensor] = field(init=False)
    _device_cache: OrderedDict[tuple[tuple[int, ...], str], BatchProgram] = field(
        default_factory=OrderedDict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.programs_by_query:
            raise ValueError("programs_by_query must not be empty.")
        if self.default_query is None:
            self.default_query = sorted(self.programs_by_query.keys())[0]
        if self.default_query not in self.programs_by_query:
            raise ValueError("default_query must be present in programs_by_query.")

        reference = self.programs_by_query[self.default_query]
        self.num_atoms = reference.num_atoms
        self.num_rules = reference.num_rules
        ref_keys = set(reference.input_fact_rules.keys())
        for query, program in self.programs_by_query.items():
            if program.num_atoms != self.num_atoms or program.num_rules != self.num_rules:
                raise ValueError(f"Program sizes mismatch for query {query}.")
            if set(program.input_fact_rules.keys()) != ref_keys:
                raise ValueError(f"Input fact keys mismatch for query {query}.")
            for key in ref_keys:
                if program.input_fact_rules[key].numel() != reference.input_fact_rules[key].numel():
                    raise ValueError(f"Input fact size mismatch for query {query} and key '{key}'.")
        self.input_fact_rules = {k: v.clone().contiguous() for k, v in reference.input_fact_rules.items()}

    def get_program(self, query: int) -> ProgramSpec:
        return self.programs_by_query.get(query, self.programs_by_query[self.default_query])

    def get_batch_for_queries(self, queries: torch.Tensor, device: torch.device | str) -> BatchProgram:
        if queries.dim() != 1:
            raise ValueError("queries must be a 1D tensor.")
        query_list = [int(q) for q in queries.detach().cpu().tolist()]
        key = (tuple(query_list), str(device))
        cached = self._device_cache.get(key)
        if cached is not None:
            self._device_cache.move_to_end(key)
            return cached

        edge_pos_list = []
        edge_neg_list = []
        head_edge_list = []
        fact_mask_list = []
        fact_values_list = []
        founded_mask_list = []
        input_rules_list: Dict[str, list[torch.Tensor]] = {k: [] for k in self.input_fact_rules.keys()}

        atom_ptr = torch.zeros(len(query_list) + 1, dtype=torch.long)
        rule_ptr = torch.zeros(len(query_list) + 1, dtype=torch.long)
        atom_offset = 0
        rule_offset = 0

        for idx, query in enumerate(query_list):
            program = self.get_program(query)

            if program.edge_pos.numel() > 0:
                edge_pos = program.edge_pos.clone()
                edge_pos[0] += atom_offset
                edge_pos[1] += rule_offset
                edge_pos_list.append(edge_pos)
            if program.edge_neg.numel() > 0:
                edge_neg = program.edge_neg.clone()
                edge_neg[0] += atom_offset
                edge_neg[1] += rule_offset
                edge_neg_list.append(edge_neg)
            if program.head_edge.numel() > 0:
                head_edge = program.head_edge.clone()
                head_edge[0] += rule_offset
                head_edge[1] += atom_offset
                head_edge_list.append(head_edge)

            if program.fact_mask.numel() > 0:
                fact_mask_list.append(program.fact_mask.clone())
            if program.fact_values.numel() > 0:
                fact_values_list.append(program.fact_values.clone())
            if program.founded_mask.numel() > 0:
                founded_mask_list.append(program.founded_mask.clone())

            for key_name, rules in program.input_fact_rules.items():
                if rules.numel() == 0:
                    continue
                input_rules_list[key_name].append(rules + rule_offset)

            atom_offset += program.num_atoms
            rule_offset += program.num_rules
            atom_ptr[idx + 1] = atom_offset
            rule_ptr[idx + 1] = rule_offset

        edge_pos = torch.cat(edge_pos_list, dim=1) if edge_pos_list else torch.empty((2, 0), dtype=torch.long)
        edge_neg = torch.cat(edge_neg_list, dim=1) if edge_neg_list else torch.empty((2, 0), dtype=torch.long)
        head_edge = torch.cat(head_edge_list, dim=1) if head_edge_list else torch.empty((2, 0), dtype=torch.long)

        fact_mask = torch.cat(fact_mask_list, dim=0) if fact_mask_list else torch.empty(0, dtype=torch.bool)
        fact_values = (
            torch.cat(fact_values_list, dim=0) if fact_values_list else torch.empty(0, dtype=torch.float32)
        )
        founded_mask = (
            torch.cat(founded_mask_list, dim=0) if founded_mask_list else torch.empty(0, dtype=torch.float32)
        )

        input_fact_rules: Dict[str, torch.Tensor] = {}
        for key_name, items in input_rules_list.items():
            if items:
                input_fact_rules[key_name] = torch.cat(items, dim=0)
            else:
                input_fact_rules[key_name] = torch.empty(0, dtype=torch.long)

        batch_program = BatchProgram(
            edge_pos=edge_pos,
            edge_neg=edge_neg,
            head_edge=head_edge,
            atom_ptr=atom_ptr,
            rule_ptr=rule_ptr,
            num_atoms=atom_offset,
            num_rules=rule_offset,
            fact_mask=fact_mask,
            fact_values=fact_values,
            input_fact_rules=input_fact_rules,
            founded_mask=founded_mask,
        )
        materialised = batch_program.materialize(torch.device(device))
        if self.max_cache_size > 0:
            self._device_cache[key] = materialised
            if len(self._device_cache) > self.max_cache_size:
                self._device_cache.popitem(last=False)
        return materialised


def _load_program_graph(program_path: Path, clingo_path: Optional[str]) -> ProgramGraph:
    clingo_cmd = resolve_clingo_command(clingo_path)
    suffix = program_path.suffix.lower()
    if suffix == ".aspif":
        return load_program_from_aspif(str(program_path), clingo_cmd, atom_list=[], compute_stable_models=False)
    return ground_program_to_aspif(str(program_path), clingo_cmd)


def load_program_graph(program_path: Path, clingo_path: Optional[str]) -> ProgramGraph:
    """Public wrapper around the program loader."""
    return _load_program_graph(program_path, clingo_path)


def _safe_tensor(tensor: Optional[torch.Tensor], length: int, dtype, fill_value=0) -> torch.Tensor:
    if tensor is None or tensor.numel() == 0:
        return torch.full((length,), fill_value=fill_value, dtype=dtype)
    return tensor.clone().to(dtype=dtype).contiguous()


def build_program_spec(
    *,
    program_path: Path,
    clingo_path: Optional[str],
    index_map: Dict[str, torch.Tensor],
    input_fact_atoms: Dict[str, torch.Tensor],
    program_graph: ProgramGraph | None = None,
) -> ProgramSpec:
    graph = program_graph or _load_program_graph(program_path, clingo_path)
    graph_data = graph.graph

    def _max_or_minus_one(tensor: torch.Tensor, dim: int) -> int:
        return int(tensor[dim].max().item()) if tensor.numel() > 0 else -1

    num_atoms = int(graph_data.num_atoms.item())
    num_rules = int(graph_data.num_rules.item())
    num_atoms = max(
        num_atoms,
        _max_or_minus_one(graph_data.edge_pos, 0) + 1,
        _max_or_minus_one(graph_data.edge_neg, 0) + 1,
        _max_or_minus_one(graph_data.head_edge, 1) + 1,
    )
    num_rules = max(
        num_rules,
        _max_or_minus_one(graph_data.edge_pos, 1) + 1,
        _max_or_minus_one(graph_data.edge_neg, 1) + 1,
        _max_or_minus_one(graph_data.head_edge, 0) + 1,
    )

    head_edge = graph_data.head_edge.clone().contiguous()
    fact_mask = _safe_tensor(getattr(graph_data, "fact_mask", None), num_rules, torch.bool, fill_value=False)
    fact_values = _safe_tensor(getattr(graph_data, "fact_values", None), num_rules, torch.float32, fill_value=0.0)
    founded_mask = _safe_tensor(getattr(graph_data, "founded_mask", None), num_atoms, torch.float32, fill_value=0.0)

    extra_edges = []
    input_fact_rules: Dict[str, torch.Tensor] = {}
    current_rule = num_rules
    for key, atom_indices in input_fact_atoms.items():
        if atom_indices.numel() == 0:
            input_fact_rules[key] = torch.empty(0, dtype=torch.long)
            continue
        new_rules = torch.arange(current_rule, current_rule + atom_indices.numel(), dtype=torch.long)
        extra_edges.append(torch.stack([new_rules, atom_indices], dim=0))
        input_fact_rules[key] = new_rules
        current_rule += atom_indices.numel()

    total_new = current_rule - num_rules
    if total_new > 0:
        merged_extra = torch.cat(extra_edges, dim=1) if len(extra_edges) > 1 else extra_edges[0]
        head_edge = torch.cat([head_edge, merged_extra], dim=1) if head_edge.numel() > 0 else merged_extra
        fact_mask = torch.cat([fact_mask, torch.ones(total_new, dtype=torch.bool)], dim=0)
        fact_values = torch.cat([fact_values, torch.zeros(total_new, dtype=torch.float32)], dim=0)
        num_rules = current_rule

    return ProgramSpec(
        atom_names=list(graph.atom_names),
        edge_pos=graph_data.edge_pos.clone().contiguous(),
        edge_neg=graph_data.edge_neg.clone().contiguous(),
        head_edge=head_edge,
        index_map={k: v.clone().contiguous() for k, v in index_map.items()},
        num_atoms=num_atoms,
        num_rules=num_rules,
        fact_mask=fact_mask,
        fact_values=fact_values,
        input_fact_rules=input_fact_rules,
        founded_mask=founded_mask,
        program_text=getattr(graph, "program_text", ""),
        grounded_aspif=getattr(graph, "grounded_aspif", ""),
    )
