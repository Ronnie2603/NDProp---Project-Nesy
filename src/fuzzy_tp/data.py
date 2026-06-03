from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm


LOGGER = logging.getLogger(__name__)


@dataclass
class ProgramGraph:
    """Lightweight container for a grounded normal program instance."""

    graph: Data
    atom_names: List[str]
    program_text: str
    stable_model: List[str]
    stable_models: List[List[str]]
    founded_atoms: List[str]

    @property
    def num_atoms(self) -> int:
        return int(self.graph.num_atoms.item())

    @property
    def num_rules(self) -> int:
        return int(self.graph.num_rules.item())


class NormalProgramGenerator:
    """Produces random normal logic programs (one head atom per rule)."""

    def __init__(
        self,
        atoms_range: Tuple[int, int],
        rules_range: Tuple[int, int],
        max_body: int,
        negation_prob: float,
        fact_rate: float = 0.2,
        seed: Optional[int] = None,
    ):
        if max_body < 1:
            raise ValueError("max_body must be at least 1.")
        self.atoms_range = atoms_range
        self.rules_range = rules_range
        self.max_body = max_body
        self.negation_prob = negation_prob
        self.fact_rate = max(0.0, min(1.0, fact_rate))
        self.random = random.Random(seed)

    def _sample_atoms(self, num_rules: int) -> List[str]:
        min_atoms, max_atoms = self.atoms_range
        upper = max(1, min(max_atoms, num_rules))
        lower = max(1, min(min_atoms, upper))
        count = self.random.randint(lower, upper) if lower < upper else lower
        return [f"a{i}" for i in range(count)]

    def _sample_body(self, atoms: Sequence[str]) -> List[str]:
        max_unique = len(atoms) * (2 if self.negation_prob > 0 else 1)
        body_len = min(self.random.randint(1, self.max_body), max_unique)
        chosen = set()
        literals: List[str] = []
        while len(literals) < body_len:
            atom = self.random.choice(atoms)
            negated = self.random.random() < self.negation_prob
            literal = f"not {atom}" if negated else atom
            if literal in chosen:
                continue
            chosen.add(literal)
            literals.append(literal)
        return literals

    def sample(self) -> Tuple[str, List[str]]:
        min_rules, max_rules = self.rules_range
        num_rules = self.random.randint(min_rules, max_rules)
        atoms = self._sample_atoms(num_rules)
        rules: List[Tuple[str, List[str]]] = []

        def _maybe_fact() -> bool:
            return self.fact_rate > 0.0 and self.random.random() < self.fact_rate

        for atom in atoms:
            body = [] if _maybe_fact() else self._sample_body(atoms)
            rules.append((atom, body))

        extra_rules = max(0, num_rules - len(atoms))
        for _ in range(extra_rules):
            head = self.random.choice(atoms)
            body = [] if _maybe_fact() else self._sample_body(atoms)
            rules.append((head, body))

        self.random.shuffle(rules)
        rendered_rules = []
        for head, body in rules:
            if body:
                rendered_rules.append(f"{head} :- {', '.join(body)}.")
            else:
                rendered_rules.append(f"{head}.")
        program_text = "\n".join(rendered_rules)
        return program_text, atoms


def resolve_clingo_command(explicit_path: Optional[str]) -> List[str]:
    if explicit_path:
        resolved = shutil.which(explicit_path) if not explicit_path.startswith("/") else explicit_path
        if resolved is None:
            raise FileNotFoundError(f"Specified clingo executable '{explicit_path}' was not found.")
        return [resolved]
    resolved = shutil.which("clingo")
    if resolved is None:
        try:
            __import__("clingo")
        except ImportError as exc:
            raise FileNotFoundError(
                "Could not locate 'clingo' on PATH and the Python clingo module is not installed. "
                "Install clingo or provide --clingo-path."
            ) from exc
        return [sys.executable, "-m", "clingo"]
    return [resolved]


def get_stable_models(
    clingo_cmd: Sequence[str],
    program_text: Optional[str] = None,
    file_path: Optional[str] = None,
    aspif: bool = False,
    enumerate_all: bool = True,
) -> List[List[str]]:
    """Returns all stable models reported by clingo."""
    if (program_text is None) == (file_path is None):
        raise ValueError("Exactly one of program_text or file_path must be provided.")
    try:
        if file_path is not None:
            cmd = list(clingo_cmd) + ["--outf=2"]
            if enumerate_all:
                cmd += ['-n', '0']
            if aspif:
                cmd.append("--mode=clingo")
            cmd.append(file_path)
            result = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                check=False,
            )
        else:
            cmd = list(clingo_cmd) + ["--outf=2", "-"]
            if enumerate_all:
                cmd += ['-n', '0']
            if aspif:
                cmd.insert(-1, "--mode=clingo")
            result = subprocess.run(
                cmd,
                input=program_text,
                text=True,
                capture_output=True,
                check=False,
            )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Failed to execute clingo ({clingo_cmd[0]}). Ensure it is installed and accessible."
        ) from exc
    if result.returncode not in (0, 10, 20, 30):
        LOGGER.error("Clingo stderr:\n%s", result.stderr.strip())
        LOGGER.error("Clingo stdout:\n%s", result.stdout.strip())
        raise RuntimeError(
            "clingo execution failed with unexpected exit code. See logs above. "
            "You can provide a different executable via --clingo-path."
        )
    if result.stderr.strip() and not "does not occur in any rule head" in result.stderr.strip():
        LOGGER.warning("Clingo reported diagnostics:\n%s", result.stderr.strip())
    payload = json.loads(result.stdout or "{}")
    if payload.get("Result") != "SATISFIABLE":
        return []
    witnesses = payload.get("Call", [{}])[0].get("Witnesses", [])
    return [witness.get("Value", []) for witness in witnesses]


def compute_stable_model(
    clingo_cmd: Sequence[str],
    program_text: Optional[str] = None,
    file_path: Optional[str] = None,
    aspif: bool = False,
) -> Optional[List[str]]:
    models = get_stable_models(
        clingo_cmd,
        program_text=program_text,
        file_path=file_path,
        aspif=aspif,
    )
    return models[0] if models else None


def atoms_to_tensor(atoms: List[str], stable_model: Sequence[str]) -> torch.Tensor:
    vector = torch.zeros(len(atoms), dtype=torch.float32)
    atom_set = set(stable_model)
    for idx, atom in enumerate(atoms):
        if atom in atom_set:
            vector[idx] = 1.0
    return vector


def program_to_graph(
    program_text: str,
    atoms: List[str],
    stable_models: Sequence[Sequence[str]],
) -> ProgramGraph:
    if not stable_models:
        raise ValueError("At least one stable model is required to build the program graph.")
    atom_to_index = {atom: idx for idx, atom in enumerate(atoms)}

    edge_pos: List[List[int]] = [[], []]
    edge_neg: List[List[int]] = [[], []]
    head_edge: List[List[int]] = [[], []]
    fact_atom_indices: set[int] = set()
    fact_rule_flags: List[bool] = []

    rule_counter = 0

    for line in program_text.splitlines():
        stripped = line.strip().rstrip(".")
        if not stripped:
            continue
        if ":-" in stripped:
            head_part, body_part = [part.strip() for part in stripped.split(":-", maxsplit=1)]
            body_literals = [lit.strip() for lit in body_part.split(",") if lit.strip()]
        else:
            head_part = stripped
            body_literals = []

        head_atom = head_part
        if head_atom == "":
            raise ValueError("Constraint rules are not supported.")
        if head_atom not in atom_to_index:
            raise ValueError(f"Unknown head atom {head_atom}.")

        head_edge[0].append(rule_counter)
        head_edge[1].append(atom_to_index[head_atom])

        is_fact_rule = len(body_literals) == 0

        for literal in body_literals:
            is_neg = literal.startswith("not ")
            atom = literal[4:] if is_neg else literal
            if atom not in atom_to_index:
                raise ValueError(f"Unknown atom {atom} in body.")
            if is_neg:
                edge_neg[0].append(atom_to_index[atom])
                edge_neg[1].append(rule_counter)
            else:
                edge_pos[0].append(atom_to_index[atom])
                edge_pos[1].append(rule_counter)

        fact_rule_flags.append(is_fact_rule)

        rule_counter += 1

        if is_fact_rule:
            fact_atom_indices.add(atom_to_index[head_atom])

    edge_pos_tensor = (
        torch.tensor(edge_pos, dtype=torch.long)
        if edge_pos[0]
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_neg_tensor = (
        torch.tensor(edge_neg, dtype=torch.long)
        if edge_neg[0]
        else torch.empty((2, 0), dtype=torch.long)
    )
    head_edge_tensor = (
        torch.tensor(head_edge, dtype=torch.long)
        if head_edge[0]
        else torch.empty((2, 0), dtype=torch.long)
    )

    p0 = torch.zeros(len(atoms), dtype=torch.float32)
    if fact_atom_indices:
        fact_indices_tensor = torch.tensor(sorted(fact_atom_indices), dtype=torch.long)
        p0[fact_indices_tensor] = 1.0
    primary_model = stable_models[0]
    target = atoms_to_tensor(atoms, primary_model)
    stable_target_tensors = [atoms_to_tensor(atoms, model) for model in stable_models]
    stable_targets = torch.stack(stable_target_tensors, dim=0)

    founded_atoms: set[str] = set(atoms)
    for model in stable_models:
        founded_atoms &= set(model)
    founded_mask = torch.tensor(
        [1.0 if atom in founded_atoms else 0.0 for atom in atoms],
        dtype=torch.float32,
    )

    graph = Data()
    graph.edge_pos = edge_pos_tensor
    graph.edge_neg = edge_neg_tensor
    graph.head_edge = head_edge_tensor
    graph.p0 = p0
    graph.y = target
    graph.num_atoms = torch.tensor(len(atoms), dtype=torch.long)
    graph.num_rules = torch.tensor(rule_counter, dtype=torch.long)
    graph.stable_targets = stable_targets
    graph.fact_mask = (
        torch.tensor(fact_rule_flags, dtype=torch.bool)
        if fact_rule_flags
        else torch.empty(0, dtype=torch.bool)
    )
    graph.fact_values = (
        torch.tensor(
            [1.0 if is_fact else 0.0 for is_fact in fact_rule_flags],
            dtype=torch.float32,
        )
        if fact_rule_flags
        else torch.empty(0, dtype=torch.float32)
    )
    graph.founded_mask = founded_mask

    return ProgramGraph(
        graph=graph,
        atom_names=atoms,
        program_text=program_text,
        stable_model=list(primary_model),
        stable_models=[list(model) for model in stable_models],
        founded_atoms=sorted(founded_atoms),
    )


def collate_programs(programs: Sequence[ProgramGraph], no_stable_model = False) -> Data:
    atom_ptr = [0]
    rule_ptr = [0]
    p0_list = []
    y_list = []
    edge_pos_list = []
    edge_neg_list = []
    head_edge_list = []
    stable_values_list = []
    stable_value_ptr = [0]
    stable_model_ptr = [0]
    fact_mask_list: List[torch.Tensor] = []
    fact_values_list: List[torch.Tensor] = []
    founded_mask_list: List[torch.Tensor] = []

    atom_offset = 0
    rule_offset = 0

    for prog in programs:
        g = prog.graph
        num_atoms = int(g.num_atoms.item())
        num_rules = int(g.num_rules.item())
        if not no_stable_model:
            stable_targets = getattr(g, "stable_targets", None)
            if stable_targets is None:
                raise ValueError("Program graph is missing stable_targets information.")

        if g.edge_pos.numel() > 0:
            edge_pos_list.append(
                g.edge_pos + torch.tensor([[atom_offset], [rule_offset]], dtype=torch.long)
            )
        if g.edge_neg.numel() > 0:
            edge_neg_list.append(
                g.edge_neg + torch.tensor([[atom_offset], [rule_offset]], dtype=torch.long)
            )
        if g.head_edge.numel() > 0:
            head_edge_list.append(
                g.head_edge + torch.tensor([[rule_offset], [atom_offset]], dtype=torch.long)
            )

        fact_mask = getattr(g, "fact_mask", None)
        if fact_mask is not None:
            fact_mask_list.append(fact_mask)
        else:
            fact_mask_list.append(torch.zeros(num_rules, dtype=torch.bool))
        fact_values = getattr(g, "fact_values", None)
        if fact_values is not None:
            fact_values_list.append(fact_values)
        else:
            default_values = torch.zeros(num_rules, dtype=torch.float32)
            if fact_mask_list[-1].numel() == num_rules:
                default_values[fact_mask_list[-1]] = 1.0
            fact_values_list.append(default_values)

        p0_list.append(g.p0)
        if not no_stable_model:
            y_list.append(g.y)
        if not no_stable_model:
            num_stable_models = int(stable_targets.size(0)) if stable_targets.numel() > 0 else 0
            for idx in range(num_stable_models):
                flat = stable_targets[idx].reshape(-1)
                stable_values_list.append(flat)
                stable_value_ptr.append(stable_value_ptr[-1] + flat.numel())
            stable_model_ptr.append(stable_model_ptr[-1] + num_stable_models)

        atom_offset += num_atoms
        rule_offset += num_rules
        atom_ptr.append(atom_offset)
        rule_ptr.append(rule_offset)

        founded = getattr(g, "founded_mask", None)
        if founded is None or founded.numel() == 0:
            founded_mask_list.append(torch.zeros(num_atoms, dtype=torch.float32))
        else:
            if founded.numel() != num_atoms:
                raise ValueError("founded_mask length does not match number of atoms.")
            founded_mask_list.append(founded)

    data = Data()
    data.p0 = torch.cat(p0_list, dim=0) if p0_list else torch.empty(0)
    if not no_stable_model:
        data.y = torch.cat(y_list, dim=0) if y_list else torch.empty(0)

    data.edge_pos = (
        torch.cat(edge_pos_list, dim=1)
        if edge_pos_list
        else torch.empty((2, 0), dtype=torch.long)
    )
    data.edge_neg = (
        torch.cat(edge_neg_list, dim=1)
        if edge_neg_list
        else torch.empty((2, 0), dtype=torch.long)
    )
    data.head_edge = (
        torch.cat(head_edge_list, dim=1)
        if head_edge_list
        else torch.empty((2, 0), dtype=torch.long)
    )

    data.atom_ptr = torch.tensor(atom_ptr, dtype=torch.long)
    data.rule_ptr = torch.tensor(rule_ptr, dtype=torch.long)
    data.num_programs = torch.tensor(len(programs), dtype=torch.long)
    data.num_atoms = torch.tensor(atom_offset, dtype=torch.long)
    data.num_rules = torch.tensor(rule_offset, dtype=torch.long)
    data.fact_mask = (
        torch.cat(fact_mask_list, dim=0)
        if fact_mask_list
        else torch.empty(0, dtype=torch.bool)
    )
    data.fact_values = (
        torch.cat(fact_values_list, dim=0)
        if fact_values_list
        else torch.empty(0, dtype=torch.float32)
    )
    data.founded_mask = (
        torch.cat(founded_mask_list, dim=0)
        if founded_mask_list
        else torch.empty(0, dtype=torch.float32)
    )
    data.stable_values = (
        torch.cat(stable_values_list, dim=0)
        if stable_values_list
        else torch.empty(0, dtype=torch.float32)
    )
    data.stable_value_ptr = torch.tensor(stable_value_ptr, dtype=torch.long)
    data.stable_model_ptr = torch.tensor(stable_model_ptr, dtype=torch.long)
    return data


def parse_program_text(program_text: str) -> Tuple[List[Tuple[str, List[str]]], List[str]]:
    lines = [line.strip() for line in program_text.splitlines() if line.strip() and not line.strip().startswith("%")]
    atoms_order: List[str] = []
    atom_set = set()
    rules: List[Tuple[str, List[str]]] = []

    def _register_atom(atom: str):
        if atom not in atom_set:
            atom_set.add(atom)
            atoms_order.append(atom)

    for line in lines:
        if ":-" in line:
            head_part, body_part = [part.strip() for part in line.rstrip(".").split(":-", maxsplit=1)]
            if not head_part:
                raise ValueError("Constraint rules are not supported.")
            head_atom = head_part
            body_literals = [
                literal.strip()
                for literal in body_part.split(",")
                if literal.strip()
            ]
        else:
            head_atom = line.rstrip(".")
            body_literals = []

        _register_atom(head_atom)
        for literal in body_literals:
            atom = literal[4:] if literal.startswith("not ") else literal
            _register_atom(atom)

        rules.append((head_atom, body_literals))

    return rules, atoms_order


def load_program_from_text(program_text: str, clingo_cmd: Sequence[str]) -> ProgramGraph:
    rules, atoms = parse_program_text(program_text)
    stable_models = get_stable_models(clingo_cmd, program_text=program_text)
    if not stable_models:
        raise ValueError("Provided program is UNSAT under clingo evaluation.")
    rendered_text = "\n".join(
        f"{head} :- {', '.join(body_literals)}." if body_literals else f"{head}."
        for head, body_literals in rules
    )
    program_graph = program_to_graph(
        program_text=rendered_text,
        atoms=atoms,
        stable_models=stable_models,
    )
    program_graph.program_text = program_text.strip()
    return program_graph


def load_program_from_file(file_path: str, clingo_cmd: Sequence[str]) -> ProgramGraph:
    with open(file_path, "r", encoding="utf-8") as f:
        program_text = f.read()

    program_graph = load_program_from_text(program_text, clingo_cmd)
    program_graph.source_path = str(Path(file_path).resolve())
    return program_graph


def load_program_from_aspif(file_path: str, clingo_cmd: Sequence[str], atom_list: Sequence[str], compute_stable_models: bool = True) -> ProgramGraph:
    with open(file_path, "r", encoding="utf-8") as f:
        aspif_text = f.read()

    symbol_map: dict[int, str] = {}
    atom_order: List[str] = []
    atom_to_index: dict[str, int] = {}

    def register(atom: str) -> int:
        if atom not in atom_to_index:
            atom_to_index[atom] = len(atom_order)
            atom_order.append(atom)
        return atom_to_index[atom]

    edge_pos: List[List[int]] = [[], []]
    edge_neg: List[List[int]] = [[], []]
    head_edge: List[List[int]] = [[], []]
    fact_atom_indices: set[int] = set()
    fact_rule_flags: List[bool] = []

    unassigned_list: List[str] = []
    for raw_line in aspif_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("asp") or line == "0":
            continue
        parts = line.split()
        kind = parts[0]
        if kind == "4":
            if len(parts) < 4:
                continue
            idx = 1
            try:
                _byte_len = int(parts[idx])
            except ValueError:
                continue
            idx += 1
            if idx >= len(parts):
                continue
            symbol = parts[idx]
            idx += 1
            if symbol.startswith('"') and symbol.endswith('"'):
                symbol = symbol[1:-1]
            if idx >= len(parts):
                continue
            try:
                num_literals = int(parts[idx])
            except ValueError:
                continue
            idx += 1
            literals: List[int] = []
            for _ in range(num_literals):
                if idx >= len(parts):
                    break
                literals.append(int(parts[idx]))
                idx += 1
            for lit in literals:
                if lit > 0 and lit not in symbol_map:
                    symbol_map[lit] = symbol
                    if symbol in atom_list:
                        atom_list.remove(symbol)
            if not literals:
                unassigned_list.append(symbol)

    for raw_line in aspif_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("asp") or line == "0":
            continue
        parts = line.split()
        kind = parts[0]
        if kind == "1":
            line_map = list(map(int, parts))
            if len(parts) != 6 or line_map[1] != 0 or line_map[2] != 1 or line_map[4] != 0 or line_map[5] != 0:
                continue
            atom_id = line_map[3]
            if atom_id not in symbol_map:
                atom_name = unassigned_list.pop(0) if unassigned_list else f"z{atom_id}"
                symbol_map[atom_id] = atom_name
                if atom_name in atom_list:
                    atom_list.remove(atom_name)
                if not compute_stable_models:
                    print(f"Warning: fact atom id {atom_id} not in symbol map, using {atom_name}.")

    for item in unassigned_list:
        if item in atom_list:
            atom_list.remove(item)

    for raw_line in aspif_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("asp") or line == "0":
            continue
        parts = line.split()
        kind = parts[0]
        if kind == "1":
            idx = 1
            if len(parts) < idx + 3:
                continue
            try:
                head_type = int(parts[idx])
                num_head_atoms = int(parts[idx + 1])
            except ValueError:
                continue
            idx += 2
            head_ids: List[int] = []
            for _ in range(num_head_atoms):
                if idx >= len(parts):
                    break
                head_ids.append(int(parts[idx]))
                idx += 1
            idx += 1
            if idx >= len(parts):
                continue
            try:
                num_body_literals = int(parts[idx])
            except ValueError:
                continue
            idx += 1
            body_lits: List[int] = []
            for _ in range(num_body_literals):
                if idx >= len(parts):
                    break
                body_lits.append(int(parts[idx]))
                idx += 1
            if not head_ids:
                raise ValueError("Constraint rules are not supported in ASPIF loader.")
            rule_idx = len(fact_rule_flags)
            for hid in head_ids:
                if abs(hid) in symbol_map:
                    head_name = symbol_map[abs(hid)]
                else:
                    head_name = unassigned_list.pop(0) if unassigned_list else f"z{abs(hid)}"
                    symbol_map[abs(hid)] = head_name
                    if head_name in atom_list:
                        atom_list.remove(head_name)
                    if not compute_stable_models:
                        print(f"Warning: head atom id {abs(hid)} not in symbol map, using {head_name}.")
                head_idx = register(head_name)
                for lit in body_lits:
                    atom_id = abs(lit)
                    if atom_id in symbol_map:
                        atom_name = symbol_map[atom_id]
                    else:
                        atom_name = atom_list.pop(0) if atom_list else f"z{atom_id}"
                        symbol_map[atom_id] = atom_name
                        if not compute_stable_models:
                            print(f"Warning: body atom id {atom_id} not in symbol map, using {atom_name}.")
                    atom_idx = register(atom_name)
                    if lit < 0:
                        edge_neg[0].append(atom_idx)
                        edge_neg[1].append(rule_idx)
                    else:
                        edge_pos[0].append(atom_idx)
                        edge_pos[1].append(rule_idx)
                head_edge[0].append(rule_idx)
                head_edge[1].append(head_idx)
                if not body_lits:
                    fact_atom_indices.add(head_idx)
            fact_rule_flags.append(len(body_lits) == 0)

    if not atom_order:
        raise ValueError("No atoms found in ASPIF file.")

    for atom in symbol_map.values():
        if atom not in atom_to_index:
            register(atom)

    if compute_stable_models:
        stable_models = get_stable_models(clingo_cmd, file_path=file_path, aspif=True)
        if not stable_models:
            raise ValueError("ASPIF program is UNSAT under clingo evaluation.")
    graph = Data()
    graph.edge_pos = (
        torch.tensor(edge_pos, dtype=torch.long)
        if edge_pos[0]
        else torch.empty((2, 0), dtype=torch.long)
    )
    graph.edge_neg = (
        torch.tensor(edge_neg, dtype=torch.long)
        if edge_neg[0]
        else torch.empty((2, 0), dtype=torch.long)
    )
    graph.head_edge = (
        torch.tensor(head_edge, dtype=torch.long)
        if head_edge[0]
        else torch.empty((2, 0), dtype=torch.long)
    )
    graph.p0 = torch.zeros(len(atom_order), dtype=torch.float32)
    if fact_atom_indices:
        fact_indices_tensor = torch.tensor(sorted(fact_atom_indices), dtype=torch.long)
        graph.p0[fact_indices_tensor] = 1.0
    if compute_stable_models:
        primary_model = stable_models[0]
        graph.y = atoms_to_tensor(atom_order, primary_model)
    graph.num_atoms = torch.tensor(len(atom_order), dtype=torch.long)
    graph.num_rules = torch.tensor(len(fact_rule_flags), dtype=torch.long)
    graph.fact_mask = (
        torch.tensor(fact_rule_flags, dtype=torch.bool)
        if fact_rule_flags
        else torch.empty(0, dtype=torch.bool)
    )
    graph.fact_values = (
        torch.tensor(
            [1.0 if is_fact else 0.0 for is_fact in fact_rule_flags],
            dtype=torch.float32,
        )
        if fact_rule_flags
        else torch.empty(0, dtype=torch.float32)
    )
    if compute_stable_models:
        founded_atoms = set(atom_order)
        for model in stable_models:
            founded_atoms &= set(model)
        graph.founded_mask = torch.tensor(
            [1.0 if atom in founded_atoms else 0.0 for atom in atom_order],
            dtype=torch.float32,
        )
    else:
        graph.founded_mask = torch.zeros(len(atom_order), dtype=torch.float32)
    if fact_rule_flags and all(fact_rule_flags):
        LOGGER.warning("ASPIF file %s contains only facts (no non-fact rules).", file_path)
        for raw_line in aspif_text.splitlines():
            LOGGER.warning("    %s", raw_line)
    
    if compute_stable_models:
        stable_target_tensors = [atoms_to_tensor(atom_order, model) for model in stable_models]
        graph.stable_targets = torch.stack(stable_target_tensors, dim=0)
    if compute_stable_models:
        return ProgramGraph(
            graph=graph,
            atom_names=atom_order,
            program_text=aspif_text.strip(),
            stable_model=primary_model,
            stable_models=[list(model) for model in stable_models],
            founded_atoms=sorted(founded_atoms),
        )
    else:
        return ProgramGraph(
            graph=graph,
            atom_names=atom_order,
            program_text=aspif_text.strip(),
            stable_model=[],
            stable_models=[],
            founded_atoms=[],
        )


def ground_program_to_aspif(file_path: str, clingo_cmd: Sequence[str], compute_stable_models: bool = True) -> ProgramGraph:
    with tempfile.NamedTemporaryFile(suffix=".aspif", delete=False, mode="w", encoding="utf-8") as tmp:
        temp_path = tmp.name
    try:
        # Extract all atom names that appear in the original program file
        with open(file_path, "r", encoding="utf-8") as f:
            original_program = f.read()
        _, atom_list = parse_program_text(original_program)
        
        cmd = list(clingo_cmd) + ["--mode=gringo", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode not in (0, 10, 20, 30):
            LOGGER.error("Clingo stderr:\n%s", result.stderr.strip())
            raise RuntimeError("Failed to ground program with clingo.")
        with open(temp_path, "w", encoding="utf-8") as out_file:
            out_file.write(result.stdout)
        program = load_program_from_aspif(temp_path, clingo_cmd, atom_list, compute_stable_models=compute_stable_models)
        program.program_text = Path(file_path).read_text(encoding="utf-8").strip()
        program.grounded_aspif = result.stdout.strip()
        return program
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def generate_dataset(
    generator: NormalProgramGenerator,
    num_instances: int,
    clingo_cmd: Sequence[str],
    max_attempts: int = 1000,
    aspif: bool = False,
    require_non_fact: bool = True,
    min_stable_models: int = 1,
    handcrafted_programs: Optional[Sequence[ProgramGraph]] = None,
) -> List[ProgramGraph]:
    if min_stable_models < 1:
        raise ValueError(f"min_stable_models must be at least 1 (got {min_stable_models}).")
    handcrafted_programs = list(handcrafted_programs or [])

    def _validate_handcrafted(program: ProgramGraph, index: int) -> None:
        stable_model_count = len(getattr(program, "stable_models", []))
        if stable_model_count < min_stable_models:
            origin = getattr(program, "source_path", None) or getattr(program, "program_text", "<inline>")
            raise ValueError(
                f"Handcrafted program #{index} ({origin}) has only {stable_model_count} stable models "
                f"(requires at least {min_stable_models})."
            )
        if require_non_fact:
            has_body_literals = program.graph.edge_pos.numel() > 0 or program.graph.edge_neg.numel() > 0
            if not has_body_literals:
                origin = getattr(program, "source_path", None) or getattr(program, "program_text", "<inline>")
                raise ValueError(
                    f"Handcrafted program #{index} ({origin}) contains only facts, "
                    "but require_non_fact=True."
                )

    dataset: List[ProgramGraph] = []
    attempts = 0
    target_count = max(num_instances, len(handcrafted_programs))
    with tqdm(total=target_count, desc="Generating programs") as pbar:
        class _ProgList(list):
            def append(self, item):
                super().append(item)
                pbar.update(1)

        dataset = _ProgList(dataset)
        if handcrafted_programs:
            for idx, program in enumerate(handcrafted_programs, start=1):
                _validate_handcrafted(program, idx)
                dataset.append(program)

        while len(dataset) < num_instances:
            program_text, atoms = generator.sample()
            attempts += 1

            if aspif:
                with tempfile.NamedTemporaryFile(
                    suffix=".lp", delete=False, mode="w", encoding="utf-8"
                ) as tmp_file:
                    tmp_file.write(program_text)
                    tmp_path = tmp_file.name
                try:
                    try:
                        program = ground_program_to_aspif(tmp_path, clingo_cmd)
                    except ValueError:
                        if attempts >= max_attempts:
                            raise RuntimeError(
                                "Unable to generate satisfiable program within attempt budget."
                            )
                        continue
                finally:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                stable_model_count = len(getattr(program, "stable_models", []))
                if stable_model_count < min_stable_models:
                    if attempts >= max_attempts:
                        raise RuntimeError(
                            "Unable to generate program meeting minimum stable-model requirement within attempt budget."
                        )
                    continue
            else:
                stable_models = get_stable_models(clingo_cmd, program_text=program_text)
                if len(stable_models) < min_stable_models:
                    if attempts >= max_attempts:
                        raise RuntimeError(
                            "Unable to generate program meeting minimum stable-model requirement within attempt budget."
                        )
                    continue
                program = program_to_graph(program_text, atoms, stable_models)

            if require_non_fact:
                has_body_literals = program.graph.edge_pos.numel() > 0 or program.graph.edge_neg.numel() > 0
                if not has_body_literals:
                    if attempts >= max_attempts:
                        raise RuntimeError(
                            "Unable to generate program with non-fact rules within attempt budget."
                        )
                    continue

            dataset.append(program)
            attempts = 0
    return list(dataset)


def format_answer_set(atom_names: Sequence[str], probs: torch.Tensor, threshold: float = 0.5) -> List[str]:
    selected = []
    for name, value in zip(atom_names, probs.tolist()):
        if value >= threshold:
            selected.append(name)
    return selected
