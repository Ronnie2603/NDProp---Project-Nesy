from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Data


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


def _strip_asp_comment(line: str) -> str:
    return line.split("%", maxsplit=1)[0].strip()


def parse_program_text(program_text: str) -> Tuple[List[Tuple[str, List[str]]], List[str]]:
    statements: List[str] = []
    current_parts: List[str] = []
    for stripped in (
        stripped
        for raw_line in program_text.splitlines()
        if (stripped := _strip_asp_comment(raw_line))
    ):
        current_parts.append(stripped)
        if stripped.endswith("."):
            statement = " ".join(current_parts).strip()
            if statement:
                statements.append(statement)
            current_parts = []
    if current_parts:
        raise ValueError("Program contains an unterminated rule without a final '.'.")

    atoms_order: List[str] = []
    atom_set = set()
    rules: List[Tuple[str, List[str]]] = []

    def _register_atom(atom: str):
        if atom not in atom_set:
            atom_set.add(atom)
            atoms_order.append(atom)

    for statement in statements:
        rule_text = statement.rstrip(".").strip()
        if ":-" in rule_text:
            head_part, body_part = [part.strip() for part in rule_text.split(":-", maxsplit=1)]
            if not head_part:
                raise ValueError("Constraint rules are not supported.")
            head_atom = head_part
            body_literals = [
                literal.strip()
                for literal in body_part.split(",")
                if literal.strip()
            ]
        else:
            head_atom = rule_text
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


def load_program_from_aspif(
    file_path: str,
    clingo_cmd: Sequence[str],
    atom_list: Optional[Sequence[str]] = None,
    compute_stable_models: bool = True,
) -> ProgramGraph:
    with open(file_path, "r", encoding="utf-8") as f:
        aspif_text = f.read()

    available_atoms = list(atom_list or [])
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
                int(parts[idx])
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
                    if symbol in available_atoms:
                        available_atoms.remove(symbol)
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
                if atom_name in available_atoms:
                    available_atoms.remove(atom_name)
                if not compute_stable_models:
                    LOGGER.warning("Fact atom id %s not in symbol map; using %s.", atom_id, atom_name)

    for item in unassigned_list:
        if item in available_atoms:
            available_atoms.remove(item)

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
                int(parts[idx])
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
                    if head_name in available_atoms:
                        available_atoms.remove(head_name)
                    if not compute_stable_models:
                        LOGGER.warning("Head atom id %s not in symbol map; using %s.", abs(hid), head_name)
                head_idx = register(head_name)
                for lit in body_lits:
                    atom_id = abs(lit)
                    if atom_id in symbol_map:
                        atom_name = symbol_map[atom_id]
                    else:
                        atom_name = available_atoms.pop(0) if available_atoms else f"z{atom_id}"
                        symbol_map[atom_id] = atom_name
                        if not compute_stable_models:
                            LOGGER.warning("Body atom id %s not in symbol map; using %s.", atom_id, atom_name)
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

    for atom in symbol_map.values():
        if atom not in atom_to_index:
            register(atom)

    for atom in available_atoms:
        if atom not in atom_to_index:
            register(atom)

    if not atom_order:
        raise ValueError("No atoms found in ASPIF file.")

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


def sanitize_aspif_for_ndprop(
    aspif_text: str,
    atom_names: Optional[Sequence[str]] = None,
) -> str:
    """
    Reduce grounded ASPIF to the normal rules supported by NDProp.

    Neural task programs commonly use exact-choice rules only to declare their
    neural input atoms. Clingo expands those choices into choice heads,
    weighted bodies, integrity constraints, and anonymous support atoms.
    NDProp injects the input atoms itself and supports only normal rules, so the
    expanded choice machinery must not enter its program graph.

    When source atom names are available, only their output records are used
    as graph roots; named outputs introduced solely during grounding are not.
    """
    lines = [line.strip() for line in aspif_text.splitlines() if line.strip()]
    source_atoms = set(atom_names) if atom_names is not None else None
    normal_rules: list[tuple[str, list[int], list[int]]] = []
    visible_atoms: set[int] = set()
    used_atoms: set[int] = set()
    external_atoms: set[int] = set()
    unconditional_outputs: list[tuple[str, str]] = []
    other_records: list[str] = []

    for line in lines:
        if line.startswith("asp"):
            continue
        if line == "0":
            continue
        parts = line.split()
        if parts[0] == "4":
            # Output statement: 4 <name-length> <name> <n-literals> <literals>.
            name = parts[2].strip('"') if len(parts) > 2 else ""
            if source_atoms is not None and name not in source_atoms:
                continue
            try:
                literal_count = int(parts[3])
                literals = [int(value) for value in parts[4 : 4 + literal_count]]
            except (ValueError, IndexError):
                other_records.append(line)
                continue
            output_atoms = {abs(literal) for literal in literals if literal}
            visible_atoms.update(output_atoms)
            used_atoms.update(output_atoms)
            if literal_count == 0:
                unconditional_outputs.append((parts[1], parts[2]))
            else:
                other_records.append(line)
            continue
        if parts[0] != "1":
            # Preserve directives the graph loader does not need but which make
            # the generated file a faithful, valid ASPIF artifact.
            if parts[0] == "5":  # External statement: 5 <atom> <value>.
                try:
                    external_atom = abs(int(parts[1]))
                    external_atoms.add(external_atom)
                    used_atoms.add(external_atom)
                except (ValueError, IndexError):
                    pass
            other_records.append(line)
            continue

        try:
            cursor = 1
            head_type = int(parts[cursor])
            cursor += 1
            head_count = int(parts[cursor])
            cursor += 1
            heads = [int(value) for value in parts[cursor : cursor + head_count]]
            cursor += head_count
            body_type = int(parts[cursor])
            cursor += 1
            body_count = int(parts[cursor])
            cursor += 1
            body = [int(value) for value in parts[cursor : cursor + body_count]]
        except (ValueError, IndexError):
            continue

        used_atoms.update(abs(atom) for atom in heads if atom)
        used_atoms.update(abs(literal) for literal in body if literal)

        # The ASPIF loader represents normal, non-constraint rules only.
        if head_type == 0 and heads and body_type == 0:
            normal_rules.append((line, heads, body))

    # Clingo omits atom ids for named outputs that it proves unconditionally
    # true. Give each such output an explicit fact so it survives reachability
    # filtering and can be represented in the NDProp graph.
    next_atom = 1
    for name_length, name in unconditional_outputs:
        while next_atom in used_atoms:
            next_atom += 1
        atom_id = next_atom
        used_atoms.add(atom_id)
        visible_atoms.add(atom_id)
        normal_rules.append((f"1 0 1 {atom_id} 0 0", [atom_id], []))
        other_records.append(f"4 {name_length} {name} 1 {atom_id}")

    # Grounding can leave anonymous atoms that are fixed constants. Substitute
    # them out of named rules so they cannot become spurious graph atoms.
    auxiliary_atoms = used_atoms.difference(visible_atoms)
    defined_auxiliary = {
        head for _, heads, _ in normal_rules for head in heads if head in auxiliary_atoms
    }
    true_auxiliary = {
        heads[0]
        for _, heads, body in normal_rules
        if len(heads) == 1 and not body and heads[0] in auxiliary_atoms
    }
    false_auxiliary = auxiliary_atoms.difference(defined_auxiliary, external_atoms)
    simplified_rules: list[tuple[str, list[int], list[int]]] = []
    for line, heads, body in normal_rules:
        if any(
            (literal > 0 and abs(literal) in false_auxiliary)
            or (literal < 0 and abs(literal) in true_auxiliary)
            for literal in body
        ):
            continue
        simplified_body = [
            literal
            for literal in body
            if not (
                (literal > 0 and abs(literal) in true_auxiliary)
                or (literal < 0 and abs(literal) in false_auxiliary)
            )
        ]
        if simplified_body != body:
            line = " ".join(
                map(str, [1, 0, len(heads), *heads, 0, len(simplified_body), *simplified_body])
            )
        simplified_rules.append((line, heads, simplified_body))
    normal_rules = simplified_rules

    # Drop anonymous facts and rules used exclusively by removed choice
    # encodings. Work backwards from named/output atoms to retain all normal
    # rules that can contribute to the task's observable atoms.
    reachable = set(visible_atoms)
    changed = True
    while changed:
        changed = False
        for _, heads, body in normal_rules:
            if any(head in reachable for head in heads):
                dependencies = {abs(literal) for literal in body if literal}
                new_atoms = dependencies.difference(reachable)
                if new_atoms:
                    reachable.update(new_atoms)
                    changed = True

    kept_rules = [
        line for line, heads, _ in normal_rules if any(head in reachable for head in heads)
    ]
    return "\n".join(["asp 1 0 0", *kept_rules, *other_records, "0"])


def ground_program_to_aspif(file_path: str, clingo_cmd: Sequence[str], compute_stable_models: bool = True) -> ProgramGraph:
    with tempfile.NamedTemporaryFile(suffix=".aspif", delete=False, mode="w", encoding="utf-8") as tmp:
        temp_path = tmp.name
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            original_program = f.read()
        _, atom_list = parse_program_text(original_program)
        
        cmd = list(clingo_cmd) + ["--mode=gringo", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode not in (0, 10, 20, 30):
            LOGGER.error("Clingo stderr:\n%s", result.stderr.strip())
            raise RuntimeError("Failed to ground program with clingo.")
        sanitized_aspif = sanitize_aspif_for_ndprop(result.stdout, atom_list)
        with open(temp_path, "w", encoding="utf-8") as out_file:
            out_file.write(sanitized_aspif)
            out_file.write("\n")
        program = load_program_from_aspif(temp_path, clingo_cmd, atom_list, compute_stable_models=compute_stable_models)
        program.program_text = Path(file_path).read_text(encoding="utf-8").strip()
        program.grounded_aspif = sanitized_aspif
        return program
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def format_answer_set(atom_names: Sequence[str], probs: torch.Tensor, threshold: float = 0.5) -> List[str]:
    selected = []
    for name, value in zip(atom_names, probs.tolist()):
        if value >= threshold:
            selected.append(name)
    return selected
