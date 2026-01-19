from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import torch

from nesy.programs import ProgramSpec, build_program_spec, load_program_graph

DEFAULT_PROGRAM_PATH = Path(__file__).resolve().parent / "programs" / "apply2x2.lp"
MIN_RESULT = -100
MAX_RESULT = 1000

_ROW_RE = re.compile(r"row\((\d+),\s*(-?\d+)\)")
_COL_RE = re.compile(r"col\((\d+),\s*(-?\d+)\)")
_DIGITS_RE = re.compile(r"digits\((\d+),\s*(\d+),\s*(\d+)\)")


def _extract_row_col_values(atom_names: List[str]) -> tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    row_values: Dict[int, set[int]] = {}
    col_values: Dict[int, set[int]] = {}
    for name in atom_names:
        row_match = _ROW_RE.fullmatch(name)
        if row_match:
            pos = int(row_match.group(1))
            val = int(row_match.group(2))
            row_values.setdefault(pos, set()).add(val)
            continue
        col_match = _COL_RE.fullmatch(name)
        if col_match:
            pos = int(col_match.group(1))
            val = int(col_match.group(2))
            col_values.setdefault(pos, set()).add(val)
    row_lists = {pos: sorted(vals) for pos, vals in row_values.items()}
    col_lists = {pos: sorted(vals) for pos, vals in col_values.items()}
    return row_lists, col_lists


def _extract_digit_triples(atom_names: List[str]) -> List[Tuple[int, int, int]]:
    triples: List[Tuple[int, int, int]] = []
    for name in atom_names:
        match = _DIGITS_RE.fullmatch(name)
        if not match:
            continue
        triples.append((int(match.group(1)), int(match.group(2)), int(match.group(3))))
    return triples


def _build_index_map(atom_names: List[str]) -> Dict[str, torch.Tensor]:
    name_to_index = {name: idx for idx, name in enumerate(atom_names)}

    def _collect(prefix: str, positions: range, values: List[int]) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        for pos in positions:
            indices: List[int] = []
            for value in values:
                atom = f"{prefix}({pos},{value})"
                if atom not in name_to_index:
                    raise ValueError(f"Atom '{atom}' not found in grounded program.")
                indices.append(name_to_index[atom])
            result[str(pos)] = torch.tensor(indices, dtype=torch.long)
        return result

    digit_triples = _extract_digit_triples(atom_names)
    if not digit_triples:
        raise ValueError("Failed to extract digits/3 atoms from grounded program.")
    digit_indices: List[int] = []
    for d1, d2, d3 in digit_triples:
        atom = f"digits({d1},{d2},{d3})"
        if atom not in name_to_index:
            raise ValueError(f"Atom '{atom}' not found in grounded program.")
        digit_indices.append(name_to_index[atom])
    op_map = _collect("op", range(4), list(range(3)))
    row_values, col_values = _extract_row_col_values(atom_names)
    if len(row_values) < 2 or len(col_values) < 2 or 0 not in row_values or 1 not in row_values or 0 not in col_values or 1 not in col_values:
        raise ValueError("Failed to extract row/col value domains from grounded program.")
    row_map: Dict[str, torch.Tensor] = {}
    col_map: Dict[str, torch.Tensor] = {}
    for pos in range(2):
        row_map.update(_collect("row", range(pos, pos + 1), row_values[pos]))
        col_map.update(_collect("col", range(pos, pos + 1), col_values[pos]))

    flat_map: Dict[str, torch.Tensor] = {}
    flat_map["digits"] = torch.tensor(digit_indices, dtype=torch.long)
    for pos, tensor in op_map.items():
        flat_map[f"op{pos}"] = tensor
    for pos, tensor in row_map.items():
        flat_map[f"row{pos}"] = tensor
    for pos, tensor in col_map.items():
        flat_map[f"col{pos}"] = tensor
    return flat_map


def extract_value_maps(atom_names: List[str]) -> tuple[List[int], List[int]]:
    row_values, col_values = _extract_row_col_values(atom_names)
    if 0 not in row_values or 1 not in row_values or 0 not in col_values or 1 not in col_values:
        raise ValueError("Missing row/col value domains in grounded program.")
    if row_values[0] != row_values[1] or col_values[0] != col_values[1]:
        raise ValueError("Row/col value domains differ between positions.")
    return row_values[0], col_values[0]


def extract_digit_triples(atom_names: List[str]) -> List[Tuple[int, int, int]]:
    triples = _extract_digit_triples(atom_names)
    if not triples:
        raise ValueError("Missing digits/3 atoms in grounded program.")
    return triples


def build_program(
    program_path: Optional[Path] = None,
    clingo_path: Optional[str] = None,
) -> ProgramSpec:
    path = program_path if program_path is not None else DEFAULT_PROGRAM_PATH
    if not path.exists():
        raise FileNotFoundError(f"Program file '{path}' does not exist.")

    graph = load_program_graph(path, clingo_path)
    index_map = _build_index_map(graph.atom_names)
    input_atoms = {"digits": index_map["digits"]}
    input_atoms.update({f"op{pos}": index_map[f"op{pos}"] for pos in range(4)})
    return build_program_spec(
        program_path=path,
        clingo_path=clingo_path,
        index_map=index_map,
        input_fact_atoms=input_atoms,
        program_graph=graph,
    )
