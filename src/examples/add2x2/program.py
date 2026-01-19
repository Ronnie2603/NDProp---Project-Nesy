from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch

from nesy.programs import ProgramSpec, build_program_spec, load_program_graph

DEFAULT_PROGRAM_PATH = Path(__file__).resolve().parent / "programs" / "add2x2.lp"


def _build_index_map(atom_names: List[str]) -> Dict[str, torch.Tensor]:
    name_to_index = {name: idx for idx, name in enumerate(atom_names)}

    def _collect(prefix: str, positions: range, values: range) -> Dict[str, torch.Tensor]:
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

    digit_map = _collect("digit", range(4), range(10))
    row_map = _collect("row", range(2), range(19))
    col_map = _collect("col", range(2), range(19))

    flat_map: Dict[str, torch.Tensor] = {}
    for pos, tensor in digit_map.items():
        flat_map[f"digit{pos}"] = tensor
    for pos, tensor in row_map.items():
        flat_map[f"row{pos}"] = tensor
    for pos, tensor in col_map.items():
        flat_map[f"col{pos}"] = tensor
    return flat_map


def build_program(
    program_path: Optional[Path] = None,
    clingo_path: Optional[str] = None,
) -> ProgramSpec:
    path = program_path if program_path is not None else DEFAULT_PROGRAM_PATH
    if not path.exists():
        raise FileNotFoundError(f"Program file '{path}' does not exist.")

    graph = load_program_graph(path, clingo_path)
    index_map = _build_index_map(graph.atom_names)
    input_atoms = {f"digit{pos}": index_map[f"digit{pos}"] for pos in range(4)}
    return build_program_spec(
        program_path=path,
        clingo_path=clingo_path,
        index_map=index_map,
        input_fact_atoms=input_atoms,
        program_graph=graph,
    )
