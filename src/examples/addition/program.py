from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch

from nesy.programs import ProgramSpec, build_program_spec, load_program_graph

DEFAULT_PROGRAM_PATH = Path(__file__).resolve().parent / "programs" / "addition.lp"


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

    digit_map = {f"digit{pos}": _digit_indices(pos) for pos in range(2)}

    sum_indices: List[int] = []
    for value in range(19):
        atom = f"sum({value})"
        if atom not in name_to_index:
            raise ValueError(f"Atom '{atom}' not found in grounded program.")
        sum_indices.append(name_to_index[atom])
    digit_map["sum"] = torch.tensor(sum_indices, dtype=torch.long)
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
    input_atoms = {f"digit{pos}": index_map[f"digit{pos}"] for pos in range(2)}
    return build_program_spec(
        program_path=path,
        clingo_path=clingo_path,
        index_map=index_map,
        input_fact_atoms=input_atoms,
        program_graph=graph,
    )
