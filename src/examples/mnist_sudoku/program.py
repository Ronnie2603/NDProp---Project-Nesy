from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from nesy.programs import ProgramSpec, build_program_spec, load_program_graph

DEFAULT_PROGRAM_PATH = Path(__file__).resolve().parent / "programs" / "mnist_sudoku.lp"


def _infer_dims(atom_names: List[str]) -> Tuple[int, int]:
    max_rc = 0
    max_v = 0
    for name in atom_names:
        if not name.startswith("digit(") or not name.endswith(")"):
            continue
        body = name[len("digit(") : -1]
        parts = body.split(",")
        if len(parts) != 3:
            continue
        try:
            r = int(parts[0])
            c = int(parts[1])
            v = int(parts[2])
        except ValueError:
            continue
        max_rc = max(max_rc, r, c)
        max_v = max(max_v, v)
    if max_rc == 0 or max_v == 0:
        raise ValueError("No digit(r,c,v) atoms found in grounded program.")
    return max_rc, max_v


def _digit_atoms(side: int, num_vals: int) -> List[str]:
    atoms: List[str] = []
    for r in range(1, side + 1):
        for c in range(1, side + 1):
            for v in range(1, num_vals + 1):
                atoms.append(f"digit({r},{c},{v})")
    return atoms


def build_program(
    program_path: Optional[Path] = None,
    clingo_path: Optional[str] = None,
) -> ProgramSpec:
    path = program_path if program_path is not None else DEFAULT_PROGRAM_PATH
    if not path.exists():
        raise FileNotFoundError(f"Program file '{path}' does not exist.")

    graph = load_program_graph(path, clingo_path)
    name_to_index = {name: idx for idx, name in enumerate(graph.atom_names)}
    side, num_vals = _infer_dims(graph.atom_names)
    digit_atoms = _digit_atoms(side, num_vals)
    indices: List[int] = []
    for atom in digit_atoms:
        if atom not in name_to_index:
            raise ValueError(f"Atom '{atom}' not found in grounded program.")
        indices.append(name_to_index[atom])

    idx_tensor = torch.tensor(indices, dtype=torch.long).view(side * side, num_vals)
    index_map: Dict[str, torch.Tensor] = {"digit": idx_tensor}
    bad_indices = [idx for name, idx in name_to_index.items() if name.startswith("bad")]
    if bad_indices:
        index_map["bad"] = torch.tensor(bad_indices, dtype=torch.long)
    input_atoms = {"digit": idx_tensor.view(-1)}
    return build_program_spec(
        program_path=path,
        clingo_path=clingo_path,
        index_map=index_map,
        input_fact_atoms=input_atoms,
        program_graph=graph,
    )
