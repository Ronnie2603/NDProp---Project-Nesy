from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch

from nesy.programs import ProgramSpec, build_program_spec, load_program_graph

from .datasets import AWA_BASE

DEFAULT_PROGRAM_PATH = Path(__file__).resolve().parent / "programs" / "awa.lp"
DEFAULT_DATA_DIR = AWA_BASE


def _read_classes(path: Path) -> List[str]:
    class_names: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            name = parts[-1].replace("+", "_")
            class_names.append(name)
    return class_names


def build_program(
    *,
    classes_file: Optional[Path] = None,
    program_path: Optional[Path] = None,
    clingo_path: Optional[str] = None,
) -> ProgramSpec:
    print("Building AwA program specification...")
    data_dir = classes_file.parent if classes_file is not None else DEFAULT_DATA_DIR
    classes_path = classes_file if classes_file is not None else data_dir / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError(f"classes file '{classes_path}' not found.")

    pred_path = data_dir / "predicates.txt"
    if not pred_path.exists():
        raise FileNotFoundError(f"predicates file '{pred_path}' not found.")
    predicates = _read_classes(pred_path)

    print(f"Using program file: {program_path}")
    program_path = program_path if program_path is not None else DEFAULT_PROGRAM_PATH
    try:
        graph = load_program_graph(program_path, clingo_path)
        fallback = False
    except Exception as e:
        print(f"Warning: Failed to load grounded program graph: {e}")
        graph = None
        fallback = True
    print(f"Loaded program graph: {graph is not None}")

    name_to_idx = {name: idx for idx, name in enumerate(graph.atom_names)} if graph is not None else {}
    attr_indices: List[int] = []
    for name in predicates:
        atom = f"attr({name})"
        if atom in name_to_idx:
            attr_indices.append(name_to_idx[atom])
    if not attr_indices and not fallback:
        raise ValueError("No attribute atoms (has/1) from predicates.txt were found in the AwA program.")
    print(f"Found {len(attr_indices)} attribute atoms from predicates.txt.")
    class_names = _read_classes(classes_path)
    class_indices: List[int] = []
    for cls in class_names:
        atom = f"class({cls})"
        if atom in name_to_idx:
            class_indices.append(name_to_idx[atom])
    if not class_indices and not fallback:
        raise ValueError("No class atoms (sel/1) from classes.txt were found in the AwA program.")

    print(f"Found {len(class_indices)} class atoms from classes.txt.")
    index_map: Dict[str, torch.Tensor] = {
        "attr": torch.tensor(attr_indices, dtype=torch.long),
        "class": torch.tensor(class_indices, dtype=torch.long),
    }
    input_atoms = {"has": index_map["attr"]}
    print(f"Loaded AwA program with {len(attr_indices)} attribute atoms and {len(class_indices)} class atoms.")

    if not fallback:
        return build_program_spec(
            program_path=program_path,
            clingo_path=clingo_path,
            index_map=index_map,
            input_fact_atoms=input_atoms,
            program_graph=graph,
        )

    print("Warning: Falling back to minimal AwA program specification without grounding.")
    matrix_path = data_dir / "predicate-matrix-binary.txt"
    if not matrix_path.exists():
        raise FileNotFoundError(f"predicate-matrix-binary.txt not found at {matrix_path}")
    import numpy as np

    mat = np.loadtxt(matrix_path, dtype=np.int64)
    if mat.ndim != 2 or mat.shape[1] != len(class_names):
        mat = mat.T
    num_attrs = len(predicates)
    num_classes = len(class_names)
    if mat.shape != (num_attrs, num_classes):
        raise ValueError(f"Predicate matrix shape {mat.shape} does not match attrs/classes ({num_attrs}, {num_classes})")

    atom_names = [f"has({name})" for name in predicates] + [f"sel({name})" for name in class_names]
    attr_indices = torch.arange(num_attrs, dtype=torch.long)
    class_indices = torch.arange(num_classes, dtype=torch.long) + num_attrs

    edge_pos: list[list[int]] = [[], []]
    head_edge: list[list[int]] = [[], []]
    rule_idx = 0
    attr_fact_rules = torch.arange(rule_idx, rule_idx + num_attrs, dtype=torch.long)
    for a_idx in range(num_attrs):
        head_edge[0].append(attr_fact_rules[a_idx].item())
        head_edge[1].append(attr_indices[a_idx].item())
    rule_idx += num_attrs

    for a_idx in range(num_attrs):
        for c_idx in range(num_classes):
            if mat[a_idx, c_idx] > 0:
                edge_pos[0].append(attr_indices[a_idx].item())
                edge_pos[1].append(rule_idx)
                head_edge[0].append(rule_idx)
                head_edge[1].append(class_indices[c_idx].item())
                rule_idx += 1
    if rule_idx == 0:
        raise ValueError("No rules constructed from predicate matrix.")

    edge_pos_tensor = torch.tensor(edge_pos, dtype=torch.long) if edge_pos[0] else torch.empty(2, 0, dtype=torch.long)
    edge_neg_tensor = torch.empty(2, 0, dtype=torch.long)
    head_edge_tensor = torch.tensor(head_edge, dtype=torch.long) if head_edge[0] else torch.empty(2, 0, dtype=torch.long)
    fact_mask = torch.zeros(rule_idx, dtype=torch.bool)
    fact_mask[attr_fact_rules] = True
    fact_values = torch.zeros(rule_idx, dtype=torch.float32)
    founded_mask = torch.zeros(len(atom_names), dtype=torch.float32)
    input_fact_rules = {"has": attr_fact_rules}

    return ProgramSpec(
        atom_names=atom_names,
        edge_pos=edge_pos_tensor,
        edge_neg=edge_neg_tensor,
        head_edge=head_edge_tensor,
        index_map={"attr": attr_indices, "class": class_indices},
        num_atoms=len(atom_names),
        num_rules=rule_idx,
        fact_mask=fact_mask,
        fact_values=fact_values,
        input_fact_rules=input_fact_rules,
        founded_mask=founded_mask,
        program_text=program_path.read_text(encoding="utf-8") if program_path.exists() else "",
        grounded_aspif="",
    )
