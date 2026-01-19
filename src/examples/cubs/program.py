from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from nesy.programs import ProgramSpec, build_program_spec, load_program_graph

from .datasets import CUB_BASE

DEFAULT_PROGRAM_PATH = Path(__file__).resolve().parent / "programs" / "cubs.lp"
DEFAULT_DATA_DIR = CUB_BASE


_NON_ALNUM = re.compile(r"[^a-zA-Z0-9_]")


def _normalize_class_name(name: str) -> str:
    cleaned = _NON_ALNUM.sub("_", name.strip()).lower()
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"c_{cleaned}"
    return cleaned


def _read_classes(path: Path) -> List[str]:
    class_names: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            class_names.append(_normalize_class_name(parts[-1]))
    return class_names


def _attribute_names(num_attrs: int) -> List[str]:
    return [f"attr_{idx + 1:03d}" for idx in range(num_attrs)]


def _num_attributes(data_dir: Path) -> int:
    matrix_path = data_dir / "attributes" / "class_attribute_labels_continuous.txt"
    if not matrix_path.exists():
        raise FileNotFoundError(f"class_attribute_labels_continuous.txt not found at {matrix_path}")
    mat = np.loadtxt(matrix_path, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"Attribute matrix at {matrix_path} must be 2D, got shape {mat.shape}.")
    return int(mat.shape[1])


def build_program(
    *,
    classes_file: Optional[Path] = None,
    program_path: Optional[Path] = None,
    clingo_path: Optional[str] = None,
) -> ProgramSpec:
    print("Building CUBS program specification...")
    data_dir = classes_file.parent if classes_file is not None else DEFAULT_DATA_DIR
    classes_path = classes_file if classes_file is not None else data_dir / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError(f"classes file '{classes_path}' not found.")

    class_names = _read_classes(classes_path)
    attr_names = _attribute_names(_num_attributes(data_dir))

    program_path = program_path if program_path is not None else DEFAULT_PROGRAM_PATH
    print(f"Using program file: {program_path}")
    graph = load_program_graph(program_path, clingo_path)

    name_to_idx = {name: idx for idx, name in enumerate(graph.atom_names)}
    attr_indices: List[int] = []
    for name in attr_names:
        atom = f"attr({name})"
        if atom in name_to_idx:
            attr_indices.append(name_to_idx[atom])
    if not attr_indices:
        raise ValueError("No attribute atoms found in the CUBS program.")

    class_indices: List[int] = []
    for cls in class_names:
        atom = f"class({cls})"
        if atom in name_to_idx:
            class_indices.append(name_to_idx[atom])
    if not class_indices:
        raise ValueError("No class atoms found in the CUBS program.")

    index_map: Dict[str, torch.Tensor] = {
        "attr": torch.tensor(attr_indices, dtype=torch.long),
        "class": torch.tensor(class_indices, dtype=torch.long),
    }
    input_atoms = {"has": index_map["attr"]}
    return build_program_spec(
        program_path=program_path,
        clingo_path=clingo_path,
        index_map=index_map,
        input_fact_atoms=input_atoms,
        program_graph=graph,
    )
