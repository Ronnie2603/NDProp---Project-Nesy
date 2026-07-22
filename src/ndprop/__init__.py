"""Shared components for NDProp over ASP program graphs."""

from .model import NDProp
from .data import (
    ProgramGraph,
    collate_programs,
    load_program_from_file,
    load_program_from_text,
    resolve_clingo_command,
    format_answer_set,
    load_program_from_aspif,
    ground_program_to_aspif,
    get_stable_models,
)
from .training import train_epoch, evaluate

__all__ = [
    "NDProp",
    "ProgramGraph",
    "collate_programs",
    "load_program_from_file",
    "load_program_from_text",
    "resolve_clingo_command",
    "format_answer_set",
    "train_epoch",
    "evaluate",
    "load_program_from_aspif",
    "ground_program_to_aspif",
    "get_stable_models",
]
