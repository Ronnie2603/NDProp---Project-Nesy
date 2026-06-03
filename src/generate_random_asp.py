#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


FAMILY_DEFAULTS = {
    "n2l": {"max_body": 2, "neg_prob": 1.0, "rules_per_atom": 2.0},
    "3lp": {"max_body": 3, "neg_prob": 0.5, "rules_per_atom": 3.0},
}

DIFFICULTY_ATOMS = {
    "easy": (8, 12),
    "medium": (13, 18),
    "hard": (19, 25),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate random ASP benchmark splits for NDProp.")
    parser.add_argument("--family", choices=sorted(FAMILY_DEFAULTS), default="n2l", help="Program family to generate.")
    parser.add_argument("--output-root", type=Path, default=None, help="Output root; defaults to benchmarks/<family> next to the repo.")
    parser.add_argument("--train-size", type=int, default=128, help="Number of train programs.")
    parser.add_argument("--test-size", type=int, default=64, help="Number of programs per difficulty test split.")
    parser.add_argument("--seed", type=int, default=7, help="Base RNG seed.")
    parser.add_argument("--max-attempts", type=int, default=10000, help="Attempt budget per split.")
    parser.add_argument("--min-stable-models", type=int, default=1, help="Minimum stable models per generated program.")
    parser.add_argument("--clingo-path", type=str, default=None, help="clingo executable.")
    return parser


def _rules_range(atoms_range: tuple[int, int], rules_per_atom: float) -> tuple[int, int]:
    return (max(1, int(atoms_range[0] * rules_per_atom)), max(1, int(atoms_range[1] * rules_per_atom)))


def _write_split(
    *,
    out_dir: Path,
    family: str,
    atoms_range: tuple[int, int],
    size: int,
    seed: int,
    clingo_cmd: list[str],
    min_stable_models: int,
    max_attempts: int,
) -> None:
    from fuzzy_tp import NormalProgramGenerator, get_stable_models

    defaults = FAMILY_DEFAULTS[family]
    generator = NormalProgramGenerator(
        atoms_range=atoms_range,
        rules_range=_rules_range(atoms_range, defaults["rules_per_atom"]),
        max_body=defaults["max_body"],
        negation_prob=defaults["neg_prob"],
        fact_rate=0.1,
        seed=seed,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    attempts = 0
    while written < size:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Could not generate {size} satisfiable programs for {out_dir}.")
        program_text, _ = generator.sample()
        stable_models = get_stable_models(clingo_cmd, program_text=program_text)
        if len(stable_models) < min_stable_models:
            continue
        (out_dir / f"program_{written:05d}.lp").write_text(program_text + "\n", encoding="utf-8")
        written += 1
    print(f"Wrote {written} programs to {out_dir}")


def main() -> None:
    args = build_parser().parse_args()
    from fuzzy_tp import resolve_clingo_command

    output_root = args.output_root or (PROJECT_ROOT.parent / "benchmarks" / args.family)
    clingo_cmd = resolve_clingo_command(args.clingo_path)

    _write_split(
        out_dir=output_root / "train",
        family=args.family,
        atoms_range=DIFFICULTY_ATOMS["medium"],
        size=args.train_size,
        seed=args.seed,
        clingo_cmd=clingo_cmd,
        min_stable_models=args.min_stable_models,
        max_attempts=args.max_attempts,
    )
    for offset, (difficulty, atoms_range) in enumerate(DIFFICULTY_ATOMS.items(), start=1):
        _write_split(
            out_dir=output_root / f"test_{difficulty}",
            family=args.family,
            atoms_range=atoms_range,
            size=args.test_size,
            seed=args.seed + offset,
            clingo_cmd=clingo_cmd,
            min_stable_models=args.min_stable_models,
            max_attempts=args.max_attempts,
        )


if __name__ == "__main__":
    main()
