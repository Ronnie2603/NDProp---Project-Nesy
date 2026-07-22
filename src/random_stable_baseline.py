from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from ndprop import get_stable_models, resolve_clingo_command
from ndprop.data import parse_program_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random baseline: sample atom sets and check if they are stable models."
    )
    parser.add_argument("--cases-dir", type=Path, required=True, help="Directory containing .lp programs.")
    parser.add_argument("--samples", type=int, default=100, help="Random candidate sets sampled per program.")
    parser.add_argument("--include-prob", type=float, default=0.5, help="Probability of including each atom.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--clingo-path", type=str, default=None, help="Optional clingo executable.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--timestamp", type=str, default=None, help="Optional timestamp.")
    return parser.parse_args()


def ensure_cases(cases_dir: Path) -> List[Path]:
    if not cases_dir.exists():
        raise FileNotFoundError(f"Cases directory '{cases_dir}' does not exist.")
    case_paths = sorted(path for path in cases_dir.glob("*.lp") if path.is_file())
    if not case_paths:
        raise FileNotFoundError(f"Cases directory '{cases_dir}' does not contain any .lp programs.")
    return case_paths


def sample_candidate(atoms: Sequence[str], rng: random.Random, include_prob: float) -> List[str]:
    return [atom for atom in atoms if rng.random() < include_prob]


def evaluate_program(
    program_path: Path,
    clingo_cmd: Sequence[str],
    rng: random.Random,
    samples: int,
    include_prob: float,
) -> Dict[str, object]:
    program_text = program_path.read_text(encoding="utf-8")
    _, atoms = parse_program_text(program_text)
    stable_models = get_stable_models(clingo_cmd, file_path=str(program_path))
    stable_set = {tuple(sorted(model)) for model in stable_models}

    hit_count = 0
    first_hit = None
    for idx in range(samples):
        candidate = sorted(sample_candidate(atoms, rng, include_prob))
        if tuple(candidate) in stable_set:
            hit_count += 1
            if first_hit is None:
                first_hit = idx

    return {
        "case": program_path.name,
        "atom_count": len(atoms),
        "stable_model_count": len(stable_models),
        "samples": samples,
        "include_prob": include_prob,
        "hit_count": hit_count,
        "success": hit_count > 0,
        "first_hit_index": first_hit,
    }


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be at least 1.")
    if not (0.0 <= args.include_prob <= 1.0):
        raise ValueError("--include-prob must be between 0 and 1.")

    cases = ensure_cases(args.cases_dir)
    clingo_cmd = resolve_clingo_command(args.clingo_path)
    rng = random.Random(args.seed)

    timestamp = args.timestamp or datetime.now().strftime("rand_baseline_%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (Path("baselines") / timestamp)
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest: List[Dict[str, object]] = []
    eval_summary: Dict[str, bool] = {}

    for case_path in cases:
        print(f"Evaluating {case_path.name}...")
        report = evaluate_program(case_path, clingo_cmd, rng, args.samples, args.include_prob)
        manifest.append(report)
        eval_summary[case_path.name] = bool(report["success"])

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "cases": manifest,
                "samples": args.samples,
                "include_prob": args.include_prob,
                "seed": args.seed,
            },
            handle,
            indent=2,
        )

    with (output_dir / "eval.json").open("w", encoding="utf-8") as handle:
        json.dump(eval_summary, handle, indent=2)

    successes = sum(1 for value in eval_summary.values() if value)
    total = len(eval_summary)
    accuracy = (successes / total * 100.0) if total else 0.0
    summary = {
        "successes": successes,
        "total": total,
        "accuracy": accuracy,
        "output_dir": str(output_dir.resolve()),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Random baseline results stored in {output_dir}")
    print(f"Accuracy: {successes}/{total} ({accuracy:.2f}%)")


if __name__ == "__main__":
    main()
