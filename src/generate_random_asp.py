#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


FAMILIES = ("n2l", "3lp")
FAMILY_OUTPUT_ROOTS = {
    "n2l": PROJECT_ROOT / "benchmarks" / "n2l",
    "3lp": PROJECT_ROOT / "benchmarks" / "3lp",
}


@dataclass(frozen=True)
class N2LSplit:
    name: str
    total: int
    seed: int
    n_range: Tuple[int, int]
    c1_range: Tuple[float, float]


@dataclass(frozen=True)
class KlpSplit:
    name: str
    total: int
    seed: int
    n_range: Tuple[int, int]


N2L_SPLITS = (
    N2LSplit("train_easy", 1000, 2025, (5, 10), (2.5, 4.0)),
    N2LSplit("train_medium", 1000, 3025, (11, 50), (5.0, 14.0)),
    N2LSplit("train_hard", 1000, 4025, (51, 100), (20.0, 34.0)),
    N2LSplit("val_easy", 100, 999, (5, 10), (2.5, 4.0)),
    N2LSplit("val_medium", 100, 1999, (11, 50), (5.0, 14.0)),
    N2LSplit("val_hard", 100, 2999, (51, 100), (20.0, 34.0)),
    N2LSplit("test_easy", 100, 3101, (5, 10), (2.5, 4.0)),
    N2LSplit("test_medium", 100, 6101, (11, 50), (5.0, 14.0)),
    N2LSplit("test_hard", 100, 7205, (51, 100), (20.0, 34.0)),
)

KLP_SPLITS = (
    KlpSplit("train_easy", 1000, 2025, (5, 10)),
    KlpSplit("val_easy", 100, 3025, (5, 10)),
    KlpSplit("test_easy", 100, 4025, (5, 10)),
    KlpSplit("train_medium", 1000, 5025, (11, 50)),
    KlpSplit("val_medium", 100, 6025, (11, 50)),
    KlpSplit("test_medium", 100, 7025, (11, 50)),
    KlpSplit("train_hard", 1000, 8025, (51, 100)),
    KlpSplit("val_hard", 100, 9025, (51, 100)),
    KlpSplit("test_hard", 100, 10025, (51, 100)),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate random ASP benchmark splits for NDProp.")
    parser.add_argument("--family", choices=list(FAMILIES) + ["both"], default="both", help="Program family to generate.")
    parser.add_argument("--output-root", type=Path, default=None, help="Output root; only valid with one family.")
    parser.add_argument("--clingo-path", type=str, default=None, help="clingo executable.")
    parser.add_argument("--min-stable-models", type=int, default=1, help="Minimum stable models per program.")
    parser.add_argument("--max-attempts", type=int, default=200000, help="Attempt budget per split.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .lp files and metadata.")
    return parser


def hashes_for_rules(rules: Sequence[str]) -> str:
    canonical = "\n".join(sorted(rules)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def resolve_cmd(explicit_path: Optional[str]) -> List[str]:
    from ndprop import resolve_clingo_command

    return resolve_clingo_command(explicit_path)


def run_clingo_json(clingo_cmd: Sequence[str], program_text: str, minimum: int, timeout: float = 10.0) -> bool:
    cmd = list(clingo_cmd) + ["--outf=2", "-n", str(minimum), "-"]
    result = subprocess.run(cmd, input=program_text, text=True, capture_output=True, check=False, timeout=timeout)
    if result.returncode not in (0, 10, 20, 30):
        raise RuntimeError(f"clingo failed with exit {result.returncode}: {result.stderr.strip()}")
    payload = json.loads(result.stdout or "{}")
    if payload.get("Result") != "SATISFIABLE":
        return False
    try:
        return int(payload.get("Models", {}).get("Number", 0)) >= minimum
    except (TypeError, ValueError):
        return False


def aspif_has_non_fact(clingo_cmd: Sequence[str], program_text: str, timeout: float = 10.0) -> bool:
    cmd = list(clingo_cmd) + ["--mode=gringo", "-"]
    result = subprocess.run(cmd, input=program_text, text=True, capture_output=True, check=False, timeout=timeout)
    if result.returncode not in (0, 10, 20, 30):
        raise RuntimeError(f"clingo ASPIF failed with exit {result.returncode}: {result.stderr.strip()}")
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("asp") or line == "0":
            continue
        parts = line.split()
        if parts[0] != "1":
            continue
        try:
            ints = [int(part) for part in parts[1:]]
        except ValueError:
            continue
        if len(ints) < 3:
            continue
        head_len = ints[1]
        idx = 2 + head_len
        if idx >= len(ints) or ints[idx] != 0:
            continue
        idx += 1
        if idx < len(ints) and ints[idx] > 0:
            return True
    return False


def prepare_split_dir(split_dir: Path, expected_total: int, overwrite: bool) -> bool:
    """Prepare a split directory and report whether it should be generated.

    Existing splits are only reused when both the expected number of programs
    and their metadata are present. This avoids silently training on an
    interrupted or otherwise incomplete generation run.
    """
    existing = list(split_dir.glob("*.lp")) if split_dir.exists() else []
    if existing and not overwrite:
        metadata_path = split_dir / "metadata.csv"
        metadata_rows = -1
        if metadata_path.is_file():
            with metadata_path.open(newline="", encoding="utf-8") as metadata_file:
                metadata_rows = sum(1 for _ in csv.DictReader(metadata_file))
        if len(existing) == expected_total and metadata_rows == expected_total:
            print(f"Split already exists at {split_dir}; skipping ({len(existing)} .lp files).")
            return False
        raise RuntimeError(
            f"Split '{split_dir}' is incomplete: found {len(existing)} of "
            f"{expected_total} .lp files and {max(metadata_rows, 0)} of "
            f"{expected_total} metadata rows. "
            "Rerun with --overwrite to regenerate it."
        )
    split_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in split_dir.glob("*.lp"):
            path.unlink()
        metadata = split_dir / "metadata.csv"
        if metadata.exists():
            metadata.unlink()
    return True


def sample_n2l_program(n: int, c1: float, rng: random.Random) -> List[str]:
    atoms = [f"a{i + 1}" for i in range(n)]
    p = c1 / n
    rules: List[str] = []
    for i, head in enumerate(atoms):
        for j, body in enumerate(atoms):
            if i == j:
                continue
            if rng.random() < p:
                rules.append(f"{head} :- not {body}.")
    return rules


def render_klp_rule(head_idx: int, pos: Sequence[int], neg: Sequence[int]) -> str:
    head = f"a{head_idx + 1}"
    body_parts = [f"a{idx + 1}" for idx in pos] + [f"not a{idx + 1}" for idx in neg]
    return f"{head} :- {', '.join(body_parts)}." if body_parts else f"{head}."


def sample_3lp_program(n: int, rng: random.Random, density: float = 5.0, k: int = 3) -> List[str]:
    atoms = list(range(n))
    target_rules = max(1, int(round(density * n)))
    rules: List[str] = []
    used: set[tuple[int, tuple[int, ...], tuple[int, ...]]] = set()
    attempts = 0
    while len(rules) < target_rules:
        attempts += 1
        if attempts > 100000:
            raise RuntimeError(f"Exceeded retries while sampling 3-LP rules (n={n}).")
        head = rng.randrange(n)
        body_atoms = rng.sample(atoms, k - 1)
        pos: List[int] = []
        neg: List[int] = []
        for atom in body_atoms:
            if rng.random() < 0.5:
                neg.append(atom)
            else:
                pos.append(atom)
        canonical = (head, tuple(sorted(pos)), tuple(sorted(neg)))
        if canonical in used:
            continue
        used.add(canonical)
        rules.append(render_klp_rule(head, sorted(pos), sorted(neg)))
    return rules


def write_rows(split_dir: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    with (split_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def generate_split(
    *,
    split_dir: Path,
    total: int,
    seed: int,
    sample: Callable[[random.Random], Tuple[List[str], Dict[str, str]]],
    clingo_cmd: Sequence[str],
    min_stable_models: int,
    max_attempts: int,
    overwrite: bool,
    fieldnames: Sequence[str],
) -> None:
    if not prepare_split_dir(split_dir, total, overwrite):
        return
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    used_hashes: set[str] = set()
    attempts = 0
    while len(rows) < total:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Exceeded attempts for split '{split_dir.name}'.")
        rules, row = sample(rng)
        if not rules:
            continue
        rule_hash = hashes_for_rules(rules)
        if rule_hash in used_hashes:
            continue
        program_text = "\n".join(rules) + "\n"
        if not run_clingo_json(clingo_cmd, program_text, min_stable_models):
            continue
        if not aspif_has_non_fact(clingo_cmd, program_text):
            continue
        file_name = f"{split_dir.name}_{len(rows):05d}.lp"
        (split_dir / file_name).write_text(program_text, encoding="ascii")
        used_hashes.add(rule_hash)
        rows.append({**row, "file": file_name, "hash": rule_hash, "seed": str(seed)})
    write_rows(split_dir, rows, fieldnames)
    print(f"[{split_dir.name}] wrote {len(rows)} programs")


def generate_n2l(output_root: Path, clingo_cmd: Sequence[str], min_stable_models: int, max_attempts: int, overwrite: bool) -> None:
    for cfg in N2L_SPLITS:
        def sample(rng: random.Random, cfg: N2LSplit = cfg) -> Tuple[List[str], Dict[str, str]]:
            n_val = rng.randint(*cfg.n_range)
            c1_val = float(f"{rng.uniform(*cfg.c1_range):.6f}")
            rules = sample_n2l_program(n_val, c1_val, rng)
            return rules, {
                "n": str(n_val),
                "c1": f"{c1_val:.6f}",
                "c2": "0.000000",
                "label": "consistent",
                "rule_count": str(len(rules)),
            }

        generate_split(
            split_dir=output_root / cfg.name,
            total=cfg.total,
            seed=cfg.seed,
            sample=sample,
            clingo_cmd=clingo_cmd,
            min_stable_models=min_stable_models,
            max_attempts=max_attempts,
            overwrite=overwrite,
            fieldnames=("file", "n", "c1", "c2", "label", "rule_count", "hash", "seed"),
        )


def generate_3lp(output_root: Path, clingo_cmd: Sequence[str], min_stable_models: int, max_attempts: int, overwrite: bool) -> None:
    for cfg in KLP_SPLITS:
        def sample(rng: random.Random, cfg: KlpSplit = cfg) -> Tuple[List[str], Dict[str, str]]:
            n_val = rng.randint(*cfg.n_range)
            rules = sample_3lp_program(n_val, rng)
            return rules, {
                "n": str(n_val),
                "k": "3",
                "l": str(len(rules)),
                "density": f"{(len(rules) / n_val):.6f}",
                "label": "consistent",
            }

        generate_split(
            split_dir=output_root / cfg.name,
            total=cfg.total,
            seed=cfg.seed,
            sample=sample,
            clingo_cmd=clingo_cmd,
            min_stable_models=min_stable_models,
            max_attempts=max_attempts,
            overwrite=overwrite,
            fieldnames=("file", "n", "k", "l", "density", "label", "hash", "seed"),
        )


def main() -> None:
    args = build_parser().parse_args()
    families = list(FAMILIES) if args.family == "both" else [args.family]
    if args.output_root is not None and len(families) > 1:
        raise ValueError("--output-root can only be used when --family is n2l or 3lp.")
    clingo_cmd = resolve_cmd(args.clingo_path)

    for family in families:
        output_root = args.output_root or FAMILY_OUTPUT_ROOTS[family]
        if family == "n2l":
            generate_n2l(output_root, clingo_cmd, args.min_stable_models, args.max_attempts, args.overwrite)
        elif family == "3lp":
            generate_3lp(output_root, clingo_cmd, args.min_stable_models, args.max_attempts, args.overwrite)
        else:
            raise ValueError(f"Unknown family: {family}")


if __name__ == "__main__":
    main()
