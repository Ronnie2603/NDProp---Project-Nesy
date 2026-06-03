import argparse
import json
import random
import re
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from fuzzy_tp import get_stable_models, resolve_clingo_command
from fuzzy_tp.data import parse_program_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DProp baseline: propagate with false decisions (no constraints)."
    )
    parser.add_argument(
        "--cases-dir",
        type=Path,
        required=True,
        help="Directory containing .lp programs to evaluate.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of randomized DProp runs per program.",
    )
    parser.add_argument(
        "--unique-trials",
        action="store_true",
        help="Require unique decision sequences per trial (resample on duplicates).",
    )
    parser.add_argument(
        "--unique-max-attempts",
        type=int,
        default=50,
        help="Max resampling attempts per trial when --unique-trials is set.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--clingo-path",
        type=str,
        default=None,
        help="Optional path to clingo executable. Falls back to PATH lookup.",
    )
    parser.add_argument(
        "--dlv-path",
        type=str,
        default=None,
        help="Optional DLV executable for WFS cross-checks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reports (defaults to baselines/<timestamp>).",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Optional timestamp override to reuse an output directory.",
    )
    return parser.parse_args()


def ensure_cases(cases_dir: Path) -> List[Path]:
    if not cases_dir.exists():
        raise FileNotFoundError(f"Cases directory '{cases_dir}' does not exist.")
    case_paths = sorted(path for path in cases_dir.glob("*.lp") if path.is_file())
    if not case_paths:
        raise FileNotFoundError(f"Cases directory '{cases_dir}' does not contain any .lp programs.")
    return case_paths


def _parse_rules(
    program_text: str, atoms: Sequence[str]
) -> Tuple[List[int], List[List[int]], List[List[int]]]:
    rules, _ = parse_program_text(program_text)
    index = {name: idx for idx, name in enumerate(atoms)}
    heads: List[int] = []
    pos: List[List[int]] = []
    neg: List[List[int]] = []
    for head_atom, body_literals in rules:
        heads.append(index[head_atom])
        pos_lits: List[int] = []
        neg_lits: List[int] = []
        for literal in body_literals:
            if literal.startswith("not "):
                neg_lits.append(index[literal[4:]])
            else:
                pos_lits.append(index[literal])
        pos.append(pos_lits)
        neg.append(neg_lits)
    return heads, pos, neg


def _build_occurrence_lists(
    atom_count: int,
    pos: Sequence[Sequence[int]],
    neg: Sequence[Sequence[int]],
) -> Tuple[List[List[int]], List[List[int]]]:
    pos_occ = [[] for _ in range(atom_count)]
    neg_occ = [[] for _ in range(atom_count)]
    for rule_idx, pos_lits in enumerate(pos):
        for atom in pos_lits:
            pos_occ[atom].append(rule_idx)
    for rule_idx, neg_lits in enumerate(neg):
        for atom in neg_lits:
            neg_occ[atom].append(rule_idx)
    return pos_occ, neg_occ


def _well_founded_initializer(
    atom_count: int,
    heads: Sequence[int],
    pos: Sequence[Sequence[int]],
    neg: Sequence[Sequence[int]],
) -> Tuple[List[bool], List[bool]]:
    all_atoms = set(range(atom_count))
    rule_count = len(heads)

    def _gl_reduct_least_model(assumed_true: set[int]) -> set[int]:
        active_rules = [
            idx
            for idx in range(rule_count)
            if not any(atom in assumed_true for atom in neg[idx])
        ]
        true_atoms: set[int] = set()
        changed = True
        while changed:
            changed = False
            for rule_idx in active_rules:
                head = heads[rule_idx]
                if head in true_atoms:
                    continue
                if all(atom in true_atoms for atom in pos[rule_idx]):
                    true_atoms.add(head)
                    changed = True
        return true_atoms

    lower: set[int] = set()
    upper: set[int] = set(all_atoms)
    while True:
        next_lower = _gl_reduct_least_model(upper)
        next_upper = _gl_reduct_least_model(next_lower)
        if next_lower == lower and next_upper == upper:
            break
        lower, upper = next_lower, next_upper

    truth = [idx in lower for idx in range(atom_count)]
    falsity = [idx not in upper for idx in range(atom_count)]
    return truth, falsity


def _parse_dlv_wfs(stdout: str) -> Tuple[set, set]:
    true_match = re.search(r"True:\s*{([^}]*)}", stdout)
    undef_match = re.search(r"Undefined:\s*{([^}]*)}", stdout)
    if not true_match or not undef_match:
        raise ValueError("DLV output did not contain True/Undefined sets.")

    def _parse_atoms(payload: str) -> set:
        payload = payload.strip()
        if not payload:
            return set()
        atoms = [atom.strip() for atom in payload.split(",") if atom.strip()]
        return set(atoms)

    return _parse_atoms(true_match.group(1)), _parse_atoms(undef_match.group(1))


def _check_wfs_with_dlv(
    program_path: Path,
    atoms: Sequence[str],
    heads: Sequence[int],
    pos: Sequence[Sequence[int]],
    neg: Sequence[Sequence[int]],
    dlv_path: Optional[str],
) -> None:
    if not dlv_path:
        print("Skipping DLV WFS cross-check because --dlv-path was not provided.")
        return
    truth, falsity = _well_founded_initializer(len(atoms), heads, pos, neg)
    true_atoms = {atom for atom, val in zip(atoms, truth) if val}
    false_atoms = {atom for atom, val in zip(atoms, falsity) if val}
    undefined_atoms = set(atoms) - true_atoms - false_atoms

    result = subprocess.run(
        [dlv_path, str(program_path), "--wellfounded"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"DLV failed for {program_path.name} (exit {result.returncode}): {result.stderr.strip()}"
        )
    dlv_true, dlv_undefined = _parse_dlv_wfs(result.stdout)

    if true_atoms != dlv_true or undefined_atoms != dlv_undefined:
        raise ValueError(
            "Well-founded model mismatch for "
            f"{program_path.name}. "
            f"Initializer true={sorted(true_atoms)} undefined={sorted(undefined_atoms)}; "
            f"DLV true={sorted(dlv_true)} undefined={sorted(dlv_undefined)}."
        )
    else:
        print(f"DLV WFS check passed for {program_path.name}.")


def _propagate(
    heads: Sequence[int],
    pos_occ: Sequence[Sequence[int]],
    neg_occ: Sequence[Sequence[int]],
    dead: List[bool],
    alive_support_count: List[int],
    need_pos: List[int],
    need_neg: List[int],
    fired: List[bool],
    truth: List[bool],
    falsity: List[bool],
    queue: deque,
    false_queue: deque,
    *,
    enable_false_propagation: bool,
) -> bool:
    while queue:
        rule_idx = queue.popleft()
        if fired[rule_idx] or dead[rule_idx]:
            continue
        fired[rule_idx] = True
        head = heads[rule_idx]
        if falsity[head]:
            return False
        if not truth[head]:
            truth[head] = True
            for r2 in pos_occ[head]:
                need_pos[r2] -= 1
                if need_pos[r2] == 0 and need_neg[r2] == 0 and not fired[r2]:
                    queue.append(r2)
            if enable_false_propagation:
                for r2 in neg_occ[head]:
                    if not dead[r2]:
                        dead[r2] = True
                        head_atom = heads[r2]
                        alive_support_count[head_atom] -= 1
                        if alive_support_count[head_atom] == 0:
                            false_queue.append(head_atom)
    return True


def _apply_false(
    atom: int,
    heads: Sequence[int],
    neg_occ: Sequence[Sequence[int]],
    pos_occ: Sequence[Sequence[int]],
    dead: List[bool],
    alive_support_count: List[int],
    need_pos: List[int],
    need_neg: List[int],
    fired: List[bool],
    truth: List[bool],
    falsity: List[bool],
    queue: deque,
    false_queue: deque,
) -> bool:
    if truth[atom]:
        return False
    if falsity[atom]:
        return True
    falsity[atom] = True
    for rule_idx in pos_occ[atom]:
        if not dead[rule_idx]:
            dead[rule_idx] = True
            head_atom = heads[rule_idx]
            alive_support_count[head_atom] -= 1
            if alive_support_count[head_atom] == 0:
                false_queue.append(head_atom)
    for rule_idx in neg_occ[atom]:
        need_neg[rule_idx] -= 1
        if need_neg[rule_idx] == 0 and need_pos[rule_idx] == 0 and not fired[rule_idx]:
            queue.append(rule_idx)
    return True


def _decide_truth_only(
    atom: int,
    neg_occ: Sequence[Sequence[int]],
    need_pos: List[int],
    need_neg: List[int],
    fired: List[bool],
    truth: List[bool],
    falsity: List[bool],
    queue: deque,
) -> bool:
    if truth[atom]:
        return False
    if falsity[atom]:
        return True
    falsity[atom] = True
    for rule_idx in neg_occ[atom]:
        need_neg[rule_idx] -= 1
        if need_neg[rule_idx] == 0 and need_pos[rule_idx] == 0 and not fired[rule_idx]:
            queue.append(rule_idx)
    return True


def _false_propagate(
    heads: Sequence[int],
    pos_occ: Sequence[Sequence[int]],
    neg_occ: Sequence[Sequence[int]],
    dead: List[bool],
    alive_support_count: List[int],
    need_pos: List[int],
    need_neg: List[int],
    fired: List[bool],
    truth: List[bool],
    falsity: List[bool],
    truth_queue: deque,
    false_queue: deque,
) -> bool:
    while false_queue:
        atom = false_queue.popleft()
        if not _apply_false(
            atom,
            heads,
            neg_occ,
            pos_occ,
            dead,
            alive_support_count,
            need_pos,
            need_neg,
            fired,
            truth,
            falsity,
            truth_queue,
            false_queue,
        ):
            return False
    return True


def dprop_run(
    atoms: Sequence[str],
    heads: Sequence[int],
    pos: Sequence[Sequence[int]],
    neg: Sequence[Sequence[int]],
    rng: random.Random,
) -> Dict[str, object]:
    atom_count = len(atoms)
    pos_occ, neg_occ = _build_occurrence_lists(atom_count, pos, neg)
    rules_by_head = [[] for _ in range(atom_count)]
    for rule_idx, head in enumerate(heads):
        rules_by_head[head].append(rule_idx)

    truth, falsity = _well_founded_initializer(atom_count, heads, pos, neg)
    fired = [False] * len(heads)
    dead = [False] * len(heads)
    alive_support_count = [len(rules_by_head[idx]) for idx in range(atom_count)]
    need_pos = [len([a for a in pos_lits if not truth[a]]) for pos_lits in pos]
    need_neg = [len([a for a in neg_lits if not falsity[a]]) for neg_lits in neg]

    queue: deque = deque()
    false_queue: deque = deque()

    decisions = 0
    decision_atoms: List[str] = []
    while True:
        undecided = [idx for idx in range(atom_count) if not truth[idx] and not falsity[idx]]
        if not undecided:
            break
        atom = rng.choice(undecided)
        decisions += 1
        decision_atoms.append(atoms[atom])
        if not _decide_truth_only(
            atom,
            neg_occ,
            need_pos,
            need_neg,
            fired,
            truth,
            falsity,
            queue,
        ):
            return {
                "success": False,
                "contradiction": True,
                "decisions": decisions,
                "decision_atoms": decision_atoms,
                "candidate": [],
            }
        while True:
            if not _propagate(
                heads,
                pos_occ,
                neg_occ,
                dead,
                alive_support_count,
                need_pos,
                need_neg,
                fired,
                truth,
                falsity,
                queue,
                false_queue,
                enable_false_propagation=False,
            ):
                return {
                    "success": False,
                    "contradiction": True,
                    "decisions": decisions,
                    "decision_atoms": decision_atoms,
                    "candidate": [],
                }
            if not queue:
                break

    candidate = [atoms[idx] for idx, value in enumerate(truth) if value]
    return {
        "success": True,
        "contradiction": False,
        "decisions": decisions,
        "decision_atoms": decision_atoms,
        "candidate": sorted(candidate),
    }


def evaluate_program(
    program_path: Path,
    clingo_cmd: Sequence[str],
    dlv_path: str,
    rng: random.Random,
    trials: int,
    unique_trials: bool,
    unique_max_attempts: int,
) -> Dict[str, object]:
    program_text = program_path.read_text(encoding="utf-8")
    _, atoms = parse_program_text(program_text)
    heads, pos, neg = _parse_rules(program_text, atoms)
    _check_wfs_with_dlv(program_path, atoms, heads, pos, neg, dlv_path)
    stable_models = get_stable_models(clingo_cmd, file_path=str(program_path))
    stable_set = {tuple(sorted(model)) for model in stable_models}

    hit_count = 0
    first_hit = None
    first_candidate: Optional[List[str]] = None
    contradictions = 0
    decision_counts: List[int] = []
    seen_sequences = set()

    for idx in range(trials):
        attempts = 0
        while True:
            run_report = dprop_run(atoms, heads, pos, neg, rng)
            seq_key = tuple(run_report["decision_atoms"])
            if not unique_trials or seq_key not in seen_sequences:
                seen_sequences.add(seq_key)
                break
            attempts += 1
            if attempts >= unique_max_attempts:
                break
        decision_counts.append(int(run_report["decisions"]))
        if run_report["contradiction"]:
            contradictions += 1
            continue
        candidate = run_report["candidate"]
        if run_report["success"] and tuple(candidate) not in stable_set:
            print(
                f"Warning: DProp success but non-stable candidate for {program_path.name}: {candidate}"
            )
            truth, falsity = _well_founded_initializer(atoms, stable_models)
            print(f"Well-founded atoms: {[atoms[idx] for idx, val in enumerate(truth) if val]}")
            print(f"Unfounded atoms: {[atoms[idx] for idx, val in enumerate(falsity) if val]}")
        if tuple(candidate) in stable_set:
            hit_count += 1
            if first_hit is None:
                first_hit = idx
                first_candidate = candidate

    return {
        "case": program_path.name,
        "atom_count": len(atoms),
        "stable_model_count": len(stable_models),
        "trials": trials,
        "unique_trials": unique_trials,
        "hit_count": hit_count,
        "success": hit_count > 0,
        "first_hit_index": first_hit,
        "first_candidate": first_candidate,
        "contradictions": contradictions,
        "avg_decisions": sum(decision_counts) / len(decision_counts) if decision_counts else 0.0,
    }


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be at least 1.")
    if args.unique_max_attempts < 1:
        raise ValueError("--unique-max-attempts must be at least 1.")

    cases = ensure_cases(args.cases_dir)
    clingo_cmd = resolve_clingo_command(args.clingo_path)
    rng = random.Random(args.seed)

    timestamp = args.timestamp or datetime.now().strftime("dprop_baseline_%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (Path("baselines") / timestamp)
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest: List[Dict[str, object]] = []
    eval_summary: Dict[str, bool] = {}

    for case_path in cases:
        print(f"Evaluating {case_path.name}...")
        report = evaluate_program(
            case_path,
            clingo_cmd,
            args.dlv_path,
            rng,
            args.trials,
            args.unique_trials,
            args.unique_max_attempts,
        )
        manifest.append(report)
        eval_summary[case_path.name] = bool(report["success"])

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "cases": manifest,
                "trials": args.trials,
                "unique_trials": args.unique_trials,
                "unique_max_attempts": args.unique_max_attempts,
                "seed": args.seed,
            },
            handle,
            indent=2,
        )

    eval_path = output_dir / "eval.json"
    with eval_path.open("w", encoding="utf-8") as handle:
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
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"DProp baseline results stored in {output_dir}")
    print(f"Accuracy: {successes}/{total} ({accuracy:.2f}%)")


if __name__ == "__main__":
    main()
