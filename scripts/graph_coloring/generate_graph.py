#!/usr/bin/env python3
"""Generate only SAT (k-colorable) graph instances, with fully-grounded ASP.

No third-party Python packages are required. Every graph is checked with an
exact backtracking coloring solver before being accepted. The ASP encoding uses
normal rules, with the odd-loop `bad` construction in place of explicit ASP
integrity constraints. It contains no variables, #const, or native constraints.

Example:
    python generate_graph_coloring.py --out graph_coloring_sat --seed 42

Each difficulty contains --train / --val / --test graphs; every instance is
SAT. Edge counts define buckets and are NOT a measured solver difficulty.
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import time
from collections import Counter
from pathlib import Path


def exact_k_coloring(n, edges, k=3):
    """Return (is_colorable, witness). UNSAT is exact after exhaustive search."""
    adj = [0] * n
    for u, v in edges:  # internal vertex ids are zero-based
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    degrees = [mask.bit_count() for mask in adj]
    colors = [-1] * n

    def solve(remaining, used):
        if remaining == 0:
            return True
        best_v = -1
        best_score = None
        best_options = []
        for v in range(n):
            if colors[v] != -1:
                continue
            forbidden = 0
            unassigned = 0
            neighbors = adj[v]
            while neighbors:
                bit = neighbors & -neighbors
                u = bit.bit_length() - 1
                if colors[u] != -1:
                    forbidden |= 1 << colors[u]
                else:
                    unassigned += 1
                neighbors ^= bit
            # Break global permutations of color names.
            options = [c for c in range(min(k, used + 1)) if not (forbidden & (1 << c))]
            if not options:
                return False
            score = (-len(options), unassigned, degrees[v])
            if best_score is None or score > best_score:
                best_v, best_score, best_options = v, score, options
        for color in best_options:
            colors[best_v] = color
            if solve(remaining - 1, max(used, color + 1)):
                return True
        colors[best_v] = -1
        return False

    sat = solve(n, 0)
    return sat, [c + 1 for c in colors] if sat else None


def grounded_normal_asp(n, edges, k):
    """Fully-grounded normal ASP; no `:- body.` constraints.

    An atom color(v,c) is supported only if the other k-1 colors for v are
    false.  The `bad :- ..., not bad.` odd loop rules out adjacent vertices
    having the same color in stable-model semantics.

    For every valid stable model, `bad` is false, so clingo's default output
    contains only color/2 atoms. No #show is needed: there are no other facts.
    """
    lines = [
        f"% Fully grounded {k}-coloring: {n} vertices, {len(edges)} edges",
        "% Exactly one color per vertex (normal rules)",
    ]
    for v in range(1, n + 1):
        for c in range(1, k + 1):
            others = [f"not color({v},{other})" for other in range(1, k + 1) if other != c]
            head = f"color({v},{c})"
            lines.append(f"{head} :- {', '.join(others)}." if others else f"{head}.")
    lines.extend(["", "% Adjacent vertices cannot share a color (normal-rule odd loop)"])
    for u, v in edges:
        for c in range(1, k + 1):
            lines.append(f"bad :- color({u + 1},{c}), color({v + 1},{c}), not bad.")
    return "\n".join(lines) + "\n"


def dimacs_col(n, edges):
    return "\n".join(["c Graph coloring instance", f"p edge {n} {len(edges)}"] +
                     [f"e {u + 1} {v + 1}" for u, v in edges]) + "\n"


def parse_ranges(text, n):
    groups = {}
    for segment in text.split(","):
        try:
            name, interval = segment.strip().split("=")
            lo, hi = map(int, interval.split(":"))
        except ValueError as exc:
            raise ValueError(f"Invalid range {segment!r}, expected easy=38:41") from exc
        if not name or name in groups or not (0 <= lo <= hi <= n * (n - 1) // 2):
            raise ValueError(f"Invalid range {segment!r}")
        groups[name] = list(range(lo, hi + 1))
    if not groups:
        raise ValueError("At least one range must be specified")
    return groups


def quotas(count, edge_counts):
    """Spread exactly count SAT instances evenly across the edge counts."""
    counts = [count // len(edge_counts) + (i < count % len(edge_counts))
              for i in range(len(edge_counts))]
    return [m for m, samples in zip(edge_counts, counts) for _ in range(samples)]


def generate(args):
    if args.n < 1 or args.k < 1 or any(x < 0 for x in (args.train, args.val, args.test)):
        raise ValueError("n and k must be positive; split sizes must be nonnegative")
    if args.k > args.n:
        raise ValueError("k cannot be greater than n")
    groups = parse_ranges(args.ranges, args.n)
    rng = random.Random(args.seed)
    possible_edges = [(u, v) for u in range(args.n) for v in range(u + 1, args.n)]
    root = Path(args.out)
    if root.exists() and any(root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output folder {root} is not empty; choose another --out or use --overwrite")
        # Delete ONLY the directories and metadata generated by this program.
        for split in ("train", "val", "test"):
            folder = root / split
            if folder.is_dir():
                shutil.rmtree(folder)
        for name in ("labels.csv", "config.json"):
            (root / name).unlink(missing_ok=True)
    root.mkdir(parents=True, exist_ok=True)

    records = []
    fingerprints = set()
    split_sizes = {"train": args.train, "val": args.val, "test": args.test}
    for difficulty, edge_counts in groups.items():
        for split, count in split_sizes.items():
            jobs = quotas(count, edge_counts)
            rng.shuffle(jobs)
            successes = Counter()
            attempts = Counter()
            for m in jobs:
                while True:
                    attempts[m] += 1
                    if attempts[m] > args.max_attempts_per_cell:
                        raise RuntimeError(
                            f"Failed to collect {count} SAT graphs for {difficulty}/{split}; "
                            f"n={args.n}, m={m}, k={args.k}. Try lowering the edge "
                            "range or increasing --max-attempts-per-cell."
                        )
                    edges = tuple(sorted(rng.sample(possible_edges, m)))
                    digest = hashlib.sha256(f"{args.n}:{args.k}:{edges}".encode()).hexdigest()
                    if digest in fingerprints:
                        continue
                    start = time.perf_counter()
                    sat, witness = exact_k_coloring(args.n, edges, args.k)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    if sat:
                        break  # UNSAT graphs are ALWAYS discarded.
                assert witness is not None and all(1 <= c <= args.k for c in witness)
                assert all(witness[u] != witness[v] for u, v in edges)
                fingerprints.add(digest)
                successes[m] += 1
                stem = f"{difficulty}_{split}_{m:04d}_sat_{successes[m]:04d}"
                folder = root / split / difficulty
                folder.mkdir(parents=True, exist_ok=True)
                col = folder / f"{stem}.col"
                lp = folder / f"{stem}.lp"
                col.write_text(dimacs_col(args.n, edges), encoding="utf-8")
                lp.write_text(grounded_normal_asp(args.n, edges, args.k), encoding="utf-8")
                records.append({
                    "id": stem,
                    "split": split,
                    "difficulty": difficulty,
                    "n_vertices": args.n,
                    "n_edges": m,
                    "k_colors": args.k,
                    "density": f"{m / len(possible_edges):.6f}" if possible_edges else "0.000000",
                    "colorable": 1,
                    "solver_ms": f"{elapsed_ms:.4f}",
                    "witness": " ".join(map(str, witness)),
                    "col_file": str(col.relative_to(root)),
                    "asp_file": str(lp.relative_to(root)),
                    "sha256": digest,
                })
            print(f"{difficulty:>6} / {split:<5}: {count} SAT (0 UNSAT)", flush=True)

    columns = ["id", "split", "difficulty", "n_vertices", "n_edges", "k_colors",
               "density", "colorable", "solver_ms", "witness", "col_file", "asp_file", "sha256"]
    with (root / "labels.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "seed": args.seed, "n": args.n, "k": args.k, "ranges": args.ranges,
        "splits_per_difficulty": split_sizes, "sat_only": True,
        "exact_solver": "pure Python DSATUR-style backtracking",
        "asp_encoding": "fully grounded normal rules; odd-loop bad, no integrity constraints",
        "difficulty_note": "Edge-count tiers are not certified computational difficulty",
    }
    (root / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {len(records)} SAT instances to {root.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="graph_coloring_sat_grounded", help="Output directory")
    parser.add_argument("--n", type=int, default=22, help="Vertices per graph")
    parser.add_argument("--k", type=int, default=3, help="Colors per graph")
    parser.add_argument("--ranges", default="easy=38:41,medium=42:45,hard=46:49",
                        help="Edge ranges per difficulty")
    parser.add_argument("--train", type=int, default=1000, help="Train graphs PER difficulty")
    parser.add_argument("--val", type=int, default=100, help="Validation graphs PER difficulty")
    parser.add_argument("--test", type=int, default=100, help="Test graphs PER difficulty")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-attempts-per-cell", type=int, default=50000,
                        help="Maximum attempts per edge count and split")
    parser.add_argument("--overwrite", action="store_true", help="Replace this script's outputs")
    generate(parser.parse_args())


if __name__ == "__main__":
    main()

