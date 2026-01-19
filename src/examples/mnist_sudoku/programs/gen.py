from __future__ import annotations

import argparse
from pathlib import Path


def _prototype_rules(lines: list[str], base: int) -> None:
    side = base * base
    bads = 0
    for r in range(1, side + 1):
        for c in range(1, side + 1):
            for v in range(1, side + 1):
                negs: list[str] = []
                # Same cell, different value.
                negs.extend(
                    f"not digit({r},{c},{v2})" for v2 in range(1, side + 1) if v2 != v
                )
                # Same row, same value.
                negs.extend(
                    f"not digit({r},{c2},{v})" for c2 in range(1, side + 1) if c2 != c
                )
                # Same column, same value.
                negs.extend(
                    f"not digit({r2},{c},{v})" for r2 in range(1, side + 1) if r2 != r
                )
                # Same block, same value.
                br = (r - 1) // base
                bc = (c - 1) // base
                for r2 in range(br * base + 1, br * base + base + 1):
                    for c2 in range(bc * base + 1, bc * base + base + 1):
                        if r2 == r and c2 == c:
                            continue
                        negs.append(f"not digit({r2},{c2},{v})")
                lines.append(f"digit({r},{c},{v}) :- " + ", ".join(negs) + ".")
            negs = [f"not digit({r},{c},{v2})" for v2 in range(1, side + 1)]
            lines.append(f"bad_{bads} :- " + ", ".join(negs) + f", not bad_{bads}.")
            bads += 1


def _row_bad(lines: list[str], side: int) -> None:
    for r in range(1, side + 1):
        for v in range(1, side + 1):
            for c1 in range(1, side + 1):
                for c2 in range(c1 + 1, side + 1):
                    lines.append(f"bad :- digit({r},{c1},{v}), digit({r},{c2},{v}), not bad.")


def _col_bad(lines: list[str], side: int) -> None:
    for c in range(1, side + 1):
        for v in range(1, side + 1):
            for r1 in range(1, side + 1):
                for r2 in range(r1 + 1, side + 1):
                    lines.append(f"bad :- digit({r1},{c},{v}), digit({r2},{c},{v}), not bad.")


def _block_bad(lines: list[str], base: int) -> None:
    side = base * base
    for br in range(0, base):
        for bc in range(0, base):
            cells = [(br * base + r, bc * base + c) for r in range(1, base + 1) for c in range(1, base + 1)]
            for v in range(1, side + 1):
                for i in range(len(cells)):
                    r1, c1 = cells[i]
                    for j in range(i + 1, len(cells)):
                        r2, c2 = cells[j]
                        lines.append(f"bad :- digit({r1},{c1},{v}), digit({r2},{c2},{v}), not bad.")


def generate(program_path: Path, *, base: int) -> None:
    side = base * base
    lines: list[str] = []
    lines.append(f"% MNIST Sudoku rules (normal rules only) for {side}x{side}")
    lines.append("")
    _prototype_rules(lines, base)
    # lines.append("")
    # _row_bad(lines, side)
    # lines.append("")
    # _col_bad(lines, side)
    # lines.append("")
    # _block_bad(lines, base)
    program_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=int, default=3, help="Base size (2 for 4x4, 3 for 9x9).")
    parser.add_argument("--out", type=Path, default=None, help="Output program path.")
    args = parser.parse_args()

    default_name = f"mnist_sudoku_{args.base * args.base}x{args.base * args.base}.lp"
    program_path = args.out if args.out is not None else (Path(__file__).resolve().parent / default_name)
    generate(program_path, base=args.base)


if __name__ == "__main__":
    main()
