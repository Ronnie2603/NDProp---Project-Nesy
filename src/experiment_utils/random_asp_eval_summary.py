from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a random ASP eval.json file.")
    parser.add_argument("eval_path", type=Path)
    args = parser.parse_args()

    data = json.loads(args.eval_path.read_text(encoding="utf-8"))
    values = list(data.values())
    total = len(values)
    success = sum(1 for value in values if value)
    accuracy = (success / total * 100.0) if total else 0.0
    print(success, total, f"{accuracy:.2f}")


if __name__ == "__main__":
    main()
