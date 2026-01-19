from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def prepare_run_directory(run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = run_root / timestamp
    suffix = 0
    while run_dir.exists():
        suffix += 1
        run_dir = run_root / f"{timestamp}_{suffix:02d}"
    run_dir.mkdir()
    return run_dir


def setup_logging(log_file: Path, level: int = logging.INFO) -> None:
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode="w"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
