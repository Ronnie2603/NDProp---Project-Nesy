import argparse
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from fuzzy_tp import (
    DecisionBasedGNN,
    NormalProgramGenerator,
    ProgramGraph,
    collate_programs,
    evaluate,
    format_answer_set,
    generate_dataset,
    ground_program_to_aspif,
    load_program_from_aspif,
    load_program_from_text,
    resolve_clingo_command,
    train_epoch,
)
from torch.utils.data import DataLoader


LOGGER = logging.getLogger(__name__)


def set_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a fuzzy TP approximator on synthetic ASP programs."
    )
    parser.add_argument("--device", default="cpu", help="Device identifier (cpu or cuda:0, ...).")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--verbosity", type=int, default=1, help="Logging verbosity level (0-2).")

    parser.add_argument("--train-instances", type=int, default=128, help="Number of training programs to generate.")
    parser.add_argument("--eval-instances", type=int, default=32, help="Number of evaluation programs to generate.")
    parser.add_argument("--atoms-range", type=int, nargs=2, default=[6, 8], help="Inclusive range for number of atoms.")
    parser.add_argument("--rules-range", type=int, nargs=2, default=[6, 10], help="Inclusive range for number of rules.")
    parser.add_argument("--max-body", type=int, default=3, help="Maximum number of literals in a body.")
    parser.add_argument("--neg-prob", type=float, default=0.5, help="Probability of default-negating a body literal.")
    parser.add_argument("--clingo-path", type=str, default=None, help="Path or name of clingo executable.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Directory containing pre-generated training instances (use all .lp/.aspif files).",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Directory containing pre-generated evaluation instances (use all .lp/.aspif files).",
    )
    parser.add_argument(
        "--train-handcrafted-dir",
        type=Path,
        action="append",
        default=None,
        help="Append handcrafted training programs from the specified directory (can be supplied multiple times).",
    )
    parser.add_argument(
        "--eval-handcrafted-dir",
        type=Path,
        action="append",
        default=None,
        help="Append handcrafted evaluation programs from the specified directory (can be supplied multiple times).",
    )
    parser.add_argument(
        "--train-handcrafted",
        action="append",
        default=None,
        help="Inline ASP program text to append to the generated training dataset (provide one per flag invocation).",
    )
    parser.add_argument(
        "--eval-handcrafted",
        action="append",
        default=None,
        help="Inline ASP program text to append to the generated evaluation dataset (provide one per flag invocation).",
    )
    parser.add_argument(
        "--min-stable-models",
        type=int,
        default=1,
        help="Minimum number of stable models required when sampling synthetic programs.",
    )

    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--model-iters",
        type=int,
        default=50,
        help="Maximum iterations of the supported-model refinement loop.",
    )
    parser.add_argument(
        "--atom-hidden",
        type=int,
        default=64,
        help="Hidden dimension for the decision controller.",
    )
    parser.add_argument(
        "--logical-dimension",
        type=int,
        default=1,
        help="Number of logical feature channels tracked per atom.",
    )
    parser.add_argument(
        "--tnorm",
        choices=["product", "lukasiewicz", "godel"],
        default="product",
        help="Fuzzy conjunction applied to rule bodies.",
    )
    parser.add_argument(
        "--tnorm-wfs",
        choices=["product", "lukasiewicz", "godel"],
        default=None,
        help="Optional t-norm used during WFS-style precomputation (defaults to --tnorm).",
    )
    parser.add_argument(
        "--loss-type",
        choices=["bce", "l1", "mse"],
        default="bce",
        help="Data loss applied to final fuzzy assignments.",
    )
    parser.add_argument("--log-every", type=int, default=10, help="Epoch interval for logging metrics.")
    parser.add_argument("--sample-threshold", type=float, default=0.5, help="Threshold for sample answer set display.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--no-benchmarks", action="store_true", help="Do not include benchmark programs in generated datasets.")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory to store run artifacts.")
    parser.add_argument(
        "--convergence-tol",
        type=float,
        default=1e-4,
        help="Tolerance for convergence detection in the refinement loop.",
    )
    parser.add_argument(
        "--disable-convergence",
        action="store_true",
        help="Disable convergence-based early stopping; always run the full iteration budget.",
    )

    return parser.parse_args()


def serialize_args(args: argparse.Namespace) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        elif isinstance(value, (list, tuple)):
            serialized[key] = [
                str(item) if isinstance(item, Path) else item
                for item in value
            ]
        else:
            serialized[key] = value
    return serialized


def log_sample_evaluation(
    sample,
    probs: torch.Tensor,
    threshold: float,
) -> Tuple[List[str], List[str]]:
    predicted = format_answer_set(sample.atom_names, probs, threshold=threshold)
    actual: List[str] = sorted(sample.stable_model)
    predicted_sorted = sorted(predicted)
    LOGGER.info("Sample evaluation program:\n%s", sample.program_text)
    LOGGER.info("Actual answer set: %s", actual)
    LOGGER.info("Predicted answer set: %s", predicted_sorted)
    LOGGER.info("Exact match: %s", predicted_sorted == actual)
    return predicted_sorted, actual


def load_programs_from_directory(directory: Path, clingo_cmd: Sequence[str]) -> List[ProgramGraph]:
    if directory is None:
        return []
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Provided directory '{directory}' does not exist or is not a directory.")

    allowed_suffixes = {".lp", ".aspif"}
    files = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in allowed_suffixes
    )
    if not files:
        raise ValueError(f"Directory '{directory}' does not contain any .lp or .aspif files.")

    programs: List[ProgramGraph] = []
    for file_path in files:
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".aspif":
                program = load_program_from_aspif(str(file_path), clingo_cmd)
                program.grounded_aspif = file_path.read_text(encoding="utf-8").strip()
            else:
                program = ground_program_to_aspif(str(file_path), clingo_cmd)
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError(f"Failed to load instance '{file_path}': {exc}") from exc
        program.source_path = str(file_path.resolve())
        programs.append(program)

    return programs



def plot_metrics(
    run_dir: Path,
    train_losses: List[float],
    eval_losses: List[float],
    train_accuracies: List[float],
    eval_accuracies: List[float],
    best_accuracy: float,
):
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, eval_losses, label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [a * 100.0 for a in train_accuracies], label="Train Accuracy")
    plt.plot(epochs, [a * 100.0 for a in eval_accuracies], label="Eval Accuracy")
    plt.axhline(best_accuracy * 100.0, linestyle="--", color="green", label="Best Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "accuracy_curve.png")
    plt.close()

def main() -> None:
    args = parse_args()
    if args.logical_dimension < 1:
        raise ValueError("--logical-dimension must be at least 1.")
    set_logging(args.verbosity)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    try:
        clingo_cmd = resolve_clingo_command(args.clingo_path)
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        raise SystemExit(1) from exc

    project_root = Path(__file__).resolve().parent.parent
    runs_root = project_root / "runs"
    runs_root.mkdir(exist_ok=True)
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = runs_root / run_dir
        base_dir = run_dir
        suffix = 1
        while run_dir.exists():
            run_dir = base_dir.parent / f"{base_dir.name}_{suffix}"
            suffix += 1
    else:
        timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = runs_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    file_handler = logging.FileHandler(run_dir / "train.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)

    config_payload = serialize_args(args)
    with open(run_dir / "config.json", "w", encoding="utf-8") as cfg_file:
        json.dump(config_payload, cfg_file, indent=2)

    generator: Optional[NormalProgramGenerator] = None

    def _get_generator() -> NormalProgramGenerator:
        nonlocal generator
        if generator is None:
            generator = NormalProgramGenerator(
                atoms_range=(args.atoms_range[0], args.atoms_range[1]),
                rules_range=(args.rules_range[0], args.rules_range[1]),
                max_body=args.max_body,
                negation_prob=args.neg_prob,
                seed=args.seed,
            )
        return generator

    def _gather_handcrafted(
        directories: Sequence[Path],
        inline_programs: Sequence[str],
        label: str,
    ) -> List[ProgramGraph]:
        programs: List[ProgramGraph] = []
        for directory in directories:
            loaded = load_programs_from_directory(directory, clingo_cmd)
            LOGGER.info(
                "Loaded %d handcrafted %s programs from %s.",
                len(loaded),
                label,
                directory,
            )
            programs.extend(loaded)
        for idx, program_text in enumerate(inline_programs, start=1):
            try:
                program = load_program_from_text(program_text, clingo_cmd)
            except Exception as exc:
                raise RuntimeError(f"Failed to load inline {label} program #{idx}: {exc}") from exc
            program.source_path = f"<inline_{label}_{idx}>"
            programs.append(program)
        if inline_programs:
            LOGGER.info("Added %d inline handcrafted %s programs.", len(inline_programs), label)
        return programs

    train_handcrafted_dirs = args.train_handcrafted_dir or []
    eval_handcrafted_dirs = args.eval_handcrafted_dir or []
    train_inline_programs = args.train_handcrafted or []
    eval_inline_programs = args.eval_handcrafted or []

    handcrafted_train_programs = _gather_handcrafted(train_handcrafted_dirs, train_inline_programs, "training")
    handcrafted_eval_programs = _gather_handcrafted(eval_handcrafted_dirs, eval_inline_programs, "evaluation")
    if handcrafted_train_programs:
        LOGGER.info("Prepared %d handcrafted training programs.", len(handcrafted_train_programs))
    if handcrafted_eval_programs:
        LOGGER.info("Prepared %d handcrafted evaluation programs.", len(handcrafted_eval_programs))

    LOGGER.info("Preparing datasets...")
    tic = time.time()
    if args.train_dir is not None:
        train_programs = load_programs_from_directory(args.train_dir, clingo_cmd)
        LOGGER.info(
            "Loaded %d training programs from %s.",
            len(train_programs),
            args.train_dir,
        )
        if handcrafted_train_programs:
            train_programs.extend(handcrafted_train_programs)
            LOGGER.info("Appended handcrafted training programs. Total training instances: %d.", len(train_programs))
    else:
        generator = _get_generator()
        train_programs = generate_dataset(
            generator,
            args.train_instances,
            clingo_cmd=clingo_cmd,
            aspif=True,
            min_stable_models=args.min_stable_models,
            handcrafted_programs=handcrafted_train_programs,
        )
        LOGGER.info(
            "Generated %d training programs (including %d handcrafted).",
            len(train_programs),
            len(handcrafted_train_programs),
        )

    if args.eval_dir is not None:
        eval_programs = load_programs_from_directory(args.eval_dir, clingo_cmd)
        LOGGER.info(
            "Loaded %d evaluation programs from %s.",
            len(eval_programs),
            args.eval_dir,
        )
        if handcrafted_eval_programs:
            eval_programs.extend(handcrafted_eval_programs)
            LOGGER.info("Appended handcrafted evaluation programs. Total evaluation instances: %d.", len(eval_programs))
    else:
        generator = _get_generator()
        eval_programs = generate_dataset(
            generator,
            args.eval_instances,
            clingo_cmd=clingo_cmd,
            aspif=True,
            handcrafted_programs=handcrafted_eval_programs,
            min_stable_models=args.min_stable_models,
        )
        LOGGER.info(
            "Generated %d evaluation programs (including %d handcrafted).",
            len(eval_programs),
            len(handcrafted_eval_programs),
        )

    if not train_programs:
        raise SystemExit("No training programs available. Provide --train-dir with .lp/.aspif files or generate a dataset.")
    if not eval_programs:
        raise SystemExit("No evaluation programs available. Provide --eval-dir with .lp/.aspif files or generate a dataset.")

    def _dump_programs(programs, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as prog_file:
            for idx, program in enumerate(programs, start=1):
                label = f"Program_{idx}"
                source_info = getattr(program, "source_path", None)
                if source_info:
                    prog_file.write(f"% {label} ({source_info})\n")
                else:
                    prog_file.write(f"% {label}\n")
                prog_file.write(program.program_text.strip() + "\n\n")

    _dump_programs(train_programs, run_dir / "train_programs.lp")
    _dump_programs(eval_programs, run_dir / "eval_programs.lp")

    train_loader = DataLoader(
        train_programs,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_programs,
    )
    eval_loader = DataLoader(
        eval_programs,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_programs,
    )
    LOGGER.info("Dataset preparation completed in %.2f seconds.", time.time() - tic)

    model = DecisionBasedGNN(
        num_iters=args.model_iters,
        tnorm=args.tnorm,
        tnorm_wfs=args.tnorm_wfs,
        hidden_dim=args.atom_hidden,
        logical_dimension=args.logical_dimension,
        use_convergence=not args.disable_convergence,
        convergence_tol=args.convergence_tol,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses: List[float] = []
    train_accuracies: List[float] = []
    eval_losses: List[float] = []
    eval_accuracies: List[float] = []
    best_accuracy = -float("inf")
    best_epoch = -1
    unsolved_report_path = run_dir / "unsolved_eval.json"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.loss_type,
        )
        unsolved_indices: List[int] = []
        eval_loss, accuracy = evaluate(
            model,
            eval_loader,
            device,
            args.loss_type,
            unsolved_indices=unsolved_indices,
            threshold=args.sample_threshold,
        )

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        eval_losses.append(eval_loss)
        eval_accuracies.append(accuracy)

        unsolved_programs: List[Dict[str, object]] = []
        if unsolved_indices:
            for idx in unsolved_indices:
                if idx < 0 or idx >= len(eval_programs):
                    continue
                program = eval_programs[idx]
                source_path = getattr(program, "source_path", None)
                label = f"Program_{idx + 1}"
                entry: Dict[str, object] = {"index": idx, "label": label}
                if source_path:
                    entry["source_path"] = source_path
                unsolved_programs.append(entry)
        report_payload = {"epoch": epoch, "count": len(unsolved_programs), "problems": unsolved_programs}
        with open(unsolved_report_path, "w", encoding="utf-8") as unsolved_file:
            json.dump(report_payload, unsolved_file, indent=2)

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            checkpoint = {
                "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "accuracy": accuracy,
                "args": serialize_args(args),
            }
            torch.save(checkpoint, run_dir / "best.pt")

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            LOGGER.info(
                "Epoch %04d | train_loss=%.4f | eval_loss=%.4f | train_stable=%.2f%% | eval_stable=%.2f%% | best_acc=%.2f%% (epoch %d)",
                epoch,
                train_loss,
                eval_loss,
                train_acc * 100.0,
                accuracy * 100.0,
                best_accuracy * 100.0,
                best_epoch,
            )
            plot_metrics(
                run_dir,
                train_losses,
                eval_losses,
                train_accuracies,
                eval_accuracies,
                best_accuracy,
            )

    LOGGER.info("Training finished.")

    sample = random.choice(eval_programs)
    with torch.no_grad():
        atom_ptr_tensor = torch.tensor([0, sample.num_atoms], dtype=torch.long, device=device)
        rule_ptr_tensor = torch.tensor([0, sample.num_rules], dtype=torch.long, device=device)
        p0_device = sample.graph.p0.to(device)
        probs = model(
            p0_device,
            sample.graph.edge_pos.to(device),
            sample.graph.edge_neg.to(device),
            sample.graph.head_edge.to(device),
            sample.num_atoms,
            sample.num_rules,
            atom_ptr=atom_ptr_tensor,
            rule_ptr=rule_ptr_tensor,
            fact_mask=sample.graph.fact_mask.to(device),
            fact_values=sample.graph.fact_values.to(device),
        ).cpu()
    predicted_sorted, actual_sorted = log_sample_evaluation(sample, probs, threshold=args.sample_threshold)

    sample_report = run_dir / "sample_prediction.txt"
    with open(sample_report, "w", encoding="utf-8") as sample_file:
        sample_file.write("Program:\n")
        sample_file.write(sample.program_text.strip() + "\n\n")
        sample_file.write(f"Actual answer set: {actual_sorted}\n")
        sample_file.write(f"Predicted answer set: {predicted_sorted}\n")
        sample_file.write(f"Exact match: {predicted_sorted == actual_sorted}\n")

    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        metrics_file.write("epoch,train_loss,train_accuracy,eval_loss,eval_accuracy\n")
        for epoch, (tr, ta, ev, acc) in enumerate(
            zip(train_losses, train_accuracies, eval_losses, eval_accuracies), start=1
        ):
            metrics_file.write(f"{epoch},{tr:.6f},{ta:.6f},{ev:.6f},{acc:.6f}\n")

    with open(run_dir / "summary.txt", "w", encoding="utf-8") as summary_file:
        summary_file.write(
            "Best Model Summary\n"
            f"- Best accuracy: {best_accuracy * 100.0:.2f}%\n"
            f"- Best epoch: {best_epoch}\n"
            f"- Checkpoint: best.pt\n"
            f"- Metrics logfile: metrics.csv\n"
        )


if __name__ == "__main__":
    main()
