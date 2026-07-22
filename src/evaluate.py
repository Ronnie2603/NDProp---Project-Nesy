import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from ndprop import (
    NDProp,
    collate_programs,
    format_answer_set,
    load_program_from_file,
    load_program_from_aspif,
    ground_program_to_aspif,
    get_stable_models,
    resolve_clingo_command,
)


def load_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not contain config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(run_dir: Path, device: torch.device) -> dict:
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not contain best.pt")
    return torch.load(checkpoint_path, map_location=device)


def instantiate_model(config: dict, device: torch.device):
    model = NDProp(
        num_iters=config.get("model_iters", 50),
        tnorm=config.get("tnorm", "product"),
        tnorm_wfs=config.get("tnorm_wfs"),
        hidden_dim=config.get("atom_hidden", config.get("hidden_dim", 64)),
        logical_dimension=config.get("logical_dimension", config.get("logical_dim", 1)),
        use_convergence=not config.get("disable_convergence", False),
        convergence_tol=config.get("convergence_tol", 1e-4),
        decision_normalization=config.get("decision_normalization", "none"),
    )
    model_config = {
        "num_iters": config.get("model_iters", 50),
        "tnorm": config.get("tnorm", "product"),
        "tnorm_wfs": config.get("tnorm_wfs"),
        "hidden_dim": config.get("atom_hidden", config.get("hidden_dim", 64)),
        "logical_dimension": config.get("logical_dimension", config.get("logical_dim", 1)),
        "use_convergence": not config.get("disable_convergence", False),
        "convergence_tol": config.get("convergence_tol", 1e-4),
        "decision_normalization": config.get("decision_normalization", "none"),
    }
    print(json.dumps(model_config, indent=2))
    return model.to(device)


def evaluate_program(
    run_dir: Path,
    program_path: Path,
    device: torch.device,
    threshold: float,
    clingo_path: Optional[str],
    aspif: bool,
    ground: bool,
    model_iters: Optional[int],
    logical_dimension_override: Optional[int] = None,
) -> Path:
    config = load_config(run_dir)
    if logical_dimension_override is not None:
        if logical_dimension_override < 1:
            raise ValueError("logical_dimension_override must be at least 1")
        config = dict(config)
        config["logical_dimension"] = logical_dimension_override
    checkpoint = load_checkpoint(run_dir, device)
    model = instantiate_model(config, device)
    if model_iters is not None and model_iters > 0:
        model.num_iters = model_iters
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    clingo_cmd = resolve_clingo_command(clingo_path or config.get("clingo_path"))

    if aspif:
        program = load_program_from_aspif(str(program_path), clingo_cmd)
    elif ground:
        program = ground_program_to_aspif(str(program_path), clingo_cmd)
    else:
        program = load_program_from_file(str(program_path), clingo_cmd)
    batched = collate_programs([program]).to(device)

    with torch.no_grad():
        probs = model(
            batched.p0,
            batched.edge_pos,
            batched.edge_neg,
            batched.head_edge,
            int(batched.num_atoms.item()),
            int(batched.num_rules.item()),
            atom_ptr=batched.atom_ptr,
            fact_mask=batched.fact_mask,
            fact_values=batched.fact_values,
        ).cpu()

    atom_ptr = batched.atom_ptr.tolist()
    start, end = atom_ptr[0], atom_ptr[1]
    probs_program = probs[start:end]
    predicted = format_answer_set(program.atom_names, probs_program, threshold=threshold)
    predicted_sorted = sorted(predicted)

    if aspif:
        stable_models = get_stable_models(clingo_cmd, file_path=str(program_path), aspif=True)
    elif ground and getattr(program, "grounded_aspif", None):
        with tempfile.NamedTemporaryFile(suffix=".aspif", mode="w", delete=False, encoding="utf-8") as tmp:
            tmp.write(program.grounded_aspif)
            tmp_path = tmp.name
        try:
            stable_models = get_stable_models(clingo_cmd, file_path=tmp_path, aspif=True)
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass
    else:
        stable_models = get_stable_models(clingo_cmd, program_text=program.program_text)

    stable_sets = [sorted(model) for model in stable_models]
    matched_model = next((model for model in stable_sets if model == predicted_sorted), None)
    is_stable = matched_model is not None

    eval_root = run_dir / "evals"
    eval_root.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    eval_dir = eval_root / timestamp
    eval_dir.mkdir(exist_ok=False)

    mode = "aspif" if aspif else "ground" if ground else "plain"
    with open(eval_dir / "report.json", "w", encoding="utf-8") as report_file:
        json.dump(
            {
                "program_path": str(program_path.resolve()),
                "predicted": predicted_sorted,
                "stable_models": stable_sets,
                "is_stable": is_stable,
                "matched_model": matched_model,
                "threshold": threshold,
                "run_dir": str(run_dir.resolve()),
                "mode": mode,
            },
            report_file,
            indent=2,
        )

    with open(eval_dir / "probabilities.csv", "w", encoding="utf-8") as prob_file:
        prob_file.write("atom,probability\n")
        for atom, prob in zip(program.atom_names, probs_program.tolist()):
            prob_file.write(f"{atom},{prob:.6f}\n")

    program_copy = eval_dir / program_path.name
    with open(program_copy, "w", encoding="utf-8") as prog_file:
        prog_file.write(program.program_text.strip() + "\n")

    grounded_aspif = getattr(program, "grounded_aspif", None)
    if grounded_aspif:
        with open(eval_dir / "grounded.aspif", "w", encoding="utf-8") as aspif_file:
            aspif_file.write(grounded_aspif + "\n")

    summary_path = eval_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(
            f"Run directory: {run_dir.resolve()}\n"
            f"Program: {program_path.resolve()}\n"
            f"Mode: {mode}\n"
            f"Threshold: {threshold}\n"
            f"Stable model found: {is_stable}\n"
            f"Predicted: {predicted_sorted}\n"
            f"Matched stable model: {matched_model}\n"
        )

    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained NDProp model on an ASP program.")
    parser.add_argument("--run-dir", required=True, help="Path to the training run directory containing best.pt")
    parser.add_argument("--file", required=True, help="Path to the ASP program file to evaluate")
    parser.add_argument("--device", default="cpu", help="Device to run evaluation on")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for selecting the answer set")
    parser.add_argument("--clingo-path", type=str, default=None, help="Path or name of clingo executable")
    parser.add_argument("--model-iters", type=int, default=None, help="Override the number of model iterations at evaluation time")
    parser.add_argument(
        "--logical-dimension",
        type=int,
        default=None,
        help="Override the logical feature dimension used by the model (defaults to run configuration).",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--aspif", action="store_true", help="Interpret the input file as ASPIF")
    mode_group.add_argument("--ground", action="store_true", help="Ground the input program before evaluation")
    args = parser.parse_args()

    device = torch.device(args.device)
    run_dir = Path(args.run_dir)
    program_path = Path(args.file)

    summary_path = evaluate_program(
        run_dir,
        program_path,
        device,
        args.threshold,
        args.clingo_path,
        args.aspif,
        args.ground,
        model_iters=args.model_iters,
        logical_dimension_override=args.logical_dimension,
    )
    print(f"Evaluation summary written to {summary_path}")


if __name__ == "__main__":
    main()
