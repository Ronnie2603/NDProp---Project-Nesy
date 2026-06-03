import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore

from evaluate import instantiate_model, load_checkpoint, load_config
from fuzzy_tp import (
    collate_programs,
    format_answer_set,
    get_stable_models,
    ground_program_to_aspif,
    resolve_clingo_command,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze decision-based GNN behaviour on curated cases.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing config.json and best.pt.")
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=Path("analysis_cases"),
        help="Directory with .lp logic programs to evaluate.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for atom activation.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu, cuda:0, etc.).")
    parser.add_argument(
        "--clingo-path",
        type=str,
        default=None,
        help="Optional path to clingo executable. Falls back to config or PATH lookup.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Optional timestamp override to reuse an analysis directory.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip generating plots and visual artefacts.",
    )
    parser.add_argument(
        "--model-iters",
        type=int,
        default=None,
        help="Override the number of refinement iterations executed by the model during analysis.",
    )
    return parser.parse_args()


def ensure_cases(cases_dir: Path) -> List[Path]:
    if not cases_dir.exists():
        raise FileNotFoundError(f"Cases directory '{cases_dir}' does not exist.")
    case_paths = sorted(path for path in cases_dir.glob("*.lp") if path.is_file())
    if not case_paths:
        raise FileNotFoundError(f"Cases directory '{cases_dir}' does not contain any .lp programs.")
    return case_paths


def to_device(data, device: torch.device):
    if hasattr(data, "to"):
        return data.to(device)
    raise TypeError("Provided data object does not define a .to(device) method.")


def _project_atoms(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce logical feature dimension to a single score per atom (match training)."""
    if tensor.dim() > 1 and tensor.size(-1) > 1:
        return tensor.mean(dim=-1)
    if tensor.dim() > 1 and tensor.size(-1) == 1:
        return tensor.squeeze(-1)
    return tensor


def _project_iterates_with_selected_dims(
    iter_probs: torch.Tensor,
    atom_ptr: torch.Tensor,
    selected_dims: torch.Tensor,
) -> torch.Tensor:
    if iter_probs.dim() != 3:
        return iter_probs
    num_atoms = iter_probs.size(1)
    logical_dim = iter_probs.size(2)
    if selected_dims.numel() == 0 or logical_dim <= 1:
        return _project_atoms(iter_probs)
    if atom_ptr.numel() > 1:
        segment_lengths = torch.tensor(
            [int(atom_ptr[i + 1].item() - atom_ptr[i].item()) for i in range(atom_ptr.numel() - 1)],
            device=iter_probs.device,
            dtype=torch.long,
        )
    else:
        segment_lengths = torch.tensor([num_atoms], device=iter_probs.device, dtype=torch.long)
    if int(segment_lengths.sum().item()) != num_atoms:
        segment_lengths = torch.tensor([num_atoms], device=iter_probs.device, dtype=torch.long)
    atom_dim_indices = torch.repeat_interleave(selected_dims[: segment_lengths.numel()], segment_lengths, dim=0)
    if atom_dim_indices.numel() < num_atoms:
        pad_dim = selected_dims[-1:].expand(num_atoms - atom_dim_indices.numel())
        atom_dim_indices = torch.cat([atom_dim_indices, pad_dim], dim=0)
    elif atom_dim_indices.numel() > num_atoms:
        atom_dim_indices = atom_dim_indices[:num_atoms]
    expanded = atom_dim_indices.unsqueeze(0).unsqueeze(-1).expand(iter_probs.size(0), num_atoms, 1)
    gathered = torch.gather(iter_probs, 2, expanded)
    return gathered.squeeze(-1)


def _has_undecidedness(tau: torch.Tensor, phi: torch.Tensor) -> bool:
    if tau.numel() == 0 or phi.numel() == 0:
        return False
    zero_tau = tau < 0.5
    zero_phi = phi < 0.5
    return bool((zero_tau & zero_phi).any().item())


def run_model(model, batched):
    outputs = model(
        batched.p0,
        batched.edge_pos,
        batched.edge_neg,
        batched.head_edge,
        int(batched.num_atoms.item()),
        int(batched.num_rules.item()),
        atom_ptr=batched.atom_ptr,
        rule_ptr=batched.rule_ptr,
        fact_mask=batched.fact_mask,
        fact_values=batched.fact_values,
        return_debug=True,
    )
    if isinstance(outputs, tuple) and len(outputs) == 2:
        probs, debug = outputs
    else:
        probs, debug = outputs, {}
    iter_probs = None
    if isinstance(debug, dict):
        iter_probs = debug.get("probabilities")
        if iter_probs is None and "iterates" in debug:
            iter_probs = debug["iterates"]
    if iter_probs is not None and torch.is_tensor(iter_probs):
        iter_probs = iter_probs.detach()
    elif iter_probs is None and torch.is_tensor(probs):
        iter_probs = probs.detach().unsqueeze(0)
    return probs.detach(), iter_probs, debug


def collect_iteration_curves(
    iterates: Sequence[torch.Tensor],
    start: int,
    end: int,
    atom_names: Sequence[str],
) -> Dict[str, List[float]]:
    curves: Dict[str, List[float]] = {atom: [] for atom in atom_names}
    for iterate in iterates:
        slice_vals = iterate[start:end]
        values = slice_vals
        if values.dim() > 1 and values.size(-1) == 1:
            values = values.squeeze(-1)
        for atom, value in zip(atom_names, values.tolist()):
            if isinstance(value, list):
                value = value[0]
            curves[atom].append(float(value))
    return curves


def _extract_rule_components(program) -> Tuple[list[list[str]], list[list[str]], list[list[str]]]:
    graph = program.graph
    num_rules = int(graph.num_rules.item())
    atom_names = program.atom_names

    heads: list[list[str]] = [[] for _ in range(num_rules)]
    pos_bodies: list[list[str]] = [[] for _ in range(num_rules)]
    neg_bodies: list[list[str]] = [[] for _ in range(num_rules)]

    head_edge = graph.head_edge.cpu()
    edge_pos = graph.edge_pos.cpu()
    edge_neg = graph.edge_neg.cpu()

    if head_edge.numel() > 0:
        for rule_idx, atom_idx in zip(head_edge[0].tolist(), head_edge[1].tolist()):
            heads[rule_idx].append(atom_names[atom_idx])

    if edge_pos.numel() > 0:
        for atom_idx, rule_idx in zip(edge_pos[0].tolist(), edge_pos[1].tolist()):
            pos_bodies[rule_idx].append(atom_names[atom_idx])

    if edge_neg.numel() > 0:
        for atom_idx, rule_idx in zip(edge_neg[0].tolist(), edge_neg[1].tolist()):
            neg_bodies[rule_idx].append(atom_names[atom_idx])

    return heads, pos_bodies, neg_bodies


def build_reduct_program(program, candidate_atoms: Sequence[str]) -> str:
    candidate_set = set(candidate_atoms)
    heads, pos_bodies, neg_bodies = _extract_rule_components(program)
    lines: list[str] = []

    for rule_idx, head_atoms in enumerate(heads):
        negative_body = neg_bodies[rule_idx]
        if any(atom in candidate_set for atom in negative_body):
            continue
        positive_body = pos_bodies[rule_idx]
        if head_atoms:
            rule_head = ",".join(head_atoms) if len(head_atoms) > 1 else head_atoms[0]
            if positive_body:
                body = ", ".join(positive_body)
                lines.append(f"{rule_head} :- {body}.")
            else:
                lines.append(f"{rule_head}.")
        else:
            if positive_body:
                body = ", ".join(positive_body)
                lines.append(f":- {body}.")
    return "\n".join(lines)


def check_supported_model(
    program,
    candidate: Sequence[str],
    clingo_cmd: Sequence[str],
) -> Tuple[bool, bool, List[str], List[str], List[str]]:
    reduct = build_reduct_program(program, candidate)
    stable_models = get_stable_models(clingo_cmd, program_text=reduct)
    stable_sorted = [sorted(m) for m in stable_models]
    candidate_sorted = sorted(candidate)
    is_candidate_model = candidate_sorted in stable_sorted
    is_supported = is_candidate_model
    support_witness = stable_sorted[0] if stable_sorted else []
    return is_supported, is_candidate_model, [], [], support_witness


def plot_prob_heatmap(
    probabilities: torch.Tensor,
    atom_names: Sequence[str],
    case_dir: Path,
    title: str,
    filename: str = "probabilities.png",
) -> Optional[Path]:
    return None  # heatmaps disabled per request


def plot_atom_curves(
    curves: Dict[str, List[float]],
    case_dir: Path,
    title: str,
    filename: str,
) -> Optional[Path]:
    if plt is None or not curves:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    atoms = list(curves.keys())
    jitter = np.linspace(-0.3, 0.3, num=max(1, len(atoms)))
    for idx, atom in enumerate(atoms):
        values = curves[atom]
        if not values:
            continue
        xs = np.arange(len(values), dtype=float) + jitter[idx % len(jitter)]
        ax.plot(xs, values, marker="o", markersize=3, linewidth=1.2, label=atom)
    ax.set_title(title)
    ax.set_xlabel("Epoch/Iteration")
    ax.set_ylabel("Value")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    out_path = case_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def analyze_case(
    model,
    batched,
    program,
    program_path: Path,
    threshold: float,
    case_dir: Path,
    clingo_cmd: Sequence[str],
    visualize: bool,
) -> Dict[str, object]:
    case_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        probs, iter_probs, debug_rollout = run_model(model, batched)

    atom_ptr = batched.atom_ptr.tolist()
    if len(atom_ptr) < 2:
        raise RuntimeError("Batched data is missing atom pointer information.")
    start, end = atom_ptr[0], atom_ptr[1]
    final_slice = _project_atoms(probs[start:end])
    predicted = sorted(format_answer_set(program.atom_names, final_slice, threshold=threshold))
    tau_projected = None
    phi_projected = None
    if isinstance(debug_rollout, dict):
        tau_projected = debug_rollout.get("tau_projected")
        phi_projected = debug_rollout.get("phi_projected")
    has_undecided = False
    if torch.is_tensor(tau_projected) and torch.is_tensor(phi_projected):
        tau_slice = _project_atoms(tau_projected[start:end])
        phi_slice = _project_atoms(phi_projected[start:end])
        has_undecided = _has_undecidedness(tau_slice, phi_slice)

    stable_models = [sorted(model_set) for model_set in get_stable_models(clingo_cmd, file_path=str(program_path))]
    is_exact_match = predicted in stable_models and not has_undecided
    iter_max_diff: List[float] = []
    iter_exact_match: List[bool] = []
    iter_stack_for_curves = None
    if iter_probs is not None and torch.is_tensor(iter_probs) and iter_probs.numel() > 0:
        if iter_probs.dim() == 1:
            iter_stack = iter_probs.unsqueeze(0)
        else:
            iter_stack = iter_probs
        selected_dims = None
        if isinstance(debug_rollout, dict):
            selected_dims = debug_rollout.get("selected_dimensions")
        if selected_dims is not None and torch.is_tensor(selected_dims):
            atom_ptr_tensor = batched.atom_ptr.to(iter_stack.device)
            iter_stack = _project_iterates_with_selected_dims(iter_stack, atom_ptr_tensor, selected_dims.to(iter_stack.device))
            iter_stack_for_curves = iter_stack
            tau_iter = debug_rollout.get("tau_iterates")
            phi_iter = debug_rollout.get("phi_iterates")
            if torch.is_tensor(tau_iter) and torch.is_tensor(phi_iter):
                tau_iter = _project_iterates_with_selected_dims(tau_iter.to(iter_stack.device), atom_ptr_tensor, selected_dims.to(iter_stack.device))
                phi_iter = _project_iterates_with_selected_dims(phi_iter.to(iter_stack.device), atom_ptr_tensor, selected_dims.to(iter_stack.device))
            else:
                tau_iter = None
                phi_iter = None
        else:
            tau_iter = None
            phi_iter = None
        iter_slices = [_project_atoms(iterate[start:end]) for iterate in iter_stack]
        for idx, iter_slice in enumerate(iter_slices):
            iter_predicted = sorted(format_answer_set(program.atom_names, iter_slice, threshold=threshold))
            iter_has_undecided = False
            if tau_iter is not None and phi_iter is not None:
                tau_slice = _project_atoms(tau_iter[idx][start:end])
                phi_slice = _project_atoms(phi_iter[idx][start:end])
                iter_has_undecided = _has_undecidedness(tau_slice, phi_slice)
            iter_ok = iter_predicted in stable_models and not iter_has_undecided
            if idx + 1 < len(iter_slices):
                diff = (iter_slices[idx + 1] - iter_slice).abs().max().item()
                iter_max_diff.append(diff)
            else:
                iter_max_diff.append(0.0)
            iter_exact_match.append(iter_ok)
    prob_plot = None
    tau_plot = None
    phi_plot = None
    prob_lines = None
    tau_lines = None
    phi_lines = None

    is_supported_model, is_candidate_model, unsupported_atoms, violated_rules, support_witness = check_supported_model(
        program, predicted, clingo_cmd
    )
    curves_source = iter_stack_for_curves if iter_stack_for_curves is not None else iter_probs
    iter_curves = collect_iteration_curves(curves_source if curves_source is not None else [], start, end, program.atom_names) if iter_probs is not None else {}
    if visualize and iter_curves:
        prob_lines = plot_atom_curves(iter_curves, case_dir, "Probabilities per atom", filename="probabilities_lines.png")
    if visualize and isinstance(debug_rollout, dict):
        tau_iter = debug_rollout.get("tau_iterates")
        phi_iter = debug_rollout.get("phi_iterates")
        if tau_iter is not None and torch.is_tensor(tau_iter) and tau_iter.numel() > 0:
            tau_curves = collect_iteration_curves(tau_iter, start, end, program.atom_names)
            tau_lines = plot_atom_curves(tau_curves, case_dir, "Tau per atom", filename="tau_lines.png")
        if phi_iter is not None and torch.is_tensor(phi_iter) and phi_iter.numel() > 0:
            phi_curves = collect_iteration_curves(phi_iter, start, end, program.atom_names)
            phi_lines = plot_atom_curves(phi_curves, case_dir, "Phi per atom", filename="phi_lines.png")

    report = {
        "case": program_path.name,
        "predicted": predicted,
        "stable_models": stable_models,
        "stable_model_count": len(stable_models),
        "is_exact_match": bool(is_exact_match),
        "has_undecided": bool(has_undecided),
        "iter_max_diff": iter_max_diff,
        "is_supported_model": bool(is_supported_model),
        "is_candidate_model": bool(is_candidate_model),
        "unsupported_atoms": unsupported_atoms,
        "violated_rules": violated_rules,
        "support_witness": support_witness,
        "iter_exact_match": iter_exact_match,
        "prob_plot": str(prob_plot) if prob_plot else None,
        "prob_line_plot": str(prob_lines) if prob_lines else None,
        "tau_plot": str(tau_lines) if tau_lines else None,
        "phi_plot": str(phi_lines) if phi_lines else None,
        "output_dir": str(case_dir),
        "debug_keys": sorted(list(debug_rollout.keys())) if isinstance(debug_rollout, dict) else [],
        "selected_dimensions": [],
        "iteration_curves": iter_curves,
    }

    with open(case_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = load_config(args.run_dir)
    checkpoint = load_checkpoint(args.run_dir, device)

    model = instantiate_model(config, device)
    if args.model_iters is not None and args.model_iters > 0:
        model.num_iters = args.model_iters
    if args.model_iters is not None and args.model_iters > 0:
        model.num_iters = args.model_iters
    model.load_state_dict(checkpoint["model_state"])
    model.use_convergence = False
    model.eval()

    cases = ensure_cases(args.cases_dir)
    timestamp = args.timestamp or datetime.now().strftime("analysis_%Y%m%d_%H%M%S")
    analysis_root = (args.run_dir / "analysis" / timestamp).resolve()
    analysis_root.mkdir(parents=True, exist_ok=False)

    clingo_cmd = resolve_clingo_command(args.clingo_path or config.get("clingo_path"))

    manifest: List[Dict[str, object]] = []
    visualize = not args.no_visualize
    summary = []

    for case_path in cases:
        print(f"\n===== Analyzing case: {case_path.name} =====")
        program = ground_program_to_aspif(str(case_path), clingo_cmd, compute_stable_models=False)
        batched = collate_programs([program], no_stable_model=True)
        batched = to_device(batched, device)
        case_dir = analysis_root / case_path.stem
        report = analyze_case(
            model,
            batched,
            program,
            case_path,
            args.threshold,
            case_dir,
            clingo_cmd,
            visualize,
        )
        manifest.append(report)
        verdict = "✓" if report["is_exact_match"] else "✗"
        dims_repr = report.get("selected_dimensions") or []
        dims_text = f" dims={dims_repr}" if dims_repr else ""
        print(
            f"{verdict} {case_path.stem}: predicted={report['predicted']} "
            f"(stable models={report['stable_model_count']}){dims_text} -> {report['output_dir']}"
        )
        summary.append(f"{case_path.stem}: {verdict}")

    manifest_path = analysis_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({"cases": manifest, "threshold": args.threshold}, handle, indent=2)

    eval_summary = {
        report["case"]: bool(report["is_exact_match"]) for report in manifest
    }
    with open(analysis_root / "eval.json", "w", encoding="utf-8") as eval_file:
        json.dump(eval_summary, eval_file, indent=2)

    print(f"\nAnalysis results stored in {analysis_root}")
    print("Summary:")
    for line in summary:
        print(f" - {line}")

    if plt is None and visualize:
        print("matplotlib not available: probability plots were skipped.")


if __name__ == "__main__":
    main()
