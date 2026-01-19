from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from fuzzy_tp import StableModelGNN
from examples.addition.program import build_program as build_addition_program
from nesy.reasoner import NeSyReasoner
from nesy.logging_utils import prepare_run_directory, setup_logging

LOGGER = logging.getLogger("nesy_pretrain")


def binarization_loss(assignments: torch.Tensor) -> torch.Tensor:
    """Smooth penalty encouraging assignments to be close to {0, 1}."""
    if assignments.dim() != 2:
        raise ValueError("Assignments must be a batch of atom valuations.")
    return (assignments * (1.0 - assignments)).mean()


def step_metrics(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    sum_probs = outputs["sum_probs"]
    sum_pred = torch.argmax(sum_probs, dim=-1)
    sum_acc = (sum_pred == batch["sum"]).float().mean()

    a_pred = torch.argmax(outputs["a_digit_probs"], dim=-1)
    b_pred = torch.argmax(outputs["b_digit_probs"], dim=-1)
    a_acc = (a_pred == batch["digit_a"]).float().mean()
    b_acc = (b_pred == batch["digit_b"]).float().mean()
    digit_acc = torch.stack([a_pred == batch["digit_a"], b_pred == batch["digit_b"]], dim=0).float().mean()
    return {"sum_acc": sum_acc, "a_acc": a_acc, "b_acc": b_acc, "digit_acc": digit_acc}


class RandomAdditionDataset(Dataset):
    def __init__(
        self,
        size: int,
        seed: int,
        logit_bias: float,
        temperature: float,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        self.size = size
        self.logit_bias = logit_bias
        self.temperature = temperature

        generator = torch.Generator()
        generator.manual_seed(seed)
        self.a_digits = torch.randint(0, 10, (size,), generator=generator, dtype=torch.long)
        self.b_digits = torch.randint(0, 10, (size,), generator=generator, dtype=torch.long)
        self.noise_a = torch.randn(size, 10, generator=generator)
        self.noise_b = torch.randn(size, 10, generator=generator)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        logits_a = self.noise_a[idx].clone()
        logits_a[self.a_digits[idx]] += self.logit_bias
        p_a = torch.softmax(logits_a / self.temperature, dim=-1).to(dtype=torch.float32)

        logits_b = self.noise_b[idx].clone()
        logits_b[self.b_digits[idx]] += self.logit_bias
        p_b = torch.softmax(logits_b / self.temperature, dim=-1).to(dtype=torch.float32)

        sum_value = self.a_digits[idx] + self.b_digits[idx]
        return {
            "p_a": p_a,
            "p_b": p_b,
            "digit_a": self.a_digits[idx],
            "digit_b": self.b_digits[idx],
            "sum": torch.tensor(sum_value, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the StableModelGNN on synthetic addition inputs.")
    parser.add_argument("--device", default="cpu", help="Computation device identifier (cpu or cuda:0, ...).")
    parser.add_argument("--seed", type=int, default=7, help="Global random seed.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--train-size", type=int, default=4096, help="Number of synthetic training samples per epoch.")
    parser.add_argument("--eval-size", type=int, default=1024, help="Number of synthetic evaluation samples.")
    parser.add_argument("--log-interval", type=int, default=100, help="Mini-batch interval for logging training stats.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader worker processes.")
    parser.add_argument("--logit-bias", type=float, default=3.0, help="Bias added to the true digit logit when sampling input probabilities.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature controlling input sharpness.")
    parser.add_argument("--iters", type=int, default=30, help="Maximum number of TP/gate/CP iterations.")
    parser.add_argument("--convergence-tol", type=float, default=1e-4, help="Tolerance for convergence detection.")
    parser.add_argument("--disable-convergence", action="store_true", help="Disable convergence-based early stopping.")
    parser.add_argument("--tnorm", choices=["product", "lukasiewicz", "godel"], default="godel", help="T-norm for TP aggregation.")
    parser.add_argument(
        "--support-agg",
        choices=["prob_sum", "max", "mean"],
        default="max",
        help="Aggregation used to combine rule supports per atom.",
    )
    parser.add_argument("--atom-hidden", type=int, default=8, help="Hidden dimension of the gating controller.")
    parser.add_argument("--rnn-type", choices=["gru", "lstm"], default="gru", help="Recurrent cell type for the gate.")
    parser.add_argument("--use-gnn-backbone", action="store_true", help="Enable the optional atom message passing backbone.")
    parser.add_argument(
        "--use-wfs-bounds",
        action="store_true",
        help="Enable iterative well-founded bound refinement during the update loop.",
    )
    parser.add_argument("--positional-init", action="store_true", help="Initialise recurrent states with positional encodings.")
    parser.add_argument("--gnn-checkpoint", type=Path, default=None, help="Optional path to a StableModelGNN checkpoint to initialise from.")
    parser.add_argument("--program-path", type=Path, default=None, help="Path to an ASP/.aspif program defining the addition task.")
    parser.add_argument("--clingo-path", type=str, default=None, help="Optional path to the clingo executable.")
    parser.add_argument("--bin-weight", type=float, default=1.0, help="Weight for the binarisation loss.")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help="Directory where run logs and checkpoints are stored (defaults to project_root/runs_nesy).",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Regenerate random training samples at the start of every epoch.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_reasoner(args: argparse.Namespace, device: torch.device) -> NeSyReasoner:
    program = build_addition_program(args.program_path, args.clingo_path)
    gnn = StableModelGNN(
        num_iters=args.iters,
        tnorm=args.tnorm,
        support_aggregation=args.support_agg,
        rnn_type=args.rnn_type,
        atom_hidden_dim=args.atom_hidden,
        use_gnn_backbone=args.use_gnn_backbone,
        use_wfs_bounds=args.use_wfs_bounds,
        use_convergence=not args.disable_convergence,
        convergence_tol=args.convergence_tol,
        positional_init=args.positional_init,
    )
    if args.gnn_checkpoint is not None:
        ckpt_path = args.gnn_checkpoint
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Provided GNN checkpoint path '{ckpt_path}' does not exist.")
        payload = torch.load(ckpt_path, map_location=device)
        state_dict = payload.get("model_state", payload)
        incompatible = gnn.load_state_dict(state_dict, strict=False)
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        if missing or unexpected:
            LOGGER.warning("Loaded GNN checkpoint with missing=%s unexpected=%s.", missing, unexpected)
        LOGGER.info("Loaded StableModelGNN weights from %s", ckpt_path)
    reasoner = NeSyReasoner(program=program, model=gnn)
    return reasoner.to(device)


def build_dataloaders(args: argparse.Namespace) -> Dict[str, DataLoader]:
    train_dataset = RandomAdditionDataset(
        size=args.train_size,
        seed=args.seed,
        logit_bias=args.logit_bias,
        temperature=args.temperature,
    )
    eval_dataset = RandomAdditionDataset(
        size=args.eval_size,
        seed=args.seed + 1,
        logit_bias=args.logit_bias,
        temperature=args.temperature,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    return {"train": train_loader, "eval": eval_loader}


def regenerate_training_data(loader: DataLoader, seed: int, logit_bias: float, temperature: float) -> None:
    dataset = loader.dataset
    if not isinstance(dataset, RandomAdditionDataset):
        return
    dataset.__init__(dataset.size, seed, logit_bias, temperature)  # type: ignore[misc]


def _reasoner_outputs(
    reasoner: NeSyReasoner,
    p_a: torch.Tensor,
    p_b: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    assignments = reasoner(p_a, p_b)
    program = reasoner.program
    index_map = {key: tensor.to(assignments.device) for key, tensor in program.index_map.items()}
    outputs: Dict[str, torch.Tensor] = {
        "assignments": assignments,
        "sum_probs": assignments.index_select(1, index_map["sum"]),
        "a_digit_probs": assignments.index_select(1, index_map["a_digit"]),
        "b_digit_probs": assignments.index_select(1, index_map["b_digit"]),
        "seed_a": p_a,
        "seed_b": p_b,
    }
    return outputs


def train_epoch(
    reasoner: NeSyReasoner,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int,
    bin_weight: float,
) -> Dict[str, float]:
    reasoner.train()
    running_loss = 0.0
    running_sum_acc = 0.0
    running_a_acc = 0.0
    running_b_acc = 0.0
    running_bin_loss = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(loader, start=1):
        p_a = batch["p_a"].to(device)
        p_b = batch["p_b"].to(device)
        sums = batch["sum"].to(device)

        optimizer.zero_grad()
        outputs = _reasoner_outputs(reasoner, p_a, p_b)
        log_sum = torch.log(outputs["sum_probs"].clamp_min(1e-8))
        loss = F.nll_loss(log_sum, sums)

        bin_loss = torch.tensor(0.0, device=device)
        if bin_weight > 0.0:
            bin_loss = binarization_loss(outputs["assignments"])

        loss = loss + bin_weight * bin_loss

        loss.backward()
        optimizer.step()

        metrics = step_metrics(
            outputs,
            {
                "sum": sums,
                "digit_a": batch["digit_a"].to(device),
                "digit_b": batch["digit_b"].to(device),
            },
        )
        running_loss += loss.item()
        running_sum_acc += metrics["sum_acc"].item()
        running_a_acc += metrics["a_acc"].item()
        running_b_acc += metrics["b_acc"].item()
        running_bin_loss += bin_loss.detach().item()
        total_batches += 1

        if batch_idx % log_interval == 0:
            LOGGER.info(
                "Batch %d/%d - loss: %.4f - sum_acc: %.3f - a_acc: %.3f - b_acc: %.3f - bin_loss: %.4f",
                batch_idx,
                len(loader),
                running_loss / total_batches,
                running_sum_acc / total_batches,
                running_a_acc / total_batches,
                running_b_acc / total_batches,
                running_bin_loss / total_batches,
            )

    return {
        "loss": running_loss / max(total_batches, 1),
        "sum_acc": running_sum_acc / max(total_batches, 1),
        "a_acc": running_a_acc / max(total_batches, 1),
        "b_acc": running_b_acc / max(total_batches, 1),
        "bin_loss": running_bin_loss / max(total_batches, 1),
    }


@torch.no_grad()
def evaluate(
    reasoner: NeSyReasoner,
    loader: DataLoader,
    device: torch.device,
    bin_weight: float,
) -> Dict[str, float]:
    reasoner.eval()
    running_loss = 0.0
    running_sum_acc = 0.0
    running_a_acc = 0.0
    running_b_acc = 0.0
    running_bin_loss = 0.0
    total_batches = 0

    for batch in loader:
        p_a = batch["p_a"].to(device)
        p_b = batch["p_b"].to(device)
        sums = batch["sum"].to(device)

        outputs = _reasoner_outputs(reasoner, p_a, p_b)
        log_sum = torch.log(outputs["sum_probs"].clamp_min(1e-8))
        loss = F.nll_loss(log_sum, sums)

        if bin_weight > 0.0:
            bin_loss_value = binarization_loss(outputs["assignments"]).item()
        else:
            bin_loss_value = 0.0
        running_bin_loss += bin_loss_value

        metrics = step_metrics(
            outputs,
            {
                "sum": sums,
                "digit_a": batch["digit_a"].to(device),
                "digit_b": batch["digit_b"].to(device),
            },
        )
        running_loss += loss.item()
        running_sum_acc += metrics["sum_acc"].item()
        running_a_acc += metrics["a_acc"].item()
        running_b_acc += metrics["b_acc"].item()
        total_batches += 1

    return {
        "loss": running_loss / max(total_batches, 1),
        "sum_acc": running_sum_acc / max(total_batches, 1),
        "a_acc": running_a_acc / max(total_batches, 1),
        "b_acc": running_b_acc / max(total_batches, 1),
        "bin_loss": running_bin_loss / max(total_batches, 1),
    }


def save_checkpoint(path: Path, reasoner: NeSyReasoner, epoch: int) -> None:
    payload = {
        "epoch": epoch,
        "model_state": reasoner.model.state_dict(),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    run_root = args.log_root if args.log_root is not None else project_root / "runs_nesy"
    run_dir = prepare_run_directory(run_root)
    setup_logging(run_dir / "train.log")
    set_seed(args.seed)
    device = torch.device(args.device)

    LOGGER.info("Run directory: %s", run_dir)

    config_payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config_payload["run_dir"] = str(run_dir)
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2))

    reasoner = build_reasoner(args, device)
    data_loaders = build_dataloaders(args)
    optimizer = torch.optim.Adam(reasoner.model.parameters(), lr=args.lr)

    metrics_path = run_dir / "metrics.jsonl"
    best_sum_acc = float("-inf")
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, args.epochs + 1):
            if args.resample:
                regenerate_training_data(
                    data_loaders["train"],
                    seed=args.seed + epoch,
                    logit_bias=args.logit_bias,
                    temperature=args.temperature,
                )
            train_stats = train_epoch(
                reasoner,
                data_loaders["train"],
                optimizer,
                device,
                args.log_interval,
                args.bin_weight,
            )
            eval_stats = evaluate(
                reasoner,
                data_loaders["eval"],
                device,
                args.bin_weight,
            )
            LOGGER.info(
                "Epoch %d/%d | train loss %.4f | train sum_acc %.3f | train bin_loss %.4f | eval loss %.4f | eval sum_acc %.3f | eval bin_loss %.4f",
                epoch,
                args.epochs,
                train_stats["loss"],
                train_stats["sum_acc"],
                train_stats["bin_loss"],
                eval_stats["loss"],
                eval_stats["sum_acc"],
                eval_stats["bin_loss"],
            )

            record = {
                "epoch": epoch,
                "train": train_stats,
                "eval": eval_stats,
            }
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()

            save_checkpoint(last_path, reasoner, epoch)
            if eval_stats["sum_acc"] > best_sum_acc:
                best_sum_acc = eval_stats["sum_acc"]
                save_checkpoint(best_path, reasoner, epoch)


if __name__ == "__main__":
    main()
