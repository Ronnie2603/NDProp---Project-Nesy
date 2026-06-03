from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None

from examples.common import NeSyTask, RandomDigitTupleDataset, default_transform, move_batch
from examples.registry import TASKS
from nesy.logging_utils import prepare_run_directory, setup_logging

LOGGER = logging.getLogger("nesy")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _mnist_test_accuracy(model: torch.nn.Module, args: argparse.Namespace, device: torch.device) -> float:
    digit_net = getattr(model, "digit_net", None)
    if digit_net is None:
        return float("nan")
    was_training = digit_net.training
    digit_net.eval()
    test_ds = datasets.MNIST(str(args.mnist_root), train=False, download=True, transform=default_transform())
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = digit_net(images)
            preds = torch.argmax(logits, dim=-1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
    if was_training:
        digit_net.train()
    return correct / max(total, 1)


def _op_test_accuracy(model: torch.nn.Module, args: argparse.Namespace, device: torch.device) -> float:
    op_net = getattr(model, "op_net", None)
    if op_net is None:
        return float("nan")
    op_root = getattr(args, "op_root", None)
    op_eval_subdir = getattr(args, "op_eval_subdir", None)
    if op_root is None or op_eval_subdir is None:
        return float("nan")
    op_path = Path(op_root) / op_eval_subdir
    if not op_path.exists():
        return float("nan")
    was_training = op_net.training
    op_net.eval()
    op_transform = transforms.Compose(
        [transforms.Grayscale(1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    test_ds = datasets.ImageFolder(root=str(op_path), transform=op_transform)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = op_net(images)
            preds = torch.argmax(logits, dim=-1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
    if was_training:
        op_net.train()
    return correct / max(total, 1)


def _serialize_history(
    train_history: List[Dict[str, float]],
    eval_history: List[Dict[str, float]],
) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    for split, history in (("train", train_history), ("eval", eval_history)):
        for entry in history:
            epoch = int(entry.get("epoch", 0))
            for key, val in entry.items():
                if key == "epoch":
                    continue
                rows.append({"epoch": epoch, "split": split, "metric": key, "value": float(val)})
    return rows


def _write_metric_artifacts(
    run_dir: Path,
    train_history: List[Dict[str, float]],
    eval_history: List[Dict[str, float]],
) -> None:
    rows = _serialize_history(train_history, eval_history)
    metric_names = sorted({row["metric"] for row in rows})
    for metric in metric_names:
        train_series = {int(e["epoch"]): float(e[metric]) for e in train_history if metric in e}
        eval_series = {int(e["epoch"]): float(e[metric]) for e in eval_history if metric in e}
        epochs = sorted(set(train_series.keys()) | set(eval_series.keys()))
        metric_csv = run_dir / f"{metric}.csv"
        with metric_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "train", "eval"])
            writer.writeheader()
            for ep in epochs:
                writer.writerow(
                    {
                        "epoch": ep,
                        "train": train_series.get(ep, ""),
                        "eval": eval_series.get(ep, ""),
                    }
                )
            if metric == "time":
                total_train = sum(train_series.values())
                total_eval = sum(eval_series.values())
                writer.writerow({"epoch": "total", "train": total_train, "eval": total_eval})
            else:
                best_train = max(train_series.values()) if train_series else ""
                best_eval = max(eval_series.values()) if eval_series else ""
                writer.writerow({"epoch": "best", "train": best_train, "eval": best_eval})
        if plt is None:
            continue
        train_vals = [train_series.get(ep, float("nan")) for ep in epochs]
        eval_vals = [eval_series.get(ep, float("nan")) for ep in epochs]
        if all(math.isnan(v) for v in train_vals) and all(math.isnan(v) for v in eval_vals):
            continue
        fig, ax = plt.subplots()
        ax.plot(epochs, train_vals, label="train")
        ax.plot(epochs, eval_vals, label="eval")
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend()
        fig.tight_layout()
        fig_path = run_dir / f"{metric}.png"
        fig.savefig(fig_path)
        plt.close(fig)


def _extract_eval_ids(dataset) -> List[str]:
    ids: List[str] = []
    try:
        if isinstance(dataset, RandomDigitTupleDataset):
            m = len(dataset.mnist)
            tgt = dataset.mnist.targets
            for idx, tup in enumerate(dataset.indices):
                digits = [int(tgt[i % m]) for i in tup]
                if len(digits) == 2:
                    s = digits[0] + digits[1]
                    ids.append(f"{tup[0]} {tup[1]} {s}")
                elif len(digits) == 3 and getattr(dataset, "label_mode", None) == "member3":
                    query = None
                    if getattr(dataset, "query_digits", None) is not None:
                        query = int(dataset.query_digits[idx])
                    label = 1 if query is not None and query in digits else 0
                    if query is None:
                        ids.append(" ".join(str(i) for i in tup))
                    else:
                        ids.append(f"{tup[0]} {tup[1]} {tup[2]} {query} {label}")
                elif len(digits) == 5 and getattr(dataset, "label_mode", None) == "member5":
                    query = None
                    if getattr(dataset, "query_digits", None) is not None:
                        query = int(dataset.query_digits[idx])
                    label = 1 if query is not None and query in digits else 0
                    if query is None:
                        ids.append(" ".join(str(i) for i in tup))
                    else:
                        ids.append(f"{tup[0]} {tup[1]} {tup[2]} {tup[3]} {tup[4]} {query} {label}")
                elif len(digits) == 4 and getattr(dataset, "label_mode", None) == "addition2":
                    s = (digits[0] * 10 + digits[1]) + (digits[2] * 10 + digits[3])
                    ids.append(f"{tup[0]} {tup[1]} {tup[2]} {tup[3]} {s}")
                elif len(digits) == 4 and getattr(dataset, "label_mode", None) == "add2x2":
                    row0 = digits[0] + digits[1]
                    row1 = digits[2] + digits[3]
                    col0 = digits[0] + digits[2]
                    col1 = digits[1] + digits[3]
                    ids.append(f"{tup[0]} {tup[1]} {tup[2]} {tup[3]} {row0} {row1} {col0} {col1}")
                else:
                    ids.append(" ".join(str(i) for i in tup))
        else:
            try:
                from examples.apply2x2.datasets import Apply2x2Dataset
            except Exception:
                Apply2x2Dataset = None  # type: ignore[assignment]
            if Apply2x2Dataset is not None and isinstance(dataset, Apply2x2Dataset):
                for example in dataset.examples:
                    ids.append(" ".join(str(i) for i in example))
            else:
                raise ValueError("Unsupported dataset type for extracting eval ids.")
    except Exception:
        print("Failed to extract eval ids.")
        pass
    return ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Neuro-symbolic reasoning runner")
    parser.add_argument("--device", default="cpu", help="Computation device identifier (cpu or cuda:0, ...).")
    parser.add_argument("--seed", type=int, default=7, help="Global random seed.")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--log-interval", type=int, default=100, help="Mini-batch interval for logging training stats.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader worker processes.")
    parser.add_argument("--log-root", type=Path, default=None, help="Directory where run logs and metrics are stored.")
    parser.add_argument("--tnorm", choices=["product", "lukasiewicz", "godel"], default="godel", help="T-norm for TP aggregation.")
    parser.add_argument(
        "--tnorm-wfs",
        choices=["product", "lukasiewicz", "godel"],
        default=None,
        help="Optional t-norm used during WFS-style precomputation (defaults to --tnorm).",
    )
    parser.add_argument("--iters", type=int, default=30, help="Maximum number of TP iterations.")
    parser.add_argument("--convergence-tol", type=float, default=1e-4, help="Tolerance for convergence detection.")
    parser.add_argument("--disable-convergence", action="store_true", help="Disable early stopping based on convergence.")
    parser.add_argument("--logical-dimension", type=int, default=1, help="Logical feature channels per atom.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension of the decision controller.")
    parser.add_argument("--use-init-hidden", action="store_true", help="Use evidence-driven initial hidden state for the controller.")
    parser.add_argument("--dataset", choices=sorted(TASKS.keys()), default="addition", help="Which neurosymbolic task to run.")
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Measure single-sample inference time and exit.",
    )
    parser.add_argument(
        "--inference-warmup",
        type=int,
        default=3,
        help="Warmup runs before timing single-sample inference.",
    )
    parser.add_argument(
        "--inference-repeats",
        type=int,
        default=20,
        help="Number of timed repeats for single-sample inference.",
    )
    return parser


def run_epoch(
    *,
    task: NeSyTask,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    log_interval: int,
) -> Tuple[Dict[str, float], float]:
    model.train(optimizer is not None)
    totals: Dict[str, float] = {}
    total_examples = 0
    start_time = time.perf_counter()
    for step_idx, batch in enumerate(loader, 1):
        loss, metrics = task.step(model, batch, device)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = _infer_batch_size(batch)
        total_examples += batch_size
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
        if log_interval and step_idx % log_interval == 0:
            LOGGER.info("Step %d | loss=%.4f", step_idx, float(metrics.get("loss", loss).detach().cpu()))
    elapsed = time.perf_counter() - start_time
    return {key: val / max(total_examples, 1) for key, val in totals.items()}, elapsed


def _infer_batch_size(batch) -> int:
    if isinstance(batch, dict):
        for value in batch.values():
            if torch.is_tensor(value) and value.dim() >= 1:
                return int(value.size(0))
    elif isinstance(batch, (list, tuple)):
        for value in batch:
            if torch.is_tensor(value) and value.dim() >= 1:
                return int(value.size(0))
    return 1


def _slice_batch_to_one(batch):
    if isinstance(batch, dict):
        out = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.dim() >= 1:
                out[key] = value[:1]
            else:
                out[key] = value
        return out
    if isinstance(batch, (list, tuple)):
        sliced = []
        for value in batch:
            if torch.is_tensor(value) and value.dim() >= 1:
                sliced.append(value[:1])
            else:
                sliced.append(value)
        return type(batch)(sliced)
    return batch


def main() -> None:
    parser = build_parser()
    base_args, _ = parser.parse_known_args()
    task = TASKS[base_args.dataset]
    task.add_args(parser)
    args = parser.parse_args()

    device = torch.device(args.device)
    set_seed(args.seed)

    run_root = args.log_root if args.log_root is not None else Path("./runs_nesy") / args.dataset
    run_dir = prepare_run_directory(run_root)
    setup_logging(run_dir / "train.log")
    LOGGER.info("Starting task=%s on device=%s", args.dataset, device)
    (run_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    task = TASKS[args.dataset]
    model = task.build_model(args, device)
    LOGGER.info("Model built: %s", model)

    loaders, eval_is_random = task.build_loaders(args)
    if eval_is_random:
        eval_ids = _extract_eval_ids(loaders["eval"].dataset)
        if eval_ids:
            (run_dir / "eval_ids.txt").write_text("\n".join(eval_ids), encoding="utf-8")

    if args.inference_only:
        model.eval()
        loader = loaders.get("eval_sb", loaders["eval"])
        batch = next(iter(loader))
        batch = _slice_batch_to_one(batch)

        with torch.no_grad():
            for _ in range(max(args.inference_warmup, 0)):
                task.step(model, batch, device)

            use_cuda_timing = device.type == "cuda" and torch.cuda.is_available()
            if use_cuda_timing:
                torch.cuda.synchronize()

            timings = []
            for _ in range(max(args.inference_repeats, 1)):
                start = time.perf_counter()
                task.step(model, batch, device)
                if use_cuda_timing:
                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)

        mean_s = sum(timings) / len(timings)
        print(f"inference_time_single_ms_mean={mean_s * 1000.0:.6f}")
        print(f"inference_time_single_ms_min={min(timings) * 1000.0:.6f}")
        print(f"inference_time_single_ms_max={max(timings) * 1000.0:.6f}")
        with (run_dir / "inference_time.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            writer.writerow(["inference_time_mean", f"{mean_s:.6f}"])
            writer.writerow(["inference_time_min", f"{min(timings):.6f}"])
            writer.writerow(["inference_time_max", f"{max(timings):.6f}"])
        return

    if args.dataset == "mnist_sudoku" and getattr(args, "mnist_sudoku_run_baselines", False):
        baselines = task.run_baselines(model, loaders, device, args)
        (run_dir / "mnist_sudoku_baselines.json").write_text(json.dumps(baselines, indent=2), encoding="utf-8")
        LOGGER.info("MNIST Sudoku baselines: %s", baselines)
        return

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    train_history: List[Dict[str, float]] = []
    eval_history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics, train_time = run_epoch(
            task=task,
            model=model,
            loader=loaders["train"],
            device=device,
            optimizer=optimizer,
            log_interval=args.log_interval,
        )
        eval_metrics, eval_time = run_epoch(
            task=task,
            model=model,
            loader=loaders["eval"],
            device=device,
            optimizer=None,
            log_interval=0,
        )
        if args.dataset in ("addition", "addition2", "add2x2", "member3", "member5", "apply2x2"):
            if args.dataset == "apply2x2":
                eval_metrics["op_test_acc"] = _op_test_accuracy(model, args, device)
            else:
                eval_metrics["mnist_test_acc"] = _mnist_test_accuracy(model, args, device)

            if epoch == args.epochs:
                _, eval_time2 = run_epoch(
                    task=task,
                    model=model,
                    loader=loaders["eval_sb"],
                    device=device,
                    optimizer=None,
                    log_interval=0,
                )
                eval_metrics["eval_sb_time"] = float(eval_time2)

        train_history.append({"epoch": float(epoch), **train_metrics, "time": float(train_time)})
        eval_history.append({"epoch": float(epoch), **eval_metrics, "time": float(eval_time)})
        _write_metric_artifacts(run_dir, train_history, eval_history)
        LOGGER.info(
            "Epoch %d | train=%s (%.2fs) | eval=%s (%.2fs)",
            epoch,
            train_metrics,
            train_time,
            eval_metrics,
            eval_time,
        )
        try:
            sample_batch = next(iter(loaders["eval"]))
            with torch.no_grad():
                model.eval()
                sample_out = model(move_batch(sample_batch, device))
            task.plot_samples(batch=sample_batch, outputs=sample_out, run_dir=run_dir, epoch=epoch)
        except Exception as exc:  # pragma: no cover - visualization best-effort
            LOGGER.warning("Skipping sample plot for epoch %d due to error: %s", epoch, exc)

    LOGGER.info("Finished training. Model saved to %s", run_dir)
    # torch.save({"model_state": model.state_dict(), "args": vars(args)}, run_dir / "best.pt")


if __name__ == "__main__":
    main()
