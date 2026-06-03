from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from examples.common import NeSyTask, gnn_from_args, move_batch
from nesy.reasoner import NeSyReasoner

from .datasets import MNISTSudokuDataset, mnist_sudoku_collate
from .model import MNISTSudokuNeSyModel, SudokuDigitNet
from .program import build_program


class MNISTSudokuTask(NeSyTask):
    name = "mnist_sudoku"
    description = "MNIST Sudoku with digit CNN feeding DecisionBasedGNN."

    def _build_digit_transform(self, dataset: str):
        if dataset == "mnist":
            mean = (0.1307,)
            std = (0.3081,)
        elif dataset == "fashion_mnist":
            mean = (0.2860,)
            std = (0.3530,)
        elif dataset == "cifar10":
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2470, 0.2435, 0.2616)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        def _to_tensor(x):
            if torch.is_tensor(x):
                return x
            return transforms.ToTensor()(x)

        return transforms.Compose(
            [
                transforms.Lambda(_to_tensor),
                transforms.Normalize(mean, std),
            ]
        )

    def _digit_loader(
        self,
        *,
        root: Path,
        digits: list[int],
        train: bool,
        batch_size: int,
        num_workers: int,
        dataset: str,
    ) -> DataLoader:
        transform = self._build_digit_transform(dataset)
        if dataset == "mnist":
            ds = datasets.MNIST(str(root), train=train, download=True, transform=transform)
        elif dataset == "fashion_mnist":
            ds = datasets.FashionMNIST(str(root), train=train, download=True, transform=transform)
        elif dataset == "cifar10":
            ds = datasets.CIFAR10(str(root), train=train, download=True, transform=transform)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        targets = ds.targets
        if not torch.is_tensor(targets):
            targets = torch.tensor(targets)
        mask = torch.isin(targets, torch.tensor(digits))
        indices = torch.nonzero(mask, as_tuple=False).view(-1).tolist()
        subset = torch.utils.data.Subset(ds, indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=train, num_workers=num_workers)

    def _pretrain_digit_net(
        self,
        *,
        digit_net: torch.nn.Module,
        digits: list[int],
        root: Path,
        device: torch.device,
        lr: float,
        epochs: int,
        batch_size: int,
        num_workers: int,
        train_fraction: float = 0.1,
        dataset: str = "mnist",
    ) -> dict:
        digit_net = digit_net.to(device)
        opt = torch.optim.Adam(digit_net.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        train_loader = self._digit_loader(
            root=root,
            digits=digits,
            train=True,
            batch_size=batch_size,
            num_workers=num_workers,
            dataset=dataset,
        )

        subset_indices = torch.randperm(len(train_loader.dataset))[:int(len(train_loader.dataset) * train_fraction)]
        subset = torch.utils.data.Subset(train_loader.dataset, subset_indices)
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        eval_loader = self._digit_loader(
            root=root,
            digits=digits,
            train=False,
            batch_size=batch_size,
            num_workers=num_workers,
            dataset=dataset,
        )
        best_state = digit_net.state_dict()
        best_acc = 0.0
        max_steps = 500
        steps = 0
        for epoch in range(max(epochs, 0)):
            digit_net.train()
            for imgs, labels in train_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = digit_net(imgs)
                targets = labels - digits[0]
                opt.zero_grad()
                loss = loss_fn(logits, targets)
                loss.backward()
                opt.step()
                steps += 1
                if steps >= max_steps:
                    break
            digit_net.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels in eval_loader:
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    logits = digit_net(imgs)
                    preds = torch.argmax(logits, dim=-1)
                    targets = labels - digits[0]
                    correct += int((preds == targets).sum().item())
                    total += int(targets.numel())
            acc = correct / max(total, 1)
            print(f"Digit pretrain epoch {epoch+1}/{epochs}: acc={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in digit_net.state_dict().items()}
            if steps >= max_steps:
                break
        return best_state

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--mnist-sudoku-root",
            type=Path,
            default=Path("./data/mnist_sudoku"),
            help="Root directory for MNIST Sudoku tensors.",
        )
        parser.add_argument(
            "--mnist-root",
            type=Path,
            default=Path("./data"),
            help="Root directory for MNIST.",
        )
        parser.add_argument("--mnist-sudoku-train-fraction", type=float, default=0.9, help="Train split fraction.")
        parser.add_argument("--mnist-sudoku-split-seed", type=int, default=0, help="Split seed.")
        parser.add_argument("--mnist-sudoku-digit-path", type=Path, default=None, help="Path to pretrained digit CNN weights.")
        parser.add_argument("--mnist-sudoku-digit-pretrain-epochs", type=int, default=0, help="Epochs to pretrain digit CNN (0 to skip).")
        parser.add_argument("--mnist-sudoku-digit-pretrain-fraction", type=float, default=0.1, help="Fraction of MNIST training data to use for digit CNN pretraining.")
        parser.add_argument("--mnist-sudoku-digit-pretrain-lr", type=float, default=1e-3, help="Learning rate for digit CNN pretraining.")
        parser.add_argument("--mnist-sudoku-digit-pretrain-path", type=Path, default=None, help="Path to save/load pretrained digit CNN.")
        parser.add_argument(
            "--mnist-sudoku-image-dataset",
            choices=["mnist", "fashion_mnist", "cifar10"],
            default="mnist",
            help="Underlying image dataset for Sudoku digits.",
        )
        parser.add_argument("--mnist-sudoku-run-baselines", action="store_true", help="Run MNIST Sudoku baselines and exit.")
        parser.add_argument(
            "--mnist-sudoku-baseline-mode",
            choices=["both", "clingo", "nn"],
            default="both",
            help="Which MNIST Sudoku baseline(s) to run.",
        )
        parser.add_argument("--mnist-sudoku-baseline-limit", type=int, default=-1, help="Limit baseline eval samples.")
        parser.add_argument("--mnist-sudoku-clingo-timeout", type=float, default=5.0, help="Timeout per clingo call (s).")
        parser.add_argument("--mnist-sudoku-max-nodes", type=int, default=200000, help="Max nodes for NN backtracking.")
        parser.add_argument("--mnist-sudoku-nn-baseline-epochs", type=int, default=100, help="Epochs for NN digit baseline.")
        parser.add_argument("--mnist-sudoku-nn-baseline-lr", type=float, default=1e-3, help="Learning rate for NN digit baseline.")
        parser.add_argument("--program-path", type=Path, default=None, help="Path to ASP program (lp/aspif).")
        parser.add_argument("--clingo-path", type=str, default=None, help="Optional clingo executable.")


    def build_model(self, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
        program = build_program(args.program_path, args.clingo_path)
        # args.fixed_atom_num = program.num_atoms
        gnn = gnn_from_args(args)
        reasoner = NeSyReasoner(program, gnn, init_hidden=args.use_init_hidden)
        num_vals = int(program.index_map["digit"].size(1))
        digits = [0] + list(range(1, num_vals + 1))
        in_channels = 1
        if args.mnist_sudoku_image_dataset == "cifar10":
            in_channels = 3
        digit_net = SudokuDigitNet(num_classes=num_vals + 1, in_channels=in_channels)

        def _load_digit_state(path: Path) -> None:
            state = torch.load(path, map_location=device, weights_only=True)
            model_state = digit_net.state_dict()
            filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
            skipped = [k for k, v in state.items() if k not in model_state or v.shape != model_state[k].shape]
            digit_net.load_state_dict(filtered, strict=False)
            digit_net.requires_grad_(False)
            if skipped:
                print(f"Loaded digit CNN from {path} with {len(skipped)} mismatched params skipped.")
            else:
                print(f"Loaded pretrained digit CNN from {path}.")

        def _print_loaded_digit_acc() -> None:
            eval_loader = self._digit_loader(
                root=args.mnist_root,
                digits=digits,
                train=False,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                dataset=args.mnist_sudoku_image_dataset,
            )
            digit_net.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in eval_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    digit_net.to(device)
                    logits = digit_net(images)
                    preds = torch.argmax(logits, dim=-1)
                    correct += int((preds == labels).sum().item())
                    total += int(labels.numel())
            acc = correct / max(total, 1)
            print(f"Digit net eval acc (loaded, {args.mnist_sudoku_image_dataset}): {acc:.4f}")
        pretrain_path = args.mnist_sudoku_digit_pretrain_path
        if pretrain_path is None:
            pretrain_path = args.mnist_sudoku_root / f"digit_net_{num_vals + 1}.pt"
        if args.mnist_sudoku_digit_pretrain_epochs > 0:
            state = self._pretrain_digit_net(
                digit_net=digit_net,
                digits=digits,
                root=args.mnist_root,
                device=device,
                lr=args.mnist_sudoku_digit_pretrain_lr,
                epochs=args.mnist_sudoku_digit_pretrain_epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                train_fraction=args.mnist_sudoku_digit_pretrain_fraction,
                dataset=args.mnist_sudoku_image_dataset,
            )
            pretrain_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, pretrain_path)
            digit_net.load_state_dict(state)
            digit_net.requires_grad_(False)
            print(f"Saved pretrained digit CNN to {pretrain_path}.")
            _print_loaded_digit_acc()
        elif args.mnist_sudoku_digit_path is not None and args.mnist_sudoku_digit_path.exists():
            _load_digit_state(args.mnist_sudoku_digit_path)
            _print_loaded_digit_acc()
        elif pretrain_path is not None and pretrain_path.exists():
            _load_digit_state(pretrain_path)
            _print_loaded_digit_acc()
        model = MNISTSudokuNeSyModel(digit_net, reasoner, program.index_map)
        return model.to(device)

    def build_loaders(self, args: argparse.Namespace) -> tuple[Dict[str, DataLoader], bool]:
        norm_transform = self._build_digit_transform(args.mnist_sudoku_image_dataset)
        train_ds = MNISTSudokuDataset(
            root=args.mnist_sudoku_root,
            train=True,
            train_fraction=args.mnist_sudoku_train_fraction,
            split_seed=args.mnist_sudoku_split_seed,
            transform=norm_transform,
        )
        eval_ds = MNISTSudokuDataset(
            root=args.mnist_sudoku_root,
            train=False,
            train_fraction=args.mnist_sudoku_train_fraction,
            split_seed=args.mnist_sudoku_split_seed,
            transform=norm_transform,
        )
        loaders = {
            "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=mnist_sudoku_collate),
            "eval": DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=mnist_sudoku_collate),
        }
        return loaders, False

    def step(self, model: torch.nn.Module, batch: Dict, device: torch.device):
        batch = move_batch(batch, device)
        outputs = model(batch, return_tau_phi=False)
        sudoku_probs = outputs["sudoku_probs"]
        target = batch["labels"].long()
        target = target.view(target.size(0), -1)

        one_hot = F.one_hot(target, num_classes=sudoku_probs.size(-1)).float()
        weight = torch.ones_like(one_hot)
        weight[one_hot == 1] = sudoku_probs.size(-1) - 1.0
        loss = F.binary_cross_entropy(sudoku_probs, one_hot, weight=weight)

        bad_bin_loss = torch.tensor(0.0, device=loss.device)
        if "bad" in model.index_map:
            bad_idx = model.index_map["bad"].to(sudoku_probs.device)
            bad_prob = outputs["assignments"].index_select(1, bad_idx).squeeze(1)
            # bad_bin_loss = torch.where(bad_prob >= 0.5, -torch.log(1-bad_prob), -torch.log(1-bad_prob*2+1e-5)).mean() 
            bad_bin_loss = (-torch.log(1-bad_prob)).mean()
            
            loss = loss + bad_bin_loss

        preds = torch.argmax(sudoku_probs, dim=-1)
        cell_acc = (preds == target).float().mean()
        grid_acc = (preds == target).all(dim=1).float().mean()
        metrics = {
            "cell_acc": cell_acc,
            "grid_acc": grid_acc,
            "loss": loss.detach(),
            "bad_bin_loss": bad_bin_loss.detach(),
        }
        return loss, metrics

    def run_baselines(self, model: torch.nn.Module, loaders: Dict[str, DataLoader], device: torch.device, args: argparse.Namespace) -> Dict[str, float]:
        import math
        import subprocess

        def _digit_probs(images: torch.Tensor, num_vals: int) -> torch.Tensor:
            batch_size, cells, _, _, _ = images.shape
            in_channels = images.size(2)
            expected = getattr(model.digit_net.encoder[0], "in_channels", in_channels)
            if in_channels != expected:
                if in_channels == 1 and expected == 3:
                    images = images.repeat(1, 1, 3, 1, 1)
                elif in_channels == 3 and expected == 1:
                    images = images[:, :, :1, :, :]
                else:
                    raise ValueError(f"Digit net expects {expected} channels but got {in_channels}.")
            flat = images.view(batch_size * cells, images.size(2), images.size(-2), images.size(-1))
            logits = model.digit_net(flat)
            if logits.size(-1) < num_vals + 1:
                raise ValueError(f"Digit net output has {logits.size(-1)} classes but need {num_vals + 1}.")
            if logits.size(-1) > num_vals + 1:
                logits = logits[:, : num_vals + 1]
            probs = torch.softmax(logits, dim=-1)
            return probs.view(batch_size, cells, probs.size(-1))

        def _run_clingo(program_path: Path, facts: str, timeout: float) -> list[tuple[int, int, int]] | None:
            cmd = [str(args.clingo_path or "clingo"), str(program_path), "-", "-n", "1", "--quiet=1,1"]
            try:
                result = subprocess.run(
                    cmd,
                    input=facts,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return None
            if result.returncode not in [0,10,20,30]:
                print(f"Clingo error (return code {result.returncode}):")
                return None
            model_line = None
            for line in result.stdout.splitlines():
                if "digit(" in line and not line.startswith("Answer"):
                    model_line = line.strip()
            if not model_line:
                return None
            atoms = []
            for atom in model_line.split():
                if not atom.startswith("digit("):
                    continue
                body = atom[len("digit(") : -1]
                parts = body.split(",")
                if len(parts) != 3:
                    continue
                try:
                    r, c, v = (int(parts[0]), int(parts[1]), int(parts[2]))
                except ValueError:
                    continue
                atoms.append((r, c, v))
            return atoms

        def _nn_solve(probs: torch.Tensor, max_nodes: int) -> torch.Tensor | None:
            side = int(math.sqrt(probs.size(0)))
            num_vals = probs.size(1)
            if side * side != probs.size(0):
                raise ValueError("Invalid cell count for Sudoku.")
            base = int(math.sqrt(side))
            if base * base != side:
                raise ValueError("Side is not a square.")
            board = torch.zeros((side, side), dtype=torch.long)
            row_used = [set() for _ in range(side)]
            col_used = [set() for _ in range(side)]
            block_used = [set() for _ in range(side)]

            cell_order = list(range(side * side))
            cell_order.sort(key=lambda idx: -float(probs[idx].max()))
            nodes = 0

            def _block_index(r: int, c: int) -> int:
                return (r // base) * base + (c // base)

            def _search(pos: int) -> bool:
                nonlocal nodes
                if pos >= len(cell_order):
                    return True
                if nodes >= max_nodes:
                    return False
                idx = cell_order[pos]
                r = idx // side
                c = idx % side
                b = _block_index(r, c)
                used = row_used[r] | col_used[c] | block_used[b]
                candidates = [(v + 1, probs[idx, v].item()) for v in range(num_vals) if (v + 1) not in used]
                candidates.sort(key=lambda x: -x[1])
                for v, _ in candidates:
                    nodes += 1
                    board[r, c] = v
                    row_used[r].add(v)
                    col_used[c].add(v)
                    block_used[b].add(v)
                    if _search(pos + 1):
                        return True
                    row_used[r].remove(v)
                    col_used[c].remove(v)
                    block_used[b].remove(v)
                board[r, c] = 0
                return False

            if _search(0):
                return board
            return None

        def _train_digit_baseline(
            *,
            num_vals: int,
            train_loader: DataLoader,
            eval_loader: DataLoader,
            device: torch.device,
            epochs: int,
            lr: float,
        ) -> tuple[float, float, float, float]:
            images = train_loader.dataset.images
            cells = int(images.size(1))
            input_dim = cells * (num_vals + 1)
            output_dim = cells * num_vals

            class GridMLP(torch.nn.Module):
                def __init__(self, in_dim: int, out_dim: int) -> None:
                    super().__init__()
                    self.net = torch.nn.Sequential(
                        torch.nn.Linear(in_dim, 512),
                        torch.nn.ReLU(),
                        torch.nn.Linear(512, out_dim),
                    )

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.net(x)

            digit_net = GridMLP(input_dim, output_dim).to(device)
            opt = torch.optim.Adam(digit_net.parameters(), lr=lr)
            loss_fn = torch.nn.CrossEntropyLoss()

            def _run_epoch(loader: DataLoader, train: bool) -> tuple[float, float]:
                correct_cells = 0
                total_cells = 0
                correct_grids = 0
                total_grids = 0
                if train:
                    digit_net.train()
                else:
                    digit_net.eval()
                for batch in loader:
                    batch = move_batch(batch, device)
                    images = batch["images"]
                    labels = batch["labels"].view(batch["labels"].size(0), -1)
                    with torch.no_grad():
                        probs = _digit_probs(images, num_vals).detach()
                    flat = probs.reshape(probs.size(0), -1)
                    if train:
                        opt.zero_grad()
                        logits = digit_net(flat).view(labels.size(0), -1, num_vals)
                        loss = loss_fn(logits.view(-1, num_vals), labels.view(-1))
                        loss.backward()
                        opt.step()
                    else:
                        with torch.no_grad():
                            logits = digit_net(flat).view(labels.size(0), -1, num_vals)
                    preds = torch.argmax(logits, dim=-1)
                    correct_cells += int((preds == labels).sum().item())
                    total_cells += int(labels.numel())
                    preds_grid = preds.view(labels.size(0), -1)
                    correct_grids += int((preds_grid == labels).all(dim=1).sum().item())
                    total_grids += int(labels.size(0))
                return correct_cells / max(total_cells, 1), correct_grids / max(total_grids, 1)

            train_acc = 0.0
            eval_acc = 0.0
            train_grid = 0.0
            eval_grid = 0.0
            best_eval_acc = 0.0
            best_eval_grid = 0.0
            for epoch in range(max(epochs, 0)):
                train_acc, train_grid = _run_epoch(train_loader, train=True)
                eval_acc, eval_grid = _run_epoch(eval_loader, train=False)
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                if eval_grid > best_eval_grid:
                    best_eval_grid = eval_grid
                print(
                    "NN baseline epoch "
                    f"{epoch+1}/{epochs}: train_acc={train_acc:.4f} train_grid={train_grid:.4f} "
                    f"eval_acc={eval_acc:.4f} eval_grid={eval_grid:.4f}"
                )
            return train_acc, best_eval_acc, train_grid, best_eval_grid

        eval_loader = loaders["eval"]
        train_loader = loaders["train"]
        num_vals = int(model.index_map["digit"].size(1))
        program_path = args.program_path if args.program_path is not None else (Path(__file__).resolve().parent / "programs" / "mnist_sudoku.lp")

        def _eval_digit_net() -> float:
            digit_eval_loader = self._digit_loader(
                root=args.mnist_root,
                digits=[0] + list(range(1, num_vals + 1)),
                train=False,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                dataset=args.mnist_sudoku_image_dataset,
            )
            correct = 0
            total = 0
            model.digit_net.eval()
            with torch.no_grad():
                for images, labels in digit_eval_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    flat = images
                    logits = model.digit_net(flat)
                    preds = torch.argmax(logits, dim=-1)
                    targets = labels
                    mapped = preds
                    correct += int((mapped == targets).sum().item())
                    total += int(targets.numel())
            return correct / max(total, 1)

        digit_acc = _eval_digit_net()
        print(f"Digit net eval acc (pretrained, {args.mnist_sudoku_image_dataset}): {digit_acc:.4f}")

        total_cells = 0
        correct_cells_clingo = 0
        correct_grids_clingo = 0
        correct_cells_nn = 0
        correct_grids_nn = 0
        seen = 0

        model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                batch = move_batch(batch, device)
                images = batch["images"]
                labels = batch["labels"]
                probs = _digit_probs(images, num_vals).cpu()
                labels = labels.view(labels.size(0), -1).cpu()
                for i in range(labels.size(0)):
                    if args.mnist_sudoku_baseline_limit > 0 and seen >= args.mnist_sudoku_baseline_limit:
                        break
                    seen += 1
                    target = labels[i]
                    side = int(math.sqrt(target.numel()))
                    target_grid = target.view(side, side) + 1
                    if args.mnist_sudoku_baseline_mode in ("both", "clingo"):
                        facts = []
                        for cell in range(target.numel()):
                            r = cell // side + 1
                            c = cell % side + 1
                            cell_probs = probs[i, cell]
                            _, pred_idx = torch.max(cell_probs, dim=0)
                            pred = int(pred_idx.item())
                            if pred not in list(range(1, num_vals + 1)):# or target[cell].item() == 0:
                                continue
                            facts.append(f":- not digit({r},{c},{pred}).")
                        atoms = _run_clingo(program_path, "\n".join(facts) + "\n", args.mnist_sudoku_clingo_timeout)
                        if atoms:
                            pred_grid = torch.zeros_like(target_grid)
                            for r, c, v in atoms:
                                if 1 <= r <= side and 1 <= c <= side:
                                    pred_grid[r - 1, c - 1] = v
                            correct = (pred_grid == target_grid).view(-1)
                            correct_cells_clingo += int(correct.sum().item())
                            correct_grids_clingo += int(correct.all().item())
                            total_cells += correct.numel()
                    if args.mnist_sudoku_baseline_mode in ("both", "nn"):
                        nn_probs = _digit_probs(images[i : i + 1], num_vals).squeeze(0).cpu()
                        if nn_probs.size(-1) == num_vals + 1:
                            nn_probs = nn_probs[:, 1:]
                        board = _nn_solve(nn_probs, args.mnist_sudoku_max_nodes)
                        if board is not None:
                            correct = (board == target_grid).view(-1)
                            correct_cells_nn += int(correct.sum().item())
                            correct_grids_nn += int(correct.all().item())
                            total_cells += correct.numel()
                if args.mnist_sudoku_baseline_limit > 0 and seen >= args.mnist_sudoku_baseline_limit:
                    break

        results: Dict[str, float] = {}
        if args.mnist_sudoku_baseline_mode in ("both", "clingo") and total_cells > 0:
            results["clingo_cell_acc"] = correct_cells_clingo / max(seen*16, 1)
            results["clingo_grid_acc"] = correct_grids_clingo / max(seen, 1)
        if args.mnist_sudoku_baseline_mode in ("both", "nn"):
            train_acc, eval_acc, train_grid, eval_grid = _train_digit_baseline(
                num_vals=num_vals,
                train_loader=train_loader,
                eval_loader=eval_loader,
                device=device,
                epochs=args.mnist_sudoku_nn_baseline_epochs,
                lr=args.mnist_sudoku_nn_baseline_lr,
            )
            results["nn_train_cell_acc"] = train_acc
            results["nn_eval_cell_acc"] = eval_acc
            results["nn_train_grid_acc"] = train_grid
            results["nn_eval_grid_acc"] = eval_grid
        results["digit_net_acc"] = digit_acc
        return results
