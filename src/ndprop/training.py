from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .data import packed_stable_model_keys
from .losses import closest_stable_loss, iter_program_targets
from .model import NDProp


def _project_atoms(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce logical feature dimension to a single score per atom."""
    if tensor.dim() > 1 and tensor.size(-1) > 1:
        return tensor.mean(dim=-1)
    if tensor.dim() > 1 and tensor.size(-1) == 1:
        return tensor.squeeze(-1)
    return tensor


def _stable_matches(binary_prediction: torch.Tensor, batch) -> List[bool]:
    """Transfer the batch once, then query each program's packed target set."""
    bits = binary_prediction.cpu().numpy()
    layout = getattr(batch, "stable_target_layout", None)
    if layout is None:
        # Previously collated batches only carry flat target values and pointers.
        atom_ptr = batch.atom_ptr.tolist()
        spans = list(zip(atom_ptr, atom_ptr[1:]))
        if getattr(batch, "stable_model_keys", None) is None:
            batch.stable_model_keys = [
                packed_stable_model_keys(rows) for _, _, rows in iter_program_targets(batch)
            ]
    else:
        spans = [(start, end) for start, end, _, _, _ in layout]
    return [
        np.packbits(bits[start:end]).tobytes() in keys
        for (start, end), keys in zip(spans, batch.stable_model_keys)
    ]


def train_epoch(
    model: NDProp,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_type: str,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    stable_matches = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        num_atoms = int(batch.num_atoms.item())
        num_rules = int(batch.num_rules.item())
        pred = model(
            batch.p0,
            batch.edge_pos,
            batch.edge_neg,
            batch.head_edge,
            num_atoms,
            num_rules,
            atom_ptr=batch.atom_ptr,
            fact_mask=batch.fact_mask,
            fact_values=batch.fact_values,
        )
        loss = closest_stable_loss(pred, batch, loss_type)
        # eps = 1e-8
        # loss = -(pred * torch.log(pred + eps) +
        #         (1 - pred) * torch.log(1 - pred + eps)).mean()
        # loss = loss + (pred * (1 - pred)). mean()  # encourage predictions to be close to 0 or 1
        if loss.requires_grad and any(p.requires_grad for p in model.parameters()):
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        projected = _project_atoms(pred)
        matches = _stable_matches(projected > 0.5, batch)
        stable_matches += sum(matches)
        total += len(matches)
    avg_loss = total_loss / max(1, total)
    accuracy = stable_matches / max(1, total)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: NDProp,
    loader: DataLoader,
    device: torch.device,
    loss_type: str,
    unsolved_indices: Optional[List[int]] = None,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    stable_matches = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        num_atoms = int(batch.num_atoms.item())
        num_rules = int(batch.num_rules.item())
        pred = model(
            batch.p0,
            batch.edge_pos,
            batch.edge_neg,
            batch.head_edge,
            num_atoms,
            num_rules,
            atom_ptr=batch.atom_ptr,
            fact_mask=batch.fact_mask,
            fact_values=batch.fact_values,
        )
        # eps = 1e-8
        # batch_loss = -(pred * torch.log(pred + eps) +
        #         (1 - pred) * torch.log(1 - pred + eps)).mean()
        batch_loss = closest_stable_loss(pred, batch, loss_type)
        # batch_loss = batch_loss + (pred * (1 - pred)). mean()  # encourage predictions to be close to 0 or 1
        total_loss += batch_loss.item()
        projected = _project_atoms(pred)
        matches = _stable_matches(projected >= threshold, batch)
        if unsolved_indices is not None:
            unsolved_indices.extend(total + i for i, matched in enumerate(matches) if not matched)
        stable_matches += sum(matches)
        total += len(matches)
    avg_loss = total_loss / max(1, total)
    accuracy = stable_matches / max(1, total)
    return avg_loss, accuracy