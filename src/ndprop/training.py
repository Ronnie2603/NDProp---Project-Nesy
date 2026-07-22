from typing import List, Optional, Tuple

import torch
from torch_geometric.loader import DataLoader

from .losses import closest_stable_loss


def _project_atoms(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce logical feature dimension to a single score per atom."""
    if tensor.dim() > 1 and tensor.size(-1) > 1:
        return tensor.mean(dim=-1)
    if tensor.dim() > 1 and tensor.size(-1) == 1:
        return tensor.squeeze(-1)
    return tensor
from .model import NDProp


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
        pred_bin = (projected > 0.5).float()
        atom_ptr = batch.atom_ptr.cpu().tolist()
        stable_model_ptr = batch.stable_model_ptr.cpu().tolist()
        stable_value_ptr = batch.stable_value_ptr.cpu().tolist()
        stable_values = batch.stable_values
        for i in range(len(atom_ptr) - 1):
            atom_start, atom_end = atom_ptr[i], atom_ptr[i + 1]
            stable_start, stable_end = stable_model_ptr[i], stable_model_ptr[i + 1]
            candidate = pred_bin[atom_start:atom_end]
            matched = False
            for stable_idx in range(stable_start, stable_end):
                value_start = stable_value_ptr[stable_idx]
                value_end = stable_value_ptr[stable_idx + 1]
                reference = stable_values[value_start:value_end]
                if reference.numel() != candidate.numel():
                    continue
                if torch.equal(candidate, reference):
                    matched = True
                    break
            stable_matches += int(matched)
            total += 1
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
    program_offset = 0
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
        pred_bin = (projected >= threshold).float()
        atom_ptr = batch.atom_ptr.cpu().tolist()
        stable_model_ptr = batch.stable_model_ptr.cpu().tolist()
        stable_value_ptr = batch.stable_value_ptr.cpu().tolist()
        stable_values = batch.stable_values
        num_programs = len(atom_ptr) - 1
        for i in range(num_programs):
            atom_start, atom_end = atom_ptr[i], atom_ptr[i + 1]
            stable_start, stable_end = stable_model_ptr[i], stable_model_ptr[i + 1]
            candidate = pred_bin[atom_start:atom_end]
            matched = False
            for stable_idx in range(stable_start, stable_end):
                value_start = stable_value_ptr[stable_idx]
                value_end = stable_value_ptr[stable_idx + 1]
                reference = stable_values[value_start:value_end]
                if reference.numel() != candidate.numel():
                    continue
                if torch.equal(candidate, reference):
                    matched = True
                    break
            stable_matches += int(matched)
            total += 1
            if not matched and unsolved_indices is not None:
                unsolved_indices.append(program_offset + i)
        program_offset += num_programs
    avg_loss = total_loss / max(1, total)
    accuracy = stable_matches / max(1, total)
    return avg_loss, accuracy
