from typing import Optional

import torch
import torch.nn.functional as F


def data_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "bce":
        return F.binary_cross_entropy(pred, target)
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"Unsupported loss_type '{loss_type}'")


def closest_stable_loss(
    pred: torch.Tensor,
    batch,
    loss_type: str,
) -> torch.Tensor:
    # Reduce logical feature dimension if present.
    if pred.dim() > 1 and pred.size(-1) > 1:
        pred = pred.mean(dim=-1)
    elif pred.dim() > 1 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)
    atom_ptr = batch.atom_ptr.tolist()
    stable_model_ptr = batch.stable_model_ptr.tolist()
    stable_value_ptr = batch.stable_value_ptr.tolist()
    stable_values = batch.stable_values

    total_loss = pred.new_tensor(0.0)
    counted = 0

    for i in range(len(atom_ptr) - 1):
        atom_start, atom_end = atom_ptr[i], atom_ptr[i + 1]
        stable_start, stable_end = stable_model_ptr[i], stable_model_ptr[i + 1]

        if stable_start >= stable_end:
            continue

        candidate = pred[atom_start:atom_end]
        best_loss: Optional[torch.Tensor] = None

        for stable_idx in range(stable_start, stable_end):
            value_start = stable_value_ptr[stable_idx]
            value_end = stable_value_ptr[stable_idx + 1]
            reference = stable_values[value_start:value_end]
            if reference.dim() > 1 and reference.size(-1) == 1:
                reference = reference.squeeze(-1)
            if reference.numel() != candidate.numel():
                continue
            reference = reference.to(dtype=candidate.dtype, device=candidate.device)
            loss_val = data_loss(candidate, reference, loss_type)
            if best_loss is None:
                best_loss = loss_val
            else:
                best_loss = torch.minimum(best_loss, loss_val)

        if best_loss is not None:
            total_loss = total_loss + best_loss
            counted += 1

    if counted == 0:
        return pred.new_tensor(0.0, device=pred.device)
    return total_loss# / counted
