from functools import reduce

import torch
import torch.nn.functional as F


def data_loss(
    pred: torch.Tensor, target: torch.Tensor, loss_type: str, reduction: str = "mean"
) -> torch.Tensor:
    if loss_type == "bce":
        return F.binary_cross_entropy(pred, target, reduction=reduction)
    if loss_type == "l1":
        return F.l1_loss(pred, target, reduction=reduction)
    if loss_type == "mse":
        return F.mse_loss(pred, target, reduction=reduction)
    raise ValueError(f"Unsupported loss_type '{loss_type}'")


def iter_program_targets(batch):
    """Yield each program's atom span and matrix of compatible target rows."""
    values = batch.stable_values
    layout = getattr(batch, "stable_target_layout", None)
    if layout is not None:
        for start, end, first, last, count in layout:
            if count and last - first == count * (end - start):
                yield start, end, values[first:last].reshape(count, end - start)
            else:
                yield start, end, values.new_empty((0, end - start))
        return

    # Batches created before stable_target_layout can contain rows of unequal length.
    atom_ptr = batch.atom_ptr.tolist()
    model_ptr = batch.stable_model_ptr.tolist()
    value_ptr = batch.stable_value_ptr.tolist()
    for i, (start, end) in enumerate(zip(atom_ptr, atom_ptr[1:])):
        rows = []
        for model in range(model_ptr[i], model_ptr[i + 1]):
            row = values[value_ptr[model]:value_ptr[model + 1]]
            if row.numel() == end - start:
                rows.append(row.reshape(-1))
        yield start, end, torch.stack(rows) if rows else values.new_empty((0, end - start))


def _closest_target_loss(candidate, targets, loss_type):
    targets = targets.to(candidate)
    if targets.size(0) == 1:
        return data_loss(candidate, targets[0], loss_type)

    # Score every model for this program in one tensor operation.
    need_backward = torch.is_grad_enabled() and (candidate.requires_grad or targets.requires_grad)
    with torch.no_grad():
        per_atom = data_loss(candidate.expand_as(targets), targets, loss_type, reduction="none")
        scores = per_atom.mean(dim=1)
        best = scores.amin()
        # A row-wise reduction can round differently from a standalone mean.
        # For nonnegative terms the relative error in a sum grows at most with
        # the number of atoms. Keep a conservative band around the best score.
        fp = torch.finfo(scores.dtype)
        margin = 8 * candidate.numel() * fp.eps
        if margin < 1:
            cutoff = best + best.abs() * margin + candidate.numel() * fp.tiny
            indices = ((scores <= cutoff) | best.isnan()).nonzero().flatten()
        else:
            indices = torch.arange(scores.numel(), device=scores.device)
        if indices.numel() > 1:
            # Match the scalar mean's reduction order and memory alignment before
            # deciding which near-ties really tie in the original implementation.
            near_rows = per_atom.index_select(0, indices).unbind()
            scalar_scores = torch.stack([row.clone().mean() for row in near_rows])
            minimum = scalar_scores.amin()
            if not need_backward:
                return minimum
            indices = indices[(scalar_scores == minimum) | minimum.isnan()]

    # Use the same scalar backward kernels and accumulation order as training
    # against the original individual targets. Only minima need an autograd graph.
    winners = targets.index_select(0, indices)
    if winners.size(0) == 1:
        return data_loss(candidate, winners[0], loss_type)
    return reduce(torch.minimum, (data_loss(candidate, row, loss_type) for row in winners.unbind()))


def closest_stable_loss(
    pred: torch.Tensor,
    batch,
    loss_type: str,
) -> torch.Tensor:
    if pred.dim() > 1 and pred.size(-1) > 1:
        pred = pred.mean(dim=-1)
    elif pred.dim() > 1 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)

    total_loss = pred.new_tensor(0.0)
    for start, end, targets in iter_program_targets(batch):
        if targets.size(0):
            total_loss = total_loss + _closest_target_loss(pred[start:end], targets, loss_type)
    return total_loss