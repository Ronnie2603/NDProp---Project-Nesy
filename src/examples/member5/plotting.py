from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from examples.common.plotting_utils import (
    PlotTheme,
    detach_outputs,
    digit_preds,
    finalize_figure,
    get_plt,
    mosaic,
    select_indices,
    show_image,
    show_text,
    styled_grid,
    to_np,
)


def plot_samples(
    *,
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    run_dir: Path,
    epoch: int,
    max_samples: int = 5,
    theme: PlotTheme | None = None,
) -> None:
    plt = get_plt()
    if plt is None:
        return
    theme = theme or PlotTheme()
    outputs = detach_outputs(outputs)

    query = batch["query"]
    member_probs = _member_probs(outputs, query)
    member_pred = digit_preds(member_probs)
    digit_probs = outputs.get("digit_probs")
    digit_probs_seeds = outputs.get("seed_digits")
    imgs = batch["imgs"]
    flat_imgs = imgs.view(imgs.size(0), imgs.size(1), imgs.size(3), imgs.size(4))

    def _make_figure(correct_only: bool):
        indices = select_indices(member_pred, batch["member"], max_samples, correct_only)
        if not indices:
            return None
        title = f"Epoch {epoch} {'correct' if correct_only else 'wrong'} samples"
        fig, axes = styled_grid(len(indices), 2, title, theme)
        for row_idx, sample_idx in enumerate(indices):
            grid = mosaic([to_np(flat_imgs[sample_idx, k]) for k in range(flat_imgs.size(1))], rows=1, cols=5)
            show_image(axes[row_idx, 0], grid, "Digits", theme)
            probs = member_probs[sample_idx].tolist()
            pred_val = int(member_pred[sample_idx])
            gt_val = int(batch["member"][sample_idx])
            digit_line = _format_digit_preds(digit_probs, sample_idx)
            digit_line_seeds = _format_digit_preds(digit_probs_seeds, sample_idx)
            text = (
                f"Query: {int(query[sample_idx])}\n"
                f"Pred: {pred_val} (p={probs[pred_val]:.2f})\n"
                f"GT: {gt_val}\n"
                f"P(member=0): {probs[0]:.2f}\n"
                f"P(member=1): {probs[1]:.2f}\n"
                f"{digit_line}\n"
                f"Seed {digit_line_seeds}"
            )
            show_text(axes[row_idx, 1], text, theme)
        finalize_figure(fig)
        return fig

    correct_fig = _make_figure(True)
    if correct_fig is not None:
        correct_fig.savefig(run_dir / "samples_correct.png", facecolor=theme.figure_face)
        plt.close(correct_fig)

    wrong_fig = _make_figure(False)
    if wrong_fig is not None:
        wrong_fig.savefig(run_dir / "samples_wrong.png", facecolor=theme.figure_face)
        plt.close(wrong_fig)


def _member_probs(outputs: Dict[str, torch.Tensor], query: torch.Tensor) -> torch.Tensor:
    member0_probs = outputs["member0_probs"]
    member1_probs = outputs["member1_probs"]
    if member0_probs.dim() > 1 and member0_probs.size(1) > 1:
        member0 = member0_probs.gather(1, query.unsqueeze(1)).squeeze(1)
        member1 = member1_probs.gather(1, query.unsqueeze(1)).squeeze(1)
    else:
        member0 = member0_probs.squeeze(1)
        member1 = member1_probs.squeeze(1)
    return torch.stack([member0, member1], dim=-1)


def _format_digit_preds(digit_probs: torch.Tensor | None, sample_idx: int) -> str:
    if digit_probs is None:
        return "Digit preds: n/a"
    preds = digit_preds(digit_probs[sample_idx]).tolist()
    confs = digit_probs[sample_idx].max(dim=-1).values.tolist()
    parts = [f"{d}({c:.2f})" for d, c in zip(preds, confs)]
    return f"Digit preds: {', '.join(parts)}"
