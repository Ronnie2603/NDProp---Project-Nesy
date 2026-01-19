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
    top_prob,
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

    sum_pred = digit_preds(outputs["sum_probs"])
    digit_probs = outputs.get("digit_probs")
    imgs = batch["imgs"]
    flat_imgs = imgs.view(imgs.size(0), imgs.size(1), imgs.size(3), imgs.size(4))

    def _format_pred_digits(sample_idx: int) -> str:
        if digit_probs is None:
            return "Pred digits: n/a"
        preds = digit_preds(digit_probs[sample_idx]).tolist()
        confs = digit_probs[sample_idx].max(dim=-1).values.tolist()
        parts = [f"{d}({c:.2f})" for d, c in zip(preds, confs)]
        return f"Pred digits: {', '.join(parts)}"

    def _make_figure(correct_only: bool):
        indices = select_indices(sum_pred, batch["sum"], max_samples, correct_only)
        if not indices:
            return None
        title = f"Epoch {epoch} {'correct' if correct_only else 'wrong'} samples"
        fig, axes = styled_grid(len(indices), 2, title, theme)
        for row_idx, sample_idx in enumerate(indices):
            grid = mosaic(
                [to_np(flat_imgs[sample_idx, j]) for j in range(flat_imgs.size(1))],
                rows=2,
                cols=2,
            )
            show_image(axes[row_idx, 0], grid, "Digits", theme)
            gt_digits = batch.get("digits")
            digit_line = _format_pred_digits(sample_idx)
            if gt_digits is not None:
                digit_line = f"GT digits: {gt_digits[sample_idx].tolist()}\n{digit_line}"
            conf_sum = top_prob(outputs["sum_probs"][sample_idx])
            gt_sum = int(batch["sum"][sample_idx].item())
            is_correct = int(sum_pred[sample_idx]) == gt_sum
            text = (
                f"{digit_line}\n"
                f"Sum pred: {int(sum_pred[sample_idx])} (p={conf_sum:.2f})\n"
                f"GT sum: {gt_sum}\n"
                f"Match: {'yes' if is_correct else 'no'}"
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
