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

    pred_a = digit_preds(outputs["a_digit_probs"])
    pred_b = digit_preds(outputs["b_digit_probs"])
    pred_sum = digit_preds(outputs["sum_probs"])

    def _make_figure(correct_only: bool):
        indices = select_indices(pred_sum, batch["sum"], max_samples, correct_only)
        if not indices:
            return None
        title = f"Epoch {epoch} {'correct' if correct_only else 'wrong'} samples"
        fig, axes = styled_grid(len(indices), 3, title, theme)
        for row_idx, sample_idx in enumerate(indices):
            gt_sum = int(batch["sum"][sample_idx].item())
            is_correct = int(pred_sum[sample_idx]) == gt_sum

            conf_a = top_prob(outputs["a_digit_probs"][sample_idx])
            conf_b = top_prob(outputs["b_digit_probs"][sample_idx])
            conf_sum = top_prob(outputs["sum_probs"][sample_idx])
            show_image(
                axes[row_idx, 0],
                to_np(batch["img_a"][sample_idx]),
                f"A pred {int(pred_a[sample_idx])} (p={conf_a:.2f})",
                theme,
            )
            show_image(
                axes[row_idx, 1],
                to_np(batch["img_b"][sample_idx]),
                f"B pred {int(pred_b[sample_idx])} (p={conf_b:.2f})",
                theme,
            )
            text = (
                f"Sum pred: {int(pred_sum[sample_idx])} (p={conf_sum:.2f})\n"
                f"GT sum: {gt_sum}\n"
                f"Match: {'yes' if is_correct else 'no'}"
            )
            show_text(axes[row_idx, 2], text, theme)
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
