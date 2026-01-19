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
    select_from_correct,
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

    row_probs = outputs["row_probs"]
    col_probs = outputs["col_probs"]
    row_probs_seeds = outputs.get("seed_digits")[:, :2, :]
    col_probs_seeds = outputs.get("seed_digits")[:, 2:, :]
    row_pred = digit_preds(row_probs)
    col_pred = digit_preds(col_probs)
    correct = [
        i
        for i in range(row_pred.size(0))
        if (row_pred[i] == batch["sums"][i, :2]).all() and (col_pred[i] == batch["sums"][i, 2:]).all()
    ]
    imgs = batch["imgs"]
    flat_imgs = imgs.view(imgs.size(0), imgs.size(1), imgs.size(3), imgs.size(4))

    def _row_col_details(sample_idx: int) -> str:
        row_conf = row_probs[sample_idx].max(dim=-1).values.tolist()
        col_conf = col_probs[sample_idx].max(dim=-1).values.tolist()
        row_part = ", ".join(f"{int(v)}({p:.2f})" for v, p in zip(row_pred[sample_idx].tolist(), row_conf))
        col_part = ", ".join(f"{int(v)}({p:.2f})" for v, p in zip(col_pred[sample_idx].tolist(), col_conf))
        row_part_seeds = ", ".join(f"{int(v)}({p:.2f})" for v, p in zip(row_pred[sample_idx].tolist(), row_probs_seeds[sample_idx].max(dim=-1).values.tolist()))
        col_part_seeds = ", ".join(f"{int(v)}({p:.2f})" for v, p in zip(col_pred[sample_idx].tolist(), col_probs_seeds[sample_idx].max(dim=-1).values.tolist()))
        gt = batch["sums"][sample_idx].tolist()
        return (
            f"Row preds: {row_part}\n"
            f"Col preds: {col_part}\n"
            f"Seed row preds: {row_part_seeds}\n"
            f"Seed col preds: {col_part_seeds}\n"
            f"GT rows: {gt[:2]}\n"
            f"GT cols: {gt[2:]}"
        )

    def _make_figure(correct_only: bool):
        indices = select_from_correct(correct, row_pred.size(0), max_samples, correct_only)
        if not indices:
            return None
        title = f"Epoch {epoch} {'correct' if correct_only else 'wrong'} samples"
        fig, axes = styled_grid(len(indices), 2, title, theme)
        for row_idx, sample_idx in enumerate(indices):
            grid = mosaic([to_np(flat_imgs[sample_idx, k]) for k in range(flat_imgs.size(1))], rows=2, cols=2)
            show_image(axes[row_idx, 0], grid, "Digits", theme)
            digits = batch.get("digits")
            prefix = f"GT digits: {digits[sample_idx].tolist()}\n" if digits is not None else ""
            text = prefix + _row_col_details(sample_idx)
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
