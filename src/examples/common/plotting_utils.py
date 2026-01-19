from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


@dataclass(frozen=True)
class PlotTheme:
    dpi: int = 150
    figure_face: str = "#f4f1ec"
    panel_face: str = "#ffffff"
    text_color: str = "#1b1a19"
    muted_color: str = "#4d4a47"
    accent_color: str = "#ad4f2e"
    grid_color: str = "#d4ccc1"


def get_plt():
    return plt


def detach_outputs(outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in outputs.items()}


def digit_preds(prob: torch.Tensor) -> torch.Tensor:
    return torch.argmax(prob, dim=-1)


def top_prob(prob: torch.Tensor) -> float:
    if prob.numel() == 0:
        return float("nan")
    return float(prob.max().item())


def select_indices(
    preds: torch.Tensor,
    targets: torch.Tensor,
    max_samples: int,
    correct_only: bool,
) -> List[int]:
    correct = [i for i, p in enumerate(preds) if int(p) == int(targets[i])]
    if correct_only:
        return correct[:min(max_samples, len(correct))]
    wrong = [i for i in range(len(preds)) if i not in correct]
    return wrong[:min(max_samples, len(wrong))]


def select_from_correct(
    correct: Sequence[int],
    total: int,
    max_samples: int,
    correct_only: bool,
) -> List[int]:
    if correct_only:
        return list(correct)[:max_samples]
    wrong = [i for i in range(total) if i not in correct]
    return (wrong + list(correct))[:max_samples]


def to_np(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 2:
        return img.cpu()
    if img.dim() == 3:
        return img.squeeze(0).cpu()
    return img.cpu()


def styled_grid(
    rows: int,
    cols: int,
    title: str,
    theme: PlotTheme,
) -> Tuple["plt.Figure", "plt.Axes"]:
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 3.25, rows * 2.3),
        dpi=theme.dpi,
        facecolor=theme.figure_face,
    )
    if rows == 1:
        axes = axes.reshape(1, -1)
    for ax in axes.flat:
        ax.set_facecolor(theme.panel_face)
        for spine in ax.spines.values():
            spine.set_color(theme.grid_color)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.suptitle(title, fontsize=12, fontweight="bold", color=theme.text_color)
    return fig, axes


def show_image(ax: "plt.Axes", img: torch.Tensor, title: str, theme: PlotTheme) -> None:
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_title(title, fontsize=9, color=theme.text_color)
    ax.axis("off")


def show_text(ax: "plt.Axes", text: str, theme: PlotTheme) -> None:
    ax.axis("off")
    ax.text(
        0.03,
        0.5,
        text,
        ha="left",
        va="center",
        fontsize=9,
        color=theme.text_color,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#ffffff",
            "edgecolor": theme.grid_color,
        },
    )


def finalize_figure(fig: "plt.Figure") -> None:
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.93))


def hstack_imgs(imgs: List[torch.Tensor], pad: int = 2) -> torch.Tensor:
    if not imgs:
        return torch.empty(0)
    spacer = torch.zeros((imgs[0].shape[0], pad), device=imgs[0].device, dtype=imgs[0].dtype)
    parts: List[torch.Tensor] = []
    for idx, img in enumerate(imgs):
        parts.append(img)
        if idx != len(imgs) - 1:
            parts.append(spacer)
    return torch.cat(parts, dim=1)


def vstack_imgs(rows: List[torch.Tensor], pad: int = 2) -> torch.Tensor:
    if not rows:
        return torch.empty(0)
    spacer = torch.zeros((pad, rows[0].shape[1]), device=rows[0].device, dtype=rows[0].dtype)
    parts: List[torch.Tensor] = []
    for idx, row in enumerate(rows):
        parts.append(row)
        if idx != len(rows) - 1:
            parts.append(spacer)
    return torch.cat(parts, dim=0)


def mosaic(imgs: List[torch.Tensor], rows: int, cols: int) -> torch.Tensor:
    if not imgs:
        return torch.empty(0)
    needed = rows * cols
    padded = imgs + [torch.zeros_like(imgs[0]) for _ in range(max(0, needed - len(imgs)))]
    grid_rows: List[torch.Tensor] = []
    idx = 0
    for _ in range(rows):
        grid_rows.append(hstack_imgs(padded[idx : idx + cols]))
        idx += cols
    return vstack_imgs(grid_rows)
