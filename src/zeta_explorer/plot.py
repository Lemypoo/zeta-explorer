"""Matplotlib plots. Needs the `plot` extra."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from zeta_explorer.riemann_siegel import z_exact
from zeta_explorer.zeros import find_zeros

if TYPE_CHECKING:
    from matplotlib.figure import Figure

SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def plot_z(t_min: float = 0.0, t_max: float = 60.0, samples: int = 1500) -> Figure:
    """Z(t) with its zeros marked."""
    import matplotlib.pyplot as plt

    ts = np.linspace(t_min, t_max, samples)
    zs = [z_exact(t) for t in ts]
    zeros = find_zeros(max(t_min, 10.0), t_max)  # Z has no zeros below t = 14.13

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=INK_SECONDARY, linewidth=0.8)
    ax.plot(ts, zs, color=SERIES, linewidth=2)
    ax.plot(zeros, [0] * len(zeros), "o", color=SERIES, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2)

    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.set_xlim(t_min, t_max)
    ax.set_xlabel("t", color=INK)
    ax.set_ylabel("Z(t)", color=INK)
    ax.set_title(
        f"Riemann-Siegel Z(t): every sign change is a zero of ζ(1/2 + it)  "
        f"({len(zeros)} zeros up to t = {t_max:g})",
        color=INK, fontsize=12, loc="left",
    )
    fig.tight_layout()
    return fig
