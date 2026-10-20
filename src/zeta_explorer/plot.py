"""Matplotlib plots. Needs the `plot` extra."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from zeta_explorer.gram import good_gram, z_values
from zeta_explorer.riemann_siegel import theta, z_exact, z_rs
from zeta_explorer.zeros import find_zeros, gram_points

if TYPE_CHECKING:
    from matplotlib.figure import Figure

SERIES = "#2a78d6"
ACCENT = "#eb6834"
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
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

    _style(ax)
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


def plot_gram(t_min: float = 262.0, t_max: float = 300.0, samples: int = 1500) -> Figure:
    """Z(t) with Gram points, the ones that break Gram's law highlighted."""
    import matplotlib.pyplot as plt

    ts = np.linspace(t_min, t_max, samples)
    n_lo = int(np.ceil(theta(t_min) / np.pi))
    n_hi = int(np.floor(theta(t_max) / np.pi))
    ns = np.arange(n_lo, n_hi + 1)
    g = gram_points(ns)
    z = z_values(g)
    good = good_gram(z, n0=n_lo)

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=INK_SECONDARY, linewidth=0.8)
    ax.plot(ts, z_values(ts), color=SERIES, linewidth=2)
    for mask, color, label in ((good, SERIES, "good"), (~good, ACCENT, "bad")):
        ax.plot(g[mask], z[mask], "o", color=color, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2, label=f"{label} Gram point", linestyle="none")
    for n, t, zt in zip(ns[~good], g[~good], z[~good], strict=True):
        ax.annotate(f"g_{n}", (t, zt), xytext=(8, -14), textcoords="offset points",
                    color=INK_SECONDARY, fontsize=9)

    _style(ax)
    ax.set_xlim(t_min, t_max)
    ax.set_xlabel("t", color=INK)
    ax.set_ylabel("Z(t)", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK_SECONDARY)
    ax.set_title("Gram points: g_n is good when (-1)^n Z(g_n) > 0", color=INK, fontsize=12,
                 loc="left")
    fig.tight_layout()
    return fig


def plot_rs_error(t_min: float = 200.0, t_max: float = 20_000.0, bins: int = 40) -> Figure:
    """Riemann-Siegel error against mpmath for 0..4 correction terms, log-log.

    Each point is the largest error over one log-spaced bin of 25 samples.
    """
    import matplotlib.pyplot as plt

    edges = np.geomspace(t_min, t_max, bins + 1)
    ts = np.concatenate(
        [np.linspace(a, b, 25, endpoint=False) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    )
    exact = np.array([z_exact(t) for t in ts])
    centers = np.sqrt(edges[:-1] * edges[1:])

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for k, color in enumerate(ORDINAL):
        err = np.abs(z_rs(ts, corrections=k) - exact).reshape(bins, 25).max(axis=1)
        ax.loglog(centers, err, color=color, linewidth=2)
        label = "C0" if k == 0 else f"C0 to C{k}"
        ax.annotate(f"{label}, $t^{{-{(2 * k + 3) / 4:g}}}$", (centers[0], err[0]),
                    xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                    color=INK_SECONDARY, fontsize=9)

    _style(ax)
    ax.set_xlim(t_min / 3.5, t_max)
    ax.set_xlabel("t", color=INK)
    ax.set_ylabel("max |error| in bin", color=INK)
    ax.set_title(
        "Riemann-Siegel error with more correction terms (floor: double-precision phases)",
        color=INK, fontsize=12, loc="left",
    )
    fig.tight_layout()
    return fig


def _style(ax) -> None:
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
