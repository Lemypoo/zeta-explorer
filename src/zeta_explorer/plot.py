from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from zeta_explorer.continuation import dirichlet_sums, em_sums, eta_sums
from zeta_explorer.gram import good_gram, z_values
from zeta_explorer.riemann_siegel import theta, z_exact, z_rs
from zeta_explorer.spacing import (
    gue_spacing,
    montgomery,
    normalized_spacings,
    pair_correlation,
    unfold,
    wigner_surmise,
)
from zeta_explorer.zeros import find_zeros, gram_points
from zeta_explorer.zeta import zeta

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
    import matplotlib.pyplot as plt

    ts = np.linspace(t_min, t_max, samples)
    zs = [z_exact(t) for t in ts]
    zeros = find_zeros(max(t_min, 10.0), t_max)

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


def phase_colors(w: np.ndarray) -> np.ndarray:
    from matplotlib.colors import hsv_to_rgb

    hue = (np.angle(w) / (2 * np.pi)) % 1
    with np.errstate(divide="ignore", invalid="ignore"):
        bands = np.log2(np.abs(w)) % 1
    value = np.where(np.isfinite(bands), 0.62 + 0.38 * bands, 1.0)
    return hsv_to_rgb(np.stack([hue, np.full_like(hue, 0.78), value], axis=-1))


def plot_domain(
    t_min: float = 0.0, t_max: float = 60.0, sigma_min: float = -5.0, sigma_max: float = 6.0,
    width: int = 1800,
) -> Figure:
    import matplotlib.pyplot as plt

    height = int(width * (sigma_max - sigma_min) / (t_max - t_min))
    t = t_min + (np.arange(width) + 0.5) * (t_max - t_min) / width
    sigma = sigma_min + (np.arange(height) + 0.5) * (sigma_max - sigma_min) / height
    s = sigma[:, None] + 1j * t[None, :]
    w = np.full(s.shape, np.inf + 0j)
    finite = s != 1
    w[finite] = zeta(s[finite])

    fig, ax = plt.subplots(figsize=(12, 12 * height / width + 1.2), facecolor=SURFACE)
    ax.imshow(phase_colors(w), origin="lower", extent=(t_min, t_max, sigma_min, sigma_max),
              aspect="auto", interpolation="lanczos")
    for line in (0, 1):
        ax.axhline(line, color=SURFACE, linewidth=0.8, alpha=0.7)
    ax.axhline(0.5, color=SURFACE, linewidth=1, linestyle=(0, (4, 4)))
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.set_xlabel("t = Im s", color=INK)
    ax.set_ylabel("σ = Re s", color=INK)
    ax.set_title(
        "Phase portrait of ζ(σ + it): hue is arg ζ, shading steps at every doubling of |ζ|",
        color=INK, fontsize=12, loc="left",
    )
    fig.tight_layout()
    return fig


def plot_spiral(t_max: float = 40.0, samples: int = 4000) -> Figure:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    t = np.linspace(0, t_max, samples)
    w = zeta(0.5 + 1j * t)
    points = np.stack([w.real, w.imag], axis=1)
    segments = np.stack([points[:-1], points[1:]], axis=1)
    zeros = find_zeros(10, t_max)

    fig, ax = plt.subplots(figsize=(8.5, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    lines = LineCollection(segments, cmap="viridis", linewidth=1.6)
    lines.set_array((t[:-1] + t[1:]) / 2)
    ax.add_collection(lines)
    ax.plot([w[0].real], [w[0].imag], "o", color=INK, markersize=6)
    ax.annotate("t = 0, ζ(1/2) = −1.4604", (w[0].real, w[0].imag), xytext=(8, -14),
                textcoords="offset points", color=INK_SECONDARY, fontsize=9)
    ax.plot([0], [0], "o", color=ACCENT, markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5)
    ax.annotate(f"origin: the curve passes here {len(zeros)} times", (0, 0), xytext=(-10, 16),
                textcoords="offset points", ha="right", color=ACCENT, fontsize=9)
    _style(ax)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("Re ζ(1/2 + it)", color=INK)
    ax.set_ylabel("Im ζ(1/2 + it)", color=INK)
    bar = fig.colorbar(lines, ax=ax, shrink=0.7, pad=0.02)
    bar.set_label("t", color=INK)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.set_title(f"ζ(1/2 + it) for 0 ≤ t ≤ {t_max:g}", color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    return fig


def plot_continuation(s: complex = 0.5 + 14.134725141734693j, N: int = 400) -> Figure:
    import matplotlib.pyplot as plt

    target = zeta(s)
    series = (
        ("Σ n^(−s), n ≤ N", dirichlet_sums(s, N), ORDINAL[0]),
        ("alternating η series / (1 − 2^(1−s))", eta_sums(s, N), ORDINAL[2]),
        ("Σ n^(−s) + Euler-Maclaurin tail", em_sums(s, N), ACCENT),
    )

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE,
                                      gridspec_kw={"width_ratios": [1, 1.25]})
    n = np.arange(1, N + 1)
    for label, sums, color in series:
        left.plot(sums.real, sums.imag, color=color, linewidth=1.2, label=label)
        right.loglog(n, np.abs(sums - target), color=color, linewidth=1.6, label=label)
    left.plot([target.real], [target.imag], "o", color=INK, markersize=6)
    left.annotate("ζ(s)", (target.real, target.imag), xytext=(8, 6),
                  textcoords="offset points", color=INK, fontsize=10,
                  bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        _style(ax)
    left.set_aspect("equal", adjustable="datalim")
    left.set_xlabel("real part", color=INK)
    left.set_ylabel("imaginary part", color=INK)
    left.set_title("Partial sums in the complex plane", color=INK, fontsize=11, loc="left")
    right.set_xlabel("N", color=INK)
    right.set_ylabel("|partial sum − ζ(s)|", color=INK)
    right.set_title("Distance to ζ(s)", color=INK, fontsize=11, loc="left")
    right.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower left")
    fig.suptitle(f"Three ways to sum ζ(s) at s = {s.real:g} + {s.imag:.4f}i", color=INK,
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_spacing(gamma: np.ndarray, bins: int = 60) -> Figure:
    import matplotlib.pyplot as plt

    gaps = normalized_spacings(gamma)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    s = np.linspace(0, 3.5, 400)
    left.hist(gaps, bins=bins, range=(0, 3.5), density=True, color=ORDINAL[0], edgecolor=SURFACE,
              linewidth=0.5, label=f"{len(gaps):,} zero spacings")
    left.plot(s, gue_spacing(s), color=INK, linewidth=1.8, label="GUE (sine kernel)")
    left.plot(s, wigner_surmise(s), color=ACCENT, linewidth=1.2, linestyle="--",
              label="Wigner surmise")
    left.plot(s, np.exp(-s), color=INK_SECONDARY, linewidth=1.2, linestyle=":", label="Poisson")
    left.set_xlim(0, 3.5)
    left.set_xlabel("normalized spacing", color=INK)
    left.set_ylabel("density", color=INK)
    left.set_title("Nearest-neighbor spacings", color=INK, fontsize=11, loc="left")

    u, r = pair_correlation(unfold(gamma), u_max=3.0, bins=bins)
    uu = np.linspace(0, 3, 400)
    right.bar(u, r, width=u[1] - u[0], color=ORDINAL[0], edgecolor=SURFACE, linewidth=0.5,
              label="zeros")
    right.plot(uu, montgomery(uu), color=INK, linewidth=1.8, label="1 − (sin πu / πu)²")
    right.axhline(1, color=INK_SECONDARY, linewidth=1.2, linestyle=":", label="Poisson")
    right.set_xlim(0, 3)
    right.set_xlabel("u (normalized distance)", color=INK)
    right.set_title("Pair correlation (Montgomery)", color=INK, fontsize=11, loc="left")

    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        _style(ax)
    left.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper right")
    right.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower right")
    fig.suptitle(
        f"Zeros {gamma[0]:,.0f} < t < {gamma[-1]:,.0f} against random matrix theory",
        color=INK, fontsize=12, x=0.01, ha="left",
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
