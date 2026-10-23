from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from zeta_explorer.continuation import dirichlet_sums, em_sums, eta_sums
from zeta_explorer.explicit import chebyshev_psi, pi_from_zeros, prime_pi, psi_from_zeros, riemann_r
from zeta_explorer.gram import good_gram, z_values
from zeta_explorer.multieval import ZWindow
from zeta_explorer.riemann_siegel import theta, z_exact, z_rs
from zeta_explorer.spacing import (
    first_zeros,
    gue_spacing,
    montgomery,
    normalized_spacings,
    pair_correlation,
    unfold,
    wigner_surmise,
)
from zeta_explorer.values import cue_cumulants, cue_density, sample_cumulants
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


def plot_explicit(
    gamma: np.ndarray, counts: tuple[int, ...] = (10, 100, 1000), x_max: float = 100.0,
    samples: int = 3000,
) -> Figure:
    import matplotlib.pyplot as plt

    x = np.linspace(2.0, x_max, samples)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)
    left.step(x, prime_pi(x), where="post", color=INK, linewidth=1.4, label="π(x)")
    left.plot(x, riemann_r(x), color=INK_SECONDARY, linewidth=1.2, linestyle="--",
              label="R(x), no zeros")
    right.step(x, chebyshev_psi(x) - x, where="post", color=INK, linewidth=1.4, label="ψ(x) − x")
    for K, color in zip(counts, (ORDINAL[0], ORDINAL[2], ACCENT), strict=True):
        label = f"{K:,} zeros"
        left.plot(x, pi_from_zeros(x, gamma[:K]), color=color, linewidth=1.2, label=label)
        right.plot(x, psi_from_zeros(x, gamma[:K]) - x, color=color, linewidth=1.2, label=label)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        _style(ax)
        ax.set_xlim(2, x_max)
        ax.set_xlabel("x", color=INK)
    left.set_title("Prime counting function", color=INK, fontsize=11, loc="left")
    right.set_title("Chebyshev ψ(x) minus its main term", color=INK, fontsize=11, loc="left")
    left.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper left")
    right.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower left", ncol=2)
    fig.suptitle("Rebuilding the primes from the zeros of ζ (explicit formulas, truncated)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_lehmer(n: int = 6709, half_width: float = 10.0, samples: int = 2000) -> Figure:
    import matplotlib.pyplot as plt

    gamma = first_zeros(n + 64)
    a, b = gamma[n - 1], gamma[n]
    t_min, t_max = a - half_width, a + half_width
    pad = 2 * (b - a)
    t = np.linspace(t_min, t_max, samples)
    zoom = np.linspace(a - pad, b + pad, samples // 2)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE,
                                      gridspec_kw={"width_ratios": [1.6, 1]})
    origin = round(a)
    for ax, ts, shift in ((left, t, 0), (right, zoom, origin)):
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=INK_SECONDARY, linewidth=0.8)
        ax.plot(ts - shift, z_values(ts), color=SERIES, linewidth=2)
        _style(ax)
    left.set_xlabel("t", color=INK)
    right.set_xlabel(f"t − {origin}", color=INK)
    inside = gamma[(gamma >= t_min) & (gamma <= t_max)]
    left.plot(inside, np.zeros_like(inside), "o", color=SERIES, markersize=6,
              markeredgecolor=SURFACE, markeredgewidth=1.5)
    left.axvspan(a - pad, b + pad, color=ACCENT, alpha=0.15, linewidth=0)
    left.set_xlim(t_min, t_max)
    left.set_ylabel("Z(t)", color=INK)
    left.set_title(f"Z(t) near t = {a:.0f}", color=INK, fontsize=11, loc="left")
    right.plot([a - origin, b - origin], [0, 0], "o", color=ACCENT, markersize=7,
               markeredgecolor=SURFACE, markeredgewidth=1.5)
    right.set_xlim(zoom[0] - origin, zoom[-1] - origin)
    right.set_title(f"Zeros {n} and {n + 1}, {b - a:.4f} apart", color=INK, fontsize=11,
                    loc="left")
    fig.suptitle("A Lehmer pair: Z barely crosses zero twice", color=INK, fontsize=12, x=0.01,
                 ha="left")
    fig.tight_layout()
    return fig


def plot_log_z(sample: np.ndarray, T: float, N: int, bins: int = 120) -> Figure:
    import matplotlib.pyplot as plt

    x = np.linspace(-8, 5, 600)
    k = sample_cumulants(sample)
    selberg = 0.5 * np.log(np.log(T))
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.hist(sample, bins=bins, range=(-8, 5), density=True, color=ORDINAL[0], edgecolor=SURFACE,
            linewidth=0.4, label=f"log|ζ(1/2 + it)|, {len(sample):,} random t, variance {k[1]:.3f}")
    ax.plot(x, cue_density(x, N), color=INK, linewidth=1.8,
            label=f"CUE, N = {N}, variance {cue_cumulants(N)[1]:.3f}")
    ax.plot(x, np.exp(-(x**2) / (2 * selberg)) / np.sqrt(2 * np.pi * selberg), color=ACCENT,
            linewidth=1.4, linestyle="--", label=f"Selberg normal, variance {selberg:.3f}")
    _style(ax)
    ax.set_xlim(-8, 5)
    ax.set_xlabel("log |ζ(1/2 + it)|", color=INK)
    ax.set_ylabel("density", color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper left")
    ax.set_title(f"Value distribution near t = {T:.3g} against random unitary matrices",
                 color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    return fig


def plot_landscape(
    t_min: float = 0.0, t_max: float = 40.0, sigma_min: float = -0.5, sigma_max: float = 2.0,
    floor: float = -1.5, ceiling: float = 1.0, width: int = 600,
) -> Figure:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    zeros = find_zeros(max(t_min, 10.0), t_max)
    t = np.union1d(np.linspace(t_min, t_max, width), zeros)
    rows = int(round(width * (sigma_max - sigma_min) / (t_max - t_min))) * 2
    sigma = np.union1d(np.linspace(sigma_min, sigma_max, rows), [0.5, 1.0])
    S, T = np.meshgrid(sigma, t, indexing="ij")
    s = S + 1j * T
    w = np.full(s.shape, ceiling)
    finite = s != 1
    with np.errstate(divide="ignore"):
        w[finite] = np.clip(np.log10(np.abs(zeta(s[finite]))), floor, ceiling)

    fig = plt.figure(figsize=(11, 6), facecolor=SURFACE)
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.set_facecolor(SURFACE)
    light = LightSource(azdeg=300, altdeg=40)
    rgb = light.shade(w, cmap=plt.get_cmap("viridis"), vmin=floor, vmax=ceiling,
                      blend_mode="soft", vert_exag=0.3)
    ax.plot_surface(T, S, w, facecolors=rgb, rstride=1, cstride=1, linewidth=0,
                    antialiased=False, shade=False)
    ax.scatter(zeros, np.full(len(zeros), 0.5), np.full(len(zeros), floor), color=ACCENT, s=16,
               depthshade=False)
    ax.view_init(elev=42, azim=-70)
    ax.set_box_aspect((2.6, 1.1, 0.7), zoom=1.2)
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(sigma_min, sigma_max)
    ax.set_zlim(floor, ceiling)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(SURFACE)
        axis._axinfo["grid"]["color"] = GRID
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.set_xlabel("t = Im s", color=INK)
    ax.set_ylabel("σ = Re s", color=INK)
    ax.set_zlabel("log₁₀ |ζ(s)|", color=INK)
    fig.text(0.02, 0.95, f"log |ζ(σ + it)|: a spike at the pole s = 1 and a pit at each of the "
             f"{len(zeros)} zeros on σ = 1/2", color=INK, fontsize=12)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


def sweep_sigma(
    t_max: float = 40.0, frames: int = 60, samples: int = 1500, amplitude: float = 0.4,
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    t = np.linspace(0, t_max, samples)
    sigmas = 0.5 + amplitude * np.sin(2 * np.pi * np.arange(frames) / frames) ** 3
    curves = zeta(sigmas[:, None] + 1j * t[None, :])
    zeros = find_zeros(10, t_max)

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=SURFACE,
                                      gridspec_kw={"width_ratios": [1, 1.35]})
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        _style(ax)
    reach = np.abs(curves[:, t > 5]).max() * 1.05
    spiral, = left.plot([], [], color=SERIES, linewidth=1.2)
    left.plot([0], [0], "o", color=ACCENT, markersize=6, markeredgecolor=SURFACE)
    left.set_xlim(-reach, reach)
    left.set_ylim(-reach, reach)
    left.set_aspect("equal")
    left.set_xlabel("Re ζ(σ + it)", color=INK)
    left.set_ylabel("Im ζ(σ + it)", color=INK)
    modulus, = right.plot([], [], color=SERIES, linewidth=1.4)
    right.plot(zeros, np.zeros(len(zeros)), "o", color=ACCENT, markersize=6,
               markeredgecolor=SURFACE)
    right.set_xlim(0, t_max)
    right.set_ylim(0, reach)
    right.set_xlabel("t", color=INK)
    right.set_ylabel("|ζ(σ + it)|", color=INK)
    title = fig.suptitle("", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()

    def draw(i):
        w = curves[i]
        spiral.set_data(w.real, w.imag)
        modulus.set_data(t, np.abs(w))
        title.set_text(f"σ = {sigmas[i]:.2f}: ζ(σ + it) for 0 ≤ t ≤ {t_max:g}, "
                       "zeros only when σ = 1/2")
        return spiral, modulus, title

    return FuncAnimation(fig, draw, frames=frames, interval=80, blit=False)


def plot_multieval(heights: ArrayLike = (1e4, 1e6, 1e8, 1e10, 1e12), points: int = 4) -> Figure:
    import time

    import matplotlib.pyplot as plt
    import mpmath

    heights = np.asarray(heights, dtype=float)
    err_direct, err_window, us_direct, us_window = [], [], [], []
    for T in heights:
        win = ZWindow(T, T + 100)
        u = np.linspace(win.u_lo, win.u_hi, points + 2)[1:-1]
        with mpmath.workdps(35):
            exact = np.array([float(mpmath.siegelz(mpmath.mpf(win.t0) + mpmath.mpf(x)))
                              for x in u])
        err_direct.append(np.abs(z_rs(win.t0 + u) - exact).max())
        err_window.append(np.abs(win.z(u) - exact).max())
        dense = np.linspace(win.u_lo, win.u_hi, 20_000)
        start = time.perf_counter()
        win.z(dense)
        us_window.append((time.perf_counter() - start) / len(dense) * 1e6)
        few = dense[:200]
        start = time.perf_counter()
        z_rs(win.t0 + few)
        us_direct.append((time.perf_counter() - start) / len(few) * 1e6)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    for ax, direct, window, label in (
        (left, err_direct, err_window, "max |Z − mpmath|"),
        (right, us_direct, us_window, "μs per point"),
    ):
        ax.set_facecolor(SURFACE)
        ax.loglog(heights, direct, "o-", color=ORDINAL[1], linewidth=2, label="direct sum")
        ax.loglog(heights, window, "o-", color=ACCENT, linewidth=2, label="NUFFT window")
        _style(ax)
        ax.set_xlabel("t", color=INK)
        ax.set_ylabel(label, color=INK)
    left.set_title("Accuracy against mpmath", color=INK, fontsize=11, loc="left")
    right.set_title("Cost per evaluation after the window is built", color=INK, fontsize=11,
                    loc="left")
    left.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper left")
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
