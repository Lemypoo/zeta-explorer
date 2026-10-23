from __future__ import annotations

import functools
import math
from collections.abc import Callable

import numpy as np
from numpy.polynomial import Chebyshev
from numpy.typing import ArrayLike

from zeta_explorer.gram import RS_FROM, good_gram, z_values
from zeta_explorer.riemann_siegel import theta, z_rs
from zeta_explorer.zeros import find_zeros, gram_points, illinois_many

S_MAX = 6.0


def zeros_between(n_lo: int, n_hi: int, rtol: float = 1e-12, max_level: int = 10) -> np.ndarray:
    g = gram_points(np.arange(n_lo, n_hi + 1))
    if g[0] < RS_FROM:
        raise ValueError(f"zeros_between works above t = {RS_FROM:g}")
    return gram_block_zeros(g, n_lo, z_values, z_rs, rtol=rtol, max_level=max_level)


def gram_block_zeros(
    g: np.ndarray,
    n0: int,
    sample: Callable[[np.ndarray], np.ndarray],
    refine: Callable[[np.ndarray], np.ndarray],
    rtol: float = 1e-12,
    atol: float = 0.0,
    max_level: int = 10,
    max_merge: int = 4,
) -> np.ndarray:
    z = sample(g)
    good = np.flatnonzero(good_gram(z, n0=n0))
    blocks = list(zip(good[:-1].tolist(), good[1:].tolist(), strict=True))
    spans: list[tuple[int, int, tuple[np.ndarray, ...]]] = []
    i = 0
    while i < len(blocks):
        j, k = blocks[i]
        found = _bracket_changes(g, z, j, k, sample, max_level)
        if found is None and spans:
            j = spans.pop()[0]
            found = _bracket_changes(g, z, j, k, sample, max_level)
        merged = 0
        while found is None and merged < max_merge and i + 1 < len(blocks):
            i += 1
            merged += 1
            k = blocks[i][1]
            found = _bracket_changes(g, z, j, k, sample, max_level)
        if found is None:
            raise RuntimeError(f"Gram blocks in [{n0 + j}, {n0 + k}) are missing zeros")
        spans.append((j, k, found))
        i += 1
    a, b, fa, fb = (np.concatenate(x) for x in zip(*(f for _, _, f in spans), strict=True))
    return illinois_many(refine, a, b, fa, fb, rtol, atol=atol)


def _bracket_changes(
    g: np.ndarray,
    z: np.ndarray,
    j: int,
    k: int,
    sample: Callable[[np.ndarray], np.ndarray],
    max_level: int,
) -> tuple[np.ndarray, ...] | None:
    ts, zs = g[j : k + 1], z[j : k + 1]
    level = 0
    while (change := np.signbit(zs[1:]) != np.signbit(zs[:-1])).sum() < k - j:
        level += 1
        if level > max_level:
            return None
        ts = np.linspace(g[j], g[k], (k - j) * 2**level + 1)
        zs = np.concatenate(([z[j]], sample(ts[1:-1]), [z[k]]))
    return ts[:-1][change], ts[1:][change], zs[:-1][change], zs[1:][change]


@functools.cache
def _low_zeros() -> tuple[int, np.ndarray]:
    m = math.ceil(theta(RS_FROM) / math.pi)
    while not good_gram(z_values(gram_points([m])), n0=m)[0]:
        m += 1
    low = np.array(find_zeros(10, float(gram_points([m])[0])))
    if len(low) != m + 1:
        raise RuntimeError(f"found {len(low)} zeros below g_{m}, expected {m + 1}")
    return m, low


def first_zeros(n: int) -> np.ndarray:
    m, low = _low_zeros()
    if n <= len(low):
        return low[:n].copy()
    hi = n + 8
    while len(high := zeros_between(m, hi)) < n - len(low):
        hi += 16
    return np.concatenate((low, high))[:n]


def unfold(gamma: ArrayLike) -> np.ndarray:
    return np.asarray(theta(np.asarray(gamma, dtype=float))) / math.pi


def normalized_spacings(gamma: ArrayLike) -> np.ndarray:
    return np.diff(unfold(gamma))


def gap_probability(s: float, m: int = 40) -> float:
    if s <= 0:
        return 1.0
    x, w = np.polynomial.legendre.leggauss(m)
    x = s * (x + 1) / 2
    w = np.sqrt(s * w / 2)
    K = np.sinc(x[:, None] - x[None, :])
    return float(np.linalg.det(np.eye(m) - w[:, None] * K * w[None, :]))


@functools.cache
def _gap_interpolant(deg: int = 80) -> Chebyshev:
    return Chebyshev.interpolate(np.vectorize(gap_probability), deg, domain=[0, S_MAX])


def gue_spacing(s: ArrayLike) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    p = _gap_interpolant().deriv(2)(np.clip(s, 0, S_MAX))
    return np.where(s < S_MAX, p, 0.0)


def gue_spacing_cdf(s: ArrayLike) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    F = 1 + _gap_interpolant().deriv(1)(np.clip(s, 0, S_MAX))
    return np.where(s < S_MAX, F, 1.0)


def wigner_surmise(s: ArrayLike) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    return 32 / math.pi**2 * s**2 * np.exp(-4 * s**2 / math.pi)


def montgomery(u: ArrayLike) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return 1 - np.sinc(u) ** 2


def pair_correlation(
    x: ArrayLike, u_max: float = 3.0, bins: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(x, dtype=float))
    starts = np.flatnonzero(x <= x[-1] - u_max)
    if len(starts) == 0:
        raise ValueError("need a window longer than u_max")
    edges = np.linspace(0, u_max, bins + 1)
    counts = np.zeros(bins)
    lag = 1
    while True:
        i = starts[starts + lag < len(x)]
        d = x[i + lag] - x[i]
        d = d[d < u_max]
        if len(d) == 0:
            break
        counts += np.histogram(d, edges)[0]
        lag += 1
    return (edges[:-1] + edges[1:]) / 2, counts / (len(starts) * (edges[1] - edges[0]))


def ks_distance(sample: ArrayLike, cdf) -> float:
    s = np.sort(np.asarray(sample, dtype=float))
    F = cdf(s)
    n = len(s)
    k = np.arange(1, n + 1)
    return float(max(np.max(k / n - F), np.max(F - (k - 1) / n)))
