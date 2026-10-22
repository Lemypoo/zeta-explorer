from __future__ import annotations

import functools
import math

import numpy as np
from numpy.polynomial import Chebyshev
from numpy.typing import ArrayLike

from zeta_explorer.gram import RS_FROM, good_gram, z_values
from zeta_explorer.riemann_siegel import theta, z_rs
from zeta_explorer.zeros import gram_points, illinois_many

S_MAX = 6.0


def zeros_between(n_lo: int, n_hi: int, rtol: float = 1e-12, max_level: int = 10) -> np.ndarray:
    ns = np.arange(n_lo, n_hi + 1)
    g = gram_points(ns)
    if g[0] < RS_FROM:
        raise ValueError(f"zeros_between works above t = {RS_FROM:g}")
    z = z_values(g)
    good = np.flatnonzero(good_gram(z, n0=n_lo))
    lo, hi, f_lo, f_hi = [], [], [], []
    for j, k in zip(good[:-1], good[1:], strict=True):
        ts, zs = g[j : k + 1], z[j : k + 1]
        level = 0
        while (change := np.signbit(zs[1:]) != np.signbit(zs[:-1])).sum() < k - j:
            level += 1
            if level > max_level:
                raise RuntimeError(f"Gram block [{n_lo + j}, {n_lo + k}) breaks Rosser's rule")
            ts = np.linspace(g[j], g[k], (k - j) * 2**level + 1)
            zs = np.concatenate(([z[j]], z_values(ts[1:-1]), [z[k]]))
        lo.append(ts[:-1][change])
        hi.append(ts[1:][change])
        f_lo.append(zs[:-1][change])
        f_hi.append(zs[1:][change])
    a, b, fa, fb = (np.concatenate(x) for x in (lo, hi, f_lo, f_hi))
    return illinois_many(z_rs, a, b, fa, fb, rtol)


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
