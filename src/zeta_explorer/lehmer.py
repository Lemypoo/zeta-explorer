from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from zeta_explorer.spacing import normalized_spacings

TWO_PI = 2 * math.pi
T0 = 9.676906787165866
LEHMER_THRESHOLD = 0.8


def smooth_count(t: ArrayLike) -> np.ndarray:
    u = np.asarray(t, dtype=float) / TWO_PI
    return u * (np.log(u) - 1) + 7 / 8


def _smooth_sum(c: np.ndarray, lo: ArrayLike, hi: ArrayLike) -> np.ndarray:
    def G(t):
        t = np.asarray(t, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            g = -np.log(t / TWO_PI) / (t - c) + np.log(np.abs((t - c) / t)) / c
        return np.where(np.isinf(t), 0.0, g)

    return (G(hi) - G(lo)) / TWO_PI


def _half_gap(t: np.ndarray) -> np.ndarray:
    return math.pi / np.log(t / TWO_PI)


def lehmer_sums(gamma: ArrayLike, n0: int = 1, window: int = 500) -> np.ndarray:
    gamma = np.asarray(gamma, dtype=float)
    n = len(gamma)
    if n < 2:
        raise ValueError("need at least two zeros")
    k = np.arange(n - 1)
    a, b = gamma[:-1], gamma[1:]
    g = np.zeros(n - 1)
    for m in range(1, window + 1):
        for j in (k + 1 + m, k - m):
            ok = (j >= 0) & (j < n)
            x = gamma[j[ok]]
            g[ok] += 1 / (x - a[ok]) ** 2 + 1 / (x - b[ok]) ** 2

    right = gamma[np.minimum(k + 1 + window, n - 1)]
    right = right + _half_gap(right)
    g += _smooth_sum(a, right, np.inf) + _smooth_sum(b, right, np.inf)
    first = np.maximum(k - window, 0)
    left = gamma[first] - _half_gap(gamma[first])
    has_left = (k - window > 0) | (n0 > 1)
    g += np.where(has_left, _smooth_sum(a, T0, left) + _smooth_sum(b, T0, left), 0.0)

    mirrored = gamma[:window]
    for x in mirrored:
        g += 1 / (x + a) ** 2 + 1 / (x + b) ** 2
    beyond = mirrored[-1] + _half_gap(mirrored[-1])
    g += _smooth_sum(-a, beyond, np.inf) + _smooth_sum(-b, beyond, np.inf)
    if n0 > 1:
        below = gamma[0] - _half_gap(gamma[0])
        g += _smooth_sum(-a, T0, below) + _smooth_sum(-b, T0, below)
    return g


def newman_bound(delta: ArrayLike, g: ArrayLike) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    g = np.asarray(g, dtype=float)
    score = delta**2 * g
    with np.errstate(invalid="ignore"):
        lam = ((1 - 1.25 * score) ** 0.8 - 1) / (2 * g)
    return np.where(score < LEHMER_THRESHOLD, lam, np.nan)


@dataclass(frozen=True)
class LehmerPair:
    n: int
    gamma: float
    delta: float
    gap: float
    score: float
    bound: float


def lehmer_pairs(
    gamma: ArrayLike, n0: int = 1, window: int = 500, threshold: float = LEHMER_THRESHOLD
) -> list[LehmerPair]:
    gamma = np.asarray(gamma, dtype=float)
    delta = np.diff(gamma)
    g = lehmer_sums(gamma, n0, window)
    score = delta**2 * g
    bound = newman_bound(delta, g)
    gap = normalized_spacings(gamma)
    return [
        LehmerPair(n0 + int(i), float(gamma[i]), float(delta[i]), float(gap[i]),
                   float(score[i]), float(bound[i]))
        for i in np.flatnonzero(score < threshold)
    ]
