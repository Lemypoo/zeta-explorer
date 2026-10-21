from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from zeta_explorer.riemann_siegel import z_exact, z_rs
from zeta_explorer.zeros import gram_points

RS_FROM = 200.0
SIGN_GUARD = 1e-6


def z_values(ts: ArrayLike) -> np.ndarray:
    ts = np.asarray(ts, dtype=float)
    z = np.empty_like(ts)
    low = ts < RS_FROM
    z[low] = [z_exact(t) for t in ts[low]]
    if np.any(~low):
        z[~low] = z_rs(ts[~low])
    close = ~low & (np.abs(z) < SIGN_GUARD)
    z[close] = [z_exact(t) for t in ts[close]]
    return z


def good_gram(z_at_gram: np.ndarray, n0: int = 0) -> np.ndarray:
    sign = np.where((n0 + np.arange(len(z_at_gram))) % 2 == 0, 1.0, -1.0)
    return sign * z_at_gram > 0


def gram_blocks(good: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(good)
    return list(zip(idx[:-1].tolist(), idx[1:].tolist(), strict=True))


def sign_changes(a: float, b: float, za: float, zb: float, need: int, max_level: int = 8) -> int:
    pieces = max(need, 1)
    for _ in range(max_level):
        pieces *= 2
        inner = z_values(np.linspace(a, b, pieces + 1)[1:-1])
        zs = np.concatenate(([za], inner, [zb]))
        found = int(np.count_nonzero(np.signbit(zs[1:]) != np.signbit(zs[:-1])))
        if found >= need:
            return found
    return found


@dataclass(frozen=True)
class GramStats:
    n_points: int
    t_max: float
    good: int
    block_lengths: Counter[int]
    first_bad: int | None
    rosser_violations: list[tuple[int, int]]

    @property
    def good_fraction(self) -> float:
        return self.good / self.n_points


def gram_stats(n_max: int) -> GramStats:
    g = gram_points(np.arange(n_max + 1))
    z = z_values(g)
    good = good_gram(z)
    blocks = gram_blocks(good)
    violations = [
        (j, k) for j, k in blocks
        if k - j > 1 and sign_changes(g[j], g[k], z[j], z[k], need=k - j) < k - j
    ]
    bad = np.flatnonzero(~good)
    return GramStats(
        n_points=n_max + 1,
        t_max=float(g[-1]),
        good=int(good.sum()),
        block_lengths=Counter(k - j for j, k in blocks),
        first_bad=int(bad[0]) if len(bad) else None,
        rosser_violations=violations,
    )
