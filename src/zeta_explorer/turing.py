from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from zeta_explorer.gram import good_gram, gram_blocks, sign_changes, z_values
from zeta_explorer.riemann_siegel import theta, z_exact
from zeta_explorer.zeros import gram_points

LEHMAN_A = 1.7
LEHMAN_B = 0.114
T_MIN = 168.0


def blocks_needed(g_p: float) -> int:
    L = math.log(g_p)
    six_pi = 6 * math.pi
    return math.ceil(
        LEHMAN_B / six_pi * L * L + (LEHMAN_A - LEHMAN_B * math.log(2 * math.pi)) / six_pi * L
    )


@dataclass(frozen=True)
class Verification:
    n: int
    g_n: float
    zeros: int
    blocks_after: int


def verify_zeros(T: float) -> Verification:
    T = max(T, T_MIN)
    n0 = math.ceil(theta(T) / math.pi)
    margin = 64
    while True:
        g = gram_points(np.arange(n0 + margin))
        z = z_values(g)
        blocks = gram_blocks(good_gram(z))
        found = _upper_bound_blocks(g, z, blocks, n0)
        if found is not None:
            break
        margin *= 2
    n, used = found

    zeros = sign_changes(10.0, g[0], z_exact(10.0), z[0], need=1)
    for j, k in blocks:
        if k > n:
            break
        zeros += 1 if k - j == 1 else sign_changes(g[j], g[k], z[j], z[k], need=k - j)
    if zeros != n + 1:
        raise RuntimeError(f"found {zeros} sign changes below g_{n}, expected {n + 1}")
    return Verification(n=n, g_n=float(g[n]), zeros=zeros, blocks_after=used)


def _upper_bound_blocks(
    g: np.ndarray, z: np.ndarray, blocks: list[tuple[int, int]], n0: int
) -> tuple[int, int] | None:
    n, run = None, 0
    for j, k in blocks:
        if j < n0:
            continue
        if n is None:
            n = j
        if k - j > 1 and sign_changes(g[j], g[k], z[j], z[k], need=k - j) < k - j:
            n, run = None, 0
            continue
        run += 1
        if run >= blocks_needed(g[k]):
            return n, run
    return None
