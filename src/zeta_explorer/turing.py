"""Turing's method: prove that no zeros were missed below a height T.

Sign changes of Z only give a lower bound on N(T), the number of zeros with
0 < Im s <= T. Turing's idea is that S(t) = N(t) - theta(t)/pi - 1 averages to
zero, so a surplus of zeros below g_n would push the Gram points above it out
of line for longer than the integral bound on S allows. Brent (1979) turned
that into a block count, with Lehman's constants for the integral bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from zeta_explorer.gram import good_gram, gram_blocks, sign_changes, z_values
from zeta_explorer.riemann_siegel import theta, z_exact
from zeta_explorer.zeros import gram_points

# |int_{t1}^{t2} S(t) dt| <= a + b log t2 for t2 > t1 > 168 (Lehman 1970).
LEHMAN_A = 1.7
LEHMAN_B = 0.114
T_MIN = 168.0


def blocks_needed(g_p: float) -> int:
    """Gram blocks that must obey Rosser's rule, ending at g_p, for Brent's theorem.

    K >= b/(6 pi) log^2 g_p + (a - b log 2pi)/(6 pi) log g_p, which Trudgian
    (2011) rounds to 0.0061 log^2 g_p + 0.08 log g_p.
    """
    L = math.log(g_p)
    six_pi = 6 * math.pi
    return math.ceil(
        LEHMAN_B / six_pi * L * L + (LEHMAN_A - LEHMAN_B * math.log(2 * math.pi)) / six_pi * L
    )


@dataclass(frozen=True)
class Verification:
    n: int  # zeros are verified up to the Gram point g_n
    g_n: float
    zeros: int  # sign changes of Z below g_n; equals N(g_n) = n + 1
    blocks_after: int  # Rosser blocks past g_n used for the upper bound


def verify_zeros(T: float) -> Verification:
    """Prove that every zero with 0 < Im s <= T is on the critical line and simple.

    Finds a good Gram point g_n >= T, counts n + 1 sign changes of Z below it
    (lower bound), and finds enough Rosser blocks past it for Brent's theorem
    to give N(g_n) <= n + 1 (upper bound). Raises if the bounds don't meet.
    """
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
    """(n, K): K Rosser blocks starting at the good point g_n, n >= n0, enough for Brent."""
    n, run = None, 0
    for j, k in blocks:
        if j < n0:
            continue
        if n is None:
            n = j
        if k - j > 1 and sign_changes(g[j], g[k], z[j], z[k], need=k - j) < k - j:
            n, run = None, 0  # Rosser fails here; restart past this block
            continue
        run += 1
        if run >= blocks_needed(g[k]):
            return n, run
    return None
