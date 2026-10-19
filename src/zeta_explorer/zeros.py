"""Finding zeros of Z(t) and computing Gram points.

Zeros: scan a grid for sign changes of Z, then refine each bracket with the
Illinois method (regula falsi that halves the weight of a stuck endpoint, so it
keeps superlinear convergence where plain regula falsi can stall). A sign-change
scan can miss two zeros closer together than the grid step; Turing's method
(not done yet) will prove none were missed.

Gram points: g_n solves theta(g_n) = n*pi. Between consecutive Gram points Z
usually changes sign exactly once ("Gram's law"), which makes them the natural
grid for counting zeros. theta is increasing and convex for t > 7, so Newton's
method started to the right of the root converges monotonically down to it.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from zeta_explorer.riemann_siegel import theta, theta_prime, z_exact

ZFunction = Callable[[float], float]


def illinois(f: ZFunction, a: float, b: float, fa: float, fb: float, tol: float = 1e-12) -> float:
    """Root of f in [a, b] where f(a) and f(b) have opposite signs."""
    if fa * fb > 0:
        raise ValueError("root is not bracketed")
    for _ in range(200):
        c = b - fb * (b - a) / (fb - fa)  # where the secant through both ends crosses 0
        fc = f(c)
        if fc == 0:
            return c
        if fc * fb < 0:
            a, fa = b, fb  # root is between b and c: old b becomes the other end
        else:
            fa /= 2  # a survives another round: halve its weight so it can't stall us
        b, fb = c, fc
        if abs(b - a) < tol:
            return b
    return b


def find_zeros(
    t_min: float, t_max: float, step: float = 0.1, z: ZFunction = z_exact, tol: float = 1e-10
) -> list[float]:
    """All zeros of Z in [t_min, t_max] detected as sign changes on a grid of `step`."""
    grid = np.arange(t_min, t_max + step, step)
    values = [z(t) for t in grid]
    zeros = []
    for i in range(len(grid) - 1):
        z0, z1 = values[i], values[i + 1]
        if z0 == 0:
            zeros.append(float(grid[i]))
        elif z0 * z1 < 0:
            zeros.append(illinois(z, float(grid[i]), float(grid[i + 1]), z0, z1, tol))
    return [x for x in zeros if t_min <= x <= t_max]


def gram_point(n: int, tol: float = 1e-12) -> float:
    """The n-th Gram point (n >= 0): the t where theta(t) = n*pi."""
    if n < 0:
        raise ValueError("Gram points here start at n = 0 (g_0 ~ 17.85)")
    target = n * math.pi
    t = 20.0
    while theta(t) < target:  # get to the right of the root
        t *= 2
    for _ in range(100):
        step = (theta(t) - target) / theta_prime(t)
        t -= step
        if abs(step) < tol * t:
            return t
    raise RuntimeError(f"Newton did not converge for n={n}")
