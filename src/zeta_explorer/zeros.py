"""Zeros of Z(t) and Gram points."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from zeta_explorer.riemann_siegel import theta, theta_prime, z_exact

ZFunction = Callable[[float], float]


def illinois(f: ZFunction, a: float, b: float, fa: float, fb: float, tol: float = 1e-12) -> float:
    """Root of f in a sign-changing bracket [a, b]."""
    if fa * fb > 0:
        raise ValueError("root is not bracketed")
    for _ in range(200):
        c = b - fb * (b - a) / (fb - fa)
        fc = f(c)
        if fc == 0:
            return c
        if fc * fb < 0:
            a, fa = b, fb
        else:
            # Plain regula falsi can keep the same endpoint forever and crawl.
            # Halving its value pulls the next secant toward the root.
            fa /= 2
        b, fb = c, fc
        if abs(b - a) < tol:
            return b
    return b


def find_zeros(
    t_min: float, t_max: float, step: float = 0.1, z: ZFunction = z_exact, tol: float = 1e-10
) -> list[float]:
    """Zeros of Z in [t_min, t_max], found as sign changes on a grid.

    Two zeros closer than `step` cancel out and get missed. Turing's method
    (not done yet) is what proves the count is complete.
    """
    grid = np.arange(t_min, t_max + step, step)
    zs = [z(t) for t in grid]
    zeros = []
    for i in range(len(grid) - 1):
        if zs[i] == 0:
            zeros.append(float(grid[i]))
        elif zs[i] * zs[i + 1] < 0:
            zeros.append(illinois(z, float(grid[i]), float(grid[i + 1]), zs[i], zs[i + 1], tol))
    return [x for x in zeros if t_min <= x <= t_max]


def gram_point(n: int, tol: float = 1e-12) -> float:
    """g_n, the solution of theta(g_n) = n pi (n >= 0, so g_0 ~ 17.85)."""
    return float(gram_points([n], tol)[0])


def gram_points(ns: ArrayLike, tol: float = 1e-12) -> np.ndarray:
    """Vectorized gram_point."""
    ns = np.asarray(ns)
    if np.any(ns < 0):
        raise ValueError("Gram points need n >= 0")
    target = ns * math.pi
    # theta is increasing and convex out here, so Newton started to the right
    # of the root walks down onto it without overshooting.
    t = np.full(ns.shape, 20.0)
    while np.any(low := theta(t) < target):
        t[low] *= 2
    for _ in range(100):
        dt = (theta(t) - target) / theta_prime(t)
        t -= dt
        if np.all(np.abs(dt) < tol * t):
            return t
    raise RuntimeError("Newton didn't converge for some Gram points")
