from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from zeta_explorer.riemann_siegel import theta, theta_prime, z_exact

ZFunction = Callable[[float], float]


def illinois(f: ZFunction, a: float, b: float, fa: float, fb: float, tol: float = 1e-12) -> float:
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
            fa /= 2
        b, fb = c, fc
        if abs(b - a) < tol:
            return b
    return b


def find_zeros(
    t_min: float, t_max: float, step: float = 0.1, z: ZFunction = z_exact, tol: float = 1e-10
) -> list[float]:
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
    return float(gram_points([n], tol)[0])


def gram_points(ns: ArrayLike, tol: float = 1e-12) -> np.ndarray:
    ns = np.asarray(ns)
    if np.any(ns < 0):
        raise ValueError("Gram points need n >= 0")
    target = ns * math.pi
    t = np.full(ns.shape, 20.0)
    while np.any(low := theta(t) < target):
        t[low] *= 2
    for _ in range(100):
        dt = (theta(t) - target) / theta_prime(t)
        t -= dt
        if np.all(np.abs(dt) < tol * t):
            return t
    raise RuntimeError("Newton didn't converge for some Gram points")


def illinois_many(
    f: Callable[[np.ndarray], np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
    fa: np.ndarray,
    fb: np.ndarray,
    rtol: float = 1e-12,
    max_iter: int = 100,
    atol: float = 0.0,
) -> np.ndarray:
    a, b, fa, fb = (np.array(x, dtype=float) for x in (a, b, fa, fb))
    if np.any(fa * fb > 0):
        raise ValueError("some roots are not bracketed")
    root = np.where(fa == 0, a, b)
    idx = np.flatnonzero((fa != 0) & (fb != 0))
    a, b, fa, fb = a[idx], b[idx], fa[idx], fb[idx]
    for _ in range(max_iter):
        if len(idx) == 0:
            return root
        c = b - fb * (b - a) / (fb - fa)
        fc = f(c)
        flip = fc * fb < 0
        a = np.where(flip, b, a)
        fa = np.where(flip, fb, fa / 2)
        b, fb = c, fc
        done = (fc == 0) | (np.abs(b - a) < np.maximum(atol, rtol * np.abs(b)))
        root[idx[done]] = b[done]
        live = ~done
        idx, a, b, fa, fb = idx[live], a[live], b[live], fa[live], fb[live]
    raise RuntimeError(f"{len(idx)} roots did not converge")
