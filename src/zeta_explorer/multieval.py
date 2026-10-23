from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from zeta_explorer.riemann_siegel import TWO_PI

_CHUNK = 1 << 20


def nufft(x: ArrayLike, c: ArrayLike, m: int, tol: float = 1e-13) -> np.ndarray:
    if m < 2 or m % 2:
        raise ValueError("m must be even and positive")
    x = np.mod(np.asarray(x, dtype=float), TWO_PI)
    c = np.asarray(c, dtype=complex)
    mr = 2 * m
    sp = math.ceil(1.5 * math.log(1 / tol) / math.pi)
    tau = math.pi * sp / (3 * m * m)
    h = TWO_PI / mr
    offsets = np.arange(-sp, sp + 1)
    grid = np.zeros(mr, dtype=complex)
    rows = max(1, _CHUNK // len(offsets))
    for i in range(0, len(x), rows):
        xs, cs = x[i : i + rows], c[i : i + rows]
        idx = np.rint(xs / h).astype(np.int64)[:, None] + offsets
        w = (np.exp(-((idx * h - xs[:, None]) ** 2) / (4 * tau)) * cs[:, None]).ravel()
        idx = (idx % mr).ravel()
        grid += np.bincount(idx, w.real, mr) + 1j * np.bincount(idx, w.imag, mr)
    k = np.arange(-m // 2, m // 2)
    return np.fft.fft(grid)[k % mr] * np.exp(tau * k**2) / (mr * math.sqrt(tau / math.pi))


def fft_size(n: int) -> int:
    best = 2 ** math.ceil(math.log2(n))
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            p = p35
            while p < n:
                p *= 2
            best = min(best, p)
            p35 *= 3
        p5 *= 5
    return best
