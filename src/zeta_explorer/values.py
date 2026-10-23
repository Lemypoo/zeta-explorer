from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from zeta_explorer.gram import z_values
from zeta_explorer.riemann_siegel import z_rs
from zeta_explorer.zeta import loggamma, zeta

INV_PHI = (math.sqrt(5) - 1) / 2


def z_extrema(gamma: ArrayLike, iters: int = 40) -> tuple[np.ndarray, np.ndarray]:
    gamma = np.asarray(gamma, dtype=float)
    a, b = gamma[:-1].copy(), gamma[1:].copy()
    c = b - INV_PHI * (b - a)
    d = a + INV_PHI * (b - a)
    fc, fd = np.abs(z_values(c)), np.abs(z_values(d))
    for _ in range(iters):
        left = fc > fd
        b = np.where(left, d, b)
        a = np.where(left, a, c)
        c, d = np.where(left, b - INV_PHI * (b - a), d), np.where(left, c, a + INV_PHI * (b - a))
        f_new = np.abs(z_values(np.where(left, c, d)))
        fc, fd = np.where(left, f_new, fd), np.where(left, fc, f_new)
    t = (a + b) / 2
    return t, z_values(t)


def records(values: ArrayLike) -> np.ndarray:
    v = np.abs(np.asarray(values, dtype=float))
    best = np.maximum.accumulate(v)
    return np.flatnonzero(np.concatenate(([True], best[1:] > best[:-1])))


def log_z_sample(T: float, n: int, rng: np.random.Generator, width: float = 0.01) -> np.ndarray:
    t = rng.uniform(T, T * (1 + width), n)
    return np.log(np.abs(z_rs(t)))


def sample_cumulants(x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    c = x - x.mean()
    k2 = np.mean(c**2)
    return np.array([x.mean(), k2, np.mean(c**3), np.mean(c**4) - 3 * k2**2])


def _polygamma_sum(m: int, N: int) -> float:
    zeta_m = float(zeta(m + 1).real)
    j = np.arange(1, N + 1)
    head = np.concatenate(([0.0], np.cumsum(1.0 / j[:-1] ** (m + 1))))
    return (-1) ** (m + 1) * math.factorial(m) * float(np.sum(zeta_m - head))


def cue_cumulants(N: int) -> np.ndarray:
    return np.array([0.0] + [(1 - 2.0 ** (1 - k)) * _polygamma_sum(k - 1, N) for k in (2, 3, 4)])


def cue_log_mgf(u: ArrayLike, N: int) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    out = np.zeros(u.shape, dtype=complex)
    for j in range(1, N + 1):
        out += loggamma(j + 1j * u) - 2 * loggamma(j + 0.5j * u) + math.lgamma(j)
    return out


def cue_density(x: ArrayLike, N: int, u_max: float = 40.0, h: float = 0.01) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    u = np.arange(0, u_max + h, h)
    M = np.exp(cue_log_mgf(u, N))
    w = np.full(u.shape, h)
    w[0] = w[-1] = h / 2
    phase = np.exp(-1j * np.multiply.outer(x, u))
    return (phase * (w * M)).real.sum(axis=-1) / math.pi
