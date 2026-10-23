from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from zeta_explorer.zeta import zeta

EULER_GAMMA = 0.5772156649015329
LOG_2PI = math.log(2 * math.pi)

_SERIES_TERMS = 160
_ASYMPTOTIC_FROM = 40.0
_ASYMPTOTIC_TERMS = 34
_CF_DEPTH = ((6.0, 140), (10.0, 90), (20.0, 50), (_ASYMPTOTIC_FROM, 20))
_CHUNK = 1 << 18


def ei(z: ArrayLike) -> complex | np.ndarray:
    w = np.array(z, dtype=complex, ndmin=1).ravel()
    r = np.abs(w)
    real = (w.imag == 0) & (w.real > 0)
    near_real = (np.abs(w.imag) < w.real) & ~real
    if np.any(w == 0):
        raise ValueError("Ei has a logarithmic singularity at 0")
    if np.any(near_real & (r >= 4) & (r < _ASYMPTOTIC_FROM)):
        raise ValueError("ei is not implemented for 4 <= |w| < 40 within pi/4 of the real axis")
    out = np.empty_like(w)
    far = r >= _ASYMPTOTIC_FROM
    series = ~far & ((r < 4) | real)
    cf = ~far & ~series
    out[far] = _ei_asymptotic(w[far])
    out[series] = _ei_series(w[series])
    out[cf] = _ei_cf(w[cf], r[cf])
    out[w.imag == 0] = out[w.imag == 0].real
    return complex(out[0]) if np.ndim(z) == 0 else out.reshape(np.shape(z))


def _ei_series(w: np.ndarray) -> np.ndarray:
    total = np.zeros_like(w)
    term = np.ones_like(w)
    for k in range(1, _SERIES_TERMS + 1):
        term = term * w / k
        total += term / k
    return EULER_GAMMA + np.log(w) + total


def _ei_asymptotic(w: np.ndarray) -> np.ndarray:
    total = np.ones_like(w)
    term = np.ones_like(w)
    for k in range(1, _ASYMPTOTIC_TERMS + 1):
        term = term * k / w
        total += term
    return np.exp(w) / w * total + 1j * math.pi * np.sign(w.imag)


def _ei_cf(w: np.ndarray, r: np.ndarray) -> np.ndarray:
    z = -w
    f = np.empty_like(z)
    lo = 0.0
    for hi, depth in _CF_DEPTH:
        band = (r >= lo) & (r < hi)
        zb = z[band]
        fb = zb + 2 * depth + 1
        for k in range(depth, 0, -1):
            fb = zb + 2 * k - 1 - k * k / fb
        f[band] = fb
        lo = hi
    return -np.exp(-z) / f + 1j * math.pi * np.sign(w.imag)


def li(x: ArrayLike) -> float | np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0) or np.any(x == 1):
        raise ValueError("li needs x > 0, x != 1")
    out = np.real(ei(np.log(x)))
    return float(out) if np.ndim(out) == 0 else out


def riemann_r(x: ArrayLike) -> float | np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("R needs x > 0")
    L = np.log(x)
    K = int(3 * np.abs(L).max()) + 60
    inv_zeta = 1 / zeta(np.arange(2.0, K + 2)).real
    total = np.ones_like(L)
    term = np.ones_like(L)
    for k in range(1, K + 1):
        term = term * L / k
        total += term * inv_zeta[k - 1] / k
    return float(total) if total.ndim == 0 else total


def _sieve(n: int) -> np.ndarray:
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[:2] = False
    for p in range(2, math.isqrt(n) + 1):
        if is_prime[p]:
            is_prime[p * p :: p] = False
    return is_prime


def prime_pi(x: ArrayLike) -> np.ndarray:
    n = np.floor(np.asarray(x, dtype=float)).astype(np.int64)
    counts = np.cumsum(_sieve(max(int(n.max()), 1)))
    return np.where(n >= 0, counts[np.maximum(n, 0)], 0)


def chebyshev_psi(x: ArrayLike) -> np.ndarray:
    n = np.floor(np.asarray(x, dtype=float)).astype(np.int64)
    top = max(int(n.max()), 1)
    mangoldt = np.zeros(top + 1)
    for p in np.flatnonzero(_sieve(top)):
        q = p
        while q <= top:
            mangoldt[q] = math.log(p)
            q *= p
    return np.cumsum(mangoldt)[np.maximum(n, 0)]


def mobius(n: int) -> int:
    sign = 1
    p = 2
    while p * p <= n:
        if n % p == 0:
            n //= p
            if n % p == 0:
                return 0
            sign = -sign
        p += 1
    return -sign if n > 1 else sign


def psi_from_zeros(x: ArrayLike, gamma: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.any(x <= 1):
        raise ValueError("the explicit formula for psi needs x > 1")
    rho = 0.5 + 1j * np.asarray(gamma, dtype=float)
    log_x = np.log(x).ravel()
    sums = np.empty_like(log_x)
    rows = max(1, _CHUNK // max(len(rho), 1))
    for i in range(0, len(log_x), rows):
        s = slice(i, i + rows)
        sums[s] = (np.exp(log_x[s, None] * rho) / rho).sum(axis=1).real
    return x - 2 * sums.reshape(x.shape) - LOG_2PI - 0.5 * np.log1p(-(x**-2))


def _tail_integral(y: np.ndarray, nodes: int = 40) -> np.ndarray:
    w, c = np.polynomial.laguerre.laggauss(nodes)
    a = np.log(y)[..., None]
    return (c / (2 * (np.exp(2 * a) - np.exp(-w)) * (a + w / 2))).sum(axis=-1)


def riemann_j(x: ArrayLike, gamma: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.any(x <= 1):
        raise ValueError("J needs x > 1")
    rho = 0.5 + 1j * np.asarray(gamma, dtype=float)
    log_x = np.log(x).ravel()
    sums = np.empty_like(log_x)
    rows = max(1, _CHUNK // max(len(rho), 1))
    for i in range(0, len(log_x), rows):
        s = slice(i, i + rows)
        sums[s] = ei(log_x[s, None] * rho).sum(axis=1).real
    out = li(x) - 2 * sums.reshape(x.shape) - math.log(2) + _tail_integral(x)
    return np.asarray(out)


def pi_from_zeros(x: ArrayLike, gamma: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros(x.shape)
    n = 1
    while np.any(live := x >= 2.0**n):
        mu = mobius(n)
        if mu:
            out[live] += mu / n * riemann_j(x[live] ** (1 / n), gamma)
        n += 1
    return out
