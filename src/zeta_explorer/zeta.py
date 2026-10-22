from __future__ import annotations

import functools
import math
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike


@functools.cache
def bernoulli(n: int) -> Fraction:
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return Fraction(1)
    if n == 1:
        return Fraction(-1, 2)
    if n % 2:
        return Fraction(0)
    total = sum(math.comb(n + 1, k) * bernoulli(k) for k in range(n))
    return -total / (n + 1)


_STIRLING = [float(bernoulli(2 * k) / (2 * k * (2 * k - 1))) for k in range(1, 11)]
_HALF_LOG_TWO_PI = 0.5 * math.log(2 * math.pi)
_CHUNK = 1 << 20


def _out(x: np.ndarray, like: ArrayLike) -> complex | np.ndarray:
    return complex(x) if np.ndim(like) == 0 else x


def loggamma(z: ArrayLike) -> complex | np.ndarray:
    w = np.array(z, dtype=complex, ndmin=1)
    if np.any(w.real <= 0):
        raise ValueError("loggamma needs Re z > 0")
    shift = np.zeros_like(w)
    while np.any(small := np.abs(w) < 15):
        shift[small] += np.log(w[small])
        w[small] += 1
    series = np.zeros_like(w)
    inv = 1 / w
    inv2 = inv * inv
    for c in _STIRLING:
        series += c * inv
        inv = inv * inv2
    out = (w - 0.5) * np.log(w) - w + _HALF_LOG_TWO_PI + series - shift
    return _out(out.reshape(np.shape(z)), z)


_M_MAX = 60
_B_OVER_FACT = [float(bernoulli(2 * k) / math.factorial(2 * k)) for k in range(_M_MAX + 2)]


def euler_maclaurin(s: complex, N: int, M: int) -> tuple[complex, float]:
    if N < 1:
        raise ValueError("N must be >= 1")
    if not 0 <= M <= _M_MAX:
        raise ValueError(f"M must be in 0..{_M_MAX}")
    s = np.array([s], dtype=complex)
    if s[0] == 1:
        raise ValueError("zeta has a pole at s = 1")
    if s.real[0] <= -(2 * M + 1):
        raise ValueError("Backlund's bound needs Re s > -(2M + 1)")
    value, bound = _euler_maclaurin(s, np.array([N]), M, tol=0.0)
    return complex(value[0]), float(bound[0])


def _euler_maclaurin(
    s: np.ndarray, N: np.ndarray, M: int, tol: float
) -> tuple[np.ndarray, np.ndarray]:
    value = _dirichlet_head(s, N)
    N_s = N ** (-s)
    value += N * N_s / (s - 1) + N_s / 2
    bound = np.zeros(len(s))
    idx = np.arange(len(s))
    poch = s.copy()
    power = N_s / N
    for k in range(1, M + 2):
        term = _B_OVER_FACT[k] * poch * power
        sk = s[idx] + 2 * k - 1
        bound[idx] = np.abs(sk) / sk.real * np.abs(term)
        if k > M:
            break
        keep = bound[idx] > tol * np.maximum(1.0, np.abs(value[idx]))
        idx, term, poch, power = idx[keep], term[keep], poch[keep], power[keep]
        if len(idx) == 0:
            break
        value[idx] += term
        poch *= (s[idx] + 2 * k - 1) * (s[idx] + 2 * k)
        power /= N[idx] ** 2
    return value, bound


def _dirichlet_head(s: np.ndarray, N: np.ndarray) -> np.ndarray:
    out = np.zeros(len(s), dtype=complex)
    if len(s) == 0 or N.max() < 2:
        return out
    n = np.arange(1, int(N.max()))
    log_n = np.log(n)
    rows = max(1, _CHUNK // len(n))
    for i in range(0, len(s), rows):
        sl = slice(i, i + rows)
        terms = np.exp(-s[sl, None] * log_n)
        terms[n >= N[sl, None]] = 0
        out[sl] = terms.sum(axis=1)
    return out


def zeta(s: ArrayLike, tol: float = 1e-16) -> complex | np.ndarray:
    z = np.array(s, dtype=complex, ndmin=1).ravel()
    if np.any(z == 1):
        raise ValueError("zeta has a pole at s = 1")
    out = np.zeros_like(z)
    left = z.real < 0
    trivial = left & (z.imag == 0) & (z.real % 2 == 0)
    right = ~left
    out[right] = _zeta_right(z[right], tol)
    reflect = left & ~trivial
    if np.any(reflect):
        sr = z[reflect]
        w = chi(sr) * _zeta_right(1 - sr, tol)
        out[reflect] = np.where(sr.imag == 0, w.real, w)
    return _out(out.reshape(np.shape(s)), s)


def _zeta_right(s: np.ndarray, tol: float) -> np.ndarray:
    N = 10 + (np.abs(s.imag) / math.pi).astype(np.int64)
    return _euler_maclaurin(s, N, _M_MAX, tol)[0]


def chi(s: ArrayLike) -> complex | np.ndarray:
    z = np.array(s, dtype=complex, ndmin=1)
    if np.any(z.real >= 1):
        raise ValueError("chi is implemented for Re s < 1")
    log_chi = z * math.log(2) + (z - 1) * math.log(math.pi)
    out = np.exp(log_chi + _log_sin(math.pi * z / 2) + loggamma(1 - z))
    return _out(out.reshape(np.shape(s)), s)


def _log_sin(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    up = z.imag >= 0
    u, d = z[up], z[~up]
    out[up] = -1j * u + np.log((np.exp(2j * u) - 1) / 2j)
    out[~up] = 1j * d + np.log((1 - np.exp(-2j * d)) / 2j)
    return out


def xi(s: ArrayLike) -> complex | np.ndarray:
    z = np.array(s, dtype=complex, ndmin=1).ravel()
    out = np.empty_like(z)
    right = z.real >= 0.5
    for mask, w in ((right, z[right]), (~right, 1 - z[~right])):
        out[mask] = _xi_right(w)
    return _out(out.reshape(np.shape(s)), s)


def _xi_right(s: np.ndarray) -> np.ndarray:
    out = np.full(len(s), 0.5 + 0j)
    ok = s != 1
    w = s[ok]
    log_pre = np.log(w / 2) - w / 2 * math.log(math.pi) + loggamma(w / 2)
    out[ok] = (w - 1) * np.exp(log_pre) * zeta(w)
    return out
