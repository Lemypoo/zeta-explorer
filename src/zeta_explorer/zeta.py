from __future__ import annotations

import cmath
import functools
import math
from fractions import Fraction

import numpy as np


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


def loggamma(z: complex) -> complex:
    z = complex(z)
    if z.real <= 0:
        raise ValueError("loggamma needs Re z > 0")
    shift = 0j
    while abs(z) < 15:
        shift += cmath.log(z)
        z += 1
    series = 0j
    w = 1 / z
    w2 = w * w
    for c in _STIRLING:
        series += c * w
        w *= w2
    return (z - 0.5) * cmath.log(z) - z + _HALF_LOG_TWO_PI + series - shift


_M_MAX = 60
_B_OVER_FACT = [float(bernoulli(2 * k) / math.factorial(2 * k)) for k in range(_M_MAX + 2)]


def euler_maclaurin(s: complex, N: int, M: int) -> tuple[complex, float]:
    if N < 1:
        raise ValueError("N must be >= 1")
    if not 0 <= M <= _M_MAX:
        raise ValueError(f"M must be in 0..{_M_MAX}")
    return _euler_maclaurin(complex(s), N, M, tol=0.0)


def _euler_maclaurin(s: complex, N: int, M: int, tol: float) -> tuple[complex, float]:
    if s == 1:
        raise ValueError("zeta has a pole at s = 1")
    if s.real <= -(2 * M + 1):
        raise ValueError("Backlund's bound needs Re s > -(2M + 1)")
    n = np.arange(1, N)
    value = complex(np.exp(-s * np.log(n)).sum())
    N_s = N ** (-s)
    value += N * N_s / (s - 1) + N_s / 2
    poch = s
    power = N_s / N
    for k in range(1, M + 2):
        term = _B_OVER_FACT[k] * poch * power
        bound = abs(s + 2 * k - 1) / (s.real + 2 * k - 1) * abs(term)
        if k > M or bound <= tol * max(1.0, abs(value)):
            break
        value += term
        poch *= (s + 2 * k - 1) * (s + 2 * k)
        power /= N * N
    return value, bound


def zeta(s: complex, tol: float = 1e-16) -> complex:
    s = complex(s)
    if s.real < 0:
        if s.imag == 0:
            if s.real % 2 == 0:
                return 0j
            return complex((chi(s) * zeta(1 - s, tol)).real)
        return chi(s) * zeta(1 - s, tol)
    N = 10 + int(abs(s.imag) / math.pi)
    return _euler_maclaurin(s, N, _M_MAX, tol)[0]


def chi(s: complex) -> complex:
    s = complex(s)
    if s.real >= 1:
        raise ValueError("chi is implemented for Re s < 1")
    log_chi = s * math.log(2) + (s - 1) * math.log(math.pi)
    return cmath.exp(log_chi + _log_sin(math.pi * s / 2) + loggamma(1 - s))


def _log_sin(z: complex) -> complex:
    if z.imag >= 0:
        return -1j * z + cmath.log((cmath.exp(2j * z) - 1) / 2j)
    return 1j * z + cmath.log((1 - cmath.exp(-2j * z)) / 2j)
