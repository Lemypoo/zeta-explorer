"""zeta(s) anywhere in the complex plane, with no mpmath in the loop."""

from __future__ import annotations

import cmath
import functools
import math
from fractions import Fraction

import numpy as np


@functools.cache
def bernoulli(n: int) -> Fraction:
    """B_n exactly, with B_1 = -1/2."""
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return Fraction(1)
    if n == 1:
        return Fraction(-1, 2)
    if n % 2:
        return Fraction(0)
    # sum_{k=0}^{n} C(n+1, k) B_k = 0, solved for B_n
    total = sum(math.comb(n + 1, k) * bernoulli(k) for k in range(n))
    return -total / (n + 1)


# Stirling with these 10 terms is good to well below 1e-16 once |z| >= 15.
_STIRLING = [float(bernoulli(2 * k) / (2 * k * (2 * k - 1))) for k in range(1, 11)]
_HALF_LOG_TWO_PI = 0.5 * math.log(2 * math.pi)


def loggamma(z: complex) -> complex:
    """log Gamma(z) for Re z > 0, continuous in z (not the principal log of Gamma)."""
    z = complex(z)
    if z.real <= 0:
        raise ValueError("loggamma needs Re z > 0")
    # Gamma(z) = Gamma(z + m) / (z (z+1) ... (z+m-1)). Summing logs instead of
    # taking the log of the product keeps the imaginary part on the right branch.
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
# B_2k / (2k)!, k = 0.._M_MAX + 1
_B_OVER_FACT = [float(bernoulli(2 * k) / math.factorial(2 * k)) for k in range(_M_MAX + 2)]


def euler_maclaurin(s: complex, N: int, M: int) -> tuple[complex, float]:
    """Euler-Maclaurin with N - 1 direct terms and M Bernoulli corrections.

    Returns (value, bound) where bound is Backlund's rigorous bound on the
    truncation error, valid for Re s > -(2M + 1).
    """
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
    # T_k = B_2k / (2k)! * s (s+1) ... (s+2k-2) * N^(-s-2k+1)
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
    """Riemann zeta at any s != 1.

    Euler-Maclaurin for Re s >= 0, the functional equation for Re s < 0.
    """
    s = complex(s)
    if s.real < 0:
        if s.imag == 0:
            if s.real % 2 == 0:
                return 0j  # trivial zero
            # chi is assembled from complex logs, so drop its rounding-level imaginary part
            return complex((chi(s) * zeta(1 - s, tol)).real)
        return chi(s) * zeta(1 - s, tol)
    # N > |t| / pi keeps |t| / (2 pi N) below 1/2, so the Bernoulli terms fall off fast.
    N = 10 + int(abs(s.imag) / math.pi)
    return _euler_maclaurin(s, N, _M_MAX, tol)[0]


def chi(s: complex) -> complex:
    """The factor in zeta(s) = chi(s) zeta(1 - s), for Re s < 1.

    chi(s) = 2^s pi^(s-1) sin(pi s / 2) Gamma(1 - s), assembled in log space:
    for large |t| the sine and Gamma factors overflow and underflow separately.
    """
    s = complex(s)
    if s.real >= 1:
        raise ValueError("chi is implemented for Re s < 1")
    log_chi = s * math.log(2) + (s - 1) * math.log(math.pi)
    return cmath.exp(log_chi + _log_sin(math.pi * s / 2) + loggamma(1 - s))


def _log_sin(z: complex) -> complex:
    """A log of sin z (any branch) that never overflows."""
    # sin z = e^(-iz) (e^(2iz) - 1) / 2i; pull out whichever exponential is large.
    if z.imag >= 0:
        return -1j * z + cmath.log((cmath.exp(2j * z) - 1) / 2j)
    return 1j * z + cmath.log((1 - cmath.exp(-2j * z)) / 2j)
