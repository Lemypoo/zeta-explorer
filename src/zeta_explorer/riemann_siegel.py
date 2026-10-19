"""The Riemann-Siegel theta and Z functions, written from scratch.

On the critical line s = 1/2 + it, zeta is complex, which makes its zeros hard to
see. Multiplying by a carefully chosen phase fixes that:

    Z(t) = exp(i * theta(t)) * zeta(1/2 + it)

is *real* for real t, with |Z(t)| = |zeta(1/2 + it)|. So zeros of zeta on the
critical line are exactly the sign changes of Z, and finding them becomes 1-D
root finding.

theta(t) has a fast asymptotic expansion (from Stirling's series for log Gamma):

    theta(t) = t/2 log(t/2pi) - t/2 - pi/8 + 1/(48t) + 7/(5760t^3) + ...

and Z has the Riemann-Siegel formula, with N = floor(sqrt(t/2pi)) and p the
fractional part of sqrt(t/2pi):

    Z(t) = 2 * sum_{n=1}^{N} cos(theta(t) - t log n) / sqrt(n)  +  R(t)
    R(t) ~ (-1)^(N-1) (t/2pi)^(-1/4) * C0(p),   C0(p) = cos(2pi(p^2 - p - 1/16)) / cos(2pi p)

The main sum needs only ~sqrt(t/2pi) terms, which is why this formula made
large-scale zero computations possible. With just the C0 correction the error
shrinks like t^(-3/4); higher corrections (C1..C4) come later.
`z_exact` wraps mpmath as the high-precision reference for testing.
"""

from __future__ import annotations

import math

import mpmath
import numpy as np

TWO_PI = 2 * math.pi


def theta(t: float) -> float:
    """Riemann-Siegel theta function. Accurate to ~1e-10 for t >= 10."""
    if t <= 0:
        raise ValueError("the asymptotic series needs t > 0 (use t >= 10 for accuracy)")
    return (
        t / 2 * math.log(t / TWO_PI)
        - t / 2
        - math.pi / 8
        + 1 / (48 * t)
        + 7 / (5760 * t**3)
        + 31 / (80640 * t**5)
        + 127 / (430080 * t**7)
    )


def theta_prime(t: float) -> float:
    """Derivative of `theta`, term by term."""
    return (
        0.5 * math.log(t / TWO_PI)
        - 1 / (48 * t**2)
        - 7 / (1920 * t**4)
        - 155 / (80640 * t**6)
        - 889 / (430080 * t**8)
    )


def c0(p: float) -> float:
    """First Riemann-Siegel correction coefficient.

    At p = 1/4 and 3/4 numerator and denominator both vanish. The limit there
    (by L'Hopital) is 1/2, so we switch to the differentiated form near them.
    """
    denom = math.cos(TWO_PI * p)
    if abs(denom) > 1e-8:
        return math.cos(TWO_PI * (p * p - p - 1 / 16)) / denom
    return math.sin(TWO_PI * (p * p - p - 1 / 16)) * (2 * p - 1) / math.sin(TWO_PI * p)


def z_rs(t: float) -> float:
    """Riemann-Siegel Z(t) from the main sum plus the C0 correction (t >= 10)."""
    a = math.sqrt(t / TWO_PI)
    n_terms = int(a)
    p = a - n_terms
    th = theta(t)
    n = np.arange(1, n_terms + 1)
    main = 2 * float(np.sum(np.cos(th - t * np.log(n)) / np.sqrt(n)))
    sign = 1 if n_terms % 2 == 1 else -1  # (-1)^(N-1)
    return main + sign * a ** -0.5 * c0(p)


def z_exact(t: float) -> float:
    """High-precision reference Z(t) from mpmath."""
    return float(mpmath.siegelz(t))


def zeta_critical(t: float) -> complex:
    """zeta(1/2 + it) via mpmath, for plots of the real and imaginary parts."""
    return complex(mpmath.zeta(mpmath.mpc(0.5, t)))
