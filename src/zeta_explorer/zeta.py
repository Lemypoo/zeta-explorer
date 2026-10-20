"""zeta(s) anywhere in the complex plane, with no mpmath in the loop."""

from __future__ import annotations

import cmath
import functools
import math
from fractions import Fraction


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
