"""Riemann-Siegel theta and Z.

Z(t) = exp(i theta(t)) zeta(1/2 + it) is real, so zeros on the critical line are
sign changes of Z. See the README for the formulas.
"""

from __future__ import annotations

import math

import mpmath
import numpy as np

TWO_PI = 2 * math.pi


def theta(t: float) -> float:
    """Stirling-series theta. Good to ~1e-10 once t >= 10."""
    if t <= 0:
        raise ValueError("theta's asymptotic series needs t > 0")
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
    return (
        0.5 * math.log(t / TWO_PI)
        - 1 / (48 * t**2)
        - 7 / (1920 * t**4)
        - 155 / (80640 * t**6)
        - 889 / (430080 * t**8)
    )


def c0(p: float) -> float:
    denom = math.cos(TWO_PI * p)
    if abs(denom) > 1e-8:
        return math.cos(TWO_PI * (p * p - p - 1 / 16)) / denom
    # 0/0 at p = 1/4 and 3/4. L'Hopital gives 1/2 at both.
    return math.sin(TWO_PI * (p * p - p - 1 / 16)) * (2 * p - 1) / math.sin(TWO_PI * p)


def z_rs(t: float) -> float:
    """Riemann-Siegel main sum plus the C0 term. Error ~ 0.03 (2pi/t)^(3/4)."""
    a = math.sqrt(t / TWO_PI)
    N = int(a)
    p = a - N
    th = theta(t)
    n = np.arange(1, N + 1)
    main = 2 * float(np.sum(np.cos(th - t * np.log(n)) / np.sqrt(n)))
    sign = 1 if N % 2 else -1  # (-1)^(N-1)
    return main + sign * a**-0.5 * c0(p)


def z_exact(t: float) -> float:
    """mpmath's Z, used as the reference."""
    return float(mpmath.siegelz(t))


def zeta_critical(t: float) -> complex:
    return complex(mpmath.zeta(mpmath.mpc(0.5, t)))
