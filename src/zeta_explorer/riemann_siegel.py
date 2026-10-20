"""Riemann-Siegel theta and Z.

Z(t) = exp(i theta(t)) zeta(1/2 + it) is real, so zeros on the critical line are
sign changes of Z. See the README for the formulas.
"""

from __future__ import annotations

import math

import mpmath
import numpy as np
from numpy.typing import ArrayLike

TWO_PI = 2 * math.pi

# Cap on the size of the (t, n) phase matrix built per chunk in the main sum.
_CHUNK = 1 << 20


def theta(t: ArrayLike) -> float | np.ndarray:
    """Stirling-series theta. Good to ~1e-10 once t >= 10."""
    t = np.asarray(t, dtype=float)
    if np.any(t <= 0):
        raise ValueError("theta's asymptotic series needs t > 0")
    out = (
        t / 2 * np.log(t / TWO_PI)
        - t / 2
        - math.pi / 8
        + 1 / (48 * t)
        + 7 / (5760 * t**3)
        + 31 / (80640 * t**5)
        + 127 / (430080 * t**7)
    )
    return float(out) if out.ndim == 0 else out


def theta_prime(t: ArrayLike) -> float | np.ndarray:
    t = np.asarray(t, dtype=float)
    out = (
        0.5 * np.log(t / TWO_PI)
        - 1 / (48 * t**2)
        - 7 / (1920 * t**4)
        - 155 / (80640 * t**6)
        - 889 / (430080 * t**8)
    )
    return float(out) if out.ndim == 0 else out


def z_rs(t: ArrayLike) -> float | np.ndarray:
    """Riemann-Siegel main sum plus the C0 term. Error ~ 0.03 (2pi/t)^(3/4).

    Accepts a scalar or an array of t.
    """
    ts = np.atleast_1d(np.asarray(t, dtype=float))
    a = np.sqrt(ts / TWO_PI)
    N = a.astype(np.int64)
    p = a - N
    sign = np.where(N % 2 == 1, 1.0, -1.0)  # (-1)^(N-1)
    out = _main_sum(ts, N) + sign / np.sqrt(a) * c0(p)
    return float(out[0]) if np.ndim(t) == 0 else out


def _main_sum(t: np.ndarray, N: np.ndarray) -> np.ndarray:
    """2 sum_{n <= N} cos(theta(t) - t log n) / sqrt(n), one row per t."""
    out = np.zeros_like(t)
    if len(t) == 0 or N.max() < 1:
        return out
    th = theta(t)
    n = np.arange(1, int(N.max()) + 1)
    log_n = np.log(n)
    inv_sqrt_n = 1 / np.sqrt(n)
    rows = max(1, _CHUNK // len(n))
    for i in range(0, len(t), rows):
        s = slice(i, i + rows)
        terms = np.cos(th[s, None] - t[s, None] * log_n) * inv_sqrt_n
        terms[n > N[s, None]] = 0
        out[s] = 2 * terms.sum(axis=1)
    return out


def c0(p: ArrayLike) -> float | np.ndarray:
    p = np.asarray(p, dtype=float)
    num = np.cos(TWO_PI * (p * p - p - 1 / 16))
    den = np.cos(TWO_PI * p)
    near = np.abs(den) < 1e-8
    # 0/0 at p = 1/4 and 3/4. L'Hopital gives 1/2 at both.
    with np.errstate(divide="ignore", invalid="ignore"):
        lhopital = np.sin(TWO_PI * (p * p - p - 1 / 16)) * (2 * p - 1) / np.sin(TWO_PI * p)
    out = np.where(near, lhopital, num / np.where(near, 1.0, den))
    return float(out) if out.ndim == 0 else out


def z_exact(t: float) -> float:
    """mpmath's Z, used as the reference."""
    return float(mpmath.siegelz(t))


def zeta_critical(t: float) -> complex:
    return complex(mpmath.zeta(mpmath.mpc(0.5, t)))
