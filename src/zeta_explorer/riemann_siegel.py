from __future__ import annotations

import functools
import math

import mpmath
import numpy as np
from numpy.typing import ArrayLike

TWO_PI = 2 * math.pi

_CHUNK = 1 << 20


def theta(t: ArrayLike) -> float | np.ndarray:
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


def z_rs(t: ArrayLike, corrections: int = 4) -> float | np.ndarray:
    coeffs = rs_coefficients()
    if not 0 <= corrections < len(coeffs):
        raise ValueError(f"corrections must be in 0..{len(coeffs) - 1}")
    ts = np.atleast_1d(np.asarray(t, dtype=float))
    a = np.sqrt(ts / TWO_PI)
    N = a.astype(np.int64)
    z = a - N - 0.5
    remainder = sum(np.polyval(coeffs[k], z) / a**k for k in range(corrections + 1))
    sign = np.where(N % 2 == 1, 1.0, -1.0)
    out = _main_sum(ts, N) + sign / np.sqrt(a) * remainder
    return float(out[0]) if np.ndim(t) == 0 else out


def _main_sum(t: np.ndarray, N: np.ndarray) -> np.ndarray:
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


_C_TERMS = (
    ((1, 1, 0, 0),),
    ((-1, 96, 3, 2),),
    ((1, 64, 2, 2), (1, 18432, 6, 4)),
    ((-1, 64, 1, 2), (-1, 3840, 5, 4), (-1, 5308416, 9, 6)),
    ((1, 128, 0, 2), (19, 24576, 4, 4), (11, 5898240, 8, 6), (1, 2038431744, 12, 8)),
)
_DEGREE = 90


@functools.cache
def rs_coefficients() -> tuple[np.ndarray, ...]:
    mp = mpmath.mp
    with mpmath.workdps(110):
        two_pi = 2 * mp.pi
        c, s = mp.cos(5 * mp.pi / 8), mp.sin(5 * mp.pi / 8)
        num = [mp.mpf(0)] * _DEGREE
        den = [mp.mpf(0)] * _DEGREE
        for k in range(0, _DEGREE, 4):
            num[k] = c * (-1) ** (k // 4) * two_pi ** (k // 2) / mp.factorial(k // 2)
        for k in range(2, _DEGREE, 4):
            num[k] = s * (-1) ** (k // 4) * two_pi ** (k // 2) / mp.factorial(k // 2)
        for k in range(0, _DEGREE, 2):
            den[k] = -((-1) ** (k // 2)) * two_pi**k / mp.factorial(k)
        psi = []
        for j in range(_DEGREE):
            psi.append((num[j] - sum(den[i] * psi[j - i] for i in range(1, j + 1))) / den[0])

        out = []
        for terms in _C_TERMS:
            ck = [mp.mpf(0)] * _DEGREE
            for numer, denom, m, j in terms:
                scale = mp.mpf(numer) / denom / mp.pi**j
                for i in range(_DEGREE - m):
                    ck[i] += scale * psi[i + m] * mp.factorial(i + m) / mp.factorial(i)
            out.append(np.array([float(x) for x in reversed(ck)]))
    return tuple(out)


def z_exact(t: float) -> float:
    return float(mpmath.siegelz(t))


def zeta_critical(t: float) -> complex:
    return complex(mpmath.zeta(mpmath.mpc(0.5, t)))
