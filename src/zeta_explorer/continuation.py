from __future__ import annotations

import numpy as np


def dirichlet_sums(s: complex, N: int) -> np.ndarray:
    n = np.arange(1, N + 1)
    return np.cumsum(n ** (-complex(s)))


def eta_sums(s: complex, N: int) -> np.ndarray:
    s = complex(s)
    n = np.arange(1, N + 1)
    terms = np.where(n % 2 == 1, 1.0, -1.0) * n ** (-s)
    return np.cumsum(terms) / (1 - 2 ** (1 - s))


def em_sums(s: complex, N: int) -> np.ndarray:
    s = complex(s)
    if s == 1:
        raise ValueError("zeta has a pole at s = 1")
    n = np.arange(1, N + 1)
    head = np.concatenate(([0j], np.cumsum(n[:-1] ** (-s))))
    return head + n ** (1 - s) / (s - 1) + n ** (-s) / 2
