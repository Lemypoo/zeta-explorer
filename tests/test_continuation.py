import mpmath
import numpy as np
import pytest

from zeta_explorer.continuation import dirichlet_sums, em_sums, eta_sums

S = 0.5 + 20j


def test_dirichlet_sums_converge_only_right_of_one():
    sums = dirichlet_sums(3 + 1j, 2000)
    assert sums[-1] == pytest.approx(complex(mpmath.zeta(3 + 1j)), abs=1e-6)
    inside = np.abs(dirichlet_sums(S, 4000))
    assert inside[-1] > 2 * inside[999]


def test_eta_sums_converge_slowly_in_the_strip():
    sums = eta_sums(S, 20_000)
    pair_mean = (sums[-1] + sums[-2]) / 2
    target = complex(mpmath.zeta(S))
    assert abs(sums[-1] - target) < 0.05
    assert abs(pair_mean - target) < 1e-4


def test_em_sums_error_decays_like_n_to_minus_three_halves():
    target = complex(mpmath.zeta(S))
    err = np.abs(em_sums(S, 4000) - target)
    assert err[3999] < 1e-4
    assert err[999] / err[3999] == pytest.approx(4**1.5, rel=0.05)


def test_em_sums_reject_pole():
    with pytest.raises(ValueError):
        em_sums(1, 10)
