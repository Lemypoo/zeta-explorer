import math

import mpmath
import numpy as np
import pytest

from zeta_explorer.explicit import (
    chebyshev_psi,
    ei,
    li,
    mobius,
    pi_from_zeros,
    prime_pi,
    psi_from_zeros,
    riemann_j,
    riemann_r,
)
from zeta_explorer.spacing import first_zeros


@pytest.fixture(scope="module")
def zeros():
    return first_zeros(1000)


def _random_points(rng, n):
    r = np.exp(rng.uniform(np.log(0.01), np.log(700), n))
    angle = rng.uniform(-np.pi, np.pi, n)
    w = r * np.exp(1j * angle)
    supported = (r < 4) | (r >= 40) | (np.abs(w.imag) >= w.real)
    return w[supported]


def test_ei_matches_mpmath():
    w = _random_points(np.random.default_rng(1), 3000)
    ours = ei(w)
    ref = np.array([complex(mpmath.ei(x)) for x in w])
    assert np.max(np.abs(ours - ref) / np.abs(ref)) < 1e-13


def test_ei_on_the_rays_the_explicit_formula_uses():
    gamma = np.array([14.134725141734693, 1419.4224809459956])
    w = (0.5 + 1j * gamma[:, None]) * np.log([2.0, 10.0, 1e6])
    ref = np.array([[complex(mpmath.ei(x)) for x in row] for row in w])
    assert ei(w) == pytest.approx(ref, rel=1e-13)


def test_ei_real_axis():
    for x in (0.3, 5.0, 39.0, 41.0, 300.0, -0.7, -12.0, -60.0):
        assert ei(x) == pytest.approx(float(mpmath.ei(x)), rel=1e-13)
        assert ei(x).imag == 0


def test_ei_refuses_the_unsupported_wedge():
    with pytest.raises(ValueError):
        ei(10 + 3j)
    with pytest.raises(ValueError):
        ei(0)


def test_li():
    assert li(2.0) == pytest.approx(1.045163780117492784, rel=1e-14)
    for x in (0.5, 1.5, 100.0, 1e12):
        assert li(x) == pytest.approx(float(mpmath.li(x)), rel=1e-13)


def test_riemann_r_matches_mpmath():
    for x in (0.5, 2.0, 10.0, 1e6, 1e15):
        assert riemann_r(x) == pytest.approx(float(mpmath.riemannr(x)), rel=1e-13)


def test_sieve_counts():
    assert prime_pi([1, 2, 10, 100, 1000, 10**6]).tolist() == [0, 1, 4, 25, 168, 78498]
    assert chebyshev_psi(100.0) == pytest.approx(94.04531122935739, rel=1e-14)
    assert chebyshev_psi(9.0) == pytest.approx(math.log(2**3 * 3**2 * 5 * 7), rel=1e-14)


def test_mobius():
    assert [mobius(n) for n in range(1, 21)] == [
        1, -1, -1, 0, -1, 1, -1, 0, 0, 1, -1, 0, -1, 1, 1, 0, -1, 0, -1, 0,
    ]


def test_log_integral_tail():
    f = lambda t: 1 / (t * (t * t - 1) * mpmath.log(t))  # noqa: E731
    for x in (2.0, 3.7, 50.0):
        J = riemann_j(x, [])
        ref = float(mpmath.li(x)) - math.log(2) + float(mpmath.quad(f, [x, mpmath.inf]))
        assert J == pytest.approx(ref, abs=1e-10)


def test_psi_from_zeros_tracks_psi(zeros):
    x = np.arange(2.5, 101, 1.0)
    err_100 = psi_from_zeros(x, zeros[:100]) - chebyshev_psi(x)
    err_1000 = psi_from_zeros(x, zeros) - chebyshev_psi(x)
    assert np.abs(err_1000).max() < 0.16
    assert np.sqrt(np.mean(err_1000**2)) < np.sqrt(np.mean(err_100**2)) / 4


def test_pi_from_zeros_rounds_to_pi(zeros):
    x = np.arange(2.5, 101, 1.0)
    approx = pi_from_zeros(x, zeros)
    assert np.abs(approx - prime_pi(x)).max() < 0.04
    x = np.arange(1000.5, 1100, 1.0)
    assert np.array_equal(np.round(pi_from_zeros(x, zeros)), prime_pi(x))


def test_explicit_formulas_take_the_midpoint_at_a_prime(zeros):
    assert psi_from_zeros(97.0, zeros) == pytest.approx(
        chebyshev_psi(97.0) - math.log(97) / 2, abs=0.05
    )
    assert pi_from_zeros(97.0, zeros) == pytest.approx(prime_pi(97.0) - 0.5, abs=0.01)


def test_explicit_formulas_reject_x_below_one():
    with pytest.raises(ValueError):
        psi_from_zeros(1.0, [14.1])
    with pytest.raises(ValueError):
        riemann_j(0.5, [14.1])
