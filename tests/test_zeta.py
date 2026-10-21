import cmath
import math
from fractions import Fraction

import mpmath
import numpy as np
import pytest

from zeta_explorer.riemann_siegel import theta, z_rs
from zeta_explorer.zeta import bernoulli, chi, euler_maclaurin, loggamma, zeta


@pytest.mark.parametrize(
    "n, expected",
    [(0, Fraction(1)), (1, Fraction(-1, 2)), (2, Fraction(1, 6)), (3, Fraction(0)),
     (4, Fraction(-1, 30)), (12, Fraction(-691, 2730)), (20, Fraction(-174611, 330))],
)
def test_bernoulli(n, expected):
    assert bernoulli(n) == expected


@pytest.mark.parametrize("n", [1, 2, 5, 10, 30])
def test_loggamma_factorials(n):
    assert loggamma(n) == pytest.approx(math.lgamma(n), abs=1e-13)


@pytest.mark.parametrize("z", [0.5, 1e-3 + 1e-3j, 0.25 + 50j, 3 - 7j, 0.1 + 1_000j, 40 + 0.5j])
def test_loggamma_matches_mpmath(z):
    expected = complex(mpmath.loggamma(z))
    assert abs(loggamma(z) - expected) < 1e-12 * max(1, abs(expected))


@pytest.mark.parametrize("t", [0.5, 2.0, 6.0, 30.0])
def test_theta_from_loggamma(t):
    th = loggamma(0.25 + 0.5j * t).imag - t / 2 * math.log(math.pi)
    assert th == pytest.approx(float(mpmath.siegeltheta(t)), abs=1e-12)


def test_loggamma_recurrence():
    z = 0.3 + 4j
    assert loggamma(z + 1) - loggamma(z) == pytest.approx(cmath.log(z), abs=1e-13)


def test_loggamma_rejects_left_half_plane():
    with pytest.raises(ValueError):
        loggamma(-0.5 + 1j)


@pytest.mark.parametrize(
    "s, expected",
    [(2, math.pi**2 / 6), (4, math.pi**4 / 90), (0, -0.5), (0.5, -1.4603545088095868),
     (3, 1.2020569031595942), (-2, 0.0), (-20, 0.0)],
)
def test_zeta_known_values(s, expected):
    assert zeta(s) == pytest.approx(expected, rel=1e-14, abs=1e-15)


@pytest.mark.parametrize("k", range(1, 12))
def test_zeta_negative_odd_integers(k):
    expected = -float(bernoulli(2 * k)) / (2 * k)
    value = zeta(1 - 2 * k)
    assert value.imag == 0
    assert value.real == pytest.approx(expected, rel=1e-13)


def test_zeta_matches_mpmath_on_random_points():
    rng = np.random.default_rng(7)
    sigma = rng.uniform(-30, 30, 300)
    t = rng.uniform(-200, 200, 300)
    for s in sigma + 1j * t:
        expected = complex(mpmath.zeta(s))
        assert abs(zeta(s) - expected) <= 1e-12 * abs(expected)


@pytest.mark.parametrize("t", [1_000.0, 10_000.0])
def test_zeta_high_on_critical_line(t):
    assert abs(zeta(0.5 + 1j * t) - complex(mpmath.zeta(0.5 + 1j * t))) < 1e-10


def test_zeta_vanishes_at_first_zero():
    assert abs(zeta(0.5 + 14.134725141734693j)) < 1e-14


def test_zeta_pole():
    with pytest.raises(ValueError):
        zeta(1)


@pytest.mark.parametrize(
    "s, N, M",
    [(2, 3, 1), (0.5 + 20j, 5, 2), (0.5 + 20j, 8, 6), (-0.5 + 3j, 4, 3), (5 - 40j, 10, 4),
     (0.1 + 100j, 20, 5)],
)
def test_backlund_bound_holds(s, N, M):
    value, bound = euler_maclaurin(s, N, M)
    err = abs(value - complex(mpmath.zeta(s)))
    assert err <= bound
    assert bound < 50 * err


def test_euler_maclaurin_validates_arguments():
    with pytest.raises(ValueError):
        euler_maclaurin(2, 0, 3)
    with pytest.raises(ValueError):
        euler_maclaurin(-10, 5, 3)


@pytest.mark.parametrize("s", [0.3 + 5j, 0.5 + 50j, 0.9 - 7j, 0.01 + 0.2j])
def test_functional_equation(s):
    assert zeta(s) == pytest.approx(chi(s) * zeta(1 - s), rel=1e-13)


@pytest.mark.parametrize("t", [10.0, 100.0, 5_000.0])
def test_chi_on_critical_line_is_theta_phase(t):
    assert chi(0.5 + 1j * t) == pytest.approx(cmath.exp(-2j * theta(t)), abs=1e-11)


@pytest.mark.parametrize("t", [250.0, 777.7, 3_000.0])
def test_z_from_own_zeta(t):
    z = cmath.exp(1j * theta(t)) * zeta(0.5 + 1j * t)
    assert abs(z.imag) < 1e-10
    assert z.real == pytest.approx(z_rs(t), abs=1e-7)
