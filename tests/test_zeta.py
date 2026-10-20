import cmath
import math
from fractions import Fraction

import mpmath
import pytest

from zeta_explorer.zeta import bernoulli, loggamma


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
    # theta(t) = Im log Gamma(1/4 + it/2) - (t/2) log pi, exact for every t
    th = loggamma(0.25 + 0.5j * t).imag - t / 2 * math.log(math.pi)
    assert th == pytest.approx(float(mpmath.siegeltheta(t)), abs=1e-12)


def test_loggamma_recurrence():
    z = 0.3 + 4j
    assert loggamma(z + 1) - loggamma(z) == pytest.approx(cmath.log(z), abs=1e-13)


def test_loggamma_rejects_left_half_plane():
    with pytest.raises(ValueError):
        loggamma(-0.5 + 1j)
