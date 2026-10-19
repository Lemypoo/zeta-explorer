import math

import mpmath
import numpy as np
import pytest

from zeta_explorer.riemann_siegel import c0, theta, theta_prime, z_exact, z_rs, zeta_critical


@pytest.mark.parametrize("t", [10, 14.13, 50, 100, 1_000, 100_000])
def test_theta_matches_mpmath(t):
    assert theta(t) == pytest.approx(float(mpmath.siegeltheta(t)), abs=1e-9)


@pytest.mark.parametrize("t", [12, 30, 200, 5_000])
def test_theta_prime_matches_numerical_derivative(t):
    h = 1e-5
    assert theta_prime(t) == pytest.approx((theta(t + h) - theta(t - h)) / (2 * h), rel=1e-7)


def test_theta_rejects_nonpositive_t():
    with pytest.raises(ValueError):
        theta(0)


@pytest.mark.parametrize("p", [0.25, 0.75])
def test_c0_removable_singularity_has_limit_one_half(p):
    assert c0(p) == pytest.approx(0.5)
    # and is continuous there
    assert c0(p + 1e-6) == pytest.approx(0.5, abs=1e-4)
    assert c0(p - 1e-6) == pytest.approx(0.5, abs=1e-4)


def test_c0_value_at_zero():
    # cos(2pi * -1/16) / cos(0) = cos(pi/8)
    assert c0(0.0) == pytest.approx(math.cos(math.pi / 8))


@pytest.mark.parametrize("t_range", [(20, 100), (100, 1_000), (1_000, 5_000)])
def test_riemann_siegel_error_decays_like_t_to_the_minus_three_quarters(t_range):
    """After the C0 term, the error is about C1 * (2pi/t)^(3/4) with |C1| < ~0.03."""
    ts = np.linspace(*t_range, 150)
    err = np.array([abs(z_rs(t) - z_exact(t)) for t in ts])
    assert np.all(err <= 0.035 * (2 * math.pi / ts) ** 0.75)


def test_z_is_real_rotation_of_zeta():
    t = 37.0
    assert abs(zeta_critical(t)) == pytest.approx(abs(z_exact(t)), rel=1e-10)
