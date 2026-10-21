import mpmath
import numpy as np
import pytest

from zeta_explorer.riemann_siegel import (
    rs_coefficients,
    theta,
    theta_prime,
    z_exact,
    z_rs,
    zeta_critical,
)


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


def psi(p):
    return mpmath.cos(2 * mpmath.pi * (p * p - p - mpmath.mpf(1) / 16)) / mpmath.cos(
        2 * mpmath.pi * p
    )


@pytest.mark.parametrize("p", [0.0, 0.1, 0.3, 0.5, 0.6, 0.9, 0.999])
def test_c0_series_matches_closed_form(p):
    assert np.polyval(rs_coefficients()[0], p - 0.5) == pytest.approx(float(psi(p)), rel=1e-14)


@pytest.mark.parametrize("p", [0.25, 0.75])
def test_c0_series_through_removable_singularity(p):
    assert np.polyval(rs_coefficients()[0], p - 0.5) == pytest.approx(0.5, abs=1e-15)


def c_reference(k, p):
    d = list(mpmath.diffs(psi, p, 12))
    pi2 = mpmath.pi**2
    c = [
        d[0],
        -d[3] / (96 * pi2),
        d[2] / (64 * pi2) + d[6] / (18432 * pi2**2),
        -d[1] / (64 * pi2) - d[5] / (3840 * pi2**2) - d[9] / (5308416 * pi2**3),
        d[0] / (128 * pi2)
        + 19 * d[4] / (24576 * pi2**2)
        + 11 * d[8] / (5898240 * pi2**3)
        + d[12] / (2038431744 * pi2**4),
    ]
    return float(c[k])


@pytest.mark.parametrize("k", range(1, 5))
@pytest.mark.parametrize("p", [0.05, 0.4, 0.8])
def test_higher_corrections_match_derivatives(k, p):
    with mpmath.workdps(40):
        expected = c_reference(k, mpmath.mpf(p))
    assert np.polyval(rs_coefficients()[k], p - 0.5) == pytest.approx(expected, abs=1e-15)


@pytest.fixture(scope="module")
def z_reference():
    ts = np.linspace(200, 2_000, 300)
    return ts, np.array([z_exact(t) for t in ts])


@pytest.mark.parametrize("k, const", [(0, 0.135), (1, 0.06), (2, 0.0105), (3, 0.033), (4, 0.0175)])
def test_rs_error_order(z_reference, k, const):
    ts, exact = z_reference
    err = np.abs(z_rs(ts, corrections=k) - exact)
    assert np.all(err <= const * ts ** (-(2 * k + 3) / 4))


def test_full_corrections_below_200():
    ts = np.linspace(20, 200, 300)
    exact = np.array([z_exact(t) for t in ts])
    assert np.max(np.abs(z_rs(ts) - exact)) < 4e-6


def test_corrections_out_of_range():
    with pytest.raises(ValueError):
        z_rs(100.0, corrections=5)


def test_z_is_real_rotation_of_zeta():
    t = 37.0
    assert abs(zeta_critical(t)) == pytest.approx(abs(z_exact(t)), rel=1e-10)


def test_z_rs_vectorized_matches_scalar():
    ts = np.array([20.0, 150.5, 999.0, 12_345.6])
    assert z_rs(ts) == pytest.approx([z_rs(t) for t in ts], abs=1e-12)
