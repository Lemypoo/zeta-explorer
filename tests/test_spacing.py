import math

import mpmath
import numpy as np
import pytest

from zeta_explorer.spacing import (
    gap_probability,
    gue_spacing,
    gue_spacing_cdf,
    ks_distance,
    montgomery,
    normalized_spacings,
    pair_correlation,
    wigner_surmise,
    zeros_between,
)

PI = math.pi


@pytest.mark.parametrize("s", [0.05, 0.1, 0.2])
def test_gap_probability_small_s_series(s):
    series = 1 - s + PI**2 * s**4 / 36 - PI**4 * s**6 / 675
    assert gap_probability(s) == pytest.approx(series, abs=2 * PI**6 * s**8 / 17640)


def test_gap_probability_converges_in_nodes():
    for s in (0.5, 2.0, 4.0):
        assert gap_probability(s, m=20) == pytest.approx(gap_probability(s, m=60), rel=1e-12)


def test_gue_spacing_small_s_series():
    s = 0.05
    series = PI**2 / 3 * s**2 - 2 * PI**4 / 45 * s**4 + PI**6 / 315 * s**6
    assert gue_spacing(s) == pytest.approx(series, rel=1e-6)


def _mp_gap(s, nodes):
    xs = [(x + 1) * s / 2 for x, _ in nodes]
    ws = [mpmath.sqrt(w * s / 2) for _, w in nodes]
    n = len(xs)
    A = mpmath.matrix(n, n)
    for i in range(n):
        for j in range(n):
            d = xs[i] - xs[j]
            k = 1 if d == 0 else mpmath.sin(mpmath.pi * d) / (mpmath.pi * d)
            A[i, j] = (i == j) - ws[i] * k * ws[j]
    return mpmath.det(A)


def test_gue_spacing_matches_high_precision_derivatives():
    with mpmath.workdps(30):
        nodes = mpmath.calculus.quadrature.GaussLegendre(mpmath.mp).calc_nodes(4, mpmath.mp.prec)
        p = float(mpmath.diff(lambda s: _mp_gap(s, nodes), 1.0, 2))
        F = 1 + float(mpmath.diff(lambda s: _mp_gap(s, nodes), 1.0, 1))
    assert gue_spacing(1.0) == pytest.approx(p, rel=1e-8)
    assert gue_spacing_cdf(1.0) == pytest.approx(F, rel=1e-10)


def test_gue_spacing_is_a_density_with_unit_mean():
    s, ds = np.linspace(0, 6, 30001, retstep=True)
    p = gue_spacing(s)
    assert p.sum() * ds == pytest.approx(1, abs=1e-8)
    assert (s * p).sum() * ds == pytest.approx(1, abs=1e-8)
    assert p.min() > -1e-8


def test_gue_close_to_but_not_equal_to_wigner():
    s = np.linspace(0, 4, 401)
    gap = np.abs(gue_spacing(s) - wigner_surmise(s)).max()
    assert 0.003 < gap < 0.01


def test_gue_cdf_tails():
    assert gue_spacing_cdf(0.0) == pytest.approx(0, abs=1e-9)
    assert gue_spacing_cdf(10.0) == 1.0
    assert gue_spacing(10.0) == 0.0


def _gue_spacings(n, trials, rng):
    out = []
    for _ in range(trials):
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        lam = np.linalg.eigvalsh((A + A.conj().T) / 2) / (2 * math.sqrt(n))
        lam = np.clip(lam, -1, 1)
        counts = n * (0.5 + (lam * np.sqrt(1 - lam**2) + np.arcsin(lam)) / PI)
        bulk = counts[n // 4 : 3 * n // 4]
        out.append(np.diff(bulk))
    return np.concatenate(out)


def test_random_gue_matrices_match_sine_kernel():
    gaps = _gue_spacings(120, 300, np.random.default_rng(3))
    assert gaps.mean() == pytest.approx(1, abs=0.01)
    assert ks_distance(gaps, gue_spacing_cdf) < 0.01
    assert ks_distance(gaps, lambda x: 1 - np.exp(-x)) > 0.25


def test_montgomery():
    assert montgomery(0.0) == 0.0
    assert montgomery(1.0) == pytest.approx(1)
    assert montgomery(0.5) == pytest.approx(1 - 4 / PI**2)


def test_pair_correlation_of_lattice_and_poisson():
    u, r = pair_correlation(np.arange(5000.0) + 0.25, u_max=2.5, bins=5)
    assert r == pytest.approx([0, 0, 2, 0, 2])
    x = np.cumsum(np.random.default_rng(0).exponential(size=200_000))
    u, r = pair_correlation(x, u_max=3.0, bins=6)
    assert r == pytest.approx(np.ones(6), abs=0.02)


def test_ks_distance_exact():
    assert ks_distance([0.5], lambda x: x) == pytest.approx(0.5)
    assert ks_distance([0.25, 0.75], lambda x: x) == pytest.approx(0.25)


def test_zeros_between_match_zetazero():
    z = zeros_between(1000, 1060)
    assert len(z) == 60
    reference = [float(mpmath.zetazero(k).imag) for k in (1002, 1030, 1061)]
    assert [z[0], z[28], z[59]] == pytest.approx(reference, abs=1e-8)


def test_zeros_between_finds_every_zero_in_bad_blocks():
    z = zeros_between(120, 160)
    assert len(z) == 40
    assert np.all(np.diff(z) > 0)


def test_unfolded_zeros_have_unit_mean_spacing():
    z = zeros_between(20_000, 22_000)
    gaps = normalized_spacings(z)
    assert gaps.mean() == pytest.approx(1, abs=1e-3)
    assert ks_distance(gaps, gue_spacing_cdf) < 0.05
    assert ks_distance(gaps, lambda x: 1 - np.exp(-x)) > 0.25


def test_zeros_between_rejects_low_heights():
    with pytest.raises(ValueError):
        zeros_between(10, 50)
