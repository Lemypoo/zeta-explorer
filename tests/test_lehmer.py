import mpmath
import numpy as np
import pytest

from zeta_explorer.lehmer import T0, lehmer_pairs, lehmer_sums, newman_bound, smooth_count
from zeta_explorer.riemann_siegel import theta
from zeta_explorer.spacing import first_zeros, zeros_between


@pytest.fixture(scope="module")
def zeros():
    return first_zeros(1500)


def _hadamard_half(t, other):
    with mpmath.workdps(20):
        t = mpmath.findroot(mpmath.siegelz, mpmath.mpf(t))
        z1, z2, z3 = (mpmath.siegelz(t, derivative=d) for d in (1, 2, 3))
        a = mpmath.mpf(1) / 4
        log_f2 = 2 * (a - t**2) / (t**2 + a) ** 2 - mpmath.re(mpmath.psi(1, a + 0.5j * t)) / 4
        return float(-(z3 / (3 * z1) - (z2 / (2 * z1)) ** 2) - log_f2 - 1 / (t - other) ** 2)


def _hadamard_g(a, b):
    return _hadamard_half(a, b) + _hadamard_half(b, a)


def test_smooth_count():
    assert smooth_count(T0) == pytest.approx(0, abs=1e-14)
    for t in (1e3, 1e5):
        assert smooth_count(t) == pytest.approx(theta(t) / np.pi + 1, abs=1e-4)


@pytest.mark.parametrize("i", [0, 50, 700])
def test_sums_match_hadamard_product(zeros, i):
    g = lehmer_sums(zeros)
    assert g[i] == pytest.approx(_hadamard_g(zeros[i], zeros[i + 1]), rel=1e-5)


def test_window_convergence(zeros):
    wide = lehmer_sums(zeros, window=500)
    narrow = lehmer_sums(zeros, window=50)
    assert np.max(np.abs(narrow / wide - 1)) < 1e-3


def test_29_lehmer_pairs_below_1000(zeros):
    pairs = [p for p in lehmer_pairs(zeros[:700]) if p.gamma + p.delta < 1000]
    assert len(pairs) == 29
    assert all(p.score < 0.8 and -1 / (2 * p.score / p.delta**2) < p.bound < 0 for p in pairs)


def test_lehmers_pair_near_7005():
    z = zeros_between(6000, 7400)
    pairs = {p.n: p for p in lehmer_pairs(z, n0=6002)}
    p = pairs[6709]
    assert p.gamma == pytest.approx(7005.062866175, abs=1e-8)
    assert p.delta == pytest.approx(0.037698498, abs=1e-8)
    g = _hadamard_g(p.gamma, p.gamma + p.delta)
    assert p.score == pytest.approx(p.delta**2 * g, rel=1e-5)
    assert p.bound == pytest.approx(newman_bound(p.delta, g), rel=1e-5)
    assert p.bound == pytest.approx(-7.112e-4, abs=1e-7)
    assert min(pairs.values(), key=lambda q: q.score).n == 6709


def test_van_de_lune_pair_gives_the_1993_bound():
    delta = 2.171392e-4 / 2
    for g in (1.0, 100.0):
        assert -5.895e-9 < newman_bound(delta, g) < -5.893e-9


def test_no_bound_without_a_lehmer_pair():
    assert np.isnan(newman_bound(1.0, 0.8))
    assert newman_bound(1.0, 0.79) < 0
