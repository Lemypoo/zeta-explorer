import math

import mpmath
import numpy as np
import pytest

from zeta_explorer.multieval import ZWindow, fft_size, nufft, zeros_fast
from zeta_explorer.riemann_siegel import theta, z_rs
from zeta_explorer.spacing import good_gram, zeros_between
from zeta_explorer.zeros import gram_points

ODLYZKO_BASE = 267653395647
ODLYZKO_1E12 = [
    1.8475231278, 2.3623669687, 2.6816309165, 2.8619899438, 3.1576654790,
    3.4342666839, 3.5808912999, 3.8344795320, 4.0584820011, 4.3570661334,
]


def _siegelz(t0, u):
    with mpmath.workdps(35):
        return float(mpmath.siegelz(mpmath.mpf(t0) + mpmath.mpf(u)))


@pytest.mark.parametrize("m", [64, 250, 1000])
@pytest.mark.parametrize("tol", [1e-6, 1e-10, 1e-13])
def test_nufft_matches_direct_sum(m, tol):
    rng = np.random.default_rng(m)
    x = rng.uniform(-10, 10, 400)
    c = rng.standard_normal(400) + 1j * rng.standard_normal(400)
    k = np.arange(-m // 2, m // 2)
    direct = np.exp(-1j * k[:, None] * x[None, :]) @ c
    assert np.abs(nufft(x, c, m, tol) - direct).max() < tol * np.abs(c).sum()


def test_nufft_needs_even_size():
    with pytest.raises(ValueError):
        nufft([0.5], [1.0], 15)


def test_fft_size_is_the_next_5_smooth_number():
    smooth = [n for n in range(1, 2000) if _is_5_smooth(n)]
    for n in range(1, 1500):
        assert fft_size(n) == next(s for s in smooth if s >= n)


def _is_5_smooth(n):
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


@pytest.mark.parametrize("T", [1e4, 1e8, 1e10])
def test_window_matches_mpmath(T):
    win = ZWindow(T, T + 100)
    for u in np.random.default_rng(int(math.log10(T))).uniform(win.u_lo, win.u_hi, 3):
        assert win.z(u)[0] == pytest.approx(_siegelz(win.t0, u), abs=2e-12)


def test_window_is_more_accurate_than_the_direct_sum_up_high():
    win = ZWindow(1e10, 1e10 + 50)
    u = 17.3
    exact = _siegelz(win.t0, u)
    assert abs(win.z(u)[0] - exact) < 1e-12
    assert abs(z_rs(win.t0 + u) - exact) > 1e-7


def test_window_across_a_change_in_the_number_of_terms():
    k = 200
    edge = 2 * math.pi * k * k
    win = ZWindow(edge - 300, edge + 300)
    assert (win.n_min, win.n) == (k - 1, k)
    for u in edge - win.t0 + np.array([-200.0, -0.01, 0.01, 150.0]):
        assert win.z(u)[0] == pytest.approx(_siegelz(win.t0, u), abs=1e-12)


def test_window_rejects_points_outside():
    win = ZWindow(1000, 1010)
    with pytest.raises(ValueError):
        win.z(win.u_hi + 1)
    with pytest.raises(ValueError):
        ZWindow(10, 20)


def test_window_gram_points():
    ns = np.arange(400_000, 400_050)
    g = gram_points(ns)
    win = ZWindow(g[0] - 1, g[-1] + 1)
    u = win.gram(ns)
    assert win.t0 + u == pytest.approx(g, abs=1e-8)
    assert theta(win.t0 + u) - ns * math.pi == pytest.approx(np.zeros(len(ns)), abs=1e-7)


def test_zeros_fast_agrees_with_zeros_between():
    win, u = zeros_fast(1_000_000, 1_000_400)
    slow = zeros_between(1_000_000, 1_000_400)
    assert len(u) == len(slow) == 400
    assert np.abs(win.t0 + u - slow).max() < 1e-6
    for i in (0, 123, 399):
        assert abs(_siegelz(win.t0, u[i])) < 1e-10
    assert np.diff(win.unfold(u)) == pytest.approx(np.diff(theta(slow)) / math.pi, abs=1e-6)


def test_zeros_near_the_trillionth_match_odlyzko():
    n = 10**12
    win, u = zeros_fast(n - 2, n + 10)
    gamma = (win.t0 - ODLYZKO_BASE) + u
    assert gamma[1:11] == pytest.approx(ODLYZKO_1E12, abs=5e-9)


def test_rosser_violation_near_the_trillionth_zero():
    n = 10**12 + 13_440
    win, u = zeros_fast(n, n + 30)
    g = win.gram(np.arange(n, n + 31))
    good = np.flatnonzero(good_gram(win.z(g), n0=n))
    assert len(u) == good[-1] - good[0]
    assert np.count_nonzero((u > g[11]) & (u < g[12])) == 3
    assert np.count_nonzero((u > g[12]) & (u < g[14])) == 0
