import numpy as np
import pytest

from zeta_explorer.multieval import fft_size, nufft


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
