import math

import mpmath
import numpy as np
import pytest

from zeta_explorer.riemann_siegel import theta, z_rs
from zeta_explorer.zeros import find_zeros, gram_point, gram_points, illinois

FIRST_THREE = [14.134725141734693, 21.022039638771555, 25.010857580145688]


def test_first_zeros_known_values():
    zeros = find_zeros(10, 26)
    assert zeros == pytest.approx(FIRST_THREE, abs=1e-9)


def test_29_zeros_below_100_match_mpmath():
    zeros = find_zeros(10, 100)
    assert len(zeros) == 29
    reference = [float(mpmath.zetazero(k).imag) for k in range(1, 30)]
    assert zeros == pytest.approx(reference, abs=1e-9)


def test_riemann_siegel_approximation_finds_the_same_zeros():
    exact = find_zeros(10, 100)
    approx = find_zeros(10, 100, z=z_rs)
    assert len(approx) == len(exact)
    assert max(abs(a - b) for a, b in zip(approx, exact, strict=True)) < 1e-5


def test_illinois_on_a_simple_function():
    f = lambda x: x**3 - 2  # noqa: E731
    root = illinois(f, 0.0, 2.0, f(0.0), f(2.0))
    assert root == pytest.approx(2 ** (1 / 3), abs=1e-11)


def test_illinois_requires_a_bracket():
    with pytest.raises(ValueError):
        illinois(math.cos, 0.0, 1.0, 1.0, 0.5)


@pytest.mark.parametrize(
    "n, expected", [(0, 17.8455995404), (1, 23.1702827012), (2, 27.6701822178)]
)
def test_gram_points_known_values(n, expected):
    assert gram_point(n) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("n", [0, 10, 1_000, 100_000])
def test_gram_point_solves_theta_equals_n_pi(n):
    assert theta(gram_point(n)) == pytest.approx(n * math.pi, abs=1e-8)


def test_gram_point_rejects_negative_index():
    with pytest.raises(ValueError):
        gram_point(-1)


def test_gram_points_batch():
    ns = np.arange(5_000)
    g = gram_points(ns)
    assert np.all(np.diff(g) > 0)
    assert theta(g) == pytest.approx(ns * math.pi, abs=1e-8)
