import numpy as np
import pytest

from zeta_explorer.gram import good_gram, gram_blocks, gram_stats, sign_changes, z_values
from zeta_explorer.riemann_siegel import z_exact
from zeta_explorer.zeros import gram_points


@pytest.fixture(scope="module")
def stats():
    return gram_stats(2_000)


def test_first_bad_gram_point_is_126(stats):
    # Hutchinson (1925): Gram's law first fails at g_126 ~ 282.45
    assert stats.first_bad == 126


def test_no_rosser_violations_early(stats):
    # The first failure of Rosser's rule is near g_13999525 (Brent 1979)
    assert stats.rosser_violations == []


def test_blocks_tile_the_gram_intervals(stats):
    assert stats.good == stats.block_lengths.total() + 1


def test_signs_match_mpmath():
    g = gram_points(np.arange(500))
    exact = np.array([z_exact(t) for t in g])
    assert np.array_equal(good_gram(z_values(g)), good_gram(exact))


def test_good_gram_offset():
    z = np.array([1.0, 1.0, -1.0])
    assert good_gram(z).tolist() == [True, False, False]
    assert good_gram(z, n0=1).tolist() == [False, True, True]


def test_gram_blocks():
    good = np.array([True, True, False, True, False, False, True, True])
    assert gram_blocks(good) == [(0, 1), (1, 3), (3, 6), (6, 7)]


def test_block_around_g126_has_two_zeros():
    g = gram_points([125, 127])
    z = z_values(g)
    assert sign_changes(g[0], g[1], z[0], z[1], need=2) == 2


def test_z_values_switches_to_riemann_siegel():
    ts = np.array([50.0, 199.0, 201.0, 5_000.0])
    assert z_values(ts) == pytest.approx([z_exact(t) for t in ts], abs=1e-8)
