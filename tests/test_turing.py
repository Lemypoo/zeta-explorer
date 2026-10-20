import math

import mpmath
import pytest

from zeta_explorer.riemann_siegel import z_rs
from zeta_explorer.turing import T_MIN, blocks_needed, verify_zeros
from zeta_explorer.zeros import find_zeros


@pytest.mark.parametrize("height", [1e3, 1e5, 1e8, 1e12, 2e12])
def test_blocks_needed_matches_trudgian_rounding(height):
    L = math.log(height)
    assert blocks_needed(height) == math.ceil(0.0061 * L * L + 0.08 * L)


def test_verify_zeros_to_1000():
    v = verify_zeros(1_000)
    assert v.g_n >= 1_000
    assert v.zeros == v.n + 1 == 649
    assert mpmath.nzeros(v.g_n) == v.zeros
    assert len(find_zeros(10, v.g_n, z=z_rs)) == v.zeros


def test_verify_zeros_starts_above_lehman_range():
    v = verify_zeros(50)
    assert v.g_n >= T_MIN
    assert v.zeros == mpmath.nzeros(v.g_n)


def test_verify_zeros_to_5000():
    v = verify_zeros(5_000)
    assert v.zeros == v.n + 1 == mpmath.nzeros(v.g_n)
    assert v.blocks_after >= blocks_needed(5_000)
