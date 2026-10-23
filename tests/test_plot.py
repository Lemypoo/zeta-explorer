import math

import matplotlib
import numpy as np
from matplotlib.animation import PillowWriter
from PIL import Image

from zeta_explorer.plot import (
    phase_colors,
    plot_continuation,
    plot_domain,
    plot_explicit,
    plot_gram,
    plot_landscape,
    plot_lehmer,
    plot_log_z,
    plot_rs_error,
    plot_spacing,
    plot_spiral,
    plot_z,
    sweep_sigma,
)
from zeta_explorer.spacing import first_zeros, zeros_between
from zeta_explorer.values import log_z_sample

matplotlib.use("Agg")


def test_plot_z_marks_every_zero(tmp_path):
    fig = plot_z(10, 30, samples=200)
    zero_markers = fig.axes[0].lines[-1]
    assert len(zero_markers.get_xdata()) == 3
    fig.savefig(tmp_path / "z.png")
    assert (tmp_path / "z.png").stat().st_size > 0


def test_plot_gram_flags_g126_and_g134():
    fig = plot_gram(262, 300, samples=200)
    (bad,) = [line for line in fig.axes[0].lines if line.get_label() == "bad Gram point"]
    assert len(bad.get_xdata()) == 2


def test_plot_rs_error_draws_one_line_per_order():
    fig = plot_rs_error(200, 400, bins=2)
    assert len(fig.axes[0].lines) == 5


def test_phase_colors_hue_and_bands():
    rgb = phase_colors(np.array([1.0, 1j, -1.0, 2.0, 0.0, np.inf]))
    assert rgb.shape == (6, 3)
    assert rgb[0].argmax() == 0
    assert rgb[2].argmax() != 0
    assert np.allclose(rgb[3], rgb[0])
    assert np.isfinite(rgb).all()


def test_plot_domain_image_size():
    fig = plot_domain(0, 10, -1, 2, width=100)
    assert fig.axes[0].images[0].get_array().shape == (30, 100, 3)


def test_plot_spiral_counts_zeros():
    fig = plot_spiral(t_max=26, samples=300)
    assert any("3 times" in text.get_text() for text in fig.axes[0].texts)


def test_plot_continuation_draws_three_series_twice():
    fig = plot_continuation(N=50)
    left, right = fig.axes
    assert len(right.lines) == 3
    assert len(left.lines) == 4


def test_plot_spacing():
    fig = plot_spacing(zeros_between(1000, 1400), bins=20)
    assert len(fig.axes) == 2
    assert "399 zero spacings" in fig.axes[0].get_legend().get_texts()[0].get_text()


def test_plot_explicit_draws_each_truncation():
    fig = plot_explicit(first_zeros(20), counts=(5, 10, 20), x_max=30, samples=100)
    left, right = fig.axes
    labels = [line.get_label() for line in left.lines]
    assert labels[-3:] == ["5 zeros", "10 zeros", "20 zeros"]
    assert len(right.lines) == 4


def test_plot_lehmer_marks_the_pair():
    fig = plot_lehmer(samples=200)
    zoom = fig.axes[1]
    assert "0.0377 apart" in zoom.get_title(loc="left")
    assert len(zoom.lines[-1].get_xdata()) == 2


def test_plot_log_z():
    T = 2 * math.pi * math.exp(8)
    fig = plot_log_z(log_z_sample(T, 2000, np.random.default_rng(1)), T, 8, bins=30)
    texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert texts[0].startswith("log|ζ(1/2 + it)|, 2,000 random t")
    assert "CUE, N = 8" in texts[1]


def test_plot_landscape_puts_a_pit_at_each_zero():
    fig = plot_landscape(t_max=30, width=120)
    pits = fig.axes[0].collections[-1]
    assert np.allclose(sorted(pits._offsets3d[0]), [14.134725, 21.022040, 25.010858], atol=1e-6)


def test_sweep_sigma_writes_every_frame(tmp_path):
    anim = sweep_sigma(t_max=30, frames=8, samples=300)
    anim.save(tmp_path / "sweep.gif", writer=PillowWriter(fps=6), dpi=40)
    assert Image.open(tmp_path / "sweep.gif").n_frames == 8

