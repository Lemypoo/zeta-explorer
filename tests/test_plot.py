import matplotlib

from zeta_explorer.plot import plot_gram, plot_rs_error, plot_z

matplotlib.use("Agg")  # headless backend for CI


def test_plot_z_marks_every_zero(tmp_path):
    fig = plot_z(10, 30, samples=200)
    zero_markers = fig.axes[0].lines[-1]
    assert len(zero_markers.get_xdata()) == 3  # 14.13, 21.02, 25.01
    fig.savefig(tmp_path / "z.png")
    assert (tmp_path / "z.png").stat().st_size > 0


def test_plot_gram_flags_g126_and_g134():
    fig = plot_gram(262, 300, samples=200)
    bad = fig.axes[0].lines[2]
    assert bad.get_label() == "bad Gram point"
    assert len(bad.get_xdata()) == 2


def test_plot_rs_error_draws_one_line_per_order():
    fig = plot_rs_error(200, 400, bins=2)
    assert len(fig.axes[0].lines) == 5
