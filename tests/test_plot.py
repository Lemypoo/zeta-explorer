import matplotlib

from zeta_explorer.plot import plot_z

matplotlib.use("Agg")  # headless backend for CI


def test_plot_z_marks_every_zero(tmp_path):
    fig = plot_z(10, 30, samples=200)
    zero_markers = fig.axes[0].lines[-1]
    assert len(zero_markers.get_xdata()) == 3  # 14.13, 21.02, 25.01
    fig.savefig(tmp_path / "z.png")
    assert (tmp_path / "z.png").stat().st_size > 0
