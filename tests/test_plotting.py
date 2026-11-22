"""plot_topomap: missing channels, colour limits, head styling (2026-11-22).

Pins: NaN / missing channels are left out of the interpolation instead of
being drawn as 0 (audit #3), the head outline is built from every channel
passed in even when values are missing, the auto colour limits follow the sign
of the data (#6, one rule shared with plot_mean_topomap), and head_kw reaches
the outline (#5).

Numbers: 97.03 is the 97th percentile (linear interpolation) of 1..100,
worked by hand as 97 + 0.03 * (98 - 97); the |values| of -100..-1 and 1..100
give the same 97.03 (sorted pairs, position 0.97 * 199 = 193.03). A flat map
must stay flat because Clough-Tocher reproduces constants exactly up to
rounding. Reproduce: ``pytest tests/test_plotting.py``.
Head geometry from the montage is in test_montage.py.
"""

import numpy as np
import pytest
from matplotlib.patches import Circle

from eegtopo import get_channel_positions, interpolate_topography, plot_topomap

from conftest import disk_xy

P97_OF_1_TO_100 = 97.03
FLAT_TOL = 1e-9  # rounding in Clough-Tocher on constant data; a NaN-as-0 dip is ~6


@pytest.fixture(scope="module")
def hydrocel():
    import mne

    names = mne.channels.make_standard_montage("GSN-HydroCel-256").ch_names
    return get_channel_positions(names, "GSN-HydroCel-256")


def _head_circle(ax):
    (circle,) = [p for p in ax.patches if isinstance(p, Circle)]
    return circle


def test_flat_map_with_one_nan_channel_stays_flat(hydrocel):
    names, xy = hydrocel
    values = np.full(len(names), 6.0)
    centre = int(np.argmin(np.hypot(*(xy - xy.mean(axis=0)).T)))  # well inside the head
    values[centre] = np.nan

    _, _, im = plot_topomap(values, xy, ch_names=names, colorbar=False)

    shown = np.ma.compressed(im.get_array())
    shown = shown[np.isfinite(shown)]
    assert shown.size > 1000, "has pixels inside the head to check"
    assert shown.min() == pytest.approx(6.0, abs=FLAT_TOL)
    assert shown.max() == pytest.approx(6.0, abs=FLAT_TOL)


def test_channel_missing_from_dict_is_not_drawn_as_zero(hydrocel):
    names, xy = hydrocel
    values = {ch: 6.0 for ch in names[1:]}  # first channel absent

    _, ax, im = plot_topomap(values, xy, ch_names=names, colorbar=False)

    shown = np.ma.compressed(im.get_array())
    assert np.nanmin(shown) == pytest.approx(6.0, abs=FLAT_TOL)
    # its dot is still drawn
    assert sum(len(c.get_offsets()) for c in ax.collections) == len(names)


def test_head_comes_from_all_channels_even_when_values_are_missing(hydrocel):
    names, xy = hydrocel
    full = np.random.default_rng(1).normal(size=len(names))
    gappy = full.copy()
    gappy[np.random.default_rng(2).random(len(names)) < 0.4] = np.nan
    assert 0.3 < np.isnan(gappy).mean() < 0.5

    _, ax_full, im_full = plot_topomap(full, xy, colorbar=False)
    _, ax_gap, im_gap = plot_topomap(gappy, xy, colorbar=False)

    assert _head_circle(ax_gap).center == _head_circle(ax_full).center
    assert _head_circle(ax_gap).radius == _head_circle(ax_full).radius
    assert im_gap.get_extent() == im_full.get_extent()
    assert np.array_equal(np.ma.getmaskarray(im_gap.get_array()), np.ma.getmaskarray(im_full.get_array()))


def test_interpolation_needs_three_finite_channels():
    xy = disk_xy(5)
    Xi, Yi = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8))
    with pytest.raises(ValueError, match="at least 3"):
        interpolate_topography(xy, np.array([1.0, 2.0, np.nan, np.nan, np.inf]), Xi, Yi)


@pytest.mark.parametrize(
    "values, expected",
    [
        (np.arange(1.0, 101.0), (0.0, P97_OF_1_TO_100)),
        (-np.arange(1.0, 101.0), (-P97_OF_1_TO_100, 0.0)),
        (np.concatenate([-np.arange(1.0, 101.0), np.arange(1.0, 101.0)]), (-P97_OF_1_TO_100, P97_OF_1_TO_100)),
        (np.zeros(100), (0.0, 1e-6)),
    ],
    ids=["all-positive", "all-negative", "mixed", "all-zero"],
)
def test_auto_vlim_follows_the_sign_of_the_data(values, expected):
    _, _, im = plot_topomap(values, disk_xy(len(values)), colorbar=False)
    assert im.get_clim() == pytest.approx(expected)


def test_auto_vlim_ignores_nan():
    values = np.append(np.arange(1.0, 101.0), np.nan)
    _, _, im = plot_topomap(values, disk_xy(len(values)), colorbar=False)
    assert im.get_clim() == pytest.approx((0.0, P97_OF_1_TO_100))


def test_head_kw_styles_the_outline():
    _, ax, _ = plot_topomap(np.arange(30.0), disk_xy(30), colorbar=False, head_kw={"linewidth": 0.5, "color": "red"})
    assert _head_circle(ax).get_linewidth() == 0.5
    assert ax.lines, "has nose and ear lines to check"
    assert all(line.get_linewidth() == 0.5 and line.get_color() == "red" for line in ax.lines)


def test_default_head_is_still_the_electrode_centroid():
    # the no-argument path must not move (callers that pass no geometry keep their figures)
    xy = disk_xy(40) + [0.3, -0.2]
    _, ax, _ = plot_topomap(np.arange(40.0), xy, colorbar=False)
    assert _head_circle(ax).center == pytest.approx(tuple(xy.mean(axis=0)))
