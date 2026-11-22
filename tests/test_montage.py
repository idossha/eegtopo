"""Montage projection and head geometry (2026-11-22).

Pins: the projection puts left/right and front/back where MNE's head frame
says they are (+x right, +y anterior), the head outline used by
TopographicAnalysis does not depend on which channels the data has (audit #4),
and channel selection returns empty results rather than raising (#11).

Numbers: none beyond signs; the orientation comes from the 10-20 layout
(T7 left, T8 right, Fpz front, Oz back). Reproduce: ``pytest tests/test_montage.py``.
"""

import numpy as np
from matplotlib.patches import Circle

import eegtopo
from eegtopo import TopographicAnalysis, create_mne_info, get_channel_positions

from conftest import CHANNELS_1020, FRONTAL, MONTAGE_1020, long_df


def test_projection_orientation_1020():
    names, xy = get_channel_positions(["T7", "T8", "Fpz", "Oz"], MONTAGE_1020)
    pos = dict(zip(names, xy))
    assert pos["T7"][0] < 0 < pos["T8"][0]
    assert pos["Oz"][1] < 0 < pos["Fpz"][1]


def test_selection_keeps_montage_order_and_drops_excluded():
    names, xy = get_channel_positions(["Oz", "Cz", "Fpz", "XYZ"], MONTAGE_1020, exclude=["Cz"])
    assert names == ["Fpz", "Oz"]
    assert xy.shape == (2, 2)


def test_no_matching_channel_returns_empty():
    assert get_channel_positions(["XYZ"], MONTAGE_1020)[0] == []
    assert get_channel_positions(["XYZ"], MONTAGE_1020)[1].shape == (0, 2)
    assert create_mne_info(["XYZ"], MONTAGE_1020) == (None, [])


def test_head_geometry_does_not_depend_on_the_data_channels():
    full = TopographicAnalysis(long_df(n_subjects=2), montage_name=MONTAGE_1020)
    back_only = long_df(n_subjects=2, channels=[c for c in CHANNELS_1020 if c not in FRONTAL])
    partial = TopographicAnalysis(back_only, montage_name=MONTAGE_1020)

    assert partial.head_center == full.head_center
    assert partial.head_radius == full.head_radius
    assert (partial.head_center, partial.head_radius) == eegtopo.head_geometry(MONTAGE_1020)

    circles = []
    for a in (full, partial):
        _, ax, _ = a.plot_topomap(np.zeros(len(a.ordered_channels)), colorbar=False)
        (c,) = [p for p in ax.patches if isinstance(p, Circle)]
        circles.append((c.center, c.radius))
    assert circles[0] == circles[1]


def test_head_geometry_honours_exclude():
    left = [c for c in get_channel_positions(CHANNELS_1020, MONTAGE_1020)[0] if c.endswith(("1", "3", "5", "7"))]
    centre_all, _ = eegtopo.head_geometry(MONTAGE_1020)
    centre_excl, _ = eegtopo.head_geometry(MONTAGE_1020, exclude=left)
    assert not np.allclose(centre_all, centre_excl)
