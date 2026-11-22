"""
Montage handling utilities for EEG topographic mapping.

This module provides functions for working with EEG montages, including
loading standard montages from MNE-Python and extracting 2D positions.
"""

from typing import List, Optional, Tuple, Any
import numpy as np


def _montage_positions(
    montage_name: str,
    ch_names: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> Tuple[List[str], np.ndarray]:
    """Channels of a standard montage in montage order (all of them when
    *ch_names* is None) minus *exclude*, with their 3-D positions."""
    import mne

    montage = mne.channels.make_standard_montage(montage_name)
    keep = set(montage.ch_names if ch_names is None else ch_names) - set(exclude or [])
    ordered = [ch for ch in montage.ch_names if ch in keep]
    pos = montage.get_positions()["ch_pos"]
    return ordered, np.array([pos[ch] for ch in ordered]).reshape(-1, 3)


def get_channel_positions(
    ch_names: List[str],
    montage_name: str = "GSN-HydroCel-256",
    exclude: Optional[List[str]] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Return 2-D projected positions for channels from a standard MNE montage.

    Parameters
    ----------
    ch_names : list[str]
        Channel names present in the data.
    montage_name : str
        MNE standard montage identifier (e.g., 'GSN-HydroCel-256',
        'standard_1020', 'biosemi64', etc.)
    exclude : list[str] | None
        Channels to exclude (e.g., rim electrodes, fiducials).

    Returns
    -------
    ordered_channels : list[str]
        Channel names in montage order (after exclusion & intersection).
        Empty when no channel matches the montage.
    xy : ndarray, shape (n_channels, 2)
        2-D projected positions; shape (0, 2) when no channel matches.
    """
    from .coordinates import azimuthal_equidistant_projection

    ordered, pos = _montage_positions(montage_name, ch_names, exclude)
    if not ordered:
        return [], np.empty((0, 2))
    return ordered, azimuthal_equidistant_projection(pos)


def head_geometry(
    montage_name: str = "GSN-HydroCel-256",
    exclude: Optional[List[str]] = None,
) -> Tuple[Tuple[float, float], float]:
    """
    Head-outline centre and radius computed from the whole montage.

    Uses the same rule as :class:`~eegtopo.HeadOutline` when it is given no
    geometry (centroid of the projected electrodes, 90th-percentile distance),
    but over every channel of *montage_name* minus *exclude*, so the outline
    does not move when a dataset lacks some channels. Pass the result to
    ``plot_topomap(head_center=..., head_radius=...)``.

    Parameters
    ----------
    montage_name : str
        MNE standard montage identifier.
    exclude : list[str] | None
        Channels to leave out of the geometry (e.g., rim electrodes).

    Returns
    -------
    head_center : tuple (x, y)
    head_radius : float
    """
    from .coordinates import azimuthal_equidistant_projection
    from .head import HeadOutline

    _, pos = _montage_positions(montage_name, exclude=exclude)
    head = HeadOutline(azimuthal_equidistant_projection(pos))
    return (head.cx, head.cy), head.radius


def create_mne_info(
    ch_names: List[str],
    montage_name: str = "GSN-HydroCel-256",
    exclude: Optional[List[str]] = None,
) -> Tuple[Any, List[str]]:
    """
    Build an MNE Info object with a subset montage.

    Useful for computing channel adjacency matrices needed for cluster-based
    permutation testing.

    Parameters
    ----------
    ch_names : list[str]
        Channel names to include.
    montage_name : str
        MNE standard montage identifier.
    exclude : list[str] | None
        Channels to exclude.

    Returns
    -------
    info : mne.Info | None
        MNE Info object with montage set, or None if no valid channels.
    ordered_channels : list[str]
        List of channel names in order.
    """
    import mne

    ordered, pos = _montage_positions(montage_name, ch_names, exclude)
    if not ordered:
        return None, []

    montage_sub = mne.channels.make_dig_montage(ch_pos=dict(zip(ordered, pos)), coord_frame="head")
    info = mne.create_info(ch_names=ordered, sfreq=100, ch_types="eeg")
    info.set_montage(montage_sub)

    return info, ordered
