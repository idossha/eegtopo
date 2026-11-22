"""
Interpolation utilities for EEG topographic mapping.

This module provides functions for smooth interpolation of EEG data
across the scalp surface.
"""

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator


def add_extrapolation_points(xy: np.ndarray, n_points: int = 16) -> np.ndarray:
    """
    Add a ring of points outside the electrode array for smoother edges.

    The ring is centred on the electrode centroid at 1.15 times the largest
    electrode distance from it. :func:`interpolate_topography` gives every
    ring point the mean of the channel values (one global value, unlike MNE,
    which uses each border point's neighbours).

    Parameters
    ----------
    xy : ndarray, shape (n, 2)
        Electrode positions
    n_points : int
        Number of extrapolation points to add

    Returns
    -------
    xy_extended : ndarray
        Original positions plus extrapolation points
    """
    # Calculate center and max radius
    center = np.mean(xy, axis=0)
    dists = np.sqrt(np.sum((xy - center) ** 2, axis=1))
    max_dist = np.max(dists)

    # Add points in a circle beyond the electrodes
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    extra_radius = max_dist * 1.15  # 15% beyond furthest electrode

    extra_x = center[0] + extra_radius * np.cos(angles)
    extra_y = center[1] + extra_radius * np.sin(angles)
    extra_points = np.column_stack([extra_x, extra_y])

    return np.vstack([xy, extra_points])


def interpolate_topography(
    xy: np.ndarray,
    values: np.ndarray,
    Xi: np.ndarray,
    Yi: np.ndarray,
    extrapolate: bool = True,
) -> np.ndarray:
    """
    Interpolate values across a 2D grid using Clough-Tocher interpolation.

    Parameters
    ----------
    xy : ndarray, shape (n, 2)
        Electrode positions
    values : ndarray, shape (n,)
        Values at each electrode; NaN or infinite values mark missing
        channels, which are left out of the interpolation
    Xi, Yi : ndarray
        Grid coordinates for interpolation
    extrapolate : bool
        Whether to add extrapolation points for smoother edges

    Returns
    -------
    Zi : ndarray
        Interpolated values on the grid

    Raises
    ------
    ValueError
        If fewer than 3 channels have finite values.
    """
    # Missing (non-finite) channels are left out, not filled
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 3:
        raise ValueError(f"need at least 3 channels with finite values, got {int(finite.sum())}")
    if not finite.all():  # indexing re-lays out xy, which changes the rounding of complete maps
        xy, values = np.asarray(xy, dtype=float)[finite], values[finite]

    if extrapolate:
        xy_extended = add_extrapolation_points(xy)
        n_extra = len(xy_extended) - len(xy)
        values_extended = np.concatenate([values, np.full(n_extra, np.mean(values))])
    else:
        xy_extended = xy
        values_extended = values

    # Use Clough-Tocher 2D interpolator for smooth surface
    interp = CloughTocher2DInterpolator(xy_extended, values_extended)
    Zi = interp(Xi, Yi)

    return Zi
