"""
Topographic plotting functions for EEG data.

This module provides functions for creating publication-quality topographic
maps of EEG data with proper head outlines and interpolation.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .head import HeadOutline
from .interpolation import interpolate_topography


def plot_topomap(
    values: Union[Dict[str, float], np.ndarray],
    xy: np.ndarray,
    ch_names: Optional[List[str]] = None,
    *,
    ax: Optional[matplotlib.axes.Axes] = None,
    cmap: str = "RdBu_r",
    vlim: Optional[Tuple[float, float]] = None,
    sig_mask: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    head: bool = True,
    head_center: Optional[Tuple[float, float]] = None,
    head_radius: Optional[float] = None,
    head_kw: Optional[Dict[str, Any]] = None,
    grid_res: int = 256,
    sig_dot_size: float = 24,
    non_sig_dot_size: float = 6,
    sig_alpha: float = 0.9,
    non_sig_alpha: float = 0.3,
    extrapolate: bool = True,
    show_labels: bool = False,
    label_offset: Tuple[float, float] = (0.02, 0.02),
    label_fontsize: int = 8,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, Any]:
    """
    Plot an interpolated topographic scalp map.

    Parameters
    ----------
    values : dict | array-like
        Per-channel scalar values. If dict, keyed by channel name. If
        array-like, must match the length of *xy*. Channels missing from the
        dict, or NaN / infinite, are left out of the interpolation (their
        dots are still drawn); at least 3 finite channels are required.
    xy : ndarray (n, 2)
        2-D electrode positions.
    ch_names : list[str] | None
        Channel names corresponding to rows of *xy*. Required when
        *values* is a dict.
    ax : matplotlib Axes | None
        Target axes. A new figure is created when ``None``.
    cmap : str
        Matplotlib colourmap name.
    vlim : tuple (vmin, vmax) | None
        Colour limits. When ``None``, from the 97th percentile ``p`` of the
        absolute finite values (at least 1e-6): ``(0, p)`` if all values are
        >= 0, ``(-p, 0)`` if all are <= 0, else ``(-p, p)``.
    sig_mask : array-like of bool | None
        Boolean mask (same length as *xy*). ``True`` = significant
        (bold dot); ``False`` = non-significant (small dot).
        When ``None``, all electrodes get small dots.
    title : str | None
        Axes title.
    colorbar : bool
        Add a colour bar.
    colorbar_label : str | None
        Label for the colorbar (e.g., 't-statistic', 'Power (µV²)', 'Counts').
    head : bool
        Draw the head outline.
    head_center : tuple (x, y) | None
        Head outline centre. When ``None`` (with *head_radius*), the
        geometry comes from the plotted channels (see
        :class:`~eegtopo.HeadOutline`); pass :func:`~eegtopo.head_geometry`
        output so the outline does not depend on which channels are present.
    head_radius : float | None
        Head outline radius; see *head_center*.
    head_kw : dict | None
        Styling passed to :meth:`HeadOutline.draw` (e.g. ``linewidth``,
        ``color``, ``nose``, ``ears``).
    grid_res : int
        Interpolation grid resolution (default 256 for smooth coverage).
    sig_dot_size : float
        Size of significant electrode markers (default 24).
    non_sig_dot_size : float
        Size of non-significant electrode markers (default 6).
    sig_alpha : float
        Alpha for significant markers.
    non_sig_alpha : float
        Alpha for non-significant markers.
    extrapolate : bool
        Whether to add extrapolation points for smoother interpolation
        at the edges (default True).
    show_labels : bool
        Show electrode names next to the markers (default False).
    label_offset : tuple (x, y)
        Offset for electrode labels from the marker position, in data
        coordinates (default (0.02, 0.02)).
    label_fontsize : int
        Font size for electrode labels (default 8).

    Returns
    -------
    fig : Figure
    ax : Axes
    im : AxesImage
    """
    xy = np.asarray(xy, dtype=float)
    n = len(xy)

    # --- Resolve values to 1-D array ---
    if isinstance(values, dict):
        if ch_names is None:
            raise ValueError("ch_names required when values is a dict")
        data = np.array(
            [values.get(ch, np.nan) for ch in ch_names],
            dtype=float,
        )
    else:
        data = np.asarray(values, dtype=float)

    if len(data) != n:
        raise ValueError(f"len(values)={len(data)} != len(xy)={n}")

    # --- Create head outline ---
    head_outline = HeadOutline(xy, head_radius=head_radius, head_center=head_center)
    Xi, Yi, xmin, xmax, ymin, ymax = head_outline.get_grid_bounds(grid_res)

    # --- Interpolation (non-finite channels are left out) ---
    Zi = interpolate_topography(xy, data, Xi, Yi, extrapolate=extrapolate)

    # --- Auto vlim ---
    if vlim is None:
        finite = data[np.isfinite(data)]
        vmax = max(np.percentile(np.abs(finite), 97), 1e-6)
        if finite.min() >= 0:
            vlim = (0, vmax)
        elif finite.max() <= 0:
            vlim = (-vmax, 0)
        else:
            vlim = (-vmax, vmax)

    # Mask outside head circle for visualization
    # But keep interpolation data for electrodes outside head
    mask = head_outline.get_mask(Xi, Yi)
    Zi_masked = Zi.copy()
    Zi_masked[~mask] = np.nan

    # --- Plot ---
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    # Plot interpolated surface
    im = ax.imshow(
        Zi_masked,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        cmap=cmap,
        vmin=vlim[0],
        vmax=vlim[1],
        aspect="equal",
        interpolation="bilinear",
    )

    # Head outline
    if head:
        head_outline.draw(ax, **(head_kw or {}))

    # Electrode markers - always show all electrodes
    if sig_mask is not None:
        sig_mask = np.asarray(sig_mask, dtype=bool)
        non_sig = ~sig_mask
        ax.scatter(
            xy[non_sig, 0],
            xy[non_sig, 1],
            s=non_sig_dot_size,
            color="black",
            alpha=non_sig_alpha,
            zorder=4,
        )
        if sig_mask.any():
            ax.scatter(
                xy[sig_mask, 0],
                xy[sig_mask, 1],
                s=sig_dot_size,
                color="black",
                alpha=sig_alpha,
                zorder=5,
            )
    else:
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=non_sig_dot_size,
            color="black",
            alpha=non_sig_alpha,
            zorder=4,
        )

    # Electrode labels
    if show_labels and ch_names is not None:
        for (x, y), ch in zip(xy, ch_names):
            ax.text(
                x + label_offset[0],
                y + label_offset[1],
                ch,
                fontsize=label_fontsize,
                ha="left",
                va="bottom",
                color="black",
                zorder=10,
            )

    if title:
        ax.set_title(title, fontsize=12, pad=10)
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
        if colorbar_label:
            cbar.set_label(colorbar_label)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")

    return fig, ax, im


def plot_cluster_topomap(
    t_obs: np.ndarray,
    clusters: List[np.ndarray],
    cluster_pv: np.ndarray,
    xy: np.ndarray,
    ch_names: Optional[List[str]] = None,
    alpha: float = 0.05,
    **kwargs,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, Any]:
    """
    Plot a t-statistic topographic map with significant clusters highlighted.

    Parameters
    ----------
    t_obs : ndarray (n_channels,)
        Observed t-statistics.
    clusters : list of ndarray
        Boolean cluster masks from cluster_permutation_test.
    cluster_pv : ndarray
        p-value per cluster.
    xy : ndarray (n, 2)
        2-D electrode positions.
    ch_names : list[str] | None
        Channel names.
    alpha : float
        A cluster is significant when its p-value is below *alpha*.
    **kwargs
        Passed to plot_topomap.

    Returns
    -------
    fig, ax, im
    """
    sig_mask = np.zeros(len(t_obs), dtype=bool)
    for cluster, pv in zip(clusters, cluster_pv):
        if pv < alpha:
            sig_mask |= cluster

    return plot_topomap(
        t_obs,
        xy,
        ch_names=ch_names,
        sig_mask=sig_mask if sig_mask.any() else None,
        **kwargs,
    )
