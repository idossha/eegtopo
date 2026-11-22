"""
High-level DataFrame interface for topographic analysis.

This module provides easy-to-use functions for researchers who want to
perform cluster-based permutation testing on channel-level EEG data stored
in pandas DataFrames.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings

from .montage import get_channel_positions, create_mne_info, head_geometry
from .plotting import plot_topomap, plot_cluster_topomap
from .statistics import cluster_permutation_test


@dataclass
class ClusterResults:
    """Container for cluster permutation test results."""

    t_obs: np.ndarray
    clusters: List[np.ndarray]
    cluster_pv: np.ndarray
    sig_mask: np.ndarray
    n_sig_clusters: int
    sig_channels: List[str]
    alpha: float = 0.05
    tfce: bool = False  # with TFCE every "cluster" is a single channel

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of clusters (significant: p < alpha)."""
        rows = []
        for i, (cluster, pv) in enumerate(zip(self.clusters, self.cluster_pv)):
            rows.append(
                {
                    "cluster_id": i + 1,
                    "n_channels": int(cluster.sum()),
                    "p_value": pv,
                    "significant": pv < self.alpha,
                }
            )
        return pd.DataFrame(rows)


def _n_significant(results: ClusterResults) -> str:
    """'3 significant clusters', or channels under TFCE."""
    unit = "channels" if results.tfce else "clusters"
    return f"{results.n_sig_clusters} significant {unit}"


class TopographicAnalysis:
    """
    High-level interface for topographic EEG analysis.

    This class handles all the details of montage setup, cluster testing,
    and visualization. Just provide a DataFrame with your data.

    Parameters
    ----------
    df : pd.DataFrame
        One row per subject, channel and (optionally) condition. The columns
        named below must exist; nothing is guessed.

    subject_col : str
        Column name for subject IDs (default: 'subject')
    channel_col : str
        Column name for channel names (default: 'channel')
    value_col : str
        Column name for values (default: 'value')
    condition_col : str | None
        Column name for conditions (default: 'condition'); None when the
        data has no conditions.
    montage_name : str
        MNE montage name (default: 'GSN-HydroCel-256')
    exclude_channels : list[str] | None
        Channels to exclude from analysis

    Attributes
    ----------
    head_center, head_radius
        Head outline geometry from the whole montage minus
        *exclude_channels* (:func:`~eegtopo.head_geometry`), used by every
        plot method so the outline does not depend on the data's channels.

    Example
    -------
    >>> df = pd.DataFrame({
    ...     'subject': ['sub-01', 'sub-01', ...],
    ...     'channel': ['E1', 'E2', ...],
    ...     'condition': ['pre', 'post'],
    ...     'power': [1.2, 2.3, ...]
    ... })
    >>>
    >>> analysis = TopographicAnalysis(df, value_col='power')
    >>> results = analysis.run_cluster_test('pre', 'post')
    >>> analysis.plot_cluster_results(results)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        subject_col: str = "subject",
        channel_col: str = "channel",
        value_col: str = "value",
        condition_col: Optional[str] = "condition",
        montage_name: str = "GSN-HydroCel-256",
        exclude_channels: Optional[List[str]] = None,
    ):
        self.df = df.copy()
        self.subject_col = subject_col
        self.channel_col = channel_col
        self.value_col = value_col
        self.condition_col = condition_col
        self.montage_name = montage_name
        self.exclude_channels = exclude_channels or []

        # Standardize column names
        self._standardize_columns()

        # Setup montage
        self._setup_montage()

    def _standardize_columns(self):
        """Rename the named columns to the internal subject/channel/value/condition."""
        col_map = {
            self.subject_col: "subject",
            self.channel_col: "channel",
            self.value_col: "value",
        }
        if self.condition_col is not None:
            col_map[self.condition_col] = "condition"

        missing = [c for c in col_map if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"Columns not found: {missing}. Available: {list(self.df.columns)}"
            )

        # A different column already carrying an internal name would be duplicated
        clash = [
            f"{col!r} -> {new!r}"
            for col, new in col_map.items()
            if col != new and new in self.df.columns and new not in col_map
        ]
        if clash:
            raise ValueError(
                f"Renaming {clash} would collide with an existing column of that name; "
                "drop or rename it first."
            )

        self.df = self.df.rename(columns=col_map)

    def _setup_montage(self):
        """Setup electrode positions from montage."""
        all_channels = self.df["channel"].unique().tolist()

        self.ordered_channels, self.xy = get_channel_positions(
            all_channels,
            montage_name=self.montage_name,
            exclude=self.exclude_channels,
        )

        if not self.ordered_channels:
            raise ValueError(
                f"No valid channels found in montage '{self.montage_name}'. "
                f"Check your channel names match the montage."
            )

        # Build MNE info for cluster testing
        self.info, _ = create_mne_info(
            all_channels,
            montage_name=self.montage_name,
            exclude=self.exclude_channels,
        )

        self.head_center, self.head_radius = head_geometry(
            self.montage_name, exclude=self.exclude_channels
        )

        # Filter data to valid channels
        self.df = self.df[self.df["channel"].isin(self.ordered_channels)]

    def build_difference_matrix(
        self,
        condition_a: str,
        condition_b: str,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build difference matrix for cluster testing.

        Parameters
        ----------
        condition_a, condition_b : str
            Condition labels to compare

        Returns
        -------
        diff_matrix : ndarray (n_subjects, n_channels)
        subjects : list[str]
            Subject IDs in the matrix

        Raises
        ------
        ValueError
            If a (subject, channel, condition) has more than one row, or if
            any subject lacks a finite value for any channel in either
            condition. Nothing is averaged, filled or dropped.
        """
        if "condition" not in self.df.columns:
            raise ValueError(
                "DataFrame must have 'condition' column for paired comparison"
            )

        pivots = {}
        for cond in (condition_a, condition_b):
            rows = self.df[self.df["condition"] == cond]
            dup = rows[rows.duplicated(["subject", "channel"], keep=False)]
            if len(dup):
                first = dup.iloc[0]
                raise ValueError(
                    f"{len(dup)} rows share a (subject, channel, condition), e.g. "
                    f"subject={first['subject']!r}, channel={first['channel']!r}, "
                    f"condition={cond!r}; aggregate them first."
                )
            pivots[cond] = rows.pivot(index="subject", columns="channel", values="value")

        subjects = pivots[condition_a].index.union(pivots[condition_b].index)
        if len(subjects) == 0:
            raise ValueError(
                f"No rows for conditions '{condition_a}' or '{condition_b}'. "
                f"Available: {sorted(self.df['condition'].unique().tolist())}"
            )

        gaps = []
        for cond in (condition_a, condition_b):
            pivots[cond] = pivots[cond].reindex(index=subjects, columns=self.ordered_channels)
            for i, j in zip(*np.nonzero(pivots[cond].isna().to_numpy())):
                gaps.append((subjects[i], self.ordered_channels[j], cond))
        if gaps:
            raise ValueError(
                f"{len(gaps)} (subject, channel, condition) cells are missing or NaN, "
                f"e.g. {gaps[:10]}; drop or impute them before testing."
            )

        diff = (pivots[condition_a] - pivots[condition_b]).to_numpy()
        return diff, list(subjects)

    def run_cluster_test(
        self,
        condition_a: str,
        condition_b: str,
        threshold: Optional[Union[float, str, Dict]] = None,
        n_permutations: int = 1024,
        tail: int = 0,
        seed: int = 42,
        n_jobs: int = 1,
        alpha: float = 0.05,
    ) -> ClusterResults:
        """
        Run cluster-based permutation test between two conditions.

        Parameters
        ----------
        condition_a, condition_b : str
            Condition labels to compare
        threshold : float | str | dict | None
            Cluster-forming threshold:
            - None or 'auto': critical t at p < 0.05 for the chosen *tail*
            - float: Direct threshold value (negative for tail=-1)
            - 'tfce': TFCE with start 0 and step 0.2 (not with tail=-1)
            - dict: MNE TFCE parameters
        n_permutations : int
            Number of permutations (default: 1024)
        tail : int
            Test tail (-1, 0, 1). Default 0 = two-tailed.
        seed : int
            Random seed for reproducibility
        n_jobs : int
            Number of parallel jobs
        alpha : float
            A cluster is significant when its p-value is below *alpha*
            (default 0.05); stored on the result.

        Returns
        -------
        ClusterResults object with test results
        """
        # Build difference matrix
        diff_matrix, subjects = self.build_difference_matrix(condition_a, condition_b)

        n_subj = diff_matrix.shape[0]
        if n_subj < 6:
            warnings.warn(
                f"Only {n_subj} subjects with both conditions. "
                "Cluster tests need at least 6 subjects for reliable results."
            )

        # Handle threshold parameter
        threshold_val = self._resolve_threshold(threshold)

        # Run cluster test
        t_obs, clusters, cluster_pv = cluster_permutation_test(
            diff_matrix,
            self.info,
            n_permutations=n_permutations,
            threshold=threshold_val,
            tail=tail,
            seed=seed,
            n_jobs=n_jobs,
        )

        # Build significance mask
        sig_mask = np.zeros(len(t_obs), dtype=bool)
        for cluster, pv in zip(clusters, cluster_pv):
            if pv < alpha:
                sig_mask |= cluster

        n_sig = int(np.sum(cluster_pv < alpha))
        sig_channels = [self.ordered_channels[i] for i in np.where(sig_mask)[0]]

        return ClusterResults(
            t_obs=t_obs,
            clusters=clusters,
            cluster_pv=cluster_pv,
            sig_mask=sig_mask,
            n_sig_clusters=n_sig,
            sig_channels=sig_channels,
            alpha=alpha,
            tfce=isinstance(threshold_val, dict),
        )

    def _resolve_threshold(
        self, threshold: Optional[Union[float, str, Dict]]
    ) -> Union[float, Dict]:
        """Resolve threshold specification to actual value."""
        if threshold is None:
            return None  # Will use default in core function

        if isinstance(threshold, (int, float)):
            return float(threshold)

        if isinstance(threshold, str):
            if threshold.lower() == "auto":
                return None
            elif threshold.lower() == "tfce":
                return dict(start=0, step=0.2)
            else:
                raise ValueError(f"Unknown threshold string: {threshold}")

        if isinstance(threshold, dict):
            return threshold

        raise ValueError(f"Invalid threshold type: {type(threshold)}")

    def plot_topomap(
        self,
        values: Union[Dict[str, float], np.ndarray],
        title: Optional[str] = None,
        cmap: str = "RdBu_r",
        vlim: Optional[Tuple[float, float]] = None,
        sig_mask: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """
        Plot topographic map.

        Parameters
        ----------
        values : dict or array
            Values per channel (dict keyed by channel name) or array matching
            the number of channels
        title : str
            Plot title
        cmap : str
            Colormap name
        vlim : tuple
            Color limits (vmin, vmax)
        sig_mask : array
            Boolean mask for significant channels
        **kwargs
            Additional arguments passed to plot_topomap

        Returns
        -------
        fig, ax, im
        """
        kwargs.setdefault("head_center", self.head_center)
        kwargs.setdefault("head_radius", self.head_radius)
        return plot_topomap(
            values,
            self.xy,
            ch_names=self.ordered_channels,
            title=title,
            cmap=cmap,
            vlim=vlim,
            sig_mask=sig_mask,
            **kwargs,
        )

    def plot_cluster_results(
        self,
        results: ClusterResults,
        title: Optional[str] = None,
        cmap: str = "RdBu_r",
        **kwargs,
    ):
        """
        Plot cluster test results with significant clusters highlighted.

        Parameters
        ----------
        results : ClusterResults
            Results from run_cluster_test()
        title : str
            Plot title
        cmap : str
            Colormap name
        **kwargs
            Additional arguments passed to plot_topomap

        Returns
        -------
        fig, ax, im
        """
        if title is None:
            title = f"t-statistic ({_n_significant(results)})"

        kwargs.setdefault("head_center", self.head_center)
        kwargs.setdefault("head_radius", self.head_radius)
        return plot_cluster_topomap(
            results.t_obs,
            results.clusters,
            results.cluster_pv,
            self.xy,
            ch_names=self.ordered_channels,
            alpha=results.alpha,
            title=title,
            cmap=cmap,
            **kwargs,
        )

    def plot_mean_topomap(
        self,
        condition: Optional[str] = None,
        title: Optional[str] = None,
        cmap: str = "YlOrRd",
        **kwargs,
    ):
        """
        Plot mean values across subjects for a condition.

        Parameters
        ----------
        condition : str | None
            Condition to plot. If None, plots grand mean across all data.
        title : str
            Plot title
        cmap : str
            Colormap name (default: 'YlOrRd' for positive values)
        **kwargs
            Additional arguments passed to plot_topomap

        Returns
        -------
        fig, ax, im
        """
        if condition is not None:
            df_cond = self.df[self.df["condition"] == condition]
            means = df_cond.groupby("channel")["value"].mean()
        else:
            means = self.df.groupby("channel")["value"].mean()

        values = means.to_dict()

        if title is None:
            title = "Mean values" + (f" - {condition}" if condition else "")

        return self.plot_topomap(values, title=title, cmap=cmap, **kwargs)


def run_cluster_analysis(
    df: pd.DataFrame,
    condition_a: str,
    condition_b: str,
    subject_col: str = "subject",
    channel_col: str = "channel",
    value_col: str = "value",
    condition_col: str = "condition",
    montage_name: str = "GSN-HydroCel-256",
    threshold: Optional[Union[float, str]] = None,
    n_permutations: int = 1024,
    seed: int = 42,
    alpha: float = 0.05,
    plot: bool = True,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    One-shot function to run complete cluster analysis.

    This is a convenience function that creates a TopographicAnalysis object,
    runs the cluster test, and optionally plots and saves results.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with subject, channel, value, and condition columns
    condition_a, condition_b : str
        Conditions to compare
    subject_col, channel_col, value_col, condition_col : str
        Column names in the DataFrame
    montage_name : str
        MNE montage name
    threshold : float | str | None
        Cluster-forming threshold
    n_permutations : int
        Number of permutations
    seed : int
        Random seed
    alpha : float
        Cluster significance level (p < alpha), default 0.05
    plot : bool
        Whether to create and return a plot
    save_path : str | None
        Path to save the plot (if plot=True)

    Returns
    -------
    dict with keys:
        'results': ClusterResults object
        'analysis': TopographicAnalysis object
        'figure': Figure object (if plot=True)
        'axes': Axes object (if plot=True)
        'image': AxesImage object (if plot=True)
    """
    # Create analysis object
    analysis = TopographicAnalysis(
        df,
        subject_col=subject_col,
        channel_col=channel_col,
        value_col=value_col,
        condition_col=condition_col,
        montage_name=montage_name,
    )

    # Run cluster test
    results = analysis.run_cluster_test(
        condition_a,
        condition_b,
        threshold=threshold,
        n_permutations=n_permutations,
        seed=seed,
        alpha=alpha,
    )

    output = {
        "results": results,
        "analysis": analysis,
    }

    # Plot if requested
    if plot:
        fig, ax, im = analysis.plot_cluster_results(
            results,
            title=f"{condition_a} vs {condition_b}\n({_n_significant(results)})",
        )
        output["figure"] = fig
        output["axes"] = ax
        output["image"] = im

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to: {save_path}")

    return output
