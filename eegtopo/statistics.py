"""
Cluster-based permutation testing for EEG topographic analysis.

This module provides statistical testing functionality using MNE-Python's
cluster-based permutation tests with spatial adjacency.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np


def cluster_permutation_test(
    data: np.ndarray,
    info: Any,
    n_permutations: int = 1024,
    threshold: Optional[Union[float, Dict]] = None,
    tail: int = 0,
    seed: int = 42,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Cluster-based permutation test on channel-level data.

    Wraps ``mne.stats.permutation_cluster_1samp_test`` with spatial
    adjacency derived from the channel montage.

    Parameters
    ----------
    data : ndarray (n_subjects, n_channels)
        Paired-difference values per subject and channel.
    info : mne.Info
        MNE Info with montage (used to compute channel adjacency).
    n_permutations : int
        Number of permutations.
    threshold : float | dict | None
        Cluster-forming threshold. Options:
        - None: critical t at p < 0.05 for the chosen *tail* (two-tailed:
          t(1 - 0.025); one-tailed: t(1 - 0.05), negated for tail=-1)
        - float: Direct t-value threshold (negative for tail=-1)
        - dict: TFCE parameters for MNE (e.g., {'start': 0, 'step': 0.2});
          not with tail=-1, see Raises
    tail : {-1, 0, 1}
        Tail of the test (0 = two-tailed).
    seed : int
        Random seed for reproducibility.
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    t_obs : ndarray (n_channels,)
        Observed t-statistic per channel.
    clusters : list of ndarray
        Boolean masks indicating cluster membership.
    cluster_pv : ndarray
        p-value per cluster.

    Raises
    ------
    ValueError
        For TFCE (dict *threshold*) with tail=-1: MNE scores TFCE as positive
        values but ranks them as negative ones for tail=-1, which inverts the
        p-values (MNE 1.12.1 and 1.13.2). Swap the conditions and use tail=1.
    """
    from mne.stats import permutation_cluster_1samp_test
    from mne.channels import find_ch_adjacency
    from scipy.stats import t as t_dist

    # shortcut: refuses TFCE with tail=-1 because MNE <= 1.13.2 inverts its p-values; lift once MNE signs TFCE scores
    if isinstance(threshold, dict) and tail == -1:
        raise ValueError(
            "TFCE with tail=-1 returns inverted p-values in MNE (checked 1.12.1 and 1.13.2); "
            "swap the two conditions (or negate the data) and use tail=1."
        )

    adjacency, _ = find_ch_adjacency(info, ch_type="eeg")

    if threshold is None:
        p = 0.025 if tail == 0 else 0.05
        threshold = t_dist.ppf(1 - p, df=data.shape[0] - 1)
        if tail == -1:
            threshold = -threshold

    t_obs, clusters, cluster_pv, _ = permutation_cluster_1samp_test(
        data,
        adjacency=adjacency,
        n_permutations=n_permutations,
        threshold=threshold,
        tail=tail,
        seed=seed,
        out_type="mask",
        n_jobs=n_jobs,
        verbose=False,
    )

    return t_obs, clusters, cluster_pv
