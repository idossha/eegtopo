"""Shared synthetic fixtures for the eegtopo tests (2026-11-22).

Everything is generated in memory from fixed seeds; no data files, no network
(MNE's standard montages ship inside the package). Run: ``pytest``.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

# MNE 1.13 renamed standard_1020 to colin27_1020 (same positions); the old name goes in 1.14
MONTAGE_1020 = "colin27_1020" if "colin27_1020" in mne.channels.get_builtin_montages() else "standard_1020"

# 31 10-20 channels covering the scalp, so the Delaunay adjacency is connected
CHANNELS_1020 = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "CP5",
    "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2",
]
FRONTAL = ["Fp1", "Fpz", "Fp2", "AF3", "AF4", "F3", "Fz", "F4"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def long_df(n_subjects=12, effect=0.0, channels=CHANNELS_1020, effect_channels=FRONTAL, seed=0):
    """Long-format rows (subject, channel, condition, value) for conditions 'a' and 'b'.

    a - b is ``effect`` on *effect_channels* and 0 elsewhere, plus N(0, 0.5) noise.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        for ch in channels:
            base = rng.normal(10.0, 1.0)
            shift = effect if ch in effect_channels else 0.0
            rows.append((f"sub-{s:02d}", ch, "a", base + shift + rng.normal(0, 0.5)))
            rows.append((f"sub-{s:02d}", ch, "b", base))
    return pd.DataFrame(rows, columns=["subject", "channel", "condition", "value"])


def disk_xy(n, seed=0):
    """n points scattered in the unit disk."""
    rng = np.random.default_rng(seed)
    r, a = np.sqrt(rng.uniform(0, 1, n)), rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(a), r * np.sin(a)])
