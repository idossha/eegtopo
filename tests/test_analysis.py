"""DataFrame interface and cluster statistics (2026-11-22).

Pins: missing or duplicated cells raise instead of being zero-filled or
averaged (audit #1); the default cluster-forming threshold follows ``tail``,
tail=-1 runs end to end, and TFCE with tail=-1 (wrong p-values in MNE) is refused (#2); alpha is one value carried
from the test to the summary and the plot (#7); named columns are required,
never guessed, and renaming cannot collide (#8).

Numbers: t critical values for df = 9 are the printed Student-t table values
t(0.975, 9) = 2.262 and t(0.95, 9) = 1.833. The synthetic effect (a - b = -3
on 8 frontal channels, noise sd 0.5, 12 subjects) is large enough that a
one-tailed cluster test must find it. Reproduce: ``pytest tests/test_analysis.py``.
"""

import numpy as np
import pandas as pd
import pytest

import mne
from eegtopo import ClusterResults, TopographicAnalysis, cluster_permutation_test, create_mne_info

from conftest import CHANNELS_1020, FRONTAL, MONTAGE_1020, long_df

T_975_DF9 = 2.262
T_95_DF9 = 1.833
TABLE_TOL = 5e-4  # the table is printed to 3 decimals


def analysis(df, **kw):
    return TopographicAnalysis(df, montage_name=MONTAGE_1020, **kw)


# ---- #1 difference matrix -------------------------------------------------


def test_difference_matrix_is_a_minus_b():
    df = long_df(n_subjects=3)
    diff, subjects = analysis(df).build_difference_matrix("a", "b")
    a = df[df.condition == "a"].pivot(index="subject", columns="channel", values="value")
    b = df[df.condition == "b"].pivot(index="subject", columns="channel", values="value")
    names = analysis(df).ordered_channels
    assert subjects == ["sub-00", "sub-01", "sub-02"]
    assert np.allclose(diff, (a - b).loc[subjects, names].to_numpy())


def test_missing_cell_raises_naming_it():
    df = long_df(n_subjects=6)
    df = df[~((df.subject == "sub-03") & (df.channel == "Cz") & (df.condition == "b"))]
    with pytest.raises(ValueError, match=r"sub-03.*Cz.*'b'"):
        analysis(df).build_difference_matrix("a", "b")


def test_nan_value_counts_as_missing():
    df = long_df(n_subjects=6)
    df.loc[(df.subject == "sub-01") & (df.channel == "Pz") & (df.condition == "a"), "value"] = np.nan
    with pytest.raises(ValueError, match=r"sub-01.*Pz"):
        analysis(df).build_difference_matrix("a", "b")


def test_subject_missing_a_condition_is_not_dropped_silently():
    df = long_df(n_subjects=6)
    df = df[~((df.subject == "sub-05") & (df.condition == "b"))]
    with pytest.raises(ValueError, match="sub-05"):
        analysis(df).build_difference_matrix("a", "b")


def test_duplicate_rows_raise_naming_them():
    df = long_df(n_subjects=6)
    dup = df[(df.subject == "sub-02") & (df.channel == "Fz") & (df.condition == "a")]
    df = pd.concat([df, dup.assign(value=dup.value + 5)])
    with pytest.raises(ValueError, match=r"sub-02.*Fz.*'a'"):
        analysis(df).build_difference_matrix("a", "b")


# ---- #2 thresholds --------------------------------------------------------


@pytest.mark.parametrize("tail, expected", [(0, T_975_DF9), (1, T_95_DF9), (-1, -T_95_DF9)])
def test_default_threshold_follows_tail(monkeypatch, tail, expected):
    seen = {}

    def fake(data, **kw):
        seen.update(kw)
        return np.zeros(data.shape[1]), [], np.array([]), None

    monkeypatch.setattr(mne.stats, "permutation_cluster_1samp_test", fake)
    info, names = create_mne_info(CHANNELS_1020, MONTAGE_1020)
    cluster_permutation_test(np.zeros((10, len(names))), info, tail=tail)
    assert seen["threshold"] == pytest.approx(expected, abs=TABLE_TOL)


def test_negative_tail_runs_end_to_end():
    a = analysis(long_df(effect=-3.0))
    res = a.run_cluster_test("a", "b", tail=-1, n_permutations=200)
    assert res.n_sig_clusters > 0
    assert set(res.sig_channels) <= set(FRONTAL)
    assert np.all(res.t_obs[res.sig_mask] < 0)


@pytest.mark.parametrize("threshold", ["tfce", {"start": 0, "step": -0.2}])
def test_tfce_with_negative_tail_is_refused(threshold):
    # MNE 1.12.1/1.13.2 rank positive TFCE scores as negative for tail=-1: frontal p = 1.0,
    # untouched channels p = 0.005 (checked by hand against tail=+1 on the negated data)
    a = analysis(long_df(effect=-3.0))
    with pytest.raises(ValueError, match="use tail=1"):
        a.run_cluster_test("a", "b", tail=-1, threshold=threshold, n_permutations=200)


# ---- #7 alpha -------------------------------------------------------------


def test_alpha_is_carried_to_summary_and_plot():
    a = analysis(long_df(n_subjects=2))
    n = len(a.ordered_channels)
    c1, c2 = np.zeros(n, bool), np.zeros(n, bool)
    c1[:3], c2[5:9] = True, True
    res = ClusterResults(
        t_obs=np.ones(n), clusters=[c1, c2], cluster_pv=np.array([0.03, 0.08]),
        sig_mask=c1 | c2, n_sig_clusters=2, sig_channels=[], alpha=0.1,
    )
    assert res.summary()["significant"].tolist() == [True, True]

    _, ax, _ = a.plot_cluster_results(res, colorbar=False)
    bold = [c for c in ax.collections if c.get_zorder() == 5]
    assert len(bold) == 1 and len(bold[0].get_offsets()) == 7


def test_run_cluster_test_uses_alpha():
    res = analysis(long_df(effect=-3.0)).run_cluster_test("a", "b", n_permutations=200, alpha=0.5)
    assert res.alpha == 0.5
    assert res.summary()["significant"].tolist() == (res.cluster_pv < 0.5).tolist()


def test_tfce_title_counts_channels():
    a = analysis(long_df(effect=-3.0))
    res = a.run_cluster_test("a", "b", threshold="tfce", n_permutations=200)
    _, ax, _ = a.plot_cluster_results(res, colorbar=False)
    assert "significant channels" in ax.get_title()


# ---- #6 one colour-limit rule -----------------------------------------------


def test_mean_topomap_uses_the_plot_topomap_rule():
    df = long_df(n_subjects=2)
    df["value"] = df.channel.map({ch: i + 1.0 for i, ch in enumerate(CHANNELS_1020)})  # means 1..31
    a = analysis(df)
    _, _, im = a.plot_mean_topomap("a", colorbar=False)
    assert im.get_clim() == pytest.approx((0.0, 1 + 0.97 * 30))  # 97th pct of 1..31, by hand

    _, _, im = a.plot_mean_topomap("a", colorbar=False, vlim=(0, 1))
    assert im.get_clim() == (0, 1)


# ---- #8 columns -----------------------------------------------------------


def test_value_column_is_not_guessed():
    df = long_df(n_subjects=2).rename(columns={"value": "count"}).assign(density=1.0)
    with pytest.raises(ValueError, match=r"\['value'\].*count.*density"):
        analysis(df)


def test_subject_column_is_not_guessed():
    df = long_df(n_subjects=2).rename(columns={"subject": "id"})
    with pytest.raises(ValueError, match=r"\['subject'\]"):
        analysis(df)


def test_rename_collision_raises():
    df = long_df(n_subjects=2).assign(power=1.0)  # has both 'value' and 'power'
    with pytest.raises(ValueError, match="collide"):
        analysis(df, value_col="power")


def test_named_columns_and_no_condition():
    df = long_df(n_subjects=2).rename(columns={"subject": "pid", "value": "power"})
    df = df[df.condition == "a"].drop(columns="condition")
    a = analysis(df, subject_col="pid", value_col="power", condition_col=None)
    assert list(a.df.columns) == ["subject", "channel", "value"]
