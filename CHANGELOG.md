# Changelog

Notable changes to eegtopo. The format is [Keep a Changelog](https://keepachangelog.com/en/2.0.0/) and the versions are [semantic](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.1.2]

Not yet released; the date is added when it is tagged. Topomaps drawn from complete data with an explicit `vlim` and no head arguments are bit-for-bit identical to 0.1.1.

### Fixed

- **Missing channels no longer pull a topomap towards zero.** NaN or infinite values, and channels absent from a values dict, were interpolated as 0 (a flat map at 6.0 dipped to -0.12 around one missing channel). They are now left out of the interpolation; their dots are still drawn, the head outline is still built from every channel passed in, and automatic colour limits use the finite values only. At least 3 finite channels are needed, otherwise `plot_topomap` and `interpolate_topography` raise `ValueError`. Maps without missing values render exactly as before.
- **Paired tests refuse incomplete or duplicated data instead of inventing values.** A missing (subject, channel) cell was filled with 0, so its difference became "value - 0"; duplicate rows were silently averaged; subjects lacking one condition were silently dropped. `build_difference_matrix`, and so `run_cluster_test` and `run_cluster_analysis`, now raise `ValueError` naming the duplicated cell or listing the first 10 missing or NaN cells.
- **One-tailed cluster tests use the one-tailed threshold.** The default cluster-forming threshold was always the positive two-tailed critical t, so `tail=-1` failed with MNE's "incompatible tail and threshold signs" and `tail=1` used a stricter cut than p < 0.05. It is now t(1 - 0.05) for `tail=1`, its negative for `tail=-1`, and t(1 - 0.025) for `tail=0` (unchanged).
- **TFCE with `tail=-1` is refused rather than run with wrong p-values.** MNE (checked 1.12.1 and 1.13.2) inverts TFCE p-values for `tail=-1`. `threshold="tfce"` or a TFCE dict with `tail=-1` now raises `ValueError` saying to swap the conditions and use `tail=1`.
- **A named column is never replaced by a guessed one.** Without `value_col`, a DataFrame with both `count` and `density` columns was analysed on `count` without a word, and a missing subject column could fall back to `id`. The named columns (defaults `subject`, `channel`, `value`, `condition`; `condition_col=None` for none) must now exist, or `TopographicAnalysis` raises `ValueError` listing the available columns. Renaming onto a different existing column of the internal name (e.g. `value_col="power"` while a `value` column also exists) raises too. Code that relied on guessing must pass its column names.
- **`plot_mean_topomap` accepts `vlim`.** It raised `TypeError` (duplicate keyword) because it always passed its own limits.
- **`HeadOutline(xy, head_radius=r)` draws radius `r`.** An explicit radius smaller than half the largest electrode distance was silently enlarged.

### Added

- **A head outline that stays put across datasets.** `head_geometry(montage_name, exclude=None)` returns the outline centre and radius computed from the whole montage, and `plot_topomap(head_center=..., head_radius=...)` and `HeadOutline(..., head_center=...)` accept them. Without them the outline is computed from the plotted channels exactly as in 0.1.1 (dropping the frontal channels moved it by 12% of the radius). `TopographicAnalysis` computes it once (`.head_center`, `.head_radius`) and uses it for every plot.
- **Head outline styling from `plot_topomap`**: `head_kw={"linewidth": 0.5, "color": "grey"}` (also `nose`, `ears`).
- **A chosen significance level throughout.** `alpha=` on `run_cluster_test` and `run_cluster_analysis` (default 0.05) is stored as `ClusterResults.alpha` and used by `summary()` and `plot_cluster_results`; 0.05 was hard-coded in the test and the summary. `ClusterResults.tfce` records a TFCE run, and plot titles then count significant channels instead of clusters.

### Changed

- **Automatic colour limits follow the sign of the data.** With `vlim=None`: all values >= 0 gives (0, p), all <= 0 gives (-p, 0), mixed gives (-p, p) as before, where p is the 97th percentile of the absolute values (at least 1e-6). This applies to `plot_topomap`, `plot_cluster_results` and `plot_mean_topomap`, whose separate rule (0, 1.1 x 95th percentile) is gone. Pass `vlim` to keep a fixed scale.
- **`TopographicAnalysis` plots draw the montage-wide head outline.** For data with every montage channel (minus `exclude_channels`) this is the outline 0.1.1 drew; it differs when channels are missing or when `.xy` has been replaced.
- **Electrode labels default to font size 8.** The 0.1.1 wheel on PyPI already used 8 while the `v0.1.1` tag in git said 4, so PyPI users see no change.
- **Python 3.10 or newer is required**, as current MNE needs; 3.10 to 3.13 are tested. 0.1.1 declared 3.8.

### Removed

- **The `nan_fill` argument of `plot_topomap`.** Missing values are left out instead of filled; passing `nan_fill` raises `TypeError`. To fill as before, do it yourself: `np.nan_to_num(values, nan=0.0)`.
- **`setup.py` and the token-based `publish.sh`.** `pyproject.toml` is the only packaging metadata, and releases are published from the tagged GitHub workflow after the tests pass.

## [0.1.1] - 2026-02-13

First release on PyPI.
