"""Metadata-stratified statistics on endothelial neighbourhood composition.

Ports ``20250123_Metadata_Endothelial_Neighborhoods_RC.ipynb``: per-donor
neighbourhood composition, its linear regression against BMI, and comparisons
by history of hypertension and by tissue (small bowel vs colon), including the
Kruskal-Wallis / Dunn's post-hoc test across tissue-by-neighbourhood groups.
"""

from __future__ import annotations

import pandas as pd
import scikit_posthocs as sp
from scipy.stats import kruskal, linregress, shapiro, ttest_ind, ttest_rel


def group_percentages(cells: pd.DataFrame, group_col: str, category_col: str) -> pd.DataFrame:
    """Percent composition of ``category_col`` within each ``group_col`` value.

    One row per ``group_col`` value (e.g. donor, or tissue area), one column
    per category, values are percentages that sum to 100 per row. Matches
    cell 8 (``area_pcts``) and the ``view`` table feeding cell 24's area plot.
    """
    rows = {}
    for group_value, sub in cells.groupby(group_col, observed=True):
        rows[group_value] = sub[category_col].value_counts(normalize=True) * 100
    return pd.DataFrame(rows).transpose()


def bmi_linear_fits(percentages: pd.DataFrame, bmi: pd.Series) -> pd.DataFrame:
    """Ordinary least-squares fit of each neighbourhood's percentage against BMI.

    Matches cell 10: one row per neighbourhood with slope ``m``, intercept
    ``b``, correlation ``r``, its square ``r2``, p-value ``p`` and
    ``std_err``, via ``scipy.stats.linregress``.
    """
    x = [float(v) for v in bmi]
    fits = {}
    for col in percentages.columns:
        y = [float(v) for v in percentages[col]]
        m, b, r, p, std_err = linregress(x, y)
        fits[col] = {"m": m, "b": b, "r": r, "p": p, "std_err": std_err}
    out = pd.DataFrame(fits).transpose()
    out["r2"] = out["r"] ** 2
    return out


def two_group_ttest(
    group_a: pd.Series, group_b: pd.Series, paired: bool = False
) -> tuple[float, float]:
    """Independent (default) or paired t-test between two value series.

    Matches the hypertension comparison (independent, cell 12) and the small
    bowel vs colon comparison (paired by donor, cell 18).
    """
    if paired:
        t_stat, p_value = ttest_rel(group_a, group_b)
    else:
        t_stat, p_value = ttest_ind(group_a, group_b)
    return float(t_stat), float(p_value)


def long_percentages_by_replicate(
    data: pd.DataFrame,
    grouping: str,
    replicate: str,
    category_col: str,
    category_values: list[str] | None = None,
) -> pd.DataFrame:
    """Long-format percentages of ``category_col`` per (``grouping``, ``replicate``).

    Matches ``swarm_box``'s data preparation (cells 13, 19 of notebook 1):
    for every combination of ``grouping`` (e.g. hypertension yes/no) and
    ``replicate`` (e.g. donor) that actually occurs in ``data``, the percent
    of each ``category_col`` value. Built as an explicit per-group loop
    (rather than the ``SeriesGroupBy.value_counts`` shorthand) because that
    shorthand introduces unobserved (grouping, replicate) combinations with a
    manufactured 0% row once ``category_col`` is categorical, which the
    notebook's original ``.groupby(...).apply(lambda x: x[...].value_counts())``
    does not.
    """
    working = data.copy()
    working[category_col] = working[category_col].astype("category")
    records = []
    for (group_value, replicate_value), sub in working.groupby(
        [grouping, replicate], observed=True
    ):
        counts = sub[category_col].value_counts(normalize=True, sort=False) * 100
        for category_value, pct in counts.items():
            if category_values is not None and category_value not in category_values:
                continue
            records.append(
                {
                    grouping: group_value,
                    replicate: replicate_value,
                    category_col: category_value,
                    "percentage": pct,
                }
            )
    return pd.DataFrame.from_records(records)


def shapiro_normality(groups: dict[tuple, pd.Series], alpha: float = 0.05) -> pd.DataFrame:
    """Shapiro-Wilk normality test for each named group.

    Matches cell 21: returns one row per group key with ``statistic``,
    ``p_value`` and ``normal`` (``p_value >= alpha``).
    """
    rows = []
    for key, values in groups.items():
        stat, p = shapiro(values)
        rows.append({"group": key, "statistic": stat, "p_value": p, "normal": p >= alpha})
    return pd.DataFrame(rows)


def kruskal_dunn(
    long_df: pd.DataFrame, value_col: str, group_col: str
) -> tuple[float, float, pd.DataFrame]:
    """Kruskal-Wallis test across groups, with Dunn's post-hoc pairwise test.

    Matches cell 22: ``group_col`` names each (neighbourhood, tissue) group;
    returns the Kruskal-Wallis ``(H, p)`` and the full Dunn's pairwise
    p-value matrix.
    """
    groups = [g[value_col].values for _, g in long_df.groupby(group_col)]
    h_stat, p_value = kruskal(*groups)
    dunn = sp.posthoc_dunn(long_df, val_col=value_col, group_col=group_col)
    return float(h_stat), float(p_value), dunn
