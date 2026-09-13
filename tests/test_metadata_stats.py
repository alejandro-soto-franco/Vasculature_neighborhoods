import pandas as pd
import pytest

from vasculature_neighborhoods.metadata_stats import (
    bmi_linear_fits,
    group_percentages,
    kruskal_dunn,
    long_percentages_by_replicate,
    shapiro_normality,
    two_group_ttest,
)


def test_group_percentages_sums_to_100_per_group():
    cells = pd.DataFrame(
        {"donor": ["D1", "D1", "D1", "D2", "D2"], "nbhd": ["A", "A", "B", "A", "B"]}
    )
    pct = group_percentages(cells, "donor", "nbhd")
    assert pct.loc["D1"].sum() == pytest.approx(100.0)
    assert pct.loc["D2"].sum() == pytest.approx(100.0)
    assert pct.loc["D1", "A"] == pytest.approx(200 / 3)


def test_bmi_linear_fits_recovers_perfect_line():
    percentages = pd.DataFrame({"A": [0.0, 10.0, 20.0, 30.0]})
    bmi = pd.Series([20.0, 22.0, 24.0, 26.0])
    fits = bmi_linear_fits(percentages, bmi)
    assert fits.loc["A", "m"] == pytest.approx(5.0)
    assert fits.loc["A", "r2"] == pytest.approx(1.0)


def test_two_group_ttest_independent_vs_paired_differ():
    a = pd.Series([1.0, 2.0, 3.0])
    b = pd.Series([1.5, 2.5, 3.5])
    t_ind, p_ind = two_group_ttest(a, b, paired=False)
    t_pair, p_pair = two_group_ttest(a, b, paired=True)
    assert t_ind != t_pair or p_ind != p_pair


def test_long_percentages_by_replicate_shape():
    data = pd.DataFrame(
        {
            "grp": [True, True, False, False],
            "rep": [0, 0, 1, 1],
            "cat": ["A", "B", "A", "B"],
        }
    )
    long_df = long_percentages_by_replicate(data, "grp", "rep", "cat")
    assert set(long_df.columns) == {"grp", "rep", "cat", "percentage"}
    assert len(long_df) == 4


def test_shapiro_normality_flags_uniform_as_non_normal_or_normal_consistently():
    groups = {("A",): pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0])}
    result = shapiro_normality(groups, alpha=0.05)
    assert set(result.columns) == {"group", "statistic", "p_value", "normal"}
    # pandas-stubs types a single `.loc[row, col]` cell access as a huge
    # union across every possible dtype; the values here are genuinely
    # bool/float, confirmed by the assertion itself.
    assert result.loc[0, "normal"] == (result.loc[0, "p_value"] >= 0.05)  # pyrefly: ignore


def test_kruskal_dunn_detects_group_difference():
    long_df = pd.DataFrame(
        {
            "percentage": [1, 2, 1, 2, 50, 51, 49, 52],
            "group_id": ["a", "a", "a", "a", "b", "b", "b", "b"],
        }
    )
    h_stat, p_value, dunn = kruskal_dunn(long_df, "percentage", "group_id")
    assert h_stat > 0
    assert p_value < 0.05
    assert float(dunn.loc["a", "b"]) < 0.05  # pyrefly: ignore
