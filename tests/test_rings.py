import numpy as np
import pandas as pd
import pytest

from vasculature_neighborhoods.rings import (
    donor_ratio_tables,
    normalize_by_subregion,
    rings_profile,
    subregion_percentages,
)


def _windows_by_k():
    # Two neighbourhoods, two k values; a handful of rows per (nbhd, k).
    out = {}
    for k in (5, 10):
        out[k] = pd.DataFrame(
            {
                "Endothelial Neighborhood": ["Nbhd A"] * 3 + ["Nbhd B"] * 3,
                "A": [1.0, 2.0, 3.0, 5.0, 6.0, 7.0],
                "B": [4.0, 3.0, 2.0, 1.0, 1.0, 1.0],
            }
        )
    return out


def test_rings_profile_indexes_rows_by_k_and_tags_neighborhood():
    windows_by_k = _windows_by_k()
    fc, niche = rings_profile(
        windows_by_k,
        ks=[5, 10],
        neighborhood_col="Endothelial Neighborhood",
        neighborhood_order=["Nbhd A", "Nbhd B"],
        sum_cols=["A", "B"],
        tissue_averages=np.array([2.0, 2.0]),
        random_state=0,
    )
    assert set(fc.index) == {5, 10}
    assert set(fc["Endothelial Neighborhood"]) == {"Nbhd A", "Nbhd B"}
    assert len(fc) == 4  # 2 neighbourhoods x 2 k values
    assert set(niche.index) == {5, 10}


def test_rings_profile_skips_absent_neighborhoods():
    windows_by_k = _windows_by_k()
    fc, _ = rings_profile(
        windows_by_k,
        ks=[5],
        neighborhood_col="Endothelial Neighborhood",
        neighborhood_order=["Nbhd A", "Missing"],
        sum_cols=["A", "B"],
        tissue_averages=np.array([2.0, 2.0]),
        random_state=0,
    )
    assert list(fc["Endothelial Neighborhood"]) == ["Nbhd A"]


def test_donor_ratio_tables_divides_large_by_small_k():
    niche_by_donor = {
        "D1": pd.DataFrame(
            {"Endothelial Neighborhood": ["Nbhd A", "Nbhd A"], "Endothelial": [10.0, 20.0]},
            index=[5, 10],
        )
    }
    ratio_df, at_large_df = donor_ratio_tables(niche_by_donor, k_small=5, k_large=10)
    assert ratio_df.loc[0, "Endothelial"] == pytest.approx(2.0)
    assert at_large_df.loc[0, "Endothelial"] == pytest.approx(20.0)
    assert ratio_df.loc[0, "Donor"] == "D1"


def test_subregion_percentages_matches_hand_computation():
    cells = pd.DataFrame(
        {
            "Cell Type": ["Endothelial", "Other", "Endothelial", "Other"],
            "Tissue Unit": ["Mucosa", "Mucosa", "Submucosa", "Submucosa"],
        }
    )
    pct = subregion_percentages(cells, "Cell Type", "Endothelial", "Tissue Unit")
    pct = pct.set_index("Subregion")["Subregion Endothelial %"]
    assert pct["Mucosa"] == pytest.approx(50.0)
    assert pct["Submucosa"] == pytest.approx(50.0)


def test_normalize_by_subregion_divides_by_baseline():
    combined = pd.DataFrame({"Endothelial Neighborhood": ["Nbhd A"], "Endothelial": [10.0]})
    mapping = {"Nbhd A": "Mucosa"}
    subregion_pcts = pd.DataFrame({"Subregion": ["Mucosa"], "Subregion Endothelial %": [5.0]})
    out = normalize_by_subregion(combined, mapping, subregion_pcts)
    assert out.loc[0, "Normalized Endothelial %"] == pytest.approx(2.0)
