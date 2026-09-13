"""Rings analysis: endothelial neighbourhood composition at increasing radii.

Ports the "Rings Analysis" and "Donor-Specific Rings Analysis" sections of
``20250820_SingleCell_NeighborhoodAnalysis_Endothelial_HuBMAP.ipynb``: for each
named endothelial neighbourhood and each window radius ``k``, cluster that
neighbourhood's k-NN windows into a single centroid (matching the notebook's
``MiniBatchKMeans(n_clusters=1)``) and read off its fold change and percent
composition. Comparing composition at a small ``k`` (e.g. 5) against a large
``k`` (e.g. 300) shows how far each neighbourhood's endothelial enrichment
extends spatially.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .neighborhoods import cluster_single_centroid, fold_change, niche_composition

__all__ = [
    "cluster_single_centroid",
    "donor_ratio_tables",
    "donor_rings_profiles",
    "normalize_by_subregion",
    "rings_profile",
    "subregion_percentages",
]


def rings_profile(
    windows_by_k: dict[int, pd.DataFrame],
    ks: list[int],
    neighborhood_col: str,
    neighborhood_order: list[str],
    sum_cols: list[str],
    tissue_averages: np.ndarray,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fold change and niche composition for every (neighbourhood, k) pair.

    ``windows_by_k[k]`` must already carry a ``neighborhood_col`` column
    (endothelial cells labelled with their merged neighbourhood, everything
    else ``NaN``/absent). Rows are stacked neighbourhood-major, matching the
    notebook's nested loop order, with the radius ``k`` as the index and a
    ``"Endothelial Neighborhood"`` column naming the neighbourhood.
    """
    fc_rows, niche_rows = [], []
    for nbhd in neighborhood_order:
        for k in ks:
            subset = windows_by_k[k]
            subset = subset[subset[neighborhood_col] == nbhd]
            if subset.empty:
                continue
            centroid = cluster_single_centroid(subset[sum_cols].values, random_state)
            fc = fold_change(centroid, tissue_averages, sum_cols).rename(index={0: k})
            niche = niche_composition(centroid, sum_cols).rename(index={0: k})
            fc["Endothelial Neighborhood"] = nbhd
            niche["Endothelial Neighborhood"] = nbhd
            fc_rows.append(fc)
            niche_rows.append(niche)
    return pd.concat(fc_rows), pd.concat(niche_rows)


def donor_rings_profiles(
    windows_by_k: dict[int, pd.DataFrame],
    donor_by_index: pd.Series,
    ks: list[int],
    neighborhood_col: str,
    neighborhood_order: list[str],
    sum_cols: list[str],
    tissue_averages: np.ndarray,
    random_state: int = 0,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """``rings_profile`` restricted to each donor in turn.

    ``donor_by_index`` maps a cell index (matching ``windows_by_k[k]``'s
    index) to its donor id. Window compositions themselves are unchanged
    (each ``unique_region`` already belongs to a single donor); this just
    selects that donor's rows before clustering.
    """
    out = {}
    for donor in donor_by_index.unique():
        donor_index = donor_by_index[donor_by_index == donor].index
        donor_windows = {
            k: df.loc[df.index.intersection(donor_index)] for k, df in windows_by_k.items()
        }
        out[donor] = rings_profile(
            donor_windows,
            ks,
            neighborhood_col,
            neighborhood_order,
            sum_cols,
            tissue_averages,
            random_state,
        )
    return out


def donor_ratio_tables(
    niche_by_donor: dict[str, pd.DataFrame],
    k_small: int,
    k_large: int,
    cell_type_col: str = "Endothelial",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-donor ratio of ``cell_type_col`` enrichment at ``k_large`` vs ``k_small``.

    Returns ``(combined_ratio_df, combined_at_k_large_df)``: the ratio table
    (percent at ``k_large`` divided by percent at ``k_small``) and the raw
    percent-at-``k_large`` table, each with one row per (donor, neighbourhood)
    and a ``"Donor"`` column.
    """
    ratio_rows, abs_rows = [], []
    for donor, niche_comb in niche_by_donor.items():
        # `.loc[[k]]` (list form) always returns a DataFrame, even when only
        # one row happens to match; `.loc[k]` degrades to a Series (and the
        # column lookup below to a bare scalar) whenever a single
        # neighbourhood is in play, e.g. in small tests or a subsetted run.
        endo_nbhd = niche_comb.loc[[k_small]]["Endothelial Neighborhood"].reset_index(drop=True)
        numerator = niche_comb.loc[[k_large]][cell_type_col].reset_index(drop=True)
        denominator = niche_comb.loc[[k_small]][cell_type_col].reset_index(drop=True)
        abs_rows.append(pd.concat([endo_nbhd, numerator], axis=1).assign(Donor=donor))
        ratio_rows.append(
            pd.concat([endo_nbhd, (numerator / denominator).rename(cell_type_col)], axis=1).assign(
                Donor=donor
            )
        )
    combined_ratio_df = pd.concat(ratio_rows, ignore_index=True)
    combined_at_k_large_df = pd.concat(abs_rows, ignore_index=True)
    return combined_ratio_df, combined_at_k_large_df


def subregion_percentages(
    cells: pd.DataFrame, cell_type_col: str, cell_type_value: str, subregion_col: str
) -> pd.DataFrame:
    """Percent of ``cell_type_value`` cells within each ``subregion_col`` value.

    Matches cell 73's ``total_counts``/``endo_counts``/``endo_percent``.
    """
    total_counts = cells[subregion_col].value_counts()
    type_counts = cells.loc[cells[cell_type_col] == cell_type_value, subregion_col].value_counts()
    pct = (type_counts / total_counts * 100).reset_index()
    pct.columns = ["Subregion", "Subregion Endothelial %"]
    return pct


def normalize_by_subregion(
    combined_at_k_df: pd.DataFrame,
    neighborhood_to_subregion: dict[str, str],
    subregion_pcts: pd.DataFrame,
    cell_type_col: str = "Endothelial",
) -> pd.DataFrame:
    """Normalise each neighbourhood's cell-type percentage by its subregion baseline.

    Matches cell 73: maps each neighbourhood to a subregion (mucosa,
    submucosa, muscularis externa), joins that subregion's overall percentage
    of ``cell_type_col`` cells, and divides to get ``"Normalized ... %"``.
    """
    df = combined_at_k_df.copy()
    df["Subregion"] = df["Endothelial Neighborhood"].map(neighborhood_to_subregion)
    df = df.merge(subregion_pcts, on="Subregion", how="left")
    df[f"Normalized {cell_type_col} %"] = df[cell_type_col] / df["Subregion Endothelial %"]
    return df
