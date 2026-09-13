"""Endothelial cell-centric neighbourhood clustering and fold-change enrichment.

Ports the "Endothelial Cell Type-Focused Neighborhood Analysis" section of
``20250820_SingleCell_NeighborhoodAnalysis_Endothelial_HuBMAP.ipynb``: cluster
the k-NN composition windows centred on one cell type with MiniBatchKMeans,
then express each cluster centroid as a log2 fold change over the tissue-wide
cell-type average.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def cluster_windows(
    windows: pd.DataFrame,
    cell_type_col: str,
    cell_types: list[str],
    sum_cols: list[str],
    n_clusters: int,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """K-means-cluster the composition windows centred on ``cell_types``.

    Returns ``(labels, cluster_centers, index)`` where ``index`` is the index
    (into the caller's cell table) of the rows that were clustered, in the
    same order as ``labels``.
    """
    subset = windows[windows[cell_type_col].isin(cell_types)]
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(subset[sum_cols].values)
    return labels, km.cluster_centers_, subset.index


def fold_change(
    cluster_centers: np.ndarray, tissue_averages: np.ndarray, sum_cols: list[str]
) -> pd.DataFrame:
    """Log2 fold change of each cluster centroid over the tissue-wide average.

    ``fc = log2((centroid + avg) / sum(centroid + avg) / avg)``, matching the
    notebook's ``fc_out``/``fc_out1`` helpers.
    """
    numerator = cluster_centers + tissue_averages
    fc = np.log2(numerator / numerator.sum(axis=1, keepdims=True) / tissue_averages)
    return pd.DataFrame(fc, columns=sum_cols)


def niche_composition(cluster_centers: np.ndarray, sum_cols: list[str]) -> pd.DataFrame:
    """Percent composition of each cluster centroid (rows sum to 100)."""
    df = pd.DataFrame(cluster_centers, columns=sum_cols)
    return df.div(df.sum(axis=1), axis=0) * 100


def merge_labels(labels: np.ndarray | pd.Series, mapping: dict) -> pd.Series:
    """Map integer cluster labels to named neighbourhoods.

    ``mapping`` keys may be ``int`` or ``str``; both label dtypes are looked up.
    """
    series = pd.Series(labels)
    str_mapping = {str(k): v for k, v in mapping.items()}
    return series.astype(str).map(str_mapping)


def cluster_single_centroid(values: np.ndarray, random_state: int = 0) -> np.ndarray:
    """Fit ``MiniBatchKMeans(n_clusters=1)`` on ``values`` and return its one centroid.

    Kept as k-means (rather than a plain column mean) to match the
    notebook's ``fc_out``/``fc_out1``/``fc_out_ring`` helpers exactly; with
    one cluster the two are numerically close but not bitwise identical,
    since MiniBatchKMeans still draws mini-batches. ``random_state`` is
    fixed for reproducibility. Used both to re-cluster a merged
    neighbourhood's pooled cells into one representative centroid (cell 30's
    ``fc_out1``) and by the rings analysis (``rings.rings_profile``).
    """
    km = MiniBatchKMeans(n_clusters=1, random_state=random_state, n_init="auto")
    km.fit(values)
    return km.cluster_centers_


def merged_neighborhood_profile(
    windows: pd.DataFrame,
    merged_label_col: str,
    merged_names: list[str],
    sum_cols: list[str],
    tissue_averages: np.ndarray,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fold change and niche composition for each named (merged) neighbourhood.

    Matches cell 30-32's ``fc_out1``: rather than averaging the raw
    sub-cluster centroids that share a merged name, this re-clusters
    (``MiniBatchKMeans(n_clusters=1)``) the pooled windows of every cell
    whose merged label is that name, into one representative centroid.
    ``windows`` must be pre-filtered to the clustered cell type and carry
    ``merged_label_col``. Returns ``(fold_change_df, niche_df)`` indexed by
    ``merged_names``; a name with no cells (possible on a small or
    subsetted run, where a cluster can go unpopulated) is skipped rather
    than raising.
    """
    fc_rows, niche_rows = [], []
    for name in merged_names:
        subset = windows[windows[merged_label_col] == name]
        if subset.empty:
            continue
        centroid = cluster_single_centroid(subset[sum_cols].values, random_state)
        fc_rows.append(fold_change(centroid, tissue_averages, sum_cols).rename(index={0: name}))
        niche_rows.append(niche_composition(centroid, sum_cols).rename(index={0: name}))
    return pd.concat(fc_rows), pd.concat(niche_rows)
