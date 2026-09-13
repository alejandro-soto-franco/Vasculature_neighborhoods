import numpy as np
import pandas as pd
import pytest

from vasculature_neighborhoods.neighborhoods import (
    cluster_single_centroid,
    cluster_windows,
    fold_change,
    merge_labels,
    merged_neighborhood_profile,
    niche_composition,
)


def test_cluster_windows_only_uses_requested_cell_types():
    windows = pd.DataFrame(
        {
            "cell_type": ["Endothelial", "Endothelial", "Other", "Other"],
            "A": [10, 12, 1, 1],
            "B": [0, 0, 9, 10],
        }
    )
    labels, centers, index = cluster_windows(
        windows,
        cell_type_col="cell_type",
        cell_types=["Endothelial"],
        sum_cols=["A", "B"],
        n_clusters=1,
        random_state=0,
    )
    assert len(labels) == 2
    assert list(index) == [0, 1]
    assert centers.shape == (1, 2)


def test_fold_change_hand_computed_example():
    tissue_avg = np.array([2.0, 2.0])
    centers = np.array([[6.0, 2.0]])  # enriched in first column
    fc = fold_change(centers, tissue_avg, ["A", "B"])
    numerator = centers + tissue_avg
    expected = np.log2(numerator / numerator.sum(axis=1, keepdims=True) / tissue_avg)
    assert fc.values == pytest.approx(expected)


def test_niche_composition_rows_sum_to_100():
    centers = np.array([[1.0, 3.0], [2.0, 2.0]])
    niche = niche_composition(centers, ["A", "B"])
    assert niche.sum(axis=1).values == pytest.approx([100.0, 100.0])


def test_merge_labels_maps_int_and_str_keys():
    labels = np.array([0, 1, 0])
    mapping = {0: "Neighbourhood A", "1": "Neighbourhood B"}
    merged = merge_labels(labels, mapping)
    assert list(merged) == ["Neighbourhood A", "Neighbourhood B", "Neighbourhood A"]


def test_cluster_single_centroid_shape():
    values = np.array([[1.0, 2.0], [1.2, 1.8], [0.9, 2.1]])
    centroid = cluster_single_centroid(values, random_state=0)
    assert centroid.shape == (1, 2)


def test_merged_neighborhood_profile_reclusters_pooled_cells():
    # Two raw sub-clusters both merged into "Group A"; the merged profile
    # must come from re-clustering their POOLED cells (matching cell 30's
    # `fc_out1`), not from averaging the two sub-cluster centroids.
    windows = pd.DataFrame(
        {
            "merged": ["Group A", "Group A", "Group A", "Group B"],
            "A": [10.0, 10.0, 10.0, 1.0],
            "B": [0.0, 0.0, 0.0, 9.0],
        }
    )
    tissue_avg = np.array([2.0, 2.0])
    fc, niche = merged_neighborhood_profile(
        windows, "merged", ["Group A", "Group B"], ["A", "B"], tissue_avg, random_state=0
    )
    assert list(fc.index) == ["Group A", "Group B"]
    assert list(niche.index) == ["Group A", "Group B"]
    # Group A pools three identical rows, so its centroid is exactly (10, 0).
    assert niche.loc["Group A", "A"] == pytest.approx(100.0)
    assert niche.loc["Group A", "B"] == pytest.approx(0.0)


def test_merged_neighborhood_profile_skips_unpopulated_names():
    windows = pd.DataFrame({"merged": ["Group A", "Group A"], "A": [10.0, 10.0], "B": [0.0, 0.0]})
    fc, niche = merged_neighborhood_profile(
        windows,
        "merged",
        ["Group A", "Group B (empty)"],
        ["A", "B"],
        np.array([2.0, 2.0]),
        random_state=0,
    )
    assert list(fc.index) == ["Group A"]
    assert list(niche.index) == ["Group A"]
