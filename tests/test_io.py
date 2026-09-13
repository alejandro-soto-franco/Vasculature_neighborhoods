import hashlib

import pandas as pd
import pytest

from vasculature_neighborhoods.io import (
    derive_area_column,
    load_cells,
    load_donor_metadata,
    required_cell_columns,
    verify_sha256,
)


def test_derive_area_column_splits_on_first_underscore_only():
    cells = pd.DataFrame({"unique_region": ["B004_Ascending", "B008_Proximal jejunum_extra"]})
    out = derive_area_column(cells, "unique_region", "area")
    assert list(out["area"]) == ["Ascending", "Proximal jejunum_extra"]


def test_verify_sha256_passes_for_matching_digest(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    verify_sha256(path, expected)  # must not raise


def test_verify_sha256_raises_for_mismatched_digest(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"hello world")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_sha256(path, "0" * 64)


def test_verify_sha256_rejects_malformed_expected_digest(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"data")
    with pytest.raises(ValueError, match="64 hex"):
        verify_sha256(path, "not-a-hash")


def test_load_cells_adds_one_hot_columns(tmp_path):
    path = tmp_path / "cells.csv"
    pd.DataFrame({"x": [1, 2], "y": [1, 2], "Cell Type": ["A", "B"]}).to_csv(path, index=False)
    cells = load_cells(path)
    assert "A" in cells.columns and "B" in cells.columns
    assert cells.loc[0, "A"] == 1
    assert cells.loc[0, "B"] == 0


def test_load_cells_usecols_drops_unlisted_columns_but_keeps_cluster_col(tmp_path):
    path = tmp_path / "cells.csv"
    pd.DataFrame({"x": [1, 2], "y": [1, 2], "Cell Type": ["A", "B"], "MUC2": [0.1, 0.2]}).to_csv(
        path, index=False
    )
    cells = load_cells(path, usecols=["x", "y"])
    assert "MUC2" not in cells.columns
    assert "Cell Type" in cells.columns  # always kept even if omitted from usecols
    assert {"A", "B"}.issubset(cells.columns)


def test_required_cell_columns_covers_every_config_column(tmp_path):
    config = {
        "columns": {
            "x": "x",
            "y": "y",
            "region": "unique_region",
            "donor": "donor",
            "cell_type": "Cell Type",
            "subregion": "Tissue Unit",
        },
        "metadata_analysis": {"tissue_column": "tissue", "region_column": "area"},
    }
    cols = required_cell_columns(config)
    # "area" (region_column) is deliberately excluded: it is not a raw
    # column, it is derived from unique_region by `derive_area_column`.
    assert set(cols) == {
        "x",
        "y",
        "unique_region",
        "donor",
        "Cell Type",
        "Tissue Unit",
        "tissue",
    }
    assert "area" not in cols


def test_load_donor_metadata_transposes_and_maps_bools(tmp_path):
    path = tmp_path / "meta.csv"
    pd.DataFrame(
        {
            "Stanford ID": ["donor age", "BMI", "History of hypertension"],
            "B001": [67, 30.2, "yes"],
            "B004": [78, 35.1, "no"],
        }
    ).to_csv(path, index=False)
    metadata = load_donor_metadata(path)
    assert list(metadata.index) == ["B001", "B004"]
    assert bool(metadata.loc["B001", "History of hypertension"]) is True
    assert bool(metadata.loc["B004", "History of hypertension"]) is False
    assert metadata.loc["B001", "BMI"] == pytest.approx(30.2)
