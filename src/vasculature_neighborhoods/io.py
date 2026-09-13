"""Loading and integrity checking for the CODEX single-cell table and donor metadata."""

from __future__ import annotations

import fcntl
import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

# Other HickeyLab-fork pipelines on this machine (cellhier, Mingl) also load
# the same ~2.9 GB CODEX table into memory; three such loads at once can
# exceed the box's RAM. Every real-data rule that loads the full table holds
# this lock for as long as that dataframe is alive, so only one such load
# happens system-wide at a time.
BIGMEM_LOCK_PATH = "/mnt/nvme/hickeylab/shared/.bigmem.lock"


@contextmanager
def bigmem_lock(lock_path: str | Path = BIGMEM_LOCK_PATH) -> Iterator[None]:
    """Hold an exclusive, cross-process lock for a big-memory section.

    Blocks until acquired. Uses the same advisory-lock mechanism as the
    shell ``flock`` command, so it serialises against any process (in any
    language) that locks the same path, not only other Python code.
    """
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def verify_sha256(path: str | Path, expected_hex: str) -> None:
    """Raise ``ValueError`` unless the file at ``path`` hashes to ``expected_hex``.

    ``expected_hex`` must be 64 hexadecimal characters (SHA-256 hex digest).
    """
    if len(expected_hex) != 64 or any(c not in "0123456789abcdef" for c in expected_hex.lower()):
        raise ValueError(f"expected_hex must be 64 hex characters, got {expected_hex!r}")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.lower() != expected_hex.lower():
        raise ValueError(f"checksum mismatch for {path}: expected {expected_hex}, got {actual}")


def load_cells(
    path: str | Path,
    cluster_col: str = "Cell Type",
    usecols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load the CODEX single-cell table and add one-hot columns for ``cluster_col``.

    Mirrors the notebook's ``pd.concat([cells, pd.get_dummies(cells[cluster_col])], axis=1)``
    step, so downstream window-sum computations can reduce over cell-type dummies directly.

    ``usecols`` restricts which columns pandas parses. The real table has
    ~47 per-marker fluorescence intensity columns (MUC2, SOX9, CD31, ...)
    that this analysis never reads (it works from ``cluster_col`` and a
    handful of per-cell metadata columns); skipping them roughly halves both
    parse time and the resulting dataframe's memory. ``cluster_col`` is
    always included even if omitted from ``usecols``.

    ``categorical_cols`` reads those columns (``cluster_col`` is always one
    of them) as pandas ``category`` dtype instead of ``object``, a further
    large memory saving for low-cardinality repeated strings (cell type,
    donor, region, ...). Do not name a column here that a caller later
    reassigns new, not-yet-seen values into (e.g. via ``.replace()``, as
    ``metadata_analysis.py`` does for its area-name remap): recent pandas
    handles that by widening the categories, but it is a sharper edge than
    a plain string column and easy to trip on a version difference.
    """
    read_cols = None
    if usecols is not None:
        read_cols = list(dict.fromkeys([*usecols, cluster_col]))
    dtype = {c: "category" for c in {*(categorical_cols or []), cluster_col}}
    # pandas-stubs' `read_csv` overloads do not resolve cleanly against a
    # plain `dict[str, str]` dtype map even though "category" is a valid
    # dtype string per pandas' own runtime behaviour and documentation.
    cells = pd.read_csv(path, usecols=read_cols, dtype=dtype)  # pyrefly: ignore
    dummies = pd.get_dummies(cells[cluster_col], dtype=int)
    return pd.concat([cells, dummies], axis=1)


def required_cell_columns(config: dict) -> list[str]:
    """Every non-marker column this pipeline reads from the cell table.

    Built from ``config["columns"]`` and ``config["metadata_analysis"]``, so
    every workflow script that only needs cluster/grouping columns (not the
    ~47 raw marker intensity columns) can pass this straight to
    ``load_cells(usecols=...)``. Does NOT include
    ``metadata_analysis.region_column`` ("area"): that column does not exist
    in the raw table and is derived by ``derive_area_column`` instead.
    """
    cols = config["columns"]
    meta_cfg = config["metadata_analysis"]
    return list(
        dict.fromkeys(
            [
                cols["x"],
                cols["y"],
                cols["region"],
                cols["donor"],
                cols["cell_type"],
                cols["subregion"],
                meta_cfg["tissue_column"],
            ]
        )
    )


def derive_area_column(cells: pd.DataFrame, region_col: str, area_col: str) -> pd.DataFrame:
    """Add the "area" column notebook 1 derives (cell 4): the part of
    ``region_col`` (e.g. ``unique_region``, "B004_Ascending") after the first
    underscore ("Ascending"). Not present as a raw column in the real table.
    """
    cells = cells.copy()
    cells[area_col] = cells[region_col].astype(str).str.split("_", n=1).str[1]
    return cells


def load_donor_metadata(path: str | Path) -> pd.DataFrame:
    """Load the donor metadata table (donors as columns, transposed to donors as rows).

    "History of ..." columns are mapped from ``{"yes", "no"}`` strings to ``bool``.
    """
    raw = pd.read_csv(path).transpose()
    raw.columns = raw.iloc[0]
    metadata = raw.iloc[1:].copy()
    for col in metadata.columns:
        if "History" in str(col):
            metadata[col] = metadata[col].map({"yes": True, "no": False, True: True, False: False})
    metadata["BMI"] = metadata["BMI"].astype(float)
    return metadata
