"""K-nearest-neighbour window construction, per tissue region.

Ported from the ``get_windows`` function copied into both upstream notebooks
(itself a copy of the windowing step in the Hickey Lab ``cellhier`` fork,
https://github.com/alejandro-soto-franco/Hierarchical-Tissue-Unit-Annotation,
commit b47c15b1409dbe8a732485b62331348ee6e6d709 at time of porting -
``knn_graph_neighborhood2.Neighborhoods``). ``cellhier`` has no installable
package yet (no ``pyproject.toml``/``setup.py``), so this module is a cleaned,
vendored copy rather than an import; swap it for ``cellhier`` once that fork
is packaged. Attribution: Hickey Lab, Hierarchical-Tissue-Unit-Annotation.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _region_neighbor_indices(
    region_cells: pd.DataFrame, x_col: str, y_col: str, n_neighbors: int
) -> np.ndarray:
    """Nearest-neighbour indices (into ``region_cells``) for every cell in the region.

    Neighbours are sorted by distance, nearest first, matching the notebook's
    ``argsort`` re-sort of ``NearestNeighbors.kneighbors`` output. Returned as
    ``int32`` (a tissue region has nowhere near 2**31 cells): with `ks` up to
    300 and every region held in memory at once, this halves what is easily
    the largest allocation `compute_windows`/`iter_windows` makes.
    """
    coords = region_cells[[x_col, y_col]].values
    fit = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
    dist, ind = fit.kneighbors(coords)
    order = dist.argsort(axis=1)
    row_offset = np.arange(ind.shape[0]) * ind.shape[1]
    sorted_ind = ind.flatten()[order + row_offset[:, None]]
    return region_cells.index.values[sorted_ind].astype(np.int32, copy=False)


def iter_windows(
    cells: pd.DataFrame,
    ks: list[int],
    sum_cols: list[str],
    x_col: str = "x",
    y_col: str = "y",
    region_col: str = "unique_region",
) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield ``(k, window_dataframe)`` for every ``k`` in ``ks``, one at a time.

    Same computation as `compute_windows`, but only one `k`'s composition
    dataframe is alive at once (the neighbour-index search, the expensive
    part, still runs once up front and is reused across every `k`). Prefer
    this over `compute_windows` on the real dataset: the caller can write
    and drop each `k`'s result before the next is built, rather than holding
    every `k` at once.
    """
    if not ks:
        raise ValueError("ks must be non-empty")
    max_k = max(ks)
    keep_cols = [x_col, y_col, region_col]

    values = cells[sum_cols].values
    # Map cell index label -> position in `values`/`cells`, built once: with
    # the RangeIndex `load_cells` produces this is the identity, but the
    # lookup stays correct for any caller-supplied (e.g. pre-filtered) index.
    pos_of_index = pd.Series(np.arange(len(cells)), index=cells.index)

    # `observed=True`: with `region_col` read as a `category` dtype (see
    # `io.load_cells`), the default groups over every category, not just
    # ones actually present. Every category is expected to be present here,
    # but stating the requirement avoids a pandas FutureWarning and a
    # silent behaviour change on a future pandas upgrade.
    region_groups = list(cells.groupby(region_col, observed=True))
    neighbor_indices: dict[object, np.ndarray] = {}
    for region_name, region_cells in region_groups:
        n_neighbors = min(max_k, len(region_cells))
        neighbor_indices[region_name] = _region_neighbor_indices(
            region_cells, x_col, y_col, n_neighbors
        )

    for k in ks:
        pieces = []
        for region_name, region_cells in region_groups:
            neighbors = neighbor_indices[region_name]
            k_eff = min(k, neighbors.shape[1])
            neighbor_positions = pos_of_index.loc[neighbors[:, :k_eff].flatten()].to_numpy(
                dtype=np.intp
            )
            summed = (
                values[neighbor_positions]
                .reshape(len(region_cells), k_eff, len(sum_cols))
                .sum(axis=1)
            )
            pieces.append(pd.DataFrame(summed, index=region_cells.index, columns=sum_cols))
        window = pd.concat(pieces, axis=0).loc[cells.index]
        yield k, pd.concat([cells[keep_cols], window], axis=1)


def compute_windows(
    cells: pd.DataFrame,
    ks: list[int],
    sum_cols: list[str],
    x_col: str = "x",
    y_col: str = "y",
    region_col: str = "unique_region",
) -> dict[int, pd.DataFrame]:
    """Compute k-nearest-neighbour composition windows for every ``k`` in ``ks``.

    For each cell, sums the one-hot cell-type columns (``sum_cols``) of its
    ``k`` nearest neighbours (by Euclidean distance in ``x_col``/``y_col``,
    computed separately per ``region_col`` so windows never cross tissue
    sections). Returns one dataframe per ``k``, indexed like ``cells``, with
    ``sum_cols`` replaced by per-window sums and the original non-numeric
    columns in ``sum_cols``'s complement dropped except for
    ``[x_col, y_col, region_col]``.

    Convenience wrapper around `iter_windows` that holds every `k` in memory
    at once; fine for tests and small inputs, but the workflow's
    `compute_windows` script uses `iter_windows` directly on the real data.
    """
    return dict(iter_windows(cells, ks, sum_cols, x_col, y_col, region_col))
