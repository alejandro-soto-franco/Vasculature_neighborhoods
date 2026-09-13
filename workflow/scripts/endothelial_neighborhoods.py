"""Snakemake entry point: cluster endothelial-centred k-NN windows into named
neighbourhoods, and label every endothelial cell with its neighbourhood.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vasculature_neighborhoods.io import (  # noqa: E402
    bigmem_lock,
    derive_area_column,
    load_cells,
    required_cell_columns,
)
from vasculature_neighborhoods.neighborhoods import (  # noqa: E402
    assign_names_by_content,
    cluster_windows,
    merged_neighborhood_profile,
)


def main() -> None:
    snakemake = globals()["snakemake"]
    config = snakemake.config
    cols = config["columns"]
    endo_cfg = config["endothelial_neighborhoods"]

    out = snakemake.output
    for path in (
        out.cells_labeled,
        out.tissue_averages,
        out.fold_change,
        out.niche_composition,
        out.clustermap_png,
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Other HickeyLab-fork pipelines on this machine load the same ~2.9 GB
    # table concurrently; held for this rule's whole lifetime (not just the
    # read) so at most one such load is resident system-wide at a time.
    with bigmem_lock():
        # Skips the ~47 unused raw marker intensity columns (see
        # `compute_windows.py`) and reads low-cardinality columns as
        # `category`; this dataframe becomes `cells_labeled.parquet`, so it
        # keeps every OTHER column any downstream rule reads, at whatever
        # dtype pandas infers for those (`area`/region_column is
        # deliberately left a plain string: `metadata_analysis.py`
        # `.replace()`s new values into it).
        cells = load_cells(
            snakemake.input.cells,
            cluster_col=cols["cell_type"],
            usecols=required_cell_columns(config),
            categorical_cols=[cols["donor"], cols["region"], cols["subregion"]],
        )
        # "area" (cell 4 of notebook 1) is not a raw column; it is the part
        # of unique_region after the donor prefix ("B004_Ascending" ->
        # "Ascending"), used later by metadata_analysis.py.
        cells = derive_area_column(
            cells, cols["region"], config["metadata_analysis"]["region_column"]
        )
        sum_cols = sorted(cells[cols["cell_type"]].astype(str).unique())
        tissue_averages = cells[sum_cols].to_numpy(dtype=float).mean(axis=0)

        windows_k = pd.read_parquet(snakemake.input.windows_k)
        # `compute_windows` keeps only [x, y, region] alongside the summed
        # composition; the center cell's own type is not part of a window's
        # composition, so it is joined back in here to filter windows by it.
        windows_k[cols["cell_type"]] = cells[cols["cell_type"]].reindex(windows_k.index)
        labels, centers, clustered_index = cluster_windows(
            windows_k,
            cell_type_col=cols["cell_type"],
            cell_types=[endo_cfg["cell_type"]],
            sum_cols=sum_cols,
            n_clusters=endo_cfg["n_neighborhoods"],
            random_state=endo_cfg["random_state"],
        )
        # Names clusters by centroid content (Hungarian-matched against
        # config's reference profiles), not by raw cluster index: see
        # README, "Named-cluster identity", and
        # `neighborhoods.assign_names_by_content`.
        names_by_raw_index = assign_names_by_content(
            centers, sum_cols, endo_cfg["reference_profiles"]
        )
        merged_names = pd.Series(labels).map(dict(enumerate(names_by_raw_index)))
        merged_names.index = clustered_index

        neighborhood_col = "Endothelial Cell-Centric Neighborhood"
        labeled = pd.Series("Non-Endothelial", index=cells.index, name=neighborhood_col)
        labeled.loc[merged_names.index] = merged_names.values
        cells_labeled = cells.copy()
        cells_labeled[neighborhood_col] = labeled

        # Fold change / niche rows, one per merged (named) neighbourhood.
        # Matches cell 30-32's `fc_out1`: rather than averaging the raw
        # cluster centroids that share a merged name, this re-clusters
        # (n_clusters=1) the pooled windows of every endothelial cell with
        # that merged label into one representative centroid.
        endothelial_windows = windows_k.loc[merged_names.index].copy()
        endothelial_windows[neighborhood_col] = merged_names.values
        merged_names_order = sorted(set(names_by_raw_index))
        fc_by_name, niche_by_name = merged_neighborhood_profile(
            endothelial_windows,
            neighborhood_col,
            merged_names_order,
            sum_cols,
            tissue_averages,
            random_state=endo_cfg["random_state"],
        )

        cells_labeled.to_parquet(out.cells_labeled)
        fc_by_name.to_csv(out.fold_change)
        niche_by_name.to_csv(out.niche_composition)
        pd.Series(tissue_averages, index=sum_cols, name="mean").to_csv(out.tissue_averages)

    fig = sns.clustermap(fc_by_name, vmin=-3, vmax=3, cmap="bwr", figsize=(10, 5))
    fig.ax_row_dendrogram.set_visible(False)
    fig.ax_col_dendrogram.set_visible(False)
    fig.fig.suptitle("Endothelial Cell-Centric Neighborhoods, Log Fold Change")
    fig.savefig(out.clustermap_png, dpi=300, bbox_inches="tight")
    fig.savefig(out.clustermap_pdf, dpi=300, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    main()
