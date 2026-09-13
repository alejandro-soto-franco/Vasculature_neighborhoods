"""Snakemake entry point: endothelial neighbourhood composition at increasing radii."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vasculature_neighborhoods.plotting import build_categorical_palette  # noqa: E402
from vasculature_neighborhoods.rings import rings_profile  # noqa: E402


def main() -> None:
    snakemake = globals()["snakemake"]
    config = snakemake.config
    ks = snakemake.params.ks
    neighborhood_col = "Endothelial Cell-Centric Neighborhood"
    plot_order = config["rings"]["plot_order"]

    cells_labeled = pd.read_parquet(snakemake.input.cells_labeled)
    tissue_averages = pd.read_csv(snakemake.input.tissue_averages, index_col=0)["mean"]
    sum_cols = list(tissue_averages.index)

    windows_by_k = {}
    for k, path in zip(ks, snakemake.input.windows_k):
        w = pd.read_parquet(path)
        w[neighborhood_col] = cells_labeled[neighborhood_col].reindex(w.index)
        windows_by_k[k] = w

    fc, niche = rings_profile(
        windows_by_k,
        ks,
        neighborhood_col,
        plot_order,
        sum_cols,
        tissue_averages.to_numpy(),
        random_state=config["windows"]["random_state"],
    )

    out = snakemake.output
    for path in (out.fold_change, out.niche_composition, out.clustermap_png):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    fc.to_csv(out.fold_change)
    niche.to_csv(out.niche_composition)

    fc_plot = fc[sum_cols]
    palette = build_categorical_palette(plot_order, config["endothelial_neighborhoods"]["colors"])
    row_colors = fc["Endothelial Neighborhood"].map(palette)
    fig = sns.clustermap(
        fc_plot,
        vmin=-3,
        vmax=3,
        cmap="bwr",
        figsize=(10, 10),
        row_cluster=False,
        col_cluster=False,
        row_colors=row_colors,
    )
    fig.ax_row_dendrogram.set_visible(False)
    fig.ax_col_dendrogram.set_visible(False)
    fig.fig.suptitle("Rings Analysis, Log Fold Change")
    n_k = len(ks)
    heatmap_ax = fig.ax_heatmap
    heatmap_ax.hlines(
        np.arange(n_k, len(fc_plot), n_k), *heatmap_ax.get_xlim(), color="black", linewidth=0.5
    )
    heatmap_ax.set_yticklabels([])
    heatmap_ax.set_ylabel("")
    heatmap_ax.set_yticks([])
    fig.savefig(out.clustermap_png, dpi=300, bbox_inches="tight")
    fig.savefig(out.clustermap_pdf, dpi=300, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    main()
