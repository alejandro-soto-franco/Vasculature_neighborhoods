"""Snakemake entry point: donor-stratified rings analysis and subregion normalisation."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vasculature_neighborhoods.rings import (  # noqa: E402
    donor_ratio_tables,
    donor_rings_profiles,
    normalize_by_subregion,
    subregion_percentages,
)


def main() -> None:
    snakemake = globals()["snakemake"]
    config = snakemake.config
    cols = config["columns"]
    ks = snakemake.params.ks
    rings_cfg = config["rings"]
    neighborhood_col = "Endothelial Cell-Centric Neighborhood"
    plot_order = rings_cfg["plot_order"]

    cells_labeled = pd.read_parquet(snakemake.input.cells_labeled)
    tissue_averages = pd.read_csv(snakemake.input.tissue_averages, index_col=0)["mean"]
    sum_cols = list(tissue_averages.index)

    windows_by_k = {}
    for k, path in zip(ks, snakemake.input.windows_k):
        w = pd.read_parquet(path)
        w[neighborhood_col] = cells_labeled[neighborhood_col].reindex(w.index)
        windows_by_k[k] = w

    donor_by_index = cells_labeled[cols["donor"]]
    profiles = donor_rings_profiles(
        windows_by_k,
        donor_by_index,
        ks,
        neighborhood_col,
        plot_order,
        sum_cols,
        tissue_averages.to_numpy(),
        random_state=config["windows"]["random_state"],
    )
    niche_by_donor = {donor: niche for donor, (_, niche) in profiles.items()}

    combined_ratio_df, combined_at_k_large_df = donor_ratio_tables(
        niche_by_donor,
        rings_cfg["k_small"],
        rings_cfg["k_large"],
        cell_type_col=config["endothelial_neighborhoods"]["cell_type"],
    )

    subregion_pcts = subregion_percentages(
        cells_labeled,
        cols["cell_type"],
        config["endothelial_neighborhoods"]["cell_type"],
        cols["subregion"],
    )
    normalized = normalize_by_subregion(
        combined_at_k_large_df,
        rings_cfg["neighborhood_to_subregion"],
        subregion_pcts,
        cell_type_col=config["endothelial_neighborhoods"]["cell_type"],
    )

    out = snakemake.output
    for path in (out.ratio_table, out.normalized_barplot_png, out.ratio_boxplot_png):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    combined_ratio_df.to_csv(out.ratio_table, index=False)
    combined_at_k_large_df.to_csv(out.at_k_large_table, index=False)
    subregion_pcts.to_csv(out.subregion_percentages, index=False)
    normalized.to_csv(out.normalized_table, index=False)

    cell_type = config["endothelial_neighborhoods"]["cell_type"]
    norm_col = f"Normalized {cell_type} %"

    plt.figure(figsize=(6, 8))
    ax = sns.barplot(
        data=normalized,
        x="Endothelial Neighborhood",
        y=norm_col,
        order=plot_order,
        edgecolor="black",
    )
    sns.stripplot(
        data=normalized,
        x="Endothelial Neighborhood",
        y=norm_col,
        order=plot_order,
        color="black",
        alpha=0.5,
        ax=ax,
    )
    plt.axhline(1.0, color="black", linestyle="--", label="No change (ratio = 1)")
    plt.title(f"Normalised {cell_type} Cell Percent by Tissue Subregion")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out.normalized_barplot_png, dpi=300, bbox_inches="tight")
    plt.savefig(out.normalized_barplot_pdf, dpi=300, bbox_inches="tight")
    plt.close("all")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined_ratio_df, x="Endothelial Neighborhood", y=cell_type, order=plot_order)
    sns.stripplot(
        data=combined_ratio_df,
        x="Endothelial Neighborhood",
        y=cell_type,
        order=plot_order,
        color="black",
        alpha=0.6,
    )
    k_small, k_large = config["rings"]["k_small"], config["rings"]["k_large"]
    plt.axhline(1.0, color="black", linestyle="--", label="No change (ratio = 1)")
    plt.ylabel(f"%{cell_type} at {k_large} / %{cell_type} at {k_small}")
    plt.title(f"Change in {cell_type} Representation from {k_small} to {k_large} Neighborhood Size")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out.ratio_boxplot_png, dpi=300, bbox_inches="tight")
    plt.savefig(out.ratio_boxplot_pdf, dpi=300, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    main()
