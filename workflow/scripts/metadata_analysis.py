"""Snakemake entry point: metadata-stratified analysis of endothelial neighbourhoods.

Ports ``20250123_Metadata_Endothelial_Neighborhoods_RC.ipynb``.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vasculature_neighborhoods.io import load_donor_metadata  # noqa: E402
from vasculature_neighborhoods.metadata_stats import (  # noqa: E402
    bmi_linear_fits,
    group_percentages,
    kruskal_dunn,
    long_percentages_by_replicate,
    shapiro_normality,
    two_group_ttest,
)
from vasculature_neighborhoods.plotting import area_plot, swarm_box  # noqa: E402


def main() -> None:
    snakemake = globals()["snakemake"]
    config = snakemake.config
    cols = config["columns"]
    meta_cfg = config["metadata_analysis"]
    neighborhood_col = "Endothelial Cell-Centric Neighborhood"
    colors = config["endothelial_neighborhoods"]["colors"]

    cells = pd.read_parquet(snakemake.input.cells_labeled)
    cells = cells[cells[neighborhood_col] != "Non-Endothelial"]
    metadata = load_donor_metadata(snakemake.input.donor_metadata)

    out = snakemake.output
    for path in out:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    # --- Area plot vs BMI, and per-neighbourhood linear fits ---------------
    donor_pcts = group_percentages(cells, cols["donor"], neighborhood_col)
    bmi = metadata[meta_cfg["bmi_column"]]
    ordered = donor_pcts.join(bmi).sort_values(meta_cfg["bmi_column"])
    donor_pcts_ordered = ordered[donor_pcts.columns]
    area_plot(donor_pcts_ordered, color_dict=colors, out_path=out.area_by_bmi)
    donor_pcts.to_csv(out.donor_area_percentages)

    fits = bmi_linear_fits(donor_pcts, bmi.reindex(donor_pcts.index))
    fits.to_csv(out.bmi_linear_fits)

    # --- Hypertension comparison --------------------------------------------
    hyp_col = meta_cfg["hypertension_column"]
    hypertensive = donor_pcts.loc[metadata[hyp_col].reindex(donor_pcts.index).astype(bool)]
    normotensive = donor_pcts.loc[~metadata[hyp_col].reindex(donor_pcts.index).astype(bool)]
    ttests = {
        col: dict(zip(("t", "p"), two_group_ttest(hypertensive[col], normotensive[col])))
        for col in donor_pcts.columns
    }
    with open(out.hypertension_ttests, "w") as f:
        json.dump(ttests, f, indent=2)

    long_hyp = cells[[cols["donor"], neighborhood_col]].copy()
    long_hyp["donor_index"] = long_hyp[cols["donor"]].map(
        {d: i for i, d in enumerate(long_hyp[cols["donor"]].unique())}
    )
    long_hyp[hyp_col] = long_hyp[cols["donor"]].map(metadata[hyp_col]).astype(bool)
    long_df = long_percentages_by_replicate(long_hyp, hyp_col, "donor_index", neighborhood_col)
    swarm_box(
        long_df, hyp_col, neighborhood_col, hue_order=[True, False], out_path=out.hypertension_swarm
    )

    # --- Small bowel vs colon comparison (paired by donor) ------------------
    tissue_col = meta_cfg["tissue_column"]
    sb = cells[cells[tissue_col] == "SB"]
    cl = cells[cells[tissue_col] == "CL"]
    sb_pcts = group_percentages(sb, cols["donor"], neighborhood_col)
    cl_pcts = group_percentages(cl, cols["donor"], neighborhood_col)
    common_donors = sb_pcts.index.intersection(cl_pcts.index)
    sb_cl_ttests = {
        col: dict(
            zip(
                ("t", "p"),
                two_group_ttest(
                    sb_pcts.loc[common_donors, col], cl_pcts.loc[common_donors, col], paired=True
                ),
            )
        )
        for col in sb_pcts.columns
    }
    with open(out.sb_cl_ttests, "w") as f:
        json.dump(sb_cl_ttests, f, indent=2)

    combined = pd.concat([sb, cl])[[cols["donor"], neighborhood_col, tissue_col]].copy()
    combined["donor_index"] = combined[cols["donor"]].map(
        {d: i for i, d in enumerate(combined[cols["donor"]].unique())}
    )
    long_tissue = long_percentages_by_replicate(
        combined, tissue_col, "donor_index", neighborhood_col
    )
    swarm_box(long_tissue, tissue_col, neighborhood_col, out_path=out.sb_cl_swarm)

    # --- Kruskal-Wallis / Dunn's post-hoc across (neighbourhood, tissue) ----
    long_tissue["group_id"] = (
        long_tissue[neighborhood_col].astype(str) + "_" + long_tissue[tissue_col].astype(str)
    )
    groups = {
        (name, tissue): sub["percentage"]
        for (name, tissue), sub in long_tissue.groupby([neighborhood_col, tissue_col])
    }
    normality = shapiro_normality(groups, alpha=meta_cfg["alpha"])
    normality.to_csv(out.shapiro_normality, index=False)
    h_stat, p_value, dunn = kruskal_dunn(long_tissue, "percentage", "group_id")
    with open(out.kruskal_wallis, "w") as f:
        json.dump({"H": h_stat, "p": p_value}, f, indent=2)
    dunn.to_csv(out.dunn_posthoc)

    # --- Area plot by tissue region -----------------------------------------
    region_col = meta_cfg["region_column"]
    area_map = {"Mid jejunum_extra": "Mid jejunum", "Proximal jejunum_extra": "Proximal jejunum"}
    cells_region = cells.copy()
    cells_region[region_col] = cells_region[region_col].replace(area_map)
    region_pcts = group_percentages(cells_region, region_col, neighborhood_col).sort_index()
    area_plot(region_pcts, color_dict=colors, figsize=(8, 6), out_path=out.area_by_region)
    region_pcts.to_csv(out.region_area_percentages)

    plt.close("all")


if __name__ == "__main__":
    main()
