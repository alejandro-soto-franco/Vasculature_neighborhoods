"""Snakemake entry point: single-region figure of endothelial cell-centric neighbourhoods.

Ports the "B010_Ileum" highlight plot (cells 22 and 27 of
``20250820_SingleCell_NeighborhoodAnalysis_Endothelial_HuBMAP.ipynb``): one
tissue section, cells coloured by their endothelial cell-centric
neighbourhood. The exploratory UMAP/Leiden cells in that notebook (48-51) are
dropped; see README, "Differences from upstream".
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vasculature_neighborhoods.plotting import catplot  # noqa: E402


def main() -> None:
    snakemake = globals()["snakemake"]
    config = snakemake.config
    cols = config["columns"]
    region = config["figures"]["region_highlight"]
    colors = config["endothelial_neighborhoods"]["colors"]
    neighborhood_col = "Endothelial Cell-Centric Neighborhood"

    cells = pd.read_parquet(snakemake.input.cells_labeled)
    sub = cells[cells[cols["region"]] == region]

    out_path = Path(snakemake.output.figure)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if sub.empty:
        raise ValueError(
            f"figures.region_highlight {region!r} is not present in this run's cells; "
            "fix config/config.yaml (or config/smoke.yaml) to name a region that exists"
        )

    palette = colors if colors else "bright"
    grid = catplot(
        sub,
        hue=neighborhood_col,
        x=cols["x"],
        y=cols["y"],
        invert_y=True,
        size=10,
        palette=palette,
        out_path=out_path,
    )
    plt.close("all")
    del grid


if __name__ == "__main__":
    main()
