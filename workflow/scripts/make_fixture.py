"""Generate a synthetic fixture shaped like the CODEX single-cell table.

Used only by the smoke config so the whole DAG can run end to end, in
minutes, without the real ~2.9 GB dataset. Column names and structure mirror
``23_09_CODEX_HuBMAP_alldata_Dryad_merged.csv``: one row per cell, an
``x``/``y`` position, a ``unique_region`` (donor + area), a ``Cell Type``, and
a ``Tissue Unit`` subregion.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def make_fixture(
    n_donors: int,
    n_areas_per_donor: int,
    n_cells_per_region: int,
    n_cell_types: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    cell_types = [f"type_{i}" for i in range(n_cell_types)]
    subregions = ["Region 1", "Region 2"]
    rows = []
    for d in range(n_donors):
        donor = f"D{d}"
        for a in range(n_areas_per_donor):
            area = f"A{a}"
            region = f"{donor}_{area}"
            xs = rng.uniform(0, 500, n_cells_per_region)
            ys = rng.uniform(0, 500, n_cells_per_region)
            # type_1 plays the role of "Endothelial": rarer than the rest.
            weights = np.array(
                [0.3 if t == "type_1" else 0.7 / (n_cell_types - 1) for t in cell_types]
            )
            weights /= weights.sum()
            types = rng.choice(cell_types, size=n_cells_per_region, p=weights)
            subregion = rng.choice(subregions, size=n_cells_per_region)
            for i in range(n_cells_per_region):
                rows.append(
                    {
                        "x": xs[i],
                        "y": ys[i],
                        "unique_region": region,
                        "donor": donor,
                        "area": area,
                        "tissue": "SB" if a % 2 == 0 else "CL",
                        "Cell Type": types[i],
                        "Tissue Unit": subregion[i],
                    }
                )
    cells = pd.DataFrame(rows)

    donors = [f"D{d}" for d in range(n_donors)]
    metadata_rows = {
        "donor age": rng.integers(20, 80, n_donors),
        "donor sex": rng.choice(["male", "female"], n_donors),
        "BMI": np.round(rng.uniform(18, 35, n_donors), 1),
        "History of diabetes": rng.choice(["yes", "no"], n_donors),
        "History of cancer": rng.choice(["yes", "no"], n_donors),
        "History of hypertension": rng.choice(["yes", "no"], n_donors),
        "History gastrointestinal disease": rng.choice(["yes", "no"], n_donors),
    }
    metadata = pd.DataFrame(metadata_rows, index=donors).transpose()
    metadata.index.name = "Stanford ID"
    return cells, metadata


def main() -> None:
    snakemake = globals()["snakemake"]
    fixture_cfg = snakemake.config["data"]["fixture"]
    cells, metadata = make_fixture(
        n_donors=fixture_cfg["n_donors"],
        n_areas_per_donor=fixture_cfg["n_areas_per_donor"],
        n_cells_per_region=fixture_cfg["n_cells_per_region"],
        n_cell_types=fixture_cfg["n_cell_types"],
        random_state=fixture_cfg["random_state"],
    )
    Path(snakemake.output.cells).parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(snakemake.output.cells, index=False)
    metadata.reset_index().to_csv(snakemake.output.donor_metadata, index=False)


if __name__ == "__main__":
    main()
