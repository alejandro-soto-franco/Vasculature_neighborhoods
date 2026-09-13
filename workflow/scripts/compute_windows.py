"""Snakemake entry point: compute k-NN composition windows for every configured k."""

import os
import sys
from pathlib import Path

# Must be set before numpy/sklearn's BLAS backend initialises, so the thread
# cap in config.yaml (compute.omp_num_threads) actually takes effect: a
# Snakemake `resources:` declaration only informs the scheduler, it does not
# export an environment variable into a `script:` rule's own interpreter.
snakemake = globals()["snakemake"]
os.environ["OMP_NUM_THREADS"] = str(snakemake.config["compute"]["omp_num_threads"])

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vasculature_neighborhoods.io import bigmem_lock, load_cells  # noqa: E402
from vasculature_neighborhoods.windows import iter_windows  # noqa: E402


def main() -> None:
    config = snakemake.config
    cols = config["columns"]
    ks = snakemake.params.ks
    out_path_by_k = dict(zip(ks, snakemake.output.windows_k))
    for path in out_path_by_k.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Other HickeyLab-fork pipelines on this machine load the same ~2.9 GB
    # table concurrently; held for this rule's whole lifetime (not just the
    # read) so at most one such load is resident system-wide at a time.
    with bigmem_lock():
        # Only the columns windowing itself touches: skips the ~47 unused raw
        # marker intensity columns, and reads low-cardinality columns as
        # `category` — together, well under this machine's per-agent budget
        # for what is this pipeline's single heaviest step.
        usecols = [cols["x"], cols["y"], cols["region"]]
        cells = load_cells(
            snakemake.input.cells,
            cluster_col=cols["cell_type"],
            usecols=usecols,
            categorical_cols=[cols["region"]],
        )
        cell_type_values = sorted(cells[cols["cell_type"]].astype(str).unique())

        # Streamed one `k` at a time (rather than `compute_windows`, which
        # holds every `k` in memory together): each `k`'s composition
        # dataframe is written and freed before the next is built.
        for k, window in iter_windows(
            cells,
            ks=ks,
            sum_cols=cell_type_values,
            x_col=cols["x"],
            y_col=cols["y"],
            region_col=cols["region"],
        ):
            window.to_parquet(out_path_by_k[k])


if __name__ == "__main__":
    main()
