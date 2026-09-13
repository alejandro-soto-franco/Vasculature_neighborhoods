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

from cellhier.knn_graph_neighborhood2 import Neighborhoods  # noqa: E402

from vasculature_neighborhoods.io import bigmem_lock, load_cells  # noqa: E402


def main() -> None:
    config = snakemake.config
    cols = config["columns"]
    ks = snakemake.params.ks
    out_path_by_k = dict(zip(ks, snakemake.output.windows_k))
    for path in out_path_by_k.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Opt-in (config.compute.bigmem_lock_path, or BIGMEM_LOCK_PATH) lock
    # around this rule's whole lifetime, not just the read: a deployment
    # running several pipelines against a shared copy of this table can
    # point every one of them at the same lock file to avoid a concurrent
    # peak; a bare run (this repo's tests, CI) takes no lock at all.
    with bigmem_lock(config["compute"].get("bigmem_lock_path")):
        # Only the columns windowing itself touches: skips the ~47 unused raw
        # marker intensity columns, a large memory reduction for what is this
        # pipeline's single heaviest step.
        usecols = [cols["x"], cols["y"], cols["region"], cols["cell_type"]]
        cells = load_cells(snakemake.input.cells, cluster_col=cols["cell_type"], usecols=usecols)
        cell_type_values = sorted(cells[cols["cell_type"]].astype(str).unique())

        # cellhier's `Neighborhoods` does its own one-hot encoding of
        # `cluster_col` internally (`add_dummies=True`); `sum_cols` names the
        # resulting dummy columns (the sorted unique cell types) it should
        # sum over each window, `keep_cols` the columns to carry through
        # unchanged. Note: unlike the notebooks' own `get_windows` (which
        # queried each tissue's coordinates against themselves, so a cell's
        # own composition was always included as its "nearest" neighbour at
        # distance 0), cellhier's current `make_windows` explicitly excludes
        # the centre cell from its own window; see README, "Differences from
        # upstream".
        #
        # One `Neighborhoods` call per `k`, run separately, rather than one
        # call for every `k` together: `Neighborhoods` sizes its per-region
        # neighbour cache to `max(ks)` and holds it for every `k` in the
        # call, so requesting all of them at once peaks at `k=300`'s memory
        # for the whole rule. This machine runs several concurrent
        # pipelines against the same table (hence `bigmem_lock` above); one
        # such run was killed by the OOM killer holding everything at once
        # (see README, "Compute notes"). Each `k` here costs its own
        # nearest-neighbour search (more total compute than the combined
        # call), trading wall time for a bounded, much smaller peak.
        for k in ks:
            neighborhoods = Neighborhoods(
                cells,
                ks=[k],
                cluster_col=cols["cell_type"],
                sum_cols=list(cell_type_values),
                keep_cols=[cols["x"], cols["y"], cols["region"]],
                X=cols["x"],
                Y=cols["y"],
                reg=cols["region"],
                add_dummies=True,
            )
            windows_by_k = neighborhoods.k_windows()
            windows_by_k[k].to_parquet(out_path_by_k[k])
            del neighborhoods, windows_by_k


if __name__ == "__main__":
    main()
