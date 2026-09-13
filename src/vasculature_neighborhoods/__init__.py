"""Endothelial-cell neighbourhood analysis of the HuBMAP intestine CODEX dataset.

Ports the analysis in Hickey et al. (2023), *Nature* 619:572-584, from two
exploratory notebooks into importable, tested functions used by the
Snakemake workflow in ``workflow/``.
"""

__all__ = [
    "io",
    "neighborhoods",
    "rings",
    "metadata_stats",
    "plotting",
]
