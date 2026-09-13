"""Figure helpers.

``area_plot`` and ``swarm_box`` are cleaned ports of the plotting functions
both upstream notebooks redefine locally in their own cells (not part of the
``cellhier`` package). The tissue-section scatter plot (the notebooks'
``catplot2``/``catplot21``) is not ported here: ``cellhier`` now ships it as
``cellhier.plot_john.catplot2`` (a pinned git dependency, see
``pyproject.toml``); callers import it directly. ``build_categorical_palette``
is this project's own helper, used to fill in a colour for every category
before calling ``catplot2`` with a partial (deliberately only
part-overridden) palette dict, which it otherwise raises a ``KeyError`` on.

Differences from the notebooks' originals: no module-level globals
(``save_path``, ``pal_temp``), figures are returned/saved via an explicit
path rather than a hardcoded prefix, and dead commented-out code is removed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def build_categorical_palette(
    categories: list[str], overrides: dict[str, str | tuple] | None = None
) -> dict[str, str | tuple]:
    """A colour for every category: ``overrides`` first, "bright" cycle for the rest.

    Matches the notebooks' repeated pattern of
    ``sns.color_palette('bright', len(categories))`` zipped against every
    category, then a couple of manual colour picks layered on top (e.g.
    "Follicle" -> mediumspringgreen). Every category gets an entry, so a
    downstream ``palette=``/``row_colors=`` lookup never falls through to a
    missing key or a silent ``NaN``.
    """
    overrides = overrides or {}
    bright = sns.color_palette("bright", len(categories))
    return {c: overrides.get(c, bright[i]) for i, c in enumerate(categories)}


def area_plot(
    percentages: pd.DataFrame,
    color_dict: dict[str, str | tuple] | None = None,
    figsize: tuple[float, float] = (8, 4),
    ylabel: str = "percent",
    out_path: str | Path | None = None,
) -> plt.Axes:
    """Stacked area plot of category percentages across an ordered grouping axis.

    ``percentages`` is indexed by the grouping variable (already sorted, e.g.
    by BMI or by tissue area) with one column per category. Matches
    ``area_plot`` in the notebooks.
    """
    order = percentages.mean().sort_values(ascending=False).index
    plot_df = percentages[order]
    # An empty or partial color_dict falls back to matplotlib's default
    # cycle for any column it does not name, rather than passing `None`
    # entries through (which pandas' plotting backend rejects outright).
    colors = None
    if color_dict:
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = [
            color_dict[c] if c in color_dict else cycle[i % len(cycle)]
            for i, c in enumerate(plot_df.columns)
        ]
    ax = plot_df.plot.area(alpha=0.8, linewidth=1, color=colors, figsize=figsize, rot=90)
    for line in ax.lines:
        line.set_color("black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(ylabel)
    plt.xticks(range(len(plot_df.index)), list(plot_df.index), rotation=90)
    plt.tight_layout()
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=False)
    if out_path:
        plt.savefig(out_path, dpi=300, transparent=True, bbox_inches="tight")
    return ax


def swarm_box(
    long_df: pd.DataFrame,
    grouping: str,
    category_col: str,
    value_col: str = "percentage",
    order: list[str] | None = None,
    hue_order: list | None = None,
    figsize: tuple[float, float] = (10, 5),
    out_path: str | Path | None = None,
) -> plt.Axes:
    """Boxplot with an overlaid swarm, one box pair per category.

    ``long_df`` is the long-format table from
    ``metadata_stats.long_percentages_by_replicate``. ``order`` is an
    explicit, label-based category order (mean-ascending by default);
    matches ``swarm_box`` in the notebooks.
    """
    if order is None:
        order = long_df.groupby(category_col)[value_col].mean().sort_values().index.to_list()
    plt.figure(figsize=figsize)
    ax = sns.boxplot(
        data=long_df,
        x=category_col,
        y=value_col,
        hue=grouping,
        dodge=True,
        order=order,
        hue_order=hue_order,
    )
    # seaborn >=0.12 draws box faces as `Patch`es on `ax.patches`, not
    # `ax.artists` (the notebook's original target, from an older seaborn);
    # captured before the swarmplot call so only the box faces are touched.
    box_patches = list(ax.patches)
    sns.swarmplot(
        data=long_df,
        x=category_col,
        y=value_col,
        hue=grouping,
        dodge=True,
        order=order,
        hue_order=hue_order,
        edgecolor="black",
        linewidth=1,
        ax=ax,
    )
    for patch in box_patches:
        # matplotlib's Patch.get_facecolor() is typed as a broad colour-spec
        # union for the setter's sake; at runtime it always returns the
        # resolved RGBA float tuple this unpacks.
        r, g, b, _ = patch.get_facecolor()  # pyrefly: ignore
        patch.set_facecolor((r, g, b, 0.3))  # pyrefly: ignore
    plt.xlabel("")
    handles, labels = ax.get_legend_handles_labels()
    n_groups = long_df[grouping].nunique()
    plt.legend(
        handles[:n_groups],
        labels[:n_groups],
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
        frameon=False,
    )
    plt.xticks(rotation=90)
    sns.despine(trim=True)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, transparent=True, bbox_inches="tight")
    return ax
