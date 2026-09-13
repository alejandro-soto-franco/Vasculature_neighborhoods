import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from cellhier.plot_john import catplot2

from vasculature_neighborhoods.plotting import area_plot, build_categorical_palette


def test_build_categorical_palette_keeps_overrides_and_fills_the_rest():
    palette = build_categorical_palette(["A", "B", "C"], {"B": "hotpink"})
    assert palette["B"] == "hotpink"
    assert set(palette) == {"A", "B", "C"}
    assert palette["A"] != palette["C"]  # distinct fill colours, not one repeated


def test_area_plot_accepts_empty_color_dict():
    percentages = pd.DataFrame({"A": [10.0, 20.0], "B": [90.0, 80.0]}, index=["d1", "d2"])
    ax = area_plot(percentages, color_dict={})
    assert ax is not None
    plt.close("all")


def test_area_plot_partial_color_dict_does_not_raise():
    # Only one of two categories named: must not pass a bare `None` colour
    # entry through to matplotlib for the other.
    percentages = pd.DataFrame({"A": [10.0, 20.0], "B": [90.0, 80.0]}, index=["d1", "d2"])
    ax = area_plot(percentages, color_dict={"A": "red"})
    assert ax is not None
    plt.close("all")


def test_build_categorical_palette_feeds_catplot2_without_keyerror():
    # cellhier's catplot2 passes a dict `palette` straight into
    # `sns.lmplot`, which raises `KeyError` on any hue value the dict does
    # not name (this crashed the real endothelial-neighbourhood highlight
    # figure, whose config only overrides two of ~12 category colours).
    # `build_categorical_palette` must fill in a full palette first.
    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 2.0, 3.0],
            "exp": ["r1", "r1", "r1", "r1"],
            "group": ["Named", "Unnamed A", "Unnamed B", "Named"],
        }
    )
    palette = build_categorical_palette(sorted(df["group"].unique()), {"Named": "red"})
    (grid,) = catplot2(df, hue="group", exp="exp", X="x", Y="y", palette=palette)
    assert grid is not None
    plt.close("all")
