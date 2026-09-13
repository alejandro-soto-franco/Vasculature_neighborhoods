import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from vasculature_neighborhoods.plotting import area_plot, build_categorical_palette, catplot


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


def test_catplot_partial_palette_dict_covers_every_hue_value():
    # A dict palette naming only some hue categories must not KeyError on
    # the rest (this crashed the real endothelial-neighbourhood highlight
    # figure, whose config only overrides two of ~12 category colours).
    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 2.0, 3.0],
            "group": ["Named", "Unnamed A", "Unnamed B", "Named"],
        }
    )
    grid = catplot(df, hue="group", palette={"Named": "red"})
    assert grid is not None
    plt.close("all")
