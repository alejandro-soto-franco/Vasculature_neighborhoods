import pandas as pd

from vasculature_neighborhoods.windows import compute_windows, iter_windows


def _synthetic_cells():
    # Two regions, 4 cells each, laid out on a line so nearest neighbours are
    # unambiguous. Cell types A/B one-hot encoded as sum_cols.
    rows = []
    for region in ("r1", "r2"):
        for i in range(4):
            rows.append(
                {
                    "x": float(i),
                    "y": 0.0,
                    "unique_region": region,
                    "type": "A" if i % 2 == 0 else "B",
                }
            )
    cells = pd.DataFrame(rows)
    dummies = pd.get_dummies(cells["type"], dtype=int)
    return pd.concat([cells, dummies], axis=1)


def test_compute_windows_sums_k_nearest_neighbors_within_region():
    cells = _synthetic_cells()
    windows = compute_windows(cells, ks=[1, 2], sum_cols=["A", "B"])
    assert set(windows) == {1, 2}

    w1 = windows[1]
    # k=1: each cell's own composition (nearest neighbour of itself is itself).
    assert (w1["A"] + w1["B"]).eq(1).all()

    w2 = windows[2]
    # k=2: sums to 2 cell-type counts per window everywhere.
    assert (w2["A"] + w2["B"]).eq(2).all()


def test_compute_windows_never_crosses_regions():
    cells = _synthetic_cells()
    # Make region r2 sit far away in space but adjacent in row order, so a
    # bug that pools all cells together (ignoring region) would pull in
    # neighbours from the wrong region.
    cells.loc[cells["unique_region"] == "r2", "x"] += 1000
    windows = compute_windows(cells, ks=[4], sum_cols=["A", "B"])
    w4 = windows[4]
    # Every region has exactly 2 A and 2 B cells, so a same-region k=4 window
    # (the whole region) always sums to (2, 2).
    assert (w4["A"] == 2).all()
    assert (w4["B"] == 2).all()


def test_compute_windows_caps_k_at_region_size():
    cells = _synthetic_cells()
    windows = compute_windows(cells, ks=[100], sum_cols=["A", "B"])
    w = windows[100]
    assert (w["A"] + w["B"]).eq(4).all()  # each region only has 4 cells total


def test_compute_windows_rejects_empty_ks():
    cells = _synthetic_cells()
    try:
        compute_windows(cells, ks=[], sum_cols=["A", "B"])
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty ks")


def test_iter_windows_yields_ks_in_order_and_matches_compute_windows():
    cells = _synthetic_cells()
    streamed = list(iter_windows(cells, ks=[1, 2], sum_cols=["A", "B"]))
    assert [k for k, _ in streamed] == [1, 2]
    batched = compute_windows(cells, ks=[1, 2], sum_cols=["A", "B"])
    for k, df in streamed:
        pd.testing.assert_frame_equal(df, batched[k])
