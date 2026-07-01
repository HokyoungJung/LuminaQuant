"""Unit tests for the additive ts_mean / ts_mean_series formulaic operators.

Covers hand-computed fixtures, the ``ts_mean == ts_sum / window`` identity,
NaN / short-window edge behaviour, and scalar-vs-series index alignment. The
operators mirror the ts_sum conventions (trailing window, ``max(1, int(window))``
normalisation, ``float | None`` scalar return, NaN-padded leading Series entries).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from lumina_quant.indicators import formulaic_operators as fops
from lumina_quant.indicators import ts_mean as ts_mean_public

VALUES = [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.mark.parametrize(
    "window, expected",
    [
        (1, 5.0),  # trailing 1 -> last value
        (2, 4.5),  # mean of [4, 5]
        (3, 4.0),  # mean of [3, 4, 5]
        (5, 3.0),  # mean of the full window
    ],
)
def test_ts_mean_matches_hand_computed(window, expected):
    assert fops.ts_mean(VALUES, window) == pytest.approx(expected)


def test_public_export_matches_module_symbol():
    # The package re-export must be the same callable defined in the module.
    assert ts_mean_public is fops.ts_mean
    assert ts_mean_public(VALUES, 3) == pytest.approx(4.0)


@pytest.mark.parametrize("window", [1, 2, 3, 4, 5])
def test_ts_mean_equals_ts_sum_over_window(window):
    total = fops.ts_sum(VALUES, window)
    assert total is not None
    assert fops.ts_mean(VALUES, window) == pytest.approx(total / window)


def test_ts_mean_short_window_returns_none():
    # Fewer samples than the window -> None, exactly mirroring ts_sum.
    assert fops.ts_sum([1.0, 2.0], 3) is None
    assert fops.ts_mean([1.0, 2.0], 3) is None


def test_ts_mean_nan_inside_window_propagates():
    values = [1.0, float("nan"), 3.0]
    # The tail is long enough, so ts_sum keeps the NaN and the mean is NaN too.
    total = fops.ts_sum(values, 3)
    assert total is not None and math.isnan(total)
    result = fops.ts_mean(values, 3)
    assert result is not None and math.isnan(result)


def test_ts_mean_non_positive_window_clamps_to_one():
    # window normalises via max(1, int(window)); trailing-1 mean is the last value.
    assert fops.ts_mean(VALUES, 0) == pytest.approx(5.0)
    assert fops.ts_mean(VALUES, -4) == pytest.approx(5.0)


def test_ts_mean_series_tail_matches_scalar():
    index = pd.RangeIndex(len(VALUES))
    series = fops.ts_mean_series(VALUES, 3, index=index)
    assert isinstance(series, pd.Series)
    assert list(series.index) == list(index)
    # Last element equals the scalar rolling mean over the full trailing window.
    assert series.iloc[-1] == pytest.approx(fops.ts_mean(VALUES, 3))
    assert series.iloc[-1] == pytest.approx(4.0)


def test_ts_mean_series_leading_incomplete_windows_are_nan():
    index = pd.RangeIndex(len(VALUES))
    series = fops.ts_mean_series(VALUES, 3, index=index)
    # First (window - 1) entries lack a full window -> NaN.
    assert math.isnan(series.iloc[0])
    assert math.isnan(series.iloc[1])
    assert not math.isnan(series.iloc[2])
    assert series.iloc[2] == pytest.approx(2.0)  # mean of [1, 2, 3]


def test_ts_mean_series_equals_ts_sum_series_over_window():
    index = pd.RangeIndex(len(VALUES))
    window = 3
    mean_series = fops.ts_mean_series(VALUES, window, index=index)
    sum_series = fops.ts_sum_series(VALUES, window, index=index)
    expected = sum_series / window
    pd.testing.assert_series_equal(mean_series, expected, check_names=False)


def test_ts_mean_series_preserves_custom_index_and_trailing_alignment():
    # Index longer than the value sequence: values align to the trailing edge and
    # leading positions stay NaN (to_series left-pads the front with NaN).
    labels = ["a", "b", "c", "d", "e", "f", "g"]
    index = pd.Index(labels)
    values = [10.0, 20.0, 30.0, 40.0]  # maps onto the last 4 labels d..g
    series = fops.ts_mean_series(values, 2, index=index)
    assert list(series.index) == labels
    # Front labels carry no data -> NaN.
    assert math.isnan(series.loc["a"])
    # Trailing 2-window means land on the aligned labels.
    assert series.loc["f"] == pytest.approx(25.0)  # mean of [20, 30]
    assert series.loc["g"] == pytest.approx(35.0)  # mean of [30, 40]
