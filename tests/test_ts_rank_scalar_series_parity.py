"""Parity tests asserting scalar ts_rank == ts_rank_series tail for all inputs."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from lumina_quant.compute import ops as compute_ops
from lumina_quant.indicators import formulaic_operators as fops


def _scalar_via_ops(values: list[float], window: int) -> float | None:
    return compute_ops.ts_rank(values, window=window)


def _series_tail(values: list[float], window: int) -> float | None:
    series = pd.Series(values, dtype=float)
    result = compute_ops.rolling_rank_series(series, window=window)
    val = result.iloc[-1]
    return None if math.isnan(val) else float(val)


def _fops_series_tail(values: list[float], window: int) -> float | None:
    idx = pd.RangeIndex(len(values))
    result = fops.ts_rank_series(values, window, index=idx)
    val = result.iloc[-1]
    return None if math.isnan(val) else float(val)


# ---------------------------------------------------------------------------
# Parametrized parity cases
# ---------------------------------------------------------------------------

PARITY_CASES = [
    # (description, values, window)
    ("ascending unique", [1.0, 2.0, 3.0, 4.0, 5.0], 5),
    ("ascending unique window=4", [1.0, 2.0, 3.0, 4.0, 5.0], 4),
    ("ascending unique window=3", [1.0, 2.0, 3.0, 4.0, 5.0], 3),
    ("descending unique", [5.0, 4.0, 3.0, 2.0, 1.0], 5),
    ("all equal", [3.0, 3.0, 3.0, 3.0], 4),
    ("ties in middle not latest", [1.0, 2.0, 2.0, 4.0, 5.0], 5),
    ("ties include latest", [1.0, 2.0, 3.0, 5.0, 5.0], 5),
    ("all tied at latest", [3.0, 3.0, 3.0, 5.0, 5.0], 5),
    ("single element window", [1.0, 2.0, 3.0], 2),
    ("large window exactly filled", [float(i) for i in range(1, 21)], 20),
    ("latest is minimum", [5.0, 4.0, 3.0, 2.0, 1.0], 4),
    ("two elements equal max", [1.0, 2.0, 3.0, 4.0, 4.0], 5),
]


@pytest.mark.parametrize("desc,values,window", PARITY_CASES, ids=[c[0] for c in PARITY_CASES])
def test_scalar_equals_series_tail(desc, values, window):
    scalar = _scalar_via_ops(values, window)
    series_val = _series_tail(values, window)
    assert scalar is not None, f"{desc}: scalar returned None unexpectedly"
    assert series_val is not None, f"{desc}: series returned None unexpectedly"
    assert abs(scalar - series_val) < 1e-10, f"{desc}: scalar={scalar} != series={series_val}"


@pytest.mark.parametrize("desc,values,window", PARITY_CASES, ids=[c[0] for c in PARITY_CASES])
def test_fops_ts_rank_equals_ts_rank_series_tail(desc, values, window):
    """fops.ts_rank (scalar) must agree with fops.ts_rank_series (vectorized) tail."""
    scalar = fops.ts_rank(values, window=window)
    series_val = _fops_series_tail(values, window)
    assert scalar is not None, f"{desc}: scalar ts_rank returned None"
    assert series_val is not None, f"{desc}: ts_rank_series returned None"
    assert abs(scalar - series_val) < 1e-10, (
        f"{desc}: ts_rank={scalar} != ts_rank_series.tail={series_val}"
    )


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_scalar_nan_latest_returns_none():
    """NaN as the latest value should return None from scalar ts_rank."""
    values = [1.0, 2.0, float("nan")]
    result = compute_ops.ts_rank(values, window=3)
    assert result is None


def test_series_nan_latest_returns_nan():
    """NaN as the latest value should leave NaN in the series output."""
    values = [1.0, 2.0, float("nan")]
    series = pd.Series(values, dtype=float)
    result = compute_ops.rolling_rank_series(series, window=3)
    assert math.isnan(result.iloc[-1])


def test_scalar_nan_in_window_excluded():
    """NaN elements inside the window are ignored; result still returned for finite latest."""
    # window of 4: [1.0, nan, 2.0, 3.0] — 3 finite values, latest=3.0 is max
    values = [1.0, float("nan"), 2.0, 3.0]
    result = compute_ops.ts_rank(values, window=4)
    assert result is not None
    # 3.0 is highest among [1, 2, 3]: below=2, equal=1 -> (2 + 1) / 3 = 1.0
    assert abs(result - 1.0) < 1e-10


def test_short_series_returns_none():
    """Series shorter than window returns None from scalar, NaN from series."""
    values = [1.0, 2.0]
    assert compute_ops.ts_rank(values, window=5) is None
    series = pd.Series(values, dtype=float)
    result = compute_ops.rolling_rank_series(series, window=5)
    assert all(math.isnan(v) for v in result.tolist())


# ---------------------------------------------------------------------------
# Canonical pct-rank values (regression)
# ---------------------------------------------------------------------------


def test_ts_rank_max_of_unique_is_one():
    """Unique maximum should rank as 1.0 (matches pandas rank(pct=True))."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert abs(compute_ops.ts_rank(values, window=5) - 1.0) < 1e-10


def test_ts_rank_min_of_unique():
    """Unique minimum of N should rank as 1/N."""
    values = [5.0, 4.0, 3.0, 2.0, 1.0]
    # latest=1.0 is minimum: below=0, equal=1 -> (0+1)/5 = 0.2
    result = compute_ops.ts_rank(values, window=5)
    assert abs(result - 0.2) < 1e-10


def test_ts_rank_tie_convention_matches_pandas():
    """Tie handling must match pandas rank(pct=True) average method."""
    values = [1.0, 2.0, 3.0, 3.0, 3.0]
    # pandas rank: positions 3,4,5 tied -> avg rank = 4; pct = 4/5 = 0.8
    scalar = compute_ops.ts_rank(values, window=5)
    pandas_pct = float(pd.Series(values).rank(pct=True).iloc[-1])
    assert abs(scalar - pandas_pct) < 1e-10


def test_rolling_rank_series_tail_matches_pandas_pct_rank():
    """rolling_rank_series tail must equal pandas rank(pct=True) for uniform inputs."""
    for values in [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, 2.0, 3.0, 3.0, 3.0],
        [3.0, 3.0, 3.0, 3.0, 3.0],
        [5.0, 4.0, 3.0, 2.0, 1.0],
    ]:
        series = pd.Series(values, dtype=float)
        actual = float(compute_ops.rolling_rank_series(series, window=5).iloc[-1])
        expected = float(series.rank(pct=True).iloc[-1])
        assert abs(actual - expected) < 1e-10, f"values={values}: got {actual}, want {expected}"
