"""Unit tests for the canonical vol-annualization helpers."""

from __future__ import annotations

import math

from lumina_quant.indicators.annualization import (
    _SECONDS_PER_YEAR,
    annualize_per_bar_vol,
    bars_per_year_from_spacing,
    median_bar_spacing_seconds,
)

_HOUR = 3600.0
_DAY = 24.0 * _HOUR


def _epochs(spacing: float, count: int, start: float = 1_700_000_000.0) -> list[float]:
    return [start + i * spacing for i in range(count)]


def test_median_spacing_hourly() -> None:
    assert median_bar_spacing_seconds(_epochs(_HOUR, 10)) == _HOUR


def test_median_spacing_ignores_duplicates_and_junk() -> None:
    times = [1_700_000_000.0, 1_700_000_000.0, "junk", None, 1_700_003_600.0, 1_700_007_200.0]
    assert median_bar_spacing_seconds(times) == _HOUR


def test_median_spacing_none_paths() -> None:
    assert median_bar_spacing_seconds([]) is None
    assert median_bar_spacing_seconds([1.0]) is None
    assert median_bar_spacing_seconds([5.0, 5.0, 5.0]) is None
    assert median_bar_spacing_seconds(None) is None
    assert median_bar_spacing_seconds(12345) is None


def test_bars_per_year_conventions() -> None:
    assert bars_per_year_from_spacing(_epochs(_DAY, 8)) == _SECONDS_PER_YEAR / _DAY
    assert bars_per_year_from_spacing(_epochs(4 * _HOUR, 8)) == _SECONDS_PER_YEAR / (4 * _HOUR)
    assert bars_per_year_from_spacing(_epochs(_HOUR, 8)) == _SECONDS_PER_YEAR / _HOUR
    # 365.25-day year: daily bars => 365.25 bars/year.
    assert bars_per_year_from_spacing(_epochs(_DAY, 8)) == 365.25


def test_annualize_engages_throttle_at_hourly_scale() -> None:
    # Per-1h-bar vol 0.005 => annualized ~ 0.005 * sqrt(8766) ~ 0.468 > 0.20:
    # the throttle a raw per-bar comparison would never engage now engages.
    bpy = bars_per_year_from_spacing(_epochs(_HOUR, 16))
    assert bpy is not None
    ann = annualize_per_bar_vol(0.005, bpy)
    assert ann is not None
    assert math.isclose(ann, 0.005 * math.sqrt(bpy), rel_tol=1e-12)
    assert ann > 0.20


def test_annualize_none_propagation_and_guards() -> None:
    assert annualize_per_bar_vol(0.01, None) is None
    assert annualize_per_bar_vol(0.01, 0.0) is None
    assert annualize_per_bar_vol(0.01, -5.0) is None
    assert annualize_per_bar_vol(-0.01, 365.25) is None
    assert annualize_per_bar_vol(float("nan"), 365.25) is None
    assert annualize_per_bar_vol("junk", 365.25) is None


def test_determinism_run_twice() -> None:
    times = _epochs(4 * _HOUR, 32)
    first = annualize_per_bar_vol(0.0123, bars_per_year_from_spacing(times))
    second = annualize_per_bar_vol(0.0123, bars_per_year_from_spacing(times))
    assert first == second
