"""Direct unit tests for the vol-target annualization helpers.

These pure helpers live in ``external_alpha_sleeves`` (the location the team lead
accepted as-landed) and bridge a PER-BAR realized-vol estimate to the annual
horizon of a ``target_vol`` knob: the bar cadence is inferred deterministically
from the median observed timestamp spacing, so a sleeve needs no timeframe
parameter and behaves correctly on 1h/4h/1d feeds alike.  They are the
load-bearing piece of the sizing-discipline vol-target fix, so they carry their
own coverage here -- median spacing (odd/even/mixed-type/degenerate), the
365.25-day annualization constant, and the None / pass-through paths.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from lumina_quant.strategies.external_alpha_sleeves import (
    _annualize_per_bar_vol,
    _bars_per_year_from_spacing,
    _median_bar_spacing_seconds,
)

_DAY = 86_400.0
_BASE_EPOCH = 1_700_000_000


def _epochs(count: int, spacing_s: int) -> list[float]:
    return [float(_BASE_EPOCH + i * spacing_s) for i in range(count)]


# --------------------------------------------------------------------------- #
# median bar spacing
# --------------------------------------------------------------------------- #


def test_median_spacing_odd_gap_count_returns_middle_gap() -> None:
    # consecutive gaps: 100, 200, 400 -> median 200
    assert _median_bar_spacing_seconds([0, 100, 300, 700]) == 200.0


def test_median_spacing_even_gap_count_averages_two_middle() -> None:
    # consecutive gaps: 100, 200, 300, 500 -> median = (200 + 300) / 2 = 250
    assert _median_bar_spacing_seconds([0, 100, 300, 600, 1100]) == 250.0


def test_median_spacing_drops_zero_and_negative_gaps() -> None:
    # consecutive gaps: 100, 0 (drop), 200, 400 -> [100, 200, 400] median 200
    assert _median_bar_spacing_seconds([0, 100, 100, 300, 700]) == 200.0


def test_median_spacing_parses_mixed_timestamp_types() -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    dts = [base + timedelta(days=i) for i in range(4)]
    assert _median_bar_spacing_seconds(dts) == _DAY
    iso = ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"]
    assert _median_bar_spacing_seconds(iso) == _DAY
    assert _median_bar_spacing_seconds([_BASE_EPOCH, _BASE_EPOCH + 86_400]) == _DAY


def test_median_spacing_none_paths() -> None:
    assert _median_bar_spacing_seconds([]) is None
    assert _median_bar_spacing_seconds([_BASE_EPOCH]) is None  # < 2 usable points
    assert _median_bar_spacing_seconds(["nope", None, "also-bad"]) is None
    assert _median_bar_spacing_seconds([5, 5, 5]) is None  # no positive gap


# --------------------------------------------------------------------------- #
# bars-per-year (365.25-day annualization constant)
# --------------------------------------------------------------------------- #


def test_bars_per_year_uses_365_25_day_constant() -> None:
    # The 365.25-day year is pinned behaviorally through the delegate: 1d bars
    # map to exactly 365.25 bars/yr (and 1h/4h to their exact multiples).
    assert _bars_per_year_from_spacing(_epochs(6, 86_400)) == 365.25  # 1d
    assert _bars_per_year_from_spacing(_epochs(6, 3_600)) == 365.25 * 24.0  # 1h -> 8766.0
    assert _bars_per_year_from_spacing(_epochs(6, 4 * 3_600)) == 365.25 * 6.0  # 4h -> 2191.5


def test_bars_per_year_none_when_spacing_unknowable() -> None:
    assert _bars_per_year_from_spacing([]) is None
    assert _bars_per_year_from_spacing([_BASE_EPOCH]) is None


# --------------------------------------------------------------------------- #
# annualize per-bar vol
# --------------------------------------------------------------------------- #


def test_annualize_matches_sqrt_bpy_closed_form() -> None:
    daily = _epochs(6, 86_400)
    bpy = _bars_per_year_from_spacing(daily)
    assert bpy is not None
    got = _annualize_per_bar_vol(0.03, daily)
    assert abs(got - 0.03 * math.sqrt(bpy)) < 1e-12
    # Canonical Moreira-Muir example: per-day 0.03 -> ~0.573 annualized.
    assert abs(got - 0.5733) < 1e-3


def test_annualize_scales_up_with_finer_spacing() -> None:
    daily = _epochs(6, 86_400)
    hourly = _epochs(6, 3_600)
    # Same per-bar vol, finer spacing -> more bars/yr -> larger annualized vol.
    assert _annualize_per_bar_vol(0.02, hourly) > _annualize_per_bar_vol(0.02, daily)


def test_annualize_none_when_spacing_unavailable() -> None:
    assert _annualize_per_bar_vol(0.5, []) is None
    assert _annualize_per_bar_vol(0.5, [_BASE_EPOCH]) is None
    assert _annualize_per_bar_vol(0.5, ["bad", None]) is None


def test_annualize_is_deterministic() -> None:
    daily = _epochs(8, 86_400)
    assert _annualize_per_bar_vol(0.037, daily) == _annualize_per_bar_vol(0.037, daily)
