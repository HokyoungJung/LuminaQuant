"""Deterministic unit tests for the multi-factor target-pool selector."""

from __future__ import annotations

import polars as pl

from lumina_quant import strategy_factory
from lumina_quant.strategy_factory.universe_selection import (
    DEFAULT_HARD_FILTERS,
    SymbolFactorBreakdown,
    TargetPoolResult,
    TargetPoolSelector,
    UniverseSelectionConfig,
    _dollar_volume,
    _trailing_range_percentile,
    price_position_factor,
    result_to_dict,
    select_target_pool_from_frames,
    volatility_factor,
    volume_factor,
    vwap_factor,
)


def _flat_series(value: float, n: int) -> list[float]:
    return [value] * n


def _ramp_series(start: float, step: float, n: int) -> list[float]:
    return [start + step * i for i in range(n)]


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------
def test_dollar_volume_median_and_none() -> None:
    close = [10.0, 20.0, 30.0]
    volume = [1.0, 1.0, 1.0]
    # median(10, 20, 30) = 20
    assert _dollar_volume(close, volume, window=3) == 20.0
    assert _dollar_volume([1.0], [1.0], window=5) is None


def test_trailing_range_percentile_none_on_short() -> None:
    assert _trailing_range_percentile([1.0], window=10) is None
    pct = _trailing_range_percentile([1.0, 2.0, 3.0, 4.0], window=4)
    # latest (4.0) is the max: below=3, equal=1 -> (3 + 0.5) / 4 = 0.875
    assert pct is not None and pct == 0.875


# ---------------------------------------------------------------------------
# price-position factor
# ---------------------------------------------------------------------------
def test_price_position_midband_high() -> None:
    # Close sits exactly at channel midpoint -> mid-band score ~1.0
    high = _flat_series(2.0, 60)
    low = _flat_series(0.0, 60)
    close = _flat_series(1.0, 60)
    score = price_position_factor(high, low, close, window=60)
    assert score is not None
    assert abs(score - 1.0) < 1e-9


def test_price_position_extreme_low_score() -> None:
    # Close at top of channel -> pos=1 -> mid-band score ~0.0
    high = _flat_series(2.0, 60)
    low = _flat_series(0.0, 60)
    close = [*_flat_series(1.0, 59), 2.0]
    score = price_position_factor(high, low, close, window=60)
    assert score is not None
    assert abs(score) < 1e-9


def test_price_position_degenerate_none() -> None:
    flat = _flat_series(1.0, 60)
    assert price_position_factor(flat, flat, flat, window=60) is None


# ---------------------------------------------------------------------------
# volume factor
# ---------------------------------------------------------------------------
def test_volume_factor_uses_rank_and_surge() -> None:
    close = _flat_series(1.0, 80)
    volume = _flat_series(100.0, 80)
    score = volume_factor(close, volume, window=64, surge_scale=2.0, dollar_volume_rank=1.0)
    assert score is not None
    # constant volume -> surge z ~0 -> surge component ~0.5; dv_rank=1.0
    assert abs(score - 0.75) < 1e-6


def test_volume_factor_none_when_no_components() -> None:
    # Too few bars for surge and no dollar-volume rank supplied.
    assert (
        volume_factor(
            [1.0, 1.0],
            [1.0, 1.0],
            window=64,
            surge_scale=2.0,
            dollar_volume_rank=None,
        )
        is None
    )


def test_volume_surge_is_bounded() -> None:
    close = _flat_series(1.0, 80)
    volume = [*_flat_series(1.0, 79), 1_000_000.0]
    score = volume_factor(close, volume, window=64, surge_scale=2.0, dollar_volume_rank=None)
    assert score is not None
    # Single spike is tanh-clipped; surge component stays within [0, 1].
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# volatility factor
# ---------------------------------------------------------------------------
def test_volatility_in_band_positive() -> None:
    # Build a series with moderate ATR% well inside [0, 1].
    close = _ramp_series(100.0, 0.0, 40)
    high = [c + 1.0 for c in close]
    low = [c - 1.0 for c in close]
    score, in_band = volatility_factor(
        high, low, close, window=14, vol_band_low=0.0, vol_band_high=1.0
    )
    assert in_band is True
    assert score is not None and score >= 0.0


def test_volatility_above_band_fails() -> None:
    # Tiny price with huge range -> ATR% > 1.0 -> out of band.
    close = _flat_series(1.0, 40)
    high = _flat_series(5.0, 40)
    low = _flat_series(0.5, 40)
    score, in_band = volatility_factor(
        high, low, close, window=14, vol_band_low=0.0, vol_band_high=1.0
    )
    assert in_band is False
    assert score is None


def test_volatility_below_band_fails() -> None:
    close = _flat_series(100.0, 40)
    high = [c + 0.01 for c in close]
    low = [c - 0.01 for c in close]
    score, in_band = volatility_factor(
        high, low, close, window=14, vol_band_low=0.01, vol_band_high=1.0
    )
    assert in_band is False
    assert score is None


def test_volatility_zero_vol_dead_flat_fails() -> None:
    # high == low == close -> ATR% == 0 -> untradeable, gated even with a 0.0
    # inclusive lower band.
    flat = _flat_series(100.0, 40)
    score, in_band = volatility_factor(
        flat, flat, flat, window=14, vol_band_low=0.0, vol_band_high=1.0
    )
    assert in_band is False
    assert score is None


# ---------------------------------------------------------------------------
# vwap factor
# ---------------------------------------------------------------------------
def test_vwap_above_flag_and_sign() -> None:
    close = [*_flat_series(100.0, 80), 110.0]
    volume = _flat_series(10.0, 81)
    score, above = vwap_factor(close, volume, window=64, sign=-1.0)
    assert above is True
    assert score is not None and score < 0.0  # mean-reversion penalizes distance


def test_vwap_zero_volume_none() -> None:
    close = _flat_series(100.0, 80)
    volume = _flat_series(0.0, 80)
    score, above = vwap_factor(close, volume, window=64, sign=-1.0)
    assert score is None
    assert above is None


# ---------------------------------------------------------------------------
# selector ranking + tie-breaks
# ---------------------------------------------------------------------------
def _good_symbol(n: int = 260) -> dict[str, list[float]]:
    close = _flat_series(100.0, n)
    high = [c + 1.0 for c in close]
    low = [c - 1.0 for c in close]
    volume = _flat_series(1000.0, n)
    return {"high": high, "low": low, "close": close, "volume": volume}


def test_select_ranks_best_first() -> None:
    series = {
        "AAA/USDT": _good_symbol(),
        # weaker liquidity (lower dollar volume)
        "BBB/USDT": {
            **_good_symbol(),
            "volume": _flat_series(1.0, 260),
        },
    }
    result = TargetPoolSelector().select(series)
    assert isinstance(result, TargetPoolResult)
    assert result.pool[0] == "AAA/USDT"


def test_select_tie_break_lexicographic() -> None:
    payload = _good_symbol()
    series = {
        "ZZZ/USDT": {key: list(val) for key, val in payload.items()},
        "AAA/USDT": {key: list(val) for key, val in payload.items()},
    }
    result = TargetPoolSelector().select(series)
    # Identical factors -> deterministic lexicographic tie-break.
    assert list(result.pool) == ["AAA/USDT", "ZZZ/USDT"]


def test_min_history_bars_flags_but_retains() -> None:
    config = UniverseSelectionConfig(pool_size=5, keep_unscored=True)
    series = {
        "AAA/USDT": _good_symbol(),
        "SHORT/USDT": _good_symbol(n=10),  # below min_history_bars (200)
    }
    result = TargetPoolSelector(config).select(series)
    by_symbol = {bd.symbol: bd for bd in result.breakdowns}
    assert by_symbol["SHORT/USDT"].passed_hard_filters is False
    assert by_symbol["SHORT/USDT"].reason == "short_history"
    # retained in breakdowns and ranked last
    assert result.breakdowns[-1].symbol == "SHORT/USDT"
    # AAA passes -> pool from passing symbols, SHORT excluded since one passes
    assert "AAA/USDT" in result.pool


def test_vol_band_reason() -> None:
    series = {
        "BLOWUP/USDT": {
            "close": _flat_series(1.0, 260),
            "high": _flat_series(5.0, 260),
            "low": _flat_series(0.5, 260),
            "volume": _flat_series(100.0, 260),
        },
    }
    result = TargetPoolSelector().select(series)
    by_symbol = {bd.symbol: bd for bd in result.breakdowns}
    assert by_symbol["BLOWUP/USDT"].passed_hard_filters is False
    assert by_symbol["BLOWUP/USDT"].reason == "vol_band"


# ---------------------------------------------------------------------------
# select_static + identity fallback
# ---------------------------------------------------------------------------
def test_select_static_identity_order_and_dedupe() -> None:
    result = TargetPoolSelector().select_static(["A/USDT", "B/USDT", "A/USDT", ""])
    assert list(result.pool) == ["A/USDT", "B/USDT"]


def test_select_empty_series_identity() -> None:
    result = TargetPoolSelector().select({})
    assert result.pool == ()
    assert result.breakdowns == ()


def test_select_all_none_factors_identity_preserved() -> None:
    # Single-bar frames -> all factors None -> still retained (keep_unscored).
    series = {
        "AAA/USDT": {
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        },
    }
    result = TargetPoolSelector().select(series)
    assert "AAA/USDT" in {bd.symbol for bd in result.breakdowns}


# ---------------------------------------------------------------------------
# select_target_pool_from_frames (polars frames)
# ---------------------------------------------------------------------------
def _frame(n: int, *, volume: float = 1000.0) -> pl.DataFrame:
    close = _flat_series(100.0, n)
    return pl.DataFrame(
        {
            "datetime": list(range(n)),
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": _flat_series(volume, n),
        }
    )


def test_from_frames_ranks_and_breaks_down() -> None:
    loaded = {
        "AAA/USDT": _frame(260, volume=5000.0),
        "BBB/USDT": _frame(260, volume=1.0),
    }
    result = select_target_pool_from_frames(loaded)
    assert {bd.symbol for bd in result.breakdowns} == {"AAA/USDT", "BBB/USDT"}
    assert result.pool[0] == "AAA/USDT"
    assert len(result.config_fingerprint) == 12


def test_from_frames_empty_uses_static_fallback() -> None:
    result = select_target_pool_from_frames({}, fallback_symbols=["X/USDT", "Y/USDT"])
    assert list(result.pool) == ["X/USDT", "Y/USDT"]


def test_from_frames_missing_column_falls_back() -> None:
    bad = pl.DataFrame({"datetime": [0, 1], "close": [1.0, 2.0]})
    result = select_target_pool_from_frames({"AAA/USDT": bad}, fallback_symbols=["AAA/USDT"])
    assert list(result.pool) == ["AAA/USDT"]


def test_from_frames_protected_always_kept() -> None:
    # A blow-up symbol would normally be gated out; protected keeps it.
    blowup = pl.DataFrame(
        {
            "datetime": list(range(260)),
            "open": _flat_series(1.0, 260),
            "high": _flat_series(5.0, 260),
            "low": _flat_series(0.5, 260),
            "close": _flat_series(1.0, 260),
            "volume": _flat_series(100.0, 260),
        }
    )
    loaded = {"ANCHOR/USDT": blowup, "AAA/USDT": _frame(260, volume=5000.0)}
    result = select_target_pool_from_frames(loaded, protected=["ANCHOR/USDT"])
    assert "ANCHOR/USDT" in result.pool


def test_from_frames_protected_brand_new_symbol() -> None:
    loaded = {"AAA/USDT": _frame(260, volume=5000.0)}
    result = select_target_pool_from_frames(loaded, protected=["NEW/USDT"])
    assert "NEW/USDT" in result.pool
    assert "NEW/USDT" in {bd.symbol for bd in result.breakdowns}


def test_from_frames_null_rows_stay_aligned() -> None:
    # A null in ONE column must drop the WHOLE row, taking the co-located close
    # spike with it. Per-column null stripping would desync the columns and
    # leave the 1e6 close spike mismatched against a shifted high, blowing ATR%
    # out of band. Correct row-alignment drops row 5 entirely -> clean low-vol.
    n = 260
    close = _flat_series(100.0, n)
    close[5] = 1_000_000.0  # spike co-located with the null high
    high = [c + 1.0 for c in _flat_series(100.0, n)]
    high[5] = None
    frame = pl.DataFrame(
        {
            "datetime": list(range(n)),
            "open": _flat_series(100.0, n),
            "high": high,
            "low": [c - 1.0 for c in _flat_series(100.0, n)],
            "close": close,
            "volume": _flat_series(1000.0, n),
        }
    )
    result = select_target_pool_from_frames({"AAA/USDT": frame})
    by_symbol = {bd.symbol: bd for bd in result.breakdowns}
    assert by_symbol["AAA/USDT"].reason != "vol_band"
    assert by_symbol["AAA/USDT"].passed_hard_filters is True
    assert "AAA/USDT" in result.pool


# ---------------------------------------------------------------------------
# config + serialization
# ---------------------------------------------------------------------------
def test_config_fingerprint_stable_and_sensitive() -> None:
    a = UniverseSelectionConfig().fingerprint()
    b = UniverseSelectionConfig().fingerprint()
    c = UniverseSelectionConfig(pool_size=99).fingerprint()
    assert a == b
    assert a != c


def test_result_to_dict_is_json_friendly() -> None:
    result = select_target_pool_from_frames({"AAA/USDT": _frame(260)})
    payload = result_to_dict(result)
    import json

    json.dumps(payload)  # must not raise
    assert "pool" in payload and "breakdowns" in payload


def test_default_hard_filters_present() -> None:
    assert "vol_band_high" in DEFAULT_HARD_FILTERS
    assert "min_history_bars" in DEFAULT_HARD_FILTERS


def test_breakdown_dataclass_frozen() -> None:
    bd = SymbolFactorBreakdown(
        symbol="A/USDT",
        raw={},
        normalized={},
        composite=0.0,
        passed_hard_filters=True,
        reason="",
    )
    assert bd.symbol == "A/USDT"


# ---------------------------------------------------------------------------
# export contract
# ---------------------------------------------------------------------------
def test_strategy_factory_exports_universe_selection() -> None:
    exported = strategy_factory.__all__
    for name in (
        "TargetPoolSelector",
        "UniverseSelectionConfig",
        "TargetPoolResult",
        "SymbolFactorBreakdown",
        "select_target_pool_from_frames",
        "DEFAULT_FACTOR_WEIGHTS",
        "DEFAULT_HARD_FILTERS",
    ):
        assert name in exported
        assert hasattr(strategy_factory, name)
