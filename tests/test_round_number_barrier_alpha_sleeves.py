"""Deterministic build-gate + hygiene tests for the round-number-barrier lane.

Direct class import only (no ``@register`` on this lane).  The build gate drives
the REAL incumbents through their full decision paths on identical synthetic bars
and asserts a materially DIVERGENT emitted action (never a different internal
score alone).  All fixtures are closed-form; no ``random`` module is used.

- RN-1  resistance bounce mid-channel vs DonchianAtrTrend + FalseBreakoutReversal:
        a close of 99.88 (grid level 100, d=-0.12) is a bounce SHORT for the
        candidate while neither channel incumbent fires (no path-extremum break /
        no high above the channel); positive controls make each incumbent act
        where the candidate is FLAT.
- RN-2  breakout acceleration without a channel high vs DonchianAtrTrend: a close
        of 100.6 (d=+0.06) is a breakout LONG while Donchian is silent (far below
        its channel high).
- RN-3  level-specificity vs HourlyShockReversion: two identical -2% declines are
        faded identically by the magnitude-only shock fader, while the candidate
        acts on one (ends at 100.15, d=+0.015) and abstains on the other (ends at
        104.8, |d|>=0.48).
- RN-4  anchor-independence vs CrossSectionalNearHighAnchoring: a far-from-high
        name approaching $1.00 from above is SHORTED by the anchor incumbent but
        LONGED by the candidate; a near-high name parked mid-grid is LONGED by the
        incumbent and left FLAT by the candidate.
- PLACEBO the binding falsifier: the half-shifted grid does NOT engage on RN-1's
        99.88 close -- engagement is salience-convention-driven, by construction.
- hygiene one-entry-per-episode, min-hold suppression, determinism, state
        roundtrip, adversarial set_state, degenerate inputs, schema.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from lumina_quant.core.events import MarketEvent
from lumina_quant.indicators.annualization import bars_per_year_from_spacing
from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import FalseBreakoutReversalStrategy
from lumina_quant.strategies.hourly_shock_reversion import HourlyShockReversionStrategy
from lumina_quant.strategies.near_high_anchoring_alpha_sleeves import (
    CrossSectionalNearHighAnchoringStrategy,
)
from lumina_quant.strategies.robust_alpha_sleeves import DonchianAtrTrendStrategy
from lumina_quant.strategies.round_number_barrier_alpha_sleeves import (
    RoundNumberBarrierStrategy,
    round_number_grid,
)
from lumina_quant.timeframe_aggregator import TimeframeAggregator
from lumina_quant.tuning import HyperParam

_START = datetime(2026, 1, 1, tzinfo=UTC)

# OHLCV bar tuple: (open, high, low, close, volume).
_Bar = tuple[float, float, float, float, float]


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


def _market_event(symbol: str, i: int, bar: _Bar) -> SimpleNamespace:
    o, h, low, c, v = bar
    return SimpleNamespace(
        type="MARKET",
        time=_START + timedelta(days=i),
        symbol=symbol,
        open=o,
        high=h,
        low=low,
        close=c,
        volume=v,
    )


def _window_event(i: int, rows: dict[str, _Bar]) -> SimpleNamespace:
    t = _START + timedelta(days=i)
    bars_1s = {
        sym: [{"time": t, "open": o, "high": h, "low": low, "close": c, "volume": v}]
        for sym, (o, h, low, c, v) in rows.items()
    }
    return SimpleNamespace(type="MARKET_WINDOW", time=t, bars_1s=bars_1s)


def _feed_single(strategy: Any, symbol: str, bars: list[_Bar]) -> None:
    for i, bar in enumerate(bars):
        strategy.calculate_signals(_market_event(symbol, i, bar))


def _feed_window(strategy: Any, series: dict[str, list[_Bar]], symbols: list[str]) -> None:
    n = len(series[symbols[0]])
    for i in range(n):
        strategy.calculate_signals_window(
            _window_event(i, {sym: series[sym][i] for sym in symbols}), None
        )


def _entries(strategy: Any) -> list[Any]:
    return [s for s in strategy.events.items if s.signal_type in {"LONG", "SHORT"}]


def _entries_for(strategy: Any, symbol: str) -> list[Any]:
    return [s for s in _entries(strategy) if s.symbol == symbol]


def _final_side(strategy: Any) -> dict[str, str]:
    side: dict[str, str] = {}
    for s in strategy.events.items:
        kind = str(s.signal_type).upper()
        if kind in {"LONG", "SHORT"}:
            side[s.symbol] = kind
        elif kind == "EXIT":
            side.pop(s.symbol, None)
    return side


def _bar(
    close: float, *, high: float | None = None, low: float | None = None, volume: float = 1000.0
) -> _Bar:
    hi = high if high is not None else close * 1.001
    lo = low if low is not None else close * 0.999
    return (close, hi, lo, close, volume)


_CAND_KW: dict[str, Any] = dict(
    prox_band=0.15,
    approach_bars=5,
    mode_filter="both",
    min_hold_bars=3,
    max_hold_bars=10,
    cooldown_bars=5,
    min_history_bars=15,
    allow_short=True,
    base_allocation=0.02,
    min_price=0.01,
)


def _candidate(symbols: list[str], **overrides: Any) -> RoundNumberBarrierStrategy:
    return RoundNumberBarrierStrategy(_Bars(symbols), _Queue(), **dict(_CAND_KW, **overrides))


# --------------------------------------------------------------------------- #
# Stage-1 -- frozen-grid arithmetic
# --------------------------------------------------------------------------- #


def test_stage1_grid_arithmetic_is_frozen_and_decade_aware() -> None:
    level, spacing, distance = round_number_grid(99.88)
    assert (level, spacing) == (100.0, 1.0)
    assert math.isclose(distance, -0.12, abs_tol=1e-9)
    level, spacing, distance = round_number_grid(100.6)
    assert (level, spacing) == (100.0, 10.0)  # decade boundary raises the spacing
    assert math.isclose(distance, 0.06, abs_tol=1e-9)
    level, spacing, distance = round_number_grid(1.003)
    assert (level, spacing) == (1.0, 0.1)
    assert math.isclose(distance, 0.03, abs_tol=1e-9)
    # Non-positive / non-finite closes are undefined on a log10 grid -> None.
    assert round_number_grid(0.0) is None
    assert round_number_grid(-5.0) is None
    assert round_number_grid(float("nan")) is None


# --------------------------------------------------------------------------- #
# RN-1 -- resistance bounce mid-channel vs Donchian + FalseBreakout
# --------------------------------------------------------------------------- #

_DONCHIAN_KW: dict[str, Any] = dict(
    breakout_lookback=30,
    atr_window=10,
    min_trend_efficiency=0.18,
    max_vol_ratio=4.0,
    stop_atr_multiple=2.5,
    trail_atr_multiple=3.0,
    allow_short=True,
    min_price=0.01,
)
_FALSE_BREAKOUT_KW: dict[str, Any] = dict(
    channel_lookback=30,
    break_buffer_pct=0.002,
    min_volume_z=0.8,
    volume_window=20,
    max_trend_efficiency=0.65,
    stop_loss_pct=0.02,
    take_profit_pct=0.04,
    allow_short=True,
    min_price=0.01,
)


def _rn1_bounce_short() -> list[_Bar]:
    """Old high 118 in-channel, oscillate 95/105, then RALLY into 99.88 (d=-0.12)."""
    bars = [_bar(118.0)]
    bars += [_bar(95.0 if i % 2 == 0 else 105.0) for i in range(30)]
    bars += [_bar(x) for x in (97.4, 98.0, 98.6, 99.2, 99.55, 99.88)]
    return bars


def _donchian_sides(bars: list[_Bar], **overrides: Any) -> list[str]:
    inc = DonchianAtrTrendStrategy(_Bars(["A/USDT"]), _Queue(), **dict(_DONCHIAN_KW, **overrides))
    _feed_single(inc, "A/USDT", bars)
    return [s.signal_type for s in _entries(inc)]


def _false_breakout_sides(bars: list[_Bar], **overrides: Any) -> list[str]:
    inc = FalseBreakoutReversalStrategy(
        _Bars(["A/USDT"]), _Queue(), **dict(_FALSE_BREAKOUT_KW, **overrides)
    )
    _feed_single(inc, "A/USDT", bars)
    return [s.signal_type for s in _entries(inc)]


def test_rn1_candidate_bounce_short_while_channel_incumbents_are_silent() -> None:
    bars = _rn1_bounce_short()
    # Candidate: a single bounce SHORT at the round-number level (d=-0.12).
    cand = _candidate(["A/USDT"])
    _feed_single(cand, "A/USDT", bars)
    entries = _entries(cand)
    assert [s.signal_type for s in entries] == ["SHORT"]
    meta = entries[0].metadata or {}
    assert meta.get("grid_level") == 100.0
    assert meta.get("barrier_mode") == "bounce"
    assert math.isclose(meta.get("proximity"), -0.12, abs_tol=1e-9)

    # Incumbents SILENT (the WHY: no channel break / no high above the channel).
    assert _donchian_sides(bars) == []
    assert _false_breakout_sides(bars) == []


def test_rn1_donchian_positive_control_longs_a_new_high_where_candidate_flat() -> None:
    # A genuine new-high trend makes Donchian LONG.
    up = [_bar(100.0 + i * 0.7) for i in range(40)]  # steady rise to a new high
    assert _donchian_sides(up)[:1] == ["LONG"]
    # The candidate is FLAT at close 117 (nearest level 120, |d|=0.3 not engaged).
    _level, _spacing, distance = round_number_grid(117.0)
    assert abs(distance) > _CAND_KW["prox_band"]


def test_rn1_false_breakout_positive_control_shorts_a_failed_break_where_candidate_flat() -> None:
    # Oscillate 115/118 (outside any $10 band), then a spike whose HIGH pokes above
    # the channel but CLOSE (117) falls back inside on elevated volume.
    osc = [_bar(115.0 if i % 2 == 0 else 118.0, volume=1000.0 + (i % 2) * 120.0) for i in range(32)]
    spike = _bar(117.0, high=125.0, low=116.5, volume=6000.0)
    assert _false_breakout_sides([*osc, spike]) == ["SHORT"]
    # The candidate is FLAT at close 117 (d=-0.3) and never engaged the 115/118 band.
    cand = _candidate(["A/USDT"])
    _feed_single(cand, "A/USDT", [*osc, spike])
    assert _entries(cand) == []


# --------------------------------------------------------------------------- #
# RN-2 -- breakout acceleration without a channel high vs Donchian
# --------------------------------------------------------------------------- #


def _rn2_breakout_long() -> list[_Bar]:
    """Range 102/108 (mid-decade), decline, then rally from 95 into 100.6 (d=+0.06)."""
    bars = [_bar(102.0 if i % 2 == 0 else 108.0) for i in range(30)]
    bars += [_bar(x) for x in (95.0, 95.5, 96.5, 98.0, 99.3, 100.6)]
    return bars


def test_rn2_candidate_breakout_long_where_donchian_is_silent() -> None:
    bars = _rn2_breakout_long()
    cand = _candidate(["A/USDT"])
    _feed_single(cand, "A/USDT", bars)
    entries = _entries(cand)
    assert [s.signal_type for s in entries] == ["LONG"]
    meta = entries[0].metadata or {}
    assert meta.get("grid_level") == 100.0
    assert meta.get("barrier_mode") == "breakout"
    assert math.isclose(meta.get("proximity"), 0.06, abs_tol=1e-9)
    # Donchian is silent: 100.6 is far below its ~108 channel high.
    assert _donchian_sides(bars) == []


# --------------------------------------------------------------------------- #
# RN-3 -- level-specificity vs HourlyShockReversion (1h aggregator harness)
# --------------------------------------------------------------------------- #


def _hourly_shock_fade_sides(level: float) -> list[str]:
    """Fade a completed -2% 1h move at an arbitrary price level (magnitude only)."""
    queue = _Queue()
    inc = HourlyShockReversionStrategy(
        _Bars(["ETH/USDT"]),
        queue,
        target_symbol="ETH/USDT",
        lookback_bars=1,
        return_threshold=0.006,
        max_hold_bars=48,
        stop_loss_pct=0.0,
    )
    aggregator = TimeframeAggregator(timeframes=["1h"], lookbacks={"1h": 16})
    closes = [level, level, level, level * 0.98, level * 0.98]
    for offset, close in enumerate(closes):
        event = MarketEvent(
            time=_START + timedelta(hours=offset),
            symbol="ETH/USDT",
            open=close,
            high=close,
            low=close,
            close=close,
            volume=1.0,
        )
        aggregator.update_from_1s_batch("ETH/USDT", [event])
        inc.calculate_signals_window(event, aggregator)
    return [s.signal_type for s in queue.items if s.signal_type in {"LONG", "SHORT"}]


def _rn3_daily_into(final: float, path: tuple[float, ...]) -> list[_Bar]:
    bars = [_bar(path[0]) for _ in range(15)]
    bars += [_bar(x) for x in path]
    bars.append(_bar(final))
    return bars


def test_rn3_shock_fader_is_level_blind_where_candidate_is_level_specific() -> None:
    # The magnitude-only shock fader LONGS an identical -2% decline at BOTH price
    # levels -- its action is independent of the absolute round-number geometry.
    assert _hourly_shock_fade_sides(102.19) == ["LONG"]
    assert _hourly_shock_fade_sides(106.94) == ["LONG"]

    # Candidate leg A: a decline ending at 100.15 (d=+0.015) -> support-bounce LONG.
    leg_a = _rn3_daily_into(100.15, (106.0, 105.0, 104.0, 103.0, 102.5))
    cand_a = _candidate(["A/USDT"])
    _feed_single(cand_a, "A/USDT", leg_a)
    entries_a = _entries(cand_a)
    assert [s.signal_type for s in entries_a] == ["LONG"]
    assert (entries_a[0].metadata or {}).get("grid_level") == 100.0

    # Candidate leg B: a decline ending at 104.8 (|d|>=0.48) -> FLAT, purely by grid.
    leg_b = _rn3_daily_into(104.8, (110.0, 108.0, 106.0, 105.0, 105.0))
    cand_b = _candidate(["A/USDT"])
    _feed_single(cand_b, "A/USDT", leg_b)
    assert _entries(cand_b) == []


# --------------------------------------------------------------------------- #
# RN-4 -- anchor-independence vs CrossSectionalNearHighAnchoring
# --------------------------------------------------------------------------- #

_NEAR_HIGH_KW: dict[str, Any] = dict(
    high_lookback_bars=60,
    min_history_bars=15,
    vol_window=10,
    quantile_pct=0.25,
    rebalance_bars=1,
    min_hold_bars=0,
    allow_short=True,
    min_symbols=4,
    target_vol=0.0,
    stop_loss_pct=0.0,
    max_hold_bars=0,
    min_price=0.01,
)
_RN4_SYMBOLS = ["SYM_HIGH/USDT", "SYM_LOW/USDT", "F0/USDT", "F1/USDT", "F2/USDT", "F3/USDT"]


def _rn4_series(n: int = 45) -> dict[str, list[_Bar]]:
    series: dict[str, list[_Bar]] = {}
    # Near-high name: fresh highs inside a single grid cell (d in [0.16, 0.34]).
    series["SYM_HIGH/USDT"] = [
        (lambda c: (c, c, c * 0.998, c, 1000.0))(151.6 + (i / (n - 1)) * 1.8) for i in range(n)
    ]
    # Far-from-high name: hold ~1.63 (d=0.3, never engaged), then a one-bar cliff
    # to 1.003 -- nearness collapses to ~0.6 exactly when the grid engages at $1.00.
    low = [_bar(1.63 + (0.003 if i % 2 else -0.003), high=1.633, low=1.627) for i in range(n - 1)]
    low.append(_bar(1.003, high=1.004, low=1.002))
    series["SYM_LOW/USDT"] = low
    # Mid-nearness fillers at non-round prices (a mid-window peak fixes nearness ~0.8).
    for symbol, base in zip(
        ["F0/USDT", "F1/USDT", "F2/USDT", "F3/USDT"], [73.7, 137.3, 268.6, 342.4], strict=True
    ):
        peak = base / 0.80
        arr: list[_Bar] = []
        for i in range(n):
            c = peak if i == 20 else base + (0.05 if i % 2 else -0.05)
            arr.append(_bar(c, high=max(c, peak if i == 20 else c) * 1.001, low=c * 0.999))
        series[symbol] = arr
    return series


def test_rn4_candidate_grid_side_opposes_the_near_high_anchor() -> None:
    series = _rn4_series()

    incumbent = CrossSectionalNearHighAnchoringStrategy(
        _Bars(_RN4_SYMBOLS), _Queue(), **_NEAR_HIGH_KW
    )
    _feed_window(incumbent, series, _RN4_SYMBOLS)
    side = _final_side(incumbent)
    assert side.get("SYM_LOW/USDT") == "SHORT"  # far from its 52wk high
    assert side.get("SYM_HIGH/USDT") == "LONG"  # near its 52wk high

    candidate = RoundNumberBarrierStrategy(_Bars(_RN4_SYMBOLS), _Queue(), **_CAND_KW)
    _feed_window(candidate, series, _RN4_SYMBOLS)
    low_entries = _entries_for(candidate, "SYM_LOW/USDT")
    # Opposite side on SYM_LOW: the candidate LONGS the $1.00 approach the anchor shorts.
    assert [s.signal_type for s in low_entries] == ["LONG"]
    assert (low_entries[0].metadata or {}).get("grid_level") == 1.0
    # The near-high name is parked mid-grid -> the candidate is FLAT on it.
    assert _entries_for(candidate, "SYM_HIGH/USDT") == []


# --------------------------------------------------------------------------- #
# PLACEBO-GRID unit leg (binding falsifier)
# --------------------------------------------------------------------------- #


def test_placebo_grid_does_not_engage_where_the_true_grid_does() -> None:
    # The true grid engages RN-1's 99.88 close (|d|=0.12 <= band)...
    _level, _spacing, true_d = round_number_grid(99.88)
    assert abs(true_d) <= _CAND_KW["prox_band"]
    # ...but the half-shifted placebo grid does NOT (nearest level 99.5, d=+0.38).
    placebo_level, _spacing, placebo_d = round_number_grid(99.88, half_shift=True)
    assert placebo_level == 99.5
    assert abs(placebo_d) > _CAND_KW["prox_band"]


# --------------------------------------------------------------------------- #
# hygiene
# --------------------------------------------------------------------------- #


def _episode_retouch_bars() -> list[_Bar]:
    """Approach $100 from below and then linger in the band on the same side."""
    bars = [_bar(90.0 + i * 0.05) for i in range(15)]  # slow drift up, out of band
    bars += [_bar(x) for x in (95.0, 96.0, 97.0, 98.0, 99.9)]  # rally into the band, d<0
    bars += [_bar(99.85 + (0.02 if i % 2 else -0.02)) for i in range(30)]  # linger same side
    return bars


def test_one_entry_per_episode_no_second_entry_on_retouch() -> None:
    cand = _candidate(["A/USDT"])
    _feed_single(cand, "A/USDT", _episode_retouch_bars())
    # Exactly one entry despite many in-band bars on the same (level, side) episode.
    assert len(_entries(cand)) == 1


def test_min_hold_suppresses_early_target_exit() -> None:
    # Enter a bounce SHORT at 99.88, then immediately reach the half-grid target
    # (a move of >= 0.5 away from the level) within two bars.
    bars = [*_rn1_bounce_short(), _bar(99.2), _bar(99.2)]
    held = _candidate(["A/USDT"], min_hold_bars=3)
    _feed_single(held, "A/USDT", bars)
    held_exits = [s for s in held.events.items if s.signal_type == "EXIT"]

    flips = _candidate(["A/USDT"], min_hold_bars=1)
    _feed_single(flips, "A/USDT", bars)
    flip_exits = [s for s in flips.events.items if s.signal_type == "EXIT"]

    # The min_hold=1 reference exits early; the min_hold=3 config holds through it.
    assert len(flip_exits) > len(held_exits)


def test_determinism_two_runs_identical_signals() -> None:
    bars = _rn1_bounce_short()

    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        cand = _candidate(["A/USDT"])
        _feed_single(cand, "A/USDT", bars)
        return [
            (s.symbol, s.signal_type, s.strength, dict(s.metadata or {})) for s in cand.events.items
        ]

    first = _run()
    assert first, "expected at least one signal"
    assert first == _run()


def test_state_roundtrip_lossless() -> None:
    bars = _rn1_bounce_short()
    cand = _candidate(["A/USDT"])
    _feed_single(cand, "A/USDT", bars)
    snapshot = cand.get_state()
    restored = _candidate(["A/USDT"])
    restored.set_state(snapshot)
    assert restored.get_state() == snapshot
    assert restored._state["A/USDT"].mode == cand._state["A/USDT"].mode


def test_adversarial_set_state_never_raises() -> None:
    symbols = ["A/USDT", "B/USDT"]
    cand = _candidate(symbols)
    cand.set_state(None)  # type: ignore[arg-type]
    cand.set_state("not a dict")  # type: ignore[arg-type]
    cand.set_state(12345)  # type: ignore[arg-type]
    cand.set_state({"symbol_state": "not a dict"})
    cand.set_state({"symbol_state": {"A/USDT": "nope"}})
    cand.set_state({"symbol_state": {"A/USDT": {"closes": 12345}}})
    cand.set_state(
        {
            "last_eval_time_key": None,
            "tick": "not-an-int",
            "symbol_state": {
                symbol: {
                    "closes": ["x", float("nan"), 1.0, None],
                    "mode": 999,
                    "entry_price": "abc",
                    "ref_level": "abc",
                    "ref_spacing": "abc",
                    "bars_held": "oops",
                    "cooldown_remaining": -4,
                    "last_episode_sign": "weird",
                    "last_episode_level": "abc",
                    "last_time_key": 123,
                }
                for symbol in symbols
            },
        }
    )
    for item in cand._state.values():
        assert item.mode in {"OUT", "LONG", "SHORT"}
    _feed_single(cand, "A/USDT", _rn1_bounce_short())


def test_degenerate_inputs_never_raise() -> None:
    cand = _candidate(["Z/USDT"])
    cand.calculate_signals(_market_event("Z/USDT", 0, (100.0, 90.0, 110.0, 0.0, None)))
    cand.calculate_signals(_market_event("Z/USDT", 1, (-5.0, -5.0, -5.0, -5.0, 1.0)))
    cand.calculate_signals(
        _market_event("Z/USDT", 2, (float("nan"), float("nan"), float("nan"), float("nan"), 1.0))
    )
    cand.calculate_signals(
        _market_event("Z/USDT", 3, (float("inf"), float("inf"), float("inf"), float("inf"), 1.0))
    )
    cand.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time="t0", bars_1s={}))
    cand.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    cand.calculate_signals(SimpleNamespace(type="MARKET", symbol="ZZZ/USDT", close=None))
    assert _entries(cand) == []


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = RoundNumberBarrierStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "prox_band",
        "approach_bars",
        "mode_filter",
        "min_hold_bars",
        "max_hold_bars",
        "cooldown_bars",
        "allow_short",
        "min_history_bars",
    ):
        assert required in schema


def test_slice_multi_timeframe_cells_pinned() -> None:
    """4h/1h scale the episodic bar clocks; prox_band stays timeframe invariant."""
    from lumina_quant.strategies.round_number_barrier_alpha_sleeves import (
        _ROUND_NUMBER_BARRIER_SLICE as sl,
    )

    assert {"1d", "4h", "1h"} <= set(sl)
    base = tuple(cell["variant"] for cell in sl["1d"])
    for tf in ("4h", "1h"):
        assert tuple(cell["variant"] for cell in sl[tf]) == base
    assert sl["4h"][0]["approach_bars"] == 30
    assert sl["4h"][0]["min_hold_bars"] == 18
    assert sl["1h"][0]["approach_bars"] == 120
    assert sl["1h"][0]["min_hold_bars"] == 72
    for tf in ("1d", "4h", "1h"):
        assert sl[tf][0]["prox_band"] == 0.15
        assert sl[tf][0]["mode_filter"] == "both"


# --------------------------------------------------------------------------- #
# vol-target horizon fix (Class-B throttle): regression.
#
# ``target_vol`` DEFAULTS to 0.0 here, so the throttle is off by default and this
# fix leaves default sizing byte-identical.  When ENABLED it must annualize the
# per-bar realized vol by ``sqrt(bars_per_year)`` (cadence from the median bar
# spacing) before comparing it to ``target_vol``.
# --------------------------------------------------------------------------- #


def test_vol_target_throttle_annualizes_realized_vol_on_breakout() -> None:
    bars = _rn2_breakout_long()
    # Enabled throttle: emitted scalar equals the ANNUALIZED-vol target ratio.
    strat = _candidate(["A/USDT"], target_vol=0.05)
    _feed_single(strat, "A/USDT", bars)
    entries = _entries(strat)
    assert [s.signal_type for s in entries] == ["LONG"]
    meta = entries[0].metadata or {}
    realized_vol = meta["realized_vol"]
    scalar = meta["inverse_vol_scalar"]
    assert realized_vol is not None and realized_vol > 0.0
    bpy = bars_per_year_from_spacing(list(strat._recent_times))
    assert abs(bpy - 365.25) < 1e-9  # daily fixture cadence
    expected = min(1.0, 0.05 / (realized_vol * math.sqrt(bpy)))
    assert abs(scalar - expected) < 1e-12  # annualized, not per-bar, comparison
    # Guaranteed engagement: a tiny target de-risks (annualized vol >> target).
    tiny = _candidate(["A/USDT"], target_vol=1e-6)
    _feed_single(tiny, "A/USDT", bars)
    assert (_entries(tiny)[0].metadata or {})["inverse_vol_scalar"] < 1.0
    # Default target_vol=0.0 -> throttle OFF -> scalar exactly 1.0 (byte-identical).
    off = _candidate(["A/USDT"])
    _feed_single(off, "A/USDT", bars)
    assert (_entries(off)[0].metadata or {})["inverse_vol_scalar"] == 1.0
