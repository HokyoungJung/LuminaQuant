"""Deterministic tests for the volume-clock momentum rider (Lane C).

Imports ``VolumeClockMomentumRiderStrategy`` DIRECTLY -- this lane ships
without the ``@register`` decorator (live-safety: registration + the
research_only tier hint are applied atomically in a later, separate wave), so
these tests deliberately carry NO registry/tier assertions and NO
candidate-wiring tests.

Covers: volume-bar construction math from known per-bar volumes (asserted via
``get_state``); a single abnormally heavy wall-clock bar deterministically
closing MANY volume bars in one step, each stamped with that bar's close
price; a clean uptrend with steady volume producing a LONG entry once enough
volume bars accumulate; the core DECORRELATION DEMONSTRATION -- the exact
same price path with volume concentrated in the flat/chop segment versus the
trend segment produces DIFFERENT entry bar indices under the volume clock,
while a plain wall-clock momentum check (which ignores volume entirely) is
blind to the difference by construction; the stale-clock liveness guard
suppressing all signals in a near-zero-volume regime regardless of price
trend; never-raise safety (zero/missing volume, degenerate closes, empty
window, adversarial set_state); get_state/set_state roundtrip equality; and
determinism (two identical runs produce bit-identical signal streams and
state). No backtest is run.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

from lumina_quant.strategies.volume_clock_alpha_sleeves import (
    VolumeClockMomentumRiderStrategy,
)

_START = datetime(2026, 1, 1, tzinfo=UTC)


class _BarsNoFeatures:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Events:
    def __init__(self) -> None:
        self.signals: list[Any] = []

    def put(self, event: Any) -> None:
        self.signals.append(event)


class _Market:
    type = "MARKET"

    def __init__(
        self,
        time: str,
        symbol: str,
        close: float,
        *,
        volume: float = 1000.0,
        high: float | None = None,
        low: float | None = None,
        open_: float | None = None,
    ) -> None:
        self.time = time
        self.open = open_ if open_ is not None else close
        self.high = high if high is not None else close * 1.0005
        self.low = low if low is not None else close * 0.9995
        self.close = close
        self.symbol = symbol
        self.volume = volume


def _ts(i: int, *, step_minutes: int = 30) -> str:
    return (_START + timedelta(minutes=step_minutes * i)).isoformat()


def _entries(events: _Events) -> list[Any]:
    return [e for e in events.signals if e.signal_type in {"LONG", "SHORT"}]


def _lcg(seed: int):
    state = seed
    while True:
        state = (1103515245 * state + 12345) % (2**31)
        yield (state / 2**31 - 0.5) * 2.0


_BASE_PARAMS: dict[str, Any] = {
    "trail_atr_mult": 4.0,
    "atr_period": 5,
    "max_adds": 0,
    "add_step_atr": 1.0,
    "vol_window": 10,
    "target_vol": 0.0,
    "max_hold_bars": 100000,
    "allow_short": True,
    "add_alloc_fraction": 0.0,
    "target_allocation": 0.0,
    "max_order_value": 0.0,
    "min_price": 0.0,
}


def _strategy(symbol: str, events: _Events, **overrides: Any) -> VolumeClockMomentumRiderStrategy:
    params: dict[str, Any] = {
        "vbar_ref_window": 10,
        "vbar_mult": 1.0,
        "vmom_bars": 5,
        "vmom_entry": 0.02,
        "max_stale_bars": 100000,
        **_BASE_PARAMS,
    }
    params.update(overrides)
    return VolumeClockMomentumRiderStrategy(_BarsNoFeatures([symbol]), events, **params)


# --------------------------------------------------------------------------- #
# Volume-bar construction math
# --------------------------------------------------------------------------- #
def test_vbar_construction_from_known_volumes() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    strategy = _strategy(
        symbol, events, vbar_ref_window=5, vbar_mult=1.0, vmom_bars=3, vmom_entry=1.0
    )
    # Constant close=10.0, volume=100.0 -> constant dollar volume 1000.0 per bar.
    # The rolling median self-normalizes to 1000.0 immediately (median of a
    # constant series is that constant), so V == this bar's dollar volume and
    # exactly ONE volume bar closes per wall-clock bar.
    for i in range(5):
        strategy.calculate_signals(_Market(_ts(i), symbol, 10.0, volume=100.0))
    state = strategy.get_state()
    vc = state["volume_clock"][symbol]
    assert vc["vbar_closes"] == [10.0, 10.0, 10.0, 10.0, 10.0], vc["vbar_closes"]
    assert vc["wall_bar_count"] == 5, vc["wall_bar_count"]
    assert vc["last_vbar_wall_index"] == 5, vc["last_vbar_wall_index"]


def test_heavy_bar_closes_multiple_volume_bars_deterministically() -> None:
    symbol = "ETH/USDT"
    events = _Events()
    # A large vmom_bars keeps the vbar_closes deque (maxlen = vmom_bars + 8)
    # big enough to hold every volume bar the heavy bar produces, so the full
    # count is directly observable (not truncated by the bounded history).
    strategy = _strategy(
        symbol, events, vbar_ref_window=20, vbar_mult=1.0, vmom_bars=45, vmom_entry=1.0
    )
    for i in range(4):
        strategy.calculate_signals(_Market(_ts(i), symbol, 10.0, volume=1.0))
    before = strategy.get_state()["volume_clock"][symbol]
    assert before["vbar_closes"] == [10.0, 10.0, 10.0, 10.0], before["vbar_closes"]
    assert before["wall_bar_count"] == 4, before["wall_bar_count"]

    # ONE abnormally heavy bar: dollar volume 2000.0 vs. a reference (median)
    # still anchored near the 10.0-baseline history -> many volume bars close
    # in a single wall-clock step, ALL stamped with this bar's close price
    # (the honest OHLCV-granularity approximation the module docstring
    # describes: there is no tick data to know the true intra-bar sequence).
    strategy.calculate_signals(_Market(_ts(4), symbol, 20.0, volume=100.0))
    after = strategy.get_state()["volume_clock"][symbol]
    assert len(after["vbar_closes"]) > 1, after["vbar_closes"]
    assert all(c == 20.0 for c in after["vbar_closes"]), after["vbar_closes"]
    # Only ONE wall-clock bar was consumed to produce this whole burst.
    assert after["wall_bar_count"] == before["wall_bar_count"] + 1, after["wall_bar_count"]
    assert after["last_vbar_wall_index"] == after["wall_bar_count"], after["last_vbar_wall_index"]


# --------------------------------------------------------------------------- #
# Clean uptrend, steady volume -> LONG entry
# --------------------------------------------------------------------------- #
def test_uptrend_steady_volume_enters_long() -> None:
    symbol = "BNB/USDT"
    events = _Events()
    strategy = _strategy(symbol, events, vbar_ref_window=10, vmom_bars=5, vmom_entry=0.02)
    price = 100.0
    for i in range(60):
        price *= 1.005
        strategy.calculate_signals(_Market(_ts(i), symbol, price, volume=1000.0))
    entries = _entries(events)
    assert entries, "expected a LONG entry once enough volume bars accumulate"
    assert entries[0].signal_type == "LONG", entries[0].signal_type
    metadata = entries[0].metadata or {}
    assert metadata.get("vclock_momentum") is not None and metadata["vclock_momentum"] > 0.0
    assert metadata.get("vbar_stale") is False, metadata


def test_downtrend_steady_volume_enters_short() -> None:
    symbol = "SOL/USDT"
    events = _Events()
    strategy = _strategy(symbol, events, vbar_ref_window=10, vmom_bars=5, vmom_entry=0.02)
    price = 100.0
    for i in range(60):
        price *= 0.995
        strategy.calculate_signals(_Market(_ts(i), symbol, price, volume=1000.0))
    entries = _entries(events)
    assert entries, "expected a SHORT entry once enough volume bars accumulate"
    assert entries[0].signal_type == "SHORT", entries[0].signal_type
    metadata = entries[0].metadata or {}
    assert metadata.get("vclock_momentum") is not None and metadata["vclock_momentum"] < 0.0


# --------------------------------------------------------------------------- #
# THE DECORRELATION DEMONSTRATION
# --------------------------------------------------------------------------- #
def _chop_then_trend_prices(
    *, chop_bars: int, trend_bars: int, seed: int, chop_noise: float, trend_growth: float
) -> list[float]:
    gen = _lcg(seed)
    price = 100.0
    closes: list[float] = []
    for _ in range(chop_bars):
        price *= math.exp(chop_noise * next(gen))
        closes.append(price)
    for _ in range(trend_bars):
        price *= trend_growth
        closes.append(price)
    return closes


def _first_entry_bar(
    symbol: str, closes: list[float], volumes: list[float], **overrides: Any
) -> tuple[int, str] | None:
    events = _Events()
    strategy = _strategy(symbol, events, **overrides)
    for i, (close, volume) in enumerate(zip(closes, volumes, strict=True)):
        strategy.calculate_signals(_Market(_ts(i), symbol, close, volume=volume))
        entries = _entries(events)
        if entries:
            return i, entries[0].signal_type
    return None


def _first_wallclock_entry_bar(
    closes: list[float], *, vmom_bars: int, vmom_entry: float
) -> int | None:
    """Return the first bar index where a plain (volume-blind) momentum check fires.

    This mirrors the SAME log-return-over-N-bars arithmetic every wall-clock
    trend sleeve in the registry uses, applied directly to wall-clock bars
    (never volume bars): it cannot see the volume layout at all, so it is
    identical for any two volume distributions over the same price path.
    """
    for i in range(vmom_bars, len(closes)):
        momentum = math.log(closes[i] / closes[i - vmom_bars])
        if abs(momentum) >= vmom_entry:
            return i
    return None


def test_volume_concentration_shifts_entry_timing_vs_wallclock() -> None:
    # The IDENTICAL price path: a flat/chop segment followed by a clean
    # uptrend segment.
    chop_bars, trend_bars = 30, 40
    closes = _chop_then_trend_prices(
        chop_bars=chop_bars,
        trend_bars=trend_bars,
        seed=7,
        chop_noise=0.0006,
        trend_growth=1.006,
    )
    vol_low, vol_high = 10.0, 50.0
    # Scenario A: volume concentrated in the CHOP segment, thin in the trend.
    volumes_a = [vol_high] * chop_bars + [vol_low] * trend_bars
    # Scenario B: volume concentrated in the TREND segment, thin in the chop.
    volumes_b = [vol_low] * chop_bars + [vol_high] * trend_bars

    overrides = {"vbar_ref_window": 20, "vbar_mult": 1.0, "vmom_bars": 5, "vmom_entry": 0.02}
    entry_a = _first_entry_bar("BTC/USDT", closes, volumes_a, **overrides)
    entry_b = _first_entry_bar("ETH/USDT", closes, volumes_b, **overrides)

    assert entry_a is not None, "expected scenario A to eventually enter"
    assert entry_b is not None, "expected scenario B to eventually enter"
    bar_a, _side_a = entry_a
    bar_b, _side_b = entry_b

    # Same price path, same entry threshold -- but a DIFFERENT volume layout
    # produces a DIFFERENT entry bar under the volume clock. This is the
    # concrete decorrelation mechanism the module docstring claims.
    assert bar_a != bar_b, (bar_a, bar_b)

    wallclock_bar = _first_wallclock_entry_bar(closes, vmom_bars=5, vmom_entry=0.02)
    assert wallclock_bar is not None
    # The plain wall-clock check is blind to volume by construction: it is the
    # SAME regardless of layout, and (in this construction) differs from BOTH
    # volume-clock timings -- demonstrating the volume clock is not simply
    # reproducing the wall-clock signal under a relabeling.
    assert wallclock_bar != bar_a, (wallclock_bar, bar_a)
    assert wallclock_bar != bar_b, (wallclock_bar, bar_b)


# --------------------------------------------------------------------------- #
# Stale-clock liveness guard
# --------------------------------------------------------------------------- #
def test_stale_clock_near_zero_volume_blocks_all_entries() -> None:
    symbol = "ADA/USDT"
    events = _Events()
    strategy = _strategy(
        symbol,
        events,
        vbar_ref_window=10,
        vmom_bars=5,
        vmom_entry=0.001,
        max_stale_bars=15,
    )
    # A strong, clean uptrend -- but with (near-)zero volume throughout, the
    # volume clock never ticks, so the liveness guard suppresses every entry
    # regardless of how strong the underlying price trend is.
    price = 100.0
    for i in range(80):
        price *= 1.01
        strategy.calculate_signals(_Market(_ts(i), symbol, price, volume=0.0))
    assert _entries(events) == []
    state = strategy.get_state()["volume_clock"][symbol]
    assert state["last_vbar_wall_index"] is None, state


# --------------------------------------------------------------------------- #
# Never-raise safety
# --------------------------------------------------------------------------- #
def test_never_raise_on_degenerate_inputs() -> None:
    symbol = "XRP/USDT"
    events = _Events()
    strategy = _strategy(symbol, events)

    class _EmptyWindow:
        type = "MARKET_WINDOW"
        bars_1s: list[Any] = []

    empty = _EmptyWindow()
    empty.symbol = symbol
    strategy.calculate_signals(empty)  # must not raise
    strategy.calculate_signals(_Market(_ts(0), symbol, 0.0, volume=0.0))  # degenerate close
    strategy.calculate_signals(_Market(_ts(1), symbol, -5.0, volume=100.0))  # negative close
    strategy.calculate_signals(_Market(_ts(2), symbol, 100.0, volume=0.0))  # zero volume
    strategy.calculate_signals(_Market(_ts(3), symbol, 100.0))  # normal bar

    class _NoVolume:
        type = "MARKET"
        time = _ts(4)
        open = 100.0
        high = 100.5
        low = 99.5
        close = 100.0
        volume = None

    no_volume = _NoVolume()
    no_volume.symbol = symbol
    strategy.calculate_signals(no_volume)  # missing volume attribute value
    assert _entries(events) == []


def test_adversarial_set_state_never_raises() -> None:
    symbol = "BTC/USDT"
    strategy = _strategy(symbol, _Events())
    strategy.set_state(None)
    strategy.set_state({})
    strategy.set_state({"volume_clock": "garbage"})
    strategy.set_state({"volume_clock": {symbol: "nope"}})
    strategy.set_state(
        {
            "volume_clock": {
                symbol: {
                    "vbar_closes": "not-a-list",
                    "dv_history": 123,
                    "dv_accum": "abc",
                    "wall_bar_count": "abc",
                    "last_vbar_wall_index": "abc",
                }
            }
        }
    )
    strategy.set_state(
        {
            "volume_clock": {
                symbol: {
                    "vbar_closes": [float("nan"), "x", None, 1.0],
                    "dv_history": [float("inf"), None, "y", 2.0],
                    "dv_accum": float("nan"),
                    "wall_bar_count": -5,
                    "last_vbar_wall_index": None,
                }
            }
        }
    )
    strategy.set_state({"volume_clock": {"NOPE/USDT": {"vbar_closes": [1.0, 2.0]}}})
    strategy.set_state({"symbol_state": {symbol: {"closes": [float("nan"), "x", None]}}})


# --------------------------------------------------------------------------- #
# State roundtrip
# --------------------------------------------------------------------------- #
def test_state_roundtrip() -> None:
    symbol = "TRX/USDT"
    events = _Events()
    strategy = _strategy(symbol, events, vbar_ref_window=10, vmom_bars=5, vmom_entry=0.02)
    price = 100.0
    for i in range(40):
        price *= 1.004
        strategy.calculate_signals(_Market(_ts(i), symbol, price, volume=750.0))
    snapshot = strategy.get_state()
    restored = _strategy(symbol, _Events(), vbar_ref_window=10, vmom_bars=5, vmom_entry=0.02)
    restored.set_state(snapshot)
    again = restored.get_state()
    assert again["volume_clock"][symbol] == snapshot["volume_clock"][symbol]
    assert again["symbol_state"][symbol] == snapshot["symbol_state"][symbol]


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_determinism_two_runs_identical() -> None:
    symbol = "DOGE/USDT"
    price_gen = _lcg(20260703)
    volume_gen = _lcg(998244353)
    price = 100.0
    closes: list[float] = []
    volumes: list[float] = []
    for _ in range(90):
        drift = 0.0025 + 0.01 * next(price_gen)
        price = max(1.0, price * (1.0 + drift))
        closes.append(price)
        volumes.append(max(0.0, 500.0 + 400.0 * next(volume_gen)))

    def _run() -> tuple[list[Any], dict[str, Any]]:
        events = _Events()
        strategy = _strategy(symbol, events, vbar_ref_window=15, vmom_bars=6, vmom_entry=0.015)
        for i, (close, volume) in enumerate(zip(closes, volumes, strict=True)):
            strategy.calculate_signals(_Market(_ts(i), symbol, close, volume=volume))
        signals = [(e.signal_type, e.symbol, e.price, e.metadata) for e in events.signals]
        return signals, strategy.get_state()

    signals_a, state_a = _run()
    signals_b, state_b = _run()
    assert signals_a == signals_b
    assert state_a == state_b
