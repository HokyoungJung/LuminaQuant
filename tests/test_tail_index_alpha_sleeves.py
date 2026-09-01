"""Deterministic tests for the Hill tail-index regime rider sleeve.

Covers: a tape whose recent loss tail becomes systematically heavier under a
downtrend (fattening regime) enters SHORT with the hill ratio in metadata; a
calm tape whose recent loss tail is thinner/more-uniform under an uptrend
(thinning regime) enters LONG; a flat tape (no losses) and a too-short tape
(insufficient history) never enter and never raise; None/empty-window/
non-finite-close safety; adversarial ``set_state`` payloads never raise;
get_state/set_state roundtrip; and bit-for-bit determinism across two
independent runs of the same seeded tape.  This sleeve is intentionally NOT
registered yet (see the deferred ``@register`` comment in the source module),
so this file imports the class directly and makes no registry assertions.
No backtest is run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from lumina_quant.strategies.tail_index_alpha_sleeves import (
    TailIndexRegimeRiderStrategy,
    _TAIL_INDEX_REGIME_RIDER_SLICE,
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
        high: float | None = None,
        low: float | None = None,
    ) -> None:
        self.time = time
        self.open = close
        self.high = high if high is not None else close * 1.0005
        self.low = low if low is not None else close * 0.9995
        self.close = close
        self.symbol = symbol
        self.volume = 1_000_000.0


def _ts(i: int, *, step_minutes: int = 30) -> str:
    return (_START + timedelta(minutes=step_minutes * i)).isoformat()


def _entries(events: _Events) -> list[Any]:
    return [e for e in events.signals if e.signal_type in {"LONG", "SHORT"}]


def _feed(strategy: Any, symbol: str, prices: list[float]) -> None:
    for i, p in enumerate(prices):
        rng = p * 0.0005
        strategy.calculate_signals(_Market(_ts(i), symbol, p, high=p + rng, low=p - rng))


def _lcg(seed: int):
    state = seed
    while True:
        state = (1103515245 * state + 12345) % (2**31)
        yield (state / 2**31 - 0.5) * 2.0


def _strategy(bars: Any, events: _Events, **overrides: Any) -> TailIndexRegimeRiderStrategy:
    kwargs: dict[str, Any] = dict(
        tail_window=40,
        recent_window=12,
        k_short=4,
        k_long=10,
        fatten_ratio=0.85,
        thin_ratio=1.15,
        trend_lookback=10,
        min_trend_roc=0.0,
        trail_atr_mult=4.0,
        atr_period=5,
        max_adds=0,
        vol_window=10,
        target_vol=0.0,
        max_hold_bars=100000,
        allow_short=True,
        add_alloc_fraction=0.0,
    )
    kwargs.update(overrides)
    return TailIndexRegimeRiderStrategy(bars, events, **kwargs)


def _fattening_downtrend_prices(
    seed: int, *, n_history: int = 60, n_recent: int = 15
) -> list[float]:
    """Calm baseline, then a downtrend with occasional large loss shocks.

    The occasional big down shocks in the recent window pull the recent Hill
    estimate well below the fuller-window baseline -- a fattening loss tail --
    while the steady negative drift confirms the downtrend.
    """
    gen = _lcg(seed)
    p = 100.0
    prices = [p]
    for _ in range(n_history):
        r = next(gen) * 0.003
        p *= 1.0 + r
        prices.append(p)
    for i in range(n_recent):
        shock = -0.04 * abs(next(gen)) if i % 4 == 0 else 0.0
        r = -0.01 + shock + next(gen) * 0.001
        p *= 1.0 + r
        prices.append(p)
    return prices


def _thinning_uptrend_prices(seed: int, *, n_history: int = 60, n_recent: int = 15) -> list[float]:
    """Calm baseline, then a strong uptrend with small, uniform pullbacks.

    The uniform-magnitude pullbacks in the recent window sit close to the
    Hill threshold -- a thinning loss tail relative to the noisier baseline --
    while the outsized alternating gains confirm the uptrend.
    """
    gen = _lcg(seed)
    p = 100.0
    prices = [p]
    for _ in range(n_history):
        r = next(gen) * 0.006
        p *= 1.0 + r
        prices.append(p)
    for i in range(n_recent):
        if i % 2 == 0:
            r = -0.003 + next(gen) * 0.0002
        else:
            r = 0.016 + next(gen) * 0.0002
        p *= 1.0 + r
        prices.append(p)
    return prices


# --------------------------------------------------------------------------- #
# Regime entries
# --------------------------------------------------------------------------- #
def test_fattening_regime_with_downtrend_enters_short() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    strategy = _strategy(_BarsNoFeatures([symbol]), events)
    _feed(strategy, symbol, _fattening_downtrend_prices(101))
    entries = _entries(events)
    assert entries, "expected a SHORT entry once the recent loss tail fattens under a downtrend"
    assert entries[0].signal_type == "SHORT", entries[0].signal_type
    metadata = entries[0].metadata or {}
    ratio = metadata.get("tail_index_ratio")
    assert ratio is not None and ratio <= 0.85, metadata
    assert metadata.get("hill_short") is not None, metadata
    assert metadata.get("hill_base") is not None, metadata


def test_thinning_regime_with_uptrend_enters_long() -> None:
    symbol = "ETH/USDT"
    events = _Events()
    strategy = _strategy(_BarsNoFeatures([symbol]), events)
    _feed(strategy, symbol, _thinning_uptrend_prices(2))
    entries = _entries(events)
    assert entries, "expected a LONG entry once the recent loss tail thins under an uptrend"
    assert entries[0].signal_type == "LONG", entries[0].signal_type
    metadata = entries[0].metadata or {}
    ratio = metadata.get("tail_index_ratio")
    assert ratio is not None and ratio >= 1.15, metadata
    assert metadata.get("hill_short") is not None, metadata
    assert metadata.get("hill_base") is not None, metadata


def test_flat_and_insufficient_history_never_enter() -> None:
    symbol = "BNB/USDT"
    # Flat tape: every return is exactly zero -> an empty loss sample -> no entry.
    events_flat = _Events()
    strategy_flat = _strategy(_BarsNoFeatures([symbol]), events_flat)
    _feed(strategy_flat, symbol, [100.0] * 60)
    assert _entries(events_flat) == []

    # Too few bars to fill tail_window -> hill metrics unavailable -> no entry.
    events_short = _Events()
    strategy_short = _strategy(_BarsNoFeatures([symbol]), events_short)
    _feed(strategy_short, symbol, _thinning_uptrend_prices(2, n_history=5, n_recent=5))
    assert _entries(events_short) == []


# --------------------------------------------------------------------------- #
# None / empty-window / non-finite safety
# --------------------------------------------------------------------------- #
def test_none_and_empty_window_safe() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    strategy = _strategy(_BarsNoFeatures([symbol]), events)

    class _EmptyWindow:
        type = "MARKET_WINDOW"
        bars_1s: list[Any] = []

    empty = _EmptyWindow()
    empty.symbol = symbol
    strategy.calculate_signals(empty)  # must not raise
    strategy.calculate_signals(_Market(_ts(0), symbol, 0.0))  # degenerate close, no raise
    strategy.calculate_signals(_Market(_ts(1), symbol, float("nan")))  # non-finite close, no raise
    assert _entries(events) == []


def test_adversarial_set_state_never_raises() -> None:
    symbol = "BTC/USDT"
    strategy = _strategy(_BarsNoFeatures([symbol]), _Events())

    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state("not a dict")  # type: ignore[arg-type]
    strategy.set_state(12345)  # type: ignore[arg-type]
    strategy.set_state([])  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": "not a dict"})
    strategy.set_state({"tail_index_returns": "not a dict"})
    strategy.set_state({"tail_index_returns": {symbol: "not a list"}})
    strategy.set_state(
        {"tail_index_returns": {symbol: [float("nan"), float("inf"), "bad", None, 1.5]}}
    )
    strategy.set_state(
        {
            "symbol_state": {
                symbol: {
                    "closes": ["x", float("nan"), float("inf"), 12.5, None],
                    "mode": 999,
                    "entry_price": "abc",
                    "bars_held": "oops",
                    "last_time_key": 123,
                }
            },
            "tail_index_returns": {symbol: [1.0, "bad", float("nan"), -0.5, None]},
        }
    )
    # The strategy must still be usable after adversarial set_state calls.
    _feed(strategy, symbol, _thinning_uptrend_prices(2))


# --------------------------------------------------------------------------- #
# State roundtrip + determinism
# --------------------------------------------------------------------------- #
def test_state_roundtrip() -> None:
    symbol = "SOL/USDT"
    events = _Events()
    strategy = _strategy(_BarsNoFeatures([symbol]), events)
    _feed(strategy, symbol, _fattening_downtrend_prices(101))
    snapshot = strategy.get_state()
    restored = _strategy(_BarsNoFeatures([symbol]), _Events())
    restored.set_state(snapshot)
    again = restored.get_state()
    assert again["tail_index_returns"][symbol] == snapshot["tail_index_returns"][symbol]
    assert again["symbol_state"][symbol]["closes"] == snapshot["symbol_state"][symbol]["closes"]
    assert again["symbol_state"][symbol]["mode"] == snapshot["symbol_state"][symbol]["mode"]


def test_determinism_same_seed_same_signals() -> None:
    symbol = "XRP/USDT"
    prices = _fattening_downtrend_prices(101)

    events_a = _Events()
    _feed(_strategy(_BarsNoFeatures([symbol]), events_a), symbol, prices)

    events_b = _Events()
    _feed(_strategy(_BarsNoFeatures([symbol]), events_b), symbol, prices)

    signals_a = [(e.signal_type, e.metadata) for e in _entries(events_a)]
    signals_b = [(e.signal_type, e.metadata) for e in _entries(events_b)]
    assert signals_a == signals_b


# --------------------------------------------------------------------------- #
# Prepared candidate-slice table (not yet wired into candidate_library)
# --------------------------------------------------------------------------- #
def test_prepared_slice_has_exactly_one_variant_per_timeframe() -> None:
    assert set(_TAIL_INDEX_REGIME_RIDER_SLICE.keys()) == {"30m", "1h", "4h", "1d"}
    for timeframe, variants in _TAIL_INDEX_REGIME_RIDER_SLICE.items():
        assert len(variants) == 1, (timeframe, variants)
        spec = variants[0]
        assert spec["recent_window"] <= spec["tail_window"], (timeframe, spec)
        assert spec["k_short"] < spec["recent_window"], (timeframe, spec)
        assert spec["k_long"] < spec["tail_window"], (timeframe, spec)
