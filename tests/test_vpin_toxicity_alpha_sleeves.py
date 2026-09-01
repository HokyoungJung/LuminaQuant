"""Deterministic tests for the VPIN-lite microstructure flow-toxicity rider.

Covers: a tape of calm balanced noise followed by a one-sided heavy-volume
decline drives the toxicity percentile to the top of its own trailing
history, arming a SHORT in ``toxicity_mode="confirm"`` (informed-flow
continuation); the same confirm mode produces NO entry over a calm-only tape
(toxicity never crosses the high gate); ``toxicity_mode="avoid"`` mirrors
this by entering in the trend direction once the toxicity percentile is LOW
(a benign-flow filter), while ``"confirm"`` stays flat over that same calm
trending phase; BVC/VPIN never-raise behaviour (missing volume, zero volume,
degenerate/non-finite closes, empty window); adversarial ``set_state``
payloads never raise; get_state/set_state roundtrip; and bit-for-bit
determinism across two independent runs of the same seeded tape.  This
sleeve is intentionally NOT registered yet (see the deferred ``@register``
comment in the source module), so this file imports the class directly and
makes no registry/candidate assertions.  No backtest is run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from lumina_quant.strategies.vpin_toxicity_alpha_sleeves import (
    VpinToxicityRiderStrategy,
    _VPIN_TOXICITY_RIDER_SLICE,
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
        volume: float,
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


def _strategy(
    bars: Any, events: _Events, *, toxicity_mode: str = "confirm", **overrides: Any
) -> VpinToxicityRiderStrategy:
    kwargs: dict[str, Any] = dict(
        bucket_mult=1.0,
        n_buckets=10,
        vpin_history=20,
        toxicity_entry=0.85,
        toxicity_calm=0.15,
        toxicity_mode=toxicity_mode,
        trend_lookback=10,
        min_trend_roc=0.01,
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
    return VpinToxicityRiderStrategy(bars, events, **kwargs)


def _feed(strategy: Any, symbol: str, rows: list[tuple[float, float]], *, start: int = 0) -> None:
    for offset, (price, volume) in enumerate(rows):
        strategy.calculate_signals(_Market(_ts(start + offset), symbol, price, volume))


def _calm_then_decline_tape(
    seed: int,
    *,
    n_calm: int = 150,
    n_decline: int = 40,
    calm_volume: float = 1000.0,
    decline_volume: float = 6000.0,
) -> list[tuple[float, float]]:
    """Calm, roughly-balanced noise, then a heavy-volume one-sided decline.

    The calm phase has small symmetric price noise at ordinary volume, so BVC
    classifies roughly balanced buy/sell flow -> a moderate, stable VPIN.  The
    decline phase drops price sharply, every bar, at several times the calm
    volume: BVC classifies it almost entirely SELL (large negative price
    change relative to the still-calm trailing sigma), so the bucket
    imbalance swings strongly negative and VPIN (and its trailing percentile)
    rises toward the top of its own history.
    """
    gen = _lcg(seed)
    price = 100.0
    rows: list[tuple[float, float]] = []
    for _ in range(n_calm):
        r = next(gen) * 0.001
        price *= 1.0 + r
        rows.append((price, calm_volume))
    for _ in range(n_decline):
        r = -0.02 + next(gen) * 0.0005
        price *= 1.0 + r
        rows.append((price, decline_volume))
    return rows


def _calm_trend_tape(
    seed: int, *, n_warm: int = 100, n_trend: int = 60, volume: float = 1000.0
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """A noisy warm-up (builds a VPIN comparison history) then a calm uptrend.

    The warm-up has comparatively large, zero-drift noise (a varied VPIN
    history to rank against).  The trend phase has much smaller per-bar noise
    plus a small consistent positive drift: each bar's price change stays
    modest relative to the trailing sigma (so BVC stays closer to balanced,
    pulling VPIN -- and its trailing percentile -- DOWN), while the
    accumulated drift over ``trend_lookback`` bars is large enough to read as
    a clear uptrend.  Returned as ``(warm_rows, trend_rows)`` so callers can
    feed them separately and inspect only the trend-phase signals.
    """
    gen = _lcg(seed)
    price = 100.0
    warm_rows: list[tuple[float, float]] = []
    for _ in range(n_warm):
        r = next(gen) * 0.004
        price *= 1.0 + r
        warm_rows.append((price, volume))
    trend_rows: list[tuple[float, float]] = []
    for _ in range(n_trend):
        r = 0.0006 + next(gen) * 0.0008
        price *= 1.0 + r
        trend_rows.append((price, volume))
    return warm_rows, trend_rows


# --------------------------------------------------------------------------- #
# confirm mode: toxic flow + trend -> continuation entry
# --------------------------------------------------------------------------- #
def test_confirm_mode_enters_short_on_toxic_one_sided_decline() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    strategy = _strategy(_BarsNoFeatures([symbol]), events, toxicity_mode="confirm")
    _feed(strategy, symbol, _calm_then_decline_tape(101))
    entries = _entries(events)
    assert entries, "expected a SHORT entry once toxicity spikes under the declining trend"
    assert entries[0].signal_type == "SHORT", entries[0].signal_type
    metadata = entries[0].metadata or {}
    assert metadata.get("toxicity_percentile") is not None
    assert metadata["toxicity_percentile"] >= 0.85, metadata
    assert metadata.get("vpin") is not None
    assert metadata.get("toxicity_mode") == "confirm"


def test_confirm_mode_no_entry_when_toxicity_never_high() -> None:
    symbol = "ETH/USDT"
    events = _Events()
    strategy = _strategy(_BarsNoFeatures([symbol]), events, toxicity_mode="confirm")
    # Calm-only tape (no decline shock): toxicity never crosses the high gate.
    _feed(strategy, symbol, _calm_then_decline_tape(101, n_calm=150, n_decline=0))
    assert _entries(events) == []


# --------------------------------------------------------------------------- #
# avoid mode: benign flow + trend -> entry; confirm mode stays flat there
# --------------------------------------------------------------------------- #
def test_avoid_mode_enters_on_calm_trend_confirm_mode_does_not() -> None:
    symbol = "BNB/USDT"
    warm_rows, trend_rows = _calm_trend_tape(1)

    events_avoid = _Events()
    strategy_avoid = _strategy(_BarsNoFeatures([symbol]), events_avoid, toxicity_mode="avoid")
    _feed(strategy_avoid, symbol, warm_rows)
    baseline_avoid = len(events_avoid.signals)
    _feed(strategy_avoid, symbol, trend_rows, start=len(warm_rows))
    new_entries_avoid = [
        e for e in events_avoid.signals[baseline_avoid:] if e.signal_type in {"LONG", "SHORT"}
    ]
    assert new_entries_avoid, "expected an entry once flow is calm during the confirmed uptrend"
    assert new_entries_avoid[0].signal_type == "LONG", new_entries_avoid[0].signal_type
    metadata = new_entries_avoid[0].metadata or {}
    assert metadata.get("toxicity_percentile") is not None
    assert metadata["toxicity_percentile"] <= 0.15, metadata
    assert metadata.get("toxicity_mode") == "avoid"

    events_confirm = _Events()
    strategy_confirm = _strategy(_BarsNoFeatures([symbol]), events_confirm, toxicity_mode="confirm")
    _feed(strategy_confirm, symbol, warm_rows)
    baseline_confirm = len(events_confirm.signals)
    _feed(strategy_confirm, symbol, trend_rows, start=len(warm_rows))
    new_entries_confirm = [
        e for e in events_confirm.signals[baseline_confirm:] if e.signal_type in {"LONG", "SHORT"}
    ]
    assert new_entries_confirm == [], new_entries_confirm


# --------------------------------------------------------------------------- #
# None / empty-window / degenerate / missing-volume safety
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
    strategy.calculate_signals(_Market(_ts(0), symbol, 0.0, 1000.0))  # degenerate close
    strategy.calculate_signals(_Market(_ts(1), symbol, float("nan"), 1000.0))  # non-finite close
    strategy.calculate_signals(_Market(_ts(2), symbol, 100.0, 0.0))  # zero volume
    market_no_volume = _Market(_ts(3), symbol, 100.0, 0.0)
    market_no_volume.volume = None  # type: ignore[assignment]
    strategy.calculate_signals(market_no_volume)  # missing volume
    market_bad_volume = _Market(_ts(4), symbol, 100.0, 0.0)
    market_bad_volume.volume = float("nan")  # type: ignore[assignment]
    strategy.calculate_signals(market_bad_volume)  # non-finite volume
    assert _entries(events) == []


def test_missing_and_zero_volume_never_raises_across_a_run() -> None:
    symbol = "SOL/USDT"
    events = _Events()
    strategy = _strategy(_BarsNoFeatures([symbol]), events)
    price = 100.0
    for i in range(80):
        price *= 1.001
        volume = 0.0 if i % 3 == 0 else 1000.0
        market = _Market(_ts(i), symbol, price, volume)
        if i % 7 == 0:
            market.volume = None  # type: ignore[assignment]
        strategy.calculate_signals(market)  # must not raise regardless of volume shape


# --------------------------------------------------------------------------- #
# Adversarial set_state
# --------------------------------------------------------------------------- #
def test_adversarial_set_state_never_raises() -> None:
    symbol = "BTC/USDT"
    strategy = _strategy(_BarsNoFeatures([symbol]), _Events())

    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state("not a dict")  # type: ignore[arg-type]
    strategy.set_state(12345)  # type: ignore[arg-type]
    strategy.set_state([])  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": "not a dict"})
    strategy.set_state({"vpin_state": "not a dict"})
    strategy.set_state({"vpin_state": {symbol: "not a dict"}})
    strategy.set_state(
        {
            "vpin_state": {
                symbol: {
                    "bucket_imbalances": 12345,
                    "vpin_history": "not iterable either",
                    "volumes": None,
                    "bucket_buy": "abc",
                    "bucket_sell": float("nan"),
                    "last_vpin": "bad",
                    "last_percentile": object(),
                }
            }
        }
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
            "vpin_state": {
                symbol: {
                    "bucket_imbalances": [float("nan"), float("inf"), "bad", None, 0.3],
                    "vpin_history": [1, 2, "bad", None],
                    "volumes": [float("nan"), -1.0, "bad"],
                    "bucket_buy": float("inf"),
                    "bucket_sell": None,
                    "last_vpin": float("nan"),
                    "last_percentile": float("inf"),
                }
            },
        }
    )
    # The strategy must still be usable after adversarial set_state calls.
    _feed(strategy, symbol, _calm_then_decline_tape(101))


# --------------------------------------------------------------------------- #
# State roundtrip + determinism
# --------------------------------------------------------------------------- #
def test_state_roundtrip() -> None:
    symbol = "XRP/USDT"
    events = _Events()
    strategy = _strategy(_BarsNoFeatures([symbol]), events)
    _feed(strategy, symbol, _calm_then_decline_tape(101))
    snapshot = strategy.get_state()

    restored = _strategy(_BarsNoFeatures([symbol]), _Events())
    restored.set_state(snapshot)
    again = restored.get_state()

    assert again["vpin_state"][symbol] == snapshot["vpin_state"][symbol]
    assert again["symbol_state"][symbol]["closes"] == snapshot["symbol_state"][symbol]["closes"]
    assert again["symbol_state"][symbol]["mode"] == snapshot["symbol_state"][symbol]["mode"]


def test_determinism_same_seed_same_signals() -> None:
    symbol = "ADA/USDT"
    rows = _calm_then_decline_tape(101)

    events_a = _Events()
    _feed(_strategy(_BarsNoFeatures([symbol]), events_a), symbol, rows)

    events_b = _Events()
    _feed(_strategy(_BarsNoFeatures([symbol]), events_b), symbol, rows)

    signals_a = [(e.signal_type, e.metadata) for e in _entries(events_a)]
    signals_b = [(e.signal_type, e.metadata) for e in _entries(events_b)]
    assert signals_a == signals_b
    assert signals_a, "expected at least one entry to compare"


# --------------------------------------------------------------------------- #
# Prepared candidate-slice table (not yet wired into candidate_library)
# --------------------------------------------------------------------------- #
def test_prepared_slice_has_exactly_one_variant_per_timeframe() -> None:
    assert set(_VPIN_TOXICITY_RIDER_SLICE.keys()) == {"30m", "1h", "4h", "1d"}
    for timeframe, variants in _VPIN_TOXICITY_RIDER_SLICE.items():
        assert len(variants) == 1, (timeframe, variants)
        spec = variants[0]
        assert spec["n_buckets"] >= 3, (timeframe, spec)
        assert 0.0 <= spec["toxicity_calm"] < spec["toxicity_entry"] <= 1.0, (timeframe, spec)
        assert spec["toxicity_mode"] in {"confirm", "avoid"}, (timeframe, spec)
