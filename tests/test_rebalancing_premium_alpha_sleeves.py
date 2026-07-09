"""Deterministic tests for RebalancingPremiumHarvestStrategy.

Direct class import only (no `@register` on this lane, so no registry/tier/
candidate-wiring assertions here -- those land with the W3 integration wave).

The load-bearing lane gate is the FORECAST-FREE BUILD GATE:

  (a-i)  the mechanical harvest -- at a rebalance the relative WINNER is
         TRIMMED (negative ``rebalance_trade``) and the relative LOSER is ADDED
         (positive trade) -- appears under BOTH a monthly and a weekly clock
         driven off the SAME price path;
  (a-ii) target weights depend only on trailing liquidity, never on a return
         forecast: two inputs identical up to time ``T`` but diverging after
         emit identical signals at/through ``T``.

Plus: drift-only between rebalances (trades occur only at clock boundaries);
run-twice determinism; get_state/set_state roundtrip + adversarial set_state
never-raise; never-raise on degenerate input; and the diversity-weighting rule
ranking by -log dollar-volume with a gross <= 1 target.

Any pseudo-randomness used to shape "neutral filler" symbols is drawn from a
small seeded linear-congruential generator (no ``random`` module), so every run
is bit-for-bit reproducible.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from collections.abc import Callable

from lumina_quant.strategies.rebalancing_premium_alpha_sleeves import (
    RebalancingPremiumHarvestStrategy,
)
from lumina_quant.tuning import HyperParam

# --------------------------------------------------------------------------- #
# LCG (deterministic, no `random` module)
# --------------------------------------------------------------------------- #


def _lcg_stream(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / float(0x7FFFFFFF)


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


def _market_event(
    symbol: str, iso_time: str, close: float, volume: float | None
) -> SimpleNamespace:
    return SimpleNamespace(
        type="MARKET",
        time=iso_time,
        symbol=symbol,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
    )


def _window_event(iso_time: str, quotes: dict[str, tuple[float, float | None]]) -> SimpleNamespace:
    """One MARKET_WINDOW carrying every symbol's bar simultaneously (clean cross-section)."""
    bars_1s: dict[str, list[dict[str, Any]]] = {}
    for symbol, (close, volume) in quotes.items():
        bars_1s[symbol] = [
            {
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": volume,
                "time": iso_time,
            }
        ]
    return SimpleNamespace(type="MARKET_WINDOW", time=iso_time, bars_1s=bars_1s)


# Four bars per month across four months -> monthly clock rebalances at the
# first bar of each month (bar indices 0, 4, 8, 12); weekly clock rebalances at
# roughly every bar (each date is about a week apart).
_MONTHS = (1, 2, 3, 4)
_DAYS = (2, 9, 16, 23)
_DATES: tuple[str, ...] = tuple(
    f"2026-{month:02d}-{day:02d}T00:00:00Z" for month in _MONTHS for day in _DAYS
)
_MONTHLY_BOUNDARIES = frozenset({0, 4, 8, 12})


def _month_of(iso_time: str) -> str:
    return iso_time[:7]


def _schedule(
    symbols: list[str],
    price_fn: Callable[[str, int], float],
    volume_fn: Callable[[str, int], float | None],
    dates: tuple[str, ...] = _DATES,
) -> list[tuple[str, dict[str, tuple[float, float | None]]]]:
    schedule: list[tuple[str, dict[str, tuple[float, float | None]]]] = []
    for idx, iso_time in enumerate(dates):
        quotes = {symbol: (price_fn(symbol, idx), volume_fn(symbol, idx)) for symbol in symbols}
        schedule.append((iso_time, quotes))
    return schedule


def _feed_windows(
    strategy: RebalancingPremiumHarvestStrategy,
    schedule: list[tuple[str, dict[str, tuple[float, float | None]]]],
) -> None:
    for iso_time, quotes in schedule:
        strategy.calculate_signals(_window_event(iso_time, quotes))


_COMMON_KWARGS: dict[str, Any] = dict(
    rebalance_period="monthly",
    weighting="equal",
    liquidity_window=4,
    min_symbols=4,
    target_gross_exposure=1.0,
    min_price=0.0,
)


def _longs(signals: list[Any]) -> list[Any]:
    return [sig for sig in signals if sig.signal_type == "LONG"]


def _non_exit(signals: list[Any]) -> list[Any]:
    return [sig for sig in signals if str(sig.signal_type).upper() != "EXIT"]


def _harvest_longs(signals: list[Any], symbol: str) -> list[Any]:
    """LONG rebalance signals for ``symbol`` that carry a mechanical trade (post-anchor)."""
    return [
        sig
        for sig in signals
        if sig.signal_type == "LONG"
        and sig.symbol == symbol
        and (sig.metadata or {}).get("rebalance_trade") is not None
    ]


# --------------------------------------------------------------------------- #
# (a-i) FORECAST-FREE BUILD GATE -- mechanical harvest under either clock
# --------------------------------------------------------------------------- #

_WINNER = "WIN/USDT"
_LOSER = "LOSE/USDT"
_CTRL = ("C0/USDT", "C1/USDT")
_HARVEST_SYMBOLS = [_WINNER, _LOSER, *_CTRL]


def _harvest_price(symbol: str, idx: int) -> float:
    if symbol == _WINNER:
        return 100.0 * (1.02**idx)  # relative winner: rises every bar
    if symbol == _LOSER:
        return 100.0 * (0.98**idx)  # relative loser: falls every bar
    return 100.0  # flat controls -> define the basket average


def _harvest_volume(_symbol: str, _idx: int) -> float:
    return 1000.0  # equal, constant -> equal-weight targets are exactly gross/n


def _run_harvest(period: str) -> list[Any]:
    schedule = _schedule(_HARVEST_SYMBOLS, _harvest_price, _harvest_volume)
    strategy = RebalancingPremiumHarvestStrategy(
        _Bars(_HARVEST_SYMBOLS),
        _Queue(),
        **dict(_COMMON_KWARGS, rebalance_period=period),
    )
    _feed_windows(strategy, schedule)
    return strategy.events.items


def test_forecast_free_mechanical_harvest_under_monthly_clock() -> None:
    signals = _run_harvest("monthly")
    winner = _harvest_longs(signals, _WINNER)
    loser = _harvest_longs(signals, _LOSER)
    assert winner, "expected at least one post-anchor rebalance for the winner"
    assert loser, "expected at least one post-anchor rebalance for the loser"
    # The mechanical harvest: the relative WINNER is trimmed (sell), the
    # relative LOSER is added (buy) -- at EVERY post-anchor rebalance.
    for sig in winner:
        assert sig.metadata["rebalance_trade"] < 0.0, sig.metadata
    for sig in loser:
        assert sig.metadata["rebalance_trade"] > 0.0, sig.metadata


def test_forecast_free_mechanical_harvest_under_weekly_clock() -> None:
    signals = _run_harvest("weekly")
    winner = _harvest_longs(signals, _WINNER)
    loser = _harvest_longs(signals, _LOSER)
    assert winner, "expected at least one post-anchor rebalance for the winner"
    assert loser, "expected at least one post-anchor rebalance for the loser"
    for sig in winner:
        assert sig.metadata["rebalance_trade"] < 0.0, sig.metadata
    for sig in loser:
        assert sig.metadata["rebalance_trade"] > 0.0, sig.metadata


def test_same_path_two_clocks_agree_on_harvest_sign() -> None:
    """Identical price path, two clocks -> same mechanical harvest direction."""
    monthly = _run_harvest("monthly")
    weekly = _run_harvest("weekly")
    for signals in (monthly, weekly):
        winner_last = _harvest_longs(signals, _WINNER)[-1]
        loser_last = _harvest_longs(signals, _LOSER)[-1]
        assert winner_last.metadata["rebalance_trade"] < 0.0
        assert loser_last.metadata["rebalance_trade"] > 0.0


# --------------------------------------------------------------------------- #
# (a-ii) target weights invariant to future-return manipulation
# --------------------------------------------------------------------------- #

_FF_SYMBOLS = ["S0/USDT", "S1/USDT", "S2/USDT", "S3/USDT"]
# Distinct constant volumes -> a non-trivial diversity weighting (so the test
# constrains WEIGHT VALUES, not just entry presence).
_FF_VOLUME = {"S0/USDT": 4000.0, "S1/USDT": 3000.0, "S2/USDT": 2000.0, "S3/USDT": 1000.0}


def _signal_tuple(sig: Any) -> tuple[str, str, float, dict[str, Any]]:
    return (sig.symbol, sig.signal_type, sig.strength, dict(sig.metadata or {}))


def test_target_weights_invariant_to_future_returns() -> None:
    split = 6  # bar index T -- includes rebalances at bars 0 and 4

    def _shared_price(symbol: str, idx: int) -> float:
        gen = _lcg_stream(seed=500 + _FF_SYMBOLS.index(symbol))
        price = 100.0
        for _ in range(idx + 1):
            price *= 1.0 + (next(gen) - 0.5) * 0.01
        return price

    def _price_a(symbol: str, idx: int) -> float:
        return _shared_price(symbol, idx)

    def _price_b(symbol: str, idx: int) -> float:
        base = _shared_price(symbol, idx)
        if idx <= split:
            return base  # identical up to T
        # Wildly diverging futures AFTER T must not change signals at/through T.
        return base * (5.0 if symbol == _FF_SYMBOLS[0] else 0.2)

    def _volume(symbol: str, _idx: int) -> float:
        return _FF_VOLUME[symbol]

    kwargs = dict(_COMMON_KWARGS, weighting="diversity", diversity_temperature=1.0)

    strat_a = RebalancingPremiumHarvestStrategy(_Bars(_FF_SYMBOLS), _Queue(), **kwargs)
    strat_b = RebalancingPremiumHarvestStrategy(_Bars(_FF_SYMBOLS), _Queue(), **kwargs)

    # Feed the shared prefix (bars 0..split) to both, snapshot the emitted
    # stream, then feed each its own divergent tail.
    prefix = _schedule(_FF_SYMBOLS, _price_a, _volume, dates=_DATES[: split + 1])
    _feed_windows(strat_a, prefix)
    _feed_windows(strat_b, prefix)
    snap_a = [_signal_tuple(sig) for sig in strat_a.events.items]
    snap_b = [_signal_tuple(sig) for sig in strat_b.events.items]

    tail_a = _schedule(_FF_SYMBOLS, _price_a, _volume, dates=_DATES[split + 1 :])
    tail_b = _schedule(_FF_SYMBOLS, _price_b, _volume, dates=_DATES[split + 1 :])
    _feed_windows(strat_a, tail_a)
    _feed_windows(strat_b, tail_b)

    # Signals emitted at/through T are identical (forecast-free / causal): the
    # divergent futures leave the <=T stream untouched.
    assert snap_a == snap_b
    assert [_signal_tuple(s) for s in strat_a.events.items][: len(snap_a)] == snap_a
    assert [_signal_tuple(s) for s in strat_b.events.items][: len(snap_b)] == snap_b
    # And the prefix actually exercised >=2 rebalances with non-uniform weights.
    prefix_longs = _longs(strat_a.events.items[: len(snap_a)])
    weights = {sig.metadata["target_weight"] for sig in prefix_longs}
    assert len(weights) >= 2, "diversity weighting should not be degenerate/uniform"


# --------------------------------------------------------------------------- #
# (b) rebalance-clock behavior: trades occur only at clock boundaries
# --------------------------------------------------------------------------- #


def test_trades_only_at_rebalance_boundaries() -> None:
    schedule = _schedule(_HARVEST_SYMBOLS, _harvest_price, _harvest_volume)
    strategy = RebalancingPremiumHarvestStrategy(
        _Bars(_HARVEST_SYMBOLS), _Queue(), **_COMMON_KWARGS
    )
    prev_len = 0
    grew_at: set[int] = set()
    for idx, (iso_time, quotes) in enumerate(schedule):
        strategy.calculate_signals(_window_event(iso_time, quotes))
        now = len(strategy.events.items)
        if now > prev_len:
            grew_at.add(idx)
        prev_len = now
    # Signals appear ONLY at monthly clock boundaries -- nothing in between
    # (drift-only holding).
    assert grew_at == _MONTHLY_BOUNDARIES, grew_at
    # Every boundary emits exactly one LONG per basket member.
    assert len(_longs(strategy.events.items)) == len(_MONTHLY_BOUNDARIES) * len(_HARVEST_SYMBOLS)


def test_boundaries_match_the_period_key() -> None:
    # The expected boundaries are exactly the bars that open a new month.
    expected = {0}
    for idx in range(1, len(_DATES)):
        if _month_of(_DATES[idx]) != _month_of(_DATES[idx - 1]):
            expected.add(idx)
    assert expected == _MONTHLY_BOUNDARIES


# --------------------------------------------------------------------------- #
# (c) determinism: two identical runs -> identical signal streams
# --------------------------------------------------------------------------- #


def test_determinism_two_runs_identical_signals() -> None:
    def _volume(symbol: str, idx: int) -> float:
        gen = _lcg_stream(seed=7000 + _FF_SYMBOLS.index(symbol))
        vol = 2000.0
        for _ in range(idx + 1):
            vol *= 1.0 + (next(gen) - 0.5) * 0.05
        return vol

    def _price(symbol: str, idx: int) -> float:
        gen = _lcg_stream(seed=idx * 17 + _FF_SYMBOLS.index(symbol))
        return 100.0 * (1.0 + (next(gen) - 0.5) * 0.2)

    kwargs = dict(_COMMON_KWARGS, weighting="diversity")

    def _run() -> list[tuple[str, str, float, dict[str, Any]]]:
        strategy = RebalancingPremiumHarvestStrategy(_Bars(_FF_SYMBOLS), _Queue(), **kwargs)
        _feed_windows(strategy, _schedule(_FF_SYMBOLS, _price, _volume))
        return [_signal_tuple(sig) for sig in strategy.events.items]

    first = _run()
    second = _run()
    assert first == second
    assert first, "expected at least one signal in this scenario"


# --------------------------------------------------------------------------- #
# (d) get_state / set_state roundtrip + adversarial set_state safety
# --------------------------------------------------------------------------- #


def test_state_roundtrip_lossless() -> None:
    schedule = _schedule(_HARVEST_SYMBOLS, _harvest_price, _harvest_volume)
    strategy = RebalancingPremiumHarvestStrategy(
        _Bars(_HARVEST_SYMBOLS), _Queue(), **_COMMON_KWARGS
    )
    _feed_windows(strategy, schedule)
    snapshot = strategy.get_state()

    restored = RebalancingPremiumHarvestStrategy(
        _Bars(_HARVEST_SYMBOLS), _Queue(), **_COMMON_KWARGS
    )
    restored.set_state(snapshot)
    again = restored.get_state()

    assert again == snapshot
    for symbol in _HARVEST_SYMBOLS:
        r = restored._state[symbol]
        o = strategy._state[symbol]
        assert list(r.closes) == list(o.closes)
        assert list(r.volumes) == list(o.volumes)
        assert r.mode == o.mode
        assert r.held_weight == o.held_weight
        assert r.anchor_price == o.anchor_price
        assert r.last_time_key == o.last_time_key
    assert restored._tick == strategy._tick
    assert restored._last_eval_time_key == strategy._last_eval_time_key
    assert restored._last_rebalance_key == strategy._last_rebalance_key


def test_adversarial_set_state_never_raises() -> None:
    symbols = ["A/USDT", "B/USDT", "C/USDT", "D/USDT", "E/USDT"]
    strategy = RebalancingPremiumHarvestStrategy(_Bars(symbols), _Queue(), **_COMMON_KWARGS)

    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state("not a dict")  # type: ignore[arg-type]
    strategy.set_state(12345)  # type: ignore[arg-type]
    strategy.set_state([])  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": "not a dict"})
    strategy.set_state({"symbol_state": {"A/USDT": "not a dict either"}})
    strategy.set_state({"symbol_state": {"A/USDT": {"closes": 12345}}})
    strategy.set_state({"symbol_state": {"A/USDT": {"closes": {"nested": "dict"}}}})
    strategy.set_state(
        {
            "last_eval_time_key": None,
            "last_rebalance_key": 999,
            "tick": "not-an-int",
            "symbol_state": {
                symbol: {
                    "closes": ["x", "y", float("nan"), float("inf"), 12.5, None],
                    "volumes": {"unexpected": "type"},
                    "mode": 999,
                    "held_weight": "abc",
                    "anchor_price": float("nan"),
                    "last_time_key": 123,
                }
                for symbol in symbols
            },
        }
    )
    for item in strategy._state.values():
        assert item.mode in {"OUT", "LONG"}
        assert item.held_weight is None or item.held_weight >= 0.0
        assert item.anchor_price is None or item.anchor_price > 0.0

    # Still functions normally afterward, using the SAME symbols it was built with.
    schedule = _schedule(symbols, _harvest_price, _harvest_volume)
    _feed_windows(strategy, schedule)  # must not raise


# --------------------------------------------------------------------------- #
# (e) never-raise on degenerate input
# --------------------------------------------------------------------------- #


def test_missing_volume_self_skips_and_never_raises() -> None:
    def _volume(_symbol: str, _idx: int) -> float | None:
        return None  # every bar arrives with volume=None -> dollar volume 0 -> excluded

    schedule = _schedule(_HARVEST_SYMBOLS, _harvest_price, _volume)
    strategy = RebalancingPremiumHarvestStrategy(
        _Bars(_HARVEST_SYMBOLS), _Queue(), **_COMMON_KWARGS
    )
    _feed_windows(strategy, schedule)  # must not raise
    # No positive dollar volume anywhere -> nothing is eligible -> no signals.
    assert _non_exit(strategy.events.items) == []
    assert all(v == 0.0 for v in strategy._state[_WINNER].volumes)


def test_degenerate_closes_never_raise() -> None:
    strategy = RebalancingPremiumHarvestStrategy(_Bars(["Z/USDT"]), _Queue(), **_COMMON_KWARGS)
    strategy.calculate_signals(_market_event("Z/USDT", _DATES[0], 0.0, 1000.0))
    strategy.calculate_signals(_market_event("Z/USDT", _DATES[1], -5.0, 1000.0))
    strategy.calculate_signals(_market_event("Z/USDT", _DATES[2], float("nan"), 1000.0))
    strategy.calculate_signals(_market_event("Z/USDT", _DATES[3], float("inf"), 1000.0))
    assert _non_exit(strategy.events.items) == []


def test_empty_and_unparseable_inputs_never_raise() -> None:
    strategy = RebalancingPremiumHarvestStrategy(
        _Bars(_HARVEST_SYMBOLS), _Queue(), **_COMMON_KWARGS
    )
    strategy.calculate_signals(
        SimpleNamespace(type="MARKET_WINDOW", symbol="Z/USDT", bars_1s={}, time="t0")
    )
    strategy.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol="ZZZ/USDT", close=None))
    # A parseable cross-section but with an UNPARSEABLE time never rebalances.
    strategy.calculate_signals(
        _window_event("not-a-timestamp", dict.fromkeys(_HARVEST_SYMBOLS, (100.0, 1000.0)))
    )
    assert strategy.events.items == []


def test_universe_below_min_symbols_self_skips() -> None:
    symbols = ["A/USDT", "B/USDT", "C/USDT"]
    kwargs = dict(_COMMON_KWARGS, min_symbols=5)
    strategy = RebalancingPremiumHarvestStrategy(_Bars(symbols), _Queue(), **kwargs)
    _feed_windows(strategy, _schedule(symbols, _harvest_price, _harvest_volume))
    assert _non_exit(strategy.events.items) == []


# --------------------------------------------------------------------------- #
# (f) diversity weighting ranks by -log dollar-volume; gross <= 1
# --------------------------------------------------------------------------- #


def test_diversity_weighting_ranks_by_dollar_volume_and_sums_to_gross() -> None:
    symbols = list(_FF_SYMBOLS)  # S0..S3 with strictly decreasing dollar volume

    def _price(_symbol: str, _idx: int) -> float:
        return 100.0  # flat -> dollar-volume rank is driven purely by volume

    def _volume(symbol: str, _idx: int) -> float:
        return _FF_VOLUME[symbol]

    gross = 0.9
    kwargs = dict(
        _COMMON_KWARGS,
        weighting="diversity",
        diversity_temperature=1.0,
        target_gross_exposure=gross,
    )
    strategy = RebalancingPremiumHarvestStrategy(_Bars(symbols), _Queue(), **kwargs)
    _feed_windows(strategy, _schedule(symbols, _price, _volume))

    # Read the FIRST rebalance (bar 0): entries, no mechanical trade yet.
    first_rebalance = [
        sig
        for sig in _longs(strategy.events.items)
        if (sig.metadata or {}).get("rebalance_key") == _month_of(_DATES[0])
    ]
    assert len(first_rebalance) == len(symbols)
    weight = {sig.symbol: sig.metadata["target_weight"] for sig in first_rebalance}
    rank = {sig.symbol: sig.metadata["rank"] for sig in first_rebalance}

    # Most liquid (largest dollar volume) -> rank 1 -> heaviest weight; strictly
    # monotone down the liquidity ladder.
    assert rank["S0/USDT"] == 1 and rank["S3/USDT"] == len(symbols)
    assert weight["S0/USDT"] > weight["S1/USDT"] > weight["S2/USDT"] > weight["S3/USDT"] > 0.0
    # Forecast-free entries carry no mechanical trade.
    assert all(sig.metadata["rebalance_trade"] is None for sig in first_rebalance)
    # Gross target is respected (weights sum to <= target_gross_exposure <= 1).
    assert sum(weight.values()) <= gross + 1e-9
    assert gross <= 1.0


# --------------------------------------------------------------------------- #
# schema sanity (not a registry/tier/candidate-wiring assertion)
# --------------------------------------------------------------------------- #


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = RebalancingPremiumHarvestStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "rebalance_period",
        "weighting",
        "diversity_temperature",
        "liquidity_window",
        "max_basket_size",
        "min_symbols",
        "target_gross_exposure",
        "min_price",
        "max_symbol_exposure_pct",
        "max_order_value",
    ):
        assert required in schema
    for cap in ("max_symbol_exposure_pct", "max_order_value"):
        assert schema[cap].tunable is False
