from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lumina_quant.core.strategy_input import StrategyInputContext
from lumina_quant.indicators.common import time_key
from lumina_quant.strategies.aggressive_return_alpha_sleeves import FundingHarvestCarryStrategy
from lumina_quant.strategies.alpha_max_research_sleeves import (
    CANONICAL_ALPHA_MAX_COMPONENT_NODES,
    ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy,
    ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy,
    ResearchOnlyFourHourFundingHarvestCarryStrategy,
)
from lumina_quant.strategies.low_turnover_trend_alpha_sleeves import (
    LowTurnoverTrendPersistenceStrategy,
)
from lumina_quant.strategies.near_high_anchoring_alpha_sleeves import (
    CrossSectionalNearHighAnchoringStrategy,
)
from lumina_quant.strategies.registry import (
    get_default_strategy_params,
    get_strategy_tier,
    resolve_strategy_class,
)


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


class _Agg:
    def __init__(self) -> None:
        self.data: dict[tuple[str, str], list[Any]] = {}

    def set_bars(self, symbol: str, timeframe: str, bars: list[Any]) -> None:
        self.data[(symbol, timeframe)] = list(bars)

    def get_bars(self, symbol: str, timeframe: str, lookback_bars: int = 1, *, n=None):
        bars = self.data.get((symbol, timeframe), [])
        count = lookback_bars if n is None else n
        return bars[-int(count) :]


def _dt(day: int = 1, hour: int = 0) -> datetime:
    return datetime(2026, 1, day, hour, tzinfo=UTC)


def _bar(t: datetime, close: float, *, high: float | None = None, low: float | None = None):
    high = close if high is None else high
    low = close if low is None else low
    return (t, close, high, low, close, 1000.0)


def _event_for(
    symbol_to_bar: dict[str, Any],
    when: datetime | None = None,
    *,
    completed_native_bars: bool = True,
):
    first = next(iter(symbol_to_bar.values()))
    return SimpleNamespace(
        type="MARKET_WINDOW",
        time=when or first[0],
        completed_native_bars=completed_native_bars,
        bars_1s={symbol: (bar,) for symbol, bar in symbol_to_bar.items()},
    )


def _json_state(strategy: Any) -> str:
    return json.dumps(strategy.get_research_indicator_state(), sort_keys=True, default=str)


def test_adapters_have_exact_research_only_rows_and_native_timeframes() -> None:
    node_path = Path(".omx/plans/alpha-max-current-trial-nodes-v1.json")
    payload = json.loads(node_path.read_text())
    current_nodes = {
        node["implementation"]: node
        for node in payload["nodes"]
        if str(node.get("implementation", "")).startswith("ResearchOnly")
    }
    expected = {
        "ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy": (
            ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy,
            LowTurnoverTrendPersistenceStrategy,
            "1d",
        ),
        "ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy": (
            ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy,
            CrossSectionalNearHighAnchoringStrategy,
            "1d",
        ),
        "ResearchOnlyFourHourFundingHarvestCarryStrategy": (
            ResearchOnlyFourHourFundingHarvestCarryStrategy,
            FundingHarvestCarryStrategy,
            "4h",
        ),
    }
    for name, (cls, original, timeframe) in expected.items():
        assert issubclass(cls, original)
        assert resolve_strategy_class(name) is cls
        assert get_strategy_tier(name) == "research_only"
        assert CANONICAL_ALPHA_MAX_COMPONENT_NODES[name]["params"] == current_nodes[name]["params"]
        assert CANONICAL_ALPHA_MAX_COMPONENT_NODES[name]["candidate_symbols"] == tuple(
            current_nodes[name]["symbols"]
        )
        assert get_default_strategy_params(name) == current_nodes[name]["params"]
        strategy = cls(_Bars(["BTCUSDT"]), _Queue())
        assert strategy.required_native_timeframes == (timeframe,)
        assert strategy.uses_timeframe_aggregator is True
        assert (
            cls.canonical_component_node()["candidate_symbols"]
            == CANONICAL_ALPHA_MAX_COMPONENT_NODES[name]["candidate_symbols"]
        )

    assert get_strategy_tier("LowTurnoverTrendPersistenceStrategy") == "research_only"
    assert get_strategy_tier("CrossSectionalNearHighAnchoringStrategy") == "research_only"


def test_daily_trend_adapter_matches_original_and_excludes_forming_duplicate_ticks() -> None:
    symbol = "BTCUSDT"
    original_events = _Queue()
    adapter_events = _Queue()
    original = LowTurnoverTrendPersistenceStrategy(_Bars([symbol]), original_events)
    adapter = ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy(_Bars([symbol]), adapter_events)
    completed = _bar(_dt(1), 100.0, high=101.0, low=99.0)
    forming_poison = _bar(_dt(2), 1_000_000.0)

    original.calculate_signals_window(_event_for({symbol: completed}))
    agg = _Agg()
    agg.set_bars(symbol, "1d", [completed, forming_poison])
    adapter.calculate_signals_window(
        SimpleNamespace(
            type="MARKET_WINDOW",
            time=_dt(2),
            event_time_watermark_ms=int(_dt(2).timestamp() * 1000),
        ),
        agg,
    )
    first_state = adapter.get_research_indicator_state()
    adapter.calculate_signals_window(
        SimpleNamespace(
            type="MARKET_WINDOW",
            time=_dt(2),
            event_time_watermark_ms=int(_dt(2).timestamp() * 1000),
        ),
        agg,
    )
    adapter.calculate_signals_window(
        SimpleNamespace(type="MARKET_WINDOW", time=_dt(1, 4), bars_1s={})
    )

    assert adapter.get_research_indicator_state() == first_state
    assert (
        first_state["symbol_state"][symbol]["closes"]
        == original.get_state()["symbol_state"][symbol]["closes"]
    )
    assert first_state["symbol_state"][symbol]["closes"] == [100.0]
    assert first_state["completed_native_count_by_symbol"] == {symbol: 1}


def test_near_high_atomic_barrier_sorted_batch_duplicate_and_missing_fail_closed() -> None:
    symbols = ["SOLUSDT", "ADAUSDT", "BTCUSDT"]
    params = {"min_symbols": 2, "min_history_bars": 1, "rebalance_bars": 1, "quantile_pct": 0.34}
    adapter = ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy(
        _Bars(symbols), _Queue(), admitted_symbols=symbols, **params
    )
    bars = {
        symbol: _bar(_dt(1), 100.0 + idx, high=110.0 + idx) for idx, symbol in enumerate(symbols)
    }

    for symbol in symbols[:-1]:
        adapter.calculate_signals_window(_event_for({symbol: bars[symbol]}))
        assert adapter._tick == 0
    adapter.calculate_signals_window(_event_for({symbols[-1]: bars[symbols[-1]]}))
    assert adapter._tick == 1
    assert tuple(adapter.symbol_list) == tuple(sorted(symbols))
    state_after = adapter.get_research_indicator_state()
    adapter.calculate_signals_window(_event_for({symbols[0]: bars[symbols[0]]}))
    assert adapter.get_research_indicator_state() == state_after

    # A conflicting duplicate after the atomic batch has closed is still hostile,
    # not silently idempotent.
    with pytest.raises(ValueError, match="conflicting_near_high_duplicate"):
        adapter.calculate_signals_window(_event_for({symbols[0]: _bar(_dt(1), 999.0)}))

    raw_like = ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy(
        _Bars(symbols), _Queue(), admitted_symbols=symbols, **params
    )
    raw_like.calculate_signals_window(
        _event_for({"ADAUSDT": _bar(_dt(5), 100.0)}, completed_native_bars=False)
    )
    assert raw_like._tick == 0

    immediate_missing = ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy(
        _Bars(symbols), _Queue(), admitted_symbols=symbols, **params
    )
    with pytest.raises(ValueError, match="incomplete_near_high_cross_section"):
        immediate_missing.calculate_signals_window(
            _event_for(
                {"ADAUSDT": _bar(_dt(6), 100.0)},
                when=_dt(7),
                completed_native_bars=False,
            )
        )

    conflict = _bar(_dt(2), 100.0)
    adapter.calculate_signals_window(_event_for({"ADAUSDT": conflict}))
    with pytest.raises(ValueError, match="conflicting_near_high_duplicate"):
        adapter.calculate_signals_window(_event_for({"ADAUSDT": _bar(_dt(2), 101.0)}))

    missing = ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy(
        _Bars(symbols), _Queue(), admitted_symbols=symbols, **params
    )
    missing.calculate_signals_window(_event_for({"ADAUSDT": _bar(_dt(3), 100.0)}))
    with pytest.raises(ValueError, match="incomplete_near_high_cross_section"):
        missing.finalize_completed_native_buckets(_dt(4))


def test_near_high_arrival_permutations_are_byte_identical() -> None:
    symbols = ["ADAUSDT", "BTCUSDT", "ETHUSDT"]
    params = {"min_symbols": 2, "min_history_bars": 1, "rebalance_bars": 1, "quantile_pct": 0.34}
    bars = {
        symbol: _bar(_dt(1), 100.0 + idx, high=110.0 + idx) for idx, symbol in enumerate(symbols)
    }
    states = set()
    for order in permutations(symbols):
        strategy = ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy(
            _Bars(symbols), _Queue(), admitted_symbols=symbols, **params
        )
        for symbol in order:
            strategy.calculate_signals_window(_event_for({symbol: bars[symbol]}))
            strategy.calculate_signals_window(_event_for({symbol: bars[symbol]}))
        states.add(_json_state(strategy))
    assert len(states) == 1


def test_near_high_full_state_matches_monolithic_across_partial_barrier_chunk() -> None:
    symbols = ["ADAUSDT", "BTCUSDT"]
    params = {
        "admitted_symbols": symbols,
        "min_symbols": 2,
        "min_history_bars": 2,
        "rebalance_bars": 1,
        "vol_window": 2,
    }
    day_1 = {
        "ADAUSDT": _bar(_dt(1), 100.0, high=105.0),
        "BTCUSDT": _bar(_dt(1), 110.0, high=115.0),
    }
    day_2 = {
        "ADAUSDT": _bar(_dt(2), 102.0, high=106.0),
        "BTCUSDT": _bar(_dt(2), 108.0, high=116.0),
    }

    monolithic = ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy(
        _Bars(symbols), _Queue(), **params
    )
    monolithic.calculate_signals_window(_event_for(day_1))
    monolithic.calculate_signals_window(_event_for({"ADAUSDT": day_2["ADAUSDT"]}))
    monolithic.calculate_signals_window(_event_for({"ADAUSDT": day_2["ADAUSDT"]}))
    monolithic.calculate_signals_window(_event_for({"BTCUSDT": day_2["BTCUSDT"]}))

    first_chunk = ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy(
        _Bars(symbols), _Queue(), **params
    )
    first_chunk.calculate_signals_window(_event_for(day_1))
    first_chunk.calculate_signals_window(_event_for({"ADAUSDT": day_2["ADAUSDT"]}))
    first_chunk._alpha_max_bound_aggregator = object()
    snapshot = first_chunk.get_state()
    chunk_state = snapshot["_alpha_max_chunk_state"]
    assert "bound_aggregator" not in chunk_state
    assert set(chunk_state["barrier_pending"][time_key(_dt(2))]) == {"ADAUSDT"}

    resumed = ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy(
        _Bars(symbols), _Queue(), **params
    )
    resumed.set_state(snapshot)
    assert resumed._alpha_max_bound_aggregator is None
    assert resumed.get_state() == snapshot

    # Replayed overlap is idempotent, then the remaining symbol closes exactly
    # one cross-section barrier.
    resumed.calculate_signals_window(_event_for({"ADAUSDT": day_2["ADAUSDT"]}))
    resumed.calculate_signals_window(_event_for({"BTCUSDT": day_2["BTCUSDT"]}))

    assert resumed.get_state() == monolithic.get_state()
    assert resumed.get_research_indicator_state() == monolithic.get_research_indicator_state()
    assert resumed._alpha_max_completed_native_count_by_symbol == {
        "ADAUSDT": 2,
        "BTCUSDT": 2,
    }


def test_near_high_full_state_preserves_failed_duplicate_and_partial_error() -> None:
    symbols = ["ADAUSDT", "BTCUSDT"]
    params = {
        "admitted_symbols": symbols,
        "min_symbols": 2,
        "min_history_bars": 1,
    }
    original = _bar(_dt(1), 100.0)
    failed = ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy(
        _Bars(symbols), _Queue(), **params
    )
    failed.calculate_signals_window(_event_for({"ADAUSDT": original}))
    with pytest.raises(ValueError, match="conflicting_near_high_duplicate"):
        failed.calculate_signals_window(_event_for({"ADAUSDT": _bar(_dt(1), 101.0)}))

    failed_snapshot = failed.get_state()
    failed_resumed = ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy(
        _Bars(symbols), _Queue(), **params
    )
    failed_resumed.set_state(failed_snapshot)
    assert failed_resumed.get_state() == failed_snapshot
    with pytest.raises(ValueError, match="conflicting_near_high_duplicate"):
        failed_resumed.calculate_signals_window(_event_for({"ADAUSDT": original}))

    partial = ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy(
        _Bars(symbols), _Queue(), **params
    )
    partial_agg = _Agg()
    partial_agg.set_bars("ADAUSDT", "1d", [_bar(_dt(2), 100.0)])
    partial_agg.set_bars("BTCUSDT", "1d", [_bar(_dt(2), 100.0)])
    partial._alpha_max_bound_aggregator = partial_agg
    assert partial.finalize_completed_native_buckets(_dt(2, 12)) == 0
    partial_snapshot = partial.get_state()

    partial_resumed = ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy(
        _Bars(symbols), _Queue(), **params
    )
    partial_resumed.set_state(partial_snapshot)
    assert partial_resumed._alpha_max_bound_aggregator is None
    assert partial_resumed.get_state() == partial_snapshot
    with pytest.raises(ValueError, match="partial_native_bucket"):
        partial_resumed.get_research_indicator_state()

    invalid = deepcopy(partial_snapshot)
    invalid["_alpha_max_chunk_state"]["completed_native_count_by_symbol"] = {"ADAUSDT": True}
    before = partial_resumed.get_state()
    with pytest.raises(ValueError, match="invalid_alpha_max_near_high_chunk_state"):
        partial_resumed.set_state(invalid)
    assert partial_resumed.get_state() == before


class _Lookup:
    def __init__(self, points: dict[str, tuple[float, int | datetime] | None]) -> None:
        self.points = points
        self.calls: list[tuple[str, str, int]] = []

    def get_latest_point(self, symbol: str, field: str, *, timestamp_ms: int):
        self.calls.append((symbol, field, timestamp_ms))
        return self.points.get(symbol)


def test_four_hour_carry_uses_bar_close_asof_funding_and_rejects_poison() -> None:
    symbol = "BTCUSDT"
    start = _dt(1, 0)
    close_ms = int((start + timedelta(hours=4)).timestamp() * 1000)
    bar = _bar(start, 100.0)
    forming_poison = _bar(start + timedelta(hours=4), 1_000_000.0)
    agg = _Agg()
    agg.set_bars(symbol, "4h", [bar, forming_poison])

    strategy = ResearchOnlyFourHourFundingHarvestCarryStrategy(
        _Bars([symbol]), _Queue(), funding_window=1, no_fight_roc_period=1
    )
    lookup = _Lookup({symbol: (0.0, close_ms)})
    strategy.calculate_signals_context(
        StrategyInputContext(
            event=SimpleNamespace(type="MARKET_WINDOW", time=start + timedelta(hours=4)),
            aggregator=agg,
            feature_lookup=lookup,
        )
    )
    assert lookup.calls == [(symbol, "funding_rate", close_ms)]
    assert strategy.get_research_indicator_state()["funding"][symbol] == [0.0]
    assert strategy.get_research_indicator_state()["symbol_state"][symbol]["closes"] == [100.0]

    future = ResearchOnlyFourHourFundingHarvestCarryStrategy(
        _Bars([symbol]), _Queue(), funding_window=1
    )
    future_agg = _Agg()
    future_agg.set_bars(symbol, "4h", [bar, forming_poison])
    with pytest.raises(ValueError, match="future_funding_point"):
        future.calculate_signals_context(
            StrategyInputContext(
                event=SimpleNamespace(type="MARKET_WINDOW", time=start + timedelta(hours=4)),
                aggregator=future_agg,
                feature_lookup=_Lookup({symbol: (0.1, close_ms + 1)}),
            )
        )

    stale = ResearchOnlyFourHourFundingHarvestCarryStrategy(
        _Bars([symbol]), _Queue(), funding_window=1
    )
    stale_agg = _Agg()
    stale_agg.set_bars(symbol, "4h", [bar, forming_poison])
    with pytest.raises(ValueError, match="stale_funding_point"):
        stale.calculate_signals_context(
            StrategyInputContext(
                event=SimpleNamespace(type="MARKET_WINDOW", time=start + timedelta(hours=4)),
                aggregator=stale_agg,
                feature_lookup=_Lookup({symbol: (0.1, close_ms - 8 * 60 * 60 * 1000 - 1)}),
            )
        )

    ambient = ResearchOnlyFourHourFundingHarvestCarryStrategy(
        _Bars([symbol]), _Queue(), funding_window=1
    )
    ambient_agg = _Agg()
    ambient_agg.set_bars(symbol, "4h", [bar, forming_poison])
    with pytest.raises(ValueError, match="ambient_feature_lookup_forbidden"):
        ambient.calculate_signals_context(
            StrategyInputContext(
                event=SimpleNamespace(type="MARKET_WINDOW", time=start + timedelta(hours=4)),
                aggregator=ambient_agg,
                feature_lookup=None,
            )
        )


@pytest.mark.parametrize(
    ("cls", "symbols"),
    [
        (ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy, ["BTCUSDT"]),
        (ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy, ["ADAUSDT", "BTCUSDT"]),
        (ResearchOnlyFourHourFundingHarvestCarryStrategy, ["BTCUSDT"]),
    ],
)
def test_indicator_capsules_reset_economic_state_and_minimum_history_gate(
    cls: Any, symbols: list[str]
) -> None:
    params = (
        {"admitted_symbols": symbols, "min_symbols": 2, "min_history_bars": 1}
        if "NearHigh" in cls.__name__
        else {}
    )
    strategy = cls(_Bars(symbols), _Queue(), **params)
    with pytest.raises(ValueError, match="insufficient_research_warmup_history"):
        strategy.validate_research_warmup_ready()
    strategy.minimum_completed_bars = 0
    for symbol in symbols:
        state_obj = strategy._state[symbol]
        state_obj.mode = "LONG"
        state_obj.entry_price = 123.0
        if hasattr(state_obj, "bars_held"):
            state_obj.bars_held = 9
        if hasattr(state_obj, "adds"):
            state_obj.adds = 2
            state_obj.stop_price = 1.0
    strategy.validate_research_warmup_ready()
    capsule = strategy.get_research_indicator_state()
    for payload in capsule["symbol_state"].values():
        assert payload.get("mode") == "OUT"
        assert payload.get("entry_price") is None
        assert payload.get("bars_held") == 0
        if "adds" in payload:
            assert payload["adds"] == 0
            assert payload["stop_price"] is None


def test_daily_trend_capsule_restores_recent_times_for_identical_vol_sizing() -> None:
    symbol = "BTCUSDT"
    continuous = ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy(
        _Bars([symbol]), _Queue(), vol_window=2
    )
    for day, close in enumerate((100.0, 110.0, 90.0), start=1):
        continuous.calculate_signals_window(_event_for({symbol: _bar(_dt(day), close)}))

    capsule = continuous.get_research_indicator_state()
    restored = ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy(
        _Bars([symbol]), _Queue(), vol_window=2
    )
    restored.set_research_indicator_state(capsule)

    next_event = _event_for({symbol: _bar(_dt(4), 105.0)})
    continuous.calculate_signals_window(next_event)
    restored.calculate_signals_window(next_event)

    continuous_closes = list(continuous._state[symbol].closes)
    restored_closes = list(restored._state[symbol].closes)
    assert capsule["recent_times"] == list(continuous._recent_times)[:-1]
    assert list(restored._recent_times) == list(continuous._recent_times)
    assert restored_closes == continuous_closes
    assert restored._vol_scaled_allocation(restored_closes, 1.0) == pytest.approx(
        continuous._vol_scaled_allocation(continuous_closes, 1.0)
    )


def test_daily_near_high_capsule_restores_recent_times_for_identical_vol_sizing() -> None:
    symbols = ["ADAUSDT", "BTCUSDT"]
    params = {
        "admitted_symbols": symbols,
        "high_lookback_bars": 3,
        "min_history_bars": 2,
        "min_symbols": 2,
        "quantile_pct": 0.5,
        "rebalance_bars": 1,
        "vol_window": 2,
    }
    continuous = ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy(
        _Bars(symbols), _Queue(), **params
    )
    ada_closes = (100.0, 120.0, 90.0, 130.0, 95.0)
    btc_closes = (100.0, 80.0, 110.0, 70.0, 115.0)
    for day, (ada_close, btc_close) in enumerate(zip(ada_closes, btc_closes), start=1):
        continuous.calculate_signals_window(
            _event_for(
                {
                    "ADAUSDT": _bar(_dt(day), ada_close, high=ada_close + 5.0),
                    "BTCUSDT": _bar(_dt(day), btc_close, high=max(120.0, btc_close + 5.0)),
                }
            )
        )

    capsule = continuous.get_research_indicator_state()
    restored = ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy(
        _Bars(symbols), _Queue(), **params
    )
    restored.set_research_indicator_state(capsule)

    next_event = _event_for(
        {
            "ADAUSDT": _bar(_dt(6), 140.0, high=145.0),
            "BTCUSDT": _bar(_dt(6), 60.0, high=125.0),
        }
    )
    continuous.calculate_signals_window(next_event)
    restored.calculate_signals_window(next_event)

    continuous_targets, continuous_vols = continuous._score_and_select()
    restored_targets, restored_vols = restored._score_and_select()
    continuous_weights, continuous_scalar = continuous._inverse_vol_weights(
        continuous_targets, continuous_vols
    )
    restored_weights, restored_scalar = restored._inverse_vol_weights(
        restored_targets, restored_vols
    )
    assert capsule["recent_times"] == list(continuous._recent_times)[:-1]
    assert list(restored._recent_times) == list(continuous._recent_times)
    assert restored_targets == continuous_targets
    assert restored_weights == pytest.approx(continuous_weights)
    assert restored_scalar == pytest.approx(continuous_scalar)
    assert continuous_scalar < 1.0


def test_boundary_finalization_matches_natural_promotion_and_partial_rejects() -> None:
    symbol = "BTCUSDT"
    completed = _bar(_dt(1), 100.0)
    next_bar = _bar(_dt(2), 101.0)

    natural = ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy(_Bars([symbol]), _Queue())
    natural.calculate_signals_window(_event_for({symbol: completed}))

    explicit = ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy(_Bars([symbol]), _Queue())
    agg = _Agg()
    agg.set_bars(symbol, "1d", [completed])
    explicit._alpha_max_bound_aggregator = agg
    assert explicit.finalize_completed_native_buckets(_dt(2)) == 1
    assert (
        explicit.get_research_indicator_state()["symbol_state"]
        == natural.get_research_indicator_state()["symbol_state"]
    )
    assert explicit.finalize_completed_native_buckets(_dt(2)) == 0

    partial = ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy(_Bars([symbol]), _Queue())
    partial_agg = _Agg()
    partial_agg.set_bars(symbol, "1d", [next_bar])
    partial._alpha_max_bound_aggregator = partial_agg
    assert partial.finalize_completed_native_buckets(_dt(2, 12)) == 0
    with pytest.raises(ValueError, match="partial_native_bucket"):
        partial.get_research_indicator_state()
