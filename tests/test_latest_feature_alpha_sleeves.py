"""Regression coverage for stale-safe CUSUM/VR and synchronized skew sleeves."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lumina_quant.indicators.variance_ratio import variance_ratio
from lumina_quant.strategies.cusum_varratio_alpha_sleeves import (
    CusumChangePointTrendRiderStrategy,
    VarianceRatioTrendRiderStrategy,
)
from lumina_quant.strategies.skew_innovation_alpha_sleeves import (
    IdiosyncraticSkewInnovationStrategy,
)
from lumina_quant.strategies.external_alpha_sleeves import _Snapshot
from lumina_quant.strategies.oi_growth_pressure_alpha_sleeves import (
    OpenInterestGrowthPressureStrategy,
    _DayRecord,
)
from lumina_quant.strategies.xs_residual_taker_flow_alpha_sleeves import (
    CrossSectionalResidualTakerFlowStrategy,
)


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = symbols


class _Events:
    def put(self, event: object) -> None:
        pass


def _cusum() -> CusumChangePointTrendRiderStrategy:
    return CusumChangePointTrendRiderStrategy(
        _Bars(["BTC/USDT"]), _Events(), cusum_vol_window=4, target_vol=0.0
    )


def _vr() -> VarianceRatioTrendRiderStrategy:
    return VarianceRatioTrendRiderStrategy(
        _Bars(["BTC/USDT"]),
        _Events(),
        vr_window=8,
        vr_k=2,
        vr_threshold=1.96,
        target_vol=0.0,
    )


def test_cusum_does_not_advance_on_repeated_or_old_feature_point() -> None:
    strategy = _cusum()
    item = strategy._state["BTC/USDT"]
    item.closes.append(100.0)
    item.last_time_key = "2026-01-02T00:00:00+00:00"
    snapshot = type("Snapshot", (), {"time": "2026-01-02T00:00:00+00:00", "close": 101.0})()
    strategy._process_symbol("BTC/USDT", snapshot)
    snapshot.time = "2026-01-01T23:30:00+00:00"
    strategy._process_symbol("BTC/USDT", snapshot)
    assert list(strategy._returns["BTC/USDT"]) == []
    assert list(item.closes) == [100.0]


def test_variance_ratio_uses_complete_fixed_lag_sample() -> None:
    strategy = _vr()
    returns = [0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.01, 0.02]
    strategy._returns["BTC/USDT"].extend(returns)
    statistic = strategy._vr("BTC/USDT")
    assert statistic is not None
    assert statistic[0] == variance_ratio(returns, 2, unbiased=True)


def test_variance_ratio_null_series_fails_closed() -> None:
    strategy = _vr()
    strategy._returns["BTC/USDT"].extend([0.0] * 8)
    assert strategy._vr("BTC/USDT") is None


def _skew_strategy() -> IdiosyncraticSkewInnovationStrategy:
    return IdiosyncraticSkewInnovationStrategy(
        _Bars(["BTC/USDT", "ETH/USDT"]),
        _Events(),
        beta_window=6,
        skew_window=3,
        min_history_bars=7,
        min_symbols=2,
    )


def test_residual_skew_uses_only_synchronized_asset_benchmark_returns() -> None:
    strategy = _skew_strategy()
    times = [f"2026-01-{day:02d}T00:00:00+00:00" for day in range(1, 16)]
    bench = [
        100.0,
        102.0,
        99.0,
        103.0,
        101.0,
        105.0,
        102.0,
        106.0,
        104.0,
        109.0,
        105.0,
        111.0,
        108.0,
        114.0,
        110.0,
    ]
    asset = [
        50.0,
        52.0,
        48.0,
        54.0,
        49.0,
        56.0,
        50.0,
        58.0,
        53.0,
        61.0,
        55.0,
        65.0,
        57.0,
        69.0,
        59.0,
    ]
    residuals = strategy._beta_hedged_residuals(asset, times, bench, times)
    assert residuals is not None
    assert len(residuals) == 2 * strategy.skew_window
    # A perfectly levered benchmark has no idiosyncratic residual variance.
    assert strategy.delta_skew_for([2.0 * value for value in bench], bench, times, times) is None


def test_skew_scores_only_common_timestamp_cross_section() -> None:
    strategy = _skew_strategy()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    benchmark = strategy._state["BTC/USDT"]
    asset = strategy._state["ETH/USDT"]
    bench_prices = [
        100.0,
        102.0,
        99.0,
        103.0,
        101.0,
        105.0,
        102.0,
        106.0,
        104.0,
        109.0,
        105.0,
        111.0,
        108.0,
        114.0,
    ]
    asset_prices = [
        50.0,
        52.0,
        48.0,
        54.0,
        49.0,
        56.0,
        50.0,
        58.0,
        53.0,
        61.0,
        55.0,
        65.0,
        57.0,
    ]
    for index, price in enumerate(bench_prices):
        key = (start + timedelta(days=index)).isoformat()
        benchmark.times.append(key)
        benchmark.closes.append(price)
    for index, price in enumerate(asset_prices):
        # All asset observations are common; the benchmark's final extra point
        # must not be paired with a stale asset close.
        key = (start + timedelta(days=index)).isoformat()
        asset.times.append(key)
        asset.closes.append(price)
    assert "ETH/USDT" not in strategy._score_symbols()


def test_skew_abstains_on_an_interior_gap_in_the_common_timestamp_grid() -> None:
    strategy = _skew_strategy()
    times = [
        "2026-01-01T00:00:00+00:00",
        "2026-01-02T00:00:00+00:00",
        "2026-01-04T00:00:00+00:00",
        "2026-01-05T00:00:00+00:00",
        "2026-01-06T00:00:00+00:00",
        "2026-01-07T00:00:00+00:00",
        "2026-01-08T00:00:00+00:00",
    ]
    for item, prices in (
        (strategy._state["BTC/USDT"], [100.0, 102.0, 99.0, 103.0, 101.0, 105.0, 102.0]),
        (strategy._state["ETH/USDT"], [50.0, 52.0, 48.0, 54.0, 49.0, 56.0, 50.0]),
    ):
        item.times.extend(times)
        item.closes.extend(prices)
    assert strategy._score_symbols() == {}


def _window_event(time: str, bars: dict[str, list[dict[str, object]]]) -> object:
    return type("Window", (), {"type": "MARKET_WINDOW", "time": time, "bars_1s": bars})()


def _skew_window(time: str, btc: float = 100.0, eth: float = 50.0) -> object:
    return _window_event(
        time,
        {
            "BTC/USDT": [{"time": time, "close": btc}],
            "ETH/USDT": [{"time": time, "close": eth}],
        },
    )


def test_skew_callbacks_require_one_complete_strictly_new_exact_time_panel() -> None:
    strategy = _skew_strategy()
    time_1 = "2026-01-05T00:00:00+00:00"
    initial = strategy.get_state()

    # MARKET callbacks and partial, skewed, and unkeyed windows cannot advance
    # either constituent or the weekly decision clock.
    strategy.calculate_signals(
        type(
            "Market", (), {"type": "MARKET", "symbol": "BTC/USDT", "time": time_1, "close": 100.0}
        )()
    )
    strategy.calculate_signals_window(_window_event(time_1, {"BTC/USDT": [{"close": 100.0}]}))
    strategy.calculate_signals_window(
        _window_event(
            time_1,
            {
                "BTC/USDT": [{"time": time_1, "close": 100.0}],
                "ETH/USDT": [{"time": "2026-01-05T01:00:00+00:00", "close": 50.0}],
            },
        )
    )
    strategy.calculate_signals_window(_skew_window("", 100.0, 50.0))
    assert strategy.get_state() == initial

    strategy.calculate_signals_window(_skew_window(time_1))
    committed = strategy.get_state()
    strategy.calculate_signals_window(_skew_window(time_1, 101.0, 51.0))
    strategy.calculate_signals_window(_skew_window("2026-01-04T00:00:00+00:00", 99.0, 49.0))
    assert strategy.get_state() == committed


def test_skew_preflight_requires_one_raw_row_at_the_event_in_utc() -> None:
    strategy = _skew_strategy()
    event_time = "2026-01-05T00:00:00Z"
    strategy.calculate_signals_window(
        _window_event(
            event_time,
            {
                "BTC/USDT": [("2026-01-04T19:00:00-05:00", 100.0, 100.0, 100.0, 100.0)],
                "ETH/USDT": [("2026-01-05T00:00:00+00:00", 50.0, 50.0, 50.0, 50.0)],
            },
        )
    )
    assert list(strategy._state["BTC/USDT"].times) == ["2026-01-05T00:00:00+00:00"]
    committed = strategy.get_state()

    for bars in (
        {
            "BTC/USDT": [
                {"time": event_time, "close": 101.0},
                {"time": event_time, "close": 102.0},
            ],
            "ETH/USDT": [{"time": event_time, "close": 51.0}],
        },
        {
            "BTC/USDT": [{"time": "not-a-time", "close": 101.0}],
            "ETH/USDT": [{"time": event_time, "close": 51.0}],
        },
        {
            "BTC/USDT": [{"time": "2026-01-04T23:59:59+00:00", "close": 101.0}],
            "ETH/USDT": [{"time": event_time, "close": 51.0}],
        },
    ):
        strategy.calculate_signals_window(_window_event(event_time, bars))
    assert strategy.get_state() == committed


def test_skew_beta_requires_the_full_preregistered_horizon() -> None:
    strategy = IdiosyncraticSkewInnovationStrategy(
        _Bars(["BTC/USDT", "ETH/USDT"]),
        _Events(),
        beta_window=10,
        skew_window=3,
        min_history_bars=7,
        min_symbols=2,
    )
    # Seven returns fund both three-bar skew windows but not the ten-return
    # preregistered beta estimate; the old expanding-beta path scored this.
    times = [f"2026-02-{day:02d}T00:00:00+00:00" for day in range(1, 9)]
    prices = [100.0 + (index % 3) for index in range(len(times))]
    assert strategy._beta_hedged_residuals(prices, times, prices, times) is None


def test_skew_minimum_history_is_exact_maximum_required_horizon() -> None:
    strategy = IdiosyncraticSkewInnovationStrategy(
        _Bars(["BTC/USDT", "ETH/USDT"]),
        _Events(),
        beta_window=10,
        skew_window=3,
        vol_window=2,
        min_history_bars=7,
        min_symbols=2,
    )
    # The schema clamps skew_window to six, so the exact resolved maximum is
    # max(beta=10, 2*skew=12, vol=2, configured=7) + one close.
    assert strategy.min_history_bars == 13


def test_skew_tied_delta_cross_section_is_not_ranked() -> None:
    strategy = IdiosyncraticSkewInnovationStrategy(
        _Bars(["BTC/USDT", "ETH/USDT", "SOL/USDT"]),
        _Events(),
        beta_window=6,
        skew_window=3,
        vol_window=2,
        min_history_bars=7,
        min_symbols=2,
    )
    times = [f"2026-01-{day:02d}T00:00:00+00:00" for day in range(1, 9)]
    benchmark = [100.0, 102.0, 99.0, 103.0, 101.0, 105.0, 102.0, 106.0]
    asset = [50.0, 52.0, 48.0, 54.0, 49.0, 56.0, 50.0, 58.0]
    for symbol, prices in (
        ("BTC/USDT", benchmark),
        ("ETH/USDT", asset),
        ("SOL/USDT", asset),
    ):
        strategy._state[symbol].times.extend(times)
        strategy._state[symbol].closes.extend(prices)
    assert strategy._score_symbols() == {}


def test_skew_state_restoration_preserves_complete_panel_continuation() -> None:
    strategy = _skew_strategy()
    for day in range(1, 4):
        time = f"2026-01-{day:02d}T00:00:00+00:00"
        strategy.calculate_signals_window(_skew_window(time, 100.0 + day, 50.0 + day))
    restored = _skew_strategy()
    restored.set_state(strategy.get_state())
    for day in range(4, 8):
        time = f"2026-01-{day:02d}T00:00:00+00:00"
        event = _skew_window(time, 100.0 + day, 50.0 + day)
        strategy.calculate_signals_window(event)
        restored.calculate_signals_window(event)
    assert restored.get_state() == strategy.get_state()


def test_skew_checkpoint_rejects_partial_noncommon_and_irregular_clocks_atomically() -> None:
    strategy = _skew_strategy()
    for day in range(1, 4):
        strategy.calculate_signals_window(_skew_window(f"2026-01-{day:02d}T00:00:00+00:00"))
    before = strategy.get_state()

    partial = strategy.get_state()
    partial["symbol_state"].pop("ETH/USDT")
    strategy.set_state(partial)
    assert strategy.get_state() == before

    noncommon = strategy.get_state()
    noncommon["symbol_state"]["ETH/USDT"]["times"][-1] = "2026-01-04T00:00:00+00:00"
    noncommon["symbol_state"]["ETH/USDT"]["last_bar_key"] = "2026-01-04T00:00:00+00:00"
    strategy.set_state(noncommon)
    assert strategy.get_state() == before

    irregular = strategy.get_state()
    for symbol in irregular["symbol_state"]:
        irregular["symbol_state"][symbol]["times"][1] = "2026-01-02T12:00:00+00:00"
    strategy.set_state(irregular)
    assert strategy.get_state() == irregular
    assert strategy._score_symbols() == {}


def test_variance_ratio_rejects_unkeyed_duplicate_and_older_bars_before_mutation() -> None:
    strategy = _vr()
    item = strategy._state["BTC/USDT"]
    item.closes.append(100.0)
    item.last_time_key = "2026-01-02T00:00:00+00:00"
    before = strategy.get_state()
    for time in ("", "2026-01-02T00:00:00+00:00", "2026-01-01T23:30:00+00:00"):
        strategy._process_symbol(
            "BTC/USDT",
            _Snapshot(time=time, open=101.0, high=101.0, low=101.0, close=101.0, volume=1.0),
        )
    assert strategy.get_state() == before


def test_variance_ratio_state_restoration_preserves_continuation() -> None:
    strategy = _vr()
    for minute in range(1, 5):
        strategy._process_symbol(
            "BTC/USDT",
            _Snapshot(
                time=f"2026-01-01T00:{minute:02d}:00+00:00",
                open=100.0 + minute,
                high=100.0 + minute,
                low=100.0 + minute,
                close=100.0 + minute,
                volume=1.0,
            ),
        )
    restored = _vr()
    restored.set_state(strategy.get_state())
    for minute in range(5, 9):
        snapshot = _Snapshot(
            time=f"2026-01-01T00:{minute:02d}:00+00:00",
            open=100.0 + minute,
            high=100.0 + minute,
            low=100.0 + minute,
            close=100.0 + minute,
            volume=1.0,
        )
        strategy._process_symbol("BTC/USDT", snapshot)
        restored._process_symbol("BTC/USDT", snapshot)
    assert restored.get_state() == strategy.get_state()


def _oi_strategy() -> OpenInterestGrowthPressureStrategy:
    return OpenInterestGrowthPressureStrategy(
        _Bars(["A", "B", "C"]),
        _Events(),
        min_symbols=3,
        min_history_days=8,
        min_oi_coverage=1.0,
        vol_window=2,
    )


def _set_oi_days(
    strategy: OpenInterestGrowthPressureStrategy, symbol: str, oi_step: float, volume: float
) -> None:
    item = strategy._state[symbol]
    start = datetime(2026, 1, 1, tzinfo=UTC).date()
    for offset in range(8):
        item.days.append(
            _DayRecord(
                day=(start + timedelta(days=offset)).isoformat(),
                oi_notional=100.0 + oi_step * offset,
                dollar_volume=volume,
                close=100.0 + offset * (1.0 + oi_step / 10.0),
            )
        )
    item.closes.extend([100.0, 101.0, 99.0, 102.0])
    item.last_committed_day = "2026-01-08"
    strategy._last_committed_day = "2026-01-08"


def test_oi_growth_uses_complete_utc_seven_day_horizon_on_irregular_cadence() -> None:
    strategy = _oi_strategy()
    _set_oi_days(strategy, "A", 1.0, 100.0)
    _set_oi_days(strategy, "B", 4.0, 100.0)
    _set_oi_days(strategy, "C", 2.0, 100.0)
    scores, _, metadata = strategy._residual_scores()
    assert set(scores) == {"A", "B", "C"}
    assert metadata["A"]["delta_oi_norm"] == 7.0 / 700.0


def test_oi_growth_fails_closed_when_notional_denominator_is_missing() -> None:
    strategy = _oi_strategy()
    _set_oi_days(strategy, "A", 1.0, 0.0)
    _set_oi_days(strategy, "B", 4.0, 0.0)
    _set_oi_days(strategy, "C", 2.0, 0.0)
    assert strategy._residual_scores() == ({}, {}, {})


def _flow_strategy() -> CrossSectionalResidualTakerFlowStrategy:
    return CrossSectionalResidualTakerFlowStrategy(
        _Bars(["A", "B", "C"]), _Events(), min_symbols=3, formation_window_bars=2, vol_window=2
    )


def _set_flow_state(
    strategy: CrossSectionalResidualTakerFlowStrategy, symbol: str, flow: float, key: str
) -> None:
    item = strategy._state[symbol]
    item.closes.extend([100.0, 101.0, 99.0])
    item.net_flows.extend([1.0, flow])
    item.turnovers.extend([2.0, 10.0])
    item.last_time_key = key


def test_residual_taker_flow_removes_equal_weight_common_current_shock() -> None:
    strategy = _flow_strategy()
    key = "2026-01-05T00:00:00+00:00"
    for symbol in strategy.symbol_list:
        _set_flow_state(strategy, symbol, 5.0, key)
    assert strategy._residual_scores() == ({}, {}, {})


def test_residual_taker_flow_requires_one_common_utc_timestamp() -> None:
    strategy = _flow_strategy()
    _set_flow_state(strategy, "A", 2.0, "2026-01-05T00:00:00+00:00")
    _set_flow_state(strategy, "B", 4.0, "2026-01-05T00:00:01+00:00")
    _set_flow_state(strategy, "C", 6.0, "2026-01-05T00:00:00+00:00")
    assert strategy._residual_scores() == ({}, {}, {})


def test_residual_taker_flow_min_hold_advances_once_per_utc_week() -> None:
    strategy = _flow_strategy()
    item = strategy._state["A"]
    item.mode = "LONG"
    strategy._evaluate("2026-01-05T00:00:00+00:00")
    strategy._evaluate("2026-01-05T12:00:00+00:00")
    assert item.decisions_held == 1
    strategy._evaluate("2026-01-12T00:00:00+00:00")
    assert item.decisions_held == 2
