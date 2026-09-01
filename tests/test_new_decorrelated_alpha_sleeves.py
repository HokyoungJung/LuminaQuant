from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from lumina_quant.indicators import (
    RollingZScoreWindow,
    ewma_volatility,
    hurst_exponent,
    rolling_quantile,
    ts_regression_intercept,
    ts_regression_rsquared,
    ts_regression_slope,
)
from lumina_quant.indicators.rolling_stats import recursive_least_squares_beta_update
from lumina_quant.strategies.registry import (
    get_default_strategy_params,
    get_strategy_names,
    get_strategy_param_schema,
)
from lumina_quant.strategy_factory import build_binance_futures_candidates
from lumina_quant.strategies.cross_sectional_anomaly_alpha_sleeves import (
    IdiosyncraticVolatilityStrategy,
    LotterySkewnessStrategy,
)


NEW_ALPHA_STRATEGIES = (
    "HurstRegimeGatedStrategy",
    "ConfidenceGatedTrendStrategy",
    "MetalsRelativeValueBasketStrategy",
    "LiquidationCascadeReversionStrategy",
    "OrderBookImbalanceReversionStrategy",
    "CrossSectionalEquityMomentumStrategy",
    "ResidualEquityMomentumStrategy",
    "BettingAgainstBetaStrategy",
    "SemisLeadLagRotationStrategy",
    "DualMomentumIndexRotationStrategy",
    "CalendarSeasonalityOverlayStrategy",
)


class _CrossBars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = symbols


class _CrossEvents:
    def put(self, item: object) -> None:
        _ = item


def _cross_window(time: str | None, closes: dict[str, float]) -> SimpleNamespace:
    return SimpleNamespace(
        type="MARKET_WINDOW",
        time=time,
        bars_1s={
            symbol: [{"time": time, "close": close, "volume": 1.0}]
            for symbol, close in closes.items()
        },
    )


def test_cross_section_windows_reject_skew_and_duplicate_rows_atomically():
    symbols = ["BTC/USDT", "ETH/USDT"]
    strategy = IdiosyncraticVolatilityStrategy(
        _CrossBars(symbols), _CrossEvents(), benchmark_symbol="BTC/USDT", min_symbols=2
    )
    skewed = SimpleNamespace(
        type="MARKET_WINDOW",
        time="2026-08-20T00:00:00+00:00",
        bars_1s={
            "BTC/USDT": [{"time": "2026-08-20T00:00:00+00:00", "close": 100.0}],
            "ETH/USDT": [{"time": "2026-08-20T00:01:00+00:00", "close": 100.0}],
        },
    )
    strategy.calculate_signals(skewed)
    assert all(not item.closes for item in strategy._state.values())

    duplicate = SimpleNamespace(
        type="MARKET_WINDOW",
        time="2026-08-20T00:00:00+00:00",
        bars_1s={
            "BTC/USDT": [
                {"time": "2026-08-20T00:00:00+00:00", "close": 100.0},
                {"time": "2026-08-20T00:00:00+00:00", "close": 101.0},
            ],
            "ETH/USDT": [{"time": "2026-08-20T00:00:00+00:00", "close": 100.0}],
        },
    )
    strategy.calculate_signals(duplicate)
    assert all(not item.closes for item in strategy._state.values())

    aligned = SimpleNamespace(
        type="MARKET_WINDOW",
        time="2026-08-20T00:00:00+00:00",
        bars_1s={
            symbol: [{"time": "2026-08-20T00:00:00+00:00", "close": 100.0}] for symbol in symbols
        },
    )
    strategy.calculate_signals(aligned)
    strategy.calculate_signals(aligned)
    assert all(len(item.closes) == 1 for item in strategy._state.values())


def test_lottery_max_uses_completed_timestamp_grouped_daily_returns():
    strategy = LotterySkewnessStrategy(
        _CrossBars(["A", "B"]), _CrossEvents(), skew_window=3, max_window=2, min_symbols=2
    )
    closes = [95.0, 100.0, 110.0, 115.5, 1_000.0]
    keys = [
        "2026-08-18T00:00:00+00:00",
        "2026-08-18T12:00:00+00:00",
        "2026-08-19T00:00:00+00:00",
        "2026-08-20T00:00:00+00:00",
        "2026-08-21T00:00:00+00:00",
    ]
    scored = strategy._lottery_score(closes, keys)
    assert scored is not None
    assert scored[1]["max_daily_return"] == pytest.approx(0.10)


def test_cross_section_history_requires_new_complete_order_independent_windows() -> None:
    symbols = ["BTC/USDT", "ETH/USDT"]
    strategy = IdiosyncraticVolatilityStrategy(
        _CrossBars(symbols), _CrossEvents(), benchmark_symbol="BTC/USDT", min_symbols=2
    )
    current = _cross_window("2026-08-20T00:00:00+00:00", {"BTC/USDT": 100.0, "ETH/USDT": 101.0})
    strategy.calculate_signals(current)
    baseline = strategy.get_state()

    # Older, unkeyed, and partial windows cannot move the panel or evaluation
    # clock. A single-symbol callback cannot fill the rejected partial panel.
    strategy.calculate_signals(
        _cross_window("2026-08-19T00:00:00+00:00", {"BTC/USDT": 99.0, "ETH/USDT": 100.0})
    )
    strategy.calculate_signals(_cross_window(None, {"BTC/USDT": 99.0, "ETH/USDT": 100.0}))
    strategy.calculate_signals(_cross_window("2026-08-21T00:00:00+00:00", {"BTC/USDT": 102.0}))
    strategy.calculate_signals(
        SimpleNamespace(
            type="MARKET",
            symbol="ETH/USDT",
            time="2026-08-21T00:00:00+00:00",
            close=1_000.0,
            volume=1.0,
        )
    )
    assert strategy.get_state() == baseline

    # Mapping insertion order is irrelevant once every configured name has one
    # exact-time row; the whole panel commits at once.
    strategy.calculate_signals(
        _cross_window("2026-08-21T00:00:00+00:00", {"ETH/USDT": 102.0, "BTC/USDT": 101.0})
    )
    assert strategy._last_eval_time_key > baseline["last_eval_time_key"]
    assert {len(item.closes) for item in strategy._state.values()} == {2}


def test_lottery_daily_state_roundtrip_continues_without_open_day_max() -> None:
    symbols = ["A", "B"]
    original = LotterySkewnessStrategy(
        _CrossBars(symbols), _CrossEvents(), skew_window=3, max_window=2, min_symbols=2
    )
    for day, close in enumerate((100.0, 110.0, 115.5, 120.0), start=18):
        original.calculate_signals(
            _cross_window(
                f"2026-08-{day}T12:00:00+00:00",
                {"A": close, "B": close * 1.01},
            )
        )
    state = original.get_state()
    assert json.loads(json.dumps(state)) == state
    restored = LotterySkewnessStrategy(
        _CrossBars(symbols), _CrossEvents(), skew_window=3, max_window=2, min_symbols=2
    )
    restored.set_state(state)
    assert restored.get_state() == state

    continuation = _cross_window("2026-08-22T12:00:00+00:00", {"A": 1_000.0, "B": 1_010.0})
    original.calculate_signals(continuation)
    restored.calculate_signals(continuation)
    assert restored.get_state() == original.get_state()
    assert original._lottery_score(
        list(original._state["A"].closes),
        list(original._close_times["A"]),
        list(original._completed_daily_closes["A"]),
    )[1]["max_daily_return"] == pytest.approx(0.05)


def test_idiosyncratic_vol_rejects_sparse_unmatched_benchmark_panel():
    symbols = ["BTC/USDT", "A", "B"]
    strategy = IdiosyncraticVolatilityStrategy(
        _CrossBars(symbols),
        _CrossEvents(),
        benchmark_symbol="BTC/USDT",
        beta_window=4,
        vol_window=2,
        rebalance_bars=1,
        min_symbols=2,
    )
    for symbol in symbols:
        strategy._state[symbol].closes.extend([100.0, 101.0, 99.0, 102.0, 103.0])
    strategy._close_times["BTC/USDT"].extend(
        [f"2026-08-{day:02d}T00:00:00+00:00" for day in range(1, 6)]
    )
    strategy._close_times["A"].extend(
        [f"2026-08-{day:02d}T00:00:00+00:00" for day in (1, 2, 4, 5, 6)]
    )
    strategy._close_times["B"].extend(
        [f"2026-08-{day:02d}T00:00:00+00:00" for day in (1, 2, 3, 5, 6)]
    )
    strategy._tick = 1
    strategy._rebalance("2026-08-06T00:00:00+00:00")
    assert all(item.mode == "OUT" for item in strategy._state.values())


def test_new_rolling_stats_primitives_handle_core_math_and_guards():
    x_values = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_values = [1.0, 3.0, 5.0, 7.0, 9.0]

    assert ts_regression_slope(x_values, y_values) == pytest.approx(2.0)
    assert ts_regression_intercept(x_values, y_values) == pytest.approx(1.0)
    assert ts_regression_rsquared(x_values, y_values) == pytest.approx(1.0)
    assert ts_regression_slope([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None

    assert rolling_quantile([10.0, 30.0, 20.0, 40.0], window=4, q=0.25) == pytest.approx(17.5)
    assert rolling_quantile([1.0, 2.0], window=2, q=1.5) is None

    returns = [0.01, -0.02, 0.03]
    decay = math.exp(-math.log(2.0) / 2.0)
    ewma_var = returns[0] * returns[0]
    for value in returns[1:]:
        ewma_var = decay * ewma_var + (1.0 - decay) * value * value
    assert ewma_volatility(returns, half_life=2, annualization=4) == pytest.approx(
        math.sqrt(ewma_var) * 2.0
    )
    assert ewma_volatility([float("nan")], half_life=2) is None

    hurst_series = [sum(1.0 if idx % 5 else -0.5 for idx in range(n)) for n in range(1, 160)]
    hurst = hurst_exponent(hurst_series, min_lag=2, max_lag=12)
    assert hurst is not None
    assert 0.0 <= hurst <= 1.0
    assert hurst_exponent([1.0, 1.0, 1.0], min_lag=2, max_lag=4) is None

    updated = recursive_least_squares_beta_update(
        0.5,
        1.0,
        x_value=4.0,
        y_value=2.0,
        forgetting_factor=0.9,
    )
    assert updated is not None
    beta, covariance = updated
    assert beta == pytest.approx(1.7244897959183672)
    assert covariance == pytest.approx(0.2040816326530613)
    assert recursive_least_squares_beta_update(1.0, 1.0, x_value=1.0, y_value=0.0) is None

    z_window = RollingZScoreWindow(3)
    assert z_window.zscore(3.0) is None
    for value in (1.0, 2.0, 3.0):
        z_window.append(value)
    assert z_window.zscore(4.0) == pytest.approx((4.0 - 2.0) / math.sqrt(2.0 / 3.0))

    restored = RollingZScoreWindow(3)
    restored.load_state(z_window.to_state())
    assert restored.to_state() == z_window.to_state()


def test_new_plugin_strategy_schemas_register_after_lazy_discovery():
    discovered = set(get_strategy_names())

    for strategy_name in NEW_ALPHA_STRATEGIES:
        assert strategy_name in discovered
        schema = get_strategy_param_schema(strategy_name)
        assert schema, strategy_name
        defaults = get_default_strategy_params(strategy_name)
        assert defaults, strategy_name
        assert set(defaults).issubset(schema)


def test_candidate_library_wires_new_alpha_sleeves_with_admission_safe_tags(monkeypatch):
    from lumina_quant.strategy_factory import candidate_library

    monkeypatch.setattr(candidate_library, "_has_perp_support_data", lambda: True)
    rows = build_binance_futures_candidates(
        timeframes=["15m", "30m", "1h", "4h", "1d"],
        symbols=[
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "TRX/USDT",
            "XAU/USDT",
            "XAG/USDT",
            "XPT/USDT",
            "XPD/USDT",
            "SPY/USDT",
            "QQQ/USDT",
            "EWY/USDT",
            "EWJ/USDT",
            "EWT/USDT",
            "IWM/USDT",
            "SOXL/USDT",
            "TSLA/USDT",
            "INTC/USDT",
            "HOOD/USDT",
            "MSTR/USDT",
            "NVDA/USDT",
            "AMD/USDT",
            "AVGO/USDT",
            "MU/USDT",
        ],
    )
    counts = {
        strategy_name: sum(1 for row in rows if row.strategy_class == strategy_name)
        for strategy_name in NEW_ALPHA_STRATEGIES
    }

    assert all(count > 0 for count in counts.values()), counts

    required_multi_tags = {"cross_sectional", "carry", "momentum"}
    for row in rows:
        if row.strategy_class not in NEW_ALPHA_STRATEGIES or len(row.symbols) < 3:
            continue
        assert row.family == "cross_sectional"
        assert required_multi_tags.issubset(row.tags)
        assert row.metadata["timeframe"] == row.timeframe
