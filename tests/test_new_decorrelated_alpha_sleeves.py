from __future__ import annotations

import math

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
    restored.load_state(z_window.to_state()["values"])
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
