from __future__ import annotations

from lumina_quant.backtesting.cost_models import CostModelParams
from lumina_quant.portfolio.cost_aware_constructor import (
    ConstructorParams,
    CostAwarePortfolioConstructor,
)


def test_penalty_effect_reduces_turnover():
    target_weights = {"BTC/USDT": 0.6, "ETH/USDT": -0.4}
    current_weights = {"BTC/USDT": 0.0, "ETH/USDT": 0.0}
    prices = {"BTC/USDT": 100.0, "ETH/USDT": 50.0}
    liquidity = {
        "BTC/USDT": {"sigma": 0.03, "adtv": 2_000_000.0, "volume": 50_000.0},
        "ETH/USDT": {"sigma": 0.04, "adtv": 1_500_000.0, "volume": 40_000.0},
    }
    cost_params = CostModelParams(spread_bps=2.0, impact_k=1.0, fees_bps=1.0)

    baseline = CostAwarePortfolioConstructor(ConstructorParams(no_trade_band_bps=0.0, turnover_penalty=0.0, cost_penalty=0.0, participation_penalty=0.0))
    penalized = CostAwarePortfolioConstructor(ConstructorParams(no_trade_band_bps=0.0, turnover_penalty=0.8, cost_penalty=1.5, participation_penalty=0.8))

    raw_orders, _ = baseline.construct_orders(
        target_weights=target_weights,
        current_weights=current_weights,
        prices=prices,
        liquidity=liquidity,
        aum=100000.0,
        cost_params=cost_params,
        max_participation=1.0,
    )
    penalized_orders, _ = penalized.construct_orders(
        target_weights=target_weights,
        current_weights=current_weights,
        prices=prices,
        liquidity=liquidity,
        aum=100000.0,
        cost_params=cost_params,
        max_participation=1.0,
    )

    raw_turnover = sum(abs(value) for value in raw_orders.values())
    penalized_turnover = sum(abs(value) for value in penalized_orders.values())
    assert penalized_turnover < raw_turnover
