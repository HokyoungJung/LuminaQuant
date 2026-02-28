from __future__ import annotations

from lumina_quant.backtesting.cost_models import (
    CostModelParams,
    ExecutionPolicy,
    estimate_cost_bps,
    simulate_market_fill,
)


def test_no_close_fill_basis_enforced_to_next_open():
    fill = simulate_market_fill(
        order_notional=1000.0,
        pending_notional=0.0,
        next_open=101.0,
        next_mid=None,
        close_price=90.0,
        adtv=1_000_000.0,
        sigma=0.01,
        bar_volume=1_000.0,
        params=CostModelParams(spread_bps=2.0, impact_k=1.0, fees_bps=1.0),
        policy=ExecutionPolicy(fill_basis="next_open", max_participation=1.0, unfilled_policy="carry", carry_decay=1.0),
    )
    assert abs(fill.basis_price - 101.0) < 1e-9


def test_participation_cap_and_carry_drop_policy():
    params = CostModelParams(spread_bps=1.0, impact_k=1.0, fees_bps=0.0)
    carry_fill = simulate_market_fill(
        order_notional=20_000.0,
        pending_notional=0.0,
        next_open=100.0,
        next_mid=None,
        close_price=99.0,
        adtv=1_000_000.0,
        sigma=0.01,
        bar_volume=100.0,
        params=params,
        policy=ExecutionPolicy(fill_basis="next_open", max_participation=0.1, unfilled_policy="carry", carry_decay=1.0),
    )
    assert abs(carry_fill.executed_notional) <= 1000.0 + 1e-9
    assert carry_fill.next_pending_notional > 0.0

    drop_fill = simulate_market_fill(
        order_notional=20_000.0,
        pending_notional=0.0,
        next_open=100.0,
        next_mid=None,
        close_price=99.0,
        adtv=1_000_000.0,
        sigma=0.01,
        bar_volume=100.0,
        params=params,
        policy=ExecutionPolicy(fill_basis="next_open", max_participation=0.1, unfilled_policy="drop", carry_decay=1.0),
    )
    assert drop_fill.next_pending_notional == 0.0


def test_monotonic_impact_cost():
    params = CostModelParams(spread_bps=1.0, impact_k=1.2, fees_bps=0.0)
    small = estimate_cost_bps(
        order_notional=1_000.0,
        sigma=0.02,
        adtv=10_000_000.0,
        bar_volume=5000.0,
        price=100.0,
        params=params,
    )
    large = estimate_cost_bps(
        order_notional=40_000.0,
        sigma=0.02,
        adtv=10_000_000.0,
        bar_volume=5000.0,
        price=100.0,
        params=params,
    )
    assert large.impact_bps > small.impact_bps
    assert large.total_bps > small.total_bps
