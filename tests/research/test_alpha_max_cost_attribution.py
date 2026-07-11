from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from lumina_quant.backtesting.execution_model import (
    ExecutionModel,
    ExecutionModelConfig,
    ExecutionPricingTrace,
)
from lumina_quant.backtesting.execution_sim import NoFillAttempt
from lumina_quant.backtesting.portfolio_backtest import FillApplicationAttribution
from lumina_quant.research.alpha_max_evidence import (
    AlphaMaxFundingBoundaryLedgerRow,
    _alpha_max_liquidation_cost_totals,
    _alpha_max_normalize_liquidation_events,
    _validate_alpha_max_engine_event_counts,
    reconcile_alpha_max_cost_attribution,
)


def _trace() -> ExecutionPricingTrace:
    records: list[ExecutionPricingTrace] = []
    model = ExecutionModel(
        ExecutionModelConfig(
            taker_fee_rate=0.0004,
            maker_fee_rate=0.0002,
            slippage_rate=0.0005,
            spread_rate=0.0002,
            leverage=3,
            margin_mode="isolated",
            maintenance_margin_rate=0.005,
            liquidation_buffer_rate=0.0005,
            funding_rate_per_8h=0.0,
            funding_interval_hours=8,
            random_seed=7,
            max_bar_volume_ratio=0.1,
            slippage_impact_model="sqrt_impact",
            slippage_impact_coefficient=0.1,
        )
    )
    model.compute_fill(
        raw_price=100.0,
        qty=2.0,
        direction="BUY",
        bar_volume=100.0,
        order_kind="MKT",
        order_id="O-1",
        attribution_sink=records.append,
    )
    return records[0]


def _application(trace: ExecutionPricingTrace) -> FillApplicationAttribution:
    return FillApplicationAttribution(
        record_type="fill_application_attribution",
        pricing_trace_hash=trace.sha256,
        pricing_trace=trace,
        timeindex="2025-01-01T00:00:00+00:00",
        symbol="BTCUSDT",
        direction="BUY",
        order_id="O-1",
        client_order_id=None,
        position_side=None,
        status=None,
        reduce_only=False,
        model_quantity=trace.executed_qty,
        model_fill_cost=trace.fill_price * trace.executed_qty,
        model_commission=trace.commission,
        applied_quantity=trace.executed_qty,
        applied_fill_cost=trace.fill_price * trace.executed_qty,
        applied_commission=trace.commission,
        reduce_only_scale=1.0,
        application_status="applied_unchanged",
        zero_applied_reason=None,
    )


def _no_fill() -> NoFillAttempt:
    return NoFillAttempt(
        record_type="no_fill_attempt",
        reason="liquidity_cap_zero_market",
        timeindex="2025-01-01T00:00:01+00:00",
        symbol="BTCUSDT",
        direction="BUY",
        requested_qty=1.0,
        executed_qty=0.0,
        unfilled_qty=1.0,
        raw_price=100.0,
        bar_volume=0.0,
        cap_ratio=0.1,
        order_id="O-2",
        order_kind="MKT",
        client_order_id=None,
        parent_order_id=None,
        remainder_of_order_id=None,
        oco_group=None,
        trigger_price=None,
        position_side=None,
        reduce_only=False,
        is_maker=False,
        rng_consumed=True,
    )


def test_pricing_application_bijection_excludes_no_fill_and_reconciles_all_cost_layers() -> None:
    trace = _trace()
    application = _application(trace)
    funding = AlphaMaxFundingBoundaryLedgerRow(
        symbol="BTCUSDT",
        boundary_ms=1_735_689_600_000,
        rate_source_timestamp_ms=1_735_689_600_000,
        price_row_timestamp_ms=1_735_689_599_000,
        price_close_timestamp_ms=1_735_689_600_000,
        qty=trace.executed_qty,
        rate=0.0001,
        price=100.0,
        payment=-0.02,
    )
    result = reconcile_alpha_max_cost_attribution(
        [trace],
        [application],
        [_no_fill()],
        [funding],
        portfolio_fee_total=application.applied_commission + 3.0,
        portfolio_funding_total=-0.02,
        liquidation_cost_total=3.0,
        portfolio_liquidation_total=3.0,
    )
    assert result.pricing_trace_count == result.application_count == 1
    assert result.no_fill_attempt_count == 1
    assert result.no_fill_excluded_from_bijection is True
    assert result.fee_reconciled is True
    assert result.funding_reconciled is True
    assert result.liquidation_reconciled is True
    assert result.complete is True


def test_engine_event_counts_keep_synthetic_liquidation_outside_pricing_bijection() -> None:
    _validate_alpha_max_engine_event_counts(
        fill_event_count=2,
        pricing_trace_count=1,
        application_count=1,
        liquidation_event_count=1,
        trade_count=2,
    )
    with pytest.raises(ValueError, match="engine_count_mismatch"):
        _validate_alpha_max_engine_event_counts(
            fill_event_count=1,
            pricing_trace_count=1,
            application_count=1,
            liquidation_event_count=1,
            trade_count=1,
        )
    with pytest.raises(ValueError, match="engine_count_mismatch"):
        _validate_alpha_max_engine_event_counts(
            fill_event_count=2,
            pricing_trace_count=1,
            application_count=2,
            liquidation_event_count=1,
            trade_count=2,
        )


def test_liquidation_commission_is_sealed_and_reconciled_as_portfolio_fee_residual() -> None:
    trace = _trace()
    application = _application(trace)
    normalized = _alpha_max_normalize_liquidation_events(
        (
            {
                "time": datetime(2026, 7, 10, 8, 0, tzinfo=UTC),
                "symbol": "BTCUSDT",
                "position_qty": 1.0,
                "entry_price": 100.0,
                "liquidation_price": 75.0,
                "close_price": 74.0,
                "fill_cost": 75.0,
                "commission": 0.03,
                "configured_margin_mode": "isolated",
                "modeled_margin_mode": "isolated",
            },
        )
    )

    liquidation_cost, portfolio_liquidation = _alpha_max_liquidation_cost_totals(
        normalized,
        (application,),
        application.applied_commission + 0.03,
    )

    assert normalized[0].fill_cost == pytest.approx(75.0)
    assert normalized[0].commission == pytest.approx(0.03)
    assert liquidation_cost == pytest.approx(0.03)
    assert portfolio_liquidation == pytest.approx(0.03)


def test_reconciliation_rejects_missing_duplicate_or_wrong_trace_and_totals() -> None:
    trace = _trace()
    application = _application(trace)
    kwargs = dict(
        no_fill_attempts=(),
        funding_ledger=(),
        portfolio_fee_total=application.applied_commission,
        portfolio_funding_total=0.0,
        liquidation_cost_total=0.0,
        portfolio_liquidation_total=0.0,
    )
    with pytest.raises(ValueError, match="pricing_application_bijection"):
        reconcile_alpha_max_cost_attribution([trace], [], **kwargs)
    with pytest.raises(ValueError, match="pricing_application_bijection"):
        reconcile_alpha_max_cost_attribution([trace], [application, application], **kwargs)

    other = _trace()
    wrong = replace(application, pricing_trace=other, pricing_trace_hash=other.sha256)
    with pytest.raises(ValueError, match="pricing_application_bijection"):
        reconcile_alpha_max_cost_attribution([trace], [wrong], **kwargs)
    with pytest.raises(ValueError, match="fee_reconciliation"):
        reconcile_alpha_max_cost_attribution(
            [trace],
            [application],
            (),
            (),
            portfolio_fee_total=999.0,
            portfolio_funding_total=0.0,
        )

    funding = AlphaMaxFundingBoundaryLedgerRow(
        symbol="BTCUSDT",
        boundary_ms=1_735_689_600_000,
        rate_source_timestamp_ms=1_735_689_600_000,
        price_row_timestamp_ms=1_735_689_599_000,
        price_close_timestamp_ms=1_735_689_600_000,
        qty=trace.executed_qty,
        rate=0.0001,
        price=100.0,
        payment=-0.02,
    )
    with pytest.raises(ValueError, match="funding_reconciliation_ledger_order"):
        reconcile_alpha_max_cost_attribution(
            [trace],
            [application],
            (),
            (funding, funding),
            portfolio_fee_total=application.applied_commission,
            portfolio_funding_total=-0.04,
        )
