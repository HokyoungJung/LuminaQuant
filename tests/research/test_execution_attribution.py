"""Tests for the offline execution-attribution kernel."""

from __future__ import annotations

import math

from lumina_quant.research.execution_attribution import (
    AttributionCostModel,
    ExecutionAttribution,
    FillEvent,
    RoundTrip,
    attribute_execution_delta,
    early_exit_bias_severity,
    late_exit_bias_severity,
    noise_bias_severity,
    overtrading_bias_severity,
    pair_round_trips_fifo,
    run_execution_attribution,
)

_TOL = 1e-9


def _simple_long_fills() -> list[FillEvent]:
    # Two buys then two sells on one symbol: FIFO should pair 1@100 with 1@110,
    # then 1@102 with 1@112.
    return [
        FillEvent(symbol="BTC", side="BUY", qty=1.0, price=100.0, timestamp=0.0, fee=0.0),
        FillEvent(symbol="BTC", side="BUY", qty=1.0, price=102.0, timestamp=1.0, fee=0.0),
        FillEvent(symbol="BTC", side="SELL", qty=1.0, price=110.0, timestamp=2.0, fee=0.0),
        FillEvent(symbol="BTC", side="SELL", qty=1.0, price=112.0, timestamp=3.0, fee=0.0),
    ]


def test_fifo_pairs_oldest_lot_first() -> None:
    trips = pair_round_trips_fifo(_simple_long_fills(), cost_model=None)
    assert len(trips) == 2
    first, second = trips
    assert first.entry_price == 100.0
    assert first.exit_price == 110.0
    assert first.gross_pnl == 10.0
    # FIFO: the second (later) buy pairs with the second sell.
    assert second.entry_price == 102.0
    assert second.exit_price == 112.0
    assert second.gross_pnl == 10.0


def test_fifo_partial_fill_splits_across_round_trips() -> None:
    fills = [
        FillEvent(symbol="ETH", side="BUY", qty=3.0, price=10.0, timestamp=0.0, fee=0.0),
        FillEvent(symbol="ETH", side="SELL", qty=1.0, price=12.0, timestamp=1.0, fee=0.0),
        FillEvent(symbol="ETH", side="SELL", qty=2.0, price=13.0, timestamp=2.0, fee=0.0),
    ]
    trips = pair_round_trips_fifo(fills, cost_model=None)
    assert len(trips) == 2
    assert math.isclose(trips[0].qty, 1.0, abs_tol=_TOL)
    assert math.isclose(trips[1].qty, 2.0, abs_tol=_TOL)
    # Both round trips share the single entry lot at price 10.
    assert trips[0].entry_price == 10.0
    assert trips[1].entry_price == 10.0
    assert math.isclose(trips[0].gross_pnl, 2.0, abs_tol=_TOL)  # (12-10)*1
    assert math.isclose(trips[1].gross_pnl, 6.0, abs_tol=_TOL)  # (13-10)*2


def test_short_round_trip_direction_and_pnl() -> None:
    fills = [
        FillEvent(symbol="SOL", side="SELL", qty=2.0, price=50.0, timestamp=0.0, fee=0.0),
        FillEvent(symbol="SOL", side="BUY", qty=2.0, price=45.0, timestamp=5.0, fee=0.0),
    ]
    trips = pair_round_trips_fifo(fills, cost_model=None)
    assert len(trips) == 1
    trip = trips[0]
    assert trip.direction == "short"
    # Short profits when exit (buy) below entry (sell): (50-45)*2 = 10.
    assert math.isclose(trip.gross_pnl, 10.0, abs_tol=_TOL)
    assert math.isclose(trip.holding_time, 5.0, abs_tol=_TOL)


def test_fees_apportioned_and_net_pnl_bps() -> None:
    model = AttributionCostModel(taker_fee_rate=0.001, maker_fee_rate=0.001)
    fills = [
        FillEvent(symbol="BTC", side="BUY", qty=1.0, price=100.0, timestamp=0.0, fee=None),
        FillEvent(symbol="BTC", side="SELL", qty=1.0, price=110.0, timestamp=1.0, fee=None),
    ]
    trips = pair_round_trips_fifo(fills, cost_model=model)
    assert len(trips) == 1
    trip = trips[0]
    # Fee derived from model: entry 100*1*0.001=0.1, exit 110*1*0.001=0.11.
    assert math.isclose(trip.entry_fee, 0.1, abs_tol=_TOL)
    assert math.isclose(trip.exit_fee, 0.11, abs_tol=_TOL)
    expected_net = 10.0 - 0.1 - 0.11
    assert math.isclose(trip.net_pnl, expected_net, abs_tol=_TOL)
    assert math.isclose(trip.net_pnl_bps, (expected_net / 100.0) * 10_000.0, abs_tol=_TOL)


def test_funding_sign_matches_execution_model_convention() -> None:
    # Positive funding rate -> long pays (a positive cost).
    model = AttributionCostModel(
        taker_fee_rate=0.0,
        maker_fee_rate=0.0,
        funding_rate_per_8h=0.01,
        funding_interval_hours=8.0,
    )
    fills = [
        FillEvent(symbol="BTC", side="BUY", qty=1.0, price=100.0, timestamp=0.0, fee=0.0),
        FillEvent(symbol="BTC", side="SELL", qty=1.0, price=100.0, timestamp=16.0, fee=0.0),
    ]
    trips = pair_round_trips_fifo(fills, cost_model=model)
    trip = trips[0]
    # 16h / 8h = 2 whole intervals; notional 100; interval_rate 0.01 -> 2.0 cost.
    assert math.isclose(trip.funding_paid, 2.0, abs_tol=_TOL)
    assert math.isclose(trip.net_pnl, -2.0, abs_tol=_TOL)


def test_severity_bounds_and_values() -> None:
    assert noise_bias_severity([0.5, -0.5, 100.0], noise_threshold_bps=2.0) == 2.0 / 3.0
    assert noise_bias_severity([]) == 0.0
    # Half of favorable excursion left on table.
    assert math.isclose(early_exit_bias_severity([50.0], [100.0]), 0.5, abs_tol=_TOL)
    assert early_exit_bias_severity([100.0], [0.0]) == 0.0
    # Giveback fully explained by adverse excursion.
    assert math.isclose(late_exit_bias_severity([40.0], [100.0], [80.0]), 0.6, abs_tol=_TOL)
    assert math.isclose(overtrading_bias_severity(2.0, 8.0), 0.2, abs_tol=_TOL)
    assert overtrading_bias_severity(0.0, 0.0) == 0.0
    for value in (
        noise_bias_severity([1.0]),
        early_exit_bias_severity([50.0], [100.0]),
        late_exit_bias_severity([40.0], [100.0], [80.0]),
        overtrading_bias_severity(2.0, 8.0),
    ):
        assert 0.0 <= value <= 1.0


def _attribution_round_trips() -> list[RoundTrip]:
    return [
        # Winner that left upside on the table; small adverse excursion.
        RoundTrip(
            symbol="BTC",
            direction="long",
            qty=1.0,
            entry_time=0.0,
            exit_time=1.0,
            entry_price=100.0,
            exit_price=105.0,
            entry_fee=0.1,
            exit_fee=0.1,
            funding_paid=0.0,
            gross_pnl=5.0,
            net_pnl=4.8,
            net_pnl_bps=48.0,
            holding_time=1.0,
            # Best favorable excursion (700 bps -> 7.0 cash) exceeds the realized
            # gross (5.0), leaving upside on the table for early/late attribution.
            mfe_bps=700.0,
            mae_bps=30.0,
        ),
        # Noise trade: |net_bps| within band.
        RoundTrip(
            symbol="ETH",
            direction="short",
            qty=2.0,
            entry_time=0.0,
            exit_time=1.0,
            entry_price=50.0,
            exit_price=50.0,
            entry_fee=0.05,
            exit_fee=0.05,
            funding_paid=0.0,
            gross_pnl=0.1,
            net_pnl=0.0,
            net_pnl_bps=0.0,
            holding_time=1.0,
            mfe_bps=10.0,
            mae_bps=10.0,
        ),
    ]


def test_attribution_buckets_conserve_total() -> None:
    attribution = attribute_execution_delta(_attribution_round_trips(), noise_threshold_bps=2.0)
    assert isinstance(attribution, ExecutionAttribution)
    # Buckets must sum to total_delta with only float residual in `missed`.
    assert math.isclose(attribution.bucket_sum(), attribution.total_delta, abs_tol=1e-9)
    assert math.isclose(
        attribution.total_delta,
        attribution.benchmark_pnl - attribution.realized_pnl,
        abs_tol=1e-9,
    )
    # Explicit friction goes to overtrading (only the non-noise trade contributes).
    assert math.isclose(attribution.overtrading, 0.2, abs_tol=_TOL)
    # missed is the residual, ~0 for a fully-allocated decomposition.
    assert abs(attribution.missed) < 1e-9


def test_attribution_empty_is_zero() -> None:
    attribution = attribute_execution_delta([], noise_threshold_bps=2.0)
    assert attribution.total_delta == 0.0
    assert attribution.bucket_sum() == 0.0


def test_pipeline_determinism_and_order_invariance() -> None:
    fills = _simple_long_fills()
    model = AttributionCostModel(taker_fee_rate=0.0005, maker_fee_rate=0.0002)
    excursions = [(120.0, 40.0), (90.0, 20.0)]

    report_a = run_execution_attribution(fills, cost_model=model, excursions=excursions)
    # Shuffle the input order: chronological sort must yield identical pairing.
    reordered = [fills[3], fills[0], fills[2], fills[1]]
    report_b = run_execution_attribution(reordered, cost_model=model, excursions=excursions)

    assert report_a.to_dict() == report_b.to_dict()
    assert report_a.severities.round_trip_count == 2
    assert math.isclose(
        report_a.attribution.bucket_sum(),
        report_a.attribution.total_delta,
        abs_tol=1e-9,
    )
