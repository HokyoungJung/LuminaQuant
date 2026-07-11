from __future__ import annotations

import json
import queue
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from lumina_quant.backtesting.execution_model import ExecutionPricingTrace
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import (
    FillApplicationAttribution,
    Portfolio,
)
from lumina_quant.core.events import FillEvent, MarketEvent


class _Config:
    INITIAL_CAPITAL = 1000.0
    TAKER_FEE_RATE = 0.001
    MAKER_FEE_RATE = 0.0002
    SLIPPAGE_RATE = 0.0005
    SPREAD_RATE = 0.0002
    LEVERAGE = 1
    MARGIN_MODE = "isolated"
    MAINTENANCE_MARGIN_RATE = 0.005
    LIQUIDATION_BUFFER_RATE = 0.0
    FUNDING_RATE_PER_8H = 0.01
    FUNDING_INTERVAL_HOURS = 8
    RANDOM_SEED = 42
    SIM_MAX_BAR_VOLUME_RATIO = 0.1
    MAX_DAILY_LOSS_PCT = 0.05
    APPLY_LIQUIDITY_CAP_TO_CONDITIONAL_FILLS = True


class _Bars:
    symbol_list = ["BTC"]

    def __init__(self):
        self.current_dt = datetime(2026, 7, 10, 0, 0, tzinfo=UTC)
        self.raw_point_calls: list[tuple[str, str, int]] = []

    def get_latest_bar_datetime(self, symbol):
        return self.current_dt

    def get_latest_bar_value(self, symbol, field):
        if field == "funding_rate":
            return 0.01
        return 100.0

    def get_latest_raw_point(self, symbol, field, *, timestamp_ms):
        self.raw_point_calls.append((symbol, field, timestamp_ms))
        if field == "funding_rate":
            return (0.01, timestamp_ms - 28_800_000, timestamp_ms)
        if field == "close":
            return (100.0, timestamp_ms - 1_000, timestamp_ms)
        return None


def _portfolio(
    *,
    fill_application_attribution_sink=None,
    funding_boundary_resolver=None,
    full_event_equity_sink=None,
):
    bars = _Bars()
    portfolio = Portfolio(
        bars,
        queue.Queue(),
        datetime(2026, 7, 10, 0, 0, tzinfo=UTC),
        _Config,
        fill_application_attribution_sink=fill_application_attribution_sink,
        funding_boundary_resolver=funding_boundary_resolver,
        full_event_equity_sink=full_event_equity_sink,
    )
    return portfolio, bars


def _handler_fill(*, qty, direction, reduce_only=False, order_id="SIM-1"):
    events = queue.Queue()
    handler = SimulatedExecutionHandler(
        events,
        _Bars(),
        _Config,
        record_cost_attribution=True,
    )
    handler.active_orders = [
        {
            "order_id": order_id,
            "symbol": "BTC",
            "type": "MKT",
            "quantity": qty,
            "direction": direction,
            "status": "PENDING",
            "position_side": "LONG" if direction == "BUY" else "SHORT",
            "reduce_only": reduce_only,
            "client_order_id": f"client-{order_id}",
            "stop_loss": None,
            "take_profit": None,
            "trailing_percent": None,
        }
    ]
    handler.check_open_orders(
        MarketEvent(
            time=datetime(2026, 7, 10, 0, 0, tzinfo=UTC),
            symbol="BTC",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=max(100.0, qty * 10.0),
        )
    )
    return events.get_nowait()


def _fill(*, qty, direction, reduce_only=False):
    return _handler_fill(qty=qty, direction=direction, reduce_only=reduce_only)


def test_fill_application_sink_records_scaled_reduce_only_fill():
    records = []
    portfolio, _ = _portfolio(fill_application_attribution_sink=records.append)
    portfolio.enforce_reduce_only = True
    portfolio.current_positions["BTC"] = -2.0

    portfolio.update_fill(_fill(qty=4.0, direction="BUY", reduce_only=True))

    assert len(records) == 1
    record = records[0]
    assert record["record_type"] == "fill_application_attribution"
    assert record["application_status"] == "applied_scaled"
    assert record["zero_applied_reason"] is None
    assert record["model_quantity"] == pytest.approx(4.0)
    assert record["applied_quantity"] == pytest.approx(2.0)
    assert record["applied_fill_cost"] == pytest.approx(record["model_fill_cost"] * 0.5)
    assert record["applied_commission"] == pytest.approx(record["model_commission"] * 0.5)
    assert record["reduce_only_scale"] == pytest.approx(0.5)
    assert portfolio.current_positions["BTC"] == pytest.approx(0.0)


def test_fill_application_sink_records_unchanged_positive_fill_bijection():
    records = []
    portfolio, _ = _portfolio(fill_application_attribution_sink=records.append)

    portfolio.update_fill(_fill(qty=1.5, direction="BUY"))

    assert len(records) == 1
    record = records[0]
    assert record["application_status"] == "applied_unchanged"
    assert record["zero_applied_reason"] is None
    assert record["model_quantity"] == pytest.approx(record["applied_quantity"])
    assert record["model_fill_cost"] == pytest.approx(record["applied_fill_cost"])
    assert record["model_commission"] == pytest.approx(record["applied_commission"])
    assert record["reduce_only_scale"] == pytest.approx(1.0)
    assert portfolio.current_positions["BTC"] == pytest.approx(1.5)
    assert portfolio.trade_count == 1


def test_fill_application_sink_missing_trace_fails_before_mutation():
    records = []
    portfolio, _ = _portfolio(fill_application_attribution_sink=records.append)
    before_positions = dict(portfolio.current_positions)
    before_holdings = dict(portfolio.current_holdings)

    invalid = FillEvent(
        timeindex=datetime(2026, 7, 10, 0, 0, tzinfo=UTC),
        symbol="BTC",
        exchange="SIM",
        quantity=1.0,
        direction="BUY",
        fill_cost=100.0,
        commission=0.1,
        metadata={},
    )
    with pytest.raises(RuntimeError, match="ExecutionPricingTrace"):
        portfolio.update_fill(invalid)

    assert records == []
    assert portfolio.current_positions == before_positions
    assert portfolio.current_holdings == before_holdings
    assert portfolio.trade_count == 0


def test_fill_application_sink_rejects_trace_event_mismatch_before_mutation():
    records = []
    portfolio, _ = _portfolio(fill_application_attribution_sink=records.append)
    fill = _fill(qty=1.0, direction="BUY")
    fill.fill_cost += 1.0
    before_positions = dict(portfolio.current_positions)
    before_holdings = dict(portfolio.current_holdings)

    with pytest.raises(ValueError, match="fill_cost does not match"):
        portfolio.update_fill(fill)

    assert records == []
    assert portfolio.current_positions == before_positions
    assert portfolio.current_holdings == before_holdings
    assert portfolio.trade_count == 0


def test_default_optional_seams_are_off_for_legacy_positive_fill():
    portfolio, _ = _portfolio()

    assert portfolio.fill_application_attribution_sink is None
    assert portfolio.funding_boundary_resolver is None
    assert portfolio.full_event_equity_sink is None

    fill = _fill(qty=1.0, direction="BUY")
    portfolio.update_fill(fill)

    assert portfolio.current_positions["BTC"] == pytest.approx(1.0)
    assert portfolio.current_holdings["cash"] == pytest.approx(
        1000.0 - fill.fill_cost - fill.commission
    )
    assert portfolio.trade_count == 1


def test_full_event_equity_sink_observes_every_point_in_order_and_preserves_identity():
    class _Collector:
        def __init__(self):
            self.points = []

        def observe(self, point):
            self.points.append(point)

    collector = _Collector()
    portfolio, bars = _portfolio(full_event_equity_sink=collector.observe)

    sink = portfolio.full_event_equity_sink
    assert sink.__self__ is collector
    assert sink.__func__ is collector.observe.__func__

    start = datetime(2026, 7, 10, 0, 0, tzinfo=UTC)
    expected = []
    for offset, equity in enumerate((1000.0, 980.0, 1025.0)):
        bars.current_dt = start + timedelta(seconds=offset)
        portfolio.current_holdings["cash"] = equity
        portfolio.update_timeindex(object())
        expected.append((bars.current_dt.timestamp(), equity))

    assert collector.points == expected
    assert list(portfolio._equity_points) == expected


def test_full_event_equity_sink_failure_is_loud():
    def fail(point):
        raise RuntimeError(f"equity sink failed: {point[1]}")

    portfolio, _ = _portfolio(full_event_equity_sink=fail)

    with pytest.raises(RuntimeError, match=r"equity sink failed: 1000\.0"):
        portfolio.update_timeindex(object())


def test_none_full_event_equity_sink_preserves_legacy_history_and_rolling_points():
    implicit, implicit_bars = _portfolio()
    explicit, explicit_bars = _portfolio(full_event_equity_sink=None)
    start = datetime(2026, 7, 10, 0, 0, tzinfo=UTC)

    for offset, equity in enumerate((1000.0, 995.0, 1010.0)):
        current = start + timedelta(seconds=offset)
        implicit_bars.current_dt = current
        explicit_bars.current_dt = current
        implicit.current_holdings["cash"] = equity
        explicit.current_holdings["cash"] = equity
        implicit.update_timeindex(object())
        explicit.update_timeindex(object())

    assert implicit.all_positions == explicit.all_positions
    assert implicit.all_holdings == explicit.all_holdings
    assert implicit._metric_totals == explicit._metric_totals
    assert implicit._metric_benchmarks == explicit._metric_benchmarks
    assert implicit._equity_points == explicit._equity_points


def test_full_event_equity_sink_is_callable_locked_and_does_not_add_owned_history():
    with pytest.raises(TypeError, match="full_event_equity_sink must be callable"):
        _portfolio(full_event_equity_sink=object())

    class _Counter:
        def __init__(self):
            self.count = 0
            self.last = None

        def observe(self, point):
            self.count += 1
            self.last = point

    counter = _Counter()
    bars = _Bars()
    portfolio = Portfolio(
        bars,
        queue.Queue(),
        datetime(2026, 7, 10, 0, 0, tzinfo=UTC),
        _Config,
        record_history=False,
        track_metrics=False,
        record_trades=False,
        full_event_equity_sink=counter.observe,
    )
    initial_positions = tuple(portfolio.all_positions)
    initial_holdings = tuple(portfolio.all_holdings)

    with pytest.raises(AttributeError):
        portfolio.full_event_equity_sink = lambda point: None
    with pytest.raises(AttributeError):
        portfolio._full_event_equity_sink = lambda point: None

    start = datetime(2026, 7, 10, 0, 0, tzinfo=UTC)
    point_count = 25_000
    for offset in range(point_count):
        portfolio._record_equity_point(start + timedelta(seconds=offset), 1000.0 + offset)

    assert counter.count == point_count
    assert counter.last == ((start + timedelta(seconds=point_count - 1)).timestamp(), 25_999.0)
    assert len(portfolio._equity_points) == portfolio._equity_points.maxlen == 20_000
    assert tuple(portfolio.all_positions) == initial_positions
    assert tuple(portfolio.all_holdings) == initial_holdings
    assert portfolio._metric_totals == []
    assert portfolio._metric_benchmarks == []


@pytest.mark.parametrize(
    ("starting_qty", "direction", "reason"),
    [
        (0.0, "SELL", "reduce_only_flat"),
        (1.0, "BUY", "reduce_only_wrong_side"),
    ],
)
def test_fill_application_sink_records_rejection_without_mutation(starting_qty, direction, reason):
    records = []
    portfolio, _ = _portfolio(fill_application_attribution_sink=records.append)
    portfolio.enforce_reduce_only = True
    portfolio.current_positions["BTC"] = starting_qty
    before_holdings = dict(portfolio.current_holdings)

    portfolio.update_fill(_fill(qty=1.0, direction=direction, reduce_only=True))

    assert len(records) == 1
    record = records[0]
    assert record["application_status"] == "rejected"
    assert record["zero_applied_reason"] == reason
    assert record["applied_quantity"] == pytest.approx(0.0)
    assert portfolio.current_positions["BTC"] == pytest.approx(starting_qty)
    assert portfolio.current_holdings == before_holdings


def test_fill_application_sink_raises_before_state_mutation():
    def sink(_record):
        raise RuntimeError("boom")

    portfolio, _ = _portfolio(fill_application_attribution_sink=sink)
    portfolio.enforce_reduce_only = True
    portfolio.current_positions["BTC"] = 1.0
    before_positions = dict(portfolio.current_positions)
    before_holdings = dict(portfolio.current_holdings)

    with pytest.raises(RuntimeError, match="boom"):
        portfolio.update_fill(_fill(qty=1.0, direction="BUY"))

    assert portfolio.current_positions == before_positions
    assert portfolio.current_holdings == before_holdings


def test_fill_application_sink_record_is_json_evidence_ready():
    records = []
    portfolio, _ = _portfolio(fill_application_attribution_sink=records.append)

    portfolio.update_fill(_fill(qty=1.0, direction="BUY"))

    assert len(records) == 1
    assert type(records[0]) is FillApplicationAttribution
    encoded = json.dumps(records[0].to_payload(), sort_keys=True)
    assert "2026-07-10T00:00:00+00:00" in encoded
    assert records[0]["pricing_trace"]["record_type"] == "execution_pricing_trace"
    assert len(records[0]["pricing_trace_hash"]) == 64
    with pytest.raises(FrozenInstanceError):
        records[0].application_status = "rejected"


def test_fill_application_sink_must_be_callable_and_constructor_bound():
    with pytest.raises(TypeError, match="fill_application_attribution_sink must be callable"):
        _portfolio(fill_application_attribution_sink=object())

    records = []
    portfolio, _ = _portfolio(fill_application_attribution_sink=records.append)

    with pytest.raises(AttributeError):
        portfolio.fill_application_attribution_sink = lambda record: None
    with pytest.raises(AttributeError):
        portfolio._fill_application_attribution_sink = lambda record: None

    portfolio.update_fill(_fill(qty=1.0, direction="BUY"))

    assert len(records) == 1


def test_same_real_pricing_trace_hash_links_unchanged_scaled_and_rejected_applications():
    fill = _fill(qty=4.0, direction="BUY", reduce_only=True)
    trace = fill.metadata["cost_attribution"]
    assert type(trace) is ExecutionPricingTrace
    records = []

    unchanged, _ = _portfolio(fill_application_attribution_sink=records.append)
    unchanged.enforce_reduce_only = True
    unchanged.current_positions["BTC"] = -4.0
    unchanged.update_fill(fill)

    scaled, _ = _portfolio(fill_application_attribution_sink=records.append)
    scaled.enforce_reduce_only = True
    scaled.current_positions["BTC"] = -2.0
    scaled.update_fill(fill)

    rejected, _ = _portfolio(fill_application_attribution_sink=records.append)
    rejected.enforce_reduce_only = True
    rejected.update_fill(fill)

    assert [record.application_status for record in records] == [
        "applied_unchanged",
        "applied_scaled",
        "rejected",
    ]
    assert len({record.pricing_trace_hash for record in records}) == 1
    assert all(record.pricing_trace is trace for record in records)
    assert records[0].pricing_trace_hash == trace.sha256
    assert records[0].canonical_json_bytes() != records[1].canonical_json_bytes()


def test_real_handler_conditional_and_remainder_have_distinct_bijective_applications():
    events = queue.Queue()
    handler = SimulatedExecutionHandler(
        events,
        _Bars(),
        _Config,
        record_cost_attribution=True,
    )
    handler.active_orders = [
        {
            "order_id": "STOP-1",
            "symbol": "BTC",
            "type": "STOP",
            "quantity": 3.0,
            "direction": "SELL",
            "status": "PENDING",
            "stop_price": 100.0,
            "position_side": "LONG",
            "reduce_only": True,
            "client_order_id": "conditional-client",
            "parent_order_id": "ENTRY-1",
            "is_protective": True,
            "oco_group": "BRACKET-1",
        }
    ]
    applications = []
    portfolio, _ = _portfolio(fill_application_attribution_sink=applications.append)
    portfolio.enforce_reduce_only = True
    portfolio.current_positions["BTC"] = 3.0

    handler.check_open_orders(
        MarketEvent(
            time=datetime(2026, 7, 10, 0, 0, tzinfo=UTC),
            symbol="BTC",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=10.0,
        )
    )
    first = events.get_nowait()
    portfolio.update_fill(first)
    assert handler.active_orders[0]["order_id"] == "STOP-1-R"

    handler.check_open_orders(
        MarketEvent(
            time=datetime(2026, 7, 10, 0, 0, 1, tzinfo=UTC),
            symbol="BTC",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=100.0,
        )
    )
    second = events.get_nowait()
    portfolio.update_fill(second)

    first_trace = first.metadata["cost_attribution"]
    second_trace = second.metadata["cost_attribution"]
    assert first_trace.order_kind == "STOP"
    assert second_trace.order_kind == "MKT"
    assert second_trace.remainder_of_order_id == "STOP-1"
    assert len(applications) == 2
    assert applications[0].pricing_trace is first_trace
    assert applications[1].pricing_trace is second_trace
    assert applications[0].pricing_trace_hash != applications[1].pricing_trace_hash
    assert portfolio.current_positions["BTC"] == pytest.approx(0.0)


def test_synthetic_liquidation_remains_distinct_with_zero_application_records():
    applications = []
    portfolio, _ = _portfolio(fill_application_attribution_sink=applications.append)
    portfolio.current_positions["BTC"] = 1.0
    liquidation = FillEvent(
        timeindex=datetime(2026, 7, 10, 0, 0, tzinfo=UTC),
        symbol="BTC",
        exchange="SIM_LIQUIDATION",
        quantity=1.0,
        direction="SELL",
        fill_cost=90.0,
        commission=0.09,
        position_side="LONG",
        status="LIQUIDATED",
        metadata={"reason": "maintenance_margin_breach"},
    )

    portfolio.update_fill(liquidation)

    assert applications == []
    assert portfolio.current_positions["BTC"] == pytest.approx(0.0)
    assert portfolio.trade_count == 1


def test_collector_record_application_bound_method_contract_is_preserved():
    class _Collector:
        def __init__(self):
            self.records = []

        def record_application(self, record):
            self.records.append(record)

    collector = _Collector()
    portfolio, _ = _portfolio(fill_application_attribution_sink=collector.record_application)
    portfolio.update_fill(_fill(qty=1.0, direction="BUY"))

    assert len(collector.records) == 1
    assert type(collector.records[0]) is FillApplicationAttribution


def test_optional_seams_are_keyword_only():
    bars = _Bars()

    with pytest.raises(TypeError):
        Portfolio(
            bars,
            queue.Queue(),
            datetime(2026, 7, 10, 0, 0, tzinfo=UTC),
            _Config,
            True,
            True,
            True,
            None,
            lambda record: None,
        )


def test_funding_boundary_resolver_uses_raw_point_accessor():
    class _Resolver:
        def __init__(self):
            self.calls = []

        def resolve(self, **kwargs):
            self.calls.append(kwargs)
            raw_point_accessor = kwargs["raw_point_accessor"]
            rate_point = raw_point_accessor(
                kwargs["symbol"], "funding_rate", timestamp_ms=kwargs["boundary_ms"]
            )
            price_point = raw_point_accessor(
                kwargs["symbol"], "close", timestamp_ms=kwargs["boundary_ms"]
            )
            return {"rate": rate_point, "price": price_point}

    resolver = _Resolver()
    portfolio, bars = _portfolio(funding_boundary_resolver=resolver)
    portfolio.current_positions["BTC"] = 2.0
    anchor_ts = datetime(2026, 7, 10, 0, 0, tzinfo=UTC).timestamp()
    portfolio._last_funding_ts["BTC"] = anchor_ts

    portfolio._apply_funding(datetime(2026, 7, 10, 8, 0, 1, tzinfo=UTC))

    assert len(resolver.calls) == 1
    call = resolver.calls[0]
    raw_accessor = call["raw_point_accessor"]
    assert raw_accessor.__self__ is bars
    assert raw_accessor.__func__ is bars.get_latest_raw_point.__func__
    assert call["boundary_ms"] == int((anchor_ts + 28_800.0) * 1000)
    assert call["qty"] == pytest.approx(2.0)
    assert portfolio._last_funding_ts["BTC"] == pytest.approx(28_800.0 + anchor_ts)
    assert portfolio.total_funding_paid == pytest.approx(2.0)
    assert portfolio.current_holdings["funding"] == pytest.approx(2.0)
    assert portfolio.current_holdings["cash"] == pytest.approx(998.0)


def test_funding_boundary_resolver_settles_each_crossed_boundary_independently():
    class _Resolver:
        def __init__(self):
            self.calls = []

        def resolve(self, **kwargs):
            assert "portfolio" not in kwargs
            self.calls.append(kwargs)
            payment_by_boundary = {
                int(datetime(2026, 7, 10, 8, 0, tzinfo=UTC).timestamp() * 1000): 1.25,
                int(datetime(2026, 7, 10, 16, 0, tzinfo=UTC).timestamp() * 1000): -0.50,
            }
            return {"payment": payment_by_boundary[kwargs["boundary_ms"]]}

    resolver = _Resolver()
    portfolio, _ = _portfolio(funding_boundary_resolver=resolver)
    portfolio.current_positions["BTC"] = 3.0
    anchor_ts = datetime(2026, 7, 10, 0, 0, tzinfo=UTC).timestamp()
    portfolio._last_funding_ts["BTC"] = anchor_ts

    portfolio._apply_funding(datetime(2026, 7, 10, 16, 0, 1, tzinfo=UTC))

    calls = resolver.calls
    assert [call["boundary_ms"] for call in calls] == [
        int((anchor_ts + 28_800.0) * 1000),
        int((anchor_ts + 57_600.0) * 1000),
    ]
    assert [call["qty"] for call in calls] == [pytest.approx(3.0), pytest.approx(3.0)]
    assert portfolio._last_funding_ts["BTC"] == pytest.approx(anchor_ts + 57_600.0)
    assert portfolio.total_funding_paid == pytest.approx(0.75)
    assert portfolio.current_holdings["cash"] == pytest.approx(999.25)
    assert portfolio.current_holdings["funding"] == pytest.approx(0.75)


def test_funding_boundary_resolver_failure_is_atomic_without_portfolio_access():
    class _Resolver:
        def __init__(self):
            self.calls = 0

        def resolve(self, **kwargs):
            assert "portfolio" not in kwargs
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("boundary unavailable")
            return {"payment": 1.25}

    resolver = _Resolver()
    portfolio, _ = _portfolio(funding_boundary_resolver=resolver)
    portfolio.current_positions["BTC"] = 3.0
    anchor_ts = datetime(2026, 7, 10, 0, 0, tzinfo=UTC).timestamp()
    portfolio._last_funding_ts["BTC"] = anchor_ts
    before_holdings = dict(portfolio.current_holdings)

    with pytest.raises(RuntimeError, match="boundary unavailable"):
        portfolio._apply_funding(datetime(2026, 7, 10, 16, 0, 1, tzinfo=UTC))

    assert portfolio.current_holdings == before_holdings
    assert portfolio.total_funding_paid == pytest.approx(0.0)
    assert portfolio._last_funding_ts["BTC"] == pytest.approx(anchor_ts)


def test_funding_boundary_batch_is_single_call_fsum_settled_and_not_replayed():
    class _BatchResolver:
        def __init__(self):
            self.calls = []

        def resolve_batch(self, requests, *, raw_point_accessor, execution_model):
            self.calls.append((requests, raw_point_accessor, execution_model))
            hostile_payments = (1.0e16, 1.0, -1.0e16)
            return tuple(
                {
                    "symbol": request["symbol"],
                    "boundary_ms": request["boundary_ms"],
                    "qty": request["qty"],
                    "payment": hostile_payments[index],
                }
                for index, request in enumerate(requests)
            )

    resolver = _BatchResolver()
    portfolio, bars = _portfolio(funding_boundary_resolver=resolver)
    portfolio.current_positions["BTC"] = 1.0
    anchor_ts = datetime(2026, 7, 10, 0, 0, tzinfo=UTC).timestamp()
    portfolio._last_funding_ts["BTC"] = anchor_ts
    latest = datetime(2026, 7, 11, 0, 0, 1, tzinfo=UTC)

    portfolio._apply_funding(latest)

    assert len(resolver.calls) == 1
    requests, raw_accessor, execution_model = resolver.calls[0]
    assert type(requests) is tuple and len(requests) == 3
    assert set(requests[0]) == {"symbol", "boundary_ms", "qty", "latest_datetime"}
    assert raw_accessor.__self__ is bars
    assert execution_model is portfolio.execution_model
    assert portfolio.current_holdings["cash"] == pytest.approx(999.0)
    assert portfolio.current_holdings["total"] == pytest.approx(999.0)
    assert portfolio.current_holdings["funding"] == pytest.approx(1.0)
    assert portfolio.total_funding_paid == pytest.approx(1.0)
    assert portfolio._last_funding_ts["BTC"] == pytest.approx(anchor_ts + 86_400.0)

    portfolio._apply_funding(latest)
    assert len(resolver.calls) == 1
    assert portfolio.total_funding_paid == pytest.approx(1.0)


def test_funding_boundary_resolver_must_expose_resolve_and_be_constructor_bound():
    with pytest.raises(TypeError, match="funding_boundary_resolver must expose"):
        _portfolio(funding_boundary_resolver=lambda **_: {"payment": 0.0})

    class _Resolver:
        def resolve(self, **kwargs):
            return {"payment": 0.0}

    portfolio, _ = _portfolio(funding_boundary_resolver=_Resolver())

    with pytest.raises(AttributeError):
        portfolio.funding_boundary_resolver = _Resolver()
    with pytest.raises(AttributeError):
        portfolio._funding_boundary_resolver = _Resolver()


def test_none_funding_boundary_resolver_keeps_legacy_clock():
    portfolio, _ = _portfolio()
    portfolio.current_positions["BTC"] = 1.0
    anchor_ts = datetime(2026, 7, 10, 0, 0, tzinfo=UTC).timestamp()
    portfolio._last_funding_ts["BTC"] = anchor_ts

    portfolio._apply_funding(datetime(2026, 7, 10, 9, 0, 0, tzinfo=UTC))

    assert portfolio._last_funding_ts["BTC"] == pytest.approx(anchor_ts + 28_800.0)
    assert portfolio.total_funding_paid == pytest.approx(1.0)
    assert portfolio.current_holdings["funding"] == pytest.approx(1.0)
