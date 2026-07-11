from __future__ import annotations

import inspect
import queue
import random
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
from datetime import UTC, datetime

import pytest

from lumina_quant.backtesting.execution_model import (
    ExecutionModel,
    ExecutionModelConfig,
    ExecutionPricingTrace,
    FillResult,
    execution_pricing_trace_sha256,
)
from lumina_quant.backtesting.execution_sim import (
    NoFillAttempt,
    SimulatedExecutionHandler,
)
from lumina_quant.core.events import MarketEvent


def _model_config(**overrides) -> ExecutionModelConfig:
    values = {
        "taker_fee_rate": 0.0004,
        "maker_fee_rate": 0.0002,
        "slippage_rate": 0.001,
        "spread_rate": 0.0004,
        "leverage": 1,
        "margin_mode": "isolated",
        "maintenance_margin_rate": 0.005,
        "liquidation_buffer_rate": 0.0,
        "funding_rate_per_8h": 0.0,
        "funding_interval_hours": 8,
        "random_seed": 17,
        "max_bar_volume_ratio": 0.1,
        "slippage_impact_model": "sqrt_impact",
        "slippage_impact_coefficient": 0.02,
        "slippage_adv_quote": 0.0,
    }
    values.update(overrides)
    return ExecutionModelConfig(**values)


class _Config:
    RANDOM_SEED = 17
    TAKER_FEE_RATE = 0.0004
    COMMISSION_RATE = 0.0004
    MAKER_FEE_RATE = 0.0002
    SLIPPAGE_RATE = 0.001
    SPREAD_RATE = 0.0004
    LEVERAGE = 1
    MARGIN_MODE = "isolated"
    MAINTENANCE_MARGIN_RATE = 0.005
    LIQUIDATION_BUFFER_RATE = 0.0
    FUNDING_RATE_PER_8H = 0.0
    FUNDING_INTERVAL_HOURS = 8
    SIM_MAX_BAR_VOLUME_RATIO = 0.1
    SLIPPAGE_IMPACT_MODEL = "sqrt_impact"
    SLIPPAGE_IMPACT_COEFFICIENT = 0.02
    SLIPPAGE_ADV_QUOTE = 0.0
    SIM_LATENCY_MIN_BARS = 1
    SIM_LATENCY_MAX_BARS = 1
    APPLY_LIQUIDITY_CAP_TO_CONDITIONAL_FILLS = True


class _Bars:
    symbol_list = ["BTC/USDT"]

    @staticmethod
    def get_latest_bar_value(symbol, field):
        _ = symbol
        return {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 100.0,
        }.get(field, 100.0)


def _market(
    *,
    timestamp: int = 1,
    open_price: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    volume: float = 100.0,
) -> MarketEvent:
    return MarketEvent(
        time=datetime(2026, 7, 10, 0, 0, timestamp, tzinfo=UTC),
        symbol="BTC/USDT",
        open=open_price,
        high=high,
        low=low,
        close=open_price,
        volume=volume,
    )


def _handler(*, enabled: bool) -> tuple[SimulatedExecutionHandler, queue.Queue]:
    events = queue.Queue()
    return (
        SimulatedExecutionHandler(
            events,
            _Bars(),
            _Config,
            record_cost_attribution=enabled,
        ),
        events,
    )


def _mkt_order(*, order_id: str = "M-1", quantity: float = 2.0) -> dict[str, object]:
    return {
        "order_id": order_id,
        "symbol": "BTC/USDT",
        "type": "MKT",
        "quantity": quantity,
        "direction": "BUY",
        "status": "PENDING",
        "position_side": "LONG",
        "reduce_only": False,
        "client_order_id": "client-mkt",
        "stop_loss": None,
        "take_profit": None,
        "trailing_percent": None,
    }


def _limit_order(*, crossed: bool = True) -> tuple[dict[str, object], MarketEvent]:
    limit_price = 100.0 if crossed else 90.0
    return (
        {
            "order_id": "L-1",
            "symbol": "BTC/USDT",
            "type": "LMT",
            "quantity": 2.0,
            "direction": "BUY",
            "limit_price": limit_price,
            "status": "PENDING",
            "position_side": "LONG",
            "reduce_only": False,
            "client_order_id": "client-limit",
            "stop_loss": None,
            "take_profit": None,
            "trailing_percent": None,
        },
        _market(low=99.0, volume=0.0),
    )


def _conditional_order(order_kind: str, *, quantity: float = 2.0) -> dict[str, object]:
    base: dict[str, object] = {
        "order_id": f"C-{order_kind}",
        "symbol": "BTC/USDT",
        "type": order_kind,
        "quantity": quantity,
        "direction": "SELL",
        "status": "PENDING",
        "stop_price": 100.0,
        "position_side": "LONG",
        "reduce_only": True,
        "client_order_id": f"client-{order_kind}",
        "parent_order_id": "ENTRY-1",
        "is_protective": True,
        "oco_group": "BRACKET-1",
    }
    if order_kind == "TRAIL_STOP":
        base.update({"trailing_percent": 0.01, "highest_price": 101.0, "lowest_price": None})
    return base


def _core_fill_values(fill) -> tuple[object, ...]:
    return (
        fill.timeindex,
        fill.symbol,
        fill.exchange,
        fill.quantity,
        fill.direction,
        fill.fill_cost,
        fill.commission,
        fill.order_id,
        fill.client_order_id,
        fill.position_side,
        fill.status,
        fill.type,
    )


def _economic_handler_state(handler) -> dict[str, object]:
    state = handler.get_state()
    state.pop("no_fill_attempt_evidence", None)
    return state


def test_compute_fill_default_off_is_signature_and_rng_neutral():
    assert [field.name for field in fields(FillResult)] == [
        "fill_price",
        "commission",
        "executed_qty",
        "unfilled_qty",
    ]
    signature = inspect.signature(ExecutionModel.compute_fill)
    assert signature.parameters["attribution_sink"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["attribution_sink"].default is None

    cfg = _model_config()
    off = ExecutionModel(cfg)
    on = ExecutionModel(cfg)
    traces: list[ExecutionPricingTrace] = []
    kwargs = {
        "raw_price": 100.0,
        "qty": 5.0,
        "direction": "BUY",
        "bar_volume": 20.0,
        "volatility": 0.02,
        "apply_liquidity_cap": True,
        "order_notional": 500.0,
    }

    off_result = off.compute_fill(**kwargs)
    on_result = on.compute_fill(**kwargs, attribution_sink=traces.append)

    assert off_result == on_result
    assert off._rng.getstate() == on._rng.getstate()
    assert len(traces) == 1
    with pytest.raises(FrozenInstanceError):
        traces[0].fill_price = 0.0

    later_off = off.compute_fill(**kwargs)
    later_on = on.compute_fill(**kwargs, attribution_sink=traces.append)
    assert later_off == later_on
    assert off._rng.getstate() == on._rng.getstate()


def test_taker_partial_trace_reconciles_every_price_component():
    traces: list[ExecutionPricingTrace] = []
    model = ExecutionModel(_model_config(random_seed=7))

    result = model.compute_fill(
        raw_price=100.0,
        qty=5.0,
        direction="BUY",
        bar_volume=20.0,
        volatility=0.02,
        apply_liquidity_cap=True,
        order_notional=500.0,
        order_kind="MKT",
        order_id="M-1",
        attribution_sink=traces.append,
    )

    assert len(traces) == 1
    trace = traces[0]
    assert trace.requested_qty == 5.0
    assert trace.executed_qty == result.executed_qty == 2.0
    assert trace.unfilled_qty == result.unfilled_qty == 3.0
    assert trace.liquidity_cap == 2.0
    assert trace.sampled_base_slip == pytest.approx(random.Random(7).uniform(0.0005, 0.0015))
    assert trace.volatility_multiplier == 2.0
    assert trace.applied_slip == pytest.approx(trace.sampled_base_slip * 2.0)
    assert trace.impact_denominator == pytest.approx(2_000.0)
    assert trace.participation == pytest.approx(0.25)
    assert trace.sqrt_impact == pytest.approx(0.01)
    components = trace.applied_slip + trace.half_spread + trace.sqrt_impact + trace.clamp_adjustment
    realized = trace.fill_price / trace.raw_price - 1.0
    assert components == pytest.approx(trace.penalty_after_clamp)
    assert realized == pytest.approx(components)
    assert trace.commission == pytest.approx(trace.fill_price * trace.executed_qty * trace.fee_rate)
    assert trace.liquidity_role == "taker"
    assert trace.rng_consumed is True


def test_maker_partial_trace_uses_exact_price_fee_and_no_rng():
    traces: list[ExecutionPricingTrace] = []
    model = ExecutionModel(_model_config(random_seed=11))
    before_rng = model._rng.getstate()

    result = model.compute_fill(
        raw_price=99.0,
        qty=4.0,
        direction="BUY",
        bar_volume=10.0,
        is_maker=True,
        apply_liquidity_cap=True,
        order_kind="LMT",
        trigger_price=99.0,
        attribution_sink=traces.append,
    )

    trace = traces[0]
    assert result.executed_qty == 1.0
    assert result.unfilled_qty == 3.0
    assert trace.fill_price == trace.raw_price == 99.0
    assert trace.fee_rate == pytest.approx(0.0002)
    assert trace.commission == pytest.approx(99.0 * 0.0002)
    assert trace.sampled_base_slip == 0.0
    assert trace.applied_slip == 0.0
    assert trace.half_spread == 0.0
    assert trace.sqrt_impact == 0.0
    assert trace.participation is None
    assert trace.impact_denominator is None
    assert trace.penalty_before_clamp == trace.penalty_after_clamp == 0.0
    assert trace.clamp_adjustment == 0.0
    assert trace.liquidity_role == "maker"
    assert trace.rng_consumed is False
    assert model._rng.getstate() == before_rng


def test_trace_records_exact_99_percent_clamp_adjustment():
    traces: list[ExecutionPricingTrace] = []
    model = ExecutionModel(
        _model_config(
            slippage_rate=0.0,
            spread_rate=0.0,
            slippage_impact_coefficient=2.0,
            slippage_adv_quote=100.0,
        )
    )

    result = model.compute_fill(
        raw_price=100.0,
        qty=1.0,
        direction="SELL",
        bar_volume=100.0,
        apply_liquidity_cap=False,
        order_notional=100.0,
        attribution_sink=traces.append,
    )

    trace = traces[0]
    assert trace.penalty_before_clamp == pytest.approx(2.0)
    assert trace.penalty_after_clamp == pytest.approx(0.99)
    assert trace.clamp_adjustment == pytest.approx(-1.01)
    assert (
        trace.applied_slip + trace.half_spread + trace.sqrt_impact + trace.clamp_adjustment
        == pytest.approx(0.99)
    )
    assert result.fill_price == pytest.approx(1.0)


def test_zero_execution_emits_no_pricing_trace_but_preserves_rng_sequence():
    cfg = _model_config()
    off = ExecutionModel(cfg)
    on = ExecutionModel(cfg)
    traces: list[ExecutionPricingTrace] = []

    off_result = off.compute_fill(
        raw_price=100.0,
        qty=2.0,
        direction="BUY",
        bar_volume=0.0,
    )
    on_result = on.compute_fill(
        raw_price=100.0,
        qty=2.0,
        direction="BUY",
        bar_volume=0.0,
        attribution_sink=traces.append,
    )

    assert off_result == on_result
    assert off_result.executed_qty == 0.0
    assert traces == []
    assert off._rng.getstate() == on._rng.getstate()


def test_pricing_sink_exception_propagates_after_canonical_rng_draw():
    cfg = _model_config()
    failed = ExecutionModel(cfg)
    reference = ExecutionModel(cfg)

    def sink(_trace):
        raise RuntimeError("pricing evidence failed")

    with pytest.raises(RuntimeError, match="pricing evidence failed"):
        failed.compute_fill(
            raw_price=100.0,
            qty=1.0,
            direction="BUY",
            bar_volume=100.0,
            attribution_sink=sink,
        )
    reference.compute_fill(
        raw_price=100.0,
        qty=1.0,
        direction="BUY",
        bar_volume=100.0,
    )
    assert failed._rng.getstate() == reference._rng.getstate()


def test_handler_activation_is_exact_constructor_owned_bool():
    off, _ = _handler(enabled=False)
    on, _ = _handler(enabled=True)

    assert off.record_cost_attribution is False
    assert off.pricing_attribution_sink is None
    assert on.record_cost_attribution is True
    assert on.pricing_attribution_sink is not None
    assert on.pricing_attribution_sink.__self__ is on
    assert on.pricing_attribution_sink.__func__ is SimulatedExecutionHandler._capture_pricing_trace
    assert off.pricing_trace_evidence == ()
    assert on.pricing_trace_evidence == ()
    assert off.no_fill_attempt_evidence == ()
    with pytest.raises(TypeError, match="exact bool"):
        SimulatedExecutionHandler(queue.Queue(), _Bars(), _Config, record_cost_attribution=1)
    with pytest.raises(AttributeError):
        on.record_cost_attribution = False
    with pytest.raises(AttributeError):
        on._record_cost_attribution = False
    with pytest.raises(AttributeError):
        on._attribution_seams_locked = False
    with pytest.raises(TypeError):
        SimulatedExecutionHandler(queue.Queue(), _Bars(), _Config, True)


def test_handler_positive_fill_on_off_core_events_orders_and_rng_match():
    off, off_events = _handler(enabled=False)
    on, on_events = _handler(enabled=True)
    off.active_orders = [_mkt_order()]
    on.active_orders = deepcopy(off.active_orders)

    market = _market(volume=100.0)
    off.check_open_orders(market)
    on.check_open_orders(market)

    off_fill = off_events.get_nowait()
    on_fill = on_events.get_nowait()
    assert _core_fill_values(off_fill) == _core_fill_values(on_fill)
    assert off_fill.metadata == {
        "reduce_only": False,
        "signal_metadata": {},
        "component_id": None,
    }
    assert "cost_attribution" not in off_fill.metadata
    on_core_metadata = dict(on_fill.metadata)
    trace = on_core_metadata.pop("cost_attribution")
    assert on_core_metadata == off_fill.metadata
    assert isinstance(trace, ExecutionPricingTrace)
    assert trace.order_id == "M-1"
    assert trace.order_kind == "MKT"
    assert trace.remainder_of_order_id is None
    assert off.pricing_trace_evidence == ()
    assert on.pricing_trace_evidence == (trace,)
    assert off.active_orders == on.active_orders
    assert "no_fill_attempt_evidence" not in off.get_state()
    assert on.get_state()["no_fill_attempt_evidence"] == []
    assert _economic_handler_state(off) == _economic_handler_state(on)

    off.active_orders = [_mkt_order(order_id="M-2")]
    on.active_orders = deepcopy(off.active_orders)
    later_market = _market(timestamp=2, volume=100.0)
    off.check_open_orders(later_market)
    on.check_open_orders(later_market)
    off_later_fill = off_events.get_nowait()
    on_later_fill = on_events.get_nowait()
    assert _core_fill_values(off_later_fill) == _core_fill_values(on_later_fill)
    assert off.pricing_trace_evidence == ()
    assert on.pricing_trace_evidence == (
        trace,
        on_later_fill.metadata["cost_attribution"],
    )
    assert _economic_handler_state(off) == _economic_handler_state(on)


@pytest.mark.parametrize("order_kind", ["STOP", "TAKE_PROFIT", "TRAIL_STOP"])
def test_positive_conditional_trace_preserves_kind_trigger_and_parent(order_kind):
    handler, events = _handler(enabled=True)
    handler.active_orders = [_conditional_order(order_kind)]

    handler.check_open_orders(_market(open_price=100.0, high=101.0, low=99.0, volume=100.0))

    fill = events.get_nowait()
    trace = fill.metadata["cost_attribution"]
    assert trace.order_kind == order_kind
    assert trace.trigger_price == 100.0
    assert trace.parent_order_id == "ENTRY-1"
    assert trace.order_id == f"C-{order_kind}"
    assert trace.is_maker is False
    assert trace.rng_consumed is True


def test_partial_remainder_trace_links_to_immediate_order_without_state_changes():
    handler, events = _handler(enabled=True)
    handler.active_orders = [_mkt_order(quantity=15.0)]

    handler.check_open_orders(_market(volume=100.0))
    first = events.get_nowait()
    first_trace = first.metadata["cost_attribution"]
    assert first_trace.requested_qty == 15.0
    assert first_trace.executed_qty == 10.0
    assert first_trace.unfilled_qty == 5.0
    assert first_trace.remainder_of_order_id is None
    assert handler.active_orders[0]["order_id"] == "M-1-R"

    handler.check_open_orders(_market(timestamp=2, volume=100.0))
    second = events.get_nowait()
    second_trace = second.metadata["cost_attribution"]
    assert second_trace.order_id == "M-1-R"
    assert second_trace.remainder_of_order_id == "M-1"
    assert second_trace.requested_qty == 5.0
    assert second_trace.executed_qty == 5.0


def test_zero_volume_market_no_fill_attempt_and_on_off_state_are_identical():
    off, off_events = _handler(enabled=False)
    on, on_events = _handler(enabled=True)
    off.active_orders = [_mkt_order(quantity=2.0)]
    on.active_orders = deepcopy(off.active_orders)

    market = _market(volume=0.0)
    off.check_open_orders(market)
    on.check_open_orders(market)

    assert off_events.empty() and on_events.empty()
    assert off.active_orders == on.active_orders
    assert _economic_handler_state(off) == _economic_handler_state(on)
    assert off.pricing_trace_evidence == ()
    assert on.pricing_trace_evidence == ()
    assert off.no_fill_attempt_evidence == ()
    assert len(on.no_fill_attempt_evidence) == 1
    record = on.no_fill_attempt_evidence[0]
    assert isinstance(record, NoFillAttempt)
    assert record.reason == "liquidity_cap_zero_market"
    assert record.requested_qty == record.unfilled_qty == 2.0
    assert record.executed_qty == 0.0
    assert record.raw_price == 100.0
    assert record.bar_volume == 0.0
    assert record.cap_ratio == pytest.approx(0.1)
    assert record.order_id == "M-1"
    assert record.order_kind == "MKT"
    assert record.is_maker is False
    assert record.rng_consumed is True
    with pytest.raises(FrozenInstanceError):
        record.reason = "changed"


def test_zero_volume_crossed_limit_records_attempt_but_non_crossed_does_not():
    crossed, crossed_events = _handler(enabled=True)
    crossed_off, crossed_off_events = _handler(enabled=False)
    crossed_order, market = _limit_order(crossed=True)
    crossed.active_orders = [crossed_order]
    crossed_off.active_orders = deepcopy(crossed.active_orders)
    before_rng = crossed.execution_model._rng.getstate()

    crossed.check_open_orders(market)
    crossed_off.check_open_orders(market)

    assert crossed_events.empty() and crossed_off_events.empty()
    assert len(crossed.no_fill_attempt_evidence) == 1
    record = crossed.no_fill_attempt_evidence[0]
    assert record.reason == "liquidity_cap_zero_limit"
    assert record.order_kind == "LMT"
    assert record.is_maker is True
    assert record.rng_consumed is False
    assert record.trigger_price == 100.0
    assert crossed.execution_model._rng.getstate() == before_rng
    assert crossed.active_orders[0]["order_id"] == "L-1-R"
    assert crossed.active_orders == crossed_off.active_orders
    assert _economic_handler_state(crossed) == _economic_handler_state(crossed_off)

    non_crossed, non_crossed_events = _handler(enabled=True)
    non_crossed_order, non_crossed_market = _limit_order(crossed=False)
    non_crossed.active_orders = [non_crossed_order]
    before_order = deepcopy(non_crossed.active_orders)
    non_crossed.check_open_orders(non_crossed_market)
    assert non_crossed_events.empty()
    assert non_crossed.no_fill_attempt_evidence == ()
    assert non_crossed.active_orders == before_order


@pytest.mark.parametrize("order_kind", ["STOP", "TAKE_PROFIT", "TRAIL_STOP"])
def test_zero_volume_triggered_conditional_records_one_attempt_and_keeps_lineage(
    order_kind,
):
    handler, events = _handler(enabled=True)
    off, off_events = _handler(enabled=False)
    handler.active_orders = [_conditional_order(order_kind)]
    off.active_orders = deepcopy(handler.active_orders)

    handler.check_open_orders(_market(open_price=100.0, high=101.0, low=99.0, volume=0.0))
    off.check_open_orders(_market(open_price=100.0, high=101.0, low=99.0, volume=0.0))

    assert events.empty() and off_events.empty()
    assert len(handler.no_fill_attempt_evidence) == 1
    record = handler.no_fill_attempt_evidence[0]
    assert record.reason == "liquidity_cap_zero_conditional"
    assert record.order_kind == order_kind
    assert record.order_id == f"C-{order_kind}"
    assert record.parent_order_id == "ENTRY-1"
    assert record.oco_group == "BRACKET-1"
    assert record.trigger_price == 100.0
    assert record.rng_consumed is True
    assert [order["order_id"] for order in handler.active_orders] == [f"C-{order_kind}-R"]
    assert handler.active_orders == off.active_orders
    assert _economic_handler_state(handler) == _economic_handler_state(off)


def test_untriggered_conditional_emits_no_attempt():
    handler, events = _handler(enabled=True)
    order = _conditional_order("STOP")
    order["stop_price"] = 50.0
    handler.active_orders = [order]

    handler.check_open_orders(_market(open_price=100.0, high=101.0, low=99.0, volume=0.0))

    assert events.empty()
    assert handler.no_fill_attempt_evidence == ()
    assert handler.active_orders == [order]


def test_no_fill_evidence_failure_is_loud_before_remainder_or_order_mutation(monkeypatch):
    handler, events = _handler(enabled=True)
    order, market = _limit_order(crossed=True)
    handler.active_orders = [order]
    before_orders = deepcopy(handler.active_orders)

    def fail(**_kwargs):
        raise RuntimeError("no-fill evidence failed")

    monkeypatch.setattr(handler, "_emit_no_fill_attempt", fail)
    with pytest.raises(RuntimeError, match="no-fill evidence failed"):
        handler.check_open_orders(market)

    assert events.empty()
    assert handler.active_orders == before_orders
    assert handler.no_fill_attempt_evidence == ()


def test_handler_pricing_sink_failure_is_loud_before_fill_or_order_mutation(monkeypatch):
    def fail(_self, _trace):
        raise RuntimeError("handler pricing sink failed")

    monkeypatch.setattr(SimulatedExecutionHandler, "_capture_pricing_trace", fail)
    handler, events = _handler(enabled=True)
    order, market = _limit_order(crossed=True)
    market.volume = 100.0
    handler.active_orders = [order]
    before_orders = deepcopy(handler.active_orders)

    with pytest.raises(RuntimeError, match="handler pricing sink failed"):
        handler.check_open_orders(market)

    assert events.empty()
    assert handler.active_orders == before_orders
    assert handler.no_fill_attempt_evidence == ()


def test_trace_canonical_hash_is_structural_strict_and_immutable():
    traces: list[ExecutionPricingTrace] = []
    model = ExecutionModel(_model_config())
    model.compute_fill(
        raw_price=100.0,
        qty=1.0,
        direction="BUY",
        bar_volume=100.0,
        order_id="CANON-1",
        attribution_sink=traces.append,
    )
    trace = traces[0]
    payload = trace.to_payload()
    expected = execution_pricing_trace_sha256(trace)

    assert trace.sha256 == expected
    assert len(expected) == 64
    payload["fill_price"] = 0.0
    assert trace.fill_price != 0.0
    assert trace.sha256 == expected
    assert replace(trace).sha256 == expected
    assert replace(trace, order_kind="STOP").sha256 != expected
    with pytest.raises(ValueError, match="nonfinite"):
        replace(trace, fill_price=float("nan")).to_payload()
    with pytest.raises(TypeError, match="unsupported"):
        replace(trace, order_id=object()).canonical_json_bytes()


def test_no_fill_state_restore_is_exact_nonduplicating_and_rng_neutral():
    full, full_events = _handler(enabled=True)
    full.active_orders = [_mkt_order(quantity=2.0)]
    full.check_open_orders(_market(volume=0.0))
    checkpoint = deepcopy(full.get_state())

    restored, restored_events = _handler(enabled=True)
    restored.set_state(checkpoint)
    restored.set_state(checkpoint)
    assert restored.no_fill_attempt_evidence == full.no_fill_attempt_evidence
    assert len(restored.no_fill_attempt_evidence) == 1
    assert restored.get_state() == checkpoint

    later = _market(timestamp=2, volume=100.0)
    full.check_open_orders(later)
    restored.check_open_orders(later)
    full_fill = full_events.get_nowait()
    restored_fill = restored_events.get_nowait()
    assert _core_fill_values(full_fill) == _core_fill_values(restored_fill)
    assert (
        full_fill.metadata["cost_attribution"].sha256
        == restored_fill.metadata["cost_attribution"].sha256
    )
    assert full.execution_model._rng.getstate() == restored.execution_model._rng.getstate()
    assert full.get_state() == restored.get_state()

    drained = restored.drain_no_fill_attempt_evidence()
    assert drained == full.no_fill_attempt_evidence
    assert restored.drain_no_fill_attempt_evidence() == ()
    assert restored.no_fill_attempt_evidence == ()


def test_no_fill_state_validation_is_atomic_and_disabled_handler_rejects_it():
    handler, _ = _handler(enabled=True)
    handler.active_orders = [_mkt_order(quantity=2.0)]
    handler.check_open_orders(_market(volume=0.0))
    before = deepcopy(handler.get_state())
    corrupt = deepcopy(before)
    corrupt["no_fill_attempt_evidence"][0]["raw_price"] = float("nan")
    corrupt["active_orders"] = []

    with pytest.raises(ValueError, match="no_fill_attempt_invalid"):
        handler.set_state(corrupt)
    assert handler.get_state() == before

    disabled, _ = _handler(enabled=False)
    disabled_before = deepcopy(disabled.get_state())
    with pytest.raises(ValueError, match="requires_attribution"):
        disabled.set_state(before)
    assert disabled.get_state() == disabled_before
