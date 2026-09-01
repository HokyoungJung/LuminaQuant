import logging
import math
import os
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import date, datetime
from typing import Any

from lumina_quant.backtesting._config_view import wrapped_runtime_config
from lumina_quant.backtesting.execution_model import (
    ExecutionModel,
    ExecutionModelConfig,
    ExecutionPricingTrace,
    FillResult,
    _config_from_attrs,
)
from lumina_quant.core.events import FillEvent

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NoFillAttempt:
    """Immutable handler evidence for a called calculation that executed nothing."""

    record_type: str
    reason: str
    timeindex: Any
    symbol: str
    direction: str
    requested_qty: float
    executed_qty: float
    unfilled_qty: float
    raw_price: float
    bar_volume: float
    cap_ratio: float | None
    order_id: str | None
    order_kind: str
    client_order_id: str | None
    parent_order_id: str | None
    remainder_of_order_id: str | None
    oco_group: str | None
    trigger_price: float | None
    position_side: str | None
    reduce_only: bool
    is_maker: bool
    rng_consumed: bool

    def to_payload(self) -> dict[str, object]:
        """Return a strict JSON-compatible checkpoint payload."""
        if type(self) is not NoFillAttempt:
            raise TypeError("no_fill_attempt must be an exact NoFillAttempt")
        payload: dict[str, object] = {}
        float_fields = {
            "requested_qty",
            "executed_qty",
            "unfilled_qty",
            "raw_price",
            "bar_volume",
            "cap_ratio",
            "trigger_price",
        }
        bool_fields = {"reduce_only", "is_maker", "rng_consumed"}
        for field in fields(NoFillAttempt):
            value = getattr(self, field.name)
            if field.name in float_fields:
                if value is None and field.name in {"cap_ratio", "trigger_price"}:
                    payload[field.name] = None
                    continue
                if type(value) is not float or not math.isfinite(value):
                    raise ValueError(f"no_fill_attempt_invalid:{field.name}")
                payload[field.name] = value
                continue
            if field.name in bool_fields:
                if type(value) is not bool:
                    raise ValueError(f"no_fill_attempt_invalid:{field.name}")
                payload[field.name] = value
                continue
            if field.name == "timeindex":
                payload[field.name] = _canonical_evidence_timeindex(value)
                continue
            if value is not None and type(value) is not str:
                raise ValueError(f"no_fill_attempt_invalid:{field.name}")
            payload[field.name] = value

        if self.record_type != "no_fill_attempt":
            raise ValueError("no_fill_attempt_record_type")
        if self.executed_qty != 0.0:
            raise ValueError("no_fill_attempt_nonzero_execution")
        if self.requested_qty < 0.0 or self.unfilled_qty < 0.0:
            raise ValueError("no_fill_attempt_quantity_bounds")
        if not math.isclose(
            self.requested_qty,
            self.unfilled_qty,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("no_fill_attempt_quantity_reconciliation")
        if self.rng_consumed is self.is_maker:
            raise ValueError("no_fill_attempt_rng_flag")
        if not self.reason or not self.symbol or not self.direction or not self.order_kind:
            raise ValueError("no_fill_attempt_required_field")
        if self.direction not in {"BUY", "SELL"}:
            raise ValueError("no_fill_attempt_direction")
        if self.order_kind not in {"MKT", "LMT", "STOP", "TAKE_PROFIT", "TRAIL_STOP"}:
            raise ValueError("no_fill_attempt_order_kind")
        if self.is_maker is not (self.order_kind == "LMT"):
            raise ValueError("no_fill_attempt_maker_kind")
        expected_reason = (
            "liquidity_cap_zero_limit"
            if self.order_kind == "LMT"
            else "liquidity_cap_zero_conditional"
            if self.order_kind in {"STOP", "TAKE_PROFIT", "TRAIL_STOP"}
            else "liquidity_cap_zero_market"
        )
        if self.reason != expected_reason:
            raise ValueError("no_fill_attempt_reason")
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> NoFillAttempt:
        """Restore one exact checkpoint record, rejecting partial/extra schemas."""
        if type(payload) is not dict:
            raise TypeError("no_fill_attempt_state_record must be an exact dict")
        expected = tuple(field.name for field in fields(cls))
        if set(payload) != set(expected):
            raise ValueError("no_fill_attempt_state_schema")
        record = cls(**{name: payload[name] for name in expected})
        record.to_payload()
        return record


@dataclass(frozen=True, slots=True)
class CapacityObservation:
    """Immutable positive-request capacity inputs captured at engine queue time."""

    record_type: str
    timeindex: Any
    symbol: str
    requested_qty: float
    raw_price: float
    bar_volume: float
    equity_before: float

    def to_payload(self) -> dict[str, object]:
        if type(self) is not CapacityObservation:
            raise TypeError("capacity_observation must be an exact CapacityObservation")
        if self.record_type != "capacity_observation":
            raise ValueError("capacity_observation_record_type")
        if not self.symbol:
            raise ValueError("capacity_observation_symbol")
        for name in ("requested_qty", "raw_price", "bar_volume", "equity_before"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"capacity_observation_invalid:{name}")
        if self.requested_qty <= 0.0 or self.raw_price <= 0.0 or self.equity_before <= 0.0:
            raise ValueError("capacity_observation_nonpositive")
        if self.bar_volume < 0.0:
            raise ValueError("capacity_observation_negative_volume")
        return {
            "bar_volume": self.bar_volume,
            "equity_before": self.equity_before,
            "raw_price": self.raw_price,
            "record_type": self.record_type,
            "requested_qty": self.requested_qty,
            "symbol": self.symbol,
            "timeindex": _canonical_evidence_timeindex(self.timeindex),
        }

    @classmethod
    def from_payload(cls, payload: object) -> CapacityObservation:
        if type(payload) is not dict or set(payload) != {
            "bar_volume",
            "equity_before",
            "raw_price",
            "record_type",
            "requested_qty",
            "symbol",
            "timeindex",
        }:
            raise ValueError("capacity_observation_state_schema")
        record = cls(**payload)
        record.to_payload()
        return record


def _canonical_evidence_timeindex(value: object) -> str | int | float | None:
    """Normalize event time without repr/default-string serializer fallbacks."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if value is None or type(value) in {str, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError("no_fill_attempt_timeindex_unsupported")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


class ExecutionHandler(ABC):
    """The ExecutionHandler abstract class handles the interaction
    between a set of order objects generated by a Portfolio and
    the ultimate set of Fill objects that actually occur in the
    market.
    """

    @abstractmethod
    def execute_order(self, event: Any) -> None:
        """Takes an Order event and executes it, producing
        a Fill event that gets placed onto the Events queue.
        """
        raise NotImplementedError


class SimulatedExecutionHandler(ExecutionHandler):
    """The simulated execution handler simply converts all order
    objects into their equivalent fill objects automatically
    without latency, slippage or fill-ratio issues.

    It allows a "Trailing Stop" which is simulated by tracking data updates.
    LMT orders are supported with strict-cross fill rules (see ExecutionModel docstring).
    """

    _LOCKED_ATTRIBUTION_SEAMS = frozenset(
        {
            "_attribution_seams_locked",
            "_record_cost_attribution",
            "_pricing_attribution_sink",
        }
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._LOCKED_ATTRIBUTION_SEAMS and getattr(
            self, "_attribution_seams_locked", False
        ):
            raise AttributeError(f"{name} is constructor-bound and cannot be reassigned")
        super().__setattr__(name, value)

    def __init__(
        self,
        events: Any,
        bars: Any,
        config: Any,
        *,
        record_cost_attribution: bool = False,
    ):
        if type(record_cost_attribution) is not bool:
            raise TypeError("record_cost_attribution must be an exact bool")
        self._attribution_seams_locked = False
        self._record_cost_attribution = record_cost_attribution
        self._pending_pricing_trace: ExecutionPricingTrace | None = None
        self._pricing_trace_evidence: list[ExecutionPricingTrace] = []
        self._pricing_attribution_sink = (
            self._capture_pricing_trace if record_cost_attribution else None
        )
        self._no_fill_attempt_evidence: list[NoFillAttempt] = []
        self._capacity_observation_evidence: list[CapacityObservation] = []
        self._capacity_equity_context: float | None = None
        self._attribution_seams_locked = True
        self.events = events
        self.bars = bars
        self.config = config
        self._order_seq = 0
        # 2026-07-03 audit finding B (config-gated, default OFF = legacy): apply
        # the bar-volume liquidity cap to triggered conditional fills too.
        self._conditional_liquidity_cap = self._execution_flag(
            config,
            "APPLY_LIQUIDITY_CAP_TO_CONDITIONAL_FILLS",
            "apply_liquidity_cap_to_conditional_fills",
        )

        # Phase 4 unified cost model — replaces FillModel + LiquidityModel for fills.
        # BacktestConfigView carries ._rt (RuntimeConfig); production path uses from_runtime.
        # Plain mock configs (unit tests) fall back to _config_from_attrs.
        _rt = wrapped_runtime_config(config)
        self.execution_model = ExecutionModel(
            ExecutionModelConfig.from_runtime(_rt)
            if _rt is not None
            else _config_from_attrs(config)
        )
        self.latency_model = LatencyModel(config)

        # Store conditional orders: { order_id: { 'symbol':..., 'type':..., 'trigger_price':..., 'parent_id':...} }
        # For simplicity, just list of order dicts
        self.active_orders: list[dict[str, Any]] = []
        self._active_orders_by_symbol: dict[str, tuple[dict[str, Any], ...]] = {}
        self._active_order_stop_bounds: dict[str, tuple[bool, float | None, float | None]] = {}
        self._active_order_index_list_id = id(self.active_orders)
        self._active_order_index_size = 0

    def _rebuild_active_order_index(self) -> None:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for order in self.active_orders:
            grouped.setdefault(str(order.get("symbol") or ""), []).append(order)
        self._active_orders_by_symbol = {
            symbol: tuple(orders) for symbol, orders in grouped.items()
        }
        stop_bounds: dict[str, tuple[bool, float | None, float | None]] = {}
        for symbol, orders in grouped.items():
            all_stop = True
            sell_stops: list[float] = []
            buy_stops: list[float] = []
            for order in orders:
                if str(order.get("type")) != "STOP" or order.get("stop_price") is None:
                    all_stop = False
                    break
                stop_price = float(order["stop_price"])
                if str(order.get("direction")).upper() == "SELL":
                    sell_stops.append(stop_price)
                elif str(order.get("direction")).upper() == "BUY":
                    buy_stops.append(stop_price)
                else:
                    all_stop = False
                    break
            stop_bounds[symbol] = (
                all_stop,
                max(sell_stops) if sell_stops else None,
                min(buy_stops) if buy_stops else None,
            )
        self._active_order_stop_bounds = stop_bounds
        self._active_order_index_list_id = id(self.active_orders)
        self._active_order_index_size = len(self.active_orders)

    def _active_orders_for_symbol(self, symbol: str) -> tuple[dict[str, Any], ...]:
        if self._active_order_index_list_id != id(
            self.active_orders
        ) or self._active_order_index_size != len(self.active_orders):
            self._rebuild_active_order_index()
        return self._active_orders_by_symbol.get(str(symbol), ())

    @property
    def record_cost_attribution(self) -> bool:
        """Whether constructor-owned execution-cost evidence is enabled."""
        return self._record_cost_attribution

    @property
    def pricing_attribution_sink(self):
        """Exact local sink passed to every canonical fill calculation when ON."""
        return self._pricing_attribution_sink

    @property
    def pricing_trace_evidence(self) -> tuple[ExecutionPricingTrace, ...]:
        """Return an immutable snapshot of every positive execution pricing trace."""
        return tuple(self._pricing_trace_evidence)

    @property
    def no_fill_attempt_evidence(self) -> tuple[NoFillAttempt, ...]:
        """Return an immutable snapshot of zero-execution pricing attempts."""
        return tuple(self._no_fill_attempt_evidence)

    @property
    def capacity_observation_evidence(self) -> tuple[CapacityObservation, ...]:
        """Return all positive-request inputs captured before queued fills apply."""
        return tuple(self._capacity_observation_evidence)

    def set_capacity_equity_context(self, equity_before: object) -> None:
        """Set the current engine-queue equity for one open-order sweep."""
        if not self.record_cost_attribution:
            return
        if type(equity_before) not in {int, float}:
            raise TypeError("capacity_equity_context_invalid")
        parsed = float(equity_before)
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise ValueError("capacity_equity_context_invalid")
        self._capacity_equity_context = parsed

    def clear_capacity_equity_context(self) -> None:
        self._capacity_equity_context = None

    def drain_no_fill_attempt_evidence(self) -> tuple[NoFillAttempt, ...]:
        """Atomically return and clear pending zero-execution evidence."""
        evidence = tuple(self._no_fill_attempt_evidence)
        self._no_fill_attempt_evidence.clear()
        return evidence

    def _capture_pricing_trace(self, trace: ExecutionPricingTrace) -> None:
        if type(trace) is not ExecutionPricingTrace:
            raise TypeError("execution pricing sink received an invalid trace")
        if self._pending_pricing_trace is not None:
            raise RuntimeError("execution pricing sink received more than one trace")
        self._pending_pricing_trace = trace
        self._pricing_trace_evidence.append(trace)

    @staticmethod
    def _remainder_parent_order_id(order: dict[str, Any]) -> str | None:
        order_id = order.get("order_id")
        if order_id is None:
            return None
        normalized = str(order_id)
        return normalized[:-2] if normalized.endswith("-R") else None

    @staticmethod
    def _order_trigger_price(order: dict[str, Any]) -> float | None:
        order_kind = str(order.get("type") or "").upper()
        if order_kind == "LMT":
            value = order.get("limit_price")
        elif order_kind in {"STOP", "TAKE_PROFIT", "TRAIL_STOP"}:
            value = order.get("stop_price")
        else:
            return None
        return float(value) if value is not None else None

    @staticmethod
    def _no_fill_reason(order_kind: str) -> str:
        if order_kind == "LMT":
            return "liquidity_cap_zero_limit"
        if order_kind in {"STOP", "TAKE_PROFIT", "TRAIL_STOP"}:
            return "liquidity_cap_zero_conditional"
        return "liquidity_cap_zero_market"

    def _emit_no_fill_attempt(
        self,
        *,
        event: Any,
        order: dict[str, Any],
        result: FillResult,
        requested_qty: float,
        raw_price: float,
        bar_volume: float,
        is_maker: bool,
        apply_liquidity_cap: bool,
    ) -> None:
        order_kind = str(order.get("type") or "MKT").upper()
        self._no_fill_attempt_evidence.append(
            NoFillAttempt(
                record_type="no_fill_attempt",
                reason=self._no_fill_reason(order_kind),
                timeindex=_canonical_evidence_timeindex(event.time),
                symbol=str(order.get("symbol") or ""),
                direction=str(order.get("direction") or "").upper(),
                requested_qty=float(requested_qty),
                executed_qty=0.0,
                unfilled_qty=float(result.unfilled_qty),
                raw_price=float(raw_price),
                bar_volume=float(bar_volume),
                cap_ratio=float(self.execution_model.cfg.max_bar_volume_ratio)
                if apply_liquidity_cap
                else None,
                order_id=str(order["order_id"]) if order.get("order_id") is not None else None,
                order_kind=order_kind,
                client_order_id=str(order["client_order_id"])
                if order.get("client_order_id") is not None
                else None,
                parent_order_id=str(order["parent_order_id"])
                if order.get("parent_order_id") is not None
                else None,
                remainder_of_order_id=self._remainder_parent_order_id(order),
                oco_group=str(order["oco_group"]) if order.get("oco_group") is not None else None,
                trigger_price=self._order_trigger_price(order),
                position_side=str(order["position_side"])
                if order.get("position_side") is not None
                else None,
                reduce_only=bool(order.get("reduce_only", False)),
                is_maker=bool(is_maker),
                rng_consumed=not bool(is_maker),
            )
        )

    def _compute_fill_for_order(
        self,
        *,
        event: Any,
        order: dict[str, Any],
        raw_price: float,
        qty: float,
        bar_volume: float,
        volatility: float,
        is_maker: bool,
        apply_liquidity_cap: bool,
        order_notional: float | None = None,
    ) -> tuple[FillResult, ExecutionPricingTrace | None]:
        if self._pending_pricing_trace is not None:
            raise RuntimeError("unconsumed execution pricing trace")
        if qty > 0.0 and self.record_cost_attribution:
            if self._capacity_equity_context is None:
                raise RuntimeError("capacity equity context missing")
            observation = CapacityObservation(
                record_type="capacity_observation",
                timeindex=_canonical_evidence_timeindex(event.time),
                symbol=str(order.get("symbol") or ""),
                requested_qty=float(qty),
                raw_price=float(raw_price),
                bar_volume=float(bar_volume),
                equity_before=self._capacity_equity_context,
            )
            observation.to_payload()
            self._capacity_observation_evidence.append(observation)
        result = self.execution_model.compute_fill(
            raw_price=float(raw_price),
            qty=float(qty),
            direction=str(order["direction"]),
            bar_volume=float(bar_volume),
            volatility=float(volatility),
            is_maker=bool(is_maker),
            apply_liquidity_cap=bool(apply_liquidity_cap),
            order_notional=order_notional,
            order_kind=str(order.get("type") or "MKT"),
            trigger_price=self._order_trigger_price(order),
            order_id=str(order["order_id"]) if order.get("order_id") is not None else None,
            client_order_id=str(order["client_order_id"])
            if order.get("client_order_id") is not None
            else None,
            parent_order_id=str(order["parent_order_id"])
            if order.get("parent_order_id") is not None
            else None,
            remainder_of_order_id=self._remainder_parent_order_id(order),
            oco_group=str(order["oco_group"]) if order.get("oco_group") is not None else None,
            attribution_sink=self.pricing_attribution_sink,
        )
        pricing_trace = self._pending_pricing_trace
        self._pending_pricing_trace = None
        if self.record_cost_attribution:
            if result.executed_qty > 0.0 and pricing_trace is None:
                raise RuntimeError("positive fill produced no execution pricing trace")
            if result.executed_qty <= 0.0:
                if pricing_trace is not None:
                    raise RuntimeError("zero execution unexpectedly produced a pricing trace")
                self._emit_no_fill_attempt(
                    event=event,
                    order=order,
                    result=result,
                    requested_qty=float(qty),
                    raw_price=float(raw_price),
                    bar_volume=float(bar_volume),
                    is_maker=bool(is_maker),
                    apply_liquidity_cap=bool(apply_liquidity_cap),
                )
        elif pricing_trace is not None:
            raise RuntimeError("disabled execution attribution captured a pricing trace")
        return result, pricing_trace

    def get_state(self) -> dict[str, Any]:
        state = {
            "active_orders": deepcopy(self.active_orders),
            "order_seq": int(self._order_seq),
            # Phase 4: execution_model._rng drives all fill randomness — must be
            # checkpointed so chunked runs produce the same sequence as a full run.
            "execution_model_rng_state": self.execution_model._rng.getstate(),
        }
        latency_get_state = getattr(self.latency_model, "get_state", None)
        if callable(latency_get_state):
            state["latency_model"] = latency_get_state()
        if self.record_cost_attribution:
            state["no_fill_attempt_evidence"] = [
                record.to_payload() for record in self._no_fill_attempt_evidence
            ]
            state["capacity_observation_evidence"] = [
                record.to_payload() for record in self._capacity_observation_evidence
            ]
        return state

    def set_state(self, state: dict[str, Any] | None) -> None:
        if not isinstance(state, dict):
            return
        evidence_key = "no_fill_attempt_evidence"
        capacity_key = "capacity_observation_evidence"
        if not self.record_cost_attribution and (evidence_key in state or capacity_key in state):
            raise ValueError("no_fill_attempt_state_requires_attribution")
        restored_evidence: list[NoFillAttempt] = []
        if self.record_cost_attribution and evidence_key in state:
            raw_evidence = state[evidence_key]
            if type(raw_evidence) is not list:
                raise TypeError("no_fill_attempt_state_must_be_an_exact_list")
            # Validate the complete evidence batch before mutating handler state.
            restored_evidence = [NoFillAttempt.from_payload(item) for item in raw_evidence]
        restored_capacity: list[CapacityObservation] = []
        if self.record_cost_attribution and capacity_key in state:
            raw_capacity = state[capacity_key]
            if type(raw_capacity) is not list:
                raise TypeError("capacity_observation_state_must_be_an_exact_list")
            restored_capacity = [CapacityObservation.from_payload(item) for item in raw_capacity]

        active_orders = state.get("active_orders")
        if isinstance(active_orders, list):
            self.active_orders = deepcopy(active_orders)
        if "order_seq" in state:
            try:
                self._order_seq = int(state.get("order_seq", 0))
            except Exception:
                pass
        # Restore execution_model rng; gracefully skip if absent (old state dicts).
        em_rng_state = state.get("execution_model_rng_state")
        if em_rng_state is not None:
            try:
                self.execution_model._rng.setstate(em_rng_state)
            except Exception:
                pass
        latency_state = state.get("latency_model")
        latency_set_state = getattr(self.latency_model, "set_state", None)
        if isinstance(latency_state, dict) and callable(latency_set_state):
            latency_set_state(latency_state)
        if self.record_cost_attribution:
            # Replacement, not extension, prevents replay duplication on repeated restore.
            self._no_fill_attempt_evidence = restored_evidence
            self._capacity_observation_evidence = restored_capacity
            self._capacity_equity_context = None

    def _next_order_id(self) -> str:
        self._order_seq += 1
        return f"SIM-{self._order_seq}"

    def _cancel_protective_orders(self, symbol: str, position_side: str | None = None) -> None:
        protected_types = {"STOP", "TAKE_PROFIT"}
        target_side = str(position_side).upper() if position_side else None
        kept: list[dict[str, Any]] = []
        for order in self.active_orders:
            if order.get("symbol") != symbol:
                kept.append(order)
                continue
            if str(order.get("type")) not in protected_types:
                kept.append(order)
                continue
            if not bool(order.get("is_protective", False)):
                kept.append(order)
                continue
            if target_side and str(order.get("position_side") or "").upper() not in {
                target_side,
                "",
            }:
                kept.append(order)
                continue
        self.active_orders = kept

    def _build_protective_orders(
        self,
        order: dict[str, Any],
        *,
        fill_price: float | None = None,
    ) -> list[dict[str, Any]]:
        if bool(order.get("reduce_only", False)):
            return []

        stop_loss = order.get("stop_loss")
        take_profit = order.get("take_profit")
        trailing_percent = order.get("trailing_percent")
        if (
            stop_loss is None
            and take_profit is None
            and (trailing_percent is None or float(trailing_percent) <= 0.0)
        ):
            return []

        position_side = order.get("position_side")
        if not position_side:
            position_side = "LONG" if order.get("direction") == "BUY" else "SHORT"

        exit_direction = "SELL" if order.get("direction") == "BUY" else "BUY"
        oco_group = f"{order.get('order_id')}-BRACKET"
        quantity = float(order.get("quantity") or 0.0)
        if quantity <= 0.0:
            return []

        out: list[dict[str, Any]] = []
        if stop_loss is not None:
            out.append(
                {
                    "order_id": self._next_order_id(),
                    "symbol": order.get("symbol"),
                    "type": "STOP",
                    "quantity": quantity,
                    "direction": exit_direction,
                    "stop_price": float(stop_loss),
                    "position_side": position_side,
                    "reduce_only": True,
                    "client_order_id": f"{order.get('client_order_id')}-SL"
                    if order.get("client_order_id")
                    else None,
                    "is_protective": True,
                    "oco_group": oco_group,
                    "parent_order_id": order.get("order_id"),
                }
            )

        if take_profit is not None:
            out.append(
                {
                    "order_id": self._next_order_id(),
                    "symbol": order.get("symbol"),
                    "type": "TAKE_PROFIT",
                    "quantity": quantity,
                    "direction": exit_direction,
                    "stop_price": float(take_profit),
                    "position_side": position_side,
                    "reduce_only": True,
                    "client_order_id": f"{order.get('client_order_id')}-TP"
                    if order.get("client_order_id")
                    else None,
                    "is_protective": True,
                    "oco_group": oco_group,
                    "parent_order_id": order.get("order_id"),
                }
            )

        if trailing_percent is not None and float(trailing_percent) > 0.0:
            trailing_value = float(trailing_percent)
            initial_price = float(fill_price) if fill_price is not None else None
            trailing_order = {
                "order_id": self._next_order_id(),
                "symbol": order.get("symbol"),
                "type": "TRAIL_STOP",
                "quantity": quantity,
                "direction": exit_direction,
                "position_side": position_side,
                "reduce_only": True,
                "client_order_id": f"{order.get('client_order_id')}-TRAIL"
                if order.get("client_order_id")
                else None,
                "is_protective": True,
                "oco_group": oco_group,
                "parent_order_id": order.get("order_id"),
                "trailing_percent": trailing_value,
                "stop_price": None,
                "highest_price": None,
                "lowest_price": None,
            }
            if exit_direction == "SELL" and initial_price is not None:
                trailing_order["highest_price"] = initial_price
                trailing_order["stop_price"] = initial_price * (1.0 - trailing_value)
            elif exit_direction == "BUY" and initial_price is not None:
                trailing_order["lowest_price"] = initial_price
                trailing_order["stop_price"] = initial_price * (1.0 + trailing_value)
            out.append(trailing_order)

        return out

    def execute_order(self, event: Any) -> None:
        """Receives OrderEvent.
        - MKT: Queues for Next Open execution (realism).
        - LMT: Queues for strict-cross fill on next bar.
        - STOP/TRAIL: triggers active monitoring.
        """
        if event.type == "ORDER":
            order_id = self._next_order_id()

            if event.order_type == "MKT":
                if bool(getattr(event, "reduce_only", False)):
                    self._cancel_protective_orders(
                        event.symbol,
                        getattr(event, "position_side", None),
                    )
                # LATENCY SIMULATION:
                # Do NOT fill immediately. Queue for Next Open.
                self.active_orders.append(
                    {
                        "order_id": order_id,
                        "symbol": event.symbol,
                        "type": "MKT",
                        "quantity": event.quantity,
                        "direction": event.direction,
                        "status": "PENDING",
                        "position_side": event.position_side,
                        "reduce_only": event.reduce_only,
                        "client_order_id": event.client_order_id,
                        "stop_loss": event.stop_loss,
                        "take_profit": event.take_profit,
                        "trailing_percent": event.trailing_percent,
                    }
                )

            elif event.order_type == "LMT":
                # LMT order: fills when bar strictly crosses limit_price.
                # BUY fills when bar_low  < limit_price (strict — not ≤).
                # SELL fills when bar_high > limit_price (strict — not ≥).
                limit_price = getattr(event, "price", None)
                if bool(getattr(event, "reduce_only", False)):
                    self._cancel_protective_orders(
                        event.symbol,
                        getattr(event, "position_side", None),
                    )
                self.active_orders.append(
                    {
                        "order_id": order_id,
                        "symbol": event.symbol,
                        "type": "LMT",
                        "quantity": event.quantity,
                        "direction": event.direction,
                        "limit_price": float(limit_price) if limit_price is not None else None,
                        "status": "PENDING",
                        "position_side": event.position_side,
                        "reduce_only": event.reduce_only,
                        "client_order_id": event.client_order_id,
                        "stop_loss": event.stop_loss,
                        "take_profit": event.take_profit,
                        "trailing_percent": event.trailing_percent,
                    }
                )

            elif event.order_type == "STOP":
                # Add to active orders
                self.active_orders.append(
                    {
                        "order_id": order_id,
                        "symbol": event.symbol,
                        "type": "STOP",
                        "quantity": event.quantity,
                        "direction": event.direction,
                        "stop_price": event.stop_price,
                        "position_side": event.position_side,
                        "reduce_only": event.reduce_only,
                        "client_order_id": event.client_order_id,
                    }
                )

            elif event.order_type == "TRAIL_STOP":
                # Add to active orders
                curr_price = self.bars.get_latest_bar_value(event.symbol, "close")
                # Initial stop price (if sent, else calculate)
                stop_price = event.stop_price

                self.active_orders.append(
                    {
                        "order_id": order_id,
                        "symbol": event.symbol,
                        "type": "TRAIL_STOP",
                        "quantity": event.quantity,
                        "direction": event.direction,
                        "stop_price": stop_price,
                        "trailing_percent": event.trailing_percent,
                        "highest_price": curr_price
                        if event.direction == "SELL"
                        else None,  # For Long Exit Trailing Stop
                        "lowest_price": curr_price
                        if event.direction == "BUY"
                        else None,  # For Short Exit Trailing Stop
                        "position_side": event.position_side,
                        "reduce_only": event.reduce_only,
                        "client_order_id": event.client_order_id,
                    }
                )

    @staticmethod
    def _execution_flag(config: Any, upper_attr: str, runtime_field: str) -> bool:
        """Resolve an execution flag from UPPERCASE attr or RuntimeConfig dotpath."""
        value = getattr(config, upper_attr, None)
        if value is not None:
            return bool(value)
        runtime = wrapped_runtime_config(config)
        execution = getattr(runtime, "execution", None) if runtime is not None else None
        if execution is not None:
            return bool(getattr(execution, runtime_field, False))
        return False

    def check_open_orders(self, event: Any) -> None:
        """Check active orders against the new MarketEvent.
        Handles MKT (Next Open), LMT (strict-cross), STOP/TP, TRAIL_STOP.
        """
        if event.type != "MARKET" or not self.active_orders:
            return
        symbol_orders = self._active_orders_for_symbol(str(event.symbol))
        if not symbol_orders:
            return
        original_active_orders = self.active_orders

        bar_open = event.open
        bar_high = event.high
        bar_low = event.low
        bar_volume = event.volume
        all_stop, highest_sell_stop, lowest_buy_stop = self._active_order_stop_bounds.get(
            str(event.symbol),
            (False, None, None),
        )
        if (
            all_stop
            and (highest_sell_stop is None or bar_low > highest_sell_stop)
            and (lowest_buy_stop is None or bar_high < lowest_buy_stop)
        ):
            return
        # Normalised bar range — used to scale slippage on volatile bars.
        volatility = (bar_high - bar_low) / bar_open if bar_open > 0 else 0.0

        next_active_orders: list[dict[str, Any]] = []
        remainder_orders: list[dict[str, Any]] = []
        closed_oco_groups: set[str] = set()
        closed_positions: set[tuple[str, str | None]] = set()

        for order in symbol_orders:
            oco_group = order.get("oco_group")
            if oco_group and str(oco_group) in closed_oco_groups:
                continue

            triggered = False
            exec_price = None
            # Pre-computed fill result for MKT and LMT (computed inside their branches
            # so we don't double-consume the rng in the unified triggered block).
            _fill_result = None
            _pricing_trace = None

            # ── MKT ORDER (Next Open) ─────────────────────────────────────────
            if order["type"] == "MKT" and order["status"] == "PENDING":
                if not self.latency_model.should_release(order):
                    next_active_orders.append(order)
                    continue
                exec_price = bar_open
                triggered = True

                original_qty = order["quantity"]
                _fill_result, _pricing_trace = self._compute_fill_for_order(
                    event=event,
                    order=order,
                    raw_price=float(exec_price),
                    qty=float(original_qty),
                    bar_volume=float(bar_volume),
                    volatility=float(volatility),
                    is_maker=False,
                    apply_liquidity_cap=True,
                    order_notional=float(original_qty) * float(exec_price),
                )

                if _fill_result.unfilled_qty > 0.0:
                    if not _env_flag("LQ_BACKTEST_SUPPRESS_PARTIAL_FILL_LOGS", False):
                        LOGGER.debug(
                            "[Realism] Partial Fill: Req %s > Limit %.4f. Filling %s "
                            "and keeping remainder.",
                            original_qty,
                            _fill_result.executed_qty,
                            _fill_result.executed_qty,
                        )
                    remainder_orders.append(
                        {
                            "order_id": f"{order['order_id']}-R",
                            "symbol": order["symbol"],
                            "type": "MKT",
                            "quantity": _fill_result.unfilled_qty,
                            "direction": order["direction"],
                            "status": "PENDING",
                            "position_side": order.get("position_side"),
                            "reduce_only": order.get("reduce_only", False),
                            "client_order_id": order.get("client_order_id"),
                            "stop_loss": order.get("stop_loss"),
                            "take_profit": order.get("take_profit"),
                            "trailing_percent": order.get("trailing_percent"),
                        }
                    )
                order["quantity"] = _fill_result.executed_qty

            # ── LMT ORDER (strict-cross fill) ─────────────────────────────────
            elif order["type"] == "LMT" and order.get("status") == "PENDING":
                limit_price = order.get("limit_price")
                if limit_price is None or float(limit_price) <= 0.0:
                    next_active_orders.append(order)
                    continue
                limit_price = float(limit_price)
                direction = str(order["direction"]).upper()
                # Strict cross: BUY fills when bar_low < limit (not ≤).
                #               SELL fills when bar_high > limit (not ≥).
                if (direction == "BUY" and bar_low < limit_price) or (
                    direction == "SELL" and bar_high > limit_price
                ):
                    exec_price = limit_price
                    triggered = True

                if triggered:
                    original_qty = order["quantity"]
                    _fill_result, _pricing_trace = self._compute_fill_for_order(
                        event=event,
                        order=order,
                        raw_price=float(exec_price),
                        qty=float(original_qty),
                        bar_volume=float(bar_volume),
                        volatility=0.0,  # LMT fills at exact price — no slippage
                        is_maker=True,
                        apply_liquidity_cap=True,
                    )
                    if _fill_result.unfilled_qty > 0.0:
                        remainder_orders.append(
                            {
                                "order_id": f"{order['order_id']}-R",
                                "symbol": order["symbol"],
                                "type": "LMT",
                                "quantity": _fill_result.unfilled_qty,
                                "direction": order["direction"],
                                "limit_price": limit_price,
                                "status": "PENDING",
                                "position_side": order.get("position_side"),
                                "reduce_only": order.get("reduce_only", False),
                                "client_order_id": order.get("client_order_id"),
                                "stop_loss": order.get("stop_loss"),
                                "take_profit": order.get("take_profit"),
                                "trailing_percent": order.get("trailing_percent"),
                            }
                        )
                    order["quantity"] = _fill_result.executed_qty

            # ── STOP ORDER ────────────────────────────────────────────────────
            elif order["type"] == "STOP":
                if order["direction"] == "SELL" and bar_low <= order["stop_price"]:
                    exec_price = order["stop_price"]
                    if exec_price > bar_open:
                        exec_price = bar_open
                    triggered = True
                elif order["direction"] == "BUY" and bar_high >= order["stop_price"]:
                    exec_price = order["stop_price"]
                    if exec_price < bar_open:
                        exec_price = bar_open
                    triggered = True

            # ── TAKE PROFIT ORDER ─────────────────────────────────────────────
            elif order["type"] == "TAKE_PROFIT":
                target = float(order["stop_price"])
                if order["direction"] == "SELL" and bar_high >= target:
                    exec_price = target
                    if bar_open > exec_price:
                        exec_price = bar_open
                    triggered = True
                elif order["direction"] == "BUY" and bar_low <= target:
                    exec_price = target
                    if bar_open < exec_price:
                        exec_price = bar_open
                    triggered = True

            # ── TRAILING STOP ─────────────────────────────────────────────────
            # Conservative intra-bar sequencing: test the trigger against the
            # stop level as it stood BEFORE this bar, THEN ratchet using this
            # bar's favorable extreme. Ratcheting first (the old behaviour)
            # assumes the favorable extreme always precedes the adverse one
            # within the bar — best-case optimism that overstates trailing-stop
            # exits by up to the bar range. Only a non-triggering bar advances
            # the trail for future bars.
            elif order["type"] == "TRAIL_STOP":
                if order["direction"] == "SELL":
                    prev_stop = order["stop_price"]
                    if prev_stop is not None and bar_low <= prev_stop:
                        exec_price = prev_stop
                        if bar_open < exec_price:
                            exec_price = bar_open
                        triggered = True
                    elif order["highest_price"] is None or bar_high > order["highest_price"]:
                        order["highest_price"] = bar_high
                        order["stop_price"] = order["highest_price"] * (
                            1.0 - order["trailing_percent"]
                        )
                elif order["direction"] == "BUY":
                    prev_stop = order["stop_price"]
                    if prev_stop is not None and bar_high >= prev_stop:
                        exec_price = prev_stop
                        if bar_open > exec_price:
                            exec_price = bar_open
                        triggered = True
                    elif order["lowest_price"] is None or bar_low < order["lowest_price"]:
                        order["lowest_price"] = bar_low
                        order["stop_price"] = order["lowest_price"] * (
                            1.0 + order["trailing_percent"]
                        )

            # ── Unified fill emission ──────────────────────────────────────────
            if triggered and exec_price is not None:
                if _fill_result is not None:
                    # MKT or LMT — fill already computed in the branch above.
                    # N5: nothing executed this bar (e.g. a zero-volume trigger bar
                    # under the always-on liquidity cap). The remainder chase, if any,
                    # was already queued in the branch above; skip the zero-qty
                    # FillEvent + protective/OCO bookkeeping and drop the spent order.
                    if _fill_result.executed_qty <= 0.0:
                        continue
                    fill_price = _fill_result.fill_price
                    comm = _fill_result.commission
                else:
                    # STOP, TAKE_PROFIT, TRAIL_STOP — legacy: no liquidity cap
                    # (aggressive fill). With apply_liquidity_cap_to_conditional_fills
                    # the cap applies and the excess chases as a MKT remainder.
                    _cond_qty = float(order["quantity"])
                    cond_result, _pricing_trace = self._compute_fill_for_order(
                        event=event,
                        order=order,
                        raw_price=float(exec_price),
                        qty=_cond_qty,
                        bar_volume=float(bar_volume),
                        volatility=float(volatility),
                        is_maker=False,
                        apply_liquidity_cap=self._conditional_liquidity_cap,
                        order_notional=_cond_qty * float(exec_price),
                    )
                    fill_price = cond_result.fill_price
                    comm = cond_result.commission
                    if self._conditional_liquidity_cap and cond_result.unfilled_qty > 0.0:
                        if not _env_flag("LQ_BACKTEST_SUPPRESS_PARTIAL_FILL_LOGS", False):
                            LOGGER.debug(
                                "[Realism] Partial Conditional Fill: Req %s > Limit "
                                "%.4f. Filling %s and chasing remainder as MKT.",
                                _cond_qty,
                                cond_result.executed_qty,
                                cond_result.executed_qty,
                            )
                        remainder_orders.append(
                            {
                                "order_id": f"{order['order_id']}-R",
                                "symbol": order["symbol"],
                                "type": "MKT",
                                "quantity": cond_result.unfilled_qty,
                                "direction": order["direction"],
                                "status": "PENDING",
                                "position_side": order.get("position_side"),
                                "reduce_only": order.get("reduce_only", False),
                                "client_order_id": order.get("client_order_id"),
                                "stop_loss": None,
                                "take_profit": None,
                                "trailing_percent": None,
                            }
                        )
                        order["quantity"] = cond_result.executed_qty

                    # N5: nothing executed on this trigger bar (e.g. a zero-volume
                    # bar under the conditional liquidity cap). Keep only the
                    # remainder chase queued above and skip the zero-qty FillEvent
                    # + OCO/protective teardown so live protection isn't dismantled
                    # on a bar where the exit filled nothing.
                    if cond_result.executed_qty <= 0.0:
                        continue

                fill_metadata = {
                    "reduce_only": order.get("reduce_only", False),
                    "signal_metadata": dict(order.get("metadata") or {}),
                    "component_id": str(
                        dict(order.get("metadata") or {}).get("component_id") or ""
                    ).strip()
                    or None,
                }
                if self.record_cost_attribution:
                    if _pricing_trace is None:
                        raise RuntimeError("positive FillEvent has no execution pricing trace")
                    fill_metadata["cost_attribution"] = _pricing_trace

                fill_event = FillEvent(
                    timeindex=event.time,
                    symbol=order["symbol"],
                    exchange="BINANCE_SIM",
                    quantity=order["quantity"],
                    direction=order["direction"],
                    fill_cost=fill_price * order["quantity"],
                    commission=comm,
                    order_id=order.get("order_id"),
                    client_order_id=order.get("client_order_id"),
                    position_side=order.get("position_side"),
                    status="FILLED",
                    metadata=fill_metadata,
                )
                self.events.put(fill_event)

                if order.get("type") in {"MKT", "LMT"}:
                    remainder_orders.extend(
                        self._build_protective_orders(order, fill_price=fill_price)
                    )

                if bool(order.get("reduce_only", False)):
                    closed_positions.add(
                        (
                            str(order.get("symbol")),
                            str(order.get("position_side")).upper()
                            if order.get("position_side")
                            else None,
                        )
                    )

                if oco_group:
                    closed_oco_groups.add(str(oco_group))
                continue

            if oco_group and str(oco_group) in closed_oco_groups:
                continue
            next_active_orders.append(order)

        if closed_positions:
            filtered: list[dict[str, Any]] = []
            for order in next_active_orders:
                if str(order.get("type")) not in {"STOP", "TAKE_PROFIT"}:
                    filtered.append(order)
                    continue
                if not bool(order.get("is_protective", False)):
                    filtered.append(order)
                    continue
                symbol = str(order.get("symbol"))
                side = (
                    str(order.get("position_side")).upper() if order.get("position_side") else None
                )
                matched = False
                for c_symbol, c_side in closed_positions:
                    if symbol != c_symbol:
                        continue
                    if c_side is None or side is None or side == c_side:
                        matched = True
                        break
                if not matched:
                    filtered.append(order)
            next_active_orders = filtered

        if remainder_orders:
            next_active_orders.extend(remainder_orders)
        membership_unchanged = len(next_active_orders) == len(symbol_orders) and all(
            current is original
            for current, original in zip(next_active_orders, symbol_orders, strict=True)
        )
        if membership_unchanged:
            return
        original_symbol_order_ids = {id(order) for order in symbol_orders}
        surviving_order_ids = {id(order) for order in next_active_orders}
        rebuilt = [
            order
            for order in original_active_orders
            if id(order) not in original_symbol_order_ids or id(order) in surviving_order_ids
        ]
        rebuilt.extend(
            order for order in next_active_orders if id(order) not in original_symbol_order_ids
        )
        self.active_orders = rebuilt
        self._rebuild_active_order_index()


class LatencyModel:
    """Simple latency model releasing queued orders on next check cycle."""

    def __init__(self, config: Any):
        seed = int(getattr(config, "RANDOM_SEED", 42))
        self.rng = random.Random(seed + 701)
        self.min_bars = max(1, int(getattr(config, "SIM_LATENCY_MIN_BARS", 1)))
        self.max_bars = max(self.min_bars, int(getattr(config, "SIM_LATENCY_MAX_BARS", 1)))

    def get_state(self) -> dict[str, Any]:
        return {"rng_state": self.rng.getstate()}

    def set_state(self, state: dict[str, Any] | None) -> None:
        if not isinstance(state, dict):
            return
        rng_state = state.get("rng_state")
        if rng_state is None:
            return
        try:
            self.rng.setstate(rng_state)
        except Exception:
            pass

    def should_release(self, order: dict[str, Any]) -> bool:
        target = order.get("_latency_target_bars")
        if target is None:
            target = int(self.rng.randint(int(self.min_bars), int(self.max_bars)))
            order["_latency_target_bars"] = int(target)
        waited = int(order.get("_latency_waited_bars", 0)) + 1
        order["_latency_waited_bars"] = int(waited)
        return int(waited) >= int(target)


__all__ = [
    "ExecutionHandler",
    "LatencyModel",
    "NoFillAttempt",
    "SimulatedExecutionHandler",
]
