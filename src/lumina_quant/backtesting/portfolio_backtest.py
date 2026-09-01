import json
import logging
import math
import os
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
from lumina_quant.backtesting._config_view import wrapped_runtime_config
from lumina_quant.backtesting.execution_model import (
    ExecutionModel,
    ExecutionModelConfig,
    ExecutionPricingTrace,
    _config_from_attrs,
    execution_pricing_trace_sha256,
)
from lumina_quant.core.events import FillEvent, OrderEvent, SignalEvent
from lumina_quant.core.order_policy import (
    canonical_order_type,
    limit_price_for_direction,
    merge_order_policy_metadata,
    normalize_limit_price_mode,
    policy_order_type,
    price_tick_size_from_sources,
)
from lumina_quant.data.feature_points import BINANCE_FUNDING_SOURCE_JITTER_TOLERANCE_MS
from lumina_quant.market_data import normalize_timeframe_token, timeframe_to_milliseconds
from lumina_quant.risk_manager import RiskManager
from lumina_quant.portfolio.strategy_quality import StrategyQualityOverlay
from lumina_quant.services.portfolio import PortfolioPerformanceService, PortfolioSizingService

LOGGER = logging.getLogger(__name__)


def _canonical_attribution_timeindex(value) -> str | int | float | None:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if value is None or type(value) in {str, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError("fill_application_timeindex_unsupported")


@dataclass(frozen=True, slots=True)
class FillApplicationAttribution(Mapping[str, object]):
    """Immutable reconciliation of one real pricing trace into portfolio state."""

    record_type: str
    pricing_trace_hash: str
    pricing_trace: ExecutionPricingTrace
    timeindex: str | int | float | None
    symbol: str
    direction: str
    order_id: str | None
    client_order_id: str | None
    position_side: str | None
    status: str | None
    reduce_only: bool
    model_quantity: float
    model_fill_cost: float
    model_commission: float
    applied_quantity: float
    applied_fill_cost: float
    applied_commission: float
    reduce_only_scale: float
    application_status: str
    zero_applied_reason: str | None

    def to_payload(self) -> dict[str, object]:
        """Return strict structured JSON evidence; never repr/stringify the trace."""
        if type(self) is not FillApplicationAttribution:
            raise TypeError("fill_application must be an exact FillApplicationAttribution")
        if self.record_type != "fill_application_attribution":
            raise ValueError("fill_application_record_type")
        if self.pricing_trace_hash != execution_pricing_trace_sha256(self.pricing_trace):
            raise ValueError("fill_application_pricing_trace_hash_mismatch")
        if self.application_status not in {"applied_unchanged", "applied_scaled", "rejected"}:
            raise ValueError("fill_application_status")
        if self.application_status == "rejected" and not self.zero_applied_reason:
            raise ValueError("fill_application_rejection_reason")
        if self.application_status != "rejected" and self.zero_applied_reason is not None:
            raise ValueError("fill_application_unexpected_rejection_reason")
        if type(self.reduce_only) is not bool:
            raise TypeError("fill_application_reduce_only")
        numeric_fields = (
            "model_quantity",
            "model_fill_cost",
            "model_commission",
            "applied_quantity",
            "applied_fill_cost",
            "applied_commission",
            "reduce_only_scale",
        )
        for name in numeric_fields:
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"fill_application_nonfinite:{name}")
        if not math.isclose(
            self.model_quantity,
            self.pricing_trace.executed_qty,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("fill_application_model_quantity_mismatch")
        if not math.isclose(
            self.model_fill_cost,
            self.pricing_trace.fill_price * self.pricing_trace.executed_qty,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("fill_application_model_cost_mismatch")
        if not math.isclose(
            self.model_commission,
            self.pricing_trace.commission,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("fill_application_model_commission_mismatch")
        if self.application_status == "applied_unchanged":
            if self.reduce_only_scale != 1.0 or any(
                not math.isclose(applied, model, rel_tol=0.0, abs_tol=1e-12)
                for applied, model in (
                    (self.applied_quantity, self.model_quantity),
                    (self.applied_fill_cost, self.model_fill_cost),
                    (self.applied_commission, self.model_commission),
                )
            ):
                raise ValueError("fill_application_unchanged_mismatch")
        elif self.application_status == "applied_scaled":
            if not 0.0 < self.reduce_only_scale < 1.0:
                raise ValueError("fill_application_scale_bounds")
            if any(
                not math.isclose(
                    applied,
                    model * self.reduce_only_scale,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for applied, model in (
                    (self.applied_quantity, self.model_quantity),
                    (self.applied_fill_cost, self.model_fill_cost),
                    (self.applied_commission, self.model_commission),
                )
            ):
                raise ValueError("fill_application_scaled_mismatch")
        elif (
            self.reduce_only_scale != 0.0
            or self.applied_quantity != 0.0
            or self.applied_fill_cost != 0.0
            or self.applied_commission != 0.0
        ):
            raise ValueError("fill_application_rejection_nonzero")
        for name in ("symbol", "direction"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"fill_application_required:{name}")
        for name in ("order_id", "client_order_id", "position_side", "status"):
            value = getattr(self, name)
            if value is not None and type(value) is not str:
                raise TypeError(f"fill_application_invalid:{name}")

        return {
            "record_type": self.record_type,
            "pricing_trace_hash": self.pricing_trace_hash,
            "pricing_trace": self.pricing_trace.to_payload(),
            "timeindex": _canonical_attribution_timeindex(self.timeindex),
            "symbol": self.symbol,
            "direction": self.direction,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "position_side": self.position_side,
            "status": self.status,
            "reduce_only": self.reduce_only,
            "model_quantity": self.model_quantity,
            "model_fill_cost": self.model_fill_cost,
            "model_commission": self.model_commission,
            "applied_quantity": self.applied_quantity,
            "applied_fill_cost": self.applied_fill_cost,
            "applied_commission": self.applied_commission,
            "reduce_only_scale": self.reduce_only_scale,
            "application_status": self.application_status,
            "zero_applied_reason": self.zero_applied_reason,
        }

    def canonical_json_bytes(self) -> bytes:
        return json.dumps(
            self.to_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def __getitem__(self, key: str) -> object:
        return self.to_payload()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_payload())

    def __len__(self) -> int:
        return len(self.to_payload())


class Portfolio:
    """The Portfolio class handles the positions and market value.
    Refactored to use Polars for equity curve storage.
    """

    _LOCKED_OPTIONAL_SEAM_ATTRS = frozenset(
        {
            "_fill_application_attribution_sink",
            "_funding_boundary_resolver",
            "_full_event_equity_sink",
            "_reporting_sampling_timeframe",
        }
    )

    def __setattr__(self, name, value):
        if name in self._LOCKED_OPTIONAL_SEAM_ATTRS and getattr(
            self, "_optional_seams_locked", False
        ):
            raise AttributeError(f"{name} is constructor-bound and cannot be reassigned")
        super().__setattr__(name, value)

    def __init__(
        self,
        bars,
        events,
        start_date,
        config,
        record_history=True,
        track_metrics=True,
        record_trades=True,
        sampling_timeframe=None,
        *,
        fill_application_attribution_sink=None,
        funding_boundary_resolver=None,
        full_event_equity_sink=None,
        reporting_sampling_timeframe=None,
    ):
        self.bars = bars
        self.events = events
        self.config = config
        if fill_application_attribution_sink is not None and not callable(
            fill_application_attribution_sink
        ):
            raise TypeError("fill_application_attribution_sink must be callable")
        if funding_boundary_resolver is not None and not any(
            callable(getattr(funding_boundary_resolver, name, None))
            for name in ("resolve_batch", "resolve")
        ):
            raise TypeError(
                "funding_boundary_resolver must expose callable resolve_batch or resolve"
            )
        if full_event_equity_sink is not None and not callable(full_event_equity_sink):
            raise TypeError("full_event_equity_sink must be callable")
        normalized_reporting_sampling_timeframe = None
        if reporting_sampling_timeframe is not None:
            try:
                normalized_reporting_sampling_timeframe = normalize_timeframe_token(
                    reporting_sampling_timeframe
                )
                reporting_interval_ms = int(
                    timeframe_to_milliseconds(normalized_reporting_sampling_timeframe)
                )
                if reporting_interval_ms <= 0:
                    raise ValueError("nonpositive reporting interval")
            except Exception as exc:
                raise ValueError("reporting_sampling_timeframe_invalid") from exc
        self._fill_application_attribution_sink = fill_application_attribution_sink
        self._funding_boundary_resolver = funding_boundary_resolver
        self._full_event_equity_sink = full_event_equity_sink
        self._reporting_sampling_timeframe = normalized_reporting_sampling_timeframe
        self._optional_seams_locked = True
        self.symbol_list = self.bars.symbol_list
        self._single_symbol = len(self.symbol_list) == 1
        self.record_history = bool(record_history)
        self.track_metrics = bool(track_metrics)
        self.record_trades = bool(record_trades)
        self.sampling_timeframe = None
        if sampling_timeframe:
            try:
                self.sampling_timeframe = normalize_timeframe_token(sampling_timeframe)
            except Exception:
                self.sampling_timeframe = None
        self._sampling_interval_ms = None
        if self.sampling_timeframe:
            try:
                effective_sampling_timeframe = (
                    self._reporting_sampling_timeframe or self.sampling_timeframe
                )
                self._sampling_interval_ms = int(
                    timeframe_to_milliseconds(effective_sampling_timeframe)
                )
            except Exception:
                self._sampling_interval_ms = None
        elif self._reporting_sampling_timeframe:
            self._sampling_interval_ms = int(
                timeframe_to_milliseconds(self._reporting_sampling_timeframe)
            )
        self._last_sample_timestamp_ms = None
        self.start_date = start_date
        self.initial_capital = self.config.INITIAL_CAPITAL

        self.all_positions = []
        self.current_positions = dict.fromkeys(self.symbol_list, 0.0)

        self.all_holdings = []
        self.current_holdings = self.construct_current_holdings()

        # Trade Log (for Visualization)
        self.trades = []
        self.trade_count = 0

        # Circuit Breaker (Safety)
        self.circuit_breaker_tripped = False
        self.day_start_equity = self.initial_capital
        self.max_daily_loss_pct = getattr(config, "MAX_DAILY_LOSS_PCT", 0.05)  # 5% default
        self.risk_per_trade = getattr(config, "RISK_PER_TRADE", 0.005)
        self.max_symbol_exposure_pct = getattr(config, "MAX_SYMBOL_EXPOSURE_PCT", 0.25)
        self.max_order_value = getattr(config, "MAX_ORDER_VALUE", 5000.0)
        self.max_order_notional_pct = getattr(config, "MAX_ORDER_NOTIONAL_PCT", 0.0)
        self.target_allocation_mode = getattr(
            config,
            "TARGET_ALLOCATION_MODE",
            "legacy_notional_cap",
        )
        self.leverage = getattr(config, "LEVERAGE", 1.0)
        self.default_stop_loss_pct = getattr(config, "DEFAULT_STOP_LOSS_PCT", 0.01)
        # Audit-hardening (fix/audit-hardening) backtest-path risk gates. Each flag
        # defaults False so the golden baseline stays byte-identical; reading prefers
        # an uppercase attr (plain-class unit-test configs) and falls back to the
        # RuntimeConfig dotpath carried on BacktestConfigView._rt for production runs.
        self.enforce_order_risk_gate = self._audit_flag(
            config,
            "ENFORCE_ORDER_RISK_GATE_IN_BACKTEST",
            "risk",
            "enforce_order_risk_gate_in_backtest",
        )
        self.attach_default_protective_stop = self._audit_flag(
            config, "ATTACH_DEFAULT_PROTECTIVE_STOP", "risk", "attach_default_protective_stop"
        )
        self.require_funding_coverage = self._audit_flag(
            config, "REQUIRE_FUNDING_COVERAGE", "execution", "require_funding_coverage"
        )
        self.enforce_reduce_only = self._audit_flag(
            config, "ENFORCE_REDUCE_ONLY", "execution", "enforce_reduce_only"
        )
        # Report defect #8 (funding timing). When True, funding is charged on
        # CROSSED wall-clock 00/08/16 UTC boundaries instead of the entry-anchored
        # 8h clock, so a sub-8h round trip that straddles a boundary pays one
        # funding event. Default False keeps the entry-anchored charging exactly
        # byte-identical (this is a backtest-numerics change, so it must be gated).
        self.funding_on_utc_boundary = self._audit_flag(
            config, "FUNDING_ON_UTC_BOUNDARY", "execution", "funding_on_utc_boundary"
        )
        # L-D funding-entry guard (pre-registered fixed rule, default OFF):
        # skip a new entry whose declared intended hold is shorter than one
        # funding interval AND would straddle the next settlement boundary.
        self.funding_entry_guard = self._audit_flag(
            config, "FUNDING_ENTRY_GUARD", "execution", "funding_entry_guard"
        )
        # Lazily-constructed RiskManager backstop for the order-time gate (only when
        # enforce_order_risk_gate is True). Mirrors the live/trader.py:1713 usage.
        self._risk_manager = None
        # Phase 4 unified cost model — funding and liquidation delegate to this.
        # BacktestConfigView carries ._rt (RuntimeConfig); use from_runtime for production.
        # Plain class configs (unit tests) use _config_from_attrs.
        _rt = wrapped_runtime_config(config)
        self.execution_model = ExecutionModel(
            ExecutionModelConfig.from_runtime(_rt)
            if _rt is not None
            else _config_from_attrs(config)
        )
        self._current_day = None
        self._last_funding_ts = dict.fromkeys(self.symbol_list)
        # Settlement evidence and position exposure advance independently. The
        # latter is sampled before every position-changing fill so delayed
        # evidence cannot reinterpret a past boundary at a later size.
        self._funding_exposure_cursor = dict.fromkeys(self.symbol_list)
        # UTC funding evidence can publish shortly after its nominal boundary.
        # Preserve the boundary exposure before same-bar fills mutate it.
        self._pending_funding_liabilities = {symbol: {} for symbol in self.symbol_list}
        self.total_funding_paid = 0.0
        self.entry_prices = dict.fromkeys(self.symbol_list)
        self.liquidation_events = []
        self._pending_liquidation = set()
        # M5: the live path disables the *simulated* liquidation engine and the
        # *simulated* funding charge — on live, liquidations and funding are real
        # exchange events, so the modeled versions must never fabricate a local
        # fill or debit cash. Defaults False so the backtest path (and the golden
        # baseline) is byte-identical; live/portfolio.LivePortfolio flips this
        # True in its __init__.
        self._live_liquidation_disabled = False
        # M5 (live only): modeled maintenance-margin breaches recorded for audit /
        # alerting instead of being applied as synthetic fills. Never populated on
        # the backtest path.
        self._modeled_liquidation_warnings = []
        # X5: cache the position-invariant liquidation price per symbol keyed by
        # (qty, entry_price) so a held position does not recompute it every bar.
        # {symbol: (qty, entry_price, liq_price_or_None)}; invalidated on fill.
        self._liq_price_cache = {}
        self._metric_totals = [float(self.initial_capital)] if self.track_metrics else []
        self._metric_benchmarks = [0.0] if self.track_metrics else []
        self._last_metric_timestamp_ms = (
            self._to_timestamp_ms(self.start_date) if self.track_metrics else None
        )
        self._equity_points = deque(maxlen=20_000)
        self.trading_frozen = False
        self.component_positions = {}

        # L-C no-trade band (bps of equity). 0.0 = OFF (byte-identical default):
        # entry / partial-exit orders below the band are dropped as sub-cost
        # churn; full exits are always exempt (position hygiene).
        self.no_trade_band_bps = float(
            getattr(config, "STRATEGY_QUALITY_NO_TRADE_BAND_BPS", 0.0) or 0.0
        )
        self.strategy_quality = StrategyQualityOverlay(config)
        # Initialize first record
        self.update_initial_record()

    @property
    def fill_application_attribution_sink(self):
        return self._fill_application_attribution_sink

    @property
    def funding_boundary_resolver(self):
        return self._funding_boundary_resolver

    @property
    def full_event_equity_sink(self):
        return self._full_event_equity_sink

    @property
    def reporting_sampling_timeframe(self):
        return self._reporting_sampling_timeframe

    def construct_current_holdings(self):
        d = dict.fromkeys(self.symbol_list, 0.0)
        d["cash"] = self.initial_capital
        d["commission"] = 0.0
        d["total"] = self.initial_capital
        d["funding"] = 0.0
        return d

    def update_initial_record(self):
        # Initial positions - Store as Tuple: (datetime, s1, s2, ...)
        # Rely on self.symbol_list order
        pos_row = [self.start_date] + [0.0 for _ in self.symbol_list]
        self.all_positions.append(tuple(pos_row))

        # Initial holdings - Store as Tuple: (datetime, cash, commission, total, s1, s2, ..., benchmark_price)
        h_row = (
            [self.start_date, self.initial_capital, 0.0, 0.0, self.initial_capital]
            + [0.0 for _ in self.symbol_list]
            + [0.0]
        )  # Benchmark Price Placeholder
        self.all_holdings.append(tuple(h_row))
        self._last_sample_timestamp_ms = self._to_timestamp_ms(self.start_date)
        self.save_portfolio_state()

    def save_portfolio_state(self):
        # We assume LiveTrader handles file I/O via get_state
        pass

    def get_state(self):
        return {
            "positions": self.current_positions,
            "holdings": self.current_holdings,
            "initial_capital": self.initial_capital,
            "circuit_breaker_tripped": self.circuit_breaker_tripped,
            "entry_prices": self.entry_prices,
            "total_funding_paid": self.total_funding_paid,
            "funding": self._funding_state(),
            "pending_liquidation": list(self._pending_liquidation),
            "liquidation_events": list(self.liquidation_events),
            "trade_count": self.trade_count,
            "trading_frozen": bool(self.trading_frozen),
            "equity_points": list(self._equity_points),
            "component_positions": self.component_positions,
            "last_sample_timestamp_ms": self._last_sample_timestamp_ms,
            "last_metric_timestamp_ms": self._last_metric_timestamp_ms,
            "strategy_quality": self.strategy_quality.get_state(),
            "current_day": self._current_day,
            "day_start_equity": self.day_start_equity,
        }

    def set_state(self, state):
        if not isinstance(state, dict):
            raise ValueError("portfolio state must be an object")
        if set(state) != set(self.get_state()):
            raise ValueError("portfolio state must contain the exact checkpoint schema")
        symbols = set(self.symbol_list)

        def finite(value, name):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"invalid {name}")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"invalid {name}")
            return value

        staged = {}
        if "positions" in state:
            raw = state["positions"]
            if not isinstance(raw, dict) or set(raw) != symbols:
                raise ValueError("positions must cover exactly known symbols")
            staged["current_positions"] = {
                symbol: finite(quantity, f"position {symbol}") for symbol, quantity in raw.items()
            }
        if "holdings" in state:
            raw = state["holdings"]
            required = {"cash", "commission", "funding", "total", *symbols}
            if not isinstance(raw, dict) or set(raw) != required:
                raise ValueError("holdings must cover exactly known fields")
            staged["current_holdings"] = {
                key: finite(value, f"holding {key}") for key, value in raw.items()
            }
        if "entry_prices" in state:
            raw = state["entry_prices"]
            if not isinstance(raw, dict) or set(raw) != symbols:
                raise ValueError("entry_prices must cover exactly known symbols")
            staged["entry_prices"] = {
                symbol: None if value is None else finite(value, f"entry price {symbol}")
                for symbol, value in raw.items()
            }
        if "funding" in state:
            funding = self._validated_funding_state(state["funding"])
            staged.update(
                _last_funding_ts=funding["settlement_cursors"],
                _funding_exposure_cursor=funding["exposure_cursors"],
                _pending_funding_liabilities=funding["liabilities"],
            )
        if "pending_liquidation" in state:
            raw = state["pending_liquidation"]
            if not isinstance(raw, list) or not all(
                isinstance(item, str) and item in symbols for item in raw
            ):
                raise ValueError("invalid pending_liquidation")
            staged["_pending_liquidation"] = set(raw)
        if "liquidation_events" in state:
            raw = state["liquidation_events"]
            if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
                raise ValueError("invalid liquidation_events")
            staged["liquidation_events"] = list(raw)
        for key in ("initial_capital", "total_funding_paid", "day_start_equity"):
            if key in state:
                staged[key] = finite(state[key], key)
        if "trade_count" in state:
            if (
                isinstance(state["trade_count"], bool)
                or not isinstance(state["trade_count"], int)
                or state["trade_count"] < 0
            ):
                raise ValueError("invalid trade_count")
            staged["trade_count"] = state["trade_count"]
        if "circuit_breaker_tripped" in state:
            if not isinstance(state["circuit_breaker_tripped"], bool):
                raise ValueError("invalid circuit_breaker_tripped")
            staged["circuit_breaker_tripped"] = state["circuit_breaker_tripped"]
        if "trading_frozen" in state:
            if not isinstance(state["trading_frozen"], bool):
                raise ValueError("invalid trading_frozen")
            staged["trading_frozen"] = state["trading_frozen"]
        for key in ("last_sample_timestamp_ms", "last_metric_timestamp_ms"):
            if key in state:
                value = state[key]
                if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                    raise ValueError(f"invalid {key}")
                staged[f"_{key}"] = value
        if "equity_points" in state:
            raw = state["equity_points"]
            if not isinstance(raw, list):
                raise ValueError("invalid equity_points")
            staged["_equity_points"] = deque(raw, maxlen=20_000)
        if "component_positions" in state:
            raw = state["component_positions"]
            if not isinstance(raw, dict):
                raise ValueError("invalid component_positions")
            staged["component_positions"] = {
                str(component): {
                    str(symbol): finite(quantity, f"component position {symbol}")
                    for symbol, quantity in dict(values).items()
                }
                for component, values in raw.items()
                if isinstance(component, str) and component and isinstance(values, dict)
            }
            if len(staged["component_positions"]) != len(raw):
                raise ValueError("invalid component_positions")
        if "current_day" in state:
            staged["_current_day"] = state["current_day"]
        if "strategy_quality" in state and not isinstance(state["strategy_quality"], dict):
            raise ValueError("invalid strategy_quality")

        previous_strategy_quality = self.strategy_quality.get_state()
        try:
            if "strategy_quality" in state:
                self.strategy_quality.set_state(state["strategy_quality"])
            for key, value in staged.items():
                setattr(self, key, value)
        except Exception:
            self.strategy_quality.set_state(previous_strategy_quality)
            raise

    def _funding_state(self):
        rows = [
            {
                "symbol": symbol,
                "boundary_ms": boundary_ms,
                "quantity": quantity,
            }
            for symbol in sorted(self.symbol_list)
            for boundary_ms, quantity in sorted(
                self._pending_funding_liabilities.get(symbol, {}).items()
            )
        ]
        return {
            "schema": "portfolio_funding.v1",
            "settlement_cursors": dict(self._last_funding_ts),
            "exposure_cursors": dict(self._funding_exposure_cursor),
            "liabilities": rows,
        }

    def _validated_funding_state(self, value):
        if not isinstance(value, dict) or set(value) != {
            "schema",
            "settlement_cursors",
            "exposure_cursors",
            "liabilities",
        }:
            raise ValueError("invalid atomic funding state")
        if value["schema"] != "portfolio_funding.v1":
            raise ValueError("unsupported funding state schema")
        symbols = set(self.symbol_list)

        def cursors(name):
            raw = value[name]
            if not isinstance(raw, dict) or set(raw) != symbols:
                raise ValueError(f"funding {name} must cover exactly known symbols")
            parsed = {}
            for symbol, cursor in raw.items():
                if cursor is None:
                    parsed[symbol] = None
                elif (
                    isinstance(cursor, bool)
                    or not isinstance(cursor, (int, float))
                    or not math.isfinite(float(cursor))
                ):
                    raise ValueError(f"invalid funding {name} cursor")
                else:
                    parsed[symbol] = float(cursor)
            return parsed

        settlement_cursors = cursors("settlement_cursors")
        exposure_cursors = cursors("exposure_cursors")
        interval_ms = int(self.execution_model.cfg.funding_interval_hours * 3_600_000)
        rows = value["liabilities"]
        if not isinstance(rows, list) or interval_ms <= 0:
            raise ValueError("invalid funding liabilities")
        liabilities = {symbol: {} for symbol in self.symbol_list}
        previous = None
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"symbol", "boundary_ms", "quantity"}:
                raise ValueError("invalid funding liability row")
            symbol = row["symbol"]
            boundary_ms = row["boundary_ms"]
            quantity = row["quantity"]
            if (
                symbol not in symbols
                or isinstance(boundary_ms, bool)
                or not isinstance(boundary_ms, int)
                or boundary_ms < 0
                or boundary_ms % interval_ms
                or isinstance(quantity, bool)
                or not isinstance(quantity, (int, float))
                or not math.isfinite(float(quantity))
                or abs(float(quantity)) < 1e-12
            ):
                raise ValueError("invalid funding liability row")
            key = (symbol, boundary_ms)
            if previous is not None and key <= previous:
                raise ValueError("funding liability rows must be canonical and unique")
            previous = key
            liabilities[symbol][boundary_ms] = float(quantity)
        for symbol in self.symbol_list:
            settlement = settlement_cursors[symbol]
            exposure = exposure_cursors[symbol]
            if settlement is not None and exposure is not None and settlement > exposure:
                raise ValueError("funding settlement cursor exceeds exposure cursor")
            for boundary_ms in liabilities[symbol]:
                if settlement is not None and boundary_ms <= round(settlement * 1000):
                    raise ValueError("pending funding liability is already settled")
                if exposure is None or boundary_ms > round(exposure * 1000):
                    raise ValueError("pending funding liability exceeds exposure cursor")
        return {
            "settlement_cursors": settlement_cursors,
            "exposure_cursors": exposure_cursors,
            "liabilities": liabilities,
        }

    @staticmethod
    def _audit_flag(config, upper_attr: str, section: str, field: str) -> bool:
        """Resolve an audit-hardening bool flag, default False.

        Prefers an uppercase attr on ``config`` (plain-class unit-test configs);
        falls back to the RuntimeConfig dotpath carried on ``config._rt`` (the
        BacktestConfigView surface used in production). Absent everywhere => False,
        so each gate is a strict no-op at its schema default.
        """
        sentinel = object()
        value = getattr(config, upper_attr, sentinel)
        if value is not sentinel:
            return bool(value)
        runtime = wrapped_runtime_config(config)
        if runtime is not None:
            section_obj = getattr(runtime, section, None)
            if section_obj is not None:
                return bool(getattr(section_obj, field, False))
        return False

    @staticmethod
    def _component_id_from_metadata(metadata) -> str | None:
        raw = dict(metadata or {})
        direct = str(raw.get("component_id") or "").strip()
        if direct:
            return direct
        signal_meta = dict(raw.get("signal_metadata") or {})
        nested = str(signal_meta.get("component_id") or "").strip()
        return nested or None

    @staticmethod
    def _scalar_from_value(value) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            parsed = float(value)
            return parsed if math.isfinite(parsed) else None
        if isinstance(value, (tuple, list)):
            for item in value:
                scalar = Portfolio._scalar_from_value(item)
                if scalar is not None:
                    return scalar
            return None
        if isinstance(value, dict):
            for key in ("value", "price", "rate", "close"):
                if key in value:
                    scalar = Portfolio._scalar_from_value(value[key])
                    if scalar is not None:
                        return scalar
            return None
        for attr in ("value", "price", "rate", "close"):
            if hasattr(value, attr):
                scalar = Portfolio._scalar_from_value(getattr(value, attr))
                if scalar is not None:
                    return scalar
        try:
            parsed = float(value)
        except Exception:
            return None
        return parsed if math.isfinite(parsed) else None

    @staticmethod
    def _event_optional_str(event, name: str) -> str | None:
        value = getattr(event, name, None)
        if value is None:
            return None
        if type(value) is not str:
            raise TypeError(f"fill_application_event_field_invalid:{name}")
        return value

    @staticmethod
    def _event_required_str(event, name: str) -> str:
        value = getattr(event, name, None)
        if type(value) is not str or not value:
            raise TypeError(f"fill_application_event_field_invalid:{name}")
        return value

    @staticmethod
    def _is_synthetic_liquidation_fill(event) -> bool:
        metadata = getattr(event, "metadata", None)
        return (
            getattr(event, "exchange", None) == "SIM_LIQUIDATION"
            and getattr(event, "status", None) == "LIQUIDATED"
            and isinstance(metadata, Mapping)
            and metadata.get("reason") == "maintenance_margin_breach"
        )

    @staticmethod
    def _require_pricing_trace(event) -> ExecutionPricingTrace:
        metadata = getattr(event, "metadata", None)
        if not isinstance(metadata, Mapping):
            raise RuntimeError("attributed FillEvent metadata is missing")
        trace = metadata.get("cost_attribution")
        if type(trace) is not ExecutionPricingTrace:
            raise RuntimeError("attributed FillEvent has no exact ExecutionPricingTrace")
        # Structural/canonical validation occurs before any portfolio mutation.
        trace.to_payload()

        quantity = float(getattr(event, "quantity", 0.0) or 0.0)
        fill_cost = getattr(event, "fill_cost", None)
        commission = getattr(event, "commission", None)
        if quantity <= 0.0 or not math.isfinite(quantity):
            raise ValueError("attributed FillEvent quantity is not positive and finite")
        if fill_cost is None or commission is None:
            raise ValueError("attributed FillEvent cost fields are missing")
        fill_cost = float(fill_cost)
        commission = float(commission)
        if not math.isfinite(fill_cost) or not math.isfinite(commission):
            raise ValueError("attributed FillEvent cost fields are non-finite")
        if not math.isclose(quantity, trace.executed_qty, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("attributed FillEvent quantity does not match pricing trace")
        if not math.isclose(
            fill_cost,
            trace.fill_price * trace.executed_qty,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("attributed FillEvent fill_cost does not match pricing trace")
        if not math.isclose(commission, trace.commission, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("attributed FillEvent commission does not match pricing trace")
        direction = Portfolio._event_required_str(event, "direction")
        if direction.upper() != trace.direction:
            raise ValueError("attributed FillEvent direction does not match pricing trace")
        if getattr(event, "order_id", None) != trace.order_id:
            raise ValueError("attributed FillEvent order_id does not match pricing trace")
        if getattr(event, "client_order_id", None) != trace.client_order_id:
            raise ValueError("attributed FillEvent client_order_id does not match pricing trace")
        return trace

    def _build_fill_application_attribution_record(
        self,
        original_event,
        applied_event,
        pricing_trace: ExecutionPricingTrace,
        *,
        application_status: str,
        zero_applied_reason: str | None,
    ) -> FillApplicationAttribution:
        model_quantity = float(getattr(original_event, "quantity", 0.0) or 0.0)
        model_fill_cost = float(getattr(original_event, "fill_cost", 0.0) or 0.0)
        model_commission = float(getattr(original_event, "commission", 0.0) or 0.0)
        applied_quantity = (
            float(getattr(applied_event, "quantity", 0.0) or 0.0) if applied_event else 0.0
        )
        applied_fill_cost = (
            float(getattr(applied_event, "fill_cost", 0.0) or 0.0) if applied_event else 0.0
        )
        applied_commission = (
            float(getattr(applied_event, "commission", 0.0) or 0.0) if applied_event else 0.0
        )
        scale = 0.0
        if application_status == "applied_unchanged":
            scale = 1.0
        elif model_quantity > 0.0 and applied_quantity > 0.0:
            scale = abs(applied_quantity / model_quantity)
        metadata = getattr(original_event, "metadata", None)
        reduce_only = metadata.get("reduce_only", False) if isinstance(metadata, Mapping) else False
        if type(reduce_only) is not bool:
            raise TypeError("fill_application_reduce_only_metadata_invalid")
        record = FillApplicationAttribution(
            record_type="fill_application_attribution",
            pricing_trace_hash=execution_pricing_trace_sha256(pricing_trace),
            pricing_trace=pricing_trace,
            timeindex=_canonical_attribution_timeindex(getattr(original_event, "timeindex", None)),
            symbol=self._event_required_str(original_event, "symbol"),
            direction=self._event_required_str(original_event, "direction").upper(),
            order_id=self._event_optional_str(original_event, "order_id"),
            client_order_id=self._event_optional_str(original_event, "client_order_id"),
            position_side=self._event_optional_str(original_event, "position_side"),
            status=self._event_optional_str(original_event, "status"),
            reduce_only=reduce_only,
            model_quantity=model_quantity,
            model_fill_cost=model_fill_cost,
            model_commission=model_commission,
            applied_quantity=applied_quantity,
            applied_fill_cost=applied_fill_cost,
            applied_commission=applied_commission,
            reduce_only_scale=scale,
            application_status=application_status,
            zero_applied_reason=zero_applied_reason,
        )
        record.to_payload()
        return record

    def _emit_fill_application_attribution(
        self,
        original_event,
        applied_event,
        pricing_trace: ExecutionPricingTrace,
        *,
        application_status: str,
        zero_applied_reason: str | None,
    ) -> None:
        sink = self.fill_application_attribution_sink
        if sink is None:
            return
        sink(
            self._build_fill_application_attribution_record(
                original_event,
                applied_event,
                pricing_trace,
                application_status=application_status,
                zero_applied_reason=zero_applied_reason,
            )
        )

    def _reduce_only_zero_reason(self, event) -> str | None:
        metadata = getattr(event, "metadata", None) or {}
        if not bool(metadata.get("reduce_only", False)):
            return None
        quantity = float(getattr(event, "quantity", 0.0) or 0.0)
        if quantity <= 0.0:
            return "zero_quantity"
        old_qty = float(self.current_positions.get(event.symbol, 0.0) or 0.0)
        fill_dir = 1.0 if event.direction == "BUY" else -1.0
        if abs(old_qty) <= 1e-12:
            return "reduce_only_flat"
        if old_qty * fill_dir > 0.0:
            return "reduce_only_wrong_side"
        return None

    def _apply_funding_boundary_resolution(self, latest_datetime, *, now_ts: float):
        raw_point_accessor = getattr(self.bars, "get_latest_raw_point", None)
        if not callable(raw_point_accessor):
            raise AttributeError("funding_boundary_resolver requires bars.get_latest_raw_point")
        resolver = self.funding_boundary_resolver
        resolve_batch_fn = getattr(resolver, "resolve_batch", None)
        resolve_fn = getattr(resolver, "resolve", None)
        if not callable(resolve_batch_fn) and not callable(resolve_fn):
            raise TypeError(
                "funding_boundary_resolver must expose callable resolve_batch or resolve"
            )
        interval_seconds = int(self.execution_model.cfg.funding_interval_hours * 3600)
        pending_payments: list[tuple[str, int, float]] = []
        pending_anchors: dict[str, float] = {}
        requests: list[dict[str, object]] = []
        used_batch_resolution = False

        for symbol in self.symbol_list:
            qty = float(self.current_positions.get(symbol, 0.0))
            if abs(qty) < 1e-12:
                continue
            last_ts = self._last_funding_ts.get(symbol)
            if last_ts is None:
                pending_anchors[symbol] = now_ts
                continue
            if now_ts <= last_ts:
                continue

            start_index = int(last_ts // interval_seconds) + 1
            end_index = int(now_ts // interval_seconds) + 1
            for boundary_index in range(start_index, end_index):
                qty = float(self.current_positions.get(symbol, 0.0))
                if abs(qty) < 1e-12:
                    break
                boundary_seconds = float(boundary_index * interval_seconds)
                boundary_ms = int(boundary_seconds * 1000.0)
                requests.append(
                    {
                        "symbol": symbol,
                        "boundary_ms": boundary_ms,
                        "qty": qty,
                        "latest_datetime": latest_datetime,
                    }
                )

        if requests and callable(resolve_batch_fn):
            used_batch_resolution = True
            # Production contract: one complete causal batch is sealed by the
            # resolver before Portfolio applies any cash mutation.  The first
            # positional argument is the immutable request sequence; only the raw
            # accessor and exact ExecutionModel are supplied as keyword capabilities.
            resolved_batch = resolve_batch_fn(
                tuple(requests),
                raw_point_accessor=raw_point_accessor,
                execution_model=self.execution_model,
            )
            if not isinstance(resolved_batch, (tuple, list)):
                raise TypeError("funding_boundary_resolver batch result must be a sequence")
            requested_by_key = {
                (str(request["symbol"]), int(request["boundary_ms"])): request
                for request in requests
            }
            paid_by_key: dict[tuple[str, int], float] = {}
            for resolved in resolved_batch:
                if isinstance(resolved, Mapping):
                    symbol = resolved.get("symbol")
                    boundary_ms = resolved.get("boundary_ms")
                    payment_raw = resolved.get("payment")
                    qty_raw = resolved.get("qty")
                else:
                    symbol = getattr(resolved, "symbol", None)
                    boundary_ms = getattr(resolved, "boundary_ms", None)
                    payment_raw = getattr(resolved, "payment", None)
                    qty_raw = getattr(resolved, "qty", None)
                if type(symbol) is not str or type(boundary_ms) is not int:
                    raise ValueError("funding_boundary_batch_identity_invalid")
                key = (symbol, boundary_ms)
                request = requested_by_key.get(key)
                if request is None or key in paid_by_key:
                    raise ValueError("funding_boundary_batch_bijection_invalid")
                if (
                    type(payment_raw) is not float
                    or not math.isfinite(payment_raw)
                    or type(qty_raw) is not float
                    or not math.isfinite(qty_raw)
                ):
                    raise ValueError("funding_boundary_batch_paid_row_invalid")
                if not math.isclose(
                    qty_raw,
                    float(request["qty"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError("funding_boundary_batch_quantity_mismatch")
                paid_by_key[key] = payment_raw
            if set(paid_by_key) != set(requested_by_key):
                raise ValueError("funding_boundary_batch_bijection_invalid")
            for request in requests:
                symbol = str(request["symbol"])
                boundary_ms = int(request["boundary_ms"])
                pending_payments.append(
                    (
                        symbol,
                        boundary_ms // (interval_seconds * 1000),
                        paid_by_key[(symbol, boundary_ms)],
                    )
                )

        elif requests:
            # Backward-compatible legacy resolver path.  Resolution remains
            # portfolio-atomic: all rows are collected and validated before cash.
            for request in requests:
                resolved = resolve_fn(
                    symbol=request["symbol"],
                    boundary_ms=request["boundary_ms"],
                    qty=request["qty"],
                    latest_datetime=request["latest_datetime"],
                    raw_point_accessor=raw_point_accessor,
                )
                if resolved is None:
                    raise ValueError("funding_boundary_resolver returned no boundary resolution")

                if isinstance(resolved, dict):
                    payment = self._scalar_from_value(resolved.get("payment"))
                    rate_value = self._scalar_from_value(
                        resolved.get("rate")
                        if "rate" in resolved
                        else resolved.get("rate_point")
                        if "rate_point" in resolved
                        else resolved.get("boundary_rate")
                    )
                    price_value = self._scalar_from_value(
                        resolved.get("price")
                        if "price" in resolved
                        else resolved.get("price_point")
                        if "price_point" in resolved
                        else resolved.get("boundary_price")
                    )
                else:
                    payment = self._scalar_from_value(getattr(resolved, "payment", None))
                    rate_value = self._scalar_from_value(
                        getattr(resolved, "rate", None)
                        if hasattr(resolved, "rate")
                        else getattr(resolved, "rate_point", None)
                        if hasattr(resolved, "rate_point")
                        else getattr(resolved, "boundary_rate", None)
                    )
                    price_value = self._scalar_from_value(
                        getattr(resolved, "price", None)
                        if hasattr(resolved, "price")
                        else getattr(resolved, "price_point", None)
                        if hasattr(resolved, "price_point")
                        else getattr(resolved, "boundary_price", None)
                    )

                if payment is None:
                    if rate_value is None or price_value is None:
                        raise ValueError(
                            "funding_boundary_resolver must provide rate/price or payment"
                        )
                    payment = self.execution_model.compute_funding_payment(
                        signed_qty=float(request["qty"]),
                        price=price_value,
                        periods=1,
                        rate=rate_value,
                    )

                pending_payments.append(
                    (
                        str(request["symbol"]),
                        int(request["boundary_ms"]) // (interval_seconds * 1000),
                        float(payment),
                    )
                )

        if not pending_payments and not pending_anchors:
            return

        if used_batch_resolution:
            # Alpha batch settlement is one exact ledger reconciliation.  fsum
            # avoids order-sensitive cancellation and the cash mutation occurs once.
            batch_total = math.fsum(payment for _, _, payment in pending_payments)
            self.current_holdings["cash"] -= batch_total
            self.current_holdings["total"] -= batch_total
            self.current_holdings["funding"] += batch_total
            self.total_funding_paid += batch_total
            for symbol, boundary_index, _ in pending_payments:
                boundary_seconds = float(boundary_index * interval_seconds)
                self._last_funding_ts[symbol] = boundary_seconds
                exposure_cursor = self._funding_exposure_cursor.get(symbol)
                self._funding_exposure_cursor[symbol] = max(
                    boundary_seconds,
                    boundary_seconds if exposure_cursor is None else float(exposure_cursor),
                )
        else:
            # Preserve legacy resolver arithmetic/order exactly.
            for symbol, boundary_index, payment in pending_payments:
                boundary_seconds = float(boundary_index * interval_seconds)
                self.current_holdings["cash"] -= payment
                self.current_holdings["total"] -= payment
                self.current_holdings["funding"] += payment
                self.total_funding_paid += payment
                self._last_funding_ts[symbol] = boundary_seconds
                exposure_cursor = self._funding_exposure_cursor.get(symbol)
                self._funding_exposure_cursor[symbol] = max(
                    boundary_seconds,
                    boundary_seconds if exposure_cursor is None else float(exposure_cursor),
                )

        for symbol, anchor_ts in pending_anchors.items():
            self._last_funding_ts[symbol] = anchor_ts
            exposure_cursor = self._funding_exposure_cursor.get(symbol)
            self._funding_exposure_cursor[symbol] = max(
                anchor_ts,
                anchor_ts if exposure_cursor is None else float(exposure_cursor),
            )

    def update_timeindex(self, event):
        """Updates the positions from the current locations to the
        latest available bar.
        """
        _ = event
        primary_symbol = self.symbol_list[0]
        latest_datetime = self.bars.get_latest_bar_datetime(primary_symbol)
        self.strategy_quality.next_bar(latest_datetime)
        self.strategy_quality.reconcile_min_hold_positions(
            self.current_positions,
            self.component_positions,
        )
        # L-C min-hold: release deferred bare EXITs whose hold just matured as
        # synthetic EXIT signals (the emitting strategy is one-shot and will
        # never re-emit). The overlay_reason marker lets them pass the gate.
        for pending in self.strategy_quality.pop_matured_pending_exits():
            metadata = dict(pending.get("metadata") or {})
            metadata["overlay_reason"] = "min_hold_released"
            metadata["min_hold_exit_key"] = str(pending.get("key") or "")
            if pending.get("component_id"):
                metadata.setdefault("component_id", pending["component_id"])
            self.events.put(
                SignalEvent(
                    strategy_id=str(pending.get("strategy_id") or "overlay"),
                    symbol=str(pending.get("symbol") or ""),
                    datetime=latest_datetime,
                    signal_type="EXIT",
                    strength=1.0,
                    client_order_id=str(pending.get("client_order_id") or "") or None,
                    metadata=metadata,
                )
            )
            self.strategy_quality.mark_pending_exit_state(
                str(pending.get("key") or ""), "DISPATCHED"
            )
        should_sample = self._should_sample(latest_datetime)
        self._update_day_boundary(latest_datetime)
        self._apply_funding(latest_datetime)
        self._check_liquidations(latest_datetime, event)

        current_positions = self.current_positions
        current_holdings = self.current_holdings
        cash = current_holdings["cash"]
        commission = current_holdings["commission"]
        collect_history = self.record_history and should_sample

        if self._single_symbol:
            symbol = primary_symbol
            qty = current_positions[symbol]
            close_price = self.bars.get_latest_bar_value(symbol, "close")
            market_value = qty * close_price if qty != 0 else 0.0
            current_holdings[symbol] = market_value
            total = cash + market_value
            current_holdings["total"] = total
            self._record_equity_point(latest_datetime, total)
            if self.track_metrics and should_sample:
                self._metric_totals.append(float(total))
                self._metric_benchmarks.append(float(close_price))
                self._last_metric_timestamp_ms = self._to_timestamp_ms(latest_datetime)

            if collect_history:
                self.all_positions.append((latest_datetime, qty))
                self.all_holdings.append(
                    (
                        latest_datetime,
                        cash,
                        commission,
                        current_holdings.get("funding", 0.0),
                        total,
                        market_value,
                        close_price,
                    )
                )
            return

        total = cash
        market_vals = [] if collect_history else None
        for symbol in self.symbol_list:
            qty = current_positions[symbol]
            market_value = (
                qty * self.bars.get_latest_bar_value(symbol, "close") if qty != 0 else 0.0
            )
            if market_vals is not None:
                market_vals.append(market_value)
            total += market_value
            current_holdings[symbol] = market_value

        current_holdings["total"] = total
        self._record_equity_point(latest_datetime, total)
        bench_price = self.bars.get_latest_bar_value(primary_symbol, "close")
        if self.track_metrics and should_sample:
            self._metric_totals.append(float(total))
            self._metric_benchmarks.append(float(bench_price))
            self._last_metric_timestamp_ms = self._to_timestamp_ms(latest_datetime)
        if not collect_history:
            return

        # Update positions
        # Tuple: (datetime, s1, s2...)
        self.all_positions.append(
            (latest_datetime, *(current_positions[s] for s in self.symbol_list))
        )

        # Store Tuple
        # Schema: (datetime, cash, commission, total, s1_val, s2_val, ..., benchmark_price)
        # Benchmark: Close price of first symbol (Primary Asset)
        history_market_vals = market_vals if market_vals is not None else []
        self.all_holdings.append(
            (
                latest_datetime,
                cash,
                commission,
                current_holdings.get("funding", 0.0),
                total,
                *history_market_vals,
                bench_price,
            )
        )

    def update_timeindex_inert_batch(
        self,
        timestamp_ms: np.ndarray,
        closes_by_symbol: Mapping[str, np.ndarray],
    ) -> None:
        """Record exact consecutive non-boundary marks with immutable positions."""
        if (
            type(timestamp_ms) is not np.ndarray
            or timestamp_ms.dtype != np.dtype(np.int64)
            or timestamp_ms.ndim != 1
            or timestamp_ms.size == 0
            or tuple(closes_by_symbol) != tuple(self.symbol_list)
            or self.strategy_quality.enabled
        ):
            raise ValueError("inert_equity_batch_invalid")
        if timestamp_ms.size > 1 and not bool(np.all(np.diff(timestamp_ms) == 1000)):
            raise ValueError("inert_equity_batch_timeline_invalid")
        first_day = datetime.fromtimestamp(int(timestamp_ms[0]) / 1000.0, UTC).date()
        last_day = datetime.fromtimestamp(int(timestamp_ms[-1]) / 1000.0, UTC).date()
        current_day = (
            self._current_day.date()
            if isinstance(self._current_day, datetime)
            else self._current_day
        )
        if (
            first_day != last_day
            or current_day != first_day
            or (
                self._sampling_interval_ms
                and bool(np.any(timestamp_ms % int(self._sampling_interval_ms) == 0))
            )
        ):
            raise ValueError("inert_equity_batch_boundary_invalid")

        cash = float(self.current_holdings["cash"])
        totals = np.full(timestamp_ms.shape, cash, dtype=np.float64)
        final_market_values: dict[str, float] = {}
        for symbol in self.symbol_list:
            closes = closes_by_symbol[symbol]
            if (
                type(closes) is not np.ndarray
                or closes.dtype != np.dtype(np.float64)
                or closes.shape != timestamp_ms.shape
                or not bool(np.all(np.isfinite(closes)))
                or not bool(np.all(closes > 0.0))
            ):
                raise ValueError("inert_equity_batch_close_invalid")
            quantity = float(self.current_positions[symbol])
            market_values = closes * quantity if quantity != 0.0 else np.zeros_like(closes)
            totals = totals + market_values
            final_market_values[symbol] = quantity * float(closes[-1]) if quantity != 0.0 else 0.0
        if not bool(np.all(np.isfinite(totals))):
            raise ValueError("inert_equity_batch_total_invalid")

        points = np.column_stack((timestamp_ms.astype(np.float64) / 1000.0, totals))
        sink = self._full_event_equity_sink
        batch_sink = getattr(sink, "update_batch", None)
        if callable(batch_sink):
            batch_sink(points)
        elif sink is not None:
            for point in points:
                sink((float(point[0]), float(point[1])))
        self._equity_points.extend(
            (float(point[0]), float(point[1])) for point in points[-self._equity_points.maxlen :]
        )
        self.strategy_quality.bar_index += int(timestamp_ms.size) - 1
        self.strategy_quality.next_bar(int(timestamp_ms[-1]))
        for symbol, market_value in final_market_values.items():
            self.current_holdings[symbol] = market_value
        self.current_holdings["total"] = float(totals[-1])

    def reconcile_final_snapshot(self):
        """Refresh the last bar snapshot after its queued fills have drained."""
        latest_datetime = self.bars.get_latest_bar_datetime(self.symbol_list[0])
        if latest_datetime is None:
            return
        market_values = []
        for symbol in self.symbol_list:
            value = float(self.current_positions[symbol]) * float(
                self.bars.get_latest_bar_value(symbol, "close")
            )
            self.current_holdings[symbol] = value
            market_values.append(value)
        total = float(self.current_holdings["cash"]) + sum(market_values)
        self.current_holdings["total"] = total
        benchmark = float(self.bars.get_latest_bar_value(self.symbol_list[0], "close"))

        if self._equity_points and self._equity_points[-1][0] == self._to_unix_seconds(
            latest_datetime
        ):
            self._equity_points[-1] = (self._equity_points[-1][0], total)
        metric_timestamp_ms = self._to_timestamp_ms(latest_datetime)
        if self.track_metrics:
            if self._metric_totals and self._last_metric_timestamp_ms == metric_timestamp_ms:
                self._metric_totals[-1] = total
                self._metric_benchmarks[-1] = benchmark
            else:
                self._metric_totals.append(total)
                self._metric_benchmarks.append(benchmark)
                self._last_metric_timestamp_ms = metric_timestamp_ms
        if not self.record_history:
            return

        positions = (latest_datetime, *(self.current_positions[s] for s in self.symbol_list))
        holdings = (
            latest_datetime,
            self.current_holdings["cash"],
            self.current_holdings["commission"],
            self.current_holdings.get("funding", 0.0),
            total,
            *market_values,
            benchmark,
        )
        if self.all_positions and self.all_positions[-1][0] == latest_datetime:
            self.all_positions[-1] = positions
            self.all_holdings[-1] = holdings
        else:
            self.all_positions.append(positions)
            self.all_holdings.append(holdings)

    def _to_timestamp_ms(self, value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            ts = int(value)
            if abs(ts) < 100_000_000_000:
                ts *= 1000
            return ts
        if isinstance(value, datetime):
            # Convention (core/engine.py): naive datetimes are UTC. Calling
            # .timestamp() on a naive datetime interprets it in the host-local
            # tz, which would skew equity-sampling cadence on a non-UTC host —
            # localize to UTC first (mirrors market_data._coerce_timestamp_ms).
            dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
            return int(dt.timestamp() * 1000)
        if isinstance(value, date):
            return int(datetime(value.year, value.month, value.day, tzinfo=UTC).timestamp() * 1000)
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return int(parsed.timestamp() * 1000)

    def _should_sample(self, latest_datetime):
        ts_ms = self._to_timestamp_ms(latest_datetime)
        if ts_ms is None:
            return True
        if self._last_sample_timestamp_ms is not None and ts_ms == self._last_sample_timestamp_ms:
            return False
        if self._sampling_interval_ms and ts_ms % self._sampling_interval_ms != 0:
            return False
        self._last_sample_timestamp_ms = ts_ms
        return True

    def update_positions_from_fill(self, fill):
        # X5: any fill can change this symbol's qty and/or entry price, both of
        # which the cached position-invariant liquidation price depends on — drop
        # the cache entry so the next bar recomputes it against fresh inputs.
        self._liq_price_cache.pop(fill.symbol, None)

        alpha_funding_anchor = None
        if self.funding_boundary_resolver is not None:
            fill_time = getattr(fill, "timeindex", None)
            if type(fill_time) is int and fill_time >= 100_000_000_000:
                # MarketWindowEvent canonicalizes its event clock to exact
                # epoch milliseconds.  Preserve that exact anchor instead of
                # forcing the Alpha-Max engine through a datetime-only seam.
                alpha_funding_anchor = fill_time / 1000.0
            elif (
                isinstance(fill_time, datetime)
                and fill_time.tzinfo is not None
                and fill_time.utcoffset() == timedelta(0)
            ):
                alpha_funding_anchor = float(fill_time.timestamp())
            else:
                raise ValueError("funding_boundary_fill_timestamp_invalid")

        fill_dir = 0
        if fill.direction == "BUY":
            fill_dir = 1
        if fill.direction == "SELL":
            fill_dir = -1

        old_qty = float(self.current_positions.get(fill.symbol, 0.0))
        fill_qty = float(fill.quantity) * fill_dir
        new_qty = old_qty + fill_qty
        if self.funding_on_utc_boundary:
            fill_ts = self._to_unix_seconds(getattr(fill, "timeindex", None))
            if fill_ts is None:
                raise ValueError("UTC funding exposure scan requires a fill timestamp")
            self._scan_funding_exposure(fill.symbol, now_ts=fill_ts, quantity=old_qty)
        self.current_positions[fill.symbol] = new_qty

        component_id = self._component_id_from_metadata(getattr(fill, "metadata", None))
        if component_id:
            component_rows = dict(self.component_positions.get(component_id) or {})
            old_component_qty = float(component_rows.get(fill.symbol, 0.0))
            new_component_qty = old_component_qty + fill_qty
            if abs(new_component_qty) <= 1e-12:
                component_rows.pop(fill.symbol, None)
            else:
                component_rows[fill.symbol] = new_component_qty
            if component_rows:
                self.component_positions[component_id] = component_rows
            else:
                self.component_positions.pop(component_id, None)

        # Maintain entry price for liquidation model.
        fill_price = None
        if fill.fill_cost is not None and fill.quantity:
            fill_price = float(fill.fill_cost) / float(fill.quantity)
        else:
            fill_price = self.bars.get_latest_bar_value(fill.symbol, "close")

        old_entry = self.entry_prices.get(fill.symbol)
        if abs(new_qty) < 1e-12:
            self.entry_prices[fill.symbol] = None
            self._pending_liquidation.discard(fill.symbol)
            # CRITICAL: clear the funding anchor when the position goes flat.
            # Otherwise _last_funding_ts retains its pre-close value through the
            # entire flat gap, and the first _apply_funding after a reopen
            # back-charges funding for that gap (sign flips for shorts → phantom
            # funding income). Re-anchoring happens lazily in _apply_funding when
            # last_ts is None on the next bar the position is held.
            self._last_funding_ts[fill.symbol] = None
            self._funding_exposure_cursor[fill.symbol] = None
            return

        # Position flip or fresh position: entry resets to current fill price.
        if old_qty == 0 or (old_qty > 0 > new_qty) or (old_qty < 0 < new_qty):
            self.entry_prices[fill.symbol] = fill_price
            self._pending_liquidation.discard(fill.symbol)
            if alpha_funding_anchor is not None:
                self._last_funding_ts[fill.symbol] = alpha_funding_anchor
            self._funding_exposure_cursor[fill.symbol] = self._to_unix_seconds(
                getattr(fill, "timeindex", None)
            )
            return

        # Adding to existing direction updates VWAP entry.
        if old_qty > 0 and fill_qty > 0:
            old_notional = abs(old_qty) * (old_entry if old_entry else fill_price)
            add_notional = abs(fill_qty) * fill_price
            self.entry_prices[fill.symbol] = (old_notional + add_notional) / abs(new_qty)
            return
        if old_qty < 0 and fill_qty < 0:
            old_notional = abs(old_qty) * (old_entry if old_entry else fill_price)
            add_notional = abs(fill_qty) * fill_price
            self.entry_prices[fill.symbol] = (old_notional + add_notional) / abs(new_qty)
            return

        # Reducing existing position keeps original entry until flat.
        if old_entry is None:
            self.entry_prices[fill.symbol] = fill_price

    def update_holdings_from_fill(self, fill):
        fill_dir = 0
        if fill.direction == "BUY":
            fill_dir = 1
        if fill.direction == "SELL":
            fill_dir = -1

        # USE ACTUAL FILL PRICE (realism)
        # If fill_cost is provided, derive unit price from it.
        # Otherwise fallback to bar close (legacy/compatibility).
        if fill.fill_cost is not None and fill.quantity > 0:
            unit_fill_price = fill.fill_cost / fill.quantity
        else:
            unit_fill_price = self.bars.get_latest_bar_value(fill.symbol, "close")

        cost = fill_dir * unit_fill_price * fill.quantity

        commission = fill.commission if fill.commission is not None else 0.0

        self.current_holdings[fill.symbol] += cost
        self.current_holdings["commission"] += commission
        self.current_holdings["cash"] -= cost + commission
        self.current_holdings["total"] -= commission

    def _clamp_reduce_only_fill(self, event):
        """Clamp a reduce-only fill to live-exchange semantics.

        A reduce-only order may only move the position toward zero, never
        through it (Binance rejects/auto-reduces the excess). Returns the fill
        to apply, ``None`` to skip entirely (reduce-only firing from flat or
        the same side), or a proportionally scaled copy when only part of the
        quantity is reducible.
        """
        metadata = getattr(event, "metadata", None) or {}
        if not bool(metadata.get("reduce_only", False)):
            return event
        old_qty = float(self.current_positions.get(event.symbol, 0.0) or 0.0)
        fill_dir = 1.0 if event.direction == "BUY" else -1.0
        if abs(old_qty) <= 1e-12 or old_qty * fill_dir > 0.0:
            return None
        quantity = float(event.quantity)
        if quantity <= 0.0:
            return None
        reducible = min(quantity, abs(old_qty))
        if reducible >= quantity - 1e-12:
            return event
        scale = reducible / quantity
        return replace(
            event,
            quantity=reducible,
            fill_cost=(float(event.fill_cost) * scale) if event.fill_cost is not None else None,
            commission=(float(event.commission) * scale if event.commission is not None else None),
        )

    def update_fill(self, event):
        if event.type == "FILL":
            pricing_trace: ExecutionPricingTrace | None = None
            attribution_sink = self.fill_application_attribution_sink
            synthetic_liquidation = self._is_synthetic_liquidation_fill(event)
            if attribution_sink is not None:
                metadata = getattr(event, "metadata", None)
                if synthetic_liquidation:
                    if isinstance(metadata, Mapping) and "cost_attribution" in metadata:
                        raise RuntimeError(
                            "synthetic liquidation must not carry an execution pricing trace"
                        )
                else:
                    pricing_trace = self._require_pricing_trace(event)

            applied_event = event
            application_status = "applied_unchanged"
            zero_applied_reason = None
            if self.enforce_reduce_only:
                zero_applied_reason = self._reduce_only_zero_reason(event)
                clamped = self._clamp_reduce_only_fill(event)
                if clamped is None:
                    if pricing_trace is not None:
                        self._emit_fill_application_attribution(
                            event,
                            None,
                            pricing_trace,
                            application_status="rejected",
                            zero_applied_reason=zero_applied_reason or "zero_quantity",
                        )
                    return
                applied_event = clamped
                if clamped is not event:
                    application_status = "applied_scaled"
            elif float(getattr(event, "quantity", 0.0) or 0.0) <= 0.0:
                application_status = "rejected"
                zero_applied_reason = "zero_quantity"
                if pricing_trace is not None:
                    self._emit_fill_application_attribution(
                        event,
                        None,
                        pricing_trace,
                        application_status=application_status,
                        zero_applied_reason=zero_applied_reason,
                    )
                if attribution_sink is not None:
                    return
            elif pricing_trace is not None:
                self._emit_fill_application_attribution(
                    event,
                    applied_event,
                    pricing_trace,
                    application_status=application_status,
                    zero_applied_reason=zero_applied_reason,
                )
            if self.enforce_reduce_only and pricing_trace is not None:
                self._emit_fill_application_attribution(
                    event,
                    applied_event,
                    pricing_trace,
                    application_status=application_status,
                    zero_applied_reason=zero_applied_reason,
                )
            event = applied_event
            old_qty = float(self.current_positions.get(event.symbol, 0.0) or 0.0)
            fill_dir = 1.0 if event.direction == "BUY" else -1.0
            new_qty = old_qty + fill_dir * float(event.quantity)
            fill_price = (
                float(event.fill_cost) / float(event.quantity)
                if event.fill_cost is not None and float(event.quantity) > 0.0
                else float(self.bars.get_latest_bar_value(event.symbol, "close") or 0.0)
            )
            # Component book qtys (pre-update: update_positions_from_fill runs
            # below) so the min-hold ledger tracks the book the fill moved.
            fill_component_id = self._component_id_from_metadata(getattr(event, "metadata", None))
            component_old_qty = None
            component_new_qty = None
            if fill_component_id:
                component_old_qty = float(
                    dict(self.component_positions.get(fill_component_id) or {}).get(
                        event.symbol, 0.0
                    )
                )
                component_new_qty = component_old_qty + fill_dir * float(event.quantity)
            self.strategy_quality.note_fill(
                event,
                old_qty,
                new_qty,
                fill_price,
                component_id=fill_component_id,
                component_old_qty=component_old_qty,
                component_new_qty=component_new_qty,
            )
            self.update_positions_from_fill(event)
            self.update_holdings_from_fill(event)
            self.trade_count += 1

            # Log Trade
            # FillEvent: timeindex, symbol, exchange, quantity, direction, fill_cost, commission
            if self.record_trades:
                self.trades.append(
                    {
                        "datetime": event.timeindex,
                        "symbol": event.symbol,
                        "direction": event.direction,
                        "quantity": event.quantity,
                        "fill_cost": event.fill_cost,
                        "commission": event.commission,
                        "price": event.fill_cost / event.quantity if event.quantity > 0 else 0.0,
                        "component_id": self._component_id_from_metadata(
                            getattr(event, "metadata", None)
                        ),
                    }
                )

            self._check_circuit_breaker()

    def _check_circuit_breaker(self):
        """Circuit Breaker: Halt trading if daily loss exceeds threshold."""
        if self.circuit_breaker_tripped:
            return  # Already tripped

        current_equity = self.current_holdings["total"]
        loss_pct = (self.day_start_equity - current_equity) / self.day_start_equity

        if loss_pct >= self.max_daily_loss_pct:
            self.circuit_breaker_tripped = True
            if os.getenv("LQ_BACKTEST_SUPPRESS_CIRCUIT_BREAKER_LOGS", "").strip().lower() not in {
                "1",
                "true",
                "yes",
                "on",
            }:
                print(
                    f"[CIRCUIT BREAKER] Daily loss {loss_pct:.2%} >= {self.max_daily_loss_pct:.2%}. HALTING TRADING."
                )

    def _update_day_boundary(self, latest_datetime):
        cur_day = self._normalize_to_date(latest_datetime)
        if cur_day is None:
            return

        if self._current_day is None:
            self._current_day = cur_day
            return

        if cur_day != self._current_day:
            self._current_day = cur_day
            self.day_start_equity = self.current_holdings["total"]
            self.circuit_breaker_tripped = False

    def _to_unix_seconds(self, value):
        if value is None:
            return None
        if isinstance(value, datetime):
            # Naive datetimes are UTC (core/engine.py convention); localize before
            # .timestamp() so funding/equity epochs are host-tz independent.
            dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
            return dt.timestamp()
        if isinstance(value, date):
            return datetime(value.year, value.month, value.day, tzinfo=UTC).timestamp()
        if isinstance(value, (int, float)):
            ts = float(value)
            if ts > 10_000_000_000:
                ts = ts / 1000.0
            return ts
        try:
            dt = datetime.fromisoformat(str(value))
        except Exception:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.timestamp()

    def _apply_funding(self, latest_datetime):
        if self._live_liquidation_disabled:
            # M5 (live path): funding is a real balance delta reconciled from the
            # exchange account by the trader — never simulate a charge here (that
            # would double-count against the real funding debit). Default False on
            # the backtest path, so this early-out is a no-op there.
            return
        interval_seconds = self.execution_model.cfg.funding_interval_hours * 3600
        default_rate_per_8h = self.execution_model.cfg.funding_rate_per_8h

        now_ts = self._to_unix_seconds(latest_datetime)
        if now_ts is None:
            return
        if self.funding_boundary_resolver is not None:
            self._apply_funding_boundary_resolution(
                datetime.fromtimestamp(now_ts, tz=UTC),
                now_ts=now_ts,
            )
            return

        for symbol in self.symbol_list:
            if self.funding_on_utc_boundary:
                self._apply_utc_boundary_funding(
                    symbol,
                    now_ts=now_ts,
                    interval_seconds=interval_seconds,
                    default_rate_per_8h=default_rate_per_8h,
                )
                continue

            qty = float(self.current_positions.get(symbol, 0.0))
            if abs(qty) < 1e-12:
                continue

            last_ts = self._last_funding_ts.get(symbol)
            if last_ts is None:
                self._last_funding_ts[symbol] = now_ts
                continue
            if now_ts <= last_ts:
                continue

            periods = int((now_ts - last_ts) // interval_seconds)
            if periods <= 0:
                continue

            rate_per_8h = self._resolve_funding_rate(symbol, default=default_rate_per_8h)
            if rate_per_8h is None:
                self._last_funding_ts[symbol] = now_ts
                continue

            price = self.bars.get_latest_bar_value(symbol, "close")
            notional = abs(qty * price)
            if notional <= 0:
                self._last_funding_ts[symbol] = now_ts
                continue

            # Delegate payment computation to the unified ExecutionModel.
            # Returns 0.0 when abs(rate) <= 1e-12 — timestamp still advances below.
            funding_payment = self.execution_model.compute_funding_payment(
                signed_qty=qty,
                price=price,
                periods=periods,
                rate=rate_per_8h,
            )
            self.current_holdings["cash"] -= funding_payment
            self.current_holdings["total"] -= funding_payment
            self.current_holdings["funding"] += funding_payment
            self.total_funding_paid += funding_payment
            self._last_funding_ts[symbol] = last_ts + periods * interval_seconds

    def settle_terminal_funding(self, as_of):
        """Settle boundary liabilities using evidence observable by ``as_of``."""
        self._apply_funding(as_of)
        if any(bool(rows) for rows in self._pending_funding_liabilities.values()):
            raise ValueError("terminal pending funding liability lacks settlement evidence")
        self.reconcile_final_snapshot()

    def _scan_funding_exposure(self, symbol, *, now_ts, quantity):
        """Freeze every boundary exposure in ``(cursor, now]`` before advancing."""
        if not math.isfinite(float(now_ts)):
            raise ValueError("invalid UTC funding exposure timestamp")
        interval_seconds = float(self.execution_model.cfg.funding_interval_hours) * 3600.0
        if not math.isfinite(interval_seconds) or interval_seconds <= 0.0:
            raise ValueError("invalid UTC funding interval")
        cursor = self._funding_exposure_cursor.get(symbol)
        if cursor is None:
            if abs(float(quantity)) >= 1e-12:
                self._funding_exposure_cursor[symbol] = float(now_ts)
            return
        if now_ts < cursor:
            raise ValueError("UTC funding exposure timestamp moved backwards")
        if now_ts == cursor:
            return
        if abs(float(quantity)) >= 1e-12:
            first_boundary = (int(cursor // interval_seconds) + 1) * int(interval_seconds)
            last_boundary = int(now_ts // interval_seconds) * int(interval_seconds)
            liabilities = self._pending_funding_liabilities.setdefault(symbol, {})
            for boundary_ts in range(first_boundary, last_boundary + 1, int(interval_seconds)):
                liabilities.setdefault(int(boundary_ts * 1000), float(quantity))
        self._funding_exposure_cursor[symbol] = float(now_ts)

    def _apply_utc_boundary_funding(
        self,
        symbol,
        *,
        now_ts,
        interval_seconds,
        default_rate_per_8h,
    ):
        """Settle UTC funding from immutable boundary exposures in exact order."""
        interval_ms = int(interval_seconds * 1000)
        liabilities = self._pending_funding_liabilities.setdefault(symbol, {})
        qty = float(self.current_positions.get(symbol, 0.0))
        self._scan_funding_exposure(symbol, now_ts=now_ts, quantity=qty)

        lookup = getattr(self.bars, "_feature_lookup", None)
        sum_fn = getattr(lookup, "funding_fee_sum_between", None)
        for boundary_ms in sorted(liabilities):
            boundary_qty = float(liabilities[boundary_ms])
            coverage = None
            if callable(sum_fn):
                coverage = sum_fn(
                    symbol,
                    start_timestamp_ms=boundary_ms - interval_ms,
                    end_timestamp_ms=min(
                        int(now_ts * 1000),
                        boundary_ms + BINANCE_FUNDING_SOURCE_JITTER_TOLERANCE_MS,
                    ),
                    interval_ms=interval_ms,
                )
            fee_sum, complete = coverage if coverage is not None else (None, False)
            if complete and fee_sum is not None:
                funding_payment = boundary_qty * float(fee_sum)
            else:
                deferred_boundary_ms = getattr(coverage, "deferred_boundary_ms", None)
                if deferred_boundary_ms is not None:
                    break
                if self.require_funding_coverage:
                    raise ValueError(
                        "require_funding_coverage: missing exact funding settlement data "
                        f"for symbol {symbol!r} at boundary {boundary_ms}"
                    )
                rate_per_8h = self._resolve_funding_rate(symbol, default=default_rate_per_8h)
                price = self.bars.get_latest_bar_value(symbol, "close")
                funding_payment = self.execution_model.compute_funding_payment(
                    signed_qty=boundary_qty,
                    price=price,
                    periods=1,
                    rate=rate_per_8h,
                )
            self.current_holdings["cash"] -= funding_payment
            self.current_holdings["total"] -= funding_payment
            self.current_holdings["funding"] += funding_payment
            self.total_funding_paid += funding_payment
            del liabilities[boundary_ms]
            self._last_funding_ts[symbol] = boundary_ms / 1000.0

    def _bar_funding_rate(self, symbol) -> float | None:
        """Return the bar-column funding rate, or ``None`` when genuinely absent.

        ``get_latest_bar_value`` returns a ``0.0`` SENTINEL when neither a bar
        column nor a feature exists, which must not be conflated with genuine
        0.0 funding data (that sentinel silently disabled the configured static
        default and the ``require_funding_coverage`` gate). Only trust the bar
        path when the handler actually carries a ``funding_rate`` column; the
        dynamic feature path is consulted separately by the caller.
        """
        col_idx = getattr(self.bars, "col_idx", None)
        if isinstance(col_idx, dict):
            if "funding_rate" not in col_idx:
                return None
            try:
                value = self.bars.get_latest_bar_value(symbol, "funding_rate")
            except Exception:
                return None
        else:
            # Handlers without a col_idx contract (test doubles, adapters) keep
            # the legacy behavior: a non-None value is trusted as-is.
            declared = getattr(self.bars, "funding_rate", ...)
            if declared is None:
                return None
            try:
                value = (
                    declared
                    if declared is not ...
                    else self.bars.get_latest_bar_value(symbol, "funding_rate")
                )
            except Exception:
                return None
        if value is None:
            return None
        try:
            parsed = float(value)
        except Exception:
            return None
        return parsed if math.isfinite(parsed) else None

    def _resolve_funding_rate(self, symbol, *, default: float) -> float | None:
        # Per-bar funding data is "present" only when a source yields a non-None
        # value. A real 0.0 is genuine data (rate is exactly zero), NOT "absent" —
        # so we must not conflate it with a missing series. Track presence
        # explicitly instead of inferring absence from a 0.0 value.
        getter = getattr(self.bars, "get_latest_feature_value", None)
        if callable(getter):
            try:
                dynamic = getter(symbol, "funding_rate")
            except Exception:
                dynamic = None
            if dynamic is not None:
                try:
                    return float(dynamic)
                except Exception:
                    pass

        fallback = self._bar_funding_rate(symbol)
        if fallback is not None:
            return fallback

        # No per-bar funding data (dynamic feature AND bar column both absent).
        # Honor a configured non-zero static default before any coverage raise:
        # a legitimately configured funding_rate_per_8h is usable coverage.
        if abs(float(default)) > 1e-12:
            return float(default)

        # Audit-hardening: when funding coverage is required and the run is
        # leveraged, refuse to silently charge 0.0 funding because truly no
        # per-bar funding data AND no usable static default were available.
        # Default OFF preserves the legacy silent-0.0 behavior.
        if self.require_funding_coverage and float(self.execution_model.cfg.leverage) > 1.0:
            raise ValueError(
                "require_funding_coverage: no per-bar funding data available for "
                f"symbol {symbol!r} on a leveraged run "
                f"(leverage={float(self.execution_model.cfg.leverage)}); refusing to "
                "charge 0.0 funding silently. Provide funding_rate feature/bar data, "
                "configure a non-zero execution.funding_rate_per_8h default, "
                "or disable execution.require_funding_coverage."
            )

        return None

    @staticmethod
    def _window_extremes_from_event(event) -> dict[str, tuple[float, float, float]]:
        """Map symbol -> (max_high, min_low, last_close) from a MARKET_WINDOW event.

        The windowed data handler advances ``get_latest_bar_value`` to only the
        LAST 1s bar of each ~20s window, so liquidation checks would miss a
        maintenance-margin breach touched anywhere else in the window. When the
        event carries per-second ``bars_1s`` rows, evaluate against the window's
        full extremes instead (long liquidates on the lowest low, short on the
        highest high). Returns an empty dict for non-window events (batch/single
        bar), so those paths keep using ``get_latest_bar_value`` unchanged.
        """
        bars_1s = getattr(event, "bars_1s", None)
        if not isinstance(bars_1s, dict) or not bars_1s:
            return {}

        def _hlc(row):
            if isinstance(row, (tuple, list)) and len(row) >= 6:
                return float(row[2]), float(row[3]), float(row[4])
            if isinstance(row, dict):
                return (
                    float(row.get("high", 0.0)),
                    float(row.get("low", 0.0)),
                    float(row.get("close", 0.0)),
                )
            high = getattr(row, "high", None)
            low = getattr(row, "low", None)
            close = getattr(row, "close", None)
            if high is None or low is None or close is None:
                return None
            return float(high), float(low), float(close)

        extremes: dict[str, tuple[float, float, float]] = {}
        for symbol, rows in bars_1s.items():
            if not rows:
                continue
            highs: list[float] = []
            lows: list[float] = []
            last_close = None
            for row in rows:
                hlc = _hlc(row)
                if hlc is None:
                    continue
                highs.append(hlc[0])
                lows.append(hlc[1])
                last_close = hlc[2]
            if not highs or last_close is None:
                continue
            extremes[str(symbol)] = (max(highs), min(lows), last_close)
        return extremes

    def _cached_liquidation_price(self, symbol, qty, entry_price):
        """Return the position-invariant liquidation price for ``symbol`` (X5).

        The result depends only on ``sign(qty)``, ``entry_price`` and fixed config
        (leverage / MMR / fee / buffer), so it is cached per symbol keyed by
        ``(qty, entry_price)`` and reused for every bar the position is held
        unchanged. On a cache miss — fresh position, size added/reduced, a flip,
        or post-fill invalidation — it recomputes via
        ``ExecutionModel.liquidation_price`` and refreshes the cache. The cached
        value is bit-identical to the uncached call for identical inputs, so the
        backtest golden baseline is unchanged.
        """
        cached = self._liq_price_cache.get(symbol)
        if cached is not None and cached[0] == qty and cached[1] == entry_price:
            return cached[2]
        liq_price = self.execution_model.liquidation_price(qty=qty, entry_price=entry_price)
        self._liq_price_cache[symbol] = (qty, entry_price, liq_price)
        return liq_price

    def _record_modeled_liquidation_breach(
        self,
        *,
        latest_datetime,
        symbol,
        qty,
        entry_price,
        liq_price,
        trigger_price,
        close_price,
        bar_high,
        bar_low,
        configured_margin_mode,
        modeled_margin_mode,
    ):
        """Record a modeled maintenance-margin breach without applying a fill (M5).

        Live-path only: the simulated liquidation engine must never enqueue a
        synthetic ``LIQUIDATED`` fill — a real liquidation, if any, arrives from
        the exchange as a genuine fill. The modeled breach is retained as an
        in-memory audit record plus a WARNING so an operator / alerting layer can
        react, but portfolio state (positions, holdings, the event queue) is left
        untouched. The record is deliberately kept out of ``liquidation_events``
        so it never inflates the real liquidation count.
        """
        self._modeled_liquidation_warnings.append(
            {
                "time": latest_datetime,
                "symbol": symbol,
                "position_qty": qty,
                "entry_price": entry_price,
                "liquidation_price": liq_price,
                "trigger_price": trigger_price,
                "close_price": close_price,
                "bar_high": bar_high,
                "bar_low": bar_low,
                "configured_margin_mode": configured_margin_mode,
                "modeled_margin_mode": modeled_margin_mode,
                "modeled_only": True,
            }
        )
        LOGGER.warning(
            "MODELED_LIQUIDATION_BREACH (live path, no simulated fill applied): "
            "symbol=%s qty=%s entry=%s liq=%s trigger=%s close=%s",
            symbol,
            qty,
            entry_price,
            liq_price,
            trigger_price,
            close_price,
        )

    def _check_liquidations(self, latest_datetime, event=None):
        # leverage <= 1 guard is implicit: execution_model.liquidation_price() returns None.
        configured_margin_mode = (
            str(getattr(self.config, "MARGIN_MODE", "isolated") or "isolated").strip().lower()
        )
        modeled_margin_mode = "isolated"

        window_extremes = self._window_extremes_from_event(event)

        for symbol in self.symbol_list:
            qty = float(self.current_positions.get(symbol, 0.0))
            if abs(qty) < 1e-12:
                continue
            if symbol in self._pending_liquidation:
                continue

            entry_price = self.entry_prices.get(symbol)
            if not entry_price or entry_price <= 0:
                continue

            symbol_extremes = window_extremes.get(symbol)
            if symbol_extremes is not None:
                bar_high, bar_low, close_price = symbol_extremes
            else:
                close_price = self.bars.get_latest_bar_value(symbol, "close")
                bar_high = self.bars.get_latest_bar_value(symbol, "high")
                bar_low = self.bars.get_latest_bar_value(symbol, "low")
            if close_price <= 0:
                continue

            # Delegate liquidation price and breach detection to ExecutionModel.
            # X5: the position-invariant liquidation price is served from a per
            # (qty, entry_price) cache (recomputed only when the position changes)
            # instead of recomputing it every bar — bit-identical result.
            liq_price = self._cached_liquidation_price(symbol, qty, entry_price)
            if liq_price is None:
                # leverage <= 1 — no liquidation possible for this symbol.
                continue

            breached, trigger_price = self.execution_model.check_liquidation(
                qty=qty,
                entry_price=entry_price,
                bar_low=bar_low,
                bar_high=bar_high,
                close_price=close_price,
            )
            if not breached:
                continue

            if self._live_liquidation_disabled:
                # M5: on the live path the simulated liquidation engine must never
                # fabricate a local fill — a real liquidation arrives as a genuine
                # exchange fill. Downgrade the modeled breach to a WARNING / audit
                # record and stop re-evaluating this symbol until its position
                # changes (the _pending_liquidation marker is cleared on the next
                # fill via update_positions_from_fill), then skip the fill path.
                self._record_modeled_liquidation_breach(
                    latest_datetime=latest_datetime,
                    symbol=symbol,
                    qty=qty,
                    entry_price=entry_price,
                    liq_price=liq_price,
                    trigger_price=trigger_price,
                    close_price=close_price,
                    bar_high=bar_high,
                    bar_low=bar_low,
                    configured_margin_mode=configured_margin_mode,
                    modeled_margin_mode=modeled_margin_mode,
                )
                self._pending_liquidation.add(symbol)
                continue

            direction = "SELL" if qty > 0 else "BUY"
            position_side = "LONG" if qty > 0 else "SHORT"
            abs_qty = abs(qty)
            leverage = self.execution_model.cfg.leverage
            fill_cost = trigger_price * abs_qty
            # Single cost path: a forced liquidation is an aggressive (taker) fill
            # at the computed trigger price — route the fee through ExecutionModel
            # rather than re-deriving fill_cost * taker_fee_rate here.
            commission = self.execution_model.commission_for(
                fill_price=trigger_price, qty=abs_qty, is_maker=False
            )
            fill_event = FillEvent(
                timeindex=latest_datetime,
                symbol=symbol,
                exchange="SIM_LIQUIDATION",
                quantity=abs_qty,
                direction=direction,
                fill_cost=fill_cost,
                commission=commission,
                position_side=position_side,
                status="LIQUIDATED",
                metadata={
                    "reason": "maintenance_margin_breach",
                    "entry_price": entry_price,
                    "liquidation_price": liq_price,
                    "trigger_price": trigger_price,
                    "bar_high": bar_high,
                    "bar_low": bar_low,
                    "close_price": close_price,
                    "leverage": leverage,
                    "configured_margin_mode": configured_margin_mode,
                    "modeled_margin_mode": modeled_margin_mode,
                },
            )
            self.events.put(fill_event)
            self.liquidation_events.append(
                {
                    "time": latest_datetime,
                    "symbol": symbol,
                    "position_qty": qty,
                    "entry_price": entry_price,
                    "liquidation_price": liq_price,
                    "trigger_price": trigger_price,
                    "bar_high": bar_high,
                    "bar_low": bar_low,
                    "close_price": close_price,
                    "fill_cost": fill_cost,
                    "commission": commission,
                    "leverage": leverage,
                    "reason": "maintenance_margin_breach",
                    "configured_margin_mode": configured_margin_mode,
                    "modeled_margin_mode": modeled_margin_mode,
                }
            )
            self._pending_liquidation.add(symbol)

    def _normalize_to_date(self, value):
        if value is None:
            return None
        if isinstance(value, date):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, (int, float)):
            # Live feeds often provide milliseconds epoch.
            ts = float(value)
            if ts > 10_000_000_000:
                ts = ts / 1000.0
            try:
                return datetime.fromtimestamp(ts, UTC).date()
            except Exception:
                return None
        try:
            return datetime.fromisoformat(str(value)).date()
        except Exception:
            return None

    def _record_equity_point(self, latest_datetime, total):
        ts = self._to_unix_seconds(latest_datetime)
        if ts is None:
            return
        point = (float(ts), float(total))
        self._equity_points.append(point)
        if self._full_event_equity_sink is not None:
            self._full_event_equity_sink(point)

    def get_rolling_loss_pct(self, window_seconds=3600):
        if window_seconds <= 0 or len(self._equity_points) < 2:
            return 0.0
        now_ts = self._equity_points[-1][0]
        cutoff = float(now_ts) - float(window_seconds)
        window = [point for point in self._equity_points if point[0] >= cutoff]
        if len(window) < 2:
            return 0.0
        start_equity = float(window[0][1])
        end_equity = float(window[-1][1])
        if start_equity <= 0:
            return 0.0
        return max(0.0, (start_equity - end_equity) / start_equity)

    def _get_symbol_limits(self, symbol):
        """Returns fallback limits from config for symbols that don't have exchange metadata."""
        market_spec = {}
        if hasattr(self.bars, "get_market_spec"):
            try:
                market_spec = self.bars.get_market_spec(symbol) or {}
                if market_spec:
                    min_qty = market_spec.get("min_qty")
                    qty_step = market_spec.get("qty_step")
                    min_notional = market_spec.get("min_notional")
                    return {
                        "min_qty": float(min_qty) if min_qty else float(self.config.MIN_TRADE_QTY),
                        "qty_step": float(qty_step)
                        if qty_step
                        else float(self.config.MIN_TRADE_QTY),
                        "min_notional": float(min_notional) if min_notional else 5.0,
                        "price_tick_size": price_tick_size_from_sources(
                            symbol,
                            market_spec=market_spec,
                            config=self.config,
                        ),
                    }
            except Exception:
                pass

        symbol_limits = getattr(self.config, "SYMBOL_LIMITS", {}) or {}
        limits = symbol_limits.get(symbol, {})
        return {
            "min_qty": float(limits.get("min_qty", self.config.MIN_TRADE_QTY)),
            "qty_step": float(limits.get("qty_step", self.config.MIN_TRADE_QTY)),
            "min_notional": float(limits.get("min_notional", 5.0)),
            "price_tick_size": price_tick_size_from_sources(
                symbol,
                market_spec=market_spec,
                config=self.config,
            ),
        }

    def _round_quantity(self, quantity, step):
        return PortfolioSizingService.round_quantity(quantity, step)

    def _risk_based_quantity(self, signal, current_price):
        """Futures-oriented position sizing:
        risk_amount = equity * risk_per_trade
        qty = risk_amount / stop_distance

        When live.go_live_stage='canary', EFFECTIVE_POSITION_FRACTION is set to
        canary_position_fraction (< 1.0) in _build_live_config_namespace.  Multiplying
        here is the single choke point for canary sizing — backtesting and live both
        route through this method via LivePortfolio re-export.
        """
        target_alloc = getattr(self.config, "TARGET_ALLOCATION", self.max_symbol_exposure_pct)
        qty = PortfolioSizingService.risk_based_quantity(
            signal=signal,
            current_price=float(current_price),
            equity=float(self.current_holdings["total"]),
            risk_per_trade=float(self.risk_per_trade),
            default_stop_loss_pct=float(self.default_stop_loss_pct),
            max_symbol_exposure_pct=float(self.max_symbol_exposure_pct),
            target_allocation=float(target_alloc),
            max_order_value=float(self.max_order_value),
            target_allocation_mode=str(self.target_allocation_mode),
            leverage=float(self.leverage),
            max_order_notional_pct=float(self.max_order_notional_pct),
            allow_metadata_risk_override=bool(
                getattr(self.config, "ALLOW_METADATA_RISK_OVERRIDE", False)
            ),
            max_leverage=float(getattr(self.config, "MAX_LEVERAGE", 0.0)),
        )
        # Apply canary position fraction: EFFECTIVE_POSITION_FRACTION == canary_position_fraction
        # when stage=canary, 1.0 otherwise.  Clamped to (0, 1] to prevent zero/negative qty.
        effective_fraction = float(getattr(self.config, "EFFECTIVE_POSITION_FRACTION", 1.0) or 1.0)
        if 0.0 < effective_fraction < 1.0:
            qty = qty * effective_fraction
        return qty

    def _validate_and_round_quantity(self, symbol, quantity, price):
        limits = self._get_symbol_limits(symbol)
        return PortfolioSizingService.validate_and_round_quantity(
            quantity=float(quantity),
            price=float(price),
            min_qty=float(limits["min_qty"]),
            qty_step=float(limits["qty_step"]),
            min_notional=float(limits["min_notional"]),
        )

    def _signal_order_type(self, signal) -> str:
        metadata = dict(getattr(signal, "metadata", {}) or {})
        requested = metadata.get("order_type")
        if requested is None:
            requested = getattr(self.config, "DEFAULT_ORDER_TYPE", "MKT")
        return policy_order_type(
            requested,
            default=getattr(self.config, "DEFAULT_ORDER_TYPE", "MKT"),
            allow_market_orders=bool(getattr(self.config, "ALLOW_MARKET_ORDERS", True)),
        )

    def _order_time_in_force(self, signal, order_type: str) -> str | None:
        if canonical_order_type(order_type, default="LMT") != "LMT":
            return None
        return (
            str(
                getattr(signal, "time_in_force", None)
                or getattr(self.config, "LIMIT_TIME_IN_FORCE", "GTC")
                or "GTC"
            )
            .strip()
            .upper()
        )

    def _order_metadata(self, signal, *, order_type, direction, reference_price, price, limits):
        if canonical_order_type(order_type, default="LMT") != "LMT":
            return dict(getattr(signal, "metadata", {}) or {})
        mode = normalize_limit_price_mode(
            getattr(self.config, "LIMIT_PRICE_MODE", "one_tick_worse")
        )
        return merge_order_policy_metadata(
            getattr(signal, "metadata", None),
            {
                "default_order_type": str(getattr(self.config, "DEFAULT_ORDER_TYPE", "MKT")),
                "resolved_order_type": "LMT",
                "direction": str(direction),
                "limit_price_mode": mode,
                "limit_price_offset_ticks": int(
                    getattr(self.config, "LIMIT_PRICE_OFFSET_TICKS", 1) or 0
                ),
                "limit_reference_price": float(reference_price),
                "limit_price": float(price),
                "price_tick_size": float(limits.get("price_tick_size", 0.0) or 0.0),
                "time_in_force": self._order_time_in_force(signal, order_type),
            },
        )

    def _order_price(
        self, signal, *, symbol: str, direction: str, current_price: float, order_type: str
    ):
        if canonical_order_type(order_type, default="LMT") != "LMT":
            return None, self._get_symbol_limits(symbol)
        reference_price = (
            float(signal.price)
            if getattr(signal, "price", None) is not None
            else float(current_price)
        )
        limits = self._get_symbol_limits(symbol)
        tick_size = float(limits.get("price_tick_size", 0.0) or 0.0)
        price = limit_price_for_direction(
            reference_price=reference_price,
            direction=direction,
            tick_size=tick_size,
            mode=getattr(self.config, "LIMIT_PRICE_MODE", "one_tick_worse"),
            offset_ticks=int(getattr(self.config, "LIMIT_PRICE_OFFSET_TICKS", 1) or 0),
        )
        return price, limits

    def _resolve_stop_loss(self, signal, *, side: str, entry_price: float):
        """Stop-loss for a new entry order.

        Returns ``signal.stop_loss`` verbatim (default behavior). When
        ``attach_default_protective_stop`` is enabled and the signal carries no
        stop, synthesizes a protective stop at ``default_stop_loss_pct`` from the
        entry price so the position never runs naked
        (LONG: entry*(1-pct); SHORT: entry*(1+pct)). Default OFF => unchanged.
        """
        signal_stop = getattr(signal, "stop_loss", None)
        if signal_stop is not None:
            return signal_stop
        if not self.attach_default_protective_stop:
            return None
        pct = float(self.default_stop_loss_pct)
        if pct <= 0.0 or entry_price <= 0.0:
            return None
        if side == "LONG":
            return float(entry_price) * (1.0 - pct)
        return float(entry_price) * (1.0 + pct)

    def _funding_entry_guard_blocks(self, signal, symbol) -> bool:
        """L-D pre-registered rule: block a sub-funding-interval entry that
        would straddle the next 00/08/16 UTC settlement boundary.

        Only fires when the flag is ON and the signal DECLARES an intended
        hold via ``intended_hold_seconds`` (or ``intended_hold_bars`` at the
        config timeframe); undeclared signals are never blocked.
        """
        if not self.funding_entry_guard:
            return False
        # The guard is defined against exchange settlement boundaries. Enabling
        # it without boundary-mode funding would make admission and accounting
        # disagree, so reject rather than silently applying an elapsed-time rule.
        if not self.funding_on_utc_boundary:
            return True
        metadata = dict(getattr(signal, "metadata", {}) or {})
        hold_s = None
        raw_seconds = metadata.get("intended_hold_seconds")
        if raw_seconds is not None:
            try:
                hold_s = float(raw_seconds)
            except TypeError, ValueError:
                return True
        if hold_s is None:
            raw_bars = metadata.get("intended_hold_bars")
            if raw_bars is None:
                return True
            try:
                bars = float(raw_bars)
            except TypeError, ValueError:
                return True
            if not math.isfinite(bars) or bars <= 0.0:
                return True
            timeframe = str(getattr(self.config, "TIMEFRAME", "") or "")
            if not timeframe:
                return True
            try:
                tf_ms = timeframe_to_milliseconds(normalize_timeframe_token(timeframe))
            except TypeError, ValueError:
                return True
            if not math.isfinite(float(tf_ms)) or float(tf_ms) <= 0.0:
                return True
            hold_s = bars * float(tf_ms) / 1000.0
        if hold_s is None or not math.isfinite(hold_s) or hold_s <= 0.0:
            return True
        interval_s = float(self.execution_model.cfg.funding_interval_hours) * 3600.0
        if not math.isfinite(interval_s) or interval_s <= 0.0:
            return True
        if hold_s >= interval_s:
            return False
        now_ts = self._to_unix_seconds(self.bars.get_latest_bar_datetime(symbol))
        if now_ts is None or not math.isfinite(float(now_ts)):
            return True
        # Both MKT and LMT orders first become eligible on the next bar. A limit
        # may fill later, but using the earliest executable timestamp is the
        # conservative, deterministic admission rule. Holds are [start, end):
        # a settlement exactly at end is not crossed.
        timeframe = str(getattr(self.config, "TIMEFRAME", "") or "")
        if not timeframe:
            return True
        try:
            tf_s = float(timeframe_to_milliseconds(normalize_timeframe_token(timeframe))) / 1000.0
        except TypeError, ValueError:
            return True
        if not math.isfinite(tf_s) or tf_s <= 0.0:
            return True
        fill_ts = float(now_ts) + max(0.0, tf_s)
        next_boundary = (int(fill_ts // interval_s) + 1) * interval_s
        return (next_boundary - fill_ts) < hold_s

    def _below_no_trade_band(self, quantity, price) -> bool:
        """True when an order's notional is below the L-C no-trade band."""
        band = float(self.no_trade_band_bps)
        if band <= 0.0:
            return False
        try:
            notional = abs(float(quantity)) * float(price)
        except TypeError, ValueError:
            return False
        equity = float(self.current_holdings.get("total", self.initial_capital) or 0.0)
        if equity <= 0.0:
            return False
        return notional < equity * band / 10_000.0

    def generate_order_from_signal(self, signal) -> OrderEvent | None:
        """Generates an OrderEvent from a SignalEvent.
        Uses risk-based sizing with exchange constraints.
        """
        order = None
        symbol = signal.symbol
        direction = signal.signal_type

        # Get current price to estimate quantity
        current_price = self.bars.get_latest_bar_value(symbol, "close")
        if current_price == 0:
            return None
        quality_decision = self.strategy_quality.apply(
            signal,
            bars=self.bars,
            current_price=float(current_price),
            current_equity=float(self.current_holdings.get("total", self.initial_capital)),
        )
        if quality_decision.signal is None:
            return None
        signal = quality_decision.signal
        symbol = signal.symbol
        direction = signal.signal_type

        position_side = signal.position_side
        if direction == "LONG":
            position_side = position_side or "LONG"
        elif direction == "SHORT":
            position_side = position_side or "SHORT"

        if direction in ("LONG", "SHORT") and self._funding_entry_guard_blocks(signal, symbol):
            return None

        if direction == "LONG":
            qty = self._risk_based_quantity(signal, current_price)
            qty = self._validate_and_round_quantity(symbol, qty, current_price)
            if qty <= 0:
                return None
            if self._below_no_trade_band(qty, current_price):
                return None
            order_type = self._signal_order_type(signal)
            price, limits = self._order_price(
                signal,
                symbol=symbol,
                direction="BUY",
                current_price=current_price,
                order_type=order_type,
            )
            order = OrderEvent(
                symbol=symbol,
                order_type=order_type,
                quantity=qty,
                direction="BUY",
                price=price,
                position_side=position_side,
                reduce_only=False,
                client_order_id=signal.client_order_id,
                stop_loss=self._resolve_stop_loss(signal, side="LONG", entry_price=current_price),
                take_profit=signal.take_profit,
                trailing_percent=signal.trailing_percent,
                time_in_force=self._order_time_in_force(signal, order_type),
                metadata=self._order_metadata(
                    signal,
                    order_type=order_type,
                    direction="BUY",
                    reference_price=signal.price if signal.price is not None else current_price,
                    price=price,
                    limits=limits,
                ),
            )
        elif direction == "SHORT":
            qty = self._risk_based_quantity(signal, current_price)
            qty = self._validate_and_round_quantity(symbol, qty, current_price)
            if qty <= 0:
                return None
            if self._below_no_trade_band(qty, current_price):
                return None
            order_type = self._signal_order_type(signal)
            price, limits = self._order_price(
                signal,
                symbol=symbol,
                direction="SELL",
                current_price=current_price,
                order_type=order_type,
            )
            order = OrderEvent(
                symbol=symbol,
                order_type=order_type,
                quantity=qty,
                direction="SELL",
                price=price,
                position_side=position_side,
                reduce_only=False,
                client_order_id=signal.client_order_id,
                stop_loss=self._resolve_stop_loss(signal, side="SHORT", entry_price=current_price),
                take_profit=signal.take_profit,
                trailing_percent=signal.trailing_percent,
                time_in_force=self._order_time_in_force(signal, order_type),
                metadata=self._order_metadata(
                    signal,
                    order_type=order_type,
                    direction="SELL",
                    reference_price=signal.price if signal.price is not None else current_price,
                    price=price,
                    limits=limits,
                ),
            )
        elif direction == "EXIT":
            metadata = dict(getattr(signal, "metadata", {}) or {})
            component_id = self._component_id_from_metadata(metadata)
            if component_id:
                cur_qty = float(
                    dict(self.component_positions.get(component_id) or {}).get(symbol, 0.0)
                )
            else:
                cur_qty = self.current_positions[symbol]
            if cur_qty != 0:
                try:
                    exit_fraction = float(metadata.get("exit_fraction", 1.0))
                except TypeError, ValueError:
                    exit_fraction = 1.0
                if not math.isfinite(exit_fraction):
                    exit_fraction = 1.0
                exit_fraction = min(1.0, max(0.0, exit_fraction))
                if exit_fraction <= 0.0:
                    return None
                if exit_fraction < 1.0 and self._below_no_trade_band(
                    abs(cur_qty) * exit_fraction, current_price
                ):
                    # Full exits stay exempt from the no-trade band (hygiene);
                    # only sub-band partial trims are dropped as churn.
                    return None
                exit_direction = "SELL" if cur_qty > 0 else "BUY"
                order_type = self._signal_order_type(signal)
                price, limits = self._order_price(
                    signal,
                    symbol=symbol,
                    direction=exit_direction,
                    current_price=current_price,
                    order_type=order_type,
                )
                order = OrderEvent(
                    symbol=symbol,
                    order_type=order_type,
                    quantity=abs(cur_qty) * exit_fraction,
                    direction=exit_direction,
                    price=price,
                    position_side="LONG" if cur_qty > 0 else "SHORT",
                    reduce_only=True,
                    client_order_id=signal.client_order_id,
                    time_in_force=self._order_time_in_force(signal, order_type),
                    metadata=self._order_metadata(
                        signal,
                        order_type=order_type,
                        direction=exit_direction,
                        reference_price=signal.price if signal.price is not None else current_price,
                        price=price,
                        limits=limits,
                    ),
                )

        return order

    def update_signal(self, event):
        if self.circuit_breaker_tripped:
            return  # Do not generate orders when breaker is tripped
        if event.type == "SIGNAL":
            order_event = self.generate_order_from_signal(event)
            if order_event is not None:
                if self.enforce_order_risk_gate and not self._passes_order_risk_gate(order_event):
                    self.strategy_quality.mark_pending_exit_state(
                        str(
                            dict(getattr(event, "metadata", {}) or {}).get("min_hold_exit_key")
                            or ""
                        ),
                        "REJECTED",
                    )
                    return  # Audit-hardening gate rejected the order; skip it.
                self.events.put(order_event)
            else:
                self.strategy_quality.mark_pending_exit_state(
                    str(dict(getattr(event, "metadata", {}) or {}).get("min_hold_exit_key") or ""),
                    "REJECTED",
                )

    def _passes_order_risk_gate(self, order_event) -> bool:
        """Run the live RiskManager.check_order backstop on a backtest order.

        Mirrors the live/trader.py order-time gate so one enforcement path governs
        both. Only invoked when ``enforce_order_risk_gate`` is True; default OFF
        leaves this path untouched and the golden baseline byte-identical.
        """
        if self._risk_manager is None:
            self._risk_manager = RiskManager(self.config)
        current_price = self.bars.get_latest_bar_value(order_event.symbol, "close")
        passed, _reason = self._risk_manager.check_order(order_event, current_price, portfolio=self)
        return bool(passed)

    def create_equity_curve_dataframe(self):
        """Creates a Polars DataFrame from the all_holdings list (list of Tuples)."""
        # Define Schema matches Tuple order
        # (datetime, cash, commission, total, s1, s2, ..., benchmark_price)
        cols = [
            "datetime",
            "cash",
            "commission",
            "funding",
            "total",
            *self.symbol_list,
            "benchmark_price",
        ]

        # Polars handles list of tuples with 'schema' or 'columns' arg
        # Note: If list is empty, this might crash, but typically not in backtest.
        self.equity_curve = pl.DataFrame(self.all_holdings, schema=cols, orient="row")

        # Calculate returns
        self.equity_curve = self.equity_curve.with_columns(
            [(pl.col("total").diff() / pl.col("total").shift(1)).alias("returns")]
        )

        # Calculate Benchmark Returns (Buy & Hold)
        # Using benchmark_price column
        self.equity_curve = self.equity_curve.with_columns(
            [
                (pl.col("benchmark_price").diff() / pl.col("benchmark_price").shift(1)).alias(
                    "benchmark_returns"
                )
            ]
        )

        # Cumprod for equity curve (normalized)
        if len(self.equity_curve) > 0:
            start_val = self.equity_curve["total"][0]
            self.equity_curve = self.equity_curve.with_columns(
                [(pl.col("total") / start_val).alias("equity_curve_norm")]
            )

    def save_equity_curve(self, filename="data/equity.csv"):
        if hasattr(self, "equity_curve") and not self.equity_curve.is_empty():
            parent = os.path.dirname(str(filename))
            if parent:
                os.makedirs(parent, exist_ok=True)
            self.equity_curve.write_csv(str(filename))

    def output_summary_stats(self):
        """Creates a list of summary statistics."""
        return PortfolioPerformanceService.build_summary_stats(
            equity_curve=self.equity_curve,
            config=self.config,
            total_funding_paid=self.total_funding_paid,
            liquidation_count=len(self.liquidation_events),
        )

    def output_summary_stats_fast(self):
        """Return lightweight stats without constructing a DataFrame.

        This is intended for optimization loops where only core objective
        metrics are needed.
        """
        return PortfolioPerformanceService.build_fast_stats(
            metric_totals=self._metric_totals,
            config=self.config,
        )

    def output_trade_log(self, filename="data/trades.csv"):
        """Outputs the trade log to a CSV file."""
        if not self.record_trades or not self.trades:
            # print("No trades generated.") # Optional: don't spam
            return

        df = pl.DataFrame(self.trades)
        parent = os.path.dirname(str(filename))
        if parent:
            os.makedirs(parent, exist_ok=True)
        df.write_csv(str(filename))
        # print(f"Trade log saved to '{filename}'")


__all__ = ["FillApplicationAttribution", "Portfolio"]
