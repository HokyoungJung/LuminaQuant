import os
from collections import deque
from datetime import UTC, date, datetime

import polars as pl
from lumina_quant.backtesting.execution_model import (
    ExecutionModel,
    ExecutionModelConfig,
    _config_from_attrs,
)
from lumina_quant.core.events import FillEvent, OrderEvent
from lumina_quant.core.order_policy import (
    canonical_order_type,
    limit_price_for_direction,
    merge_order_policy_metadata,
    normalize_limit_price_mode,
    policy_order_type,
    price_tick_size_from_sources,
)
from lumina_quant.market_data import normalize_timeframe_token, timeframe_to_milliseconds
from lumina_quant.risk_manager import RiskManager
from lumina_quant.services.portfolio import PortfolioPerformanceService, PortfolioSizingService


class Portfolio:
    """The Portfolio class handles the positions and market value.
    Refactored to use Polars for equity curve storage.
    """

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
    ):
        self.bars = bars
        self.events = events
        self.config = config
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
                self._sampling_interval_ms = int(timeframe_to_milliseconds(self.sampling_timeframe))
            except Exception:
                self._sampling_interval_ms = None
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
        # Lazily-constructed RiskManager backstop for the order-time gate (only when
        # enforce_order_risk_gate is True). Mirrors the live/trader.py:1713 usage.
        self._risk_manager = None
        # Phase 4 unified cost model — funding and liquidation delegate to this.
        # BacktestConfigView carries ._rt (RuntimeConfig); use from_runtime for production.
        # Plain class configs (unit tests) use _config_from_attrs.
        _rt = getattr(config, "_rt", None)
        self.execution_model = ExecutionModel(
            ExecutionModelConfig.from_runtime(_rt)
            if _rt is not None
            else _config_from_attrs(config)
        )
        self._current_day = None
        self._last_funding_ts = dict.fromkeys(self.symbol_list)
        self.total_funding_paid = 0.0
        self.entry_prices = dict.fromkeys(self.symbol_list)
        self.liquidation_events = []
        self._pending_liquidation = set()
        self._metric_totals = [float(self.initial_capital)] if self.track_metrics else []
        self._metric_benchmarks = [0.0] if self.track_metrics else []
        self._equity_points = deque(maxlen=20_000)
        self.trading_frozen = False
        self.component_positions = {}

        # Initialize first record
        self.update_initial_record()

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
            "last_funding_ts": self._last_funding_ts,
            "pending_liquidation": list(self._pending_liquidation),
            "liquidation_events": list(self.liquidation_events),
            "trade_count": self.trade_count,
            "trading_frozen": bool(self.trading_frozen),
            "equity_points": list(self._equity_points),
            "component_positions": self.component_positions,
            "last_sample_timestamp_ms": self._last_sample_timestamp_ms,
            "current_day": self._current_day,
            "day_start_equity": self.day_start_equity,
        }

    def set_state(self, state):
        if "positions" in state:
            self.current_positions = state["positions"]
        if "holdings" in state:
            self.current_holdings = state["holdings"]
        if "initial_capital" in state:
            self.initial_capital = state["initial_capital"]
        if "circuit_breaker_tripped" in state:
            self.circuit_breaker_tripped = state["circuit_breaker_tripped"]
        if "entry_prices" in state and isinstance(state["entry_prices"], dict):
            self.entry_prices = state["entry_prices"]
        if "total_funding_paid" in state:
            self.total_funding_paid = float(state["total_funding_paid"])
        if "last_funding_ts" in state and isinstance(state["last_funding_ts"], dict):
            self._last_funding_ts = state["last_funding_ts"]
        if "pending_liquidation" in state:
            self._pending_liquidation = set(state["pending_liquidation"])
        if "liquidation_events" in state and isinstance(state["liquidation_events"], list):
            self.liquidation_events = list(state["liquidation_events"])
        if "trade_count" in state:
            self.trade_count = int(state["trade_count"])
        if "trading_frozen" in state:
            self.trading_frozen = bool(state["trading_frozen"])
        if "equity_points" in state and isinstance(state["equity_points"], list):
            self._equity_points = deque(state["equity_points"], maxlen=20_000)
        if "component_positions" in state and isinstance(state["component_positions"], dict):
            self.component_positions = {
                str(component): {
                    str(symbol): float(quantity) for symbol, quantity in dict(symbols or {}).items()
                }
                for component, symbols in dict(state["component_positions"]).items()
            }
        if "last_sample_timestamp_ms" in state:
            raw_last_sample = state.get("last_sample_timestamp_ms")
            try:
                self._last_sample_timestamp_ms = (
                    int(raw_last_sample) if raw_last_sample is not None else None
                )
            except Exception:
                self._last_sample_timestamp_ms = None
        if "current_day" in state:
            self._current_day = state.get("current_day")
        if "day_start_equity" in state:
            try:
                self.day_start_equity = float(state.get("day_start_equity"))
            except Exception:
                pass

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
        runtime = getattr(config, "_rt", None)
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

    def update_timeindex(self, event):
        """Updates the positions from the current locations to the
        latest available bar.
        """
        _ = event
        primary_symbol = self.symbol_list[0]
        latest_datetime = self.bars.get_latest_bar_datetime(primary_symbol)
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
        fill_dir = 0
        if fill.direction == "BUY":
            fill_dir = 1
        if fill.direction == "SELL":
            fill_dir = -1

        old_qty = float(self.current_positions.get(fill.symbol, 0.0))
        fill_qty = float(fill.quantity) * fill_dir
        new_qty = old_qty + fill_qty
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
            return

        # Position flip or fresh position: entry resets to current fill price.
        if old_qty == 0 or (old_qty > 0 > new_qty) or (old_qty < 0 < new_qty):
            self.entry_prices[fill.symbol] = fill_price
            self._pending_liquidation.discard(fill.symbol)
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

    def update_fill(self, event):
        if event.type == "FILL":
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
        interval_seconds = self.execution_model.cfg.funding_interval_hours * 3600
        default_rate_per_8h = self.execution_model.cfg.funding_rate_per_8h

        now_ts = self._to_unix_seconds(latest_datetime)
        if now_ts is None:
            return

        for symbol in self.symbol_list:
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

        try:
            fallback = self.bars.get_latest_bar_value(symbol, "funding_rate")
        except Exception:
            fallback = None
        if fallback is not None:
            try:
                return float(fallback)
            except Exception:
                pass

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
            liq_price = self.execution_model.liquidation_price(qty=qty, entry_price=entry_price)
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
                    "close_price": close_price,
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
        self._equity_points.append((float(ts), float(total)))

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

        position_side = signal.position_side
        if direction == "LONG":
            position_side = position_side or "LONG"
        elif direction == "SHORT":
            position_side = position_side or "SHORT"

        if direction == "LONG":
            qty = self._risk_based_quantity(signal, current_price)
            qty = self._validate_and_round_quantity(symbol, qty, current_price)
            if qty <= 0:
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
            component_id = self._component_id_from_metadata(signal.metadata)
            if component_id:
                cur_qty = float(
                    dict(self.component_positions.get(component_id) or {}).get(symbol, 0.0)
                )
            else:
                cur_qty = self.current_positions[symbol]
            if cur_qty != 0:
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
                    quantity=abs(cur_qty),
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
                    return  # Audit-hardening gate rejected the order; skip it.
                self.events.put(order_event)

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


__all__ = ["Portfolio"]
