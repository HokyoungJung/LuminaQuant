import os
from collections import deque
from datetime import date, datetime

import polars as pl
from lumina_quant.events import FillEvent, OrderEvent
from lumina_quant.market_data import normalize_timeframe_token, timeframe_to_milliseconds
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
        self.default_stop_loss_pct = getattr(config, "DEFAULT_STOP_LOSS_PCT", 0.01)
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
            "trade_count": self.trade_count,
            "trading_frozen": bool(self.trading_frozen),
            "equity_points": list(self._equity_points),
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
        if "trade_count" in state:
            self.trade_count = int(state["trade_count"])
        if "trading_frozen" in state:
            self.trading_frozen = bool(state["trading_frozen"])
        if "equity_points" in state and isinstance(state["equity_points"], list):
            self._equity_points = deque(state["equity_points"], maxlen=20_000)

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
        self._check_liquidations(latest_datetime)

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
            return int(value.timestamp() * 1000)
        if isinstance(value, date):
            return int(datetime(value.year, value.month, value.day).timestamp() * 1000)
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None
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
            return value.timestamp()
        if isinstance(value, date):
            return datetime(value.year, value.month, value.day).timestamp()
        if isinstance(value, (int, float)):
            ts = float(value)
            if ts > 10_000_000_000:
                ts = ts / 1000.0
            return ts
        try:
            dt = datetime.fromisoformat(str(value))
            return dt.timestamp()
        except Exception:
            return None

    def _apply_funding(self, latest_datetime):
        interval_hours = max(1, int(getattr(self.config, "FUNDING_INTERVAL_HOURS", 8)))
        interval_seconds = interval_hours * 3600
        rate_per_8h = float(getattr(self.config, "FUNDING_RATE_PER_8H", 0.0))
        if rate_per_8h == 0.0:
            return

        now_ts = self._to_unix_seconds(latest_datetime)
        if now_ts is None:
            return

        interval_rate = rate_per_8h * (interval_hours / 8.0)
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

            price = self.bars.get_latest_bar_value(symbol, "close")
            notional = abs(qty * price)
            if notional <= 0:
                self._last_funding_ts[symbol] = now_ts
                continue

            # Positive funding rate: longs pay, shorts receive.
            signed = 1.0 if qty > 0 else -1.0
            funding_payment = signed * notional * interval_rate * periods
            self.current_holdings["cash"] -= funding_payment
            self.current_holdings["total"] -= funding_payment
            self.current_holdings["funding"] += funding_payment
            self.total_funding_paid += funding_payment
            self._last_funding_ts[symbol] = last_ts + periods * interval_seconds

    def _check_liquidations(self, latest_datetime):
        leverage = max(1, int(getattr(self.config, "LEVERAGE", 1)))
        mmr = float(getattr(self.config, "MAINTENANCE_MARGIN_RATE", 0.005))
        liq_buffer = float(getattr(self.config, "LIQUIDATION_BUFFER_RATE", 0.0))
        if leverage <= 1:
            return

        fee_rate = float(
            getattr(
                self.config,
                "TAKER_FEE_RATE",
                getattr(self.config, "COMMISSION_RATE", 0.001),
            )
        )

        def calc_liq_price(entry_price, qty):
            """Approximate isolated USDT-M liquidation price with maintenance margin and fee/buffer safety.
            Long  : entry * (1 - 1/L + MMR + fee + buffer)
            Short : entry * (1 + 1/L - MMR - fee - buffer)
            """
            if qty > 0:
                factor = 1.0 - (1.0 / leverage) + mmr + fee_rate + liq_buffer
                factor = max(0.0, min(factor, 1.0))
                return entry_price * factor
            factor = 1.0 + (1.0 / leverage) - mmr - fee_rate - liq_buffer
            factor = max(1.0, factor)
            return entry_price * factor

        for symbol in self.symbol_list:
            qty = float(self.current_positions.get(symbol, 0.0))
            if abs(qty) < 1e-12:
                continue
            if symbol in self._pending_liquidation:
                continue

            entry_price = self.entry_prices.get(symbol)
            if not entry_price or entry_price <= 0:
                continue

            close_price = self.bars.get_latest_bar_value(symbol, "close")
            bar_high = self.bars.get_latest_bar_value(symbol, "high")
            bar_low = self.bars.get_latest_bar_value(symbol, "low")
            if close_price <= 0:
                continue

            liq_price = calc_liq_price(entry_price, qty)

            if qty > 0:
                breached = (bar_low > 0 and bar_low <= liq_price) or close_price <= liq_price
                direction = "SELL"
                position_side = "LONG"
                trigger_price = bar_low if (bar_low > 0 and bar_low <= liq_price) else close_price
            else:
                breached = (bar_high > 0 and bar_high >= liq_price) or close_price >= liq_price
                direction = "BUY"
                position_side = "SHORT"
                trigger_price = (
                    bar_high if (bar_high > 0 and bar_high >= liq_price) else close_price
                )

            if not breached:
                continue

            abs_qty = abs(qty)
            fill_cost = trigger_price * abs_qty
            commission = fill_cost * fee_rate
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
                return datetime.utcfromtimestamp(ts).date()
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
        if hasattr(self.bars, "get_market_spec"):
            try:
                spec = self.bars.get_market_spec(symbol) or {}
                if spec:
                    min_qty = spec.get("min_qty")
                    qty_step = spec.get("qty_step")
                    min_notional = spec.get("min_notional")
                    return {
                        "min_qty": float(min_qty) if min_qty else float(self.config.MIN_TRADE_QTY),
                        "qty_step": float(qty_step)
                        if qty_step
                        else float(self.config.MIN_TRADE_QTY),
                        "min_notional": float(min_notional) if min_notional else 5.0,
                    }
            except Exception:
                pass

        symbol_limits = getattr(self.config, "SYMBOL_LIMITS", {}) or {}
        limits = symbol_limits.get(symbol, {})
        return {
            "min_qty": float(limits.get("min_qty", self.config.MIN_TRADE_QTY)),
            "qty_step": float(limits.get("qty_step", self.config.MIN_TRADE_QTY)),
            "min_notional": float(limits.get("min_notional", 5.0)),
        }

    def _round_quantity(self, quantity, step):
        return PortfolioSizingService.round_quantity(quantity, step)

    def _risk_based_quantity(self, signal, current_price):
        """Futures-oriented position sizing:
        risk_amount = equity * risk_per_trade
        qty = risk_amount / stop_distance
        """
        target_alloc = getattr(self.config, "TARGET_ALLOCATION", self.max_symbol_exposure_pct)
        return PortfolioSizingService.risk_based_quantity(
            signal=signal,
            current_price=float(current_price),
            equity=float(self.current_holdings["total"]),
            risk_per_trade=float(self.risk_per_trade),
            default_stop_loss_pct=float(self.default_stop_loss_pct),
            max_symbol_exposure_pct=float(self.max_symbol_exposure_pct),
            target_allocation=float(target_alloc),
            max_order_value=float(self.max_order_value),
        )

    def _validate_and_round_quantity(self, symbol, quantity, price):
        limits = self._get_symbol_limits(symbol)
        return PortfolioSizingService.validate_and_round_quantity(
            quantity=float(quantity),
            price=float(price),
            min_qty=float(limits["min_qty"]),
            qty_step=float(limits["qty_step"]),
            min_notional=float(limits["min_notional"]),
        )

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
            order = OrderEvent(
                symbol=symbol,
                order_type="MKT",
                quantity=qty,
                direction="BUY",
                position_side=position_side,
                reduce_only=False,
                client_order_id=signal.client_order_id,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                time_in_force=signal.time_in_force,
                metadata=signal.metadata,
            )
        elif direction == "SHORT":
            qty = self._risk_based_quantity(signal, current_price)
            qty = self._validate_and_round_quantity(symbol, qty, current_price)
            if qty <= 0:
                return None
            order = OrderEvent(
                symbol=symbol,
                order_type="MKT",
                quantity=qty,
                direction="SELL",
                position_side=position_side,
                reduce_only=False,
                client_order_id=signal.client_order_id,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                time_in_force=signal.time_in_force,
                metadata=signal.metadata,
            )
        elif direction == "EXIT":
            cur_qty = self.current_positions[symbol]
            if cur_qty != 0:
                order = OrderEvent(
                    symbol=symbol,
                    order_type="MKT",
                    quantity=abs(cur_qty),
                    direction="SELL" if cur_qty > 0 else "BUY",
                    position_side="LONG" if cur_qty > 0 else "SHORT",
                    reduce_only=True,
                    client_order_id=signal.client_order_id,
                    time_in_force=signal.time_in_force,
                    metadata=signal.metadata,
                )

        return order

    def update_signal(self, event):
        if self.circuit_breaker_tripped:
            return  # Do not generate orders when breaker is tripped
        if event.type == "SIGNAL":
            order_event = self.generate_order_from_signal(event)
            if order_event is not None:
                self.events.put(order_event)

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
