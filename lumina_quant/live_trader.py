import atexit
import os
import queue
import time

from lumina_quant.config import LiveConfig
from lumina_quant.engine import TradingEngine
from lumina_quant.events import OrderEvent
from lumina_quant.exchanges import get_exchange
from lumina_quant.interfaces import ExchangeInterface
from lumina_quant.risk_manager import RiskManager
from lumina_quant.runtime_cache import RuntimeCache
from lumina_quant.utils.audit_store import AuditStore
from lumina_quant.utils.logging_utils import setup_logging
from lumina_quant.utils.notification import NotificationManager
from lumina_quant.utils.persistence import StateManager


class LiveTrader(TradingEngine):
    """The LiveTrader engine."""

    def __init__(
        self,
        symbol_list,
        data_handler_cls,
        execution_handler_cls,
        portfolio_cls,
        strategy_cls,
        strategy_params=None,
        strategy_name=None,
        stop_file="",
        external_run_id="",
    ):
        self.logger = setup_logging("LiveTrader")
        self._audit_closed = True
        self.run_id = None
        self.symbol_list = symbol_list
        self.events = queue.Queue()
        self.config = LiveConfig
        self.config.validate()
        default_strategy_name = getattr(strategy_cls, "__name__", strategy_cls.__class__.__name__)
        self.strategy_name = str(strategy_name or default_strategy_name)
        self.strategy_params = dict(strategy_params or {})
        self.stop_file = str(stop_file or "")
        self.external_run_id = str(external_run_id or "")
        self.state_manager = StateManager()
        self.risk_manager = RiskManager(self.config)  # NEW
        self.audit_store: AuditStore = AuditStore(self.config.POSTGRES_DSN)
        self.run_id = self.audit_store.start_run(
            mode="live",
            metadata={
                "symbols": self.symbol_list,
                "exchange": self.config.EXCHANGE,
                "mode": self.config.MODE,
                "strategy": self.strategy_name,
                "strategy_params": self.strategy_params,
                "stop_file": self.stop_file,
            },
            run_id=self.external_run_id or None,
        )
        self._audit_closed = False
        self.heartbeat_interval_sec = max(1, int(self.config.HEARTBEAT_INTERVAL_SEC))
        self.reconciliation_interval_sec = max(
            5,
            int(self.config.RECONCILIATION_INTERVAL_SEC),
        )
        self._last_heartbeat_monotonic = time.monotonic()
        self._last_reconciliation_monotonic = 0.0
        self._last_equity_snapshot_monotonic = 0.0
        self._equity_snapshot_interval_sec = max(
            1,
            int(self.config.HEARTBEAT_INTERVAL_SEC),
        )
        self._last_drift_signature = ()
        self._last_dual_leg_signature = ()
        self._reconciliation_drift_events = 0
        self._flatten_inflight = False
        self._last_order_reconciliation_monotonic = 0.0
        self._order_reconciliation_interval_sec = max(
            5,
            int(self.config.RECONCILIATION_INTERVAL_SEC),
        )
        self.runtime_cache = RuntimeCache()
        self.outbox_events: list[dict] = []
        atexit.register(self._close_audit_store)

        # Initialize Notification Manager
        self.notifier = NotificationManager(
            self.config.TELEGRAM_BOT_TOKEN, self.config.TELEGRAM_CHAT_ID
        )
        self.notifier.send_message(
            f"🚀 **LuminaQuant Started**\nSymbols: {symbol_list}\nStrategy: {self.strategy_name}"
        )

        # Initialize Exchange
        self.logger.info("Initializing Exchange...")
        self.exchange = get_exchange(self.config)

        # Initialize Handlers with Exchange
        self.data_handler = data_handler_cls(
            self.events, self.symbol_list, self.config, self.exchange
        )
        self.execution_handler = execution_handler_cls(
            self.events, self.data_handler, self.config, self.exchange
        )
        if hasattr(self.execution_handler, "set_order_state_callback"):
            self.execution_handler.set_order_state_callback(self._on_order_state)

        self.portfolio = portfolio_cls(self.data_handler, self.events, time.time(), self.config)
        self.strategy = strategy_cls(self.data_handler, self.events, **self.strategy_params)

        # Initialize Base Engine
        super().__init__(
            self.events,
            self.data_handler,
            self.strategy,
            self.portfolio,
            self.execution_handler,
        )

        # Load State
        self._load_state()

    def _load_state(self):
        state = self.state_manager.load_state()
        if state:
            self.logger.info("Restoring state from disk...")
            try:
                if "portfolio" in state:
                    self.portfolio.set_state(state["portfolio"])
                if "strategy" in state:
                    self.strategy.set_state(state["strategy"])
                if isinstance(state.get("runtime_cache"), dict):
                    self.runtime_cache.restore(state["runtime_cache"])
                    if hasattr(self.execution_handler, "rehydrate_orders"):
                        self.execution_handler.rehydrate_orders(self.runtime_cache.open_orders)
                if isinstance(state.get("outbox_events"), list):
                    self.outbox_events = list(state["outbox_events"])
                self.logger.info("State restored.")
            except Exception as e:
                self.logger.error(f"Failed to restore state: {e}")

    def _save_state(self):
        try:
            state = {
                "portfolio": self.portfolio.get_state(),
                "strategy": self.strategy.get_state(),
                "runtime_cache": self.runtime_cache.snapshot(),
                "outbox_events": self.outbox_events[-2000:],
            }
            self.state_manager.save_state(state)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def _is_stop_requested(self):
        if not self.stop_file:
            return False
        return os.path.exists(self.stop_file)

    def _close_audit_store(self, status=None):
        if self._audit_closed:
            return
        try:
            if status and self.run_id:
                self.audit_store.end_run(self.run_id, status=status)
        except Exception:
            pass
        try:
            self.audit_store.close()
        except Exception:
            pass
        self._audit_closed = True

    def _on_order_state(self, state_payload):
        self.runtime_cache.update_order_state(state_payload.get("order_id"), state_payload)
        self._append_outbox("order_state", state_payload)
        try:
            self.audit_store.log_order_state(self.run_id, state_payload)
        except Exception as exc:
            self.logger.error("Failed to persist order state event: %s", exc)
            return
        self._save_state()

        state = str(state_payload.get("state", "")).upper()
        if state not in {"REJECTED", "TIMEOUT", "CANCELED"}:
            return
        details = {
            "state": state,
            "symbol": state_payload.get("symbol"),
            "client_order_id": state_payload.get("client_order_id"),
            "exchange_order_id": state_payload.get("order_id"),
            "message": state_payload.get("message"),
            "metadata": state_payload.get("metadata") or {},
        }
        self.audit_store.log_risk_event(
            self.run_id,
            reason=f"ORDER_{state}",
            details=details,
        )
        msg = f"⚠️ **Order {state}** {details['symbol']} (client_id={details['client_order_id']})"
        self.notifier.send_message(msg)

    def on_fill(self, event):
        """Hook from TradingEngine to save state on fill."""
        self._append_outbox(
            "fill",
            {
                "symbol": event.symbol,
                "direction": event.direction,
                "quantity": event.quantity,
                "fill_cost": event.fill_cost,
                "status": event.status,
                "order_id": event.order_id,
                "timestamp_ns": getattr(event, "timestamp_ns", None),
                "sequence": getattr(event, "sequence", None),
            },
        )
        msg = f"✅ **FILL**: {event.direction} {event.quantity} {event.symbol} @ {event.fill_cost}"
        self.logger.info(msg)
        self.notifier.send_message(msg)
        self.audit_store.log_fill(self.run_id, event)
        self._save_state()

    def _append_outbox(self, event_type: str, payload: dict) -> None:
        item = {
            "event_type": str(event_type),
            "event_time": time.time(),
            "payload": dict(payload or {}),
        }
        self.outbox_events.append(item)
        if len(self.outbox_events) > 5000:
            self.outbox_events = self.outbox_events[-5000:]

    def _emit_heartbeat(self, force=False):
        now_mono = time.monotonic()
        if not force and (now_mono - self._last_heartbeat_monotonic) < self.heartbeat_interval_sec:
            return
        self._last_heartbeat_monotonic = now_mono
        details = {"queue_size": self.events.qsize()}
        if hasattr(self.execution_handler, "tracked_orders"):
            details["tracked_orders"] = len(self.execution_handler.tracked_orders)
        details["reconciliation_drift_events"] = self._reconciliation_drift_events
        self.audit_store.log_heartbeat(self.run_id, status="ALIVE", details=details)

    def _reconcile_positions(self, force=False):
        now_mono = time.monotonic()
        if (
            not force
            and (now_mono - self._last_reconciliation_monotonic) < self.reconciliation_interval_sec
        ):
            return
        self._last_reconciliation_monotonic = now_mono

        try:
            exchange_positions = self.exchange.get_all_positions()
        except Exception as exc:
            self.logger.error("Reconciliation failed to fetch exchange positions: %s", exc)
            self.audit_store.log_risk_event(
                self.run_id,
                reason="RECONCILIATION_ERROR",
                details={"error": str(exc)},
            )
            return

        exchange_position_legs = {}
        if hasattr(self.exchange, "get_all_position_legs"):
            try:
                exchange_position_legs = self.exchange.get_all_position_legs() or {}
            except Exception as exc:
                self.logger.error("Failed to fetch position legs for reconciliation: %s", exc)
                self.audit_store.log_risk_event(
                    self.run_id,
                    reason="POSITION_LEG_RECONCILIATION_ERROR",
                    details={"error": str(exc)},
                )

        local = self.portfolio.current_positions
        self.runtime_cache.update_positions(exchange_positions)
        if exchange_position_legs:
            self.runtime_cache.update_position_legs(exchange_position_legs)
        else:
            self.runtime_cache.update_position_legs({})

        if str(getattr(self.config, "POSITION_MODE", "")).upper() == "HEDGE":
            dual_leg_signature = tuple(
                sorted(
                    (
                        symbol,
                        round(float(legs.get("LONG", 0.0)), 8),
                        round(float(legs.get("SHORT", 0.0)), 8),
                    )
                    for symbol, legs in exchange_position_legs.items()
                    if float(legs.get("LONG", 0.0)) > 0.0 and float(legs.get("SHORT", 0.0)) > 0.0
                )
            )
            if dual_leg_signature and dual_leg_signature != self._last_dual_leg_signature:
                self._last_dual_leg_signature = dual_leg_signature
                self.audit_store.log_risk_event(
                    self.run_id,
                    reason="HEDGE_DUAL_LEG_DETECTED",
                    details={"legs": exchange_position_legs},
                )
            if not dual_leg_signature:
                self._last_dual_leg_signature = ()

        drift = []
        for symbol in self.symbol_list:
            local_qty = float(local.get(symbol, 0.0))
            exchange_qty = float(exchange_positions.get(symbol, 0.0))
            delta = exchange_qty - local_qty
            if abs(delta) > 1e-9:
                drift.append(
                    {
                        "symbol": symbol,
                        "local_qty": local_qty,
                        "exchange_qty": exchange_qty,
                        "delta": delta,
                    }
                )

        signature = tuple(
            sorted(
                (
                    item["symbol"],
                    round(item["local_qty"], 8),
                    round(item["exchange_qty"], 8),
                )
                for item in drift
            )
        )
        if drift and signature != self._last_drift_signature:
            self._last_drift_signature = signature
            self._reconciliation_drift_events += 1
            self.audit_store.log_risk_event(
                self.run_id,
                reason="RECONCILIATION_DRIFT",
                details={
                    "drift_count": len(drift),
                    "drift": drift,
                },
            )
            self.notifier.send_message(
                f"⚠️ **Reconciliation Drift** detected for {len(drift)} symbol(s)."
            )
        if not drift:
            self._last_drift_signature = ()

    def _reconcile_orders(self, force=False):
        now_mono = time.monotonic()
        if (
            not force
            and (now_mono - self._last_order_reconciliation_monotonic)
            < self._order_reconciliation_interval_sec
        ):
            return
        self._last_order_reconciliation_monotonic = now_mono

        if hasattr(self.execution_handler, "reconcile_open_orders"):
            records = self.execution_handler.reconcile_open_orders() or []
            for item in records:
                try:
                    self.audit_store.log_order_reconciliation(self.run_id, item)
                except Exception as exc:
                    self.logger.error("Failed to persist order reconciliation event: %s", exc)

    def _set_trade_freeze(self, *, enabled, reason, details=None):
        self.portfolio.trading_frozen = bool(enabled)
        event_reason = "TRADE_FREEZE_ON" if enabled else "TRADE_FREEZE_OFF"
        self.audit_store.log_risk_event(
            self.run_id,
            reason=event_reason,
            details={"reason": reason, **dict(details or {})},
        )
        status = "ON" if enabled else "OFF"
        self.notifier.send_message(f"⚠️ **Trade Freeze {status}**: {reason}")

    def _queue_reduce_only_order(self, *, symbol, quantity, direction, position_side):
        if float(quantity) <= 1e-12:
            return False
        event = OrderEvent(
            symbol=symbol,
            order_type="MKT",
            quantity=abs(float(quantity)),
            direction=direction,
            position_side=position_side,
            reduce_only=True,
        )
        self.events.put(event)
        return True

    def _flatten_all_positions(self, *, reason, details=None):
        orders_sent = 0
        legs = {}
        if hasattr(self.exchange, "get_all_position_legs"):
            try:
                legs = self.exchange.get_all_position_legs() or {}
            except Exception as exc:
                self.logger.error("Flatten could not fetch position legs: %s", exc)

        for symbol in self.symbol_list:
            payload = legs.get(symbol) if isinstance(legs, dict) else None
            if not isinstance(payload, dict):
                continue
            long_qty = float(payload.get("LONG", 0.0) or 0.0)
            short_qty = float(payload.get("SHORT", 0.0) or 0.0)
            if self._queue_reduce_only_order(
                symbol=symbol,
                quantity=long_qty,
                direction="SELL",
                position_side="LONG",
            ):
                orders_sent += 1
            if self._queue_reduce_only_order(
                symbol=symbol,
                quantity=short_qty,
                direction="BUY",
                position_side="SHORT",
            ):
                orders_sent += 1

        if orders_sent > 0:
            self.audit_store.log_risk_event(
                self.run_id,
                reason="FLATTEN_ALL_TRIGGERED",
                details={"reason": reason, "orders_sent": orders_sent, **dict(details or {})},
            )
            self.notifier.send_message(
                f"🛑 **Flatten All Triggered**: {reason} (orders={orders_sent})"
            )
            return orders_sent

        for symbol in self.symbol_list:
            qty = float(self.portfolio.current_positions.get(symbol, 0.0))
            if self._queue_reduce_only_order(
                symbol=symbol,
                quantity=abs(qty),
                direction="SELL" if qty > 0 else "BUY",
                position_side="LONG" if qty > 0 else "SHORT",
            ):
                orders_sent += 1

        self.audit_store.log_risk_event(
            self.run_id,
            reason="FLATTEN_ALL_TRIGGERED",
            details={"reason": reason, "orders_sent": orders_sent, **dict(details or {})},
        )
        self.notifier.send_message(f"🛑 **Flatten All Triggered**: {reason} (orders={orders_sent})")
        return orders_sent

    def _evaluate_risk_guards(self):
        passed, reason, action, details = self.risk_manager.evaluate_portfolio_risk(self.portfolio)
        if passed:
            self._flatten_inflight = False
            if getattr(self.portfolio, "trading_frozen", False):
                self._set_trade_freeze(enabled=False, reason="risk_recovered", details=details)
            return

        if action == "FREEZE":
            if not getattr(self.portfolio, "trading_frozen", False):
                self._set_trade_freeze(enabled=True, reason=reason, details=details)
            return

        if action == "FLATTEN":
            if not getattr(self.portfolio, "trading_frozen", False):
                self._set_trade_freeze(enabled=True, reason=reason, details=details)
            if not self._flatten_inflight:
                orders_sent = self._flatten_all_positions(reason=reason, details=details)
                self._flatten_inflight = orders_sent > 0
            return

        self.audit_store.log_risk_event(
            self.run_id,
            reason="RISK_POLICY_BREACH",
            details={"reason": reason, **dict(details or {})},
        )

    def _sync_portfolio(self):
        """Syncs the internal portfolio state with the exchange state."""
        if isinstance(self.exchange, ExchangeInterface):
            self.logger.info("Syncing Portfolio with Exchange...")

            try:
                # 1. Sync Cash
                balance = self.exchange.get_balance("USDT")
                if balance > 0:
                    self.logger.info(f"Exchange USDT Balance: {balance}")
                    self.portfolio.current_holdings["cash"] = balance
                    self.runtime_cache.update_account({"cash": float(balance)})
                    # If total is 0 (first run), init with balance.
                    if self.portfolio.current_holdings["total"] == self.portfolio.initial_capital:
                        self.portfolio.initial_capital = balance

                # 2. Sync Positions
                exchange_positions = self.exchange.get_all_positions()
                self.logger.info(f"Exchange Positions: {exchange_positions}")

                for s, qty in exchange_positions.items():
                    if s in self.symbol_list:
                        self.portfolio.current_positions[s] = qty

                # 3. Recalculate Total Holdings (Cash + Position Value)
                total_equity = self.portfolio.current_holdings["cash"]
                for s in self.symbol_list:
                    qty = self.portfolio.current_positions.get(s, 0.0)
                    if qty != 0:
                        last_price = self.data_handler.get_latest_bar_value(s, "close")
                        # If price is 0 (no data yet), we can't value it perfectly.
                        if last_price > 0:
                            total_equity += qty * last_price

                self.portfolio.current_holdings["total"] = total_equity

                # 4. Reconcile Strategy State with Portfolio State
                if hasattr(self.strategy, "bought"):
                    for s in self.symbol_list:
                        position_qty = self.portfolio.current_positions.get(s, 0)
                        strategy_status = self.strategy.bought.get(s, "OUT")

                        if strategy_status != "OUT" and position_qty == 0:
                            msg = f"⚠️ **State Mismatch**: {s} Strategy={strategy_status}, Portfolio=0. Resetting Strategy to OUT."
                            self.logger.warning(msg)
                            self.notifier.send_message(msg)
                            self.strategy.bought[s] = "OUT"
                            self.audit_store.log_risk_event(
                                self.run_id,
                                reason="STATE_MISMATCH",
                                details={
                                    "symbol": s,
                                    "strategy_status": strategy_status,
                                    "position_qty": position_qty,
                                },
                            )
                        elif strategy_status == "OUT" and position_qty != 0:
                            msg = f"⚠️ **State Mismatch**: {s} Strategy=OUT, Portfolio={position_qty}. Syncing Strategy."
                            self.logger.warning(msg)
                            self.notifier.send_message(msg)
                            self.strategy.bought[s] = "LONG" if position_qty > 0 else "SHORT"
                            self.audit_store.log_risk_event(
                                self.run_id,
                                reason="STATE_MISMATCH",
                                details={
                                    "symbol": s,
                                    "strategy_status": strategy_status,
                                    "position_qty": position_qty,
                                },
                            )

                self.logger.info(f"Portfolio Sync Completed. Total Equity: {total_equity}")
            except Exception as e:
                self.logger.error(f"Portfolio Sync Failed: {e}")
                self.audit_store.log_risk_event(
                    self.run_id,
                    reason="EXCHANGE_SYNC_ERROR",
                    details={"error": str(e)},
                )

    def handle_market_event(self, event):
        """Override to save equity curve on every bar."""
        self.runtime_cache.update_market(
            event.symbol,
            {
                "time": event.time,
                "timestamp_ns": getattr(event, "timestamp_ns", None),
                "sequence": getattr(event, "sequence", None),
                "close": float(event.close),
                "volume": float(event.volume),
            },
        )
        super().handle_market_event(event)

        # Save Live Equity snapshot periodically to reduce per-bar overhead.
        now_mono = time.monotonic()
        should_snapshot = (
            now_mono - self._last_equity_snapshot_monotonic >= self._equity_snapshot_interval_sec
        )
        if should_snapshot:
            self._last_equity_snapshot_monotonic = now_mono
            self.portfolio.create_equity_curve_dataframe()
            if self.config.STORAGE_EXPORT_CSV:
                self.portfolio.save_equity_curve(os.path.join("data", "live_equity.csv"))
        self.audit_store.log_equity(
            self.run_id,
            timeindex=event.time,
            total=self.portfolio.current_holdings.get("total", 0.0),
            cash=self.portfolio.current_holdings.get("cash", 0.0),
            metadata={"symbol": event.symbol},
        )

    def handle_fill_event(self, event):
        """Override to save trades on every fill."""
        super().handle_fill_event(event)  # This calls self.on_fill(event) too

        # Save Live Trades
        if self.config.STORAGE_EXPORT_CSV:
            self.portfolio.output_trade_log(os.path.join("data", "live_trades.csv"))

    def run(self):
        """Main Live Trading Loop."""
        self.logger.info(f"Starting Live Trading on {self.symbol_list}...")
        self._sync_portfolio()

        while True:
            try:
                if self._is_stop_requested():
                    self.logger.warning("Stop file detected. Shutting down live trader gracefully.")
                    self.notifier.send_message(
                        "⏹️ **Stop Requested** via control file. Stopping trader."
                    )
                    self._save_state()
                    self.data_handler.continue_backtest = False
                    if hasattr(self.data_handler, "ws_running"):
                        self.data_handler.ws_running = False
                    self._close_audit_store(status="STOPPED")
                    break

                self._emit_heartbeat(force=False)
                self._reconcile_positions(force=False)
                self._reconcile_orders(force=False)
                self._evaluate_risk_guards()
                # In Live mode, data_handler is threaded and pushes events autonomously.
                # We just blocking-wait for events.
                event = self.events.get(True, timeout=10)  # Wait up to 10s

                # Add logging for events if needed (optional overlay on process_event)
                # But engine.py handles basic routing.
                # We might want logging wrappers?
                # For now, rely on engine.
                # But wait, engine doesn't log!
                # I should override handle_xxx methods or add logging to engine.
                # To be Clean OOP, let's keep engine distinct and perhaps add logging there or here.
                # simpler: just log here then call process.

                if event is not None:
                    if event.type == "MARKET":
                        self.logger.debug(f"Market Event: {event.symbol}")
                    elif event.type == "SIGNAL":
                        self.logger.info(f"Signal Event: {event.signal_type}")
                    if event.type == "ORDER":
                        self.logger.info(f"Order Event: {event.direction}")
                        # RISK CHECK
                        current_price = self.data_handler.get_latest_bar_value(
                            event.symbol, "close"
                        )
                        passed, reason = self.risk_manager.check_order(
                            event, current_price, portfolio=self.portfolio
                        )
                        if not passed:
                            msg = f"⛔ **Risk Reject**: {reason}"
                            self.logger.warning(msg)
                            self.notifier.send_message(msg)
                            self.audit_store.log_risk_event(
                                self.run_id,
                                reason="ORDER_REJECTED",
                                details={
                                    "symbol": event.symbol,
                                    "direction": event.direction,
                                    "quantity": event.quantity,
                                    "reason": reason,
                                },
                            )
                            continue  # Skip processing this event
                        self.audit_store.log_order(self.run_id, event, status="NEW")

                    self.process_event(event)

            except queue.Empty:
                # Heartbeat
                self._emit_heartbeat(force=True)
                self._reconcile_positions(force=False)
                self._reconcile_orders(force=False)
                self._evaluate_risk_guards()
                if hasattr(self.execution_handler, "check_open_orders"):
                    self.execution_handler.check_open_orders(None)
            except KeyboardInterrupt:
                self.logger.info("Stopping Live Trader...")
                self._save_state()  # Save on exit
                self.data_handler.continue_backtest = False
                if hasattr(self.data_handler, "ws_running"):
                    self.data_handler.ws_running = False
                self._close_audit_store(status="STOPPED")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                self._save_state()  # Save on crash attempt
                self.audit_store.log_risk_event(
                    self.run_id, reason="MAIN_LOOP_ERROR", details={"error": str(e)}
                )

    def __del__(self):
        self._close_audit_store(status="STOPPED")
