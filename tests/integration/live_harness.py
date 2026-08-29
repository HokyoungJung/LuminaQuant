"""Scripted fake-exchange harness for end-to-end LiveTrader integration tests.

Audit finding C7 (real_money_readiness_audit_20260706): there is no test that
wires the REAL ``LiveExecutionHandler`` + live ``Portfolio`` + ``RiskManager``
under a real trader loop with only the exchange faked. All three historical
live CRITICALs (retry double-send, flatten latch, SIGNAL-before-FILL ordering)
were seam bugs invisible to the ``SimpleNamespace``-stubbed unit tests.

This module provides:

* :class:`ScriptedFakeExchange` — an ``ExchangeInterface`` with deterministic
  fills and injectable faults (retryable 5xx/timeouts, non-retryable rejects
  such as ``-2022`` reduce-only, resting/partial fills).
* :func:`build_live_trader` — constructs a real ``LiveTrader`` (real execution
  handler, real live portfolio, real risk manager) with only the exchange,
  audit store, notifier, state manager and logging faked.
* :func:`drive_events` — drains the trader event queue exactly the way
  ``LiveTrader.run()``'s inner dispatch does (risk check for ORDER events,
  ``process_event`` routing, per-event exception capture) without the blocking
  10s ``queue.get`` / watchdog / user-stream machinery.

The harness is import-only (no test functions); the C7 and V4 test modules
import from it.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import lumina_quant.live.trader as trader_mod
from lumina_quant.configuration import get_default_runtime_config
from lumina_quant.live.execution_live import LiveExecutionHandler
from lumina_quant.live.portfolio import get_live_portfolio_cls
from lumina_quant.live.trader import LiveTrader, _build_live_config_namespace
from lumina_quant.strategies.rsi_strategy import RsiStrategy

DEFAULT_MARK_PRICE = 100.0


class RetryableExchangeError(RuntimeError):
    """Mimics a transient Binance 5xx/timeout (``_is_retryable_exception`` True)."""

    def __init__(self, message: str = "temporarily unavailable", status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class RejectExchangeError(RuntimeError):
    """Mimics a hard Binance rejection (e.g. ``-2022`` ReduceOnly rejected).

    Carries no retryable marker (no 5xx status_code, no timeout/429 token) so
    ``_is_retryable_exception`` returns False and the submit is NOT retried.
    """

    def __init__(self, message: str = "ReduceOnly Order is rejected.", error_code: int = -2022):
        super().__init__(message)
        self.error_code = error_code
        self.status_code = None


class ScriptedFakeExchange:
    """Deterministic, fault-injectable ExchangeInterface stand-in.

    Fill behaviour per ``execute_order``:

    * ``"immediate"`` (default) — the order fully fills on submit (status
      ``closed``), the round-trip that a marketable order takes.
    * ``"rest"`` — the order is accepted but stays open with zero fill (status
      ``new``); it will never fill on its own. Used to model an unfilled
      reduce-only that rests past ORDER_TIMEOUT.
    * ``"partial"`` — the order fills ``partial_qty`` on submit and the rest
      rests open (status ``partially_filled``).

    Fault injection: ``submit_faults`` / ``fetch_faults`` / ``cancel_faults``
    are FIFO deques of Exception instances applied (and consumed) before the
    default behaviour of the matching method. A queued exception is raised; when
    the deque empties the method resumes normal deterministic behaviour.
    """

    def __init__(
        self,
        *,
        symbols: list[str] | None = None,
        fill_mode: str = "immediate",
        partial_qty: float = 0.0,
        mark_price: float = DEFAULT_MARK_PRICE,
        balance: float = 100_000.0,
    ) -> None:
        self.symbols = list(symbols or ["BTC/USDT"])
        self.fill_mode = str(fill_mode)
        self.partial_qty = float(partial_qty)
        self.mark_price = float(mark_price)
        self.balance = float(balance)
        self._next_id = 0
        # order_id -> live state dict (ccxt-shaped for _normalize_order consumers)
        self.orders: dict[str, dict[str, Any]] = {}
        self.submit_faults: deque[Exception] = deque()
        self.fetch_faults: deque[Exception] = deque()
        self.cancel_faults: deque[Exception] = deque()
        # When set, EVERY execute_order raises this (models a persistent rejection
        # such as a standing -2022 reduce-only reject across an entire flatten).
        self.persistent_submit_error: Exception | None = None
        # observability for assertions
        self.submit_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[str] = []
        self.positions: dict[str, float] = {}
        self.position_legs: dict[str, dict[str, float]] = {}

    # -- request plane -------------------------------------------------------
    def execute_order(self, *, symbol, type, side, quantity, price=None, params=None) -> dict:
        self.submit_calls.append(
            {
                "symbol": symbol,
                "type": type,
                "side": side,
                "quantity": float(quantity),
                "price": price,
                "params": dict(params or {}),
            }
        )
        if self.persistent_submit_error is not None:
            raise self.persistent_submit_error
        if self.submit_faults:
            raise self.submit_faults.popleft()

        self._next_id += 1
        order_id = f"EX-{self._next_id}"
        qty = float(quantity)
        avg = float(price or self.mark_price)
        if self.fill_mode == "rest":
            state = {"status": "new", "filled": 0.0}
        elif self.fill_mode == "partial":
            state = {"status": "partially_filled", "filled": max(0.0, min(self.partial_qty, qty))}
        else:  # immediate
            state = {"status": "closed", "filled": qty}
        record = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "type": type,
            "amount": qty,
            "price": avg,
            "average": avg,
            "status": state["status"],
            "filled": state["filled"],
            "remaining": max(0.0, qty - state["filled"]),
            "clientOrderId": (params or {}).get("newClientOrderId")
            or (params or {}).get("clientOrderId"),
        }
        self.orders[order_id] = record
        return dict(record)

    def fetch_order(self, order_id, symbol=None, params=None) -> dict:
        if self.fetch_faults:
            raise self.fetch_faults.popleft()
        # idempotent-retry lookup by clientOrderId returns "not found" here so the
        # handler treats a failed-then-retried submit as safe-to-resubmit.
        if order_id is None:
            return {}
        return dict(self.orders.get(str(order_id), {}))

    def fetch_open_orders(self, symbol=None) -> list[dict]:
        return [
            dict(rec)
            for rec in self.orders.values()
            if str(rec.get("status")) in {"new", "open", "partially_filled"}
        ]

    def cancel_order(self, order_id, symbol=None) -> bool:
        self.cancel_calls.append(str(order_id))
        if self.cancel_faults:
            raise self.cancel_faults.popleft()
        rec = self.orders.get(str(order_id))
        if rec is not None and str(rec.get("status")) in {"new", "open", "partially_filled"}:
            rec["status"] = "canceled"
        return True

    # -- account plane -------------------------------------------------------
    def get_balance(self, currency: str = "USDT") -> float:
        return float(self.balance)

    def get_all_positions(self) -> dict[str, float]:
        return dict(self.positions)

    def get_all_position_legs(self) -> dict[str, dict[str, float]]:
        return {k: dict(v) for k, v in self.position_legs.items()}

    def load_markets(self) -> dict:
        return {sym: {"symbol": sym} for sym in self.symbols}

    def set_leverage(self, symbol, leverage) -> bool:  # pragma: no cover - trivial
        return True

    def set_margin_mode(self, symbol, margin_mode) -> bool:  # pragma: no cover - trivial
        return True


class _FakeDataHandler:
    def __init__(self, events, symbol_list, config, exchange):
        self.events = events
        self.symbol_list = list(symbol_list)
        self.config = config
        self.exchange = exchange
        self.continue_backtest = True

    def get_latest_bar_value(self, symbol, val_type):
        if val_type == "close":
            return DEFAULT_MARK_PRICE
        return 0.0

    def get_latest_bar_datetime(self, symbol):
        return datetime(2026, 1, 1, tzinfo=UTC)

    def get_market_spec(self, symbol):
        return {"price_tick_size": 0.1, "qty_step": 0.001, "min_qty": 0.0, "min_notional": 0.0}

    def shutdown(self, join_timeout=5.0):
        return None


class _FakeStrategy:
    decision_cadence_seconds = 1

    def __init__(self, *args, **kwargs):
        pass

    def calculate_signals(self, event):
        return None

    def get_state(self):
        return {}

    def set_state(self, state):
        return None


class FakeAuditStore:
    def __init__(self, dsn=""):
        self.risk_events: list[tuple[str, dict]] = []
        self.fills: list[Any] = []
        self.orders: list[Any] = []
        self.order_states: list[Any] = []

    def start_run(self, **kwargs):
        return "run-e2e"

    def end_run(self, *args, **kwargs):
        return None

    def log_risk_event(self, run_id, reason=None, details=None):
        self.risk_events.append((str(reason), dict(details or {})))

    def log_order(self, run_id, event, status=None):
        self.orders.append((event, status))

    def log_order_state(self, run_id, payload):
        self.order_states.append(payload)

    def log_fill(self, run_id, event):
        self.fills.append(event)

    def log_equity(self, *args, **kwargs):
        return None

    def log_heartbeat(self, *args, **kwargs):
        return None

    def log_order_reconciliation(self, *args, **kwargs):
        return None

    def close(self):
        return None

    def reasons(self) -> list[str]:
        return [reason for reason, _ in self.risk_events]


class FakeNotifier:
    def __init__(self, *args, **kwargs):
        self.messages: list[str] = []

    def send_message(self, message):
        self.messages.append(str(message))


class SharedStateManager:
    """In-memory StateManager shared between two trader instances (restart test)."""

    file_path = "data/state.json"

    def __init__(self, store: dict | None = None):
        # Bound to a caller-provided dict so a "restarted" trader sees the prior
        # trader's saved payload.
        self._store = store if store is not None else {}

    def load_state(self):
        return dict(self._store.get("state") or {})

    def save_state(self, state):
        self._store["state"] = dict(state)
        return True


def build_live_trader(
    *,
    exchange: ScriptedFakeExchange,
    symbols: list[str] | None = None,
    mode: str = "real",
    order_state_source: str = "polling",
    state_store: dict | None = None,
    config_overrides: dict[str, Any] | None = None,
    strategy_cls: type | None = None,
    strategy_name: str | None = None,
) -> tuple[LiveTrader, ScriptedFakeExchange, FakeAuditStore, FakeNotifier]:
    """Construct a real LiveTrader with only the exchange (+ side effects) faked.

    Returns (trader, exchange, audit_store, notifier). The trader is fully wired
    with the REAL ``LiveExecutionHandler``, the REAL live ``Portfolio`` (M5
    liquidation/funding disabled) and a REAL ``RiskManager``.
    """
    symbols = list(symbols or ["BTC/USDT"])
    rt = get_default_runtime_config()
    # This harness intentionally submits orders through a fake exchange. Model
    # that deployment as canary so the real admission boundary exercises its
    # executable-stage path; get_exchange remains patched below, so no real
    # endpoint can be contacted.
    rt.live.go_live_stage = "canary"
    rt.live.mode = str(mode)
    rt.live.testnet = False
    cfg = _build_live_config_namespace(rt, symbols=symbols)
    cfg.ORDER_STATE_SOURCE = str(order_state_source)
    cfg.ALLOW_MARKET_ORDERS = True
    cfg.DEFAULT_ORDER_TYPE = "MKT"
    cfg.POSTGRES_DSN = ""
    cfg.TELEGRAM_BOT_TOKEN = ""
    cfg.TELEGRAM_CHAT_ID = ""
    cfg.ORDER_TIMEOUT = 1
    for key, value in dict(config_overrides or {}).items():
        setattr(cfg, key, value)

    audit_holder: dict[str, Any] = {}
    notifier_holder: dict[str, Any] = {}

    class _Audit(FakeAuditStore):
        def __init__(self, dsn=""):
            super().__init__(dsn)
            audit_holder["obj"] = self

    class _Notifier(FakeNotifier):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            notifier_holder["obj"] = self

    saved = {
        "LiveConfig": trader_mod.LiveConfig,
        "AuditStore": trader_mod.AuditStore,
        "NotificationManager": trader_mod.NotificationManager,
        "StateManager": trader_mod.StateManager,
        "get_exchange": trader_mod.get_exchange,
        "setup_logging": trader_mod.setup_logging,
    }
    trader_mod.LiveConfig = cfg
    trader_mod.AuditStore = _Audit
    trader_mod.NotificationManager = _Notifier
    trader_mod.StateManager = lambda: SharedStateManager(state_store)
    trader_mod.get_exchange = lambda _cfg: exchange
    trader_mod.setup_logging = lambda _name: logging.getLogger("live-harness")
    try:
        trader = LiveTrader(
            symbol_list=symbols,
            data_handler_cls=_FakeDataHandler,
            execution_handler_cls=LiveExecutionHandler,
            portfolio_cls=get_live_portfolio_cls(),
            strategy_cls=strategy_cls or RsiStrategy,
            strategy_name=strategy_name,
        )
    finally:
        for name, value in saved.items():
            setattr(trader_mod, name, value)

    trader._live_readiness_verified = True
    trader._startup_reconciliation_complete = True
    trader._startup_state = "ready"
    trader._materialized_stale_block_active = False
    trader._data_silence_block_active = False
    now_ns = time.time_ns()
    now_ms = now_ns // 1_000_000
    trader._record_market_freshness(
        SimpleNamespace(
            bars_1s={
                symbol: [
                    (
                        now_ms,
                        DEFAULT_MARK_PRICE,
                        DEFAULT_MARK_PRICE,
                        DEFAULT_MARK_PRICE,
                        DEFAULT_MARK_PRICE,
                        1.0,
                    )
                ]
                for symbol in symbols
            },
            timestamp_ns=now_ns,
            sequence=1,
            lag_ms=0,
            time=now_ms,
            event_time_watermark_ms=now_ms,
            is_stale=False,
        )
    )
    return trader, exchange, audit_holder["obj"], notifier_holder["obj"]


def drive_events(
    trader: LiveTrader,
    *,
    run_risk_check: bool = True,
    max_iterations: int = 200,
) -> dict[str, Any]:
    """Drain and process every queued event the way ``LiveTrader.run()`` does.

    Mirrors the inner dispatch of the real run loop: ORDER events pass through
    ``risk_manager.check_order`` (unless disabled), everything routes through
    ``process_event``; a per-event exception is captured (as the run loop's
    ``except Exception`` branch would) instead of aborting so fault-injection
    scenarios can assert downstream convergence / alerts.
    """
    processed: list[str] = []
    errors: list[Exception] = []
    rejected: list[str] = []
    iterations = 0
    while not trader.events.empty() and iterations < max_iterations:
        iterations += 1
        event = trader.events.get_nowait()
        etype = str(getattr(event, "type", "")).upper()
        try:
            if etype == "ORDER" and run_risk_check:
                price = trader.data_handler.get_latest_bar_value(event.symbol, "close")
                passed, reason = trader.risk_manager.check_order(
                    event, price, portfolio=trader.portfolio
                )
                if not passed:
                    rejected.append(str(reason))
                    continue
            trader.process_event(event)
            processed.append(etype)
        except Exception as exc:  # faithful to run()'s per-event except-branch capture
            errors.append(exc)
    return {
        "processed": processed,
        "errors": errors,
        "rejected": rejected,
        "iterations": iterations,
    }


def open_long_position(trader: LiveTrader, symbol: str, qty: float, price: float) -> None:
    """Seed a real open LONG position through the real fill path (no shortcuts)."""
    from lumina_quant.core.events import FillEvent

    fill = FillEvent(
        timeindex=trader.data_handler.get_latest_bar_datetime(symbol),
        symbol=symbol,
        exchange="TEST",
        quantity=abs(float(qty)),
        direction="BUY",
        fill_cost=abs(float(qty)) * float(price),
        commission=0.0,
        order_id="SEED",
        client_order_id="SEED",
        position_side="LONG",
        status="FILLED",
    )
    trader.portfolio.update_fill(fill)
