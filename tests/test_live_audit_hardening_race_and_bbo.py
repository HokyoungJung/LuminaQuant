"""Audit-hardening (fix/audit-hardening) regression tests.

Covers three HIGH findings:
  1. Live order/fill-state race — concurrent fill emission must not double-count.
  2. Stale BBO — limit slippage guard must fail closed against an aged quote.
  3. Testnet user-stream host — must point at the testnet WS host when testnet.
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import pytest

from lumina_quant.core.events import OrderEvent
from lumina_quant.live.binance_user_stream import (
    _PROD_WS_BASE_URL,
    _TESTNET_WS_BASE_URL,
    BinanceUserStreamClient,
)
from lumina_quant.live.execution_live import LiveExecutionHandler
from lumina_quant.live.trader import LiveTrader


class _Bars:
    def __init__(self) -> None:
        self._book: dict[str, dict] = {}

    @staticmethod
    def get_latest_bar_value(symbol, val_type):
        _ = (symbol, val_type)
        return 100.0

    @staticmethod
    def get_latest_bar_datetime(symbol):
        _ = symbol
        return datetime(2026, 1, 1)

    def set_book_ticker(self, symbol: str, snapshot: dict | None) -> None:
        if snapshot is None:
            self._book.pop(symbol, None)
        else:
            self._book[symbol] = dict(snapshot)

    def get_latest_book_ticker(self, symbol: str):
        snapshot = self._book.get(symbol)
        return dict(snapshot) if isinstance(snapshot, dict) else None


class _Config:
    EXCHANGE_ID = "BINANCE"
    MARKET_TYPE = "future"
    TAKER_FEE_RATE = 0.0004
    MAKER_FEE_RATE = 0.0002
    ORDER_TIMEOUT = 10
    ORDER_STATE_SOURCE = "user_stream"
    RECONCILIATION_POLL_FALLBACK_ENABLED = True
    ALLOW_MARKET_ORDERS = True
    REQUIRE_BBO_FOR_LIMIT_ORDERS = True
    MAX_BBO_SPREAD_BPS_AT_SUBMIT = 100.0
    MAX_ESTIMATED_ONE_WAY_SLIPPAGE_BPS = 100.0
    PAPER_EXCHANGE_PROTECTIVE_ORDERS = False
    MODE = "paper"


class _Exchange:
    def __init__(self) -> None:
        self.queried = 0

    @staticmethod
    def execute_order(**kwargs):
        _ = kwargs
        return {"id": "ord-1", "status": "open", "filled": 0.0, "amount": 1.0, "price": 100.0}

    @staticmethod
    def fetch_open_orders(symbol=None):
        _ = symbol
        return []

    def fetch_order(self, order_id, symbol=None):
        # Polling fallback observes the order fully filled with the SAME cumulative
        # quantity the user-stream already reported — both paths see filled=1.0.
        _ = (order_id, symbol)
        self.queried += 1
        return {
            "id": "ord-1",
            "status": "closed",
            "filled": 1.0,
            "amount": 1.0,
            "average": 100.0,
            "price": 100.0,
        }

    @staticmethod
    def cancel_order(order_id, symbol=None):
        _ = (order_id, symbol)
        return True


def _fresh_book(symbol: str = "BTCUSDT") -> dict:
    now_ms = int(time.time() * 1000)
    return {
        "symbol": symbol,
        "exchange_ts_ms": now_ms,
        "receive_ts_ms": now_ms,
        "bid_price": 99.9,
        "ask_price": 100.1,
        "bid_quantity": 5.0,
        "ask_quantity": 5.0,
        "mid_price": 100.0,
        "spread_bps": 20.0,
    }


def _drain_fills(events: queue.Queue) -> list:
    fills = []
    while not events.empty():
        evt = events.get_nowait()
        if getattr(evt, "type", "") == "FILL":
            fills.append(evt)
    return fills


# --------------------------------------------------------------------------- #
# Finding 1: concurrent fill emission must not double-count.
# --------------------------------------------------------------------------- #
def test_concurrent_user_stream_and_poll_do_not_double_count_fill():
    events: queue.Queue = queue.Queue()
    handler = LiveExecutionHandler(events, _Bars(), _Config, _Exchange())

    # Track an open order as if execute_order had submitted it.
    order = OrderEvent("BTC/USDT", "MKT", 1.0, "BUY", client_order_id="LQ-race-1")
    handler.tracked_orders["ord-1"] = {
        "event": order,
        "symbol": "BTC/USDT",
        "last_filled": 0.0,
        "state": "OPEN",
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    handler.client_id_to_order["LQ-race-1"] = "ord-1"

    full_fill_payload = {
        "event_type": "executionReport",
        "exchange_ts_ms": 1_700_000_000_000,
        "symbol": "BTCUSDT",
        "order_id": "ord-1",
        "client_order_id": "LQ-race-1",
        "exec_type": "TRADE",
        "order_status": "FILLED",
        "cum_fill_qty": 1.0,
        "last_fill_qty": 1.0,
        "last_fill_price": 100.0,
        "trade_id": 1,
        "side": "BUY",
    }

    barrier = threading.Barrier(2)

    def _stream() -> None:
        barrier.wait()
        handler.ingest_user_stream_event(dict(full_fill_payload))

    def _poll() -> None:
        barrier.wait()
        # Force the polling-fallback path (event=None) to also observe filled=1.0.
        handler.check_open_orders(None)

    t1 = threading.Thread(target=_stream)
    t2 = threading.Thread(target=_poll)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    fills = _drain_fills(events)
    total = round(sum(float(f.quantity) for f in fills), 8)
    assert total == 1.0, (
        f"fill double-counted/lost: emitted total={total} across {len(fills)} fills"
    )
    assert "ord-1" not in handler.tracked_orders


def test_repeated_calls_with_same_fill_emit_once():
    events: queue.Queue = queue.Queue()
    handler = LiveExecutionHandler(events, _Bars(), _Config, _Exchange())

    order = OrderEvent("BTC/USDT", "MKT", 1.0, "BUY", client_order_id="LQ-once-1")
    handler.tracked_orders["ord-1"] = {
        "event": order,
        "symbol": "BTC/USDT",
        "last_filled": 0.0,
        "state": "OPEN",
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    handler.client_id_to_order["LQ-once-1"] = "ord-1"

    payload = {
        "event_type": "executionReport",
        "exchange_ts_ms": 1_700_000_000_000,
        "symbol": "BTCUSDT",
        "order_id": "ord-1",
        "client_order_id": "LQ-once-1",
        "exec_type": "TRADE",
        "order_status": "FILLED",
        "cum_fill_qty": 1.0,
        "last_fill_qty": 1.0,
        "last_fill_price": 100.0,
        "trade_id": 7,
        "side": "BUY",
    }
    # Same event delivered twice (duplicate redelivery) must emit only one fill.
    handler.ingest_user_stream_event(dict(payload))
    handler.ingest_user_stream_event(dict(payload))

    fills = _drain_fills(events)
    assert round(sum(float(f.quantity) for f in fills), 8) == 1.0


# --------------------------------------------------------------------------- #
# Finding 2: stale BBO fail-closed; fresh passes; disabled (0.0) skips.
# --------------------------------------------------------------------------- #
def _limit_order() -> OrderEvent:
    return OrderEvent("BTC/USDT", "LMT", 1.0, "BUY", price=100.0, client_order_id="LQ-bbo-1")


def test_stale_bbo_is_rejected_fail_closed():
    bars = _Bars()
    bars.set_book_ticker("BTC/USDT", _fresh_book())

    class _StaleConfig(_Config):
        MAX_BBO_AGE_SECONDS = 0.05

    handler = LiveExecutionHandler(queue.Queue(), bars, _StaleConfig, _Exchange())
    event = _limit_order()

    # Prime the cache so the snapshot content's first-seen monotonic time is set,
    # then let it age past the limit without the quote changing (websocket stall).
    handler._latest_bbo_snapshot_with_age("BTC/USDT")
    time.sleep(0.12)

    with pytest.raises(RuntimeError, match="stale"):
        handler._validate_limit_slippage_guard(event)
    check = (event.metadata or {}).get("slippage_guard_check") or {}
    assert check.get("status") == "stale_bbo"


def test_fresh_bbo_passes():
    bars = _Bars()
    bars.set_book_ticker("BTC/USDT", _fresh_book())

    class _FreshConfig(_Config):
        MAX_BBO_AGE_SECONDS = 5.0

    handler = LiveExecutionHandler(queue.Queue(), bars, _FreshConfig, _Exchange())
    event = _limit_order()

    handler._validate_limit_slippage_guard(event)
    check = (event.metadata or {}).get("slippage_guard_check") or {}
    assert check.get("status") == "passed"


def test_bbo_age_check_disabled_when_zero_skips():
    bars = _Bars()
    bars.set_book_ticker("BTC/USDT", _fresh_book())

    class _DisabledConfig(_Config):
        MAX_BBO_AGE_SECONDS = 0.0

    handler = LiveExecutionHandler(queue.Queue(), bars, _DisabledConfig, _Exchange())
    event = _limit_order()

    # Even after aging, the legacy (disabled) path must not reject on staleness.
    handler._latest_bbo_snapshot_with_age("BTC/USDT")
    time.sleep(0.05)
    handler._validate_limit_slippage_guard(event)
    check = (event.metadata or {}).get("slippage_guard_check") or {}
    assert check.get("status") == "passed"


def test_invalidate_bbo_cache_forces_missing_then_fail_closed():
    bars = _Bars()
    bars.set_book_ticker("BTC/USDT", _fresh_book())

    class _AgeConfig(_Config):
        MAX_BBO_AGE_SECONDS = 5.0

    handler = LiveExecutionHandler(queue.Queue(), bars, _AgeConfig, _Exchange())
    handler._latest_bbo_snapshot_with_age("BTC/USDT")

    # Simulate websocket gap-recovery wiping the upstream book + cache invalidation.
    bars.set_book_ticker("BTC/USDT", None)
    handler.invalidate_bbo_cache("BTC/USDT")

    event = _limit_order()
    with pytest.raises(RuntimeError, match="fresh BBO snapshot"):
        handler._validate_limit_slippage_guard(event)


# --------------------------------------------------------------------------- #
# Finding 3: testnet user-stream host vs prod by flag.
# --------------------------------------------------------------------------- #
def test_user_stream_ws_host_is_prod_by_default():
    client = BinanceUserStreamClient(exchange=SimpleNamespace())
    assert client.testnet is False
    assert client._ws_base_url() == _PROD_WS_BASE_URL


def test_user_stream_ws_host_is_testnet_when_flag_set():
    exchange = SimpleNamespace(config=SimpleNamespace(IS_TESTNET=True))
    client = BinanceUserStreamClient(exchange=exchange)
    assert client.testnet is True
    assert client._ws_base_url() == _TESTNET_WS_BASE_URL


def test_user_stream_ws_host_prefers_explicit_rest_client_url_over_fallback():
    exchange = SimpleNamespace(
        config=SimpleNamespace(IS_TESTNET=True),
        rest_client=SimpleNamespace(ws_base_url="wss://stream.binancefuture.com"),
    )
    client = BinanceUserStreamClient(exchange=exchange)
    assert client._ws_base_url() == "wss://stream.binancefuture.com"


# --------------------------------------------------------------------------- #
# Finding 1 (trader side): resuming user-stream events relinquishes the
# fallback-poll window so the polling path stops mutating the same order state.
# --------------------------------------------------------------------------- #
class _AuditStore:
    def log_risk_event(self, *args, **kwargs):
        _ = (args, kwargs)


def _build_trader_for_relinquish() -> LiveTrader:
    trader = LiveTrader.__new__(LiveTrader)
    trader.order_state_source = "user_stream"
    trader.reconciliation_poll_fallback_enabled = True
    trader.reconciliation_fallback_window_seconds = 60.0
    trader.user_stream_stale_timeout_seconds = 30.0
    trader._fallback_poll_until_monotonic = 0.0
    trader._last_fallback_reason = ""
    trader._user_stream_last_event_monotonic = time.monotonic()
    trader.audit_store = _AuditStore()
    trader.run_id = "test"
    trader._audit_closed = True
    trader.runtime_cache = SimpleNamespace(update_stream_state=lambda *_a, **_k: None)
    trader.outbox_events = []
    trader._append_outbox = lambda *_a, **_k: None
    return trader


def test_execution_report_relinquishes_active_fallback_window():
    trader = _build_trader_for_relinquish()

    ingested: list[dict] = []
    trader.execution_handler = SimpleNamespace(
        ingest_user_stream_event=lambda payload: ingested.append(payload)
    )

    # Simulate an active fallback window from a prior stall.
    trader._activate_fallback_polling("user_stream_stale")
    assert trader._fallback_poll_until_monotonic > time.monotonic()

    trader._on_user_stream_event(
        {"event_type": "executionReport", "order_id": "ord-1", "exchange_ts_ms": 1}
    )

    assert len(ingested) == 1
    # Window closed -> polling no longer required while the stream is healthy.
    assert trader._fallback_poll_until_monotonic == 0.0
    assert trader._is_fallback_polling_required() is False
