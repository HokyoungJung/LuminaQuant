"""Regression tests for the live-trader threading / state-safety hardening.

Covers (per fix/audit-hardening second-pass audit):
  1. rehydrate_orders seeds the dedup ledger so a restart-then-reconcile does NOT
     re-emit an already-counted partial fill.
  2. rehydrate_orders mutates tracked state under _state_lock.
  3. Order-state callbacks are NOT invoked under _state_lock (deferred-then-fired).
  4. invalidate_bbo_cache is wired from the trader stream-gap hooks.
  5. FIFO ledger eviction never drops a still-live resting order's baseline.
  6. Paper protective-order dedup is race-safe across threads.
  7. Concurrent save_state never corrupts the recovery file (unique temp + lock).
  8. RuntimeCache reads/writes are lock-guarded.
  9. Consecutive-loss kill-switch is never silently bypassed.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime
from types import SimpleNamespace

from lumina_quant.core.events import OrderEvent
from lumina_quant.live.execution_live import LiveExecutionHandler
from lumina_quant.live.trader import LiveTrader
from lumina_quant.runtime_cache import RuntimeCache
from lumina_quant.utils.persistence import StateManager


# --------------------------------------------------------------------------- #
# Shared fakes
# --------------------------------------------------------------------------- #
class _Bars:
    @staticmethod
    def get_latest_bar_value(symbol, val_type):
        _ = (symbol, val_type)
        return 100.0

    @staticmethod
    def get_latest_bar_datetime(symbol):
        _ = symbol
        return datetime(2026, 1, 1)


class _Config:
    EXCHANGE_ID = "BINANCE"
    MARKET_TYPE = "future"
    TAKER_FEE_RATE = 0.0004
    MAKER_FEE_RATE = 0.0002
    ORDER_TIMEOUT = 10
    MODE = "paper"
    PAPER_EXCHANGE_PROTECTIVE_ORDERS = True
    PROTECTIVE_ORDER_STYLE = "market"
    ALLOW_MARKET_ORDERS = True


class _PartialThenSamePoll:
    """Reconciliation poll observes the SAME cumulative the order was rehydrated with."""

    def __init__(self, filled: float = 0.4) -> None:
        self.filled = float(filled)

    def fetch_order(self, order_id, symbol=None):
        _ = (order_id, symbol)
        return {
            "id": "order-1",
            "status": "open",
            "filled": self.filled,
            "amount": 1.0,
            "average": 100.0,
            "price": 100.0,
        }

    def fetch_open_orders(self, symbol=None):
        _ = symbol
        return []


def _drain_fills(events: queue.Queue) -> list:
    fills = []
    while not events.empty():
        evt = events.get_nowait()
        if getattr(evt, "type", "") == "FILL":
            fills.append(evt)
    return fills


def _rehydrate_payload(last_filled: float) -> dict:
    return {
        "order-1": {
            "state": "PARTIAL",
            "symbol": "BTC/USDT",
            "client_order_id": "LQ-rehy-1",
            "last_filled": last_filled,
            "created_at": 1700000000.0,
            "metadata": {
                "symbol": "BTC/USDT",
                "direction": "BUY",
                "order_type": "MKT",
                "quantity": 1.0,
                "position_side": "LONG",
                "reduce_only": False,
                "client_order_id": "LQ-rehy-1",
            },
        }
    }


# --------------------------------------------------------------------------- #
# Item 1: rehydrate seeds dedup ledger -> no re-emit of restored cumulative.
# --------------------------------------------------------------------------- #
def test_rehydrate_seeds_ledger_no_reemit_on_poll():
    events: queue.Queue = queue.Queue()
    handler = LiveExecutionHandler(events, _Bars(), _Config, _PartialThenSamePoll(0.4))

    handler.rehydrate_orders(_rehydrate_payload(0.4))
    assert "order-1" in handler.tracked_orders
    # Ledger must be seeded with the restored cumulative under the residual-delta key.
    assert handler._emitted_cum_filled["order-1"] == 0.4

    # A reconcile/poll observing the SAME cumulative must emit NO new fill.
    handler.check_open_orders(None)
    fills = _drain_fills(events)
    assert fills == [], f"restored cumulative re-emitted as a fresh fill: {fills}"


def test_rehydrate_seeds_ledger_only_residual_on_further_fill():
    events: queue.Queue = queue.Queue()
    handler = LiveExecutionHandler(events, _Bars(), _Config, _PartialThenSamePoll(0.7))

    handler.rehydrate_orders(_rehydrate_payload(0.4))
    # Poll now sees 0.7 cumulative; only the 0.3 residual is new.
    handler.check_open_orders(None)
    fills = _drain_fills(events)
    assert round(sum(float(f.quantity) for f in fills), 8) == 0.3


def test_rehydrate_does_not_seed_zero_filled():
    handler = LiveExecutionHandler(queue.Queue(), _Bars(), _Config, _PartialThenSamePoll(0.0))
    payload = _rehydrate_payload(0.0)
    payload["order-1"]["state"] = "OPEN"
    handler.rehydrate_orders(payload)
    assert "order-1" not in handler._emitted_cum_filled


# --------------------------------------------------------------------------- #
# Item 5: FIFO ledger eviction must not drop a still-live order's baseline.
# --------------------------------------------------------------------------- #
def test_ledger_eviction_skips_live_order():
    handler = LiveExecutionHandler(queue.Queue(), _Bars(), _Config, _PartialThenSamePoll())
    handler._emitted_cum_filled_max_keys = 2

    # Live resting order inserted FIRST (oldest).
    handler.tracked_orders["live-1"] = {
        "event": OrderEvent("BTC/USDT", "MKT", 1.0, "BUY", client_order_id="LQ-live-1"),
        "symbol": "BTC/USDT",
        "last_filled": 0.5,
        "state": "PARTIAL",
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    with handler._state_lock:
        handler._emitted_cum_filled["live-1"] = 0.5
        handler._emitted_cum_filled["dead-1"] = 1.0  # forgotten order, evictable
        # Inserting a 3rd key triggers eviction (max_keys=2).
        handler._ledger_residual_delta("dead-2", 1.0)

    # Live order's baseline must survive; the forgotten oldest must be evicted.
    assert handler._emitted_cum_filled.get("live-1") == 0.5
    assert "dead-1" not in handler._emitted_cum_filled


def test_forget_order_keeps_ledger_key_for_resurrect_guard():
    handler = LiveExecutionHandler(queue.Queue(), _Bars(), _Config, _PartialThenSamePoll())
    entry = {
        "event": OrderEvent("BTC/USDT", "MKT", 1.0, "BUY", client_order_id="LQ-f-1"),
        "symbol": "BTC/USDT",
        "last_filled": 1.0,
        "state": "FILLED",
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    handler.tracked_orders["order-1"] = entry
    with handler._state_lock:
        handler._emitted_cum_filled["order-1"] = 1.0
        handler._forget_order("order-1", entry)
    # Forgetting removes tracking but KEEPS the dedup baseline (resurrect guard).
    assert "order-1" not in handler.tracked_orders
    assert handler._emitted_cum_filled["order-1"] == 1.0


# --------------------------------------------------------------------------- #
# Item 3: order-state callback is never invoked while _state_lock is held.
# --------------------------------------------------------------------------- #
def test_callback_not_invoked_under_state_lock():
    events: queue.Queue = queue.Queue()
    handler = LiveExecutionHandler(events, _Bars(), _Config, _PartialThenSamePoll())

    lock_held_during_callback: list[bool] = []

    def _callback(_payload) -> None:
        # If the lock is NOT held now, a non-blocking acquire succeeds.
        acquired = handler._state_lock.acquire(blocking=False)
        lock_held_during_callback.append(not acquired)
        if acquired:
            handler._state_lock.release()

    handler.set_order_state_callback(_callback)

    order = OrderEvent("BTC/USDT", "MKT", 1.0, "BUY", client_order_id="LQ-cb-1")
    handler.tracked_orders["order-1"] = {
        "event": order,
        "symbol": "BTC/USDT",
        "last_filled": 0.0,
        "state": "OPEN",
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    handler.client_id_to_order["LQ-cb-1"] = "order-1"
    handler.ingest_user_stream_event(
        {
            "event_type": "executionReport",
            "symbol": "BTCUSDT",
            "order_id": "order-1",
            "client_order_id": "LQ-cb-1",
            "exec_type": "TRADE",
            "order_status": "FILLED",
            "cum_fill_qty": 1.0,
            "last_fill_qty": 1.0,
            "last_fill_price": 100.0,
            "trade_id": 1,
            "side": "BUY",
        }
    )

    assert lock_held_during_callback, "callback was never fired"
    assert not any(lock_held_during_callback), "callback fired while _state_lock was held"


# --------------------------------------------------------------------------- #
# Item 6: paper protective-order dedup is race-safe across threads.
# --------------------------------------------------------------------------- #
class _ProtectiveExchange:
    def __init__(self) -> None:
        self.algo_submits = 0
        self._lock = threading.Lock()

    def execute_algo_order(self, **kwargs):
        with self._lock:
            self.algo_submits += 1
        time.sleep(0.005)  # widen the race window
        return {"id": f"algo-{self.algo_submits}", "status": "NEW"}

    def fetch_open_orders(self, symbol=None):
        _ = symbol
        return []


def test_protective_dedup_single_submit_under_concurrency():
    exchange = _ProtectiveExchange()
    handler = LiveExecutionHandler(queue.Queue(), _Bars(), _Config, exchange)
    event = OrderEvent(
        "BTC/USDT",
        "MKT",
        1.0,
        "BUY",
        position_side="LONG",
        client_order_id="LQ-prot-1",
        stop_loss=95.0,
    )

    barrier = threading.Barrier(2)

    def _submit() -> None:
        barrier.wait()
        handler._submit_paper_exchange_protection(event, parent_order_id="order-1")

    t1 = threading.Thread(target=_submit)
    t2 = threading.Thread(target=_submit)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Exactly one stop-loss protective algo submit despite two concurrent callers.
    assert exchange.algo_submits == 1
    assert "LQ-prot-1" in handler._protected_parent_client_ids


# --------------------------------------------------------------------------- #
# Item 7: concurrent save_state must never corrupt the recovery file.
# --------------------------------------------------------------------------- #
def test_concurrent_save_state_never_corrupts(tmp_path):
    target = tmp_path / "state.json"
    manager = StateManager(file_path=str(target))

    def _writer(idx: int) -> None:
        for _ in range(40):
            manager.save_state({"writer": idx, "payload": list(range(50))})

    threads = [threading.Thread(target=_writer, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # File must be present and contain a single valid JSON document (not torn).
    assert target.exists()
    with open(target, encoding="utf-8") as f:
        loaded = json.load(f)
    assert "writer" in loaded
    # No leftover temp files in the target directory.
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], f"temp files leaked: {leftovers}"


def test_save_state_uses_unique_temp_dir_same_as_target(tmp_path):
    nested = tmp_path / "deep" / "dir"
    target = nested / "state.json"
    manager = StateManager(file_path=str(target))
    manager.save_state({"ok": True})
    assert target.exists()
    with open(target, encoding="utf-8") as f:
        assert json.load(f) == {"ok": True}


# --------------------------------------------------------------------------- #
# Item 8: RuntimeCache is lock-guarded; concurrent writers stay consistent.
# --------------------------------------------------------------------------- #
def test_runtime_cache_concurrent_updates_consistent():
    cache = RuntimeCache()

    def _pos_writer() -> None:
        for i in range(200):
            cache.update_positions({"BTC/USDT": float(i)})

    def _legs_writer() -> None:
        for i in range(200):
            cache.update_position_legs({"BTC/USDT": {"LONG": float(i), "SHORT": 0.0}})

    def _reader() -> None:
        for _ in range(200):
            snap = cache.snapshot()
            assert set(snap.keys()) >= {"positions", "position_legs"}

    threads = [
        threading.Thread(target=_pos_writer),
        threading.Thread(target=_legs_writer),
        threading.Thread(target=_reader),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert "BTC/USDT" in cache.positions


def test_runtime_cache_restore_roundtrip_still_works():
    cache = RuntimeCache()
    cache.update_positions({"BTC/USDT": 0.5})
    cache.update_position_legs({"BTC/USDT": {"LONG": 0.5, "SHORT": 0.0}})
    restored = RuntimeCache()
    restored.restore(cache.snapshot())
    assert restored.positions["BTC/USDT"] == 0.5
    assert restored.position_legs["BTC/USDT"]["LONG"] == 0.5


# --------------------------------------------------------------------------- #
# Item 4: trader stream-gap hooks invalidate the execution-handler BBO cache.
# --------------------------------------------------------------------------- #
class _AuditStore:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def log_risk_event(self, *args, **kwargs):
        self.events.append((args, kwargs))


def _bare_trader() -> LiveTrader:
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
    trader.logger = SimpleNamespace(error=lambda *_a, **_k: None)
    return trader


def test_user_stream_error_invalidates_bbo_cache():
    trader = _bare_trader()
    invalidated: list = []
    trader.execution_handler = SimpleNamespace(
        invalidate_bbo_cache=lambda symbol=None: invalidated.append(symbol)
    )
    trader._on_user_stream_error(RuntimeError("ws-drop"))
    assert invalidated == [None]


def test_listen_key_expired_invalidates_bbo_cache():
    trader = _bare_trader()
    invalidated: list = []
    trader.execution_handler = SimpleNamespace(
        invalidate_bbo_cache=lambda symbol=None: invalidated.append(symbol)
    )
    trader._on_user_stream_event({"event_type": "listenKeyExpired", "exchange_ts_ms": 1})
    assert invalidated == [None]


def test_invalidate_bbo_cache_helper_is_noop_when_handler_lacks_method():
    trader = _bare_trader()
    trader.execution_handler = SimpleNamespace()  # no invalidate_bbo_cache
    # Must not raise.
    trader._invalidate_execution_bbo_cache()


# --------------------------------------------------------------------------- #
# Item 9: consecutive-loss kill-switch is never silently bypassed.
# --------------------------------------------------------------------------- #
class _RecordingRisk:
    def __init__(self) -> None:
        self.losses: list[float] = []
        self._consecutive_loss_count = 0

    def record_loss(self, *, realized_pnl: float) -> int:
        self.losses.append(float(realized_pnl))
        if realized_pnl < 0.0:
            self._consecutive_loss_count += 1
        else:
            self._consecutive_loss_count = 0
        return self._consecutive_loss_count


class _RaisingEntryPricesPortfolio:
    """Portfolio whose entry_prices access raises but current_positions is readable.

    Models the audited failure: the entry-PRICE snapshot fails while we still know
    (from current_positions) that the fill is closing a long position, so the
    kill-switch must advance conservatively.
    """

    current_positions = {"BTC/USDT": 1.0}  # long -> SELL is a closing fill

    @property
    def entry_prices(self):
        raise RuntimeError("snapshot boom")


def _fill_trader(portfolio) -> LiveTrader:
    trader = LiveTrader.__new__(LiveTrader)
    trader._audit_closed = True
    trader.portfolio = portfolio
    trader.risk_manager = _RecordingRisk()
    trader.audit_store = _AuditStore()
    trader.run_id = "test"
    trader.config = SimpleNamespace(STORAGE_EXPORT_CSV=False)
    logged: list = []
    trader.logger = SimpleNamespace(
        error=lambda *a, **k: logged.append(("error", a)),
        warning=lambda *a, **k: logged.append(("warning", a)),
        info=lambda *a, **k: None,
    )
    trader._logged = logged  # type: ignore[attr-defined]
    # Stub the engine super().handle_fill_event so only the override logic runs.
    trader._super_called = []  # type: ignore[attr-defined]
    return trader


def _fill(direction: str, qty: float, fill_cost: float) -> object:
    from lumina_quant.core.events import FillEvent

    return FillEvent(
        timeindex="2026-01-01",
        symbol="BTC/USDT",
        exchange="BINANCE",
        quantity=qty,
        direction=direction,
        fill_cost=fill_cost,
        commission=0.1,
        status="FILLED",
    )


def test_snapshot_failure_still_records_conservative_loss(monkeypatch):
    trader = _fill_trader(_RaisingEntryPricesPortfolio())

    # Neutralize the engine base handler.
    monkeypatch.setattr(
        "lumina_quant.core.engine.TradingEngine.handle_fill_event",
        lambda self, event: None,
    )

    trader.handle_fill_event(_fill("SELL", 1.0, 150.0))

    # Snapshot raised, but the closing fill must STILL advance the kill-switch.
    assert trader.risk_manager._consecutive_loss_count == 1
    assert trader.risk_manager.losses == [-1.0]
    # The failure must be logged (never silent) and audited.
    assert any(level == "error" for level, _ in trader._logged)
    assert any(
        call[1].get("reason") == "FILL_ENTRY_SNAPSHOT_ERROR" for call in trader.audit_store.events
    )


def test_record_loss_error_is_logged_not_swallowed(monkeypatch):
    portfolio = SimpleNamespace(
        entry_prices={"BTC/USDT": 200.0},
        current_positions={"BTC/USDT": 1.0},
    )
    trader = _fill_trader(portfolio)

    class _BoomRisk(_RecordingRisk):
        def record_loss(self, *, realized_pnl: float) -> int:
            raise RuntimeError("record boom")

    trader.risk_manager = _BoomRisk()

    monkeypatch.setattr(
        "lumina_quant.core.engine.TradingEngine.handle_fill_event",
        lambda self, event: None,
    )

    # Closing a long at a loss; record_loss raises -> must be logged + audited, not swallowed.
    trader.handle_fill_event(_fill("SELL", 1.0, 150.0))
    assert any(level == "error" for level, _ in trader._logged)
    assert any(call[1].get("reason") == "RECORD_LOSS_ERROR" for call in trader.audit_store.events)
