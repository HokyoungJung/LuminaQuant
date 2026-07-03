import hashlib
import logging
import threading
import time
from types import SimpleNamespace
from typing import Any

from lumina_quant.backtesting.execution_model import (  # Phase 5 structural gate: live/ must share unified cost model
    ExecutionModel,
    ExecutionModelConfig,
    _config_from_attrs,
)
from lumina_quant.backtesting.execution_sim import ExecutionHandler
from lumina_quant.core.events import FillEvent
from lumina_quant.core.order_policy import (
    canonical_order_type,
    limit_price_for_direction,
    normalize_limit_price_mode,
    price_tick_size_from_sources,
)
from lumina_quant.core.protocols import ExchangeInterface
from lumina_quant.live.binance_market_stream import normalize_stream_symbol
from lumina_quant.live.order_gateway import OrderGateway
from lumina_quant.live.order_state_machine import OrderStateMachine
from lumina_quant.live.order_state_projector import OrderStateProjector

STATE_NEW = "NEW"
STATE_SUBMITTED = "SUBMITTED"
STATE_ACKED = "ACKED"
STATE_OPEN = "OPEN"
STATE_PARTIAL = "PARTIAL"
STATE_FILLED = "FILLED"
STATE_CANCELED = "CANCELED"
STATE_REJECTED = "REJECTED"
STATE_TIMEOUT = "TIMEOUT"

TERMINAL_STATES = {STATE_FILLED, STATE_CANCELED, STATE_REJECTED, STATE_TIMEOUT}
_PROTECTIVE_PARAM_KEYS = {
    "stopLoss",
    "takeProfit",
    "stopLossPrice",
    "takeProfitPrice",
    "stopLossTriggerPrice",
    "takeProfitTriggerPrice",
}
_PROTECTIVE_CLIENT_DIGEST_LEN = len("0123456789abcdef01234567")


def _is_retryable_exception(exc: Exception) -> bool:
    retryable_names = {
        "NetworkError",
        "RequestTimeout",
        "ExchangeNotAvailable",
        "DDoSProtection",
    }
    if exc.__class__.__name__ in retryable_names:
        return True

    status = getattr(exc, "status_code", None)
    if status in {429, 500, 502, 503, 504}:
        return True

    msg = str(exc).lower()
    return any(token in msg for token in ["timeout", "temporarily", "429", "rate limit"])


class LiveExecutionHandler(ExecutionHandler):
    """Handles order execution via an ExchangeInterface."""

    def __init__(self, events, bars, config, exchange: ExchangeInterface):
        self.events = events
        self.bars = bars
        self.config = config
        self.exchange = exchange
        self.logger = logging.getLogger("LiveExecutionHandler")
        self.order_timeout_sec = max(1, int(getattr(config, "ORDER_TIMEOUT", 10)))
        # Single reentrant lock guarding ALL reads/writes of tracked_orders /
        # last_filled and the fill-emission path.  The user-stream thread and the
        # polling-fallback path both mutate this state concurrently; without the
        # lock a fill can be double-counted or lost (corrupt position).  Reentrant
        # so the public entrypoints can nest internal helpers that also acquire it.
        self._state_lock = threading.RLock()
        self.tracked_orders = {}
        self.client_id_to_order = {}
        # Cross-path fill ledger: highest cumulative-filled qty already emitted per
        # order_id (across user-stream AND polling).  Guards the resurrect-after-
        # forget race where one path terminally resolves + forgets an order while
        # the other still re-creates it and re-emits the same fill.  Bounded FIFO.
        self._emitted_cum_filled: dict[str, float] = {}
        self._emitted_cum_filled_max_keys = 100_000
        self._state_callback = None
        # Best-bid/offer freshness guard (live-only, fail-closed).  Cache the most
        # recent snapshot together with the monotonic time its *content* last
        # changed so a websocket stall (no new ticks) ages out instead of passing
        # orders against a stale quote.  config.MAX_BBO_AGE_SECONDS<=0 => disabled.
        self.max_bbo_age_seconds = float(getattr(config, "MAX_BBO_AGE_SECONDS", 0.0) or 0.0)
        self._bbo_cache: dict[str, tuple[dict[str, Any], float]] = {}
        self.order_gateway = OrderGateway(exchange)
        self.order_state_source = (
            str(getattr(config, "ORDER_STATE_SOURCE", "polling") or "polling").strip().lower()
        )
        self.poll_fallback_enabled = bool(
            getattr(config, "RECONCILIATION_POLL_FALLBACK_ENABLED", True)
        )
        self.state_machine = OrderStateMachine()
        self.state_projector = OrderStateProjector(state_machine=self.state_machine)
        self._last_exchange_open_ids: set[str] = set()
        self._last_exchange_open_signature: tuple[tuple[str, ...], ...] = tuple()
        self._last_exchange_open_snapshot_ok = False
        self._last_exchange_open_snapshot_ts = 0.0
        self.protective_orders: dict[str, dict[str, Any]] = {}
        self._protected_parent_client_ids: set[str] = set()
        # Dedicated lock guarding the protective-order dedup set check-and-add.
        # _submit_paper_exchange_protection runs OUTSIDE _state_lock from both the
        # user-stream and polling threads; this lock makes the dedup atomic without
        # being held across the (network) protective-order submission.
        self._protection_lock = threading.Lock()
        # Unified cost model — same formula as backtest (Phase 5 structural gate).
        # Used for paper/testnet simulated fill commission and pre-trade
        # liquidation_price / cost estimation in all modes.
        _rt = getattr(config, "_rt", None)
        if _rt is not None:
            _em_cfg = ExecutionModelConfig.from_runtime(_rt, mode="live")
        else:
            _em_cfg = _config_from_attrs(config)  # unit-test path: mock config without _rt
        self._execution_model: ExecutionModel = ExecutionModel(_em_cfg)

    def set_order_state_callback(self, callback) -> None:
        """Set a callback to receive order-state transition events."""
        self._state_callback = callback

    def _call_with_retry(self, fn, *args, retries=3, delay=1.0, backoff=2.0, **kwargs):
        attempt = 0
        wait = delay
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                attempt += 1
                if attempt >= retries or not _is_retryable_exception(exc):
                    raise
                self.logger.warning(
                    "Retryable API error in %s (%s/%s): %s",
                    getattr(fn, "__name__", "call"),
                    attempt,
                    retries,
                    exc,
                )
                time.sleep(wait)
                wait *= backoff

    def _lookup_order_by_client_id(self, symbol, client_order_id):
        """Return the exchange order matching ``client_order_id``, or ``None``.

        Used to make order submission idempotent: after a retryable submit
        failure the outcome is UNKNOWN (the exchange may have accepted the
        order even though the response failed), so the order must be looked up
        by client id before any resubmission.
        """
        if not client_order_id:
            return None
        try:
            fetched = self.exchange.fetch_order(
                None, symbol, params={"origClientOrderId": str(client_order_id)}
            )
            if fetched and (fetched.get("id") or fetched.get("clientOrderId")):
                return dict(fetched)
        except Exception:
            pass
        try:
            for item in self.order_gateway.query_open_orders(symbol) or []:
                cid = item.get("clientOrderId") or (item.get("info") or {}).get("clientOrderId")
                if str(cid or "") == str(client_order_id):
                    return dict(item)
        except Exception:
            pass
        return None

    def _submit_with_idempotent_retry(
        self,
        *,
        symbol,
        order_type,
        side,
        quantity,
        price,
        params,
        client_order_id,
        retries=3,
        delay=1.0,
        backoff=2.0,
    ):
        """Submit an order with query-before-resubmit retry semantics.

        A plain retry loop double-sends on the documented Binance "unknown
        outcome" path (5xx/timeout after the matching engine accepted the
        order) because a clientOrderId becomes reusable once the original
        order fills. Every retry therefore first queries the exchange for the
        client id and ADOPTS the existing order instead of resubmitting
        (2026-07-03 audit fix #3a).
        """
        attempt = 0
        wait = delay
        while True:
            try:
                return self.order_gateway.submit(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    quantity=quantity,
                    price=price,
                    params=params,
                )
            except Exception as exc:
                attempt += 1
                if attempt >= retries or not _is_retryable_exception(exc):
                    raise
                existing = self._lookup_order_by_client_id(symbol, client_order_id)
                if existing is not None:
                    self.logger.warning(
                        "Submit response failed but the exchange accepted the order; "
                        "adopting it instead of resubmitting (client_id=%s, id=%s)",
                        client_order_id,
                        existing.get("id"),
                    )
                    return existing
                self.logger.warning(
                    "Retryable API error in submit (%s/%s), no order found for "
                    "client_id=%s -> safe to resubmit: %s",
                    attempt,
                    retries,
                    client_order_id,
                    exc,
                )
                time.sleep(wait)
                wait *= backoff

    def _normalize_status(self, status):
        if status is None:
            return "unknown"
        return str(status).strip().lower()

    def _to_state(self, status: str, filled_qty: float, amount: float) -> str:
        status = self._normalize_status(status)
        if status in {"closed", "filled"}:
            return STATE_FILLED
        if status in {"canceled", "cancelled", "expired"}:
            return STATE_CANCELED
        if status in {"rejected"}:
            return STATE_REJECTED
        if status in {"partially_filled", "partial"}:
            if filled_qty > 0 and amount > 0 and filled_qty < amount:
                return STATE_PARTIAL
            return STATE_OPEN
        if status in {"new", "pending"}:
            if filled_qty > 0 and amount > 0 and filled_qty < amount:
                return STATE_PARTIAL
            if filled_qty <= 0:
                return STATE_ACKED
            return STATE_OPEN
        if status in {"open"}:
            if filled_qty > 0 and amount > 0 and filled_qty < amount:
                return STATE_PARTIAL
            return STATE_OPEN
        return STATE_OPEN

    def _make_client_order_id(self, event):
        parts = [
            str(getattr(event, "symbol", "")),
            str(getattr(event, "order_type", "")),
            str(getattr(event, "direction", "")),
            f"{float(getattr(event, 'quantity', 0.0)):.10f}",
            f"{float(getattr(event, 'price', 0.0) or 0.0):.10f}",
            str(getattr(event, "position_side", "") or ""),
            "1" if bool(getattr(event, "reduce_only", False)) else "0",
            str(getattr(event, "time_in_force", "") or ""),
            str(getattr(event, "timestamp_ns", "") or ""),
            str(getattr(event, "sequence", "") or ""),
            str(getattr(event, "stop_loss", "") or ""),
            str(getattr(event, "take_profit", "") or ""),
        ]
        token = "|".join(parts)
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]
        return f"LQ-{digest}"

    def _notify_state(
        self,
        *,
        order_id: str | None,
        entry: dict[str, Any] | None,
        state: str,
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
        defer_to: list[dict[str, Any]] | None = None,
    ) -> None:
        """Prepare an order-state callback payload.

        The downstream callback (trader._on_order_state) performs blocking disk +
        network + DB I/O, so it must never run under ``self._state_lock``.  Callers
        that hold the lock pass ``defer_to`` (a list collected while locked); the
        prepared payload is appended to it and fired by ``_fire_deferred_notifications``
        AFTER the lock is released — exactly once, preserving call order.  When
        ``defer_to`` is None (no lock held) the callback fires immediately.
        """
        callback = self._state_callback
        if callback is None:
            return
        payload = {
            "order_id": order_id,
            "state": state,
            "message": message,
            "metadata": metadata or {},
            "event_time": time.time(),
        }
        if entry:
            event = entry.get("event")
            payload["symbol"] = entry.get("symbol")
            payload["client_order_id"] = getattr(event, "client_order_id", None)
            payload["last_filled"] = float(entry.get("last_filled", 0.0))
            payload["created_at"] = float(entry.get("created_at", 0.0))
            payload_meta = payload["metadata"]
            payload_meta["symbol"] = entry.get("symbol")
            payload_meta["direction"] = getattr(event, "direction", None)
            payload_meta["order_type"] = getattr(event, "order_type", None)
            payload_meta["quantity"] = float(getattr(event, "quantity", 0.0) or 0.0)
            payload_meta["price"] = getattr(event, "price", None)
            payload_meta["position_side"] = getattr(event, "position_side", None)
            payload_meta["reduce_only"] = bool(getattr(event, "reduce_only", False))
            payload_meta["time_in_force"] = getattr(event, "time_in_force", None)
            payload_meta["stop_loss"] = getattr(event, "stop_loss", None)
            payload_meta["take_profit"] = getattr(event, "take_profit", None)
            payload_meta["client_order_id"] = getattr(event, "client_order_id", None)
        if defer_to is not None:
            defer_to.append(payload)
            return
        self._dispatch_notification(payload)

    def _dispatch_notification(self, payload: dict[str, Any]) -> None:
        callback = self._state_callback
        if callback is None:
            return
        try:
            callback(payload)
        except Exception as exc:  # pragma: no cover - defensive callback guard
            self.logger.error("Order state callback failed: %s", exc)

    def _fire_deferred_notifications(self, deferred: list[dict[str, Any]]) -> None:
        """Fire callback payloads collected under the lock, after it is released.

        Preserves enqueue order and fires each exactly once so no blocking disk /
        network / DB I/O is performed while ``self._state_lock`` is held.
        """
        for payload in deferred:
            self._dispatch_notification(payload)

    def _forget_order(self, order_id: str, entry: dict[str, Any]) -> None:
        self.tracked_orders.pop(order_id, None)
        event = entry.get("event")
        client_order_id = getattr(event, "client_order_id", None)
        if client_order_id and self.client_id_to_order.get(client_order_id) == order_id:
            self.client_id_to_order.pop(client_order_id, None)
        # NOTE: the dedup-ledger key is intentionally NOT removed here.  It must
        # outlive _forget_order so the resurrect-after-forget race is still guarded
        # (one path terminally resolves + forgets while the other re-creates the
        # order from the same exchange snapshot and would otherwise re-emit the same
        # cumulative).  The key is reclaimed lazily by _evict_ledger_keys, which only
        # evicts keys whose order is no longer live (i.e. already forgotten).

    def _ledger_residual_delta(self, order_id: str | None, new_cumulative: float) -> float:
        """Return the not-yet-emitted residual for ``order_id`` and record it.

        Must be called under ``self._state_lock``.  Both the user-stream and the
        polling paths route fills through this single chokepoint so a fill is
        emitted at most once even if one path terminally resolves and forgets the
        order while the other re-creates it from the same exchange snapshot.
        """
        if order_id is None:
            return max(0.0, float(new_cumulative))
        key = str(order_id)
        already = float(self._emitted_cum_filled.get(key, 0.0))
        residual = max(0.0, float(new_cumulative) - already)
        if residual > 0.0:
            self._emitted_cum_filled[key] = float(new_cumulative)
            self._evict_ledger_keys()
        return residual

    def _evict_ledger_keys(self) -> None:
        """Bound the dedup ledger, but never evict a key whose order is still live.

        Eviction pops oldest-inserted keys; if the oldest belongs to an order still
        in ``tracked_orders`` (a long-resting partial fill), dropping its baseline
        would re-emit its full cumulative on the next poll/stream event.  Skip such
        live keys and only evict terminally-resolved / forgotten ones.  Must be
        called under ``self._state_lock``.
        """
        max_keys = int(self._emitted_cum_filled_max_keys)
        if len(self._emitted_cum_filled) <= max_keys:
            return
        for key in list(self._emitted_cum_filled.keys()):
            if len(self._emitted_cum_filled) <= max_keys:
                break
            if key in self.tracked_orders:
                continue
            self._emitted_cum_filled.pop(key, None)

    def _build_event_stub(self, payload: dict[str, Any]) -> SimpleNamespace:
        metadata = dict(payload.get("metadata") or {})
        return SimpleNamespace(
            symbol=str(payload.get("symbol") or metadata.get("symbol") or ""),
            direction=str(metadata.get("direction") or "BUY"),
            order_type=str(metadata.get("order_type") or "MKT"),
            quantity=float(metadata.get("quantity") or 0.0),
            price=metadata.get("price"),
            position_side=metadata.get("position_side"),
            reduce_only=bool(metadata.get("reduce_only", False)),
            client_order_id=str(
                payload.get("client_order_id") or metadata.get("client_order_id") or ""
            ),
            time_in_force=metadata.get("time_in_force"),
            stop_loss=metadata.get("stop_loss"),
            take_profit=metadata.get("take_profit"),
            metadata=metadata,
            type="ORDER",
        )

    def rehydrate_orders(self, open_orders: dict[str, dict[str, Any]]) -> None:
        # Guarded by _state_lock: the user-stream thread (started before _load_state
        # calls rehydrate) already mutates tracked_orders/client_id_to_order under
        # the same lock via ingest_user_stream_event.
        with self._state_lock:
            for order_id, payload in dict(open_orders or {}).items():
                if not isinstance(payload, dict):
                    continue
                state = str(payload.get("state") or "").upper()
                if state in TERMINAL_STATES:
                    continue
                event = self._build_event_stub(payload)
                oid = str(order_id)
                last_filled = float(payload.get("last_filled") or 0.0)
                created_at = float(payload.get("created_at") or time.time())
                self.tracked_orders[oid] = {
                    "event": event,
                    "symbol": event.symbol,
                    "last_filled": last_filled,
                    "state": state or STATE_OPEN,
                    "created_at": created_at,
                    "updated_at": time.time(),
                }
                if event.client_order_id:
                    self.client_id_to_order[event.client_order_id] = oid
                # Seed the dedup ledger with the already-counted cumulative filled
                # using the SAME key derivation _ledger_residual_delta uses (str(oid)).
                # Without this, the first reconcile/poll after restart re-emits the
                # restored cumulative as a fresh FillEvent (double-counted position).
                if last_filled > 0.0:
                    self._emitted_cum_filled[oid] = last_filled

    def ingest_user_stream_event(self, payload: dict[str, Any]) -> None:
        """Apply one normalized user-stream event to tracked order state."""
        if not isinstance(payload, dict):
            return
        event_type = str(payload.get("event_type") or "").strip()
        if event_type != "executionReport":
            # Account/balance events are owned by trader/runtime cache projection.
            return

        raw_order_id = payload.get("order_id")
        if raw_order_id in {None, ""}:
            return
        order_id = str(raw_order_id)

        # Read-modify-write of tracked_orders/last_filled, fill dedup (projector
        # idempotency) and fill emission are performed atomically so the polling
        # fallback cannot interleave and double-count / lose the same fill.  No
        # blocking network I/O is performed under the lock; protective-order
        # submission (network) is deferred until after the lock is released.
        emit_protection_for: Any | None = None
        deferred_notifications: list[dict[str, Any]] = []
        with self._state_lock:
            entry = self.tracked_orders.get(order_id)

            if entry is None:
                symbol = normalize_stream_symbol(str(payload.get("symbol") or ""))
                synthetic = SimpleNamespace(
                    symbol=symbol,
                    direction=str(payload.get("side") or "BUY"),
                    order_type="MKT",
                    quantity=float(
                        payload.get("cum_fill_qty") or payload.get("last_fill_qty") or 0.0
                    ),
                    price=float(payload.get("last_fill_price") or 0.0),
                    position_side=payload.get("position_side"),
                    reduce_only=bool(payload.get("reduce_only", False)),
                    client_order_id=str(payload.get("client_order_id") or ""),
                    time_in_force=None,
                    stop_loss=None,
                    take_profit=None,
                    metadata={},
                    type="ORDER",
                )
                entry = {
                    "event": synthetic,
                    "symbol": symbol,
                    "last_filled": 0.0,
                    "state": STATE_SUBMITTED,
                    "created_at": time.time(),
                    "updated_at": time.time(),
                }
                self.tracked_orders[order_id] = entry
                if synthetic.client_order_id:
                    self.client_id_to_order[synthetic.client_order_id] = order_id

            previous_state = str(entry.get("state") or STATE_SUBMITTED).upper()
            previous_filled = float(entry.get("last_filled") or 0.0)
            projection = self.state_projector.project_execution_report(
                payload,
                previous_state=previous_state,
                previous_filled=previous_filled,
            )
            if not projection.accepted:
                if projection.reason == "ORDER_INVALID_TRANSITION":
                    self._notify_state(
                        order_id=order_id,
                        entry=entry,
                        state=previous_state,
                        message="invalid_transition",
                        metadata={"reason": projection.reason, "event_key": projection.event_key},
                        defer_to=deferred_notifications,
                    )
                # Do NOT fire callbacks or return here: that would run the blocking
                # callback I/O under _state_lock (violating the lines ~212-213
                # invariant). Fall through to the single post-lock fire below.
            else:
                if projection.fill_delta > 0:
                    residual = self._ledger_residual_delta(order_id, projection.cumulative_filled)
                    if residual > 0:
                        order_payload = {
                            "id": order_id,
                            "status": payload.get("order_status"),
                            "filled": projection.cumulative_filled,
                            "amount": max(
                                float(getattr(entry.get("event"), "quantity", 0.0) or 0.0),
                                projection.cumulative_filled,
                            ),
                            "average": float(payload.get("last_fill_price") or 0.0),
                            "price": float(payload.get("last_fill_price") or 0.0),
                        }
                        self._emit_fill_event(
                            entry.get("event"),
                            order_payload,
                            float(residual),
                            status=str(payload.get("order_status") or "").lower(),
                            submitted_at=entry.get("created_at"),
                        )

                entry["last_filled"] = float(projection.cumulative_filled)
                entry["state"] = str(projection.next_state).upper()
                entry["updated_at"] = time.time()
                self._notify_state(
                    order_id=order_id,
                    entry=entry,
                    state=entry["state"],
                    metadata={"event_key": projection.event_key},
                    defer_to=deferred_notifications,
                )
                if str(entry["state"]).upper() in TERMINAL_STATES:
                    if str(entry["state"]).upper() == STATE_FILLED:
                        emit_protection_for = entry.get("event")
                    self._forget_order(order_id, entry)

        self._fire_deferred_notifications(deferred_notifications)
        if emit_protection_for is not None:
            self._submit_paper_exchange_protection(emit_protection_for, parent_order_id=order_id)

    def _build_reconciliation_payload(
        self,
        *,
        order_id: str,
        entry: dict[str, Any],
        local_state: str,
        exchange_state: str,
        local_filled: float,
        exchange_filled: float,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = entry.get("event")
        return {
            "order_id": order_id,
            "symbol": entry.get("symbol") or getattr(event, "symbol", None),
            "client_order_id": getattr(event, "client_order_id", None),
            "local_state": local_state,
            "exchange_state": exchange_state,
            "local_filled": float(local_filled),
            "exchange_filled": float(exchange_filled),
            "reason": str(reason),
            "metadata": dict(metadata or {}),
        }

    def reconcile_open_orders(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        try:
            exchange_open = list(self.order_gateway.query_open_orders(None) or [])
        except Exception as exc:
            self._last_exchange_open_snapshot_ok = False
            self.logger.error("Failed to fetch open orders for reconciliation: %s", exc)
            return records
        self._last_exchange_open_snapshot_ok = True
        self._last_exchange_open_snapshot_ts = time.time()

        open_ids = {str(item.get("id")) for item in exchange_open if item.get("id")}
        self._last_exchange_open_ids = set(open_ids)
        self._last_exchange_open_signature = tuple((order_id,) for order_id in sorted(open_ids))

        tracked_ids = set(self.tracked_orders.keys())
        for row in exchange_open:
            if not isinstance(row, dict):
                continue
            order_id = str(row.get("id") or "")
            if not order_id or order_id in tracked_ids:
                continue

            symbol = normalize_stream_symbol(str(row.get("symbol") or ""))
            if not symbol:
                continue
            side = str(row.get("side") or "buy").upper()
            direction = "BUY" if side == "BUY" else "SELL"
            client_order_id = str(row.get("clientOrderId") or row.get("client_order_id") or "")
            status = self._normalize_status(row.get("status"))
            filled_qty = float(row.get("filled") or 0.0)
            amount = float(row.get("amount") or row.get("remaining") or filled_qty or 0.0)
            state = self._to_state(status, filled_qty, max(amount, filled_qty))
            if state in TERMINAL_STATES:
                continue

            synthetic = SimpleNamespace(
                symbol=symbol,
                direction=direction,
                order_type=str(row.get("type") or "MKT").upper(),
                quantity=max(float(amount), float(filled_qty)),
                price=row.get("price"),
                position_side=row.get("positionSide"),
                reduce_only=bool(row.get("reduceOnly", False)),
                client_order_id=client_order_id,
                time_in_force=row.get("timeInForce"),
                stop_loss=None,
                take_profit=None,
                metadata={},
                type="ORDER",
            )
            entry = {
                "event": synthetic,
                "symbol": symbol,
                "last_filled": filled_qty,
                "state": state,
                "created_at": time.time(),
                "updated_at": time.time(),
            }
            with self._state_lock:
                if order_id in self.tracked_orders:
                    # The user-stream thread tracked this order between the open-orders
                    # snapshot and now; do not clobber its (authoritative) fill state.
                    continue
                self.tracked_orders[order_id] = entry
                if client_order_id:
                    self.client_id_to_order[client_order_id] = order_id

            self._notify_state(
                order_id=order_id,
                entry=entry,
                state=str(state),
                message="rehydrated_exchange_open_order",
            )
            records.append(
                self._build_reconciliation_payload(
                    order_id=order_id,
                    entry=entry,
                    local_state=STATE_SUBMITTED,
                    exchange_state=str(state),
                    local_filled=0.0,
                    exchange_filled=filled_qty,
                    reason="UNTRACKED_OPEN_ORDER_REHYDRATED",
                    metadata={"status": status},
                )
            )

        for order_id, entry in list(self.tracked_orders.items()):
            if order_id in open_ids:
                continue
            order_event = entry.get("event")
            symbol = getattr(order_event, "symbol", None)
            if not symbol:
                continue
            local_state = str(entry.get("state") or STATE_OPEN)
            local_filled = float(entry.get("last_filled") or 0.0)
            try:
                latest = self._call_with_retry(
                    self.order_gateway.query_order,
                    order_id,
                    symbol,
                    retries=2,
                    delay=0.5,
                )
            except Exception as exc:
                self.logger.error("Order reconciliation poll failed for %s: %s", order_id, exc)
                records.append(
                    self._build_reconciliation_payload(
                        order_id=order_id,
                        entry=entry,
                        local_state=local_state,
                        exchange_state=local_state,
                        local_filled=local_filled,
                        exchange_filled=local_filled,
                        reason="POLL_ERROR",
                        metadata={"error": str(exc)},
                    )
                )
                continue
            if not latest:
                records.append(
                    self._build_reconciliation_payload(
                        order_id=order_id,
                        entry=entry,
                        local_state=local_state,
                        exchange_state=local_state,
                        local_filled=local_filled,
                        exchange_filled=local_filled,
                        reason="MISSING_ORDER",
                    )
                )
                continue
            status = self._normalize_status(latest.get("status"))
            filled_now = float(latest.get("filled") or 0.0)
            total_amount = float(latest.get("amount") or getattr(order_event, "quantity", 0.0))
            exchange_state = self._to_state(status, filled_now, total_amount)
            emit_protection_for: Any | None = None
            deferred_notifications: list[dict[str, Any]] = []
            # Apply the polled reconciliation atomically: re-read last_filled under
            # the lock so a concurrent user-stream fill is not double-counted.
            with self._state_lock:
                delta = self._ledger_residual_delta(order_id, filled_now)
                if delta > 0:
                    self._emit_fill_event(
                        order_event,
                        latest,
                        delta,
                        status=status,
                        submitted_at=entry.get("created_at"),
                    )
                    entry["last_filled"] = filled_now
                    records.append(
                        self._build_reconciliation_payload(
                            order_id=order_id,
                            entry=entry,
                            local_state=local_state,
                            exchange_state=exchange_state,
                            local_filled=local_filled,
                            exchange_filled=filled_now,
                            reason="FILL_DELTA",
                            metadata={"delta": delta},
                        )
                    )
                entry["state"] = exchange_state
                entry["updated_at"] = time.time()
                self._notify_state(
                    order_id=order_id,
                    entry=entry,
                    state=entry["state"],
                    defer_to=deferred_notifications,
                )
                if exchange_state != local_state:
                    records.append(
                        self._build_reconciliation_payload(
                            order_id=order_id,
                            entry=entry,
                            local_state=local_state,
                            exchange_state=exchange_state,
                            local_filled=local_filled,
                            exchange_filled=filled_now,
                            reason="STATE_CHANGE",
                        )
                    )
                if entry["state"] in TERMINAL_STATES:
                    records.append(
                        self._build_reconciliation_payload(
                            order_id=order_id,
                            entry=entry,
                            local_state=local_state,
                            exchange_state=exchange_state,
                            local_filled=local_filled,
                            exchange_filled=filled_now,
                            reason="TERMINAL_RESOLVED",
                        )
                    )
                    if entry["state"] == STATE_FILLED:
                        emit_protection_for = order_event
                    self._forget_order(order_id, entry)
            self._fire_deferred_notifications(deferred_notifications)
            if emit_protection_for is not None:
                self._submit_paper_exchange_protection(
                    emit_protection_for, parent_order_id=order_id
                )
        return records

    def exchange_open_order_count(self) -> int:
        """Most recent exchange open-order count observed during reconciliation."""
        return len(self._last_exchange_open_ids)

    def tracked_order_signature(self) -> tuple[tuple[str, ...], ...]:
        rows: list[tuple[str, ...]] = []
        for order_id, entry in sorted(self.tracked_orders.items()):
            event = entry.get("event")
            rows.append(
                (
                    str(order_id),
                    str(entry.get("state") or ""),
                    str(round(float(entry.get("last_filled") or 0.0), 8)),
                    str(getattr(event, "client_order_id", None) or ""),
                )
            )
        return tuple(rows)

    def exchange_open_order_signature(self) -> tuple[tuple[str, ...], ...]:
        return tuple(self._last_exchange_open_signature)

    def exchange_open_snapshot_ready(self) -> bool:
        """Whether exchange open-order snapshot has been fetched successfully at least once."""
        return bool(self._last_exchange_open_snapshot_ok)

    def _build_exchange_params(self, event):
        params = {}
        if event.metadata and isinstance(event.metadata, dict):
            params.update(event.metadata.get("exchange_params", {}))

        if getattr(event, "client_order_id", None):
            params.setdefault("newClientOrderId", event.client_order_id)
            params.setdefault("clientOrderId", event.client_order_id)

        market_type = getattr(self.config, "MARKET_TYPE", "spot")
        if market_type == "future":
            if event.position_side:
                params["positionSide"] = event.position_side.upper()
            params["reduceOnly"] = bool(event.reduce_only)

        if event.time_in_force:
            params["timeInForce"] = event.time_in_force
        return params

    def _validate_protective_order_params(self, event, params: dict[str, Any]) -> None:
        has_protection = any(
            getattr(event, field, None) is not None for field in ("stop_loss", "take_profit")
        )
        if not has_protection:
            return
        if any(key in params for key in _PROTECTIVE_PARAM_KEYS):
            return
        live_mode = str(getattr(self.config, "MODE", "paper") or "paper").strip().lower()
        if live_mode != "real":
            if self._paper_exchange_protection_enabled():
                return
            self.logger.warning(
                "Proceeding without exchange-side protective params in %s mode for %s; "
                "stop_loss/take_profit remain unmanaged unless explicit exchange_params are provided.",
                live_mode or "paper",
                getattr(event, "symbol", ""),
            )
            return
        raise RuntimeError(
            "Real-mode live protective orders require explicit exchange_params mapping. "
            "Provide stop/take-profit exchange_params or omit stop_loss/take_profit."
        )

    def _paper_exchange_protection_enabled(self) -> bool:
        live_mode = str(getattr(self.config, "MODE", "paper") or "paper").strip().lower()
        if live_mode == "real":
            return False
        market_type = str(getattr(self.config, "MARKET_TYPE", "spot") or "spot").strip().lower()
        if market_type != "future":
            return False
        return bool(getattr(self.config, "PAPER_EXCHANGE_PROTECTIVE_ORDERS", True))

    @staticmethod
    def _protective_close_side(event) -> str:
        return "sell" if str(getattr(event, "direction", "")).upper() == "BUY" else "buy"

    @staticmethod
    def _protective_position_side(event) -> str | None:
        position_side = str(getattr(event, "position_side", "") or "").strip().upper()
        if position_side in {"LONG", "SHORT"}:
            return position_side
        return "LONG" if str(getattr(event, "direction", "")).upper() == "BUY" else "SHORT"

    @staticmethod
    def _protective_client_algo_id(parent_client_order_id: str, suffix: str) -> str:
        digest = hashlib.sha256(f"{parent_client_order_id}|{suffix}".encode()).hexdigest()
        return f"LQ-P-{suffix}-{digest[:_PROTECTIVE_CLIENT_DIGEST_LEN]}"

    def _protective_order_style(self) -> str:
        style = str(getattr(self.config, "PROTECTIVE_ORDER_STYLE", "limit") or "limit")
        style = style.strip().lower()
        if style == "market" and bool(getattr(self.config, "ALLOW_MARKET_ORDERS", False)):
            return "market"
        return "limit"

    def _market_spec_for_order(self, symbol: str) -> dict[str, Any]:
        if hasattr(self.bars, "get_market_spec"):
            try:
                spec = self.bars.get_market_spec(symbol) or {}
                if isinstance(spec, dict):
                    return dict(spec)
            except Exception:
                pass
        return {}

    def _protective_limit_params(
        self,
        *,
        symbol: str,
        close_side: str,
        trigger_price: float,
    ) -> dict[str, Any]:
        market_spec = self._market_spec_for_order(symbol)
        tick_size = price_tick_size_from_sources(
            symbol,
            market_spec=market_spec,
            config=self.config,
        )
        mode = normalize_limit_price_mode(
            getattr(self.config, "LIMIT_PRICE_MODE", "one_tick_worse")
        )
        offset_ticks = int(getattr(self.config, "LIMIT_PRICE_OFFSET_TICKS", 1) or 0)
        limit_price = limit_price_for_direction(
            reference_price=float(trigger_price),
            direction=str(close_side).upper(),
            tick_size=tick_size,
            mode=mode,
            offset_ticks=offset_ticks,
        )
        return {
            "price": float(limit_price),
            "timeInForce": str(getattr(self.config, "LIMIT_TIME_IN_FORCE", "GTC") or "GTC").upper(),
            "limitPriceMode": mode,
            "limitPriceOffsetTicks": offset_ticks,
            "priceTickSize": float(tick_size or 0.0),
        }

    def _build_paper_protective_algo_specs(self, event) -> list[dict[str, Any]]:
        if bool(getattr(event, "reduce_only", False)):
            return []
        if not self._paper_exchange_protection_enabled():
            return []
        metadata = dict(getattr(event, "metadata", None) or {})
        exchange_params = dict(metadata.get("exchange_params") or {})
        if any(key in exchange_params for key in _PROTECTIVE_PARAM_KEYS):
            return []
        quantity = float(getattr(event, "quantity", 0.0) or 0.0)
        if quantity <= 0.0:
            return []
        side = self._protective_close_side(event)
        style = self._protective_order_style()
        position_side = self._protective_position_side(event)
        parent_client_id = str(getattr(event, "client_order_id", "") or "")
        base_params = {
            "algoType": "CONDITIONAL",
            "positionSide": position_side,
            "workingType": "CONTRACT_PRICE",
            "priceProtect": "true",
            "parentClientOrderId": parent_client_id,
        }
        specs: list[dict[str, Any]] = []
        stop_loss = getattr(event, "stop_loss", None)
        if stop_loss is not None:
            params = dict(base_params)
            params.update(
                {
                    "triggerPrice": float(stop_loss),
                    "clientAlgoId": self._protective_client_algo_id(parent_client_id, "SL"),
                    "protectionRole": "stop_loss",
                }
            )
            order_type = "STOP_MARKET"
            if style == "limit":
                params.update(
                    self._protective_limit_params(
                        symbol=event.symbol,
                        close_side=side,
                        trigger_price=float(stop_loss),
                    )
                )
                order_type = "STOP"
            specs.append(
                {
                    "type": order_type,
                    "side": side,
                    "quantity": quantity,
                    "params": params,
                }
            )
        take_profit = getattr(event, "take_profit", None)
        if take_profit is not None:
            params = dict(base_params)
            params.update(
                {
                    "triggerPrice": float(take_profit),
                    "clientAlgoId": self._protective_client_algo_id(parent_client_id, "TP"),
                    "protectionRole": "take_profit",
                }
            )
            order_type = "TAKE_PROFIT_MARKET"
            if style == "limit":
                params.update(
                    self._protective_limit_params(
                        symbol=event.symbol,
                        close_side=side,
                        trigger_price=float(take_profit),
                    )
                )
                order_type = "TAKE_PROFIT"
            specs.append(
                {
                    "type": order_type,
                    "side": side,
                    "quantity": quantity,
                    "params": params,
                }
            )
        return specs

    def _submit_paper_exchange_protection(self, event, *, parent_order_id: str | None) -> None:
        if event is None:
            return
        parent_client_id = str(getattr(event, "client_order_id", "") or "")
        if not parent_client_id:
            return
        specs = self._build_paper_protective_algo_specs(event)
        if not specs:
            return
        # Atomic check-and-reserve so two threads (user-stream + polling) cannot both
        # submit protective orders for the same parent.  The reservation is added
        # BEFORE the (network) submission and rolled back only if nothing was
        # submitted; the lock is never held across submit_algo.
        with self._protection_lock:
            if parent_client_id in self._protected_parent_client_ids:
                return
            self._protected_parent_client_ids.add(parent_client_id)
        submitted: list[dict[str, Any]] = []
        for spec in specs:
            params = dict(spec["params"])
            params["parentOrderId"] = parent_order_id
            try:
                order = self._call_with_retry(
                    self.order_gateway.submit_algo,
                    symbol=event.symbol,
                    type=str(spec["type"]),
                    side=str(spec["side"]),
                    quantity=float(spec["quantity"]),
                    params=params,
                )
            except Exception as exc:
                self.logger.error("Paper/testnet protective algo order submit failed: %s", exc)
                self._notify_state(
                    order_id=parent_order_id,
                    entry={
                        "event": event,
                        "symbol": event.symbol,
                        "last_filled": float(getattr(event, "quantity", 0.0) or 0.0),
                        "created_at": time.time(),
                    },
                    state=STATE_REJECTED,
                    message="paper_protective_order_submit_failed",
                    metadata={
                        "error": str(exc),
                        "protective_type": str(spec["type"]),
                        "parent_client_order_id": parent_client_id,
                    },
                )
                continue
            protective_id = str(order.get("id") or params.get("clientAlgoId") or "")
            record = {
                "parent_order_id": parent_order_id,
                "parent_client_order_id": parent_client_id,
                "symbol": event.symbol,
                "type": str(spec["type"]),
                "side": str(spec["side"]),
                "quantity": float(spec["quantity"]),
                "params": params,
                "order": dict(order),
                "created_at": time.time(),
                "status": str(order.get("status") or STATE_SUBMITTED).upper(),
            }
            if protective_id:
                with self._protection_lock:
                    self.protective_orders[protective_id] = record
            submitted.append(record)
        if not submitted:
            # Nothing made it to the exchange; release the reservation so a later
            # attempt (e.g. after a transient outage) can retry the protection.
            with self._protection_lock:
                self._protected_parent_client_ids.discard(parent_client_id)
            return
        metadata = {
            "parent_order_id": parent_order_id,
            "parent_client_order_id": parent_client_id,
            "protective_order_count": len(submitted),
            "protective_order_types": [str(item["type"]) for item in submitted],
            "paper_testnet_only": True,
            "real_money_execution": False,
        }
        self._notify_state(
            order_id=parent_order_id,
            entry={
                "event": event,
                "symbol": event.symbol,
                "last_filled": float(getattr(event, "quantity", 0.0) or 0.0),
                "created_at": time.time(),
            },
            state=STATE_SUBMITTED,
            message="paper_exchange_protective_orders_submitted",
            metadata=metadata,
        )

    def _slippage_guard_policy_for_event(self, event) -> dict[str, Any]:
        metadata = dict(getattr(event, "metadata", None) or {})
        for key in ("live_slippage_guard_policy", "slippage_guard_policy"):
            value = metadata.get(key)
            if isinstance(value, dict):
                return dict(value)
        nested = metadata.get("signal_metadata")
        if isinstance(nested, dict):
            for key in ("live_slippage_guard_policy", "slippage_guard_policy"):
                value = nested.get(key)
                if isinstance(value, dict):
                    return dict(value)

        policy: dict[str, Any] = {}
        max_spread = getattr(self.config, "MAX_BBO_SPREAD_BPS_AT_SUBMIT", None)
        max_slippage = getattr(self.config, "MAX_ESTIMATED_ONE_WAY_SLIPPAGE_BPS", None)
        if max_spread is not None:
            policy["max_bbo_spread_bps_at_submit"] = float(max_spread)
        if max_slippage is not None:
            policy["max_estimated_one_way_slippage_bps"] = float(max_slippage)
        if bool(getattr(self.config, "REQUIRE_BBO_FOR_LIMIT_ORDERS", False)):
            policy["require_bbo_snapshot"] = True
        if bool(getattr(self.config, "ALLOW_MARKET_ORDERS", False)) is False:
            policy.setdefault("market_fallback_allowed", False)
        return policy

    def _latest_bbo_snapshot(self, symbol: str) -> dict[str, Any] | None:
        snapshot, _age = self._latest_bbo_snapshot_with_age(symbol)
        return snapshot

    @staticmethod
    def _bbo_identity(snapshot: dict[str, Any]) -> tuple[Any, ...]:
        """Content signature used to detect whether the quote actually changed."""
        return (
            snapshot.get("exchange_ts_ms"),
            snapshot.get("receive_ts_ms"),
            snapshot.get("bid_price"),
            snapshot.get("ask_price"),
            snapshot.get("bid_quantity"),
            snapshot.get("ask_quantity"),
        )

    def _latest_bbo_snapshot_with_age(
        self, symbol: str
    ) -> tuple[dict[str, Any] | None, float | None]:
        """Return (snapshot, monotonic-age-seconds).

        The age is measured from the monotonic instant the snapshot *content* last
        changed.  A stalled websocket keeps returning the same quote, so its age
        keeps growing and the freshness guard fails closed once it exceeds the
        configured limit.  Age is ``None`` (unknown) when no snapshot is available.
        """
        getter = getattr(self.bars, "get_latest_book_ticker", None)
        snapshot: dict[str, Any] | None = None
        if callable(getter):
            try:
                fetched = getter(symbol)
            except Exception:
                fetched = None
            if isinstance(fetched, dict):
                snapshot = dict(fetched)
        now = time.monotonic()
        with self._state_lock:
            if snapshot is None:
                self._bbo_cache.pop(str(symbol), None)
                return None, None
            identity = self._bbo_identity(snapshot)
            cached = self._bbo_cache.get(str(symbol))
            if cached is not None and self._bbo_identity(cached[0]) == identity:
                first_seen = float(cached[1])
            else:
                first_seen = now
                self._bbo_cache[str(symbol)] = (dict(snapshot), now)
            return snapshot, max(0.0, now - first_seen)

    def invalidate_bbo_cache(self, symbol: str | None = None) -> None:
        """Drop cached BBO snapshot(s) so a stale quote is not reused after a
        websocket gap-recovery / reconnect.  ``symbol=None`` clears all symbols.
        """
        with self._state_lock:
            if symbol is None:
                self._bbo_cache.clear()
            else:
                self._bbo_cache.pop(str(symbol), None)

    @staticmethod
    def _bbo_spread_bps(snapshot: dict[str, Any]) -> float | None:
        try:
            bid = float(snapshot.get("bid_price") or snapshot.get("bid") or 0.0)
            ask = float(snapshot.get("ask_price") or snapshot.get("ask") or 0.0)
        except Exception:
            return None
        if bid <= 0.0 or ask <= 0.0 or ask < bid:
            return None
        mid = (bid + ask) / 2.0
        if mid <= 0.0:
            return None
        return ((ask - bid) / mid) * 10_000.0

    @staticmethod
    def _estimated_one_way_slippage_bps(event, snapshot: dict[str, Any]) -> float | None:
        try:
            bid = float(snapshot.get("bid_price") or snapshot.get("bid") or 0.0)
            ask = float(snapshot.get("ask_price") or snapshot.get("ask") or 0.0)
            limit_price = float(getattr(event, "price", 0.0) or 0.0)
        except Exception:
            return None
        if bid <= 0.0 or ask <= 0.0 or ask < bid or limit_price <= 0.0:
            return None
        mid = (bid + ask) / 2.0
        if mid <= 0.0:
            return None
        direction = str(getattr(event, "direction", "") or "").strip().upper()
        if direction == "BUY":
            return max(0.0, ((limit_price - mid) / mid) * 10_000.0)
        if direction == "SELL":
            return max(0.0, ((mid - limit_price) / mid) * 10_000.0)
        return None

    def _record_slippage_guard_check(
        self,
        event,
        *,
        snapshot: dict[str, Any] | None,
        spread_bps: float | None,
        estimated_bps: float | None,
        status: str,
        reason: str | None = None,
    ) -> None:
        metadata = dict(getattr(event, "metadata", None) or {})
        metadata["slippage_guard_check"] = {
            "status": str(status),
            "reason": reason,
            "bbo_spread_bps_at_submit": spread_bps,
            "estimated_one_way_slippage_bps_at_submit": estimated_bps,
            "bbo_snapshot": dict(snapshot or {}),
            "market_fallback_allowed": False,
        }
        event.metadata = metadata

    def _validate_limit_slippage_guard(self, event) -> None:
        if canonical_order_type(getattr(event, "order_type", "LMT"), default="LMT") != "LMT":
            return
        policy = self._slippage_guard_policy_for_event(event)
        if not policy:
            return

        require_bbo = bool(
            policy.get("require_bbo_snapshot")
            or getattr(self.config, "REQUIRE_BBO_FOR_LIMIT_ORDERS", False)
        )
        snapshot, bbo_age_seconds = self._latest_bbo_snapshot_with_age(
            str(getattr(event, "symbol", ""))
        )
        if not snapshot:
            self._record_slippage_guard_check(
                event,
                snapshot=None,
                spread_bps=None,
                estimated_bps=None,
                status="missing_bbo",
                reason="missing_bbo_snapshot",
            )
            if require_bbo:
                self._notify_state(
                    order_id=None,
                    entry={
                        "event": event,
                        "symbol": getattr(event, "symbol", None),
                        "last_filled": 0.0,
                        "created_at": time.time(),
                    },
                    state=STATE_REJECTED,
                    message="slippage_guard_missing_bbo",
                    metadata={"market_fallback_allowed": False},
                )
                raise RuntimeError(
                    "Limit slippage guard requires a fresh BBO snapshot; order skipped with no market fallback."
                )
            return

        # Fail-closed freshness guard: after a websocket stall/reconnect the cached
        # BBO can be stale; reject rather than validate against a dislocated quote.
        # MAX_BBO_AGE_SECONDS<=0 disables the check (legacy behavior).
        if self.max_bbo_age_seconds > 0.0 and (
            bbo_age_seconds is None or bbo_age_seconds > self.max_bbo_age_seconds
        ):
            reason = (
                "bbo_age_unknown"
                if bbo_age_seconds is None
                else f"bbo_age_seconds={bbo_age_seconds:.6g}>{self.max_bbo_age_seconds:.6g}"
            )
            self._record_slippage_guard_check(
                event,
                snapshot=snapshot,
                spread_bps=self._bbo_spread_bps(snapshot),
                estimated_bps=self._estimated_one_way_slippage_bps(event, snapshot),
                status="stale_bbo",
                reason=reason,
            )
            self._notify_state(
                order_id=None,
                entry={
                    "event": event,
                    "symbol": getattr(event, "symbol", None),
                    "last_filled": 0.0,
                    "created_at": time.time(),
                },
                state=STATE_REJECTED,
                message="slippage_guard_stale_bbo",
                metadata={
                    "reason": reason,
                    "bbo_age_seconds": bbo_age_seconds,
                    "max_bbo_age_seconds": self.max_bbo_age_seconds,
                    "market_fallback_allowed": False,
                },
            )
            raise RuntimeError(
                "Limit slippage guard BBO snapshot is stale; order skipped with no market fallback: "
                + reason
            )

        spread_bps = self._bbo_spread_bps(snapshot)
        estimated_bps = self._estimated_one_way_slippage_bps(event, snapshot)
        max_spread = policy.get("max_bbo_spread_bps_at_submit")
        max_estimated = policy.get("max_estimated_one_way_slippage_bps")
        breach_reasons: list[str] = []
        if max_spread is not None and spread_bps is not None and spread_bps > float(max_spread):
            breach_reasons.append(
                f"bbo_spread_bps_at_submit={spread_bps:.6g}>{float(max_spread):.6g}"
            )
        if (
            max_estimated is not None
            and estimated_bps is not None
            and estimated_bps > float(max_estimated)
        ):
            breach_reasons.append(
                "estimated_one_way_slippage_bps_at_submit="
                f"{estimated_bps:.6g}>{float(max_estimated):.6g}"
            )

        if breach_reasons:
            reason = ";".join(breach_reasons)
            self._record_slippage_guard_check(
                event,
                snapshot=snapshot,
                spread_bps=spread_bps,
                estimated_bps=estimated_bps,
                status="breach",
                reason=reason,
            )
            self._notify_state(
                order_id=None,
                entry={
                    "event": event,
                    "symbol": getattr(event, "symbol", None),
                    "last_filled": 0.0,
                    "created_at": time.time(),
                },
                state=STATE_REJECTED,
                message="slippage_guard_breach",
                metadata={
                    "reason": reason,
                    "bbo_spread_bps_at_submit": spread_bps,
                    "estimated_one_way_slippage_bps_at_submit": estimated_bps,
                    "market_fallback_allowed": False,
                },
            )
            raise RuntimeError(
                "Limit slippage guard breached; order skipped with no market fallback: " + reason
            )

        self._record_slippage_guard_check(
            event,
            snapshot=snapshot,
            spread_bps=spread_bps,
            estimated_bps=estimated_bps,
            status="passed",
        )

    def _emit_fill_event(self, event, order, filled_qty, status, submitted_at=None):
        if filled_qty <= 0:
            return

        fill_price = (
            order.get("average")
            or order.get("price")
            or self.bars.get_latest_bar_value(event.symbol, "close")
            or 0.0
        )
        reference_price = (
            getattr(event, "price", None)
            or self.bars.get_latest_bar_value(event.symbol, "close")
            or order.get("price")
            or order.get("average")
            or 0.0
        )
        reference_price = float(reference_price or 0.0)
        realized_slippage_bps = None
        if reference_price > 0.0 and float(fill_price or 0.0) > 0.0:
            if str(getattr(event, "direction", "")).upper() == "BUY":
                realized_slippage_bps = (
                    (float(fill_price) - reference_price) / reference_price
                ) * 10_000.0
            else:
                realized_slippage_bps = (
                    (reference_price - float(fill_price)) / reference_price
                ) * 10_000.0
        submit_to_fill_ms = None
        if submitted_at is not None:
            try:
                submit_to_fill_ms = max(0, int((time.time() - float(submitted_at)) * 1000.0))
            except Exception:
                submit_to_fill_ms = None
        live_mode = str(getattr(self.config, "MODE", "paper") or "paper").strip().lower()
        _order_type = str(getattr(event, "order_type", "MKT") or "MKT").upper()
        _is_maker = _order_type == "LMT"
        if float(fill_price or 0.0) > 0.0:
            if live_mode == "real":
                # Real fills: the exchange already determined the fill price (slippage
                # is in the price, not to be re-applied).  Use commission_for — the
                # single fee path: fill_price * qty * fee_rate, maker vs taker.
                # Prefer exchange-reported fee (ccxt order["fee"]["cost"]) when present;
                # commission_for is the fallback so config-driven rates are consistent.
                _exchange_fee: float | None = None
                try:
                    _fee_info = order.get("fee") if isinstance(order, dict) else None
                    if isinstance(_fee_info, dict) and float(_fee_info.get("cost") or 0.0) > 0.0:
                        _exchange_fee = float(_fee_info["cost"])
                except Exception:
                    pass
                commission = (
                    _exchange_fee
                    if _exchange_fee is not None
                    else self._execution_model.commission_for(
                        fill_price=float(fill_price),
                        qty=float(filled_qty),
                        is_maker=_is_maker,
                    )
                )
            else:
                # Paper / testnet: simulate slippage + commission through the unified
                # ExecutionModel so backtest ↔ live cost paths share one formula.
                # apply_liquidity_cap=False: qty already determined by the simulation.
                _fill_result = self._execution_model.compute_fill(
                    raw_price=float(fill_price),
                    qty=float(filled_qty),
                    direction=str(getattr(event, "direction", "BUY")),
                    bar_volume=0.0,
                    is_maker=_is_maker,
                    apply_liquidity_cap=False,
                )
                commission = _fill_result.commission
        else:
            commission = 0.0
        fill_event = FillEvent(
            timeindex=self.bars.get_latest_bar_datetime(event.symbol),
            symbol=event.symbol,
            exchange=getattr(self.config, "EXCHANGE_ID", "EXCHANGE"),
            quantity=filled_qty,
            direction=event.direction,
            fill_cost=fill_price * filled_qty,
            commission=commission,
            order_id=order.get("id"),
            client_order_id=event.client_order_id,
            position_side=event.position_side,
            status=status.upper(),
            metadata={
                "reduce_only": event.reduce_only,
                "order_type": getattr(event, "order_type", None),
                "reference_price": (reference_price if reference_price > 0.0 else None),
                "decision_price": getattr(event, "price", None),
                "fill_price": float(fill_price),
                "realized_slippage_bps": realized_slippage_bps,
                "submit_to_fill_ms": submit_to_fill_ms,
                "signal_metadata": dict(getattr(event, "metadata", None) or {}),
                "component_id": str(
                    dict(getattr(event, "metadata", None) or {}).get("component_id") or ""
                ).strip()
                or None,
            },
        )
        self.events.put(fill_event)
        self.logger.info(
            "Fill emitted: %s %s qty=%s @ %s status=%s",
            event.symbol,
            event.direction,
            filled_qty,
            fill_price,
            status,
        )

    def get_balance(self):
        """Returns the free USDT balance."""
        return self._call_with_retry(self.exchange.get_balance, "USDT", retries=3, delay=2)

    def get_all_positions(self):
        """Returns a dict of {symbol: quantity} for open positions."""
        return self._call_with_retry(self.exchange.get_all_positions, retries=3, delay=1)

    def execute_order(self, event):
        """Executes an OrderEvent on the Exchange."""
        if event.type != "ORDER":
            return

        side = "buy" if event.direction == "BUY" else "sell"
        requested_order_type = canonical_order_type(
            getattr(event, "order_type", "LMT"), default="LMT"
        )
        market_guard_configured = hasattr(self.config, "ALLOW_MARKET_ORDERS")
        if (
            requested_order_type == "MKT"
            and market_guard_configured
            and not bool(getattr(self.config, "ALLOW_MARKET_ORDERS", False))
        ):
            raise RuntimeError(
                "Market orders are disabled by live.allow_market_orders=false; "
                "set live.default_order_type=LMT or explicitly enable market orders."
            )
        if requested_order_type == "LMT" and (
            getattr(event, "price", None) is None
            or float(getattr(event, "price", 0.0) or 0.0) <= 0.0
        ):
            raise RuntimeError(
                "LIMIT orders require a positive limit price before live submission."
            )
        event.order_type = requested_order_type
        self._validate_limit_slippage_guard(event)
        order_type = "limit" if requested_order_type == "LMT" else "market"
        event.client_order_id = event.client_order_id or self._make_client_order_id(event)

        with self._state_lock:
            if event.client_order_id in self.client_id_to_order:
                self.logger.warning(
                    "Duplicate client_order_id detected, skipping submit: %s",
                    event.client_order_id,
                )
                return

        params = self._build_exchange_params(event)
        self._validate_protective_order_params(event, params)
        self.logger.info(
            "Submitting order symbol=%s side=%s qty=%s type=%s client_id=%s",
            event.symbol,
            side,
            event.quantity,
            order_type,
            event.client_order_id,
        )
        self._notify_state(
            order_id=None,
            entry={
                "event": event,
                "symbol": event.symbol,
                "last_filled": 0.0,
                "created_at": time.time(),
            },
            state=STATE_SUBMITTED,
        )

        order = self._submit_with_idempotent_retry(
            symbol=event.symbol,
            order_type=order_type,
            side=side,
            quantity=event.quantity,
            price=event.price,
            params=params,
            client_order_id=event.client_order_id,
            retries=3,
            delay=1,
        )

        order_id = order.get("id")
        status = self._normalize_status(order.get("status"))
        filled_qty = float(order.get("filled") or 0.0)
        total_amount = float(order.get("amount") or event.quantity)
        state = self._to_state(status, filled_qty, total_amount)

        if not order_id:
            self.logger.error(
                "Exchange order id is missing for client_id=%s", event.client_order_id
            )
            return

        # Register tracking and emit the submit-response fill atomically.  If the
        # user-stream thread already created/advanced this order_id between the
        # network submit and here, only emit the residual delta and keep the
        # higher cumulative last_filled so the fill is counted exactly once.
        emit_protection_for: Any | None = None
        deferred_notifications: list[dict[str, Any]] = []
        with self._state_lock:
            self.client_id_to_order[event.client_order_id] = order_id
            existing = self.tracked_orders.get(order_id)
            prior_filled = float(existing.get("last_filled", 0.0)) if existing else 0.0
            self.tracked_orders[order_id] = {
                "event": event,
                "symbol": event.symbol,
                "last_filled": max(filled_qty, prior_filled),
                "state": state,
                "created_at": (
                    float(existing.get("created_at", time.time())) if existing else time.time()
                ),
                "updated_at": time.time(),
            }
            self._notify_state(
                order_id=order_id,
                entry=self.tracked_orders[order_id],
                state=state,
                defer_to=deferred_notifications,
            )

            if state == STATE_FILLED:
                response_filled = filled_qty if filled_qty > 0 else total_amount
                delta = self._ledger_residual_delta(order_id, response_filled)
                if delta > 0:
                    self._emit_fill_event(
                        event,
                        order,
                        delta,
                        status="filled",
                        submitted_at=self.tracked_orders.get(order_id, {}).get("created_at"),
                    )
                emit_protection_for = event
                self._forget_order(order_id, self.tracked_orders.get(order_id, {}))
            elif state in {STATE_OPEN, STATE_NEW, STATE_ACKED, STATE_PARTIAL}:
                self.logger.info("Order tracked order_id=%s state=%s", order_id, state)
            elif state in {STATE_CANCELED, STATE_REJECTED}:
                self.logger.warning("Order terminal without fill id=%s state=%s", order_id, state)
                self._notify_state(
                    order_id=order_id,
                    entry=self.tracked_orders.get(order_id),
                    state=state,
                    defer_to=deferred_notifications,
                )
                self._forget_order(order_id, self.tracked_orders.get(order_id, {}))
            else:
                self.logger.info("Order tracked id=%s state=%s", order_id, state)

        self._fire_deferred_notifications(deferred_notifications)
        if emit_protection_for is not None:
            self._submit_paper_exchange_protection(emit_protection_for, parent_order_id=order_id)

    def check_open_orders(self, event=None):
        """Poll tracked exchange orders and emit delta fills for partial/full executions."""
        if self.order_state_source == "user_stream":
            force_poll = bool(event is None)
            if not self.poll_fallback_enabled or not force_poll:
                return
        for order_id, entry in list(self.tracked_orders.items()):
            order_event = entry["event"]
            previous_state = str(entry.get("state", STATE_OPEN))
            now = time.time()

            if (
                previous_state
                in {STATE_NEW, STATE_SUBMITTED, STATE_ACKED, STATE_OPEN, STATE_PARTIAL}
                and now - float(entry.get("created_at", now)) >= self.order_timeout_sec
            ):
                canceled = False
                try:
                    canceled = bool(self.order_gateway.cancel(order_id, order_event.symbol))
                except Exception as exc:
                    self.logger.error("Failed to cancel timed-out order %s: %s", order_id, exc)
                timeout_state = STATE_TIMEOUT if canceled else STATE_OPEN
                if timeout_state == STATE_TIMEOUT:
                    timeout_message = "order_timeout"
                    try:
                        latest_after_cancel = self._call_with_retry(
                            self.order_gateway.query_order,
                            order_id,
                            order_event.symbol,
                            retries=2,
                            delay=0.5,
                        )
                    except Exception:
                        latest_after_cancel = {}

                    emit_protection_for: Any | None = None
                    deferred_notifications: list[dict[str, Any]] = []
                    # Apply the post-cancel reconciliation atomically: re-read
                    # last_filled under the lock so a concurrent user-stream fill
                    # cannot be double-counted, then emit only the residual delta.
                    with self._state_lock:
                        if latest_after_cancel:
                            status_after_cancel = self._normalize_status(
                                latest_after_cancel.get("status")
                            )
                            filled_after_cancel = float(latest_after_cancel.get("filled") or 0.0)
                            amount_after_cancel = float(
                                latest_after_cancel.get("amount") or order_event.quantity
                            )
                            delta_after_cancel = self._ledger_residual_delta(
                                order_id, filled_after_cancel
                            )
                            if delta_after_cancel > 0:
                                self._emit_fill_event(
                                    order_event,
                                    latest_after_cancel,
                                    delta_after_cancel,
                                    status=status_after_cancel,
                                    submitted_at=entry.get("created_at"),
                                )
                                entry["last_filled"] = filled_after_cancel
                            terminal_after_cancel = self._to_state(
                                status_after_cancel,
                                filled_after_cancel,
                                amount_after_cancel,
                            )
                            if terminal_after_cancel in TERMINAL_STATES:
                                entry["state"] = terminal_after_cancel
                                timeout_message = "timeout_reconciled"
                            else:
                                entry["state"] = STATE_TIMEOUT
                        else:
                            entry["state"] = STATE_TIMEOUT

                        entry["updated_at"] = now
                        self._notify_state(
                            order_id=order_id,
                            entry=entry,
                            state=entry["state"],
                            message=timeout_message,
                            defer_to=deferred_notifications,
                        )
                        if entry["state"] == STATE_FILLED:
                            emit_protection_for = order_event
                        self._forget_order(order_id, entry)
                    self._fire_deferred_notifications(deferred_notifications)
                    if emit_protection_for is not None:
                        self._submit_paper_exchange_protection(
                            emit_protection_for, parent_order_id=order_id
                        )
                    continue

            try:
                latest = self._call_with_retry(
                    self.order_gateway.query_order,
                    order_id,
                    order_event.symbol,
                    retries=3,
                    delay=1,
                )
            except Exception as exc:
                self.logger.error("Failed to poll order %s: %s", order_id, exc)
                continue
            if not latest:
                continue

            status = self._normalize_status(latest.get("status"))
            filled_now = float(latest.get("filled") or 0.0)
            total_amount = float(latest.get("amount") or order_event.quantity)
            emit_protection_for = None
            deferred_notifications = []
            # Apply the polled result atomically: re-read last_filled under the lock
            # so a concurrent user-stream fill (same order) is never double-counted
            # or lost — only the residual cumulative delta is emitted.
            with self._state_lock:
                delta = self._ledger_residual_delta(order_id, filled_now)
                if delta > 0:
                    self._emit_fill_event(
                        order_event,
                        latest,
                        delta,
                        status=status,
                        submitted_at=entry.get("created_at"),
                    )
                    entry["last_filled"] = filled_now

                entry["state"] = self._to_state(status, filled_now, total_amount)
                entry["updated_at"] = time.time()
                if entry["state"] != previous_state:
                    self._notify_state(
                        order_id=order_id,
                        entry=entry,
                        state=entry["state"],
                        defer_to=deferred_notifications,
                    )
                if entry["state"] in TERMINAL_STATES:
                    if entry["state"] == STATE_FILLED:
                        emit_protection_for = order_event
                    self._forget_order(order_id, entry)
            self._fire_deferred_notifications(deferred_notifications)
            if emit_protection_for is not None:
                self._submit_paper_exchange_protection(
                    emit_protection_for, parent_order_id=order_id
                )
