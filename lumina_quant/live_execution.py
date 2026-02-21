import hashlib
import logging
import time
from types import SimpleNamespace
from typing import Any

from lumina_quant.events import FillEvent
from lumina_quant.execution import ExecutionHandler
from lumina_quant.interfaces import ExchangeInterface

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
        self.tracked_orders = {}
        self.client_id_to_order = {}
        self._state_callback = None

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
    ) -> None:
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
        try:
            callback(payload)
        except Exception as exc:  # pragma: no cover - defensive callback guard
            self.logger.error("Order state callback failed: %s", exc)

    def _forget_order(self, order_id: str, entry: dict[str, Any]) -> None:
        self.tracked_orders.pop(order_id, None)
        event = entry.get("event")
        client_order_id = getattr(event, "client_order_id", None)
        if client_order_id and self.client_id_to_order.get(client_order_id) == order_id:
            self.client_id_to_order.pop(client_order_id, None)

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
            exchange_open = list(self.exchange.fetch_open_orders(None) or [])
        except Exception as exc:
            self.logger.error("Failed to fetch open orders for reconciliation: %s", exc)
            return records

        open_ids = {str(item.get("id")) for item in exchange_open if item.get("id")}
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
                    self.exchange.fetch_order,
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
            delta = filled_now - float(entry.get("last_filled", 0.0))
            if delta > 0:
                self._emit_fill_event(order_event, latest, delta, status=status)
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
            self._notify_state(order_id=order_id, entry=entry, state=entry["state"])
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
                self._forget_order(order_id, entry)
        return records

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

    def _estimate_commission(self, fill_price, quantity):
        fee_rate = getattr(
            self.config,
            "TAKER_FEE_RATE",
            getattr(self.config, "COMMISSION_RATE", 0.0),
        )
        return float(fill_price * quantity * fee_rate)

    def _emit_fill_event(self, event, order, filled_qty, status):
        if filled_qty <= 0:
            return

        fill_price = (
            order.get("average")
            or order.get("price")
            or self.bars.get_latest_bar_value(event.symbol, "close")
            or 0.0
        )
        commission = self._estimate_commission(fill_price, filled_qty)
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
            metadata={"reduce_only": event.reduce_only},
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
        order_type = "market" if event.order_type != "LMT" else "limit"
        event.client_order_id = event.client_order_id or self._make_client_order_id(event)

        if event.client_order_id in self.client_id_to_order:
            self.logger.warning(
                "Duplicate client_order_id detected, skipping submit: %s",
                event.client_order_id,
            )
            return

        params = self._build_exchange_params(event)
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

        order = self._call_with_retry(
            self.exchange.execute_order,
            symbol=event.symbol,
            type=order_type,
            side=side,
            quantity=event.quantity,
            price=event.price,
            params=params,
            retries=3,
            delay=1,
        )

        order_id = order.get("id")
        status = self._normalize_status(order.get("status"))
        filled_qty = float(order.get("filled") or 0.0)
        total_amount = float(order.get("amount") or event.quantity)
        state = self._to_state(status, filled_qty, total_amount)

        if order_id:
            self.client_id_to_order[event.client_order_id] = order_id
            self.tracked_orders[order_id] = {
                "event": event,
                "symbol": event.symbol,
                "last_filled": filled_qty,
                "state": state,
                "created_at": time.time(),
                "updated_at": time.time(),
            }
            self._notify_state(order_id=order_id, entry=self.tracked_orders[order_id], state=state)
        else:
            self.logger.error(
                "Exchange order id is missing for client_id=%s", event.client_order_id
            )
            return

        if state == STATE_FILLED:
            if filled_qty <= 0:
                filled_qty = total_amount
            self._emit_fill_event(event, order, filled_qty, status="filled")
            self._forget_order(order_id, self.tracked_orders.get(order_id, {}))
        elif state in {STATE_OPEN, STATE_NEW, STATE_ACKED, STATE_PARTIAL}:
            self.logger.info("Order tracked order_id=%s state=%s", order_id, state)
        elif state in {STATE_CANCELED, STATE_REJECTED}:
            self.logger.warning("Order terminal without fill id=%s state=%s", order_id, state)
            self._notify_state(
                order_id=order_id, entry=self.tracked_orders.get(order_id), state=state
            )
            self._forget_order(order_id, self.tracked_orders.get(order_id, {}))
        else:
            self.logger.info("Order tracked id=%s state=%s", order_id, state)

    def check_open_orders(self, event=None):
        """Poll tracked exchange orders and emit delta fills for partial/full executions."""
        _ = event
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
                    canceled = bool(self.exchange.cancel_order(order_id, order_event.symbol))
                except Exception as exc:
                    self.logger.error("Failed to cancel timed-out order %s: %s", order_id, exc)
                timeout_state = STATE_TIMEOUT if canceled else STATE_OPEN
                if timeout_state == STATE_TIMEOUT:
                    timeout_message = "order_timeout"
                    try:
                        latest_after_cancel = self._call_with_retry(
                            self.exchange.fetch_order,
                            order_id,
                            order_event.symbol,
                            retries=2,
                            delay=0.5,
                        )
                    except Exception:
                        latest_after_cancel = {}

                    if latest_after_cancel:
                        status_after_cancel = self._normalize_status(
                            latest_after_cancel.get("status")
                        )
                        filled_after_cancel = float(latest_after_cancel.get("filled") or 0.0)
                        amount_after_cancel = float(
                            latest_after_cancel.get("amount") or order_event.quantity
                        )
                        delta_after_cancel = filled_after_cancel - float(
                            entry.get("last_filled", 0.0)
                        )
                        if delta_after_cancel > 0:
                            self._emit_fill_event(
                                order_event,
                                latest_after_cancel,
                                delta_after_cancel,
                                status=status_after_cancel,
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
                    )
                    self._forget_order(order_id, entry)
                    continue

            try:
                latest = self._call_with_retry(
                    self.exchange.fetch_order,
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
            delta = filled_now - float(entry.get("last_filled", 0.0))
            if delta > 0:
                self._emit_fill_event(order_event, latest, delta, status=status)
                entry["last_filled"] = filled_now

            entry["state"] = self._to_state(status, filled_now, total_amount)
            entry["updated_at"] = time.time()
            if entry["state"] != previous_state:
                self._notify_state(order_id=order_id, entry=entry, state=entry["state"])
            if entry["state"] in TERMINAL_STATES:
                self._forget_order(order_id, entry)
