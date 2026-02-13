import logging
import time
import uuid
from typing import Any

from lumina_quant.events import FillEvent
from lumina_quant.execution import ExecutionHandler
from lumina_quant.interfaces import ExchangeInterface

STATE_NEW = "NEW"
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
        if status in {"open", "new", "partially_filled", "partial"}:
            if filled_qty > 0 and amount > 0 and filled_qty < amount:
                return STATE_PARTIAL
            if status == "new":
                return STATE_NEW
            return STATE_OPEN
        return STATE_OPEN

    def _make_client_order_id(self):
        return f"LQ-{uuid.uuid4().hex[:24]}"

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

    def _build_exchange_params(self, event):
        params = {}
        if event.metadata and isinstance(event.metadata, dict):
            params.update(event.metadata.get("exchange_params", {}))

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
        event.client_order_id = event.client_order_id or self._make_client_order_id()

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
            self._notify_state(
                order_id=order_id, entry=self.tracked_orders.get(order_id), state=state
            )
            self._forget_order(order_id, self.tracked_orders.get(order_id, {}))
        elif state in {STATE_OPEN, STATE_NEW, STATE_PARTIAL}:
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
                previous_state in {STATE_NEW, STATE_OPEN, STATE_PARTIAL}
                and now - float(entry.get("created_at", now)) >= self.order_timeout_sec
            ):
                canceled = False
                try:
                    canceled = bool(self.exchange.cancel_order(order_id, order_event.symbol))
                except Exception as exc:
                    self.logger.error("Failed to cancel timed-out order %s: %s", order_id, exc)
                timeout_state = STATE_TIMEOUT if canceled else STATE_OPEN
                if timeout_state == STATE_TIMEOUT:
                    entry["state"] = STATE_TIMEOUT
                    entry["updated_at"] = now
                    self._notify_state(
                        order_id=order_id,
                        entry=entry,
                        state=STATE_TIMEOUT,
                        message="order_timeout",
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
