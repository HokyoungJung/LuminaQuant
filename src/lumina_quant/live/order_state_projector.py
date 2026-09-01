"""User-stream order/account event projector with idempotency guards."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from lumina_quant.live.order_state_machine import OrderStateMachine


@dataclass(slots=True)
class ProjectionResult:
    accepted: bool
    event_key: str
    next_state: str
    fill_delta: float
    cumulative_filled: float
    reason: str


class OrderStateProjector:
    """Project user-stream events into local order-state transitions."""

    def __init__(
        self,
        *,
        dedupe_max_keys: int = 100_000,
        state_machine: OrderStateMachine | None = None,
    ) -> None:
        self.dedupe_max_keys = max(1000, int(dedupe_max_keys))
        self._seen: OrderedDict[str, int] = OrderedDict()
        self.state_machine = state_machine or OrderStateMachine()

    @staticmethod
    def build_execution_report_key(payload: dict) -> str:
        symbol = str(payload.get("symbol") or payload.get("s") or "")
        order_id = str(payload.get("order_id") or payload.get("i") or "")
        exec_type = str(payload.get("exec_type") or payload.get("x") or "")
        order_status = str(payload.get("order_status") or payload.get("X") or "")
        event_ts = int(payload.get("exchange_ts_ms") or payload.get("E") or 0)
        trade_id = payload.get("trade_id")
        if trade_id is None:
            trade_id = payload.get("t")
        if trade_id in {None, "", -1}:
            last_fill_qty = float(payload.get("last_fill_qty") or payload.get("l") or 0.0)
            cum_fill_qty = float(payload.get("cum_fill_qty") or payload.get("z") or 0.0)
            trade_id = f"{exec_type}:{order_status}:{last_fill_qty}:{cum_fill_qty}:{event_ts}"
        return f"usr:er:{symbol}:{order_id}:{trade_id}"

    @staticmethod
    def build_account_key(payload: dict) -> str:
        event_type = str(payload.get("event_type") or payload.get("e") or "")
        event_ts = int(payload.get("exchange_ts_ms") or payload.get("E") or 0)
        body = str(payload)
        return f"usr:{event_type}:{event_ts}:{hash(body)}"

    def _check_duplicate(self, key: str, event_ts_ms: int) -> bool:
        if key in self._seen:
            return True
        self._seen[key] = int(event_ts_ms)
        while len(self._seen) > int(self.dedupe_max_keys):
            self._seen.popitem(last=False)
        return False

    def project_execution_report(
        self,
        payload: dict,
        *,
        previous_state: str,
        previous_filled: float,
    ) -> ProjectionResult:
        key = self.build_execution_report_key(payload)
        event_ts = int(payload.get("exchange_ts_ms") or payload.get("E") or 0)
        if self._check_duplicate(key, event_ts):
            return ProjectionResult(
                accepted=False,
                event_key=key,
                next_state=str(previous_state).upper(),
                fill_delta=0.0,
                cumulative_filled=float(previous_filled),
                reason="DUPLICATE_EVENT",
            )

        mapped = self.state_machine.map_binance_execution_report(payload)
        transition = self.state_machine.transition(str(previous_state).upper(), str(mapped).upper())

        cumulative = float(payload.get("cum_fill_qty") or payload.get("z") or previous_filled)
        delta = max(0.0, float(cumulative) - float(previous_filled))
        return ProjectionResult(
            accepted=bool(transition.accepted),
            event_key=key,
            next_state=str(transition.next_state).upper(),
            fill_delta=float(delta),
            cumulative_filled=float(cumulative),
            reason=str(transition.reason),
        )


__all__ = ["OrderStateProjector", "ProjectionResult"]
