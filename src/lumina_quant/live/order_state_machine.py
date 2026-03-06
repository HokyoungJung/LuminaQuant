"""Formal live order-state transition policy."""

from __future__ import annotations

from dataclasses import dataclass

STATE_SUBMITTED = "SUBMITTED"
STATE_ACKED = "ACKED"
STATE_OPEN = "OPEN"
STATE_PARTIAL = "PARTIAL"
STATE_FILLED = "FILLED"
STATE_CANCELED = "CANCELED"
STATE_REJECTED = "REJECTED"
STATE_TIMEOUT = "TIMEOUT"


TERMINAL_STATES = {STATE_FILLED, STATE_CANCELED, STATE_REJECTED, STATE_TIMEOUT}


_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    STATE_SUBMITTED: {
        STATE_ACKED,
        STATE_OPEN,
        STATE_PARTIAL,
        STATE_FILLED,
        STATE_REJECTED,
        STATE_CANCELED,
        STATE_TIMEOUT,
    },
    STATE_ACKED: {
        STATE_OPEN,
        STATE_PARTIAL,
        STATE_FILLED,
        STATE_REJECTED,
        STATE_CANCELED,
        STATE_TIMEOUT,
    },
    STATE_OPEN: {STATE_PARTIAL, STATE_FILLED, STATE_CANCELED, STATE_REJECTED, STATE_TIMEOUT},
    STATE_PARTIAL: {STATE_PARTIAL, STATE_FILLED, STATE_CANCELED, STATE_REJECTED, STATE_TIMEOUT},
    STATE_FILLED: set(),
    STATE_CANCELED: set(),
    STATE_REJECTED: set(),
    STATE_TIMEOUT: set(),
}


@dataclass(slots=True)
class TransitionResult:
    previous_state: str
    next_state: str
    accepted: bool
    reason: str


class OrderStateMachine:
    """State machine for stream-driven order lifecycle transitions."""

    @staticmethod
    def map_binance_execution_report(payload: dict) -> str:
        exec_type = str(payload.get("exec_type") or payload.get("x") or "").upper()
        order_status = str(payload.get("order_status") or payload.get("X") or "").upper()
        if order_status in {"REJECTED"}:
            return STATE_REJECTED
        if order_status in {"CANCELED", "CANCELLED"}:
            return STATE_CANCELED
        if order_status in {"EXPIRED"}:
            return STATE_TIMEOUT
        if order_status in {"FILLED"}:
            return STATE_FILLED
        if order_status in {"PARTIALLY_FILLED"}:
            return STATE_PARTIAL
        if exec_type in {"NEW"}:
            return STATE_ACKED
        if order_status in {"NEW", "PENDING_NEW"}:
            return STATE_ACKED
        return STATE_OPEN

    @staticmethod
    def transition(previous_state: str, next_state: str) -> TransitionResult:
        previous = str(previous_state or STATE_SUBMITTED).upper()
        nxt = str(next_state or previous).upper()
        if previous == nxt:
            return TransitionResult(
                previous_state=previous, next_state=nxt, accepted=True, reason="NOOP"
            )
        if previous in TERMINAL_STATES:
            return TransitionResult(
                previous_state=previous,
                next_state=previous,
                accepted=False,
                reason="TERMINAL_STATE_IMMUTABLE",
            )
        allowed = _ALLOWED_TRANSITIONS.get(previous, set())
        if nxt in allowed:
            return TransitionResult(
                previous_state=previous, next_state=nxt, accepted=True, reason="OK"
            )
        return TransitionResult(
            previous_state=previous,
            next_state=previous,
            accepted=False,
            reason="ORDER_INVALID_TRANSITION",
        )


__all__ = ["TERMINAL_STATES", "OrderStateMachine", "TransitionResult"]
