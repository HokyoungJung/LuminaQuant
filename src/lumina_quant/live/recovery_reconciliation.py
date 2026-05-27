"""Recovery and fallback reconciliation service for live runtime."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

ReconciliationSignature = tuple[tuple[str, ...], ...]


@dataclass(slots=True)
class ReconciliationOutcome:
    converged: bool
    elapsed_seconds: float
    tracked_orders: int
    cache_open_orders: int
    exchange_open_orders: int | None = None
    exchange_snapshot_ready: bool | None = None
    tracked_cache_signature_match: bool | None = None
    exchange_signature_match: bool | None = None


def _normalize_signature(payload: object) -> ReconciliationSignature:
    if payload is None:
        return tuple()
    items: list[tuple[str, ...]] = []
    for row in tuple(payload if isinstance(payload, (tuple, list, set)) else (payload,)):
        if isinstance(row, (tuple, list)):
            items.append(tuple(str(item) for item in row))
        else:
            items.append((str(row),))
    return tuple(sorted(items))


def _signature_ids(signature: ReconciliationSignature) -> tuple[str, ...]:
    return tuple(item[0] for item in signature if item)


class RecoveryReconciliationService:
    """Run startup and fallback reconciliation loops with deterministic timeout behavior."""

    def __init__(
        self,
        *,
        reconcile_positions: Callable[[], None],
        reconcile_orders: Callable[[], None],
        tracked_orders_count: Callable[[], int],
        cache_open_orders_count: Callable[[], int],
        exchange_open_orders_count: Callable[[], int] | None = None,
        exchange_snapshot_ready: Callable[[], bool] | None = None,
        tracked_order_signature: Callable[[], object] | None = None,
        cache_open_order_signature: Callable[[], object] | None = None,
        exchange_open_order_signature: Callable[[], object] | None = None,
    ) -> None:
        self._reconcile_positions = reconcile_positions
        self._reconcile_orders = reconcile_orders
        self._tracked_orders_count = tracked_orders_count
        self._cache_open_orders_count = cache_open_orders_count
        self._exchange_open_orders_count = exchange_open_orders_count
        self._exchange_snapshot_ready = exchange_snapshot_ready
        self._tracked_order_signature = tracked_order_signature
        self._cache_open_order_signature = cache_open_order_signature
        self._exchange_open_order_signature = exchange_open_order_signature

    def startup_converge(self, *, timeout_seconds: float = 90.0) -> ReconciliationOutcome:
        started = time.monotonic()
        timeout = max(1.0, float(timeout_seconds))
        while time.monotonic() - started <= timeout:
            self._reconcile_positions()
            self._reconcile_orders()
            tracked = int(self._tracked_orders_count())
            cached = int(self._cache_open_orders_count())
            exchange_count = (
                int(self._exchange_open_orders_count())
                if self._exchange_open_orders_count is not None
                else None
            )
            snapshot_ready = (
                bool(self._exchange_snapshot_ready())
                if self._exchange_snapshot_ready is not None
                else None
            )
            tracked_signature = (
                _normalize_signature(self._tracked_order_signature())
                if self._tracked_order_signature is not None
                else None
            )
            cache_signature = (
                _normalize_signature(self._cache_open_order_signature())
                if self._cache_open_order_signature is not None
                else None
            )
            exchange_signature = (
                _normalize_signature(self._exchange_open_order_signature())
                if self._exchange_open_order_signature is not None
                else None
            )
            tracked_cache_signature_match = (
                tracked_signature == cache_signature
                if tracked_signature is not None and cache_signature is not None
                else None
            )
            exchange_signature_match = (
                _signature_ids(tracked_signature or tuple()) == _signature_ids(exchange_signature)
                if tracked_signature is not None and exchange_signature is not None
                else None
            )
            converged = tracked == cached and (
                exchange_count is None or tracked == int(exchange_count)
            )
            if tracked_cache_signature_match is False:
                converged = False
            if exchange_signature_match is False:
                converged = False
            if snapshot_ready is False:
                converged = False
            if converged:
                return ReconciliationOutcome(
                    converged=True,
                    elapsed_seconds=float(time.monotonic() - started),
                    tracked_orders=tracked,
                    cache_open_orders=cached,
                    exchange_open_orders=exchange_count,
                    exchange_snapshot_ready=snapshot_ready,
                    tracked_cache_signature_match=tracked_cache_signature_match,
                    exchange_signature_match=exchange_signature_match,
                )
            time.sleep(1.0)
        tracked = int(self._tracked_orders_count())
        cached = int(self._cache_open_orders_count())
        exchange_count = (
            int(self._exchange_open_orders_count())
            if self._exchange_open_orders_count is not None
            else None
        )
        snapshot_ready = (
            bool(self._exchange_snapshot_ready())
            if self._exchange_snapshot_ready is not None
            else None
        )
        tracked_signature = (
            _normalize_signature(self._tracked_order_signature())
            if self._tracked_order_signature is not None
            else None
        )
        cache_signature = (
            _normalize_signature(self._cache_open_order_signature())
            if self._cache_open_order_signature is not None
            else None
        )
        exchange_signature = (
            _normalize_signature(self._exchange_open_order_signature())
            if self._exchange_open_order_signature is not None
            else None
        )
        return ReconciliationOutcome(
            converged=False,
            elapsed_seconds=float(time.monotonic() - started),
            tracked_orders=tracked,
            cache_open_orders=cached,
            exchange_open_orders=exchange_count,
            exchange_snapshot_ready=snapshot_ready,
            tracked_cache_signature_match=(
                tracked_signature == cache_signature
                if tracked_signature is not None and cache_signature is not None
                else None
            ),
            exchange_signature_match=(
                _signature_ids(tracked_signature or tuple()) == _signature_ids(exchange_signature)
                if tracked_signature is not None and exchange_signature is not None
                else None
            ),
        )


__all__ = ["ReconciliationOutcome", "RecoveryReconciliationService"]
