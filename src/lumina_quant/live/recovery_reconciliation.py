"""Recovery and fallback reconciliation service for live runtime."""

from __future__ import annotations

import time
from dataclasses import dataclass
from collections.abc import Callable


@dataclass(slots=True)
class ReconciliationOutcome:
    converged: bool
    elapsed_seconds: float
    tracked_orders: int
    cache_open_orders: int
    exchange_open_orders: int | None = None
    exchange_snapshot_ready: bool | None = None


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
    ) -> None:
        self._reconcile_positions = reconcile_positions
        self._reconcile_orders = reconcile_orders
        self._tracked_orders_count = tracked_orders_count
        self._cache_open_orders_count = cache_open_orders_count
        self._exchange_open_orders_count = exchange_open_orders_count
        self._exchange_snapshot_ready = exchange_snapshot_ready

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
            converged = tracked == cached and (
                exchange_count is None or tracked == int(exchange_count)
            )
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
        return ReconciliationOutcome(
            converged=False,
            elapsed_seconds=float(time.monotonic() - started),
            tracked_orders=tracked,
            cache_open_orders=cached,
            exchange_open_orders=exchange_count,
            exchange_snapshot_ready=snapshot_ready,
        )


__all__ = ["ReconciliationOutcome", "RecoveryReconciliationService"]
