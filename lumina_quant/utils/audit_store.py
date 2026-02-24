"""Runtime audit store backed by local PostgreSQL state tables."""

from __future__ import annotations

import os
import threading
import uuid
from datetime import UTC, datetime
from typing import Any

from lumina_quant.postgres_state import PostgresStateRepository, payload_fingerprint


def _utc_iso() -> str:
    return datetime.now(UTC).isoformat()


class AuditStore:
    """Compatibility audit API implemented on top of PostgresStateRepository."""

    def __init__(self, db_path: str = ""):
        resolved_dsn = (
            str(db_path or "").strip()
            or str(os.getenv("LQ_POSTGRES_DSN", "")).strip()
            or str(os.getenv("LQ__STORAGE__POSTGRES_DSN", "")).strip()
        )
        if not resolved_dsn:
            raise ValueError(
                "Postgres DSN is required. Set LQ_POSTGRES_DSN (or storage.postgres_dsn)."
            )
        self.db_path = resolved_dsn
        self._repo = PostgresStateRepository(dsn=resolved_dsn)
        self._lock = threading.Lock()
        self._repo.initialize_schema()

    def start_run(self, mode: str, metadata: dict[str, Any] | None = None, run_id: str | None = None) -> str:
        resolved_run_id = str(run_id or uuid.uuid4())
        with self._lock:
            return self._repo.start_run(mode=mode, metadata=metadata or {}, run_id=resolved_run_id)

    def end_run(self, run_id: str, status: str = "COMPLETED", metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._repo.end_run(run_id=run_id, status=status, metadata=metadata or {})

    def log_order(self, run_id: str, order_event: Any, status: str = "NEW", exchange_order_id: str | None = None) -> None:
        payload = dict(getattr(order_event, "metadata", {}) or {})
        created_at = str(getattr(order_event, "created_at", "") or _utc_iso())
        with self._lock:
            price_value = getattr(order_event, "price", None)
            self._repo.upsert_order(
                run_id=run_id,
                created_at=created_at,
                symbol=str(getattr(order_event, "symbol", "")),
                side=str(getattr(order_event, "direction", "")),
                order_type=str(getattr(order_event, "order_type", "")),
                quantity=float(getattr(order_event, "quantity", 0.0) or 0.0),
                price=(float(price_value) if price_value is not None else None),
                status=str(status),
                client_order_id=str(getattr(order_event, "client_order_id", "") or ""),
                exchange_order_id=exchange_order_id,
                metadata=payload,
            )

    def log_fill(self, run_id: str, fill_event: Any) -> None:
        fill_cost = getattr(fill_event, "fill_cost", None)
        qty = float(getattr(fill_event, "quantity", 0.0) or 0.0)
        fill_price = None
        if fill_cost is not None and qty:
            fill_price = float(fill_cost) / qty
        with self._lock:
            self._repo.upsert_fill(
                run_id=run_id,
                fill_time=str(getattr(fill_event, "timeindex", "") or _utc_iso()),
                symbol=str(getattr(fill_event, "symbol", "")),
                side=str(getattr(fill_event, "direction", "")),
                quantity=qty,
                fill_price=fill_price,
                fill_cost=float(fill_cost) if fill_cost is not None else None,
                commission=float(getattr(fill_event, "commission", 0.0) or 0.0),
                client_order_id=str(getattr(fill_event, "client_order_id", "") or ""),
                exchange_order_id=str(getattr(fill_event, "order_id", "") or ""),
                status=str(getattr(fill_event, "status", "") or ""),
                metadata=dict(getattr(fill_event, "metadata", {}) or {}),
            )

    def log_equity(
        self,
        run_id: str,
        timeindex: str | datetime,
        total: float,
        cash: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._repo.upsert_equity(
                run_id=run_id,
                timeindex=timeindex,
                total=float(total),
                cash=float(cash) if cash is not None else None,
                metadata=metadata or {},
            )

    def log_risk_event(self, run_id: str, reason: str, details: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._repo.upsert_risk_event(run_id=run_id, reason=reason, details=details or {})

    def log_heartbeat(self, run_id: str, status: str = "ALIVE", details: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._repo.upsert_heartbeat(
                run_id=run_id,
                status=status,
                details=details or {},
                worker_id=str((details or {}).get("worker_id", "") or ""),
            )

    def log_order_state(self, run_id: str, state_payload: dict[str, Any]) -> None:
        details = dict(state_payload.get("metadata") or {})
        if "last_filled" in state_payload:
            details["last_filled"] = state_payload.get("last_filled")
        if "created_at" in state_payload:
            details["created_at"] = state_payload.get("created_at")
        with self._lock:
            self._repo.upsert_order_state_event(
                run_id=run_id,
                event_time=str(state_payload.get("event_time") or _utc_iso()),
                state=str(state_payload.get("state", "UNKNOWN")),
                symbol=state_payload.get("symbol"),
                client_order_id=state_payload.get("client_order_id"),
                exchange_order_id=state_payload.get("order_id"),
                message=state_payload.get("message"),
                details=details,
            )

    def log_order_reconciliation(self, run_id: str, payload: dict[str, Any]) -> None:
        """Persist reconciliation as a risk event for compatibility."""
        reason = str(payload.get("reason") or "ORDER_RECONCILIATION")
        details = dict(payload.get("metadata") or {})
        details.update(
            {
                "symbol": payload.get("symbol"),
                "client_order_id": payload.get("client_order_id"),
                "order_id": payload.get("order_id"),
                "local_state": payload.get("local_state"),
                "exchange_state": payload.get("exchange_state"),
                "local_filled": payload.get("local_filled"),
                "exchange_filled": payload.get("exchange_filled"),
            }
        )
        dedupe_key = payload_fingerprint(run_id, reason, details)
        with self._lock:
            self._repo.upsert_risk_event(
                run_id=run_id,
                reason=reason,
                details=details,
                dedupe_key=dedupe_key,
            )

    def close(self) -> None:
        return
