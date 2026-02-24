"""PostgreSQL-backed runtime state repository with idempotent writes."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

SCHEMA_SQL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        mode TEXT NOT NULL,
        started_at TIMESTAMPTZ NOT NULL,
        ended_at TIMESTAMPTZ,
        status TEXT,
        metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        updated_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS equity (
        id BIGSERIAL PRIMARY KEY,
        run_id TEXT NOT NULL,
        timeindex TIMESTAMPTZ NOT NULL,
        total DOUBLE PRECISION NOT NULL,
        cash DOUBLE PRECISION,
        metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        UNIQUE (run_id, timeindex)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS orders (
        id BIGSERIAL PRIMARY KEY,
        run_id TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        order_type TEXT NOT NULL,
        quantity DOUBLE PRECISION NOT NULL,
        price DOUBLE PRECISION,
        status TEXT,
        client_order_id TEXT NOT NULL,
        exchange_order_id TEXT,
        metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        UNIQUE (run_id, client_order_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fills (
        id BIGSERIAL PRIMARY KEY,
        run_id TEXT NOT NULL,
        dedupe_key TEXT NOT NULL,
        fill_time TIMESTAMPTZ NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        quantity DOUBLE PRECISION NOT NULL,
        fill_price DOUBLE PRECISION,
        fill_cost DOUBLE PRECISION,
        commission DOUBLE PRECISION,
        client_order_id TEXT,
        exchange_order_id TEXT,
        status TEXT,
        metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        UNIQUE (run_id, dedupe_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS positions (
        id BIGSERIAL PRIMARY KEY,
        run_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        position_side TEXT NOT NULL,
        quantity DOUBLE PRECISION NOT NULL,
        entry_price DOUBLE PRECISION,
        mark_price DOUBLE PRECISION,
        unrealized_pnl DOUBLE PRECISION,
        updated_at TIMESTAMPTZ NOT NULL,
        metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        UNIQUE (run_id, symbol, position_side)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS risk_events (
        id BIGSERIAL PRIMARY KEY,
        run_id TEXT NOT NULL,
        dedupe_key TEXT NOT NULL,
        event_time TIMESTAMPTZ NOT NULL,
        reason TEXT NOT NULL,
        details_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        UNIQUE (run_id, dedupe_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS heartbeats (
        id BIGSERIAL PRIMARY KEY,
        run_id TEXT NOT NULL,
        dedupe_key TEXT NOT NULL,
        worker_id TEXT,
        heartbeat_time TIMESTAMPTZ NOT NULL,
        status TEXT,
        details_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        UNIQUE (run_id, dedupe_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS order_state_events (
        id BIGSERIAL PRIMARY KEY,
        run_id TEXT NOT NULL,
        dedupe_key TEXT NOT NULL,
        event_time TIMESTAMPTZ NOT NULL,
        symbol TEXT,
        client_order_id TEXT,
        exchange_order_id TEXT,
        state TEXT NOT NULL,
        message TEXT,
        details_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        UNIQUE (run_id, dedupe_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS optimization_results (
        id BIGSERIAL PRIMARY KEY,
        run_id TEXT NOT NULL,
        stage TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        params_json JSONB NOT NULL,
        sharpe DOUBLE PRECISION,
        cagr TEXT,
        mdd TEXT,
        train_sharpe DOUBLE PRECISION,
        robustness_score DOUBLE PRECISION,
        extra_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        UNIQUE (run_id, stage, fingerprint)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS workflow_jobs (
        job_id TEXT PRIMARY KEY,
        workflow TEXT NOT NULL,
        status TEXT NOT NULL,
        requested_mode TEXT,
        strategy TEXT,
        command_json JSONB,
        env_json JSONB,
        pid BIGINT,
        run_id TEXT,
        started_at TIMESTAMPTZ,
        ended_at TIMESTAMPTZ,
        exit_code INTEGER,
        log_path TEXT,
        stop_file TEXT,
        metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        last_updated TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_equity_run_timeindex ON equity(run_id, timeindex DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_orders_run_created_at ON orders(run_id, created_at DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_fills_run_fill_time ON fills(run_id, fill_time DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_risk_events_run_event_time ON risk_events(run_id, event_time DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_heartbeats_run_time ON heartbeats(run_id, heartbeat_time DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_order_state_events_run_time
    ON order_state_events(run_id, event_time DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_optimization_results_run_stage
    ON optimization_results(run_id, stage, created_at DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_workflow_jobs_status_started
    ON workflow_jobs(status, started_at DESC)
    """,
)


class CursorLike(Protocol):
    """Structural protocol for DB cursor objects."""

    def execute(self, query: str, params: tuple[Any, ...] | None = None) -> Any:
        ...


class ConnectionLike(Protocol):
    """Structural protocol for DB connection objects."""

    def cursor(self) -> Any:
        ...

    def commit(self) -> None:
        ...

    def close(self) -> None:
        ...


def _ensure_utc(value: datetime | str | None, *, fallback_now: bool) -> datetime:
    if value is None:
        if not fallback_now:
            raise ValueError("Datetime value is required.")
        return datetime.now(tz=UTC)
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            if fallback_now:
                return datetime.now(tz=UTC)
            raise ValueError("Datetime string cannot be empty.")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return _ensure_utc(value, fallback_now=False).isoformat()
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    return value


def canonical_json_dumps(payload: Any) -> str:
    """Serialize JSON payload using deterministic key ordering."""
    normalized = _normalize_json_value(payload if payload is not None else {})
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)


def payload_fingerprint(*parts: Any) -> str:
    """Return deterministic hash token used for idempotent dedupe keys."""
    material = canonical_json_dumps({"parts": list(parts)})
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class PostgresStateRepository:
    """Postgres runtime repository for run/audit/workflow state."""

    dsn: str | None = None
    connection_factory: Callable[[], ConnectionLike] | None = None
    connection: ConnectionLike | None = None

    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(tz=UTC)

    @contextmanager
    def _open_connection(self) -> Iterator[ConnectionLike]:
        if self.connection is not None:
            yield self.connection
            return

        if self.connection_factory is not None:
            conn = self.connection_factory()
            try:
                yield conn
            finally:
                conn.close()
            return

        dsn = str(self.dsn or "").strip()
        if not dsn:
            raise ValueError("Postgres DSN is required when no explicit connection is provided.")

        conn = _connect_postgres(dsn)
        try:
            yield conn
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        """Create required schema objects if they do not already exist."""
        with self._open_connection() as conn:
            with conn.cursor() as cursor:
                for statement in SCHEMA_SQL:
                    cursor.execute(statement)
            conn.commit()

    def upsert_run(
        self,
        *,
        run_id: str,
        mode: str,
        status: str,
        metadata: dict[str, Any] | None = None,
        started_at: datetime | str | None = None,
        ended_at: datetime | str | None = None,
        updated_at: datetime | str | None = None,
    ) -> None:
        """Insert/update one run row keyed by run_id."""
        started = _ensure_utc(started_at, fallback_now=True)
        ended = _ensure_utc(ended_at, fallback_now=False) if ended_at is not None else None
        updated = _ensure_utc(updated_at, fallback_now=True)
        metadata_json = canonical_json_dumps(metadata or {})

        sql = """
            INSERT INTO runs(run_id, mode, started_at, ended_at, status, metadata_json, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (run_id)
            DO UPDATE SET
                mode = EXCLUDED.mode,
                started_at = LEAST(runs.started_at, EXCLUDED.started_at),
                ended_at = COALESCE(EXCLUDED.ended_at, runs.ended_at),
                status = EXCLUDED.status,
                metadata_json = COALESCE(runs.metadata_json, '{}'::jsonb) || EXCLUDED.metadata_json,
                updated_at = EXCLUDED.updated_at
        """
        with self._open_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, (run_id, mode, started, ended, status, metadata_json, updated))
            conn.commit()

    def start_run(self, mode: str, metadata: dict[str, Any] | None = None, run_id: str | None = None) -> str:
        """Create or refresh a RUNNING run row and return run_id."""
        resolved_run_id = str(run_id or uuid.uuid4())
        self.upsert_run(
            run_id=resolved_run_id,
            mode=mode,
            status="RUNNING",
            metadata=metadata or {},
            started_at=self._now_utc(),
            updated_at=self._now_utc(),
        )
        return resolved_run_id

    def end_run(
        self,
        run_id: str,
        *,
        status: str = "COMPLETED",
        metadata: dict[str, Any] | None = None,
        ended_at: datetime | str | None = None,
    ) -> None:
        """Mark a run as finished while preserving existing mode/start values."""
        ended = _ensure_utc(ended_at, fallback_now=True)
        updated = self._now_utc()
        metadata_json = canonical_json_dumps(metadata or {})
        sql = """
            INSERT INTO runs(run_id, mode, started_at, ended_at, status, metadata_json, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (run_id)
            DO UPDATE SET
                ended_at = COALESCE(EXCLUDED.ended_at, runs.ended_at),
                status = EXCLUDED.status,
                metadata_json = COALESCE(runs.metadata_json, '{}'::jsonb) || EXCLUDED.metadata_json,
                updated_at = EXCLUDED.updated_at
        """
        with self._open_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    (
                        run_id,
                        "unknown",
                        ended,
                        ended,
                        status,
                        metadata_json,
                        updated,
                    ),
                )
            conn.commit()

    def upsert_equity(
        self,
        *,
        run_id: str,
        timeindex: datetime | str,
        total: float,
        cash: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert/update one equity point keyed by (run_id, timeindex)."""
        sql = """
            INSERT INTO equity(run_id, timeindex, total, cash, metadata_json)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (run_id, timeindex)
            DO UPDATE SET
                total = EXCLUDED.total,
                cash = EXCLUDED.cash,
                metadata_json = EXCLUDED.metadata_json
        """
        with self._open_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    (
                        run_id,
                        _ensure_utc(timeindex, fallback_now=False),
                        float(total),
                        float(cash) if cash is not None else None,
                        canonical_json_dumps(metadata or {}),
                    ),
                )
            conn.commit()

    def upsert_order(
        self,
        *,
        run_id: str,
        created_at: datetime | str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
        status: str | None = None,
        client_order_id: str | None = None,
        exchange_order_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Insert/update one order row keyed by (run_id, client_order_id)."""
        resolved_client_order_id = str(client_order_id or "").strip() or payload_fingerprint(
            run_id,
            symbol,
            side,
            order_type,
            float(quantity),
            price,
            exchange_order_id,
            _ensure_utc(created_at, fallback_now=False).isoformat(),
        )
        sql = """
            INSERT INTO orders(
                run_id, created_at, symbol, side, order_type, quantity, price,
                status, client_order_id, exchange_order_id, metadata_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (run_id, client_order_id)
            DO UPDATE SET
                created_at = EXCLUDED.created_at,
                symbol = EXCLUDED.symbol,
                side = EXCLUDED.side,
                order_type = EXCLUDED.order_type,
                quantity = EXCLUDED.quantity,
                price = EXCLUDED.price,
                status = EXCLUDED.status,
                exchange_order_id = COALESCE(EXCLUDED.exchange_order_id, orders.exchange_order_id),
                metadata_json = EXCLUDED.metadata_json
        """
        with self._open_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    (
                        run_id,
                        _ensure_utc(created_at, fallback_now=False),
                        str(symbol),
                        str(side),
                        str(order_type),
                        float(quantity),
                        float(price) if price is not None else None,
                        str(status) if status is not None else None,
                        resolved_client_order_id,
                        str(exchange_order_id) if exchange_order_id is not None else None,
                        canonical_json_dumps(metadata or {}),
                    ),
                )
            conn.commit()
        return resolved_client_order_id

    def upsert_fill(
        self,
        *,
        run_id: str,
        fill_time: datetime | str,
        symbol: str,
        side: str,
        quantity: float,
        fill_price: float | None = None,
        fill_cost: float | None = None,
        commission: float | None = None,
        client_order_id: str | None = None,
        exchange_order_id: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
    ) -> str:
        """Insert/update one fill row keyed by (run_id, dedupe_key)."""
        fill_time_utc = _ensure_utc(fill_time, fallback_now=False)
        resolved_key = str(dedupe_key or "").strip() or payload_fingerprint(
            fill_time_utc.isoformat(),
            symbol,
            side,
            float(quantity),
            fill_price,
            fill_cost,
            commission,
            client_order_id,
            exchange_order_id,
            status,
            metadata or {},
        )
        sql = """
            INSERT INTO fills(
                run_id, dedupe_key, fill_time, symbol, side, quantity,
                fill_price, fill_cost, commission, client_order_id,
                exchange_order_id, status, metadata_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (run_id, dedupe_key)
            DO UPDATE SET
                fill_time = EXCLUDED.fill_time,
                symbol = EXCLUDED.symbol,
                side = EXCLUDED.side,
                quantity = EXCLUDED.quantity,
                fill_price = EXCLUDED.fill_price,
                fill_cost = EXCLUDED.fill_cost,
                commission = EXCLUDED.commission,
                client_order_id = EXCLUDED.client_order_id,
                exchange_order_id = EXCLUDED.exchange_order_id,
                status = EXCLUDED.status,
                metadata_json = EXCLUDED.metadata_json
        """
        with self._open_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    (
                        run_id,
                        resolved_key,
                        fill_time_utc,
                        str(symbol),
                        str(side),
                        float(quantity),
                        float(fill_price) if fill_price is not None else None,
                        float(fill_cost) if fill_cost is not None else None,
                        float(commission) if commission is not None else None,
                        str(client_order_id) if client_order_id is not None else None,
                        str(exchange_order_id) if exchange_order_id is not None else None,
                        str(status) if status is not None else None,
                        canonical_json_dumps(metadata or {}),
                    ),
                )
            conn.commit()
        return resolved_key

    def upsert_position(
        self,
        *,
        run_id: str,
        symbol: str,
        position_side: str,
        quantity: float,
        entry_price: float | None = None,
        mark_price: float | None = None,
        unrealized_pnl: float | None = None,
        updated_at: datetime | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert/update one position row keyed by (run_id, symbol, position_side)."""
        sql = """
            INSERT INTO positions(
                run_id, symbol, position_side, quantity, entry_price,
                mark_price, unrealized_pnl, updated_at, metadata_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (run_id, symbol, position_side)
            DO UPDATE SET
                quantity = EXCLUDED.quantity,
                entry_price = EXCLUDED.entry_price,
                mark_price = EXCLUDED.mark_price,
                unrealized_pnl = EXCLUDED.unrealized_pnl,
                updated_at = EXCLUDED.updated_at,
                metadata_json = EXCLUDED.metadata_json
        """
        with self._open_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    (
                        run_id,
                        str(symbol),
                        str(position_side),
                        float(quantity),
                        float(entry_price) if entry_price is not None else None,
                        float(mark_price) if mark_price is not None else None,
                        float(unrealized_pnl) if unrealized_pnl is not None else None,
                        _ensure_utc(updated_at, fallback_now=True),
                        canonical_json_dumps(metadata or {}),
                    ),
                )
            conn.commit()

    def upsert_risk_event(
        self,
        *,
        run_id: str,
        event_time: datetime | str | None = None,
        reason: str,
        details: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
    ) -> str:
        """Insert/update one risk event row keyed by (run_id, dedupe_key)."""
        event_time_utc = _ensure_utc(event_time, fallback_now=True)
        resolved_key = str(dedupe_key or "").strip() or payload_fingerprint(
            event_time_utc.isoformat(), reason, details or {}
        )
        sql = """
            INSERT INTO risk_events(run_id, dedupe_key, event_time, reason, details_json)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (run_id, dedupe_key)
            DO UPDATE SET
                event_time = EXCLUDED.event_time,
                reason = EXCLUDED.reason,
                details_json = EXCLUDED.details_json
        """
        with self._open_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    (
                        run_id,
                        resolved_key,
                        event_time_utc,
                        str(reason),
                        canonical_json_dumps(details or {}),
                    ),
                )
            conn.commit()
        return resolved_key

    def upsert_heartbeat(
        self,
        *,
        run_id: str,
        heartbeat_time: datetime | str | None = None,
        status: str | None = None,
        worker_id: str | None = None,
        details: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
    ) -> str:
        """Insert/update one heartbeat row keyed by (run_id, dedupe_key)."""
        heartbeat_time_utc = _ensure_utc(heartbeat_time, fallback_now=True)
        resolved_key = str(dedupe_key or "").strip() or payload_fingerprint(
            heartbeat_time_utc.isoformat(), worker_id, status, details or {}
        )
        sql = """
            INSERT INTO heartbeats(
                run_id, dedupe_key, worker_id, heartbeat_time, status, details_json
            ) VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (run_id, dedupe_key)
            DO UPDATE SET
                worker_id = EXCLUDED.worker_id,
                heartbeat_time = EXCLUDED.heartbeat_time,
                status = EXCLUDED.status,
                details_json = EXCLUDED.details_json
        """
        with self._open_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    (
                        run_id,
                        resolved_key,
                        str(worker_id) if worker_id is not None else None,
                        heartbeat_time_utc,
                        str(status) if status is not None else None,
                        canonical_json_dumps(details or {}),
                    ),
                )
            conn.commit()
        return resolved_key

    def upsert_order_state_event(
        self,
        *,
        run_id: str,
        event_time: datetime | str | None = None,
        state: str,
        symbol: str | None = None,
        client_order_id: str | None = None,
        exchange_order_id: str | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
    ) -> str:
        """Insert/update one order-state event row keyed by (run_id, dedupe_key)."""
        event_time_utc = _ensure_utc(event_time, fallback_now=True)
        resolved_key = str(dedupe_key or "").strip() or payload_fingerprint(
            event_time_utc.isoformat(),
            state,
            symbol,
            client_order_id,
            exchange_order_id,
            message,
            details or {},
        )
        insert_sql = """
            INSERT INTO order_state_events(
                run_id, dedupe_key, event_time, symbol, client_order_id,
                exchange_order_id, state, message, details_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (run_id, dedupe_key)
            DO UPDATE SET
                event_time = EXCLUDED.event_time,
                symbol = EXCLUDED.symbol,
                client_order_id = EXCLUDED.client_order_id,
                exchange_order_id = EXCLUDED.exchange_order_id,
                state = EXCLUDED.state,
                message = EXCLUDED.message,
                details_json = EXCLUDED.details_json
        """
        order_update_sql = """
            UPDATE orders
            SET status = %s,
                exchange_order_id = COALESCE(%s, exchange_order_id),
                metadata_json = COALESCE(metadata_json, '{}'::jsonb) || %s::jsonb
            WHERE run_id = %s AND client_order_id = %s
        """
        details_payload = dict(details or {})
        if message and "message" not in details_payload:
            details_payload["message"] = message

        with self._open_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    insert_sql,
                    (
                        run_id,
                        resolved_key,
                        event_time_utc,
                        str(symbol) if symbol is not None else None,
                        str(client_order_id) if client_order_id is not None else None,
                        str(exchange_order_id) if exchange_order_id is not None else None,
                        str(state),
                        str(message) if message is not None else None,
                        canonical_json_dumps(details_payload),
                    ),
                )
                if client_order_id:
                    cursor.execute(
                        order_update_sql,
                        (
                            str(state),
                            str(exchange_order_id) if exchange_order_id is not None else None,
                            canonical_json_dumps(details_payload),
                            run_id,
                            str(client_order_id),
                        ),
                    )
            conn.commit()
        return resolved_key

    def upsert_optimization_result(
        self,
        *,
        run_id: str,
        stage: str,
        params: dict[str, Any],
        created_at: datetime | str | None = None,
        sharpe: float | None = None,
        cagr: str | None = None,
        mdd: str | None = None,
        train_sharpe: float | None = None,
        robustness_score: float | None = None,
        extra: dict[str, Any] | None = None,
        fingerprint: str | None = None,
    ) -> str:
        """Insert/update one optimization row keyed by (run_id, stage, fingerprint)."""
        params_payload = dict(params or {})
        extra_payload = dict(extra or {})
        resolved_fingerprint = str(fingerprint or "").strip() or payload_fingerprint(
            params_payload,
            sharpe,
            cagr,
            mdd,
            train_sharpe,
            robustness_score,
            extra_payload,
        )
        sql = """
            INSERT INTO optimization_results(
                run_id, stage, fingerprint, created_at, params_json,
                sharpe, cagr, mdd, train_sharpe, robustness_score, extra_json
            ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (run_id, stage, fingerprint)
            DO UPDATE SET
                created_at = EXCLUDED.created_at,
                params_json = EXCLUDED.params_json,
                sharpe = EXCLUDED.sharpe,
                cagr = EXCLUDED.cagr,
                mdd = EXCLUDED.mdd,
                train_sharpe = EXCLUDED.train_sharpe,
                robustness_score = EXCLUDED.robustness_score,
                extra_json = EXCLUDED.extra_json
        """
        with self._open_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    (
                        run_id,
                        str(stage),
                        resolved_fingerprint,
                        _ensure_utc(created_at, fallback_now=True),
                        canonical_json_dumps(params_payload),
                        float(sharpe) if sharpe is not None else None,
                        str(cagr) if cagr is not None else None,
                        str(mdd) if mdd is not None else None,
                        float(train_sharpe) if train_sharpe is not None else None,
                        float(robustness_score) if robustness_score is not None else None,
                        canonical_json_dumps(extra_payload),
                    ),
                )
            conn.commit()
        return resolved_fingerprint

    def upsert_optimization_rows(
        self, run_id: str, stage: str, rows: list[dict[str, Any]]
    ) -> list[str]:
        """Persist optimization rows and return deterministic fingerprints."""
        fingerprints: list[str] = []
        for row in rows:
            payload = dict(row)
            params = dict(payload.pop("params", {}) or {})
            sharpe = payload.pop("sharpe", None)
            cagr = payload.pop("cagr", None)
            mdd = payload.pop("mdd", None)
            train_sharpe = payload.pop("train_sharpe", None)
            robustness_score = payload.pop("robustness_score", None)
            created_at = payload.pop("created_at", None)
            fingerprints.append(
                self.upsert_optimization_result(
                    run_id=run_id,
                    stage=stage,
                    params=params,
                    created_at=created_at,
                    sharpe=sharpe,
                    cagr=cagr,
                    mdd=mdd,
                    train_sharpe=train_sharpe,
                    robustness_score=robustness_score,
                    extra=payload,
                )
            )
        return fingerprints

    def upsert_workflow_job(
        self,
        *,
        job_id: str,
        workflow: str,
        status: str,
        requested_mode: str | None = None,
        strategy: str | None = None,
        command: Any = None,
        env: Any = None,
        pid: int | None = None,
        run_id: str | None = None,
        started_at: datetime | str | None = None,
        ended_at: datetime | str | None = None,
        exit_code: int | None = None,
        log_path: str | None = None,
        stop_file: str | None = None,
        metadata: dict[str, Any] | None = None,
        last_updated: datetime | str | None = None,
    ) -> None:
        """Insert/update one workflow job row keyed by job_id."""
        sql = """
            INSERT INTO workflow_jobs(
                job_id, workflow, status, requested_mode, strategy,
                command_json, env_json, pid, run_id, started_at, ended_at,
                exit_code, log_path, stop_file, metadata_json, last_updated
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s::jsonb, %s::jsonb, %s, %s, %s, %s,
                %s, %s, %s, %s::jsonb, %s
            )
            ON CONFLICT (job_id)
            DO UPDATE SET
                workflow = EXCLUDED.workflow,
                status = EXCLUDED.status,
                requested_mode = EXCLUDED.requested_mode,
                strategy = EXCLUDED.strategy,
                command_json = EXCLUDED.command_json,
                env_json = EXCLUDED.env_json,
                pid = EXCLUDED.pid,
                run_id = EXCLUDED.run_id,
                started_at = EXCLUDED.started_at,
                ended_at = EXCLUDED.ended_at,
                exit_code = EXCLUDED.exit_code,
                log_path = EXCLUDED.log_path,
                stop_file = EXCLUDED.stop_file,
                metadata_json = EXCLUDED.metadata_json,
                last_updated = EXCLUDED.last_updated
        """
        with self._open_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql,
                    (
                        str(job_id),
                        str(workflow),
                        str(status),
                        str(requested_mode) if requested_mode is not None else None,
                        str(strategy) if strategy is not None else None,
                        canonical_json_dumps(command if command is not None else {}),
                        canonical_json_dumps(env if env is not None else {}),
                        int(pid) if pid is not None else None,
                        str(run_id) if run_id is not None else None,
                        _ensure_utc(started_at, fallback_now=False) if started_at is not None else None,
                        _ensure_utc(ended_at, fallback_now=False) if ended_at is not None else None,
                        int(exit_code) if exit_code is not None else None,
                        str(log_path) if log_path is not None else None,
                        str(stop_file) if stop_file is not None else None,
                        canonical_json_dumps(metadata or {}),
                        _ensure_utc(last_updated, fallback_now=True),
                    ),
                )
            conn.commit()


def _connect_postgres(dsn: str) -> ConnectionLike:
    try:
        import psycopg

        return psycopg.connect(dsn)
    except ModuleNotFoundError:
        pass

    try:
        import psycopg2

        return psycopg2.connect(dsn)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "No PostgreSQL driver found. Install psycopg>=3 or psycopg2."
        ) from exc
