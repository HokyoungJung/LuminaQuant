"""Risk/health payload helpers for the Next dashboard migration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from lumina_quant.dashboard.overview_service import resolve_dashboard_postgres_dsn
from lumina_quant.postgres_state import _connect_postgres


def empty_risk_health_payload(*, reason: str) -> dict[str, Any]:
    return {
        "as_of": datetime.now(UTC).isoformat(),
        "run_id": "",
        "summary": {
            "risk_event_count": 0,
            "heartbeat_count": 0,
            "order_state_count": 0,
        },
        "risk_events": [],
        "heartbeats": [],
        "order_states": [],
        "status": reason,
    }


def load_risk_health_payload(*, dsn: str | None = None, limit: int = 25) -> dict[str, Any]:
    if dsn is not None and not str(dsn).strip():
        return empty_risk_health_payload(reason="missing_dsn")
    resolved_dsn = resolve_dashboard_postgres_dsn(dsn)
    if not resolved_dsn:
        return empty_risk_health_payload(reason="missing_dsn")

    conn = _connect_postgres(resolved_dsn)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT run_id
                FROM runs
                ORDER BY started_at DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
        if row is None:
            return empty_risk_health_payload(reason="no_runs")
        run_id = str(row[0] or "")

        def _load(query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description or ()]
            frame = pd.DataFrame(rows, columns=columns)
            if frame.empty:
                return []
            records: list[dict[str, Any]] = []
            for _, item in frame.iterrows():
                record: dict[str, Any] = {}
                for key, value in item.items():
                    if pd.isna(value):
                        record[str(key)] = None
                    elif "time" in str(key):
                        record[str(key)] = pd.to_datetime(
                            value, errors="coerce", utc=True
                        ).isoformat()
                    else:
                        record[str(key)] = value
                records.append(record)
            return records

        risk_events = _load(
            """
            SELECT event_time, reason
            FROM risk_events
            WHERE run_id = %s
            ORDER BY event_time DESC
            LIMIT %s
            """,
            (run_id, int(max(1, limit))),
        )
        heartbeats = _load(
            """
            SELECT heartbeat_time, status
            FROM heartbeats
            WHERE run_id = %s
            ORDER BY heartbeat_time DESC
            LIMIT %s
            """,
            (run_id, int(max(1, limit))),
        )
        order_states = _load(
            """
            SELECT event_time, symbol, state, message
            FROM order_state_events
            WHERE run_id = %s
            ORDER BY event_time DESC
            LIMIT %s
            """,
            (run_id, int(max(1, limit))),
        )

        return {
            "as_of": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "summary": {
                "risk_event_count": len(risk_events),
                "heartbeat_count": len(heartbeats),
                "order_state_count": len(order_states),
            },
            "risk_events": risk_events,
            "heartbeats": heartbeats,
            "order_states": order_states,
            "status": "ok",
        }
    finally:
        conn.close()


__all__ = ["empty_risk_health_payload", "load_risk_health_payload"]
