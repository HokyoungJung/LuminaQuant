"""Optimization result persistence backed by PostgreSQL."""

from __future__ import annotations

import os
from typing import Any

from lumina_quant.postgres_state import PostgresStateRepository


def save_optimization_rows(
    db_path: str,
    run_id: str,
    stage: str,
    rows: list[dict[str, Any]],
) -> None:
    """Persist optimization rows into Postgres optimization_results."""
    if not rows:
        return
    dsn = str(db_path or "").strip() or str(os.getenv("LQ_POSTGRES_DSN", "")).strip()
    if not dsn:
        raise ValueError("Postgres DSN is required for optimization result persistence.")
    repo = PostgresStateRepository(dsn=dsn)
    repo.initialize_schema()
    repo.upsert_optimization_rows(run_id=str(run_id), stage=str(stage), rows=list(rows))
