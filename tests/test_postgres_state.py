from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from lumina_quant.postgres_state import PostgresStateRepository


@dataclass(slots=True)
class FakeCursor:
    queries: list[tuple[str, tuple | None]] = field(default_factory=list)

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: tuple | None = None):
        self.queries.append((query, params))


@dataclass(slots=True)
class FakeConnection:
    cursor_obj: FakeCursor = field(default_factory=FakeCursor)
    commit_count: int = 0

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.commit_count += 1

    def close(self) -> None:
        return None


def test_initialize_schema_creates_required_tables():
    conn = FakeConnection()
    repo = PostgresStateRepository(connection=conn)

    repo.initialize_schema()

    queries = "\n".join(query for query, _ in conn.cursor_obj.queries)
    for table_name in (
        "runs",
        "equity",
        "orders",
        "fills",
        "positions",
        "risk_events",
        "heartbeats",
        "order_state_events",
        "optimization_results",
        "workflow_jobs",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table_name}" in queries
    assert conn.commit_count == 1


def test_upsert_sql_uses_on_conflict_for_idempotency():
    conn = FakeConnection()
    repo = PostgresStateRepository(connection=conn)
    now = datetime(2026, 1, 1, tzinfo=UTC)

    repo.upsert_run(run_id="run-1", mode="live", status="RUNNING", started_at=now)
    repo.upsert_equity(run_id="run-1", timeindex=now, total=100.0)
    repo.upsert_order(
        run_id="run-1",
        created_at=now,
        symbol="BTC/USDT",
        side="BUY",
        order_type="LIMIT",
        quantity=1.0,
        client_order_id="order-1",
    )
    repo.upsert_fill(
        run_id="run-1",
        fill_time=now,
        symbol="BTC/USDT",
        side="BUY",
        quantity=1.0,
        dedupe_key="fill-1",
    )
    repo.upsert_position(
        run_id="run-1",
        symbol="BTC/USDT",
        position_side="LONG",
        quantity=1.0,
        updated_at=now,
    )
    repo.upsert_risk_event(run_id="run-1", event_time=now, reason="LIMIT")
    repo.upsert_heartbeat(run_id="run-1", heartbeat_time=now, status="ALIVE")
    repo.upsert_order_state_event(run_id="run-1", event_time=now, state="NEW", dedupe_key="state-1")
    repo.upsert_optimization_result(
        run_id="run-1",
        stage="search",
        params={"alpha": 1.0},
        created_at=now,
    )
    repo.upsert_workflow_job(job_id="job-1", workflow="live", status="RUNNING", started_at=now)

    query_text = "\n".join(query for query, _ in conn.cursor_obj.queries)
    assert "ON CONFLICT (run_id)" in query_text
    assert "ON CONFLICT (run_id, timeindex)" in query_text
    assert "ON CONFLICT (run_id, client_order_id)" in query_text
    assert "ON CONFLICT (run_id, dedupe_key)" in query_text
    assert "ON CONFLICT (run_id, symbol, position_side)" in query_text
    assert "ON CONFLICT (run_id, stage, fingerprint)" in query_text
    assert "ON CONFLICT (job_id)" in query_text


def test_optimization_fingerprint_is_deterministic_for_key_order_variants():
    conn = FakeConnection()
    repo = PostgresStateRepository(connection=conn)
    now = datetime(2026, 1, 1, tzinfo=UTC)

    first = repo.upsert_optimization_result(
        run_id="run-1",
        stage="search",
        params={"beta": 2, "alpha": 1},
        created_at=now,
        extra={"nested": {"z": 3, "a": 1}},
    )
    second = repo.upsert_optimization_result(
        run_id="run-1",
        stage="search",
        params={"alpha": 1, "beta": 2},
        created_at=now,
        extra={"nested": {"a": 1, "z": 3}},
    )

    assert first == second
