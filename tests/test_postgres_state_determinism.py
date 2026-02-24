from __future__ import annotations

from dataclasses import dataclass, field

from lumina_quant.postgres_state import (
    PostgresStateRepository,
    canonical_json_dumps,
    payload_fingerprint,
)


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

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_canonical_json_and_fingerprint_are_order_invariant():
    left = {"beta": 2, "alpha": 1, "nested": {"z": 9, "a": 1}}
    right = {"nested": {"a": 1, "z": 9}, "alpha": 1, "beta": 2}

    assert canonical_json_dumps(left) == canonical_json_dumps(right)
    assert payload_fingerprint(left) == payload_fingerprint(right)


def test_risk_event_and_heartbeat_dedupe_keys_stable_for_reordered_payloads():
    conn = FakeConnection()
    repo = PostgresStateRepository(connection=conn)

    risk_a = repo.upsert_risk_event(
        run_id="run-1",
        event_time="2026-01-05T00:00:00Z",
        reason="LIMIT",
        details={"threshold": 0.3, "window": 5},
    )
    risk_b = repo.upsert_risk_event(
        run_id="run-1",
        event_time="2026-01-05T00:00:00Z",
        reason="LIMIT",
        details={"window": 5, "threshold": 0.3},
    )

    hb_a = repo.upsert_heartbeat(
        run_id="run-1",
        heartbeat_time="2026-01-05T00:00:00Z",
        worker_id="worker-a",
        status="ALIVE",
        details={"lag_ms": 3, "queue": "main"},
    )
    hb_b = repo.upsert_heartbeat(
        run_id="run-1",
        heartbeat_time="2026-01-05T00:00:00Z",
        worker_id="worker-a",
        status="ALIVE",
        details={"queue": "main", "lag_ms": 3},
    )

    assert risk_a == risk_b
    assert hb_a == hb_b
