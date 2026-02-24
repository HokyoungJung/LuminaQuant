from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from lumina_quant.parquet_market_data import ParquetMarketDataRepository
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


def _sample_1s_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime": [
                "2026-01-05T00:00:00Z",
                "2026-01-05T00:00:01Z",
                "2026-01-05T00:00:59Z",
                "2026-01-05T00:01:00Z",
            ],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_parquet_and_postgres_state_smoke_flow(tmp_path: Path):
    parquet_root = tmp_path / "market"
    parquet_repo = ParquetMarketDataRepository(parquet_root)

    rows_written = parquet_repo.upsert_1s(exchange="binance", symbol="BTC/USDT", rows=_sample_1s_frame())
    assert rows_written == 4

    minute_frame = parquet_repo.load_ohlcv(exchange="binance", symbol="BTC/USDT", timeframe="1m")
    assert minute_frame.height == 2
    assert minute_frame["open"].to_list() == [100.0, 103.0]

    conn = FakeConnection()
    state_repo = PostgresStateRepository(connection=conn)
    state_repo.initialize_schema()

    run_id = state_repo.start_run(mode="backtest", metadata={"symbol": "BTC/USDT"}, run_id="run-smoke")
    now = datetime(2026, 1, 5, tzinfo=UTC)

    state_repo.upsert_equity(run_id=run_id, timeindex=now, total=10000.0, cash=9800.0)
    state_repo.upsert_order(
        run_id=run_id,
        created_at=now,
        symbol="BTC/USDT",
        side="BUY",
        order_type="LIMIT",
        quantity=1.0,
        price=100.0,
        status="NEW",
        client_order_id="order-1",
    )
    state_repo.upsert_fill(
        run_id=run_id,
        fill_time=now,
        symbol="BTC/USDT",
        side="BUY",
        quantity=1.0,
        fill_price=100.0,
        status="FILLED",
        dedupe_key="fill-1",
    )
    state_repo.upsert_workflow_job(
        job_id="job-1",
        workflow="backtest",
        status="RUNNING",
        run_id=run_id,
        started_at=now,
    )

    all_queries = "\n".join(query for query, _ in conn.cursor_obj.queries)
    assert "CREATE TABLE IF NOT EXISTS runs" in all_queries
    assert "ON CONFLICT (run_id, timeindex)" in all_queries
    assert "ON CONFLICT (run_id, client_order_id)" in all_queries
    assert "ON CONFLICT (run_id, dedupe_key)" in all_queries
    assert "ON CONFLICT (job_id)" in all_queries
    assert conn.commit_count >= 6
