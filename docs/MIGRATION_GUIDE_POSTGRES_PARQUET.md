# Migration Guide: Legacy Storage -> Local PostgreSQL + Parquet

This guide covers the local-only storage migration for LuminaQuant runtime workflows.

## Target Architecture

- **Market data**: partitioned Parquet files (`ParquetMarketDataRepository`)
- **Run/audit/workflow state**: PostgreSQL (`PostgresStateRepository`)
- **Analytics/resampling**: Polars weekly-chunk pipelines
- **Compute**: CPU/GPU resolution via `compute_engine`

## New Environment Variables

- `LQ_POSTGRES_DSN` (required): PostgreSQL DSN for runtime state
- `LQ_GPU_MODE` (`auto|cpu|gpu|forced-gpu`)
- `LQ_GPU_DEVICE` (optional device index, e.g. `0`, `cuda:0`)
- `LQ_GPU_VERBOSE` (`0|1`, optional)

## Postgres Schema Coverage

`PostgresStateRepository.initialize_schema()` creates idempotent tables with unique keys:

- `runs` (`run_id` PK)
- `equity` (`UNIQUE(run_id, timeindex)`)
- `orders` (`UNIQUE(run_id, client_order_id)`)
- `fills` (`UNIQUE(run_id, dedupe_key)`)
- `positions` (`UNIQUE(run_id, symbol, position_side)`)
- `risk_events` (`UNIQUE(run_id, dedupe_key)`)
- `heartbeats` (`UNIQUE(run_id, dedupe_key)`)
- `order_state_events` (`UNIQUE(run_id, dedupe_key)`)
- `optimization_results` (`UNIQUE(run_id, stage, fingerprint)`)
- `workflow_jobs` (`job_id` PK)

All write paths use `INSERT ... ON CONFLICT ... DO UPDATE` for replay-safe/idempotent behavior.

## One-Time Setup

```bash
# 1) Export DSN
export LQ_POSTGRES_DSN='postgresql://user:pass@127.0.0.1:5432/luminaquant'

# 2) Initialize schema
uv run python scripts/init_postgres_schema.py
```

Optional DDL preview:

```bash
uv run python scripts/init_postgres_schema.py --print-ddl
```

## Parquet Compaction

Compact one date partition:

```bash
uv run python scripts/compact_parquet_market_data.py \
  --root-path data/market_parquet \
  --exchange binance \
  --symbol BTC/USDT \
  --date 2026-02-01
```

Compact all partitions for a series:

```bash
uv run python scripts/compact_parquet_market_data.py \
  --root-path data/market_parquet \
  --exchange binance \
  --symbol BTC/USDT
```

## Recommended Cutover Sequence

1. Initialize PostgreSQL schema.
2. Switch state writes (runs/equity/orders/fills/etc.) to `PostgresStateRepository`.
3. Switch OHLCV reads/writes to `ParquetMarketDataRepository`.
4. Enable weekly chunked Polars resampling pipeline.
5. Remove legacy storage config usage and dead code paths.
6. Run integration + determinism tests.

## Validation Checklist

```bash
uv run pytest tests/test_integration_parquet_postgres.py tests/test_week_chunk_determinism.py
```

- Schema creation succeeds repeatedly.
- Replayed writes are idempotent (no duplicate semantic rows).
- Parquet compaction preserves latest row per timestamp.
- Weekly-chunk outputs are deterministic across repeated runs.

## Rollback Strategy

If PostgreSQL is unavailable, stop new runs and restore from the previous release build.
Do **not** dual-write to deprecated storage engines in this migration target.
