# External Data Contracts

LuminaQuant supports user-managed external data in both backtest and live workflows.

## Backtest external data

Use canonical OHLCV with columns:
- `datetime`
- `open`
- `high`
- `low`
- `close`
- `volume`

Supported forms:
- CSV
- parquet
- in-memory/test fixture payloads

### CLI
```bash
uv run lq backtest --data-source external --external-data-root var/data/external/backtest
uv run lq optimize --data-source external --external-data-root var/data/external/backtest
```

### Config
```yaml
backtest:
  data_source: external
  external:
    source_kind: csv
    root_path: var/data/external/backtest
    symbol_map:
      BTC/USDT: BTCUSDT.csv
```

### File naming
For a symbol like `BTC/USDT`, the loader accepts common forms such as:
- `BTCUSDT.csv`
- `BTC_USDT.csv`
- `BTC-USDT.csv`
- `BTCUSDT.parquet`
- `BTC_USDT.parquet`
- `BTC-USDT.parquet`

Explicit `external` mode is **fail-fast**: if required data is missing, LuminaQuant raises an error instead of silently falling back.

## Live external data

Use:
```yaml
live:
  market_data_source: external
  external:
    source_kind: jsonl   # jsonl | parquet | pipe
    path: var/data/external/live_windows.jsonl
    schema: market_window_v1
    poll_seconds: 2
    allow_stale_seconds: 45
```

### `market_window_v1` JSONL
One JSON object per line:

```json
{
  "time": 1773396000000,
  "window_seconds": 20,
  "bars_1s": {
    "BTC/USDT": [
      [1773395980000, 67000.0, 67010.0, 66990.0, 67005.0, 12.3]
    ]
  },
  "event_time_watermark_ms": 1773396000000,
  "commit_id": "external-feed-001",
  "lag_ms": 250,
  "is_stale": false
}
```

### `ohlcv_1s_v1` parquet
Canonical 1-second OHLCV parquet with the same six OHLCV columns. The live external handler turns those rows into the `MARKET_WINDOW` contract internally.

## Strategy contract

Strategies should consume canonical contracts, not vendor-specific raw payloads.

Current additive strategy declarations:
- `required_inputs`
- `required_features`
- `preferred_contract`
- optional `calculate_signals_context(context)`

Legacy `calculate_signals` and `calculate_signals_window` remain supported.

## Fail-fast semantics
- Explicit external modes never silently fall back.
- Missing required strategy inputs/features are runtime errors in live and explicit external backtest modes.
