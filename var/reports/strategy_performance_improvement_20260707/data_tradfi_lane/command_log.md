## Task 1 command log

### uv run python scripts/collect_binance_1m_research_universe.py --help

```text
usage: collect_binance_1m_research_universe.py [-h]
                                               [--source {data-vision,fapi}]
                                               [--db-path DB_PATH]
                                               [--exchange EXCHANGE]
                                               [--since SINCE] [--until UNTIL]
                                               [--symbols [SYMBOLS ...]]
                                               [--universe-source {static,fapi-tradfi,static-plus-fapi-tradfi}]
                                               [--workers WORKERS]
                                               [--limit LIMIT]
                                               [--retries RETRIES]
                                               [--base-wait-sec BASE_WAIT_SEC]
                                               [--request-sleep-sec REQUEST_SLEEP_SEC]
                                               [--global-request-interval-sec GLOBAL_REQUEST_INTERVAL_SEC]
                                               [--no-resume] [--dry-run]
                                               [--report-dir REPORT_DIR]

Collect Binance USD-M Futures 1m klines for the extended research universe.
This is a direct 1m OHLCV collector for broad research/shadow scans. It does
not collect raw aggTrades, does not derive 1s bars, and does not place orders.
```

### uv run python scripts/collect_binance_1m_research_universe.py --source fapi --since 2026-07-07T00:00:00Z --until 2026-07-06T00:00:00Z --workers 1 --dry-run --report-dir var/reports/strategy_performance_improvement_20260707/data_tradfi_lane

```json
{
  "dry_run": true,
  "event": "start",
  "new_tradfi_since_static_snapshot_count": 18,
  "since_utc": "2026-07-07T00:00:00Z",
  "source": "fapi",
  "symbol_count": 128,
  "universe_source": "static-plus-fapi-tradfi",
  "until_utc": "2026-07-06T00:00:59.999000Z",
  "workers": 1
}
{
  "event": "report",
  "latest": "var/reports/strategy_performance_improvement_20260707/data_tradfi_lane/binance_1m_research_universe_collection_latest.json",
  "summary": {
    "empty_count": 0,
    "error_count": 0,
    "fetched_rows": 0,
    "missing_files": 0,
    "ok_count": 0,
    "request_count": 0,
    "source_files": 0,
    "up_to_date_count": 128,
    "upserted_rows": 0
  }
}
```

### uv run python scripts/compact_wal_to_monthly_parquet.py --root-path data/market_parquet --exchange binance

```text
No symbols found for compaction.
exit_status: 0
```

### Summary artifacts

- Coverage summary: `var/reports/strategy_performance_improvement_20260707/data_tradfi_lane/tradfi_data_coverage_summary_latest.json`
- Verification log: `var/reports/strategy_performance_improvement_20260707/data_tradfi_lane/verification_log.md`
- Subagent findings: `var/reports/strategy_performance_improvement_20260707/data_tradfi_lane/subagent_findings_task1.md`
- Full-universe claim status: `not_claimed_full_data_refresh`
