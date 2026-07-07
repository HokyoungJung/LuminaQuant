# TradFi data lane coverage summary

- Generated UTC: `2026-07-07T11:36:11.441774Z`
- Command log: `var/reports/strategy_performance_improvement_20260707/data_tradfi_lane/command_log.md`
- Verification log: `var/reports/strategy_performance_improvement_20260707/data_tradfi_lane/verification_log.md`
- Collector report: `var/reports/strategy_performance_improvement_20260707/data_tradfi_lane/binance_1m_research_universe_collection_latest.json`
- Command mode: `dry_run_discovery_no_data_write`

## Coverage counts
- `collector_error_count`: `0`
- `collector_request_count`: `0`
- `collector_up_to_date_count`: `128`
- `core_crypto_snapshot_count`: `10`
- `discovered_tradfi_trading_count`: `118`
- `fetched_rows`: `0`
- `new_tradfi_since_static_snapshot_count`: `18`
- `selected_symbol_count`: `128`
- `static_tradfi_snapshot_count`: `100`
- `upserted_rows`: `0`

## Newly discovered TradFi symbols since static snapshot
`ALABUSDT`, `BSPUSDT`, `CATUSDT`, `CIENUSDT`, `FLEXUSDT`, `KLACUSDT`, `KORUUSDT`, `KSTRUSDT`, `LRCXUSDT`, `MVLLUSDT`, `SMCIUSDT`, `SONYUSDT`, `SQQQUSDT`, `STRCUSDT`, `TERUSDT`, `TQQQUSDT`, `TTWOUSDT`, `TXNUSDT`

## Full-universe claim status
- Claim: `not_claimed_full_data_refresh`
- Reason: Task 1 validated supported dynamic universe discovery with a no-write dry-run; full historical/current data refresh was not run because it is an external/heavy data job and would need staged resource/API handling.
- Safe next command: `uv run python scripts/collect_binance_1m_research_universe.py --source data-vision --universe-source static-plus-fapi-tradfi --db-path data/market_parquet --exchange binance --since 2025-01-01T00:00:00Z --workers 4 --global-request-interval-sec 1.0`

## Compaction status
- Command: `uv run python scripts/compact_wal_to_monthly_parquet.py --root-path data/market_parquet --exchange binance`
- Exit status: `0`
- Result: No symbols found for compaction; no WAL compaction performed in this worktree.

## Subagent evidence
- Subagents spawned: `2`
- Subagent model: `gpt-5.4-mini`
- Serial repo searches before spawn: `0`
- Findings file: `var/reports/strategy_performance_improvement_20260707/data_tradfi_lane/subagent_findings_task1.md`

## Known test gaps
- No direct regression test for resolve_default_symbols(..., universe_source="fapi-tradfi").
- No direct regression test for universe_discovery_payload(..., explicit_symbols=True).
- Collector main/report aggregation is not unit-tested with monkeypatched exchangeInfo/download/upsert hooks.
- Coverage-inventory JSON/markdown artifact shape is not directly covered by a dedicated test file.

## Notes
- Static research_universe snapshot was left unchanged for reproducibility.
- Collector default static-plus-fapi-tradfi selected the frozen snapshot plus current Binance TRADIFI_PERPETUAL/USDT discovery.
- Dry-run used an end-before-start window so no market-data downloads or writes were performed; exchangeInfo discovery was still validated.
