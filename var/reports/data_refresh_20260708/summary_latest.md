# Data refresh summary — 2026-07-08

- Mode: isolated worktree/branch; main/private-main was not modified by this task.
- Branch: `data-refresh-test-20260708`
- Worktree: `/home/hoky/Quants-agent/LuminaQuant-worktrees/data-refresh-test-20260708`
- Data root: `data/market_parquet_refresh_20260708` (local ignored parquet, not committed)
- Source: `Binance USD-M Futures /fapi/v1/klines`
- Timeframe: `1m`
- Total selected universe discovered: `128`
- Static TradFi snapshot: `100`
- Current discovered TradFi trading: `118`
- New TradFi since static snapshot: `18`
- Collection window: `2026-07-07T00:00:00Z` → `2026-07-08T11:20:00Z`
- Initial fetched/upserted rows: `38124` / `38124`
- Incremental fetched/upserted rows: `54` / `54`
- Combined fetched/upserted rows: `38178` / `38178`
- Errors: `0`
- Validation status: `pass`
- Validation rows: `38178` total, `2121` per symbol
- Validation totals: duplicates `0`, non-1m gaps `0`, invalid OHLCV `0`
- Targeted tests: `23 passed, 1 skipped in 0.77s`
- Compaction check: `No symbols found for compaction.`

## New symbols collected

`ALABUSDT`, `BSPUSDT`, `CATUSDT`, `CIENUSDT`, `FLEXUSDT`, `KLACUSDT`, `KORUUSDT`, `KSTRUSDT`, `LRCXUSDT`, `MVLLUSDT`, `SMCIUSDT`, `SONYUSDT`, `SQQQUSDT`, `STRCUSDT`, `TERUSDT`, `TQQQUSDT`, `TTWOUSDT`, `TXNUSDT`

## Notes

- This is an explicit latest-data update for newly discovered TradFi symbols only, not a full historical universe backfill.
- Actual parquet data remains local and ignored to avoid repo bloat and to isolate this work from concurrent sessions.
- Report JSON/Markdown/log artifacts are the only commit candidates on this separate branch.
