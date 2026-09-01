# Strategy performance-improvement command log

generated_at_utc: 2026-07-07T11:34:40Z

```sh
$ uv run python scripts/collect_binance_1m_research_universe.py --help
```
```text
Using CPython 3.14.5
Creating virtual environment at: .venv
Installed 29 packages in 59ms
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

options:
  -h, --help            show this help message and exit
  --source {data-vision,fapi}
                        Historical source. data-vision avoids Binance REST
                        weight limits; fapi uses /fapi/v1/klines.
  --db-path DB_PATH
  --exchange EXCHANGE
  --since SINCE
  --until UNTIL         UTC ISO; default is latest full UTC day for data-
                        vision, latest closed minute for fapi
  --symbols [SYMBOLS ...]
  --universe-source {static,fapi-tradfi,static-plus-fapi-tradfi}
                        Default symbol discovery when --symbols is omitted.
                        static preserves the frozen static snapshot; fapi-
                        tradfi uses current exchangeInfo core+TradFi; static-
                        plus-fapi-tradfi keeps the snapshot and appends newly
                        listed TradFi perps.
  --workers WORKERS
  --limit LIMIT
  --retries RETRIES
  --base-wait-sec BASE_WAIT_SEC
  --request-sleep-sec REQUEST_SLEEP_SEC
  --global-request-interval-sec GLOBAL_REQUEST_INTERVAL_SEC
                        Minimum interval between Binance kline requests across
                        all workers.
  --no-resume
  --dry-run
  --report-dir REPORT_DIR
```

exit_code: 0

```sh
$ uv run python scripts/compact_wal_to_monthly_parquet.py --help
```
```text
usage: compact_wal_to_monthly_parquet.py [-h] [--root-path ROOT_PATH]
                                         [--exchange EXCHANGE]
                                         [--symbols SYMBOLS] [--keep-wal]

Compact WAL records into bounded monthly parquet files per symbol.

options:
  -h, --help            show this help message and exit
  --root-path ROOT_PATH
                        Market data root path (default: config market parquet
                        path).
  --exchange EXCHANGE   Exchange key (default: config market exchange).
  --symbols SYMBOLS     Comma-separated symbols (e.g. BTC/USDT,ETH/USDT). If
                        empty, auto-discover.
  --keep-wal            Do not truncate wal.bin after successful compaction
                        (advances watermark only).
```

exit_code: 0

```sh
$ uv run python scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py --help
```
```text
usage: run_alpha_zoo_69_asset_monthly_refit_walkforward.py [-h]
                                                           [--data-root DATA_ROOT]
                                                           [--symbols SYMBOLS]
                                                           [--timeframes TIMEFRAMES]
                                                           [--slippage-bps SLIPPAGE_BPS]
                                                           [--train-start TRAIN_START]
                                                           [--first-oos-start FIRST_OOS_START]
                                                           [--bar-minutes BAR_MINUTES]
                                                           [--max-folds MAX_FOLDS]
                                                           [--allocation-fraction ALLOCATION_FRACTION]
                                                           [--asset-trials ASSET_TRIALS]
                                                           [--profile-trials PROFILE_TRIALS]
                                                           [--hybrid-trials HYBRID_TRIALS]
                                                           [--source-symbol-workers SOURCE_SYMBOL_WORKERS]
                                                           [--seed SEED]
                                                           [--families FAMILIES]
                                                           [--bridge-protocol-manifest BRIDGE_PROTOCOL_MANIFEST]
                                                           [--recompute-from-json RECOMPUTE_FROM_JSON]
                                                           [--augment-row-level-selectors]
                                                           [--checkpoint-interval CHECKPOINT_INTERVAL]
                                                           [--checkpoint-markdown-interval CHECKPOINT_MARKDOWN_INTERVAL]
                                                           [--output-json OUTPUT_JSON]
                                                           [--output-md OUTPUT_MD]

Monthly refit walk-forward for top no-OOS 69-asset Alpha Zoo candidates.
Protocol: * refit date is the first day of each OOS month; * train is
expanding from ``--train-start`` to the bar immediately before the 2-month
validation window; * validation is the prior two calendar months; * locked OOS
is the next one calendar month, truncated to latest available data; *
candidate/parameter search never sees the OOS month. This runner is
deliberately report-only. It rebuilds the high no-OOS strategy families that
were discussed for live handoff: 1. per-asset/profile Optuna source profiles +
static guarded + v3.5/v3.6 hybrid; 2. individual-sleeve-first robust
portfolios, then static/v3.5/v3.6 hybrid; 3. strict live-efficiency repair
pass from the same source params; 4. relaxed MDD-guarded efficiency repair
pass from the same source params.

options:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT
  --symbols SYMBOLS
  --timeframes TIMEFRAMES
  --slippage-bps SLIPPAGE_BPS
  --train-start TRAIN_START
  --first-oos-start FIRST_OOS_START
  --bar-minutes BAR_MINUTES
  --max-folds MAX_FOLDS
  --allocation-fraction ALLOCATION_FRACTION
  --asset-trials ASSET_TRIALS
  --profile-trials PROFILE_TRIALS
  --hybrid-trials HYBRID_TRIALS
  --source-symbol-workers SOURCE_SYMBOL_WORKERS
                        Parallel workers for per-symbol source Optuna tuning.
                        Default 1 preserves the historical sequential path; >1
                        prewarms shared features and tunes symbols
                        concurrently with deterministic per-symbol seeds.
  --seed SEED
  --families FAMILIES   Comma-separated families. profile_optuna is always
                        included; optional: individual_robust,
                        asset_timeframe_leverage,
                        tradfi_us_equity_session_switch,
                        tradfi_vol_managed_v1, tradfi_momentum_regime_v1,
                        tradfi_intraday_session_v1, tradfi_overnight_split_v1,
                        factor_regime_router_v1, strict_efficiency,
                        relaxed_efficiency, teacher_leaf_blend,
                        strict_calm_leaf_selector,
                        regime_opportunity_leaf_switch,
                        lagged_shadow_leaf_router.
  --bridge-protocol-manifest BRIDGE_PROTOCOL_MANIFEST
  --recompute-from-json RECOMPUTE_FROM_JSON
                        Fast path: load an existing walk-forward JSON and
                        recompute clean/research dependency flags plus
                        aggregate reports without rerunning Optuna.
  --augment-row-level-selectors
                        Append post-OOS shadow single-leaf selectors from
                        existing fold rows. Selection uses train/validation
                        row metrics only and copies exact source OOS metrics;
                        use with --recompute-from-json for fast strategy
                        diagnostics.
  --checkpoint-interval CHECKPOINT_INTERVAL
                        Write JSON checkpoint every N folds during full
                        reruns. Use 0 to write only initial/final artifacts;
                        default preserves fold-level recovery.
  --checkpoint-markdown-interval CHECKPOINT_MARKDOWN_INTERVAL
                        Render markdown checkpoint every N folds during full
                        reruns. Default 0 skips expensive growing markdown
                        renders until final output.
  --output-json OUTPUT_JSON
  --output-md OUTPUT_MD
```

exit_code: 0

```sh
$ uv run python scripts/research/build_strategy_performance_improvement_report.py --help
```
```text
usage: build_strategy_performance_improvement_report.py [-h]
                                                        [--source-json SOURCE_JSON]
                                                        [--output-json OUTPUT_JSON]
                                                        [--output-md OUTPUT_MD]
                                                        [--command-log-path COMMAND_LOG_PATH]
                                                        [--worker-count WORKER_COUNT]
                                                        [--full-universe-claim-status FULL_UNIVERSE_CLAIM_STATUS]
                                                        [--require-source]

Build the strategy performance-improvement WF evidence envelope. The monthly-
refit walk-forward runner owns the expensive research computation. This thin
wrapper normalizes its JSON (or records a source-missing blocker) into one
review artifact with the fields needed by the 2026-07-07 team handoff.

options:
  -h, --help            show this help message and exit
  --source-json SOURCE_JSON
  --output-json OUTPUT_JSON
  --output-md OUTPUT_MD
  --command-log-path COMMAND_LOG_PATH
  --worker-count WORKER_COUNT
  --full-universe-claim-status FULL_UNIVERSE_CLAIM_STATUS
                        Override auto-derived claim status; use a
                        not_claimed_* value for blocker artifacts.
  --require-source      Exit non-zero if --source-json is missing after
                        writing the blocker envelope.
```

exit_code: 0
