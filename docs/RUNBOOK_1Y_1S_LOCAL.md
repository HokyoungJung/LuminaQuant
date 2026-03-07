# 1Y+ 1s Local Runbook (8GB RAM / 8GB VRAM)

This runbook is for **local-only uv runtime** and the current LuminaQuant stack:
- market data: monthly parquet + binary WAL
- control plane: PostgreSQL
- compute: Polars (GPU auto fallback)

---

## 0) One-time setup

```bash
cd /path/to/<REPO_DIR>
uv sync --extra optimize --extra dev --extra live
# Optional on Linux x86_64 + CUDA 12
# uv sync --extra gpu
uv run python scripts/init_postgres_schema.py --dsn "$LQ_POSTGRES_DSN"
```

`<REPO_DIR>` examples:
- `Quants-agent` (private source-of-truth)
- `LuminaQuant` (public mirror)

Use the default 12-symbol universe via env override:

```bash
export LQ__TRADING__SYMBOLS='["BTC/USDT","ETH/USDT","XRP/USDT","BNB/USDT","SOL/USDT","TRX/USDT","DOGE/USDT","ADA/USDT","TON/USDT","AVAX/USDT","XAU/USDT","XAG/USDT"]'
```

---

## 1) Backfill 1-second data (1 year+)

```bash
uv run python scripts/sync_binance_ohlcv.py \
  --symbols BTC/USDT ETH/USDT XRP/USDT BNB/USDT SOL/USDT TRX/USDT DOGE/USDT ADA/USDT TON/USDT AVAX/USDT XAU/USDT XAG/USDT \
  --timeframe 1s \
  --db-path data/market_parquet \
  --exchange-id binance \
  --market-type future \
  --since 2025-01-01T00:00:00+00:00 \
  --until 2025-12-31T23:59:59+00:00 \
  --limit 1000 \
  --max-batches 100000 \
  --retries 3 \
  --no-export-csv
```

Compact WAL into bounded monthly parquet files:

```bash
uv run python scripts/compact_wal_to_monthly_parquet.py \
  --root-path data/market_parquet \
  --exchange binance
```

Raw-first pipeline (collector -> materializer -> trader):

```bash
uv run python scripts/collect_binance_aggtrades_raw.py \
  --symbols BTC/USDT,ETH/USDT \
  --db-path data/market_parquet \
  --periodic --poll-seconds 2 --cycles 2

uv run python scripts/materialize_market_windows.py \
  --symbols BTC/USDT,ETH/USDT \
  --timeframes 1s,1m,5m,15m,30m,1h,4h,1d \
  --db-path data/market_parquet \
  --periodic --poll-seconds 5 --cycles 2

uv run lq live
```

Notes:
- Without explicit `--start-date/--end-date`, the periodic materializer only
  re-reads the UTC date partitions that can still change from the latest
  committed `1s` manifest (default bundle => usually current UTC day so far;
  actual span depends on the largest required timeframe and anchor gap).
- Use `--full-rebuild` for intentional historical rebuilds or raw backfills that
  land earlier than the latest committed materializer anchor.

Live fail-fast contract:
- committed data missing/parity fatal -> process exits with code `2`
- no empty MARKET_WINDOW fallback is allowed
- recovery: restore committed manifests, restart collector/materializer/live in order

Pre-live committed-data verification (copy/paste):

```bash
uv run python - <<'PY'
from lumina_quant.storage.parquet import ParquetMarketDataRepository

repo = ParquetMarketDataRepository("data/market_parquet")
for symbol in ("BTC/USDT", "ETH/USDT"):
    frame = repo.load_committed_ohlcv_chunked(
        exchange="binance",
        symbol=symbol,
        timeframe="1s",
    )
    print(symbol, "rows=", frame.height, "latest=", frame["datetime"].max())
PY
```

Rollout gate commands:

```bash
uv run python scripts/ci/export_market_window_gate_metrics.py \
  --input logs/live/market_window_metrics.ndjson \
  --output reports/live_rollout/baseline_gate_metrics.json \
  --window-hours 24 --require-flag false

uv run python scripts/ci/export_market_window_gate_metrics.py \
  --input logs/live/market_window_metrics.ndjson \
  --output reports/live_rollout/canary_gate_metrics.json \
  --window-hours 24 --require-flag true

uv run python scripts/ci/check_market_window_rollout_gates.py \
  --baseline reports/live_rollout/baseline_gate_metrics.json \
  --canary reports/live_rollout/canary_gate_metrics.json \
  --max-p95-payload-bytes 131072 \
  --max-queue-lag-increase-pct 5 \
  --max-fail-fast-incidents 0
```

---

## 2) Runtime knobs for 8GB-safe runs

```bash
export LQ_GPU_MODE=auto
export LQ_GPU_DEVICE=0
export LQ_GPU_VERBOSE=0

export LQ__BACKTEST__SKIP_AHEAD_ENABLED=1
export LQ__BACKTEST__CHUNK_DAYS=7            # tune 1..60
export LQ__BACKTEST__CHUNK_WARMUP_BARS=0

export LQ_BACKTEST_LOW_MEMORY=1
export LQ_BACKTEST_PERSIST_OUTPUT=0
export LQ__STORAGE__WAL_MAX_BYTES=268435456
export LQ__STORAGE__WAL_COMPACT_ON_THRESHOLD=1
export LQ__STORAGE__WAL_COMPACTION_INTERVAL_SECONDS=3600
export LQ_AUTO_COLLECT_DB=0
```

---

## 3) 1Y 1s backtest (memory-profiled)

`--low-memory` is auto-enabled for windows longer than 30 days (use `--no-low-memory` to override).

```bash
/usr/bin/time -v \
uv run lq backtest \
  --data-source db \
  --market-db-path data/market_parquet \
  --market-exchange binance \
  --base-timeframe 1s \
  --no-persist-output \
  --no-auto-collect-db \
  --run-id bt-1y-1s-$(date +%Y%m%d-%H%M%S) \
2>&1 | tee logs/backtest_1y_1s.log
```

Extract peak RSS (KB):

```bash
grep "Maximum resident set size" logs/backtest_1y_1s.log
```

---

## 4) 1Y 1s optimization (OOM-safe profile)

```bash
/usr/bin/time -v \
uv run lq optimize \
  --data-source db \
  --market-db-path data/market_parquet \
  --market-exchange binance \
  --base-timeframe 1s \
  --folds 3 \
  --n-trials 20 \
  --max-workers 1 \
  --oos-days 30 \
  --no-auto-collect-db \
  --run-id opt-1y-1s-$(date +%Y%m%d-%H%M%S) \
2>&1 | tee logs/optimize_1y_1s.log
```

---

## 5) Pass/Fail gates (local hardware)

- Backtest/opt run completes with exit code 0
- No OOM-kill (`dmesg -T | grep -i -E "killed process|out of memory"` should be empty for run window)
- Peak RSS stays below practical limit (recommend target: **< 7.2 GiB** on 8GB host)
- No fallback contract regressions:
  - `bash scripts/ci/architecture_gate_live_data.sh` passes
  - `bash scripts/ci/architecture_gate_market_window_contract.sh` passes
  - `uv run python scripts/audit_hardcoded_params.py` → `new=0`
  - `uv run python scripts/check_architecture.py` passes

---

## 6) If memory is still too high

1. Lower chunk size:
   ```bash
   export LQ__BACKTEST__CHUNK_DAYS=3
   ```
2. Keep optimization worker at 1:
   ```bash
   --max-workers 1
   ```
3. Ensure low-memory output is active:
   ```bash
   --low-memory --no-persist-output
   ```
4. Re-run WAL compaction before next run.

---

## 7) Streamlit GUI (monitor + launcher)

```bash
uv run python -m streamlit run apps/dashboard/app.py
```

Live real mode remains gated by dashboard arming phrase: **ENABLE REAL**.
