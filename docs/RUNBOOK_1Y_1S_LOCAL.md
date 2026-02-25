# 1Y+ 1s Local Runbook (8GB RAM / 8GB VRAM)

This runbook is for **local-only uv runtime** and the current LuminaQuant stack:
- market data: monthly parquet + binary WAL
- control plane: PostgreSQL
- compute: Polars (GPU auto fallback)

---

## 0) One-time setup

```bash
cd /home/hoky/Quants-agent/LuminaQuant
uv sync --all-extras
uv run python scripts/init_postgres_schema.py --dsn "$LQ_POSTGRES_DSN"
```

Use symbols (top10 + XAU/XAG) via env override:

```bash
export LQ__TRADING__SYMBOLS='["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT","ADA/USDT","DOGE/USDT","TRX/USDT","LTC/USDT","LINK/USDT","XAU/USDT","XAG/USDT"]'
```

---

## 1) Backfill 1-second data (1 year+)

```bash
uv run python scripts/sync_binance_ohlcv.py \
  --symbols BTC/USDT ETH/USDT BNB/USDT SOL/USDT XRP/USDT ADA/USDT DOGE/USDT TRX/USDT LTC/USDT LINK/USDT XAU/USDT XAG/USDT \
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

---

## 2) Runtime knobs for 8GB-safe runs

```bash
export LQ_GPU_MODE=auto
export LQ_GPU_DEVICE=0
export LQ_GPU_VERBOSE=0

export LQ_SKIP_AHEAD=1
export LQ_BT_CHUNK_DAYS=7            # tune 1..60
export LQ_BT_CHUNK_WARMUP_BARS=0

export LQ_BACKTEST_LOW_MEMORY=1
export LQ_BACKTEST_PERSIST_OUTPUT=0
export LQ_AUTO_COLLECT_DB=0
```

---

## 3) 1Y 1s backtest (memory-profiled)

```bash
/usr/bin/time -v \
uv run python run_backtest.py \
  --data-source db \
  --market-db-path data/market_parquet \
  --market-exchange binance \
  --base-timeframe 1s \
  --low-memory \
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
uv run python optimize.py \
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
  - `uv run python scripts/audit_hardcoded_params.py` → `new=0`
  - `uv run python scripts/check_architecture.py` passes

---

## 6) If memory is still too high

1. Lower chunk size:
   ```bash
   export LQ_BT_CHUNK_DAYS=3
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
uv run python -m streamlit run dashboard.py
```

Live real mode remains gated by dashboard arming phrase: **ENABLE REAL**.

