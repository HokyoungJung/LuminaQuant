# 1년+ 1초 로컬 런북 (8GB RAM / 8GB VRAM)

이 문서는 **로컬 전용 uv 런타임**과 현재 LuminaQuant 스택 기준 운영 절차입니다.

- 시장 데이터: monthly parquet + binary WAL
- 컨트롤 플레인: PostgreSQL
- 계산: Polars (GPU 우선, 필요 시 fallback)

---

## 0) 1회 설정

```bash
cd /path/to/<REPO_DIR>
uv sync --extra optimize --extra dev --extra live
# Linux x86_64 + CUDA 12 환경이면 권장
# uv sync --extra gpu
uv run python scripts/init_postgres_schema.py --dsn "$LQ_POSTGRES_DSN"
```

`<REPO_DIR>` 예시:
- `Quants-agent` (private source-of-truth)
- `LuminaQuant` (public mirror)

기본 12-symbol universe override:

```bash
export LQ__TRADING__SYMBOLS='["BTC/USDT","ETH/USDT","XRP/USDT","BNB/USDT","SOL/USDT","TRX/USDT","DOGE/USDT","ADA/USDT","TON/USDT","AVAX/USDT","XAU/USDT","XAG/USDT"]'
```

---

## 1) 1초 데이터 백필 (1년+)

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

WAL → monthly parquet compact:

```bash
uv run python scripts/compact_wal_to_monthly_parquet.py \
  --root-path data/market_parquet \
  --exchange binance
```

Raw-first collector/materializer/trader:

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

---

## 2) 8GB-safe runtime knobs

```bash
export LQ_GPU_MODE=gpu
export LQ_GPU_DEVICE=0
export LQ_GPU_VERBOSE=0

export LQ__BACKTEST__SKIP_AHEAD_ENABLED=1
export LQ__BACKTEST__CHUNK_DAYS=7
export LQ__BACKTEST__CHUNK_WARMUP_BARS=0

export LQ_BACKTEST_LOW_MEMORY=1
export LQ_BACKTEST_PERSIST_OUTPUT=0
export LQ__STORAGE__WAL_MAX_BYTES=268435456
export LQ__STORAGE__WAL_COMPACT_ON_THRESHOLD=1
export LQ__STORAGE__WAL_COMPACTION_INTERVAL_SECONDS=3600
export LQ_AUTO_COLLECT_DB=0
```

---

## 3) 1Y 1s 백테스트

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

Peak RSS 확인:

```bash
grep "Maximum resident set size" logs/backtest_1y_1s.log
```

---

## 4) 1Y 1s 최적화

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

## 5) Pass / Fail 게이트

- backtest / optimize run exit code 0
- OOM-kill 없음
- Peak RSS가 현실적 한도 내 유지 (권장: **< 7.2 GiB**)
- fallback contract regression 없음
  - `bash scripts/ci/architecture_gate_live_data.sh`
  - `bash scripts/ci/architecture_gate_market_window_contract.sh`
  - `uv run python scripts/audit_hardcoded_params.py` → `new=0`
  - `uv run python scripts/check_architecture.py`

---

## 6) 메모리가 여전히 높다면

1. chunk size 감소
   ```bash
   export LQ__BACKTEST__CHUNK_DAYS=3
   ```
2. optimization worker는 1 유지
   ```bash
   --max-workers 1
   ```
3. low-memory output 강제
   ```bash
   --low-memory --no-persist-output
   ```
4. 다음 run 전에 WAL compaction 재실행
