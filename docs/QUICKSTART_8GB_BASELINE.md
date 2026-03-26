# 8GB Baseline Quickstart (Install → Smoke → Shadow Live)

This quickstart is the fastest path to validate LuminaQuant on an **8GB RAM baseline** with one official gate command.

## 1) Install

```bash
uv python pin 3.13
uv sync --extra optimize --extra dev --extra live
```

Optional (Linux x86_64 + CUDA 12):

```bash
uv sync --extra gpu
```

## 2) Smoke backtest (memory-profiled)

```bash
mkdir -p logs reports/benchmarks
/usr/bin/time -v \
  uv run lq backtest --data-source csv --no-persist-output --no-auto-collect-db \
  2>&1 | tee logs/8gb_smoke_backtest.log
```

## 3) Benchmark + 8GB baseline gate

```bash
/usr/bin/time -v \
  uv run python scripts/benchmark_backtest.py \
    --iters 1 \
    --warmup 0 \
    --output reports/benchmarks/8gb_smoke.json \
  2>&1 | tee logs/8gb_benchmark_time.log

uv run python scripts/verify_8gb_baseline.py \
  --benchmark reports/benchmarks/8gb_smoke.json \
  --time-log logs/8gb_benchmark_time.log \
  --oom-log logs/8gb_benchmark_time.log \
  --skip-dmesg \
  --rss-limit-gib 7.2 \
  --disk-budget-gib 30 \
  --output reports/benchmarks/8gb_baseline_gate.json
```

Gate checks covered by `scripts/verify_8gb_baseline.py`:
- RSS threshold (`< 7.2 GiB`)
- OOM evidence scan (provided run logs via `--oom-log`, optional kernel scan if `--skip-dmesg` is removed)
- disk budget snapshot (`data`, `logs`, `reports` by default)
- benchmark JSON parse sanity

## 4) Replay + shadow-live smoke

Replay regression smoke:

```bash
uv run pytest tests/test_replay.py -q
```

Shadow-live dry run (paper mode by default):

```bash
STOP_FILE=/tmp/lq.shadow.stop
rm -f "$STOP_FILE"
uv run lq live --no-selection --run-id shadow-$(date +%Y%m%d-%H%M%S) --stop-file "$STOP_FILE"
```

## 5) Dashboard status

The interactive dashboard runtime is not shipped on `public/main`.
Use the CLI flow in this quickstart instead.

## 6) Safe stop

From another shell:

```bash
touch /tmp/lq.shadow.stop
```

(If you launched with a different path, touch that stop-file path instead.)

## 7) Cleanup

```bash
rm -f /tmp/lq.shadow.stop
uv run python scripts/cleanup_ghost_runs.py --dsn "$LQ_POSTGRES_DSN" --stale-sec 300 --startup-grace-sec 90 --apply
```

Artifacts to keep for evidence:
- `logs/8gb_smoke_backtest.log`
- `logs/8gb_benchmark_time.log`
- `reports/benchmarks/8gb_smoke.json`
- `reports/benchmarks/8gb_baseline_gate.json`
