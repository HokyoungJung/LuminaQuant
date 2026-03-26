# 8GB 기준 Quickstart (설치 → 스모크 → 섀도우 라이브)

이 문서는 **8GB RAM / 8GB VRAM** 기준에서 LuminaQuant를 빠르게 검증하는 최소 절차입니다.

## 1) 설치

```bash
uv python pin 3.13
uv sync --extra optimize --extra dev --extra live
```

선택 사항 (Linux x86_64 + CUDA 12):

```bash
uv sync --extra gpu
```

## 2) 스모크 백테스트 (메모리 프로파일 포함)

```bash
mkdir -p logs reports/benchmarks
/usr/bin/time -v \
  uv run lq backtest --data-source csv --no-persist-output --no-auto-collect-db \
  2>&1 | tee logs/8gb_smoke_backtest.log
```

## 3) 벤치마크 + 8GB 게이트

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

`scripts/verify_8gb_baseline.py`가 확인하는 항목:
- RSS 기준 (`< 7.2 GiB`)
- OOM 시그니처 스캔 (`--oom-log`, 필요 시 dmesg)
- 디스크 사용량 스냅샷 (`data`, `logs`, `reports`)
- 벤치마크 JSON 파싱 유효성

## 4) 리플레이 + 섀도우 라이브 스모크

리플레이 회귀 스모크:

```bash
uv run pytest tests/test_replay.py -q
```

섀도우 라이브(dry-run, 기본 paper 모드):

```bash
STOP_FILE=/tmp/lq.shadow.stop
rm -f "$STOP_FILE"
uv run lq live --no-selection --run-id shadow-$(date +%Y%m%d-%H%M%S) --stop-file "$STOP_FILE"
```

## 5) 대시보드 상태

인터랙티브 대시보드 런타임은 `public/main`에 포함되지 않습니다.
이 퀵스타트의 CLI 흐름을 사용하세요.

## 6) 안전 종료

다른 셸에서:

```bash
touch /tmp/lq.shadow.stop
```

(실행 시 다른 stop-file 경로를 썼다면 해당 파일 경로를 touch 하세요.)

## 7) 정리

```bash
rm -f /tmp/lq.shadow.stop
uv run python scripts/cleanup_ghost_runs.py --dsn "$LQ_POSTGRES_DSN" --stale-sec 300 --startup-grace-sec 90 --apply
```

증빙 아티팩트:
- `logs/8gb_smoke_backtest.log`
- `logs/8gb_benchmark_time.log`
- `reports/benchmarks/8gb_smoke.json`
- `reports/benchmarks/8gb_baseline_gate.json`
