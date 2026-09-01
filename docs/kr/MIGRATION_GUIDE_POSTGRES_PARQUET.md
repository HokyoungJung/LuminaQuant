# 마이그레이션 가이드: 레거시 저장소 → 로컬 PostgreSQL + Parquet

이 문서는 LuminaQuant 런타임 워크플로우를 로컬 PostgreSQL + Parquet 구조로 옮길 때의 기준 절차를 설명합니다.

## 목표 아키텍처

- **시장 데이터**: 파티션된 Parquet 파일 (`ParquetMarketDataRepository`)
- **실행/감사/워크플로 상태**: PostgreSQL (`PostgresStateRepository`)
- **분석/리샘플링**: Polars weekly-chunk 파이프라인
- **계산 엔진**: `compute_engine`를 통한 CPU/GPU 해석

## 새 환경 변수

- `LQ_POSTGRES_DSN` (필수): 런타임 상태 저장용 PostgreSQL DSN
- `LQ_GPU_MODE` (`gpu|auto|cpu|forced-gpu`, 프로젝트 기본값: `gpu`)
- `LQ_GPU_DEVICE` (선택): 장치 인덱스 (`0`, `cuda:0` 등)
- `LQ_GPU_VERBOSE` (`0|1`, 선택)

## PostgreSQL 스키마 커버리지

`PostgresStateRepository.initialize_schema()`는 다음과 같은 idempotent 테이블을 생성합니다.

- `runs` (`run_id` PK)
- `equity` (`UNIQUE(run_id, timeindex)`)
- `orders` (`UNIQUE(run_id, client_order_id)`)
- `fills` (`UNIQUE(run_id, dedupe_key)`)
- `positions` (`UNIQUE(run_id, symbol, position_side)`)
- `risk_events` (`UNIQUE(run_id, dedupe_key)`)
- `heartbeats` (`UNIQUE(run_id, dedupe_key)`)
- `order_state_events` (`UNIQUE(run_id, dedupe_key)`)
- `optimization_results` (`UNIQUE(run_id, stage, fingerprint)`)
- `workflow_jobs` (`job_id` PK)

모든 쓰기 경로는 `INSERT ... ON CONFLICT ... DO UPDATE`를 사용해 replay-safe / idempotent 동작을 보장합니다.

## 1회 설정

```bash
# 1) DSN export
export LQ_POSTGRES_DSN='postgresql://user:pass@127.0.0.1:5432/luminaquant'

# 2) 스키마 초기화
uv run python scripts/init_postgres_schema.py
```

DDL만 미리 보고 싶다면:

```bash
uv run python scripts/init_postgres_schema.py --print-ddl
```

## Parquet compaction

단일 날짜 파티션 compact:

```bash
uv run python scripts/compact_parquet_market_data.py \
  --root-path data/market_parquet \
  --exchange binance \
  --symbol BTC/USDT \
  --date 2026-02-01
```

전체 파티션 compact:

```bash
uv run python scripts/compact_parquet_market_data.py \
  --root-path data/market_parquet \
  --exchange binance \
  --symbol BTC/USDT
```

## 권장 cutover 순서

1. PostgreSQL 스키마를 초기화합니다
2. 상태 쓰기(runs/equity/orders/fills 등)를 `PostgresStateRepository`로 전환합니다
3. OHLCV 읽기/쓰기를 `ParquetMarketDataRepository`로 전환합니다
4. weekly chunked Polars 리샘플링 파이프라인을 활성화합니다
5. 레거시 저장소 설정/죽은 코드 경로를 제거합니다
6. integration + determinism 테스트를 수행합니다

## 검증 체크리스트

```bash
uv run pytest tests/test_integration_parquet_postgres.py tests/test_week_chunk_determinism.py
```

- 스키마 생성이 반복 실행에도 성공하는지
- replay된 쓰기가 idempotent한지
- Parquet compaction이 timestamp별 최신 row를 보존하는지
- weekly-chunk 출력이 반복 실행에서도 결정론적인지

## 롤백 전략

PostgreSQL이 불안정하면 새 run을 중지하고 이전 release build로 복귀하세요.

이 마이그레이션 대상에서는 deprecated 저장소 엔진과 **dual-write를 하지 않습니다**.
