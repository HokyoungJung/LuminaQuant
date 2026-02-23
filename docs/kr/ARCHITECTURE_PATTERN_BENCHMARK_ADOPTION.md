# 아키텍처 패턴 벤치마크 적용 현황 (LuminaQuant)

이 문서는 LuminaQuant에 적용한 고성능 엔진 패턴과 검증 방법을 정리합니다.

## 적용된 패턴

1. 결정론적 이벤트 식별자 (`timestamp_ns` + `sequence`)
   - 파일: `lumina_quant/events.py`, `lumina_quant/event_clock.py`, `lumina_quant/engine.py`
   - 검증: 리플레이 정렬 테스트, 단조 증가(monotonic) 검증.

2. 프로세스 내부 메시지 버스 (publish/subscribe, request/response, point-to-point)
   - 파일: `lumina_quant/message_bus.py`, `lumina_quant/engine.py`
   - 검증: `tests/test_message_bus.py`, 아키텍처 벤치마크의 bus 처리량.

3. 라이브 복구를 위한 중앙 런타임 캐시 + 아웃박스 영속화
   - 파일: `lumina_quant/runtime_cache.py`, `lumina_quant/live_trader.py`
   - 검증: `tests/test_runtime_cache.py`, 재시작 상태 스냅샷/복원 경로.

4. 최적화용 1회성 데이터셋 빌드 + frozen 배열
   - 파일: `lumina_quant/optimization/frozen_dataset.py`, `optimize.py`, `lumina_quant/data.py`
   - 검증: trial 루프에서 Polars 프레임 금지(`optimize.py` 하드 가드), pre-frozen tuple row 사용.

5. 컴파일된 평가 커널 경로 (Numba/native)
   - 파일: `lumina_quant/optimization/fast_eval.py`, `lumina_quant/optimization/native_backend.py`, `lumina_quant/services/portfolio.py`
   - 검증: `tests/test_fast_eval.py`, `tests/test_parity_fast_eval.py`, 커널 처리량 벤치마크.

6. Polars/Numba 동시 과구독(oversubscription) 제어
   - 파일: `lumina_quant/optimization/threading_control.py`, `optimize.py`
   - 검증: 시작 로그의 Numba thread 설정 값.

7. 실행 시뮬레이션 모델 모듈화
   - 파일: `lumina_quant/execution.py`
   - 검증: 보호 주문/상태머신 회귀 테스트.

8. 2단계 최적화기 (빠른 prefilter -> 전체 이벤트 리플레이)
   - 파일: `optimize.py`
   - 검증: 학습 단계 로그의 `[Two-Stage]` prefilter/replay 메시지.

## 런타임 활성화

```bash
uv sync --extra optimize --extra dev --extra live
winget install --id Rustlang.Rustup -e --accept-source-agreements --accept-package-agreements --disable-interactivity
export PATH="$PATH:/c/Users/<user>/.cargo/bin"
```

기본 백엔드 동작:

- `lumina_quant/optimization/native_backend.py`가 후보(`numba/python`, C DLL, Rust DLL)를 벤치마크하고, 가장 빠른 백엔드를 기본값으로 선택합니다.
- native 후보는 기준 경로와의 결과 일치(허용 오차 내)일 때만 채택합니다.
- 환경 변수:
  - `LQ_NATIVE_BACKEND` (기본 `auto`): 백엔드 강제 지정 (`auto|python|numba|native`)
  - `LQ_NATIVE_AUTO_SELECT` (기본 `1`): 자동 최적 선택 on/off
  - `LQ_NATIVE_MIN_SPEEDUP` (기본 `0.0`): native 전환 최소 상대 속도 이득
  - `LQ_NATIVE_BENCH_LOOPS` (기본 `256`): 마이크로 벤치 루프 수
  - `LQ_NATIVE_METRICS_DLL`: 명시 DLL 강제 지정

## 크로스 플랫폼 검증

- GitHub Actions 워크플로: `.github/workflows/cross-platform-ci.yml`
- 매트릭스 대상: Windows, Ubuntu, macOS
- 검증 단계: `uv sync`, native 백엔드 빌드, 린트, 회귀 테스트

## 네이티브 빌드 명령

```bash
uv run python scripts/build_native_backends.py --backend all
native\c_metrics\build_msvc_x64.bat
native\rust_metrics\build_release.bat
```

## 벤치마크 명령

```bash
uv run python scripts/benchmark_dataset_build.py --symbols 6 --rows 50000
uv run python scripts/benchmark_optimization_kernel.py --bars 10000 --evals 1000
uv run python scripts/benchmark_architecture.py --bars 20000 --evals 3000 --messages 200000 --updates 100000 --events 100000
uv run python scripts/benchmark_native_compare.py --bars 50000 --evals 5000
```

## 최신 벤치마크 스냅샷 (로컬)

- `benchmark_dataset_build.py --symbols 4 --rows 20000`
  - `rows_per_sec=2314198.77`
- `benchmark_optimization_kernel.py --bars 10000 --evals 1000`
  - `evals_per_sec=45863.57`
- `benchmark_architecture.py --bars 15000 --evals 1500 --messages 100000 --updates 50000 --events 50000`
  - `metric_eval_per_sec=26285.54`
  - `bus_publish_per_sec=2844505.13`
  - `cache_update_per_sec=740138.76`
  - `replay_sort_events_per_sec=1951516.52`
- `benchmark_native_compare.py --bars 20000 --evals 2000`
  - `native_backend_name=numba`
  - `python_eval_per_sec=5331.90`
  - `numba_eval_per_sec=19307.07`
  - `native_eval_per_sec=20957.53`
- `benchmark_native_compare.py --bars 20000 --evals 2000 --dll native/c_metrics/build/lumina_metrics.dll`
  - `native_eval_per_sec=15268.97`
- `benchmark_native_compare.py --bars 20000 --evals 2000 --dll native/rust_metrics/target/release/lumina_metrics.dll`
  - `native_eval_per_sec=17499.70`

## 2단계 최적화 프로파일 스냅샷

명령:

```bash
LQ_OPT_PROFILE=1 LQ_TWO_STAGE_OPT=1 LQ_TWO_STAGE_TOPK_RATIO=0.5 LQ_TWO_STAGE_PREFILTER_FRACTION=0.4 LQ_MIN_TRAIN_DAYS=1 \
uv run python optimize.py --folds 1 --n-trials 6 --max-workers 2 --oos-days 1 --data-source csv --no-auto-collect-db
```

결과 요약:

- 2단계 prefilter/replay 경로 실행 확인 (`[Two-Stage]` 로그).
- 멀티프로세스 리플레이에서 유효 결과가 없을 때 단일 워커 재시도 안전 경로 실행.
- 런타임 프로파일:
  - `feature/indicator: 0.0241s`
  - `simulation core: 0.0714s`
  - `orchestration: 0.0738s`
  - `avg simulation/call: 0.004762s`

## 검증 명령

```bash
uv run pytest tests/test_message_bus.py tests/test_runtime_cache.py tests/test_replay.py
uv run pytest tests/test_parity_fast_eval.py tests/test_fast_eval.py tests/test_frozen_dataset.py tests/test_event_clock.py tests/test_system_assembly.py tests/test_portfolio_fast_stats.py
uv run pytest tests/test_execution_protective_orders.py tests/test_live_execution_state_machine.py tests/test_lookahead.py
uv run pytest tests/test_optimize_two_stage.py tests/test_native_backend.py tests/test_strategy_registry_defaults.py
uv run ruff check lumina_quant optimize.py strategies/registry.py strategies/rsi_strategy.py strategies/moving_average.py scripts/benchmark_architecture.py scripts/benchmark_dataset_build.py scripts/benchmark_native_compare.py tests/test_message_bus.py tests/test_runtime_cache.py tests/test_replay.py tests/test_optimize_two_stage.py
```
