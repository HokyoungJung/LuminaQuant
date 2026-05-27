# Rust Native 가속 정책

LuminaQuant는 외부/운영 API를 Python으로 유지하고, 실제로 빠른 hot kernel만 Rust로 옮깁니다. Rust 경로는 출력 parity, 의미 있는 속도 개선, 8GB 세션 메모리 예산을 모두 만족할 때만 기본 경로로 승격합니다.

## 현재 결정 (2026-05-27)

| Kernel | Python API | Native backend | 기본 결정 | 근거 |
| :--- | :--- | :--- | :--- | :--- |
| raw aggTrades → canonical 1s OHLCV | `lumina_quant.data.raw_first_lineage.raw_aggtrades_to_1s_frame` | `native/rust_rawfirst` | **release library가 빌드되어 있으면 Rust를 자동 사용** | 로컬 200k-trade synthetic benchmark: Python `0.0496s/eval`, Rust `0.0246s/eval`, frame parity 유지, 약 `2.01x` 빠름. |
| optimization metric evaluator | `lumina_quant.optimization.fast_eval.evaluate_threshold_strategy` | `native/rust_metrics` / `native/c_metrics` | **Numba/Python auto-selection 유지, Rust 강제 전환 금지** | 로컬 native-compare benchmark: auto-selected Numba 약 `11.1k eval/s`, forced Rust 약 `7.9k eval/s`; parity는 맞지만 현재 host에서는 Rust가 더 느림. |

## 빌드와 벤치마크

native release library 빌드:

```bash
uv run python scripts/build_native_backends.py --backend all
```

raw-first Python/Rust 비교:

```bash
uv run python scripts/benchmark_rawfirst_backend.py \
  --trades 200000 \
  --seconds 60000 \
  --evals 3 \
  --backend both \
  --require-rust \
  --require-speedup \
  --min-speedup 1.05
```

metric evaluator backend 선택 확인:

```bash
uv run python scripts/benchmark_native_compare.py --bars 50000 --evals 5000
```

최신 로컬 근거 파일은 `var/reports/native_acceleration_20260527/` 아래에 저장되어 있습니다.

## 런타임 제어

Raw-first backend:

- `LQ_RAW_FIRST_BACKEND=auto` (기본): `native/rust_rawfirst/target/release/liblumina_rawfirst.so`가 있으면 Rust를 로드하고, 없으면 진단 정보를 남기고 Python으로 fallback.
- `LQ_RAW_FIRST_BACKEND=rust`: Rust를 필수로 요구하고 없으면 실패.
- `LQ_RAW_FIRST_BACKEND=python`: Python/Polars 경로 강제.
- `LQ_RAW_FIRST_BACKEND_DLL=/path/to/liblumina_rawfirst.so`: Rust library 명시 경로.

Metric evaluator backend:

- `LQ_NATIVE_BACKEND=auto` (기본): native와 fallback을 비교해 parity가 맞고 더 빠른 쪽을 선택.
- `LQ_NATIVE_BACKEND=native|numba|python`: 진단용 backend 강제.
- `LQ_NATIVE_METRICS_DLL=/path/to/liblumina_metrics.so`: native metrics library 명시 경로.
- `LQ_NATIVE_MIN_SPEEDUP`: auto 승격에 필요한 최소 native speedup.

## 승격 규칙

Rust 전환은 아래 조건을 모두 만족할 때만 허용합니다.

1. Python wrapper/API가 안정적으로 유지된다.
2. deterministic input에서 Python-vs-Rust parity test가 있다.
3. 유지보수 환경에서 local benchmark가 의미 있는 속도 개선을 보인다.
4. benchmark가 8GB 메모리 예산 안에서 돈다.
5. research/live safety semantics가 바뀌지 않는다. artifact가 요구하는 `ready_for_real=false`, `real_money_execution=false`는 계속 hard-false여야 한다.

Rust backend가 기존 vectorized/Numba 경로보다 느리면 optional로만 두고, Python selector가 더 빠른 경로를 선택하게 둡니다.
