# Rust Native 가속 정책

LuminaQuant는 외부/운영 API를 Python으로 유지하고, 실제로 빠른 hot kernel만 Rust로 옮깁니다. Rust 경로는 출력 parity, 의미 있는 속도 개선, 8GB 세션 메모리 예산을 모두 만족할 때만 기본 경로로 승격합니다.

## 현재 아키텍처 (2026-07-03)

모든 native kernel은 단일 pyo3 크레이트 `native/lumina_compute`에 있고, cdylib로 빌드되어 `lumina_quant._compute`로 import됩니다(모듈 이름은 `native/lumina_compute/pyproject.toml`의 `[tool.maturin] module-name`으로 고정). 이 구조는 기존에 각자 `.so`/DLL과 전용 환경변수 override를 가지고 있던 5개의 개별 ctypes 크레이트(`native/rust_rawfirst`, `native/rust_hybrid_optuna`, `native/rust_live_signals`, `native/rust_metrics`, `native/c_metrics`)를 대체했습니다. 이 크레이트들과 그 DLL 경로는 더 이상 존재하지 않으며, 옛 환경변수는 하위 호환을 위해 계속 인식되지만 무시됩니다(아래 런타임 제어 참고). `native/lumina_compute/src/lib.rs` 주석에 따르면 마이그레이션된 각 kernel의 계산 로직은 대체된 ctypes 크레이트와 bit-identical합니다.

`lumina_quant._compute`가 export하는 함수: `evaluate_metrics`, `simulate_symbol_fold`, `debounced_state_signal`, `trailing_state_signal`, `evaluate_hybrid_optuna_portfolio`, `aggregate_raw_aggtrades_to_1s`, `append_ohlcv_1s_wal`, 그리고 (2026-07-03 추가) `build_info` (아래 버전 handshake 절 참고).

| Kernel | Python API | 기본 결정 | 근거 |
| :--- | :--- | :--- | :--- |
| raw aggTrades → canonical 1s OHLCV | `lumina_quant.data.raw_first_lineage.raw_aggtrades_to_1s_frame` | **`lumina_quant._compute`가 import 가능하면 자동 사용** | 로컬 200k-trade synthetic benchmark(통합 이전 `native/rust_rawfirst` 기준): Python `0.0496s/eval`, Rust `0.0246s/eval`, frame parity 유지, 약 `2.01x` 빠름. |
| Alpha Zoo Optuna hybrid portfolio loop | `scripts/research/run_alpha_zoo_integer_leverage_optuna_hybrid_decision.py::_portfolio_returns_for_params` | **`lumina_quant._compute`가 import 가능하고 input이 finite이면 자동 사용** | 로컬 20k×3 synthetic benchmark(통합 이전 `native/rust_hybrid_optuna` 기준): Python `2.5493s/eval`, Rust `0.0107s/eval`, 약 `238.07x` 빠름; return diff `1.01e-11`, exposed-weight diff `9.15e-09`로 `1e-8` tolerance 이내. |
| Alpha Zoo live state-signal state machine | `lumina_quant.alpha_zoo.optuna_hybrid_signals.debounced_state_signal` / `trailing_state_signal` | **`lumina_quant._compute`가 import 가능하면 자동 사용** | 로컬 50k-row synthetic benchmark(통합 이전 `native/rust_live_signals` 기준): Python `0.1349s/eval`, Rust `0.000544s/eval`, 약 `247.80x` 빠름; debounced/trailing state 배열이 정확히 일치. |
| Alpha Zoo fold 단위 심볼 시뮬레이션 | `lumina_quant.alpha_zoo.native_alpha_fold_backend.simulate_symbol_arrays` (`scripts/research/run_alpha_zoo_clean_new_alpha_discovery.py`에서 호출) | **`lumina_quant._compute`가 import 가능하면 자동 사용** | 별도 ctypes 전신 없이 pyo3 크레이트로 바로 마이그레이션됨; parity는 `tests/test_native_alpha_fold_backend.py`, `tests/test_native_backend_parity.py`로 커버. |
| live `MARKET_WINDOW` event 생성 / rolling aggregation | `lumina_quant.core.market_window_contract.build_market_window_event` / `lumina_quant.live.market_window_rolling.RollingWindowAggregator` | **이 경계는 Rust 추가 금지; 내부 canonical-row fast path 사용** | profiling 결과 병목은 numeric compute가 아니라 Python row normalization/schema validation 반복이었습니다. 로컬 5-symbol/300s benchmark: generic builder `0.0005508s/eval`, trusted builder `0.000001006s/eval`(`builder ~547x`), rolling aggregation `16.5k ticks/s`, RSS `42.5MB`. `MarketWindowEvent`는 결국 Python tuple payload가 필요해 Rust FFI 전환 이득이 작습니다. |
| optimization metric evaluator | `lumina_quant.optimization.native_backend.evaluate_metrics_backend` (`lumina_quant.services.portfolio`에서 사용) | **import 시점에 pyo3 vs Numba/Python을 벤치마크해서 더 빠르고 parity가 맞는 쪽을 유지** | 로컬 native-compare benchmark: auto-selected Numba 약 `11.1k eval/s`, forced pyo3 약 `7.9k eval/s`; parity는 맞지만 현재 host에서는 Numba가 더 빨라 기본으로 유지. |
| Binary WAL 1s OHLCV append | `lumina_quant.storage.wal.native_backend.append_ohlcv_frame_native` (`ParquetOhlcvRepository.upsert_1s`에서 사용) | **pyo3 kernel을 항상 먼저 시도하고, 안 되면 순수 Python `BinaryWAL.append`로 투명하게 fallback** | 두 경로 모두 byte-identical한 WAL record를 쓰기 때문에 "Python 강제" 스위치 자체가 없습니다. |

## 빌드

```bash
uv run python scripts/build_native_backends.py           # maturin develop --release (editable install)
uv run python scripts/build_native_backends.py --wheel   # maturin build --release (wheel)
```

스크립트는 `$VIRTUAL_ENV/bin`(없으면 repo의 `.venv`)에서 `maturin`을 찾아 `native/lumina_compute` 안에서 실행합니다. 동등한 직접 명령:

```bash
cd native/lumina_compute
cargo build --release      # compile-check만 함; 활성 Python 환경에 설치되지 않음
maturin develop --release  # 빌드 + 활성 venv에 lumina_quant._compute로 설치
```

재빌드 후 실행할 native/parity 테스트:

```bash
PYTHONPATH=. uv run python -m pytest tests -q -k "native or hybrid or compute"
```

## Native kernel 버전 handshake (2026-07-03)

`lumina_quant._compute`는 editable extension이라, 체크아웃된 크레이트 소스와 컴파일된 `.so`가 서로 어긋날 수 있습니다: `native/lumina_compute/src/lib.rs`를 수정하고 `maturin develop`을 다시 돌리지 않으면 이전에 빌드된 kernel이 에러 없이 그대로 로드됩니다. 이를 잡기 위해:

- 크레이트는 `build_info()`를 export하며, 컴파일 시점의 `env!("CARGO_PKG_VERSION")`(즉 `native/lumina_compute/Cargo.toml`의 버전)을 반환합니다.
- `lumina_quant._compute` import에 성공한 모든 `native_*_backend` 모듈은 프로세스당 한 번씩 `lumina_quant._native_kernel_version.check_native_kernel_version()`을 호출합니다. 이 함수는 `native/lumina_compute/Cargo.toml`에서 기대 버전을 읽어 `build_info()`와 비교합니다:
  - 버전이 일치 → 조용히 통과.
  - 버전이 다름 → `logger.warning("stale native kernel: ...")` 한 번, 재빌드 방법(`maturin develop --release` 또는 `scripts/build_native_backends.py`) 안내.
  - `build_info()`가 없음(이 handshake 이전에 빌드된 `.so`) → kernel이 handshake 이전 버전이라는 경고 한 번.
  - `native/lumina_compute/Cargo.toml`을 찾을 수 없음(예: 크레이트 소스가 없는 설치된 wheel) → 비교 대상이 없으므로 조용히 skip.
- 이 체크는 절대 raise하지 않습니다. 순수 진단용이며, `tests/test_native_kernel_version.py`가 순수 함수 `compare_native_version(expected, reported) -> ok|stale|missing`를 단독으로 검증합니다.

## 런타임 제어

Alpha Zoo fold 단위 심볼 시뮬레이션:

- `LQ_ALPHA_FOLD_BACKEND=auto` (기본): `lumina_quant._compute`를 import할 수 있으면 pyo3 kernel 사용, 아니면 Python 경로.
- `LQ_ALPHA_FOLD_BACKEND=rust`: pyo3 kernel을 필수로 요구하고 없으면 실패.
- `LQ_ALPHA_FOLD_BACKEND=python`: Python 경로 강제.
- `LQ_ALPHA_FOLD_DLL`: 환경변수 하위 호환용으로만 남아 있으며 무시됩니다.

Alpha Zoo Optuna hybrid portfolio backend:

- `LQ_HYBRID_OPTUNA_BACKEND=auto` (기본): import 가능하고 return matrix가 finite이면 pyo3 kernel 사용, 아니면 Python.
- `LQ_HYBRID_OPTUNA_BACKEND=rust`: pyo3 kernel을 필수로 요구하고 없으면 실패.
- `LQ_HYBRID_OPTUNA_BACKEND=python`: 기존 Python loop 강제.
- `LQ_HYBRID_OPTUNA_DLL`: 환경변수 하위 호환용으로만 남아 있으며 무시됩니다.
- `auto` 모드에서는 kernel이 없거나 호출이 실패하면 이유별로 한 번씩 경고를 남깁니다(`_warn_once`, 2026-07-03 audit fix). 이전처럼 훨씬 느린 Python 경로로 조용히 넘어가지 않습니다.

Live Alpha Zoo state-signal backend:

- `LQ_LIVE_SIGNAL_BACKEND=auto` (기본) / `rust` / `python`, 위와 동일한 의미.
- `LQ_LIVE_SIGNAL_BACKEND_DLL`: 환경변수 하위 호환용으로만 남아 있으며 무시됩니다.
- `auto` 모드는 서로 다른 fallback 사유별로 한 번씩 경고를 남깁니다(`_warn_auto_fallback_once`, 2026-07-03 audit fix).

Raw aggTrades → 1s OHLCV backend:

- `LQ_RAW_FIRST_BACKEND=auto` (기본) / `rust` / `python`, 위와 동일한 의미.
- `LQ_RAW_FIRST_BACKEND_DLL`: 환경변수 하위 호환용으로만 남아 있으며 무시됩니다.
- `auto` 모드는 서로 다른 fallback 사유별로 한 번씩 경고를 남깁니다(2026-07-03 audit fix).

Binary WAL append:

- backend 선택 환경변수가 없습니다. `ParquetOhlcvRepository.upsert_1s`는 항상 `append_ohlcv_frame_native`를 먼저 호출하고, extension을 쓸 수 없거나 호출이 실패하면 경고 없이 순수 Python `BinaryWAL.append`로 넘어갑니다.

Optimization metric evaluator:

- `LQ_NATIVE_BACKEND=auto` (기본): import 시점에 pyo3 kernel과 Numba/Python을 벤치마크해서(`_select_fastest_backend`) 더 빠르고 parity가 맞는 쪽을 유지.
- `LQ_NATIVE_BACKEND=native|numba|python`: 진단용 backend 강제.
- `LQ_NATIVE_AUTO_SELECT=0|false`: 속도 비교를 건너뛰고 parity만 확인되면 pyo3 kernel을 무조건 사용.
- `LQ_NATIVE_MIN_SPEEDUP`: auto-selection이 pyo3를 선택하기 위한 최소 속도비.
- `LQ_NATIVE_BENCH_LOOPS`, `LQ_NATIVE_SELECTION_TOL`: 시작 시점 벤치마크의 loop 횟수와 parity tolerance 조정.
- 이 모듈은 fallback 시 경고를 남기지 않습니다 — 벤치마크 기반 selector가 import 시점에 고른 backend를 그대로 유지할 뿐입니다.

## 벤치마크 스크립트

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

Optuna hybrid portfolio loop Python/Rust 비교:

```bash
uv run python scripts/benchmark_optuna_hybrid_backend.py \
  --rows 20000 \
  --columns 3 \
  --evals 10 \
  --version v3_5 \
  --backend both \
  --require-rust \
  --require-speedup \
  --min-speedup 1.2
```

live state-signal Python/Rust 비교:

```bash
uv run python scripts/benchmark_live_signal_backend.py \
  --rows 50000 \
  --evals 20 \
  --backend both \
  --require-rust
```

live `MARKET_WINDOW` 생성/rolling aggregation benchmark:

```bash
uv run python scripts/benchmark_market_window_contract.py \
  --symbols 5 \
  --window-seconds 300 \
  --ticks 5000 \
  --evals 500
```

metric evaluator backend 선택 확인:

```bash
uv run python scripts/benchmark_native_compare.py --bars 50000 --evals 5000
```

최신 로컬 근거 파일은 `var/reports/native_acceleration_20260527/` 아래에 저장되어 있습니다.

## 승격 규칙

Rust 전환은 아래 조건을 모두 만족할 때만 허용합니다.

1. Python wrapper/API가 안정적으로 유지된다.
2. deterministic input에서 Python-vs-Rust parity test가 있다.
3. 유지보수 환경에서 local benchmark가 의미 있는 속도 개선을 보인다.
4. benchmark가 8GB 메모리 예산 안에서 돈다.
5. research/live safety semantics가 바뀌지 않는다. artifact가 요구하는 `ready_for_real=false`, `real_money_execution=false`는 계속 hard-false여야 한다.

Rust backend가 기존 vectorized/Numba 경로보다 느리면 optional로만 두고, Python selector가 더 빠른 경로를 선택하게 둡니다.
