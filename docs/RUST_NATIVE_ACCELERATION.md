# Rust Native Acceleration Policy

LuminaQuant keeps the public/runtime API in Python and moves only proven hot kernels to Rust. A Rust path is promoted only when it preserves output parity, improves local runtime materially, and stays within the 8GB session memory budget.

## Current decisions (2026-05-27)

| Kernel | Python API | Native backend | Default decision | Evidence |
| :--- | :--- | :--- | :--- | :--- |
| Raw aggTrades → canonical 1s OHLCV | `lumina_quant.data.raw_first_lineage.raw_aggtrades_to_1s_frame` | `native/rust_rawfirst` | **Use Rust automatically when the release library is built** | Local 200k-trade synthetic benchmark: Python `0.0496s/eval`, Rust `0.0246s/eval`, about `2.01x` faster with frame parity. |
| Optimization metric evaluator | `lumina_quant.optimization.fast_eval.evaluate_threshold_strategy` | `native/rust_metrics` / `native/c_metrics` | **Keep Numba/Python auto-selection; do not force Rust yet** | Local native-compare benchmark: auto-selected Numba around `11.1k eval/s`; forced Rust around `7.9k eval/s`, parity OK but slower on this host. |

## Build and benchmark

Build native release libraries:

```bash
uv run python scripts/build_native_backends.py --backend all
```

Benchmark raw-first Python vs Rust:

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

Benchmark metric evaluator backend selection:

```bash
uv run python scripts/benchmark_native_compare.py --bars 50000 --evals 5000
```

Latest local evidence is stored under `var/reports/native_acceleration_20260527/`.

## Runtime controls

Raw-first backend:

- `LQ_RAW_FIRST_BACKEND=auto` (default): load Rust when `native/rust_rawfirst/target/release/liblumina_rawfirst.so` exists, otherwise fall back to Python with diagnostics.
- `LQ_RAW_FIRST_BACKEND=rust`: require Rust and fail if unavailable.
- `LQ_RAW_FIRST_BACKEND=python`: force the Python/Polars path.
- `LQ_RAW_FIRST_BACKEND_DLL=/path/to/liblumina_rawfirst.so`: explicit Rust library override.

Metric evaluator backend:

- `LQ_NATIVE_BACKEND=auto` (default): benchmark native vs fallback and choose the faster parity-safe backend.
- `LQ_NATIVE_BACKEND=native|numba|python`: force a backend for diagnostics.
- `LQ_NATIVE_METRICS_DLL=/path/to/liblumina_metrics.so`: explicit native metrics library override.
- `LQ_NATIVE_MIN_SPEEDUP`: minimum native speedup for auto-promotion.

## Promotion rule

A kernel may be migrated to Rust only when all of the following hold:

1. Python wrapper/API stays stable.
2. Python-vs-Rust parity is tested on deterministic inputs.
3. Local benchmark shows material speedup on the maintained environment.
4. The benchmark stays under the 8GB memory budget.
5. Research/live safety semantics are unchanged (`ready_for_real=false` and `real_money_execution=false` remain hard-false where artifacts require them).

If a Rust backend loses to an existing vectorized/Numba path, keep it optional and let the Python selector choose the faster backend.
