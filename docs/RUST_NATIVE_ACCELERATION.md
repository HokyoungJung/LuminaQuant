# Rust Native Acceleration Policy

LuminaQuant keeps the public/runtime API in Python and moves only proven hot kernels to Rust. A Rust path is promoted only when it preserves output parity, improves local runtime materially, and stays within the 8GB session memory budget.

## Current architecture (2026-07-03)

All native kernels live in a single pyo3 crate, `native/lumina_compute`, built as a cdylib and imported as `lumina_quant._compute` (the module name is fixed by `native/lumina_compute/pyproject.toml`'s `[tool.maturin] module-name`). This replaced five separate ctypes crates (`native/rust_rawfirst`, `native/rust_hybrid_optuna`, `native/rust_live_signals`, `native/rust_metrics`, `native/c_metrics`), each of which used to ship its own `.so`/DLL with a dedicated environment-variable override. Those crates and their DLL paths no longer exist; the retired env vars are still accepted for backward compatibility but are ignored (see Runtime controls below). Per `native/lumina_compute/src/lib.rs`, computation logic for every migrated kernel is bit-identical to the ctypes crate it replaced.

`lumina_quant._compute` exports: `evaluate_metrics`, `simulate_symbol_fold`, `debounced_state_signal`, `trailing_state_signal`, `evaluate_hybrid_optuna_portfolio`, `aggregate_raw_aggtrades_to_1s`, `append_ohlcv_1s_wal`, and `build_info` (added 2026-07-03; see the version handshake section below).

| Kernel | Python API | Default decision | Evidence |
| :--- | :--- | :--- | :--- |
| Raw aggTrades → canonical 1s OHLCV | `lumina_quant.data.raw_first_lineage.raw_aggtrades_to_1s_frame` | **Use `lumina_quant._compute` automatically when it is importable** | Local 200k-trade synthetic benchmark (pre-consolidation `native/rust_rawfirst`): Python `0.0496s/eval`, Rust `0.0246s/eval`, about `2.01x` faster with frame parity. |
| Alpha Zoo Optuna hybrid portfolio loop | `scripts/research/run_alpha_zoo_integer_leverage_optuna_hybrid_decision.py::_portfolio_returns_for_params` | **Use `lumina_quant._compute` automatically when importable and input is finite** | Local 20k×3 synthetic benchmark (pre-consolidation `native/rust_hybrid_optuna`): Python `2.5493s/eval`, Rust `0.0107s/eval`, about `238.07x` faster; return diff `1.01e-11`, exposed-weight diff `9.15e-09` within `1e-8` tolerance. |
| Alpha Zoo live state-signal machines | `lumina_quant.alpha_zoo.optuna_hybrid_signals.debounced_state_signal` / `trailing_state_signal` | **Use `lumina_quant._compute` automatically when importable** | Local 50k-row synthetic benchmark (pre-consolidation `native/rust_live_signals`): Python `0.1349s/eval`, Rust `0.000544s/eval`, about `247.80x` faster; debounced/trailing state arrays matched exactly. |
| Alpha Zoo fold-level symbol simulation | `lumina_quant.alpha_zoo.native_alpha_fold_backend.simulate_symbol_arrays` (called from `scripts/research/run_alpha_zoo_clean_new_alpha_discovery.py`) | **Use `lumina_quant._compute` automatically when importable** | Migrated directly into the pyo3 crate (no standalone ctypes predecessor); parity covered by `tests/test_native_alpha_fold_backend.py` and `tests/test_native_backend_parity.py`. |
| Live `MARKET_WINDOW` event construction / rolling aggregation | `lumina_quant.core.market_window_contract.build_market_window_event` / `lumina_quant.live.market_window_rolling.RollingWindowAggregator` | **Do not add Rust for this boundary; use the internal canonical-row fast path** | Profiling showed repeated Python row normalization/schema validation, not numeric compute, dominated the live window path. Local 5-symbol/300s benchmark: generic builder `0.0005508s/eval`, trusted builder `0.000001006s/eval` (`~547x` for the builder) and rolling aggregation `16.5k ticks/s`, RSS `42.5MB`. Rust FFI would still need Python tuple conversion for `MarketWindowEvent`. |
| Optimization metric evaluator | `lumina_quant.optimization.native_backend.evaluate_metrics_backend` (used by `lumina_quant.services.portfolio`) | **Benchmark pyo3 vs. Numba/Python at import time; keep whichever is faster and parity-safe** | Local native-compare benchmark: auto-selected Numba around `11.1k eval/s`; forced pyo3 around `7.9k eval/s`, parity OK but slower on this host, so Numba stays selected by default. |
| Binary WAL 1s OHLCV append | `lumina_quant.storage.wal.native_backend.append_ohlcv_frame_native` (used by `ParquetOhlcvRepository.upsert_1s`) | **Always try the pyo3 kernel first; fall back to the pure-Python `BinaryWAL.append` transparently** | No backend-selection env var: this path has no "force Python" knob because both paths write byte-identical WAL records. |

## Build

```bash
uv run python scripts/build_native_backends.py           # maturin develop --release (editable install)
uv run python scripts/build_native_backends.py --wheel   # maturin build --release (wheel)
```

The script resolves `maturin` from `$VIRTUAL_ENV/bin` (or the repo's `.venv` if `VIRTUAL_ENV` is unset) and runs it inside `native/lumina_compute`. Equivalent direct commands:

```bash
cd native/lumina_compute
cargo build --release      # compile-check only; does not install into the active Python env
maturin develop --release  # build + install as lumina_quant._compute into the active venv
```

Native/parity tests to run after any rebuild:

```bash
PYTHONPATH=. uv run python -m pytest tests -q -k "native or hybrid or compute"
```

## Native kernel version handshake (2026-07-03)

Because `lumina_quant._compute` is an editable extension, the checked-out crate source and the compiled `.so` can drift silently: editing `native/lumina_compute/src/lib.rs` without rerunning `maturin develop` leaves the previously-built kernel loaded with no error. To catch this:

- The crate exposes `build_info()`, returning `env!("CARGO_PKG_VERSION")` at compile time (the version declared in `native/lumina_compute/Cargo.toml`).
- Every `native_*_backend` module that successfully imports `lumina_quant._compute` calls `lumina_quant._native_kernel_version.check_native_kernel_version()` once per process. It reads the expected version straight from `native/lumina_compute/Cargo.toml` and compares it against `build_info()`:
  - Versions match → silent.
  - Versions differ → one `logger.warning("stale native kernel: ...")` suggesting a rebuild (`maturin develop --release` or `scripts/build_native_backends.py`).
  - `build_info()` is missing (a `.so` built before this handshake existed) → one warning noting the loaded kernel predates the handshake.
  - `native/lumina_compute/Cargo.toml` is not found (e.g. an installed wheel with no checked-out crate source) → skipped silently; there is nothing to compare against.
- The check never raises. It is purely diagnostic and is exercised in isolation by `tests/test_native_kernel_version.py` (pure `compare_native_version(expected, reported) -> ok|stale|missing`).

## Runtime controls

Alpha Zoo fold-level symbol simulation:

- `LQ_ALPHA_FOLD_BACKEND=auto` (default): use the pyo3 kernel when `lumina_quant._compute` is importable, otherwise the Python path.
- `LQ_ALPHA_FOLD_BACKEND=rust`: require the pyo3 kernel; raise if unavailable.
- `LQ_ALPHA_FOLD_BACKEND=python`: force the Python path.
- `LQ_ALPHA_FOLD_DLL`: retained only for env-var backward compatibility; ignored.

Alpha Zoo Optuna hybrid portfolio backend:

- `LQ_HYBRID_OPTUNA_BACKEND=auto` (default): use the pyo3 kernel when importable and the return matrix is finite, otherwise Python.
- `LQ_HYBRID_OPTUNA_BACKEND=rust`: require the pyo3 kernel; raise if unavailable.
- `LQ_HYBRID_OPTUNA_BACKEND=python`: force the original Python loop.
- `LQ_HYBRID_OPTUNA_DLL`: retained only for env-var backward compatibility; ignored.
- In `auto` mode, an unavailable kernel or a failed call now logs one warning per distinct reason (`_warn_once`, 2026-07-03 audit fix) instead of silently degrading to the much slower Python path.

Live Alpha Zoo state-signal backend:

- `LQ_LIVE_SIGNAL_BACKEND=auto` (default) / `rust` / `python`, same semantics as above.
- `LQ_LIVE_SIGNAL_BACKEND_DLL`: retained only for env-var backward compatibility; ignored.
- `auto` mode logs one warning per distinct fallback reason (`_warn_auto_fallback_once`, 2026-07-03 audit fix) instead of silently falling back.

Raw aggTrades → 1s OHLCV backend:

- `LQ_RAW_FIRST_BACKEND=auto` (default) / `rust` / `python`, same semantics as above.
- `LQ_RAW_FIRST_BACKEND_DLL`: retained only for env-var backward compatibility; ignored.
- `auto` mode logs one warning per distinct fallback reason (2026-07-03 audit fix) instead of silently falling back.

Binary WAL append:

- No backend-selection env var. `ParquetOhlcvRepository.upsert_1s` always calls `append_ohlcv_frame_native` first and falls back to the pure-Python `BinaryWAL.append` transparently (no warning) whenever the extension is unavailable or the call raises.

Optimization metric evaluator:

- `LQ_NATIVE_BACKEND=auto` (default): benchmark the pyo3 kernel against Numba/Python at import time (`_select_fastest_backend`) and keep whichever is faster and parity-safe.
- `LQ_NATIVE_BACKEND=native|numba|python`: force a backend for diagnostics.
- `LQ_NATIVE_AUTO_SELECT=0|false`: skip the speed comparison and use the pyo3 kernel unconditionally once parity is confirmed.
- `LQ_NATIVE_MIN_SPEEDUP`: minimum pyo3-over-fallback speed ratio required for auto-selection to prefer pyo3.
- `LQ_NATIVE_BENCH_LOOPS`, `LQ_NATIVE_SELECTION_TOL`: tune the startup benchmark's loop count and output-parity tolerance.
- This module does not warn on fallback — the benchmark-based selector simply keeps whichever backend it picked at import time.

## Benchmark scripts

Benchmark raw-first Python vs. Rust:

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

Benchmark Optuna hybrid portfolio loop Python vs. Rust:

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

Benchmark live state-signal Python vs. Rust:

```bash
uv run python scripts/benchmark_live_signal_backend.py \
  --rows 50000 \
  --evals 20 \
  --backend both \
  --require-rust
```

Benchmark live `MARKET_WINDOW` construction/rolling aggregation:

```bash
uv run python scripts/benchmark_market_window_contract.py \
  --symbols 5 \
  --window-seconds 300 \
  --ticks 5000 \
  --evals 500
```

Benchmark metric evaluator backend selection:

```bash
uv run python scripts/benchmark_native_compare.py --bars 50000 --evals 5000
```

Latest local evidence is stored under `var/reports/native_acceleration_20260527/`.

## Promotion rule

A kernel may be migrated to Rust only when all of the following hold:

1. Python wrapper/API stays stable.
2. Python-vs-Rust parity is tested on deterministic inputs.
3. Local benchmark shows material speedup on the maintained environment.
4. The benchmark stays under the 8GB memory budget.
5. Research/live safety semantics are unchanged (`ready_for_real=false` and `real_money_execution=false` remain hard-false where artifacts require them).

If a Rust backend loses to an existing vectorized/Numba path, keep it optional and let the Python selector choose the faster backend.
