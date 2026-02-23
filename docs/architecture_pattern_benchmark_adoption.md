# Architecture Pattern Benchmark Adoption (LuminaQuant)

This document tracks high-performance engine patterns adopted in LuminaQuant, with measurable checks.

## Adopted patterns

1. Deterministic event identity (`timestamp_ns` + `sequence`)
   - Files: `lumina_quant/events.py`, `lumina_quant/event_clock.py`, `lumina_quant/engine.py`
   - Check: replay ordering tests and monotonic assertions.

2. In-process message bus (publish/subscribe, request/response, point-to-point)
   - Files: `lumina_quant/message_bus.py`, `lumina_quant/engine.py`
   - Check: `tests/test_message_bus.py` and bus throughput in architecture benchmark.

3. Central runtime cache + outbox persistence for live recovery
   - Files: `lumina_quant/runtime_cache.py`, `lumina_quant/live_trader.py`
   - Check: `tests/test_runtime_cache.py` and restart-safe state snapshot path.

4. One-time dataset build and frozen arrays for optimization
   - Files: `lumina_quant/optimization/frozen_dataset.py`, `optimize.py`, `lumina_quant/data.py`
   - Check: no Polars frame allowed in trial loop (`optimize.py` hard guard), pre-frozen tuple rows.

5. Compiled evaluation kernel path (Numba/native)
   - Files: `lumina_quant/optimization/fast_eval.py`, `lumina_quant/optimization/native_backend.py`, `lumina_quant/services/portfolio.py`
   - Check: `tests/test_fast_eval.py`, `tests/test_parity_fast_eval.py`, kernel throughput benchmark.

6. Oversubscription control for Polars/Numba phases
   - Files: `lumina_quant/optimization/threading_control.py`, `optimize.py`
   - Check: startup log line for configured Numba threads.

7. Modular execution simulation models
   - Files: `lumina_quant/execution.py`
   - Check: protective order and execution state-machine regression tests.

8. Two-stage optimizer (fast prefilter -> full event-driven replay)
   - Files: `optimize.py`
   - Check: train-phase logs include `[Two-Stage]` prefilter/replay messages.

## Runtime activation

```bash
uv sync --extra optimize --extra dev --extra live
winget install --id Rustlang.Rustup -e --accept-source-agreements --accept-package-agreements --disable-interactivity
export PATH="$PATH:/c/Users/<user>/.cargo/bin"
```

Default backend behavior:

- `lumina_quant/optimization/native_backend.py` benchmarks available candidates (`numba/python`, C DLL, Rust DLL) and selects the fastest backend by default.
- Native candidates are only accepted when output parity matches baseline within tolerance.
- Environment controls:
  - `LQ_NATIVE_BACKEND` (`auto` default): force backend (`auto|python|numba|native`).
  - `LQ_NATIVE_AUTO_SELECT` (`1` default): enable/disable automatic fastest selection.
  - `LQ_NATIVE_MIN_SPEEDUP` (`0.0` default): minimum relative speedup required for native takeover.
  - `LQ_NATIVE_BENCH_LOOPS` (`256` default): micro-benchmark loop count.
  - `LQ_NATIVE_METRICS_DLL`: explicit DLL override.

## Cross-platform validation

- GitHub Actions workflow: `.github/workflows/cross-platform-ci.yml`
- Matrix targets: Windows, Ubuntu, macOS
- Validation steps: `uv sync`, native backend build, lint, regression tests

## Native build commands

```bash
uv run python scripts/build_native_backends.py --backend all
native\c_metrics\build_msvc_x64.bat
native\rust_metrics\build_release.bat
```

## Benchmark commands

```bash
uv run python scripts/benchmark_dataset_build.py --symbols 6 --rows 50000
uv run python scripts/benchmark_optimization_kernel.py --bars 10000 --evals 1000
uv run python scripts/benchmark_architecture.py --bars 20000 --evals 3000 --messages 200000 --updates 100000 --events 100000
uv run python scripts/benchmark_native_compare.py --bars 50000 --evals 5000
```

## Latest benchmark snapshot (local)

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

## Two-stage optimization profile snapshot

Command:

```bash
LQ_OPT_PROFILE=1 LQ_TWO_STAGE_OPT=1 LQ_TWO_STAGE_TOPK_RATIO=0.5 LQ_TWO_STAGE_PREFILTER_FRACTION=0.4 LQ_MIN_TRAIN_DAYS=1 \
uv run python optimize.py --folds 1 --n-trials 6 --max-workers 2 --oos-days 1 --data-source csv --no-auto-collect-db
```

Result highlights:

- Two-stage prefilter and replay path executed (`[Two-Stage]` logs present).
- Fallback replay safety path executed when multiprocess replay returned no valid rows.
- Runtime profile:
  - `feature/indicator: 0.0241s`
  - `simulation core: 0.0714s`
  - `orchestration: 0.0738s`
  - `avg simulation/call: 0.004762s`

## Verification commands

```bash
uv run pytest tests/test_message_bus.py tests/test_runtime_cache.py tests/test_replay.py
uv run pytest tests/test_parity_fast_eval.py tests/test_fast_eval.py tests/test_frozen_dataset.py tests/test_event_clock.py tests/test_system_assembly.py tests/test_portfolio_fast_stats.py
uv run pytest tests/test_execution_protective_orders.py tests/test_live_execution_state_machine.py tests/test_lookahead.py
uv run pytest tests/test_optimize_two_stage.py tests/test_native_backend.py tests/test_strategy_registry_defaults.py
uv run ruff check lumina_quant optimize.py strategies/registry.py strategies/rsi_strategy.py strategies/moving_average.py scripts/benchmark_architecture.py scripts/benchmark_dataset_build.py scripts/benchmark_native_compare.py tests/test_message_bus.py tests/test_runtime_cache.py tests/test_replay.py tests/test_optimize_two_stage.py
```
