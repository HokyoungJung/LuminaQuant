# Final CI summary — Strategy Performance Improvement

Generated: `2026-07-08T11:48:05Z`

## Decision

Local CI-equivalent gates are green. The full-universe walk-forward now completed against existing local direct 1m-derived `data/market_parquet` data: `110/110` requested symbols loaded, `11` folds completed, and `1733` fold-candidate rows evaluated. This is a measurement artifact only: no real-money, paper, testnet, live, or order execution was run, and no clean/live promotion is approved.

## Current checks

- `uv run --extra dev ruff format --check .`: **PASS** — `1033 files already formatted`
- `uv run --extra dev ruff check .`: **PASS** — `All checks passed!`
- `uv run --extra dev pytest -q tests/test_strategy_performance_improvement_report.py tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py tests/integration/test_engine_golden.py tests/test_research_selection_flags_config.py` with local `.env` temporarily hidden: **PASS** — `70 passed in 33.53s`
- `uv run --extra dev pytest -q` with local `.env` temporarily hidden: **PASS** — `3574 passed, 20 skipped, 3 xfailed in 201.01s`
- `uv run python scripts/build_native_backends.py`: **PASS** — `lumina_compute` built successfully for CPython 3.14, source hash `729fce3d4dca4fab`
- `uv run --extra dev pytest -q tests/test_native_backend_parity.py tests/test_native_kernel_version.py`: **PASS** — `32 passed in 0.24s`
- Full-universe WF command with `--source-symbol-workers 2 --checkpoint-interval 0 --checkpoint-markdown-interval 0`: **PASS** — `11` folds, `110/110` loaded symbols, peak RSS `1834.566 MiB`
- Normalized WF report build: **PASS** — `full_universe_claim_status=claimed_loaded_all_requested_symbols_completed_walkforward`; leak checks pass with zero locked-OOS selection/weighting/sizing/tiebreak use

## Prior CI evidence retained

- Dashboard gates: `npm install`, `lint`, `test`, `typecheck`, `build`: **PASS** (`60` dashboard tests)
- Coverage gates: **PASS** — total `79%`, financial core `83% >= 70%`, live/exchanges/core `70% >= 65%`
- Architecture, hardcoded-param, docs, Binance-native, benchmark/8GB gates: **PASS**
- Benchmark/8GB: median `0.05218s`, `12149` bars/sec, peak RSS `254.19 MiB` under the 8GB gate

## Claims

- Strategy candidate status: research/shadow-only; no clean/live promotion.
- Full-universe WF status: completed on existing local data, but output remains diagnostic/report-only and non-deployable.
- Raw WF `candidate_tier` labels may contain legacy paper/testnet/live words; artifact-level promotability and this final summary are authoritative and do not approve paper/testnet/live/real-money execution.
- Native/Rust status: existing pyo3/native backend built and parity/version tests passed; no new Rust hot-path code was introduced because no measured bottleneck required it.
