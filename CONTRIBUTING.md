# Contributing to LuminaQuant

Thanks for contributing.

## 1) Development Setup

```bash
uv sync --extra optimize --extra dev --extra live
```

Optional extras:

```bash
uv sync --extra gpu  # Linux x86_64 + CUDA 12
uv sync --extra mt5  # Windows MT5 bridge
```

## 2) Required Local Checks (CI parity)

Run these before opening a PR:

```bash
uv run ruff check .
uv run python scripts/check_architecture.py
uv run python scripts/audit_hardcoded_params.py
uv run python scripts/verify_docs.py
uv run pytest tests/test_audit_hardcoded_params.py tests/test_compute_engine.py tests/test_timeframe_panel_and_liquidity.py tests/test_native_backend.py tests/test_optimize_two_stage.py tests/test_message_bus.py tests/test_runtime_cache.py tests/test_system_assembly.py tests/test_ohlcv_loader.py tests/test_execution_protective_orders.py tests/test_live_execution_state_machine.py tests/test_lookahead.py tests/test_readme_quickstart_paths.py tests/test_verify_8gb_baseline_script.py
mkdir -p logs reports/benchmarks
/usr/bin/time -v uv run python scripts/benchmark_backtest.py --iters 1 --warmup 0 --output reports/benchmarks/ci_smoke_local.json 2>&1 | tee logs/ci_smoke_local.time.log
uv run python scripts/verify_8gb_baseline.py --benchmark reports/benchmarks/ci_smoke_local.json --time-log logs/ci_smoke_local.time.log --oom-log logs/ci_smoke_local.time.log --skip-dmesg --output reports/benchmarks/ci_8gb_gate_local.json
```

## 3) Minimum Viable No-Infra Smoke Check

```bash
uv run python scripts/minimum_viable_run.py
```

## 4) Scope and Style

- Prefer small, surgical changes over large refactors.
- Keep docs (`README.md`, `README_KR.md`, related runbooks) consistent with runtime behavior.

## 5) Pull Requests

- Fill the PR template checklist.
- Include commands run and short evidence in PR description.
- Highlight any behavior changes in backtest/live safety paths.
