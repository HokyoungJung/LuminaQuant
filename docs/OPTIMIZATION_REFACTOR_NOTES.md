# Optimization Refactor Notes

This document captures the latest performance-focused refactor that preserves
existing behavior while reducing runtime and temporary allocations.

## Scope

- Keep event flow and strategy semantics unchanged.
- Keep backtest/live/optimization outputs compatible with existing tests.
- Improve hot paths in data loading, strategy calculations, and portfolio stats.

## Module Boundary Updates

### Shared search policy module

- Added `lumina_quant/optimization/search_policy.py`.
- Purpose: centralize Optuna study creation, Optuna parameter suggestion from
  the canonical config schema, bounded deterministic grid enumeration, and
  optimization-result policy metadata.
- Public helpers:
  - `run_optuna_study(...)`
  - `suggest_params_from_optuna_config(...)`
  - `build_bounded_grid_combinations(...)`
  - `optimization_search_policy_payload(...)`
- Policy:
  - use Optuna for tunable/high-dimensional search;
  - use bounded grid only for small deterministic enumerations with an explicit
    justification and optional cap;
  - keep locked-OOS flags false for selection/objective/pruning/parameter fitting
    unless a diagnostic artifact explicitly labels otherwise.

### Standard live refit policy

- Live-facing Alpha Zoo refits must refresh committed data first, reserve the
  latest 8 complete weeks as validation, tune every exposed strategy-internal
  parameter with Optuna, and then final-refit the selected frozen profile on
  train+validation.
- The selection fit/learn phase uses train only; selection/reporting may inspect
  train+validation. Live final-refit mode intentionally does not reserve or use a
  locked-OOS/test set.
- Warmup remains a train-window ratio and is part of the Optuna search space.
- Grid rows are comparison baselines only, not the default optimizer.
- Real-money flags remain hard-false in research and live handoff artifacts.

### New compute loader module

- Added `lumina_quant/compute/ohlcv_loader.py`.
- Purpose: centralize OHLCV normalization and CSV loading shared by backtest and
  optimization paths.
- Public helpers:
  - `OHLCVFrameLoader`
  - `normalize_ohlcv_frame(...)`
  - `load_csv_ohlcv(...)`

### Updated usage sites

- `lumina_quant/data.py` now uses `OHLCVFrameLoader` for preloaded and CSV data.
- `src/lumina_quant/cli/optimize.py` now uses the same loader for CSV fallback and DB frame
  normalization.
- `src/lumina_quant/cli/optimize.py` now delegates grid combination generation
  and Optuna study creation to `lumina_quant.optimization.search_policy`.
- `scripts/research/optuna_tune_hybrid_online_portfolio.py` now uses
  `run_optuna_study(...)` instead of open-coding a local `create_study(...)`
  loop.

## Performance Changes

### Data handler

- Reduced temporary allocations in tail access:
  - `get_latest_bars(...)`
  - `get_latest_bars_values(...)`
- Avoids full `deque -> list` conversion when only tail values are needed.

### Pair strategy

- `lumina_quant/strategies/pair_trading_zscore.py` now keeps incremental return histories.
- Removes repeated full-history return reconstruction in `_vol_spread_zscore()`.
- Uses iterator-based tail aggregation in ATR filter.

### Top-cap momentum strategy

- `lumina_quant/strategies/topcap_tsmom.py` now computes BTC regime MA from iterator slices
  instead of materializing a full list each call.

### Portfolio and performance stats

- `lumina_quant/portfolio.py`
  - Avoids market-value list allocation when history capture is disabled.
  - Uses vectorized first-valid benchmark price lookup in summary stats.
- `lumina_quant/utils/performance.py`
  - `create_drawdowns(...)` rewritten with NumPy vectorized operations while
    preserving previous output contract.

## Validation Evidence

All tests pass after refactor:

- `51 passed` via `uv run python -m pytest`

Search-policy tranche:

- `uv run pytest -q tests/test_optimization_search_policy.py tests/test_param_registry.py tests/test_portfolio_optimizer_core.py tests/test_strategy_alias_compat.py`
  → `19 passed`
- `uv run ruff check src/lumina_quant/optimization/search_policy.py src/lumina_quant/optimization/__init__.py src/lumina_quant/cli/optimize.py scripts/research/optuna_tune_hybrid_online_portfolio.py tests/test_optimization_search_policy.py`
  → passed

Benchmarks (median of 2 measured iterations, 1 warmup):

- RSI, single symbol (`BTC/USDT`)
  - before: `0.07699s`
  - after: `0.07301s`
  - speedup: `~5.17%` faster (`~1.05x`)

- MA cross, two symbols (`BTC/USDT,ETH/USDT`)
  - before: `0.10927s`
  - after: `0.09783s`
  - speedup: `~10.48%` faster (`~1.12x`)

Benchmark artifacts:

- `reports/benchmarks/baseline_before_refactor.json`
- `reports/benchmarks/baseline_before_refactor_ma2.json`
- `reports/benchmarks/after_refactor_rsi1.json`
- `reports/benchmarks/after_refactor_ma2.json`
