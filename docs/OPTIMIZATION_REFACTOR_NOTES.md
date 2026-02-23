# Optimization Refactor Notes

This document captures the latest performance-focused refactor that preserves
existing behavior while reducing runtime and temporary allocations.

## Scope

- Keep event flow and strategy semantics unchanged.
- Keep backtest/live/optimization outputs compatible with existing tests.
- Improve hot paths in data loading, strategy calculations, and portfolio stats.

## Module Boundary Updates

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
- `optimize.py` now uses the same loader for CSV fallback and DB frame
  normalization.

## Performance Changes

### Data handler

- Reduced temporary allocations in tail access:
  - `get_latest_bars(...)`
  - `get_latest_bars_values(...)`
- Avoids full `deque -> list` conversion when only tail values are needed.

### Pair strategy

- `strategies/pair_trading_zscore.py` now keeps incremental return histories.
- Removes repeated full-history return reconstruction in `_vol_spread_zscore()`.
- Uses iterator-based tail aggregation in ATR filter.

### Top-cap momentum strategy

- `strategies/topcap_tsmom.py` now computes BTC regime MA from iterator slices
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

- `51 passed` via `python -m pytest`

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
