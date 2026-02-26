# Worker-3 Strategy Factory Report

## Scope delivered

This contribution adds a worker-3 strategy-factory bundle focused on Binance futures candidate generation and shortlist selection across **1s ~ 1d** timeframes for **top-10 market-cap crypto + XAU/XAG**.

## Deliverables

### 1) Strategy candidate set (multiple files)

- `lumina_quant/strategies/candidate_regime_breakout.py`
  - `RegimeBreakoutCandidateStrategy`
  - Regime-aware breakout logic using trend slope, range position, volatility gating.
- `lumina_quant/strategies/candidate_vol_compression_reversion.py`
  - `VolatilityCompressionReversionStrategy`
  - Volatility-compression mean-reversion logic using z-score + volatility ratio.
- `lumina_quant/strategies/factory_candidate_set.py`
  - Deterministic candidate-universe builder for top-cap symbols/timeframes.
  - Includes both new candidate strategies and existing families (`RollingBreakoutStrategy`, `TopCapTimeSeriesMomentumStrategy`).

### 2) Fast indicator/operator module

- `lumina_quant/indicators/factory_fast.py`
  - Optional Numba acceleration where available.
  - Adds:
    - `rolling_slope_latest`
    - `rolling_range_position_latest`
    - `volatility_ratio_latest`
    - `composite_momentum_latest`

### 3) Tuning/selection pipeline scripts

- `scripts/export_strategy_factory_candidates.py`
  - Generates strategy candidate universe JSON.
- `scripts/select_strategy_factory_shortlist.py`
  - Aggregates multi-report candidates and outputs a diversified shortlist.
  - Supports caps by strategy/timeframe/symbol and pass/trade filters.

### 4) Tests for new logic

- `tests/test_factory_fast_indicators.py`
- `tests/test_factory_candidate_set.py`
- `tests/test_candidate_strategies_worker3.py`
- `tests/test_strategy_factory_shortlist_script.py`

## Usage commands

### Export candidate universe

```bash
uv run python scripts/export_strategy_factory_candidates.py --pretty
```

### Build a shortlisted portfolio candidate file from prior OOS reports

```bash
uv run python scripts/select_strategy_factory_shortlist.py \
  --report-glob "reports/oos_guarded_multistrategy_oos_*.json" \
  --mode oos \
  --max-selected 32 \
  --max-per-strategy 8 \
  --max-per-timeframe 6 \
  --max-per-symbol 4 \
  --min-trades 1 \
  --pretty
```

### Verify

```bash
uv run ruff check \
  lumina_quant/indicators/factory_fast.py \
  lumina_quant/strategies/candidate_regime_breakout.py \
  lumina_quant/strategies/candidate_vol_compression_reversion.py \
  lumina_quant/strategies/factory_candidate_set.py \
  scripts/export_strategy_factory_candidates.py \
  scripts/select_strategy_factory_shortlist.py \
  tests/test_factory_fast_indicators.py \
  tests/test_factory_candidate_set.py \
  tests/test_candidate_strategies_worker3.py \
  tests/test_strategy_factory_shortlist_script.py

uv run pytest -q \
  tests/test_factory_fast_indicators.py \
  tests/test_factory_candidate_set.py \
  tests/test_candidate_strategies_worker3.py \
  tests/test_strategy_factory_shortlist_script.py
```
