# Quant Research Factory (Binance Futures, 1s~1d, Top10 + XAU/XAG)

This workflow is designed to create and evaluate a **large strategy candidate universe** quickly.

## Scope

- Venue: Binance futures
- Symbols: Top 10 crypto majors + XAU/XAG
- Timeframes: 1s, 1m, 5m, 15m, 30m, 1h, 4h, 1d
- Data source: existing market parquet root (`data/market_parquet` by default)

## New Components

- Fast indicator helpers:
  - `lumina_quant/indicators/factory_fast.py`
  - `lumina_quant/indicators/futures_fast.py`
- New strategy candidates:
  - `strategies/candidate_regime_breakout.py`
  - `strategies/candidate_vol_compression_reversion.py`
- Strategy-candidate universe builders:
  - `strategies/factory_candidate_set.py`
  - `lumina_quant/strategy_factory/*`
- Candidate export and pipeline scripts:
  - `scripts/export_strategy_factory_candidates.py`
  - `scripts/run_strategy_factory_pipeline.py`
  - `scripts/run_mass_strategy_research.py` (alias)

## Typical Workflow

### 1) Export large candidate universe

```bash
uv run python scripts/export_strategy_factory_candidates.py \
  --output reports/strategy_factory_candidates.json \
  --pretty
```

### 2) Run full pipeline (manifest -> research -> shortlist)

```bash
uv run python scripts/run_strategy_factory_pipeline.py \
  --db-path data/market_parquet \
  --mode standard \
  --timeframes 1m 5m 15m \
  --seeds 20260221
```

### 2-1) Apply shortlist weighting + single-strategy performance filters

```bash
uv run python scripts/run_strategy_factory_pipeline.py \
  --db-path data/market_parquet \
  --mode standard \
  --timeframes 1m 5m 15m \
  --seeds 20260221 \
  --single-min-score 0.0 \
  --single-min-return 0.0 \
  --single-min-sharpe 0.7 \
  --single-min-trades 20 \
  --drop-single-without-metrics \
  --set-max-per-asset 2 \
  --set-max-sets 16
```

Default single-strategy floor policy: `score >= 0`, `return >= 0`, `sharpe >= 0.7`, `trades >= 20`.

Output defaults to:

- `reports/strategy_factory_candidates_*.json`
- `reports/strategy_factory_report_*.json`
- `reports/strategy_factory_shortlist_*.json`
- `reports/strategy_factory_shortlist_*.md`

Shortlist JSON now includes:
- per-row `portfolio_weight` (unless `--disable-weights`)
- `portfolio_sets` built from successful single-asset rows
- policy metadata (`single_min_*`, `allow_multi_asset`, `portfolio_set_count`, etc.)

### 3) Dry-run only (no heavy search)

```bash
uv run python scripts/run_strategy_factory_pipeline.py --dry-run
```

### 4) Alias command

```bash
uv run python scripts/run_mass_strategy_research.py --dry-run
```

## Notes

- For wide sweeps, increase `--max-runs` and provide multiple `--seeds`.
- Candidate manifest is deterministic for fixed symbols/timeframes.
- Shortlist is diversification-aware (family/timeframe caps).
- Direct multi-asset mixes are excluded by default (enable with `--allow-multi-asset`).
