# Quant Research Factory (Binance Futures, 1s~1d, Top10 + XAU/XAG)

This workflow is designed to create and evaluate a **large strategy candidate universe** quickly.

## Scope

- Venue: Binance futures
- Symbols: Top 10 crypto majors + XAU/XAG
- Timeframes: 1s, 1m, 5m, 15m, 30m, 1h, 4h, 1d
- Data source: existing market DB (`data/lq_market.sqlite3` by default)

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
  --db-path data/lq_market.sqlite3 \
  --market-type future \
  --mode oos \
  --strategy-set all
```

Output defaults to:

- `reports/strategy_factory/strategy_factory_candidates_*.json`
- `reports/strategy_factory/strategy_factory_shortlist.json`
- `reports/strategy_factory/strategy_factory_shortlist.md`

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
