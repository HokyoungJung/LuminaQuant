# Worker-4 Strategy Factory Report

## Scope Delivered

This contribution adds an isolated **strategy factory pipeline** focused on Binance futures research over:

- Timeframes: `1s, 1m, 5m, 15m, 30m, 1h, 4h, 1d`
- Universe: top 10 market-cap crypto + `XAU/USDT`, `XAG/USDT`
- Families: trend, mean reversion, breakout, market-neutral pair trading

## New Deliverables

### 1) Strategy candidate set (multi-file)

- `lumina_quant/strategy_factory/candidate_library.py`
  - Generates a large, diversified candidate universe for:
    - `RsiStrategy`
    - `MovingAverageCrossStrategy`
    - `MeanReversionStdStrategy`
    - `RollingBreakoutStrategy`
    - `VwapReversionStrategy`
    - `TopCapTimeSeriesMomentumStrategy`
    - `PairTradingZScoreStrategy`
  - Produces JSON-ready manifest with family/strategy/timeframe counts.

### 2) Fast indicator/operator module

- `lumina_quant/indicators/fast_ops.py`
  - Numba-compatible rolling z-score kernel
  - Numba-compatible rolling percentile-rank kernel
  - Cross-sectional rank normalization
  - Volatility-adjusted momentum score helper

### 3) Tuning/selection pipeline for multi-timeframe and multi-symbol evaluation

- `scripts/run_strategy_factory_pipeline.py`
  - Writes candidate manifest to `reports/`
  - Produces a local report payload and shortlist candidates (no external orchestrator required)
  - Builds diversified portfolio shortlist from selected team output
  - Emits JSON + Markdown shortlist artifacts

- `lumina_quant/strategy_factory/pipeline.py`
  - Report-path extraction
  - Research command assembly
  - Shortlist payload builder
  - Markdown summary renderer

- `lumina_quant/strategy_factory/selection.py`
  - Hurdle scoring
  - Identity dedupe
  - Family inference
  - Diversified shortlist selection constraints

### 4) Documentation and usage commands

- This report file documents workflow, files, and commands.

### 5) Tests for new logic

- `tests/test_strategy_factory_candidate_library.py`
- `tests/test_strategy_factory_selection.py`
- `tests/test_fast_ops.py`
- `tests/test_strategy_factory_pipeline.py`

## Usage

Generate candidate universe only:

```bash
python scripts/run_strategy_factory_pipeline.py --dry-run
```

Run full pipeline and create shortlist artifacts:

```bash
python scripts/run_strategy_factory_pipeline.py \
  --db-path data/market_parquet \
  --exchange binance \
  --market-type future \
  --mode oos \
  --strategy-set all
```

Control shortlist size/diversification:

```bash
python scripts/run_strategy_factory_pipeline.py \
  --shortlist-max-total 24 \
  --shortlist-max-per-family 8 \
  --shortlist-max-per-timeframe 6
```

## Notes

- The pipeline intentionally reuses existing OOS team-research tooling for robust evaluation.
- Added modules are isolated to reduce collision risk with concurrent worker edits.
