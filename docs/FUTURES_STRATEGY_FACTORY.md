# Futures Strategy Factory (Binance USDT-M + XAU/XAG)

`scripts/run_research_hurdle.py` builds a large strategy-candidate universe and
produces a diversified shortlist from existing research reports.

## What it produces

1. **Candidate universe** (JSON): parameterized strategy specs across:
   - `TopCapTimeSeriesMomentumStrategy`
   - `RollingBreakoutStrategy`
   - `MeanReversionStdStrategy`
   - `VwapReversionStrategy`
   - `RsiStrategy`
   - `MovingAverageCrossStrategy`
   - `PairTradingZScoreStrategy`
   - `LagConvergenceStrategy`
2. **Shortlist JSON + CSV** with family/timeframe/symbol diversification.
   - weak single-strategy rows can be filtered by score/return/sharpe/trades floors
   - direct multi-asset rows can be excluded (default in pipeline mode)
3. **Markdown report** with summary mix + reusable commands.
4. **Portfolio sets**: combinations of successful single-asset strategies with normalized `portfolio_weight`.

## Fast indicator inputs used for ranking bias

The shortlist ranker applies a lightweight regime bias using:

- `rolling_log_return_volatility_latest`
- `normalized_true_range_latest`
- `volume_shock_zscore_latest`
- `trend_efficiency_latest`

All are implemented in `lumina_quant/indicators/futures_fast.py` with optional Numba kernels.

## Usage

### Build shortlist from latest team research reports

```bash
uv run python scripts/run_research_hurdle.py \
  --mode oos \
  --report-glob "reports/strategy_team_research_oos_*.json" \
  --score-config configs/score_config.example.json \
  --max-report-files 20 \
  --max-shortlist 64
```

### Dry-run (no files written)

```bash
uv run python scripts/run_research_hurdle.py --dry-run
```

### Build weighted shortlist from research reports

```bash
uv run python scripts/select_research_shortlist.py \
  --report-glob "reports/oos_guarded_multistrategy_oos_*.json" \
  --mode oos \
  --score-config configs/score_config.example.json \
  --single-min-score 0.0 \
  --single-min-return 0.0 \
  --single-min-sharpe 0.7 \
  --single-min-trades 20 \
  --min-trades 5 \
  --max-selected 32
```

By default this keeps a single-asset-first portfolio construction path and emits `portfolio_sets`.
Use `--allow-multi-asset` only if you explicitly want direct multi-asset rows in shortlist.

## Scoring config template

Use `configs/score_config.example.json` as a base template.

- `strategy_shortlist` section → `scripts/select_research_shortlist.py`
- `research_hurdle` section → `scripts/run_research_hurdle.py`
- The same file can also be reused for:
  - `candidate_research` (`scripts/run_research_candidates.py`)
  - `portfolio_optimization` (`scripts/run_portfolio_optimization.py`)

### Custom symbol/timeframe set

```bash
uv run python scripts/run_research_hurdle.py \
  --symbols BTC/USDT ETH/USDT BNB/USDT XAU/USDT XAG/USDT \
  --timeframes 1s 1m 5m 15m 1h 4h 1d
```

## Notes

- Defaults target the mission scope: **timeframes `1s~1d`**, **top10 + XAU/XAG**.
- If no report exists, the script still emits a factory-seeded shortlist.
- Use `scripts/run_research_pipeline.py` or `scripts/run_bulk_research.py` first for richer shortlist scoring.
