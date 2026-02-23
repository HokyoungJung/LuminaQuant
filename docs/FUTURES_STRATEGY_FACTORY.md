# Futures Strategy Factory (Binance USDT-M + XAU/XAG)

`scripts/futures_strategy_factory.py` builds a large strategy-candidate universe and
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
3. **Markdown report** with summary mix + reusable commands.

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
python scripts/futures_strategy_factory.py \
  --mode oos \
  --report-glob "reports/strategy_team_research_oos_*.json" \
  --max-report-files 20 \
  --max-shortlist 64
```

### Dry-run (no files written)

```bash
python scripts/futures_strategy_factory.py --dry-run
```

### Custom symbol/timeframe set

```bash
python scripts/futures_strategy_factory.py \
  --symbols BTC/USDT ETH/USDT BNB/USDT XAU/USDT XAG/USDT \
  --timeframes 1s 1m 5m 15m 1h 4h 1d
```

## Notes

- Defaults target the mission scope: **timeframes `1s~1d`**, **top10 + XAU/XAG**.
- If no report exists, the script still emits a factory-seeded shortlist.
- Use `scripts/run_strategy_team_research.py` first for richer shortlist scoring.
