# Final Validation vs Continuity Validation

## Canonical market-data lineage

LuminaQuant now treats **Binance USDⓈ-M Futures aggTrades** as the single canonical raw market-data source.

Required lineage:
- raw aggTrades
- deterministic 1s OHLCV derived from raw aggTrades
- higher timeframes derived by deterministic resampling from real lower-timeframe data
- incomplete last buckets dropped
- no synthetic data in final validation
- no kline endpoint as canonical validation truth

## Historical + live data policy

- Historical backfills use Binance Futures public aggTrades archives when available.
- Recent gaps use native Binance Futures REST aggTrades.
- Live tails use native Binance Futures aggTrade streams.
- Live trading / balances / positions / orders / user stream use native Binance USDⓈ-M Futures APIs only.
- CCXT is removed from production-critical Binance runtime and historical data paths.

## Validation artifacts

### Continuity validation

Continuity validation is the **extension-style monitoring artifact**.

Use it to answer:
- Did the incumbent remain directionally consistent after the previously saved OOS end?
- Did the extension window introduce a short-term stability break?

Continuity validation is not the final sign-off artifact.

### Final exact-window validation

Final validation is the **latest-real-complete-candle anchored source of truth**.

Required behavior:
- discover all required symbol/timeframe/feature inputs for the incumbent portfolio
- find the latest common timestamp supported by real data only
- anchor validation to a complete bar for every required input
- rebuild the original train/val/oos window by trimming from the left
- rerun the incumbent portfolio on that exact shifted window
- report true portfolio return-stream metrics separately from weighted component summaries
- fail loudly if required real inputs are missing

## Metric semantics

`portfolio_metrics` are reserved for true portfolio return-stream metrics, such as:
- total_return
- cagr
- sharpe
- sortino
- calmar
- max_drawdown
- volatility

`weighted_component_summaries` contain weighted component aggregates that are **not** true portfolio-return-derived metrics, such as:
- trade_count
- turnover
- benchmark_corr

## Risk-free reference

Reporting supports:
- `RISK_FREE_MODE = zero | us_treasury_constant | us_treasury_series`
- `RISK_FREE_TENOR = 3m`
- `RISK_FREE_ANNUAL = <float>`
- `SORTINO_TARGET_MODE = zero | same_as_rf | explicit`
- `SORTINO_TARGET_ANNUAL = <float>`

Default reporting should use the **US 3-month Treasury bill** reference unless a workflow explicitly requires zero-rate comparability.
