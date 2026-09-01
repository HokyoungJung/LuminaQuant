# Derivatives Flow Squeeze / Exhaustion notes

Updated: 2026-05-02T12:24:34.537529Z

Provided sandbox/Desktop files were not present, so the strategy was reconstructed from the user-supplied spec and wired into the repo as `DerivativesFlowSqueezeStrategy` plus `derivatives_flow_squeeze_mode`.

## Decision

**Not deployment-ready.** The new alpha family is implemented and live-equivalent tested, but failed the gate:

- train total return: `-9.0995%`; train MDD `9.4097%`; train trades `9382`
- val total return: `-0.5571%`; val MDD `1.9324%`; val Sharpe `-0.012084293944808604`; val Sortino `-0.01056696301591784`; val trades `1522`
- blocking reasons: `train_total_return_below_floor;val_total_return_not_positive;val_sharpe_not_positive;val_sortino_not_positive`
- incumbent targets were not beaten: user pair-spread OOS return 8.18% / Sharpe 2.761 / MDD 4.52%, profit moonshot baseline val +0.264933%, boost val +0.509082%.

## Source-backed idea ledger

- https://link.springer.com/article/10.1007/s11408-025-00474-9 (accessed 2026-05-02; Published 2025-03-27)
  - Core idea: Large-cap cryptocurrency momentum can suffer severe crashes; volatility management can mitigate crash/tail behavior.
  - Repo variant: Per-symbol realized-volatility multiplier cuts allocation in high realized-volatility windows.
  - Failure addressed: Adaptive/boost fragility and high-volatility crash sizing.
- https://www.research.ed.ac.uk/en/publications/anatomy-of-cryptocurrency-perpetual-futures-returns/ (accessed 2026-05-02; Research output page accessed 2026-05-02)
  - Core idea: Cryptocurrency perpetual futures returns have basis, momentum, volume, size, and volatility predictor families.
  - Repo variant: Funding crowding filter plus OI/volume expansion and price-volume flow continuation.
  - Failure addressed: Price-only breakout false positives and missing derivatives carry context.
- https://www.sciencedirect.com/science/article/pii/S1386418126000029 (accessed 2026-05-02; Available online 2026-01-15)
  - Core idea: Order flow has explanatory and predictive power for cryptocurrency returns.
  - Repo variant: Continuation entries require taker buy/sell imbalance; current historical run uses an explicit OHLCV proxy because true taker fields are not materialized.
  - Failure addressed: Breakouts with no flow confirmation.
- https://www.sciencedirect.com/science/article/pii/S0378426625000317 (accessed 2026-05-02; Journal of Banking & Finance, June 2025)
  - Core idea: Short reversal/liquidity provision premium in crypto is tied to realized variance, crash risk, tail risk, and liquidity conditions.
  - Repo variant: Liquidation exhaustion sleeve fades only after forced-flow shock and price reclaim, under volatility-managed sizing.
  - Failure addressed: Naive shock fade/reversion losses.
- https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History (accessed 2026-05-02; Binance docs copyright 2026)
  - Core idea: USD-M funding history exposes fundingRate, fundingTime, and markPrice.
  - Repo variant: Strategy consumes feature_points funding_rate through timestamp-aligned lookup.
  - Failure addressed: Crowding/funding overheat filter for continuation and exhaustion entries.
- https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics (accessed 2026-05-02; Binance docs copyright 2026)
  - Core idea: Open-interest statistics expose sumOpenInterest and timestamp but API is latest-month limited.
  - Repo variant: Strategy supports open_interest expansion; train/val use quote-volume proxy because historical true OI is absent before 2026-03.
  - Failure addressed: False breakouts without participation expansion.
- https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Taker-BuySell-Volume (accessed 2026-05-02; Binance docs copyright 2026)
  - Core idea: Taker buy/sell volume exposes buyVol, sellVol, ratio, and timestamp but API is latest-30-days limited.
  - Repo variant: Feature schema now has taker buy/sell base/quote columns; backtest flags true-taker absence and uses OHLCV directional proxy only as research fallback.
  - Failure addressed: Need timestamp-aligned order-flow confirmation.
- https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Liquidation-Order-Streams (accessed 2026-05-02; Binance docs copyright 2026)
  - Core idea: forceOrder stream publishes largest liquidation snapshot per symbol at 1000ms cadence when liquidation occurs.
  - Repo variant: Strategy supports liquidation_long/short notional shocks, but current feature store has zero liquidation rows so the sleeve cannot be deployment-proven.
  - Failure addressed: Need forced-flow exhaustion confirmation instead of naive reversal.

## Implementation summary

- `dfse_top5_exhaustion_plus_flow` weight `0.55`: `DerivativesFlowSqueezeStrategy` over `BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, TRX/USDT`.
- `dfse_fast_liquidation_reversal` weight `0.25`: `DerivativesFlowSqueezeStrategy` over `BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, TRX/USDT`.
- `dfse_basis_flow_continuation` weight `0.2`: `DerivativesFlowSqueezeStrategy` over `BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, TRX/USDT`.
- The strategy combines price impulse, taker-flow/proxy imbalance, OI/volume expansion, funding crowding filter, liquidation-exhaustion confirmation, and volatility-managed sizing.
- Feature schema now includes taker buy/sell base/quote volume fields for future timestamp-aligned materialization.
- `--fail-fast-alpha-gate` was added to the live-equivalent revalidation script to avoid wasting full val/OOS replay once train-only promotion gates already fail.

## Latest data refresh

- `BTC/USDT` OHLCV: `2026-05-02T06:46:32Z` -> `2026-05-02T11:44:59Z`, live rows `66833`, derived 1s `17907`
- `ETH/USDT` OHLCV: `2026-05-02T06:46:32Z` -> `2026-05-02T11:44:59Z`, live rows `54095`, derived 1s `17907`
- `BNB/USDT` OHLCV: `2026-05-02T06:46:32Z` -> `2026-05-02T11:44:59Z`, live rows `20026`, derived 1s `17907`
- `SOL/USDT` OHLCV: `2026-05-02T06:46:32Z` -> `2026-05-02T11:44:59Z`, live rows `31553`, derived 1s `17907`
- `TRX/USDT` OHLCV: `2026-05-02T06:46:32Z` -> `2026-05-02T11:44:59Z`, live rows `16816`, derived 1s `17907`

## Feature coverage caveat

- Funding/mark/index support is present.
- True OI starts only around 2026-03-07/08, after train and val windows, so train/val used quote-volume proxy for OI expansion.
- Historical taker buy/sell fields are schema-supported but not materialized; train/val used OHLCV directional proxy.
- Liquidation rows are zero in current support inventory, so the liquidation-exhaustion sleeve cannot be promotion-proven.

## Backtest iteration log

| Run | Status | Train return | Val return | Train trades | Val trades | Max RSS KB | Note |
|---|---|---:|---:|---:|---:|---:|---|
| `derivatives_flow_squeeze_mode` | `live_equivalent_validated` | `0.0000%` | `0.0000%` | `0` | `0` | `7172652` | Initial true-OI/taker/liquidation version: no train/val trades because true OI was unavailable in those splits. |
| `derivatives_flow_squeeze_mode_v2_proxy_oi` | `live_equivalent_validated` | `0.0000%` | `0.0000%` | `0` | `0` | `7172320` | Allowed volume/OI proxy but still no trades because window aggregation and zero feature fallback were incomplete. |
| `derivatives_flow_squeeze_mode_v3_window_proxy` | `live_equivalent_validated` | `0.0000%` | `0.0000%` | `0` | `0` | `7151524` | Fixed MARKET_WINDOW aggregation but zero-valued OI/taker features still blocked fallback. |
| `derivatives_flow_squeeze_mode_v4_zero_feature_fallback` | `aborted_partial` | `n/a` | `n/a` | `None` | `None` | `None` | Enabled zero-feature fallback; trades appeared but sizing was too aggressive and partial run showed train loss-circuit fragility, so it was stopped. |
| `derivatives_flow_squeeze_mode_v5_sized_floor` | `live_equivalent_validated` | `-9.0995%` | `-0.5571%` | `9382` | `1522` | `7179756` | Full live-equivalent run after sizing floor and risk clamps. |
| `derivatives_flow_squeeze_mode_v6_latest_data` | `live_equivalent_validated` | `-9.0995%` | `-0.5571%` | `9382` | `1522` | `345772` | Latest-data rerun after OHLCV/feature refresh to 2026-05-02T11:45Z; train/val reused equivalent checkpoint and matched v5. |

## Failure analysis

1. The intended derivatives-flow edge was not actually observable in train/val: true taker-flow and liquidation data were missing, and true OI begins only after validation.
2. With proxies enabled, the strategy overtraded noisy 1s OHLCV pressure: train trades 9,382 and val trades 1,522, but returns were negative after live-equivalent fills/slippage.
3. The volatility governor controlled drawdown enough to pass MDD caps, but not enough to turn the proxy signal positive.
4. Therefore this is a useful implementation scaffold and data-gap diagnostic, not a promoted alpha.

## Next alpha family

Before retesting Derivatives Flow Squeeze for promotion, materialize real timestamp-aligned taker imbalance from Binance aggTrades and liquidation shocks from forceOrder data. Then test a narrower liquidation-exhaustion-only sleeve with stricter reclaim/hold filters. Do not treat OHLCV proxy flow or volume-proxy OI as deployment evidence.
