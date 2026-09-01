# Derivatives Flow Squeeze test report

- generated_at: `2026-05-02T12:24:34.537768Z`
- decision: `failed_live_equivalent_gate_not_deployment_ready`
- mode: `derivatives_flow_squeeze_mode`
- latest evidence: `var/reports/derivatives_flow_squeeze_20260502/live_equivalent/derivatives_flow_squeeze_mode_v6_latest_data/live_equivalent_revalidation_latest.json`

## Gate result

- train return `-9.0995%` vs floor `>-2.5%` / active alpha floor `>-3.0%`: **fail**
- val return `-0.5571%` vs baseline `+0.264933%`, boost `+0.509082%`, positive return gate: **fail**
- train MDD `9.4097%`, val MDD `1.9324%`: MDD gate passed but insufficient because return/Sharpe/Sortino failed
- liquidations train/val `0/0`
- alpha blocking reasons: `train_total_return_below_floor;val_total_return_not_positive;val_sharpe_not_positive;val_sortino_not_positive`

## Why not promoted

- Feature-completeness gate failed for the actual alpha thesis: train/val do not have true OI, taker-flow, or liquidation history aligned to timestamps.
- The fallback proxies generated many trades but negative train/val returns under the live-equivalent engine.
- OOS incumbent target from the prompt (8.18% return / 2.761 Sharpe / 4.52% MDD) was not challenged because train/val gate already failed.

## Computation optimization

- Added `--fail-fast-alpha-gate` to skip later splits after train-only alpha-gate failure during research iteration.
- Killed the redundant v6 long replay once train failed, then finalized from checkpoint in 1.48s with max RSS 345,772 KB.

## Sources

- https://link.springer.com/article/10.1007/s11408-025-00474-9 — accessed 2026-05-02; idea: Large-cap cryptocurrency momentum can suffer severe crashes; volatility management can mitigate crash/tail behavior.
- https://www.research.ed.ac.uk/en/publications/anatomy-of-cryptocurrency-perpetual-futures-returns/ — accessed 2026-05-02; idea: Cryptocurrency perpetual futures returns have basis, momentum, volume, size, and volatility predictor families.
- https://www.sciencedirect.com/science/article/pii/S1386418126000029 — accessed 2026-05-02; idea: Order flow has explanatory and predictive power for cryptocurrency returns.
- https://www.sciencedirect.com/science/article/pii/S0378426625000317 — accessed 2026-05-02; idea: Short reversal/liquidity provision premium in crypto is tied to realized variance, crash risk, tail risk, and liquidity conditions.
- https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History — accessed 2026-05-02; idea: USD-M funding history exposes fundingRate, fundingTime, and markPrice.
- https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics — accessed 2026-05-02; idea: Open-interest statistics expose sumOpenInterest and timestamp but API is latest-month limited.
- https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Taker-BuySell-Volume — accessed 2026-05-02; idea: Taker buy/sell volume exposes buyVol, sellVol, ratio, and timestamp but API is latest-30-days limited.
- https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Liquidation-Order-Streams — accessed 2026-05-02; idea: forceOrder stream publishes largest liquidation snapshot per symbol at 1000ms cadence when liquidation occurs.
