# Profit Moonshot external alpha screen

- generated_at: `2026-05-05T08:59:15.759571+00:00`
- data_window: `2025-01-01` → `2026-05-04`
- roundtrip_cost_assumption: `0.001800`
- peak_rss_mib: `2619.11`
- process_rule: no full backtest unless the cheap raw-first screen survives train/val/OOS gates.

## External basis used
- [A Seesaw Effect in the Cryptocurrency Market](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3465924): Cross-cryptocurrency intraday return predictability and lead-lag/seesaw effects.
- [Anatomy of cryptocurrency perpetual futures returns](https://www.research.ed.ac.uk/en/publications/anatomy-of-cryptocurrency-perpetual-futures-returns/): Perpetual-futures return predictors include basis, momentum, volume and price-volume factors.
- [Binance USDⓈ-M Funding Rate History API](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History): Funding-rate replay field used by the funding/taker screen.
- [Binance USDⓈ-M Taker Buy/Sell Volume API](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Taker-BuySell-Volume): Taker buy/sell volume is the raw flow field behind the taker-flow screen.
- [Binance USDⓈ-M Open Interest Statistics API](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics): Documents the one-month OI-history limit that makes OI replay unsuitable for old OOS claims here.

## Decision

- `funding_taker_flow`: rejected for full backtest; train/val-looking candidates failed OOS after costs or had too few OOS events.
- `cross_crypto_slow_diffusion`: only family with screen survivors; still research-only, not deployment-ready, until a single survivor passes live-equivalent backtest.
- No gross-exposure increase was used as an alpha source.

## Funding / taker-flow rejected leaders

| Candidate | Train n/edge | Val n/edge | OOS n/edge | OOS hit | Reason |
|---|---:|---:|---:|---:|---|
| funding_crowding_unwind ETH/USDT horizon=8h params={'funding_abs': 7.5e-05, 'flow_abs': 0.12, 'momentum_abs': 0.004} | 77 / -0.0563% | 8 / 1.0000% | 0 / n/a | n/a | train_or_validation_post_cost_edge_non_positive |
| funding_crowding_unwind ETH/USDT horizon=8h params={'funding_abs': 7.5e-05, 'flow_abs': 0.12, 'momentum_abs': 0.0015} | 95 / -0.0531% | 10 / 0.9996% | 5 / -1.1162% | 20.0000% | train_or_validation_post_cost_edge_non_positive |
| funding_crowding_unwind SOL/USDT horizon=8h params={'funding_abs': 0.00015, 'flow_abs': 0.06, 'momentum_abs': 0.0025} | 23 / -0.0512% | 16 / 0.8893% | 4 / -0.9353% | 25.0000% | train_or_validation_post_cost_edge_non_positive |
| funding_crowding_unwind ETH/USDT horizon=8h params={'funding_abs': 7.5e-05, 'flow_abs': 0.12, 'momentum_abs': 0.001} | 96 / -0.0740% | 10 / 0.9996% | 5 / -1.1162% | 20.0000% | train_or_validation_post_cost_edge_non_positive |
| funding_crowding_unwind ETH/USDT horizon=8h params={'funding_abs': 7.5e-05, 'flow_abs': 0.12, 'momentum_abs': 0.0025} | 88 / -0.0816% | 10 / 0.9996% | 3 / -1.2455% | 33.3333% | train_or_validation_post_cost_edge_non_positive |
| funding_crowding_unwind ETH/USDT horizon=8h params={'funding_abs': 7.5e-05, 'flow_abs': 0.04, 'momentum_abs': 0.004} | 271 / -0.0400% | 27 / 0.6202% | 12 / -0.6413% | 8.3333% | train_or_validation_post_cost_edge_non_positive |
| funding_crowding_unwind SOL/USDT horizon=4h params={'funding_abs': 0.00015, 'flow_abs': 0.04, 'momentum_abs': 0.0015} | 31 / 0.3253% | 25 / 0.0294% | 7 / -0.2704% | 28.5714% | oos_post_cost_edge_non_positive |
| funding_crowding_unwind ETH/USDT horizon=8h params={'funding_abs': 7.5e-05, 'flow_abs': 0.04, 'momentum_abs': 0.0015} | 370 / -0.0394% | 35 / 0.5352% | 22 / -0.5705% | 22.7273% | train_or_validation_post_cost_edge_non_positive |

## Lead-lag screen survivors

| Candidate | Train n/edge | Val n/edge | OOS n/edge | OOS hit | Reason |
|---|---:|---:|---:|---:|---|
| BTC/USDT→ETH/USDT lag=2h horizon=8h params={'leader_abs_ret_min': 0.015, 'target_underreaction_cap': 999.0} | 359 / 0.0591% | 93 / 0.1047% | 63 / 0.0456% | 52.3810% | screen_survivor |
| SOL/USDT→ETH/USDT lag=1h horizon=8h params={'leader_abs_ret_min': 0.015, 'target_underreaction_cap': 999.0} | 700 / 0.0372% | 95 / 0.0540% | 49 / 0.0376% | 57.1429% | screen_survivor |

## Lead-lag rejected top-ranked examples

| Candidate | Train n/edge | Val n/edge | OOS n/edge | OOS hit | Reason |
|---|---:|---:|---:|---:|---|
| BTC/USDT→ETH/USDT lag=1h horizon=1h params={'leader_abs_ret_min': 0.0015, 'target_underreaction_cap': 999.0} | 5269 / -0.1712% | 913 / -0.1204% | 963 / -0.1678% | 47.0405% | post_cost_edge_not_positive_in_all_splits |
| BTC/USDT→ETH/USDT lag=1h horizon=1h params={'leader_abs_ret_min': 0.0015, 'target_underreaction_cap': 0.5} | 732 / -0.1788% | 129 / -0.0711% | 115 / -0.1585% | 47.8261% | post_cost_edge_not_positive_in_all_splits |
| BTC/USDT→ETH/USDT lag=1h horizon=1h params={'leader_abs_ret_min': 0.0015, 'target_underreaction_cap': 0.25} | 349 / -0.2078% | 65 / 0.0094% | 44 / -0.0709% | 52.2727% | post_cost_edge_not_positive_in_all_splits |
| BTC/USDT→ETH/USDT lag=1h horizon=1h params={'leader_abs_ret_min': 0.0015, 'target_underreaction_cap': 0.0} | 0 / n/a | 0 / n/a | 0 / n/a | n/a | insufficient_split_events_for_live_equivalent_followup |
| BTC/USDT→ETH/USDT lag=1h horizon=1h params={'leader_abs_ret_min': 0.0025, 'target_underreaction_cap': 999.0} | 3584 / -0.1695% | 670 / -0.1102% | 684 / -0.1836% | 45.6140% | post_cost_edge_not_positive_in_all_splits |
| BTC/USDT→ETH/USDT lag=1h horizon=1h params={'leader_abs_ret_min': 0.0025, 'target_underreaction_cap': 0.5} | 466 / -0.1539% | 85 / -0.0465% | 62 / -0.1584% | 46.7742% | post_cost_edge_not_positive_in_all_splits |
| BTC/USDT→ETH/USDT lag=1h horizon=1h params={'leader_abs_ret_min': 0.0025, 'target_underreaction_cap': 0.25} | 220 / -0.2138% | 37 / 0.1126% | 20 / -0.0280% | 45.0000% | insufficient_split_events_for_live_equivalent_followup |
| BTC/USDT→ETH/USDT lag=1h horizon=1h params={'leader_abs_ret_min': 0.0025, 'target_underreaction_cap': 0.0} | 0 / n/a | 0 / n/a | 0 / n/a | n/a | insufficient_split_events_for_live_equivalent_followup |

## Stop condition for next phase

Run at most one full live-equivalent backtest next: the top lead-lag survivor only. If it fails train/val/OOS or current-tail live-equivalent gates, reject the family instead of retuning thresholds inside the backtest loop.
