# Profit Moonshot external-alpha validation report

- generated_at: `2026-05-03T08:56:33.067842+00:00`
- decision: `profit_moonshot_leadlag_slow_diffusion_mode` is live-equivalent validated; keep `profit_moonshot_momentum_hybrid_safe_mode` as conservative fallback until blended portfolio risk is reviewed.
- process: external source → raw-first ex-ante screen → one survivor full backtest. No threshold-mining inside full backtest; no gross-exposure bump.

## External sources used
- [A Seesaw Effect in the Cryptocurrency Market](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3465924): Cross-cryptocurrency intraday return predictability and lead-lag/seesaw effects.
- [Anatomy of cryptocurrency perpetual futures returns](https://www.research.ed.ac.uk/en/publications/anatomy-of-cryptocurrency-perpetual-futures-returns/): Perpetual-futures return predictors include basis, momentum, volume and price-volume factors.
- [Binance USDⓈ-M Funding Rate History API](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History): Funding-rate replay field used by the funding/taker screen.
- [Binance USDⓈ-M Taker Buy/Sell Volume API](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Taker-BuySell-Volume): Taker buy/sell volume is the raw flow field behind the taker-flow screen.
- [Binance USDⓈ-M Open Interest Statistics API](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics): Documents the one-month OI-history limit that makes OI replay unsuitable for old OOS claims here.

## Latest-tail refresh evidence
- status: `completed`
- collection_cutoff_utc: `2026-05-03T08:32:27Z`
- refresh peak RSS: `4424.3984375` MiB

## Raw-first screen outcome
- funding/taker-flow survivors: `0` → rejected for full backtest.
- lead-lag survivors: `2` → only top survivor was promoted to full backtest.

| Family | Candidate | Train n/edge | Val n/edge | OOS n/edge | Decision |
|---|---|---:|---:|---:|---|
| funding/taker | `ETH/USDT 8h {'flow_abs': 0.12, 'funding_abs': 7.5e-05, 'momentum_abs': 0.004}` | 77 / -0.0563% | 8 / 1.0000% | 0 / n/a | rejected: `train_or_validation_post_cost_edge_non_positive` |
| funding/taker | `ETH/USDT 8h {'flow_abs': 0.12, 'funding_abs': 7.5e-05, 'momentum_abs': 0.0015}` | 95 / -0.0531% | 10 / 0.9996% | 5 / -1.1162% | rejected: `train_or_validation_post_cost_edge_non_positive` |
| funding/taker | `SOL/USDT 8h {'flow_abs': 0.06, 'funding_abs': 0.00015, 'momentum_abs': 0.0025}` | 23 / -0.0512% | 16 / 0.8893% | 4 / -0.9353% | rejected: `train_or_validation_post_cost_edge_non_positive` |
| lead-lag | `BTC/USDT→ETH/USDT 2h 8h {'leader_abs_ret_min': 0.015, 'target_underreaction_cap': 999.0}` | 359 / 0.0591% | 93 / 0.1047% | 63 / 0.0456% | screen survivor |
| lead-lag | `SOL/USDT→ETH/USDT 1h 8h {'leader_abs_ret_min': 0.015, 'target_underreaction_cap': 999.0}` | 700 / 0.0372% | 95 / 0.0540% | 49 / 0.0376% | screen survivor |

## Live-equivalent backtest result

| Split | Return | MDD | Sharpe | Sortino | Trades | Liquidations |
|---|---:|---:|---:|---:|---:|---:|
| train | 3.1274% | 9.1302% | 0.006798 | 0.006917 | 176 | 0 |
| val | 0.6833% | 1.2601% | 0.028293 | 0.028537 | 40 | 0 |
| oos | 0.2209% | 1.0096% | 0.011127 | 0.014112 | 38 | 0 |

- status: `live_equivalent_validated`; selection_eligible: `True`; selection_score: `12.964718`
- full backtest RSS: `1905364 KB` (<8GB), elapsed `13:40.07`
- OOS is report-only in the framework, but this candidate has complete raw-first OOS replay and positive OOS result.

## Failure notes

- The earlier funding/taker implementation attempt was aborted before completion after the user correction; it is not counted as evidence.
- Funding/taker-flow looked attractive in validation for some thresholds, but failed OOS post-cost checks or had no/too few OOS events, so no full backtest was run for that family.
- Existing `hybrid_safe` remains the conservative fallback, but it lacks this new candidate’s complete OOS/raw-first validation.

## Follow-up: same-risk leadlag ensemble rejected

A second full run was limited to one mode: `profit_moonshot_leadlag_slow_diffusion_ensemble_mode`, adding the raw-first SOL→ETH survivor as a 40% sleeve while reducing BTC→ETH to 60% so total ETH target allocation stayed 0.8%. It is **not promoted**: OOS return was `-0.3059%`, OOS Sharpe `-0.037976`, OOS Sortino `-0.044528`, with 0 liquidations and max RSS 2,501,688 KB. The promoted external-alpha decision therefore remains `profit_moonshot_leadlag_slow_diffusion_mode`. Detailed failure evidence: `var/reports/profit_moonshot_20260501/external_alpha/leadlag_ensemble_followup_report_20260503.md`.
