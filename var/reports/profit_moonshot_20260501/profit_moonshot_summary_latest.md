# Profit Moonshot Research Summary

Generated: `2026-05-05T11:25:00Z`; operator override patched: `2026-05-05T11:25:43Z`; latest follow-up patched: `2026-05-06T13:26:37Z`
Decision: `operator_oos_override_candidate_found`
Candidates scanned by generated ranker: `1050`
Promotion-eligible candidates by generated val-ranker: `35`

## Operator OOS override candidate

- Candidate: `profit_moonshot_hourly_shock_reversion_eth_12h_mode`
- Source: `var/reports/profit_moonshot_20260501/current_tail_20260505/live_equivalent/profit_moonshot_hourly_shock_reversion_eth_12h_mode/live_equivalent_revalidation_latest.json`
- OOS return / MDD: `+0.8284%` / `0.2819%`
- OOS Sharpe / Sortino: `0.100651` / `0.128321`
- OOS trades / liquidations: `30` / `0`
- Rationale: generated ranker is validation-split biased; user gate requires complete raw-first train/val/OOS evidence. This is current OOS-return best, not deployment-ready.

## Sharpe-focused shadow

- Candidate: `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode`
- OOS return / MDD: `+0.7206%` / `0.1778%`
- OOS Sharpe / Sortino: `0.111225` / `0.135831`
- Versus OOS-return best: Sharpe `+10.51%`, MDD reduction `36.93%`, return `-13.01%`.
- Decision: keep as risk-adjusted shadow; do not call it deployment-ready because Sharpe is still below `1.0`.

## External material used

- [Binance USDⓈ-M Futures: Open Interest Statistics](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics)
- [Binance USDⓈ-M Futures: Taker Buy/Sell Volume](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Taker-BuySell-Volume)
- [Binance USDⓈ-M Futures: Funding Rate History](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History)
- [arXiv 2212.06888: Crypto perpetual futures / funding context](https://arxiv.org/abs/2212.06888)

## Generated val-ranker promoted candidate (not operator-promoted)

- Generated ranker candidate: `profit_moonshot_momentum_hybrid_safe_mode`
- Generated primary split: `val` return `+0.2837%` Sharpe `0.010168`
- Operator status: not promoted because it fails the stricter current-tail OOS gate / Sharpe demand.

## Top Ranked Candidates

| rank | candidate | source | split | return | MDD | Sharpe | Sortino | trades | liq | final equity | blockers |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `profit_moonshot_momentum_hybrid_safe_mode` | `live_equivalent` | `val` | 0.28% | 1.04% | 0.010 | 0.010 | 183 | 0 | 10028.407 | - |
| 2 | `profit_moonshot_momentum_hybrid_safe_mode` | `live_equivalent` | `val` | 0.28% | 1.04% | 0.010 | 0.010 | 183 | 0 | 10028.407 | - |
| 3 | `profit_moonshot_momentum_hybrid_safe_mode` | `live_equivalent` | `val` | 0.28% | 1.04% | 0.010 | 0.010 | 183 | 0 | 10028.407 | - |
| 4 | `profit_moonshot_hourly_shock_reversion_eth_mode` | `live_equivalent` | `val` | 1.04% | 0.88% | 0.066 | 0.068 | 66 | 0 | 10104.251 | - |
| 5 | `profit_moonshot_derivatives_taker_flow_sparse_mode` | `live_equivalent` | `val` | 0.08% | 0.05% | 0.055 | 0.027 | 108 | 0 | 10007.985 | - |
| 6 | `profit_moonshot_filtered_shock_reversion_diversified_mode` | `live_equivalent` | `val` | 0.09% | 1.34% | 0.003 | 0.003 | 182 | 0 | 10008.580 | - |
| 7 | `profit_moonshot_momentum_hybrid_core_mode` | `live_equivalent` | `val` | 0.25% | 1.01% | 0.009 | 0.009 | 138 | 0 | 10025.524 | - |
| 8 | `profit_moonshot_momentum_hybrid_return_mode` | `live_equivalent` | `val` | 0.27% | 1.01% | 0.010 | 0.009 | 134 | 0 | 10026.897 | - |
| 9 | `profit_moonshot_hourly_shock_reversion_eth_12h_dense_mode` | `live_equivalent` | `val` | 0.63% | 0.44% | 0.071 | 0.081 | 48 | 0 | 10063.326 | - |
| 10 | `profit_moonshot_hourly_shock_reversion_eth_12h_mode` | `live_equivalent` | `val` | 0.63% | 0.44% | 0.071 | 0.081 | 47 | 0 | 10063.269 | - |
| 11 | `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode` | `live_equivalent` | `val` | 0.50% | 0.27% | 0.063 | 0.072 | 47 | 0 | 10050.235 | - |
| 12 | `profit_moonshot_adaptive_momentum_asym_dynamic_mode` | `live_equivalent` | `val` | 0.00% | 0.07% | 0.000 | 0.000 | 102 | 0 | 10000.001 | - |
| 13 | `profit_moonshot_adaptive_momentum_asym_dynamic_mode` | `live_equivalent` | `val` | 0.00% | 0.07% | 0.000 | 0.000 | 102 | 0 | 0.000 | - |
| 14 | `profit_moonshot_leadlag_slow_diffusion_mode` | `live_equivalent` | `val` | 0.68% | 1.26% | 0.028 | 0.029 | 40 | 0 | 10068.491 | - |
| 15 | `profit_moonshot_leadlag_slow_diffusion_mode` | `live_equivalent` | `val` | 0.68% | 1.26% | 0.028 | 0.029 | 40 | 0 | 10068.491 | - |
| 16 | `profit_moonshot_adaptive_momentum_mode` | `live_equivalent` | `val` | 0.26% | 0.75% | 0.012 | 0.012 | 52 | 0 | 0.000 | - |
| 17 | `profit_moonshot_adaptive_momentum_mode` | `live_equivalent` | `val` | 0.26% | 0.75% | 0.012 | 0.012 | 52 | 0 | 10026.530 | - |
| 18 | `profit_moonshot_adaptive_momentum_boost_mode` | `live_equivalent` | `val` | 0.51% | 1.36% | 0.015 | 0.015 | 56 | 0 | 10051.054 | - |
| 19 | `profit_moonshot_adaptive_momentum_boost_mode` | `live_equivalent` | `val` | 0.51% | 1.36% | 0.015 | 0.015 | 56 | 0 | 0.000 | - |
| 20 | `profit_moonshot_adaptive_momentum_boost_mode` | `live_equivalent` | `val` | 0.51% | 1.36% | 0.015 | 0.015 | 56 | 0 | 10051.054 | - |
| 21 | `profit_moonshot_adaptive_momentum_120_mode` | `live_equivalent` | `val` | 0.33% | 0.93% | 0.013 | 0.012 | 52 | 0 | 0.000 | - |
| 22 | `profit_moonshot_adaptive_momentum_120_mode` | `live_equivalent` | `val` | 0.33% | 0.93% | 0.013 | 0.012 | 52 | 0 | 10032.962 | - |
| 23 | `profit_moonshot_adaptive_momentum_vol_target_mode` | `live_equivalent` | `val` | 0.40% | 1.13% | 0.013 | 0.013 | 54 | 0 | 10039.730 | - |
| 24 | `profit_moonshot_adaptive_momentum_vol_target_mode` | `live_equivalent` | `val` | 0.40% | 1.13% | 0.013 | 0.013 | 54 | 0 | 0.000 | - |
| 25 | `profit_moonshot_adaptive_momentum_vol_target_132_mode` | `live_equivalent` | `val` | 0.42% | 1.19% | 0.013 | 0.013 | 54 | 0 | 10041.849 | - |

## Blocker Summary

- `val_sharpe_not_positive`: 10
- `val_sortino_not_positive`: 10
- `val_total_return_not_positive`: 10
- `train_trade_count_below_min`: 5
- `val_trade_count_below_min`: 5
- `legacy_train_val_mdd_gate_failed`: 4
- `status=ready_for_live_equivalent_backtest`: 4
- `status_not_validated:ready_for_live_equivalent_backtest`: 4

## New alpha follow-up — taker-flow exhaustion

- Implemented `TakerFlowExhaustionReversalStrategy` with true taker-flow feature requirement, funding cap, realized-vol cap, UTC session filter, same `target_allocation=0.008` / `max_order_value=175.0`, and a cooldown risk-control variant.
- Screen artifact: `var/reports/profit_moonshot_20260501/current_tail_20260505/taker_flow_exhaustion_screen/taker_flow_exhaustion_screen_20260505.json` (`1,296,000` combinations, `506` survivors, peak RSS `2577.43 MiB`).
- Full live-equivalent verdict: **no variant promoted**. Raw overlapping-event edge did not survive one-position/fee/partial-fill path realism.

| mode | train ret | val ret | OOS ret | OOS Sharpe | decision |
|---|---:|---:|---:|---:|---|
| `profit_moonshot_taker_flow_exhaustion_eth_mode` | +0.0089% | +0.0000% | +0.0000% | 0.000000 | rejected: zero val/OOS coverage after cadence/chunk phase |
| `profit_moonshot_taker_flow_exhaustion_eth_reactive_mode` | -0.0019% | +0.0210% | -0.0291% | -0.085988 | rejected: train/OOS negative |
| `profit_moonshot_taker_flow_exhaustion_eth_hold_mode` | -1.3901% | -0.0041% | -0.0875% | -0.031058 | rejected: wider hold worsened train and kept val/OOS negative |
| `profit_moonshot_taker_flow_exhaustion_eth_slow_momentum_mode` | -0.2422% | -0.3210% | -0.0507% | -0.036304 | rejected: cooldown risk control still failed all splits |

Detailed report: `var/reports/profit_moonshot_20260501/current_tail_20260505/taker_flow_exhaustion_new_alpha_report_20260505.md`.


## New alpha follow-up — precious-metal pair aggressive mode

- Implemented `TimeframePairZScoreReversionStrategy` and mode `profit_moonshot_precious_metal_pair_aggressive_mode` covering XAU/XAG and XPT/XPD with explicit target-allocation, max-order, stop/take-profit/max-hold, beta/correlation, and current-volume guards.
- External rationale: CME gold/silver ratio + metals spread products support the relative-value family, but mean-reversion cost literature requires engine/fill-aware validation.
- Tail refresh attempted for XAU/XAG/XPT/XPD (`2026-03-28` to `2026-05-05`): peak RSS `1537.184 MiB`; XAU raw archive reached `date=2026-05-04`; XAG/XPT/XPD tail refresh did not complete after CloudFront/no-progress stall.
- Standard raw-first live-equivalent gate: `blocked_missing_raw_first_market_data` for all four metals train/val windows.
- Legacy-windowed split engine result: train `-0.0570%` Sharpe `-0.024200`; val `+0.1914%` Sharpe `0.037224`; OOS `-0.0478%` Sharpe `-0.164066`; liquidations `0`.
- Decision: **not promoted**. Keep `profit_moonshot_hourly_shock_reversion_eth_12h_mode` as OOS-return best, `profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode` as Sharpe/MDD shadow, and `profit_moonshot_momentum_hybrid_safe_mode` as conservative legacy candidate.
- Detailed report: `var/reports/profit_moonshot_20260501/current_tail_20260505/precious_metal_pair_aggressive/precious_metal_pair_aggressive_report_20260505.md`.

## Useful-alpha execution follow-up — 2026-05-06

- Report: `var/reports/profit_moonshot_20260501/current_tail_20260506/useful_alpha_execution_report_20260506.md`.
- Raw-first coverage check: ETH/BTC/SOL train and val complete; OOS through `2026-05-05` missing `2026-05-05`, so full tests used complete OOS end `2026-05-04`; metals remain raw-first blocked.
- Feature inventory: BTC/ETH/SOL funding, OI, and taker-flow are available; liquidation rows are `0`, so liquidation confirmation is not usable yet.
- Stateful ETH 12h shock replay: `130` filter specs, `8` replay-relative survivors, `0` absolute final-gate survivors.
- Full live-equivalent survivors tested one at a time:
  - `profit_moonshot_hourly_shock_reversion_eth_12h_taker_flow_guard_mode`: OOS `+0.5871%`, MDD `0.3203%`, Sharpe `0.070688`; rejected.
  - `profit_moonshot_hourly_shock_reversion_eth_12h_sol_regime_guard_mode`: OOS `+0.3221%`, MDD `0.9275%`, Sharpe `0.014160`; rejected.
  - `profit_moonshot_hourly_shock_reversion_eth_12h_funding_taker_flow_guard_mode`: OOS `+0.1759%`, MDD `0.2204%`, Sharpe `0.043476`; train `-0.2259%` with `1` liquidation; rejected.
- Decision: **no new successful alpha promoted**. Keep ETH 12h shock reversion as OOS-return best and funding guard as Sharpe/MDD shadow; new modes remain shadow/audit artifacts only, not live-selection-supported modes.

## Leverage/rebalancing cadence follow-up — 2026-05-06

- Cadence sweep report: `var/reports/profit_moonshot_20260501/current_tail_20260506/cadence_sweep/profit_moonshot_cadence_sweep_latest.md`.
- Exact screen: `174` cadence variants; no exposure/gross/max-order increase.
- Best screen survivor `profit_moonshot_adaptive_momentum_boost_mode__cadence_1b` is discarded: report-capped train `-100.0000%` / MDD `100.0000%` after equity breach, raw arithmetic `-111.5894%` / `175.7880%`, `5` train liquidations; OOS `-0.9619%` / Sharpe `0.012010` / MDD `33.0023%`.
- Decision: **no cadence/rebalance alpha promoted**; the equity-breach candidate is a failure artifact, not useful alpha. Incumbent ETH 12h shock and funding-guard shadow remain unchanged.
## Multiasset exchange expansion follow-up — 2026-05-07

- Report: `var/reports/profit_moonshot_20260501/current_tail_20260506/multiasset_exchange_expansion/multiasset_exchange_alpha_execution_report_20260506.md`.
- Hyperliquid read-only public `/info` collection upserted `35211` feature rows; funding history covered train/val/OOS, but OI/mark were current snapshot/context only.
- Tickmill/MT5 read-only collection was blocked because `LQ_MT5_BRIDGE_PYTHON / LQ__LIVE__MT5_BRIDGE_PYTHON is not configured.`.
- Stateful replay evaluated `38` Hyperliquid/Tickmill confirmation/regime filters; replay survivors `0`, success candidates `0`, max RSS `6710.816 MiB`.
- No live-equivalent full backtest was run because there was no replay survivor. Incumbent and Sharpe/MDD shadow remain unchanged.
