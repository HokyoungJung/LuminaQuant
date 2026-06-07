# Codex performance re-research — lagged leaf router grid diagnostic (2026-06-07)

## Executive result

- Goal: 기존 85-symbol/router/Optuna lineage에서 성능이 구린 지점을 다시 파고, `no_nested_oos_mining`, `execution_cost_gate`, `theory_plausibility_gate`를 유지하면서 성적 개선 후보를 찾는다.
- Best **exact source-metric** diagnostic: `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled`.
  - OOS compounded `197.37%`, annualized approx `269.80%`, max fold OOS MDD `27.69%`, monthly equity MDD `4.50%`, positive folds `5/10`, PF `30.04`.
  - This improves the canonical exact lagged-router comparator from `61.40%` comp / `77.62%` ann to `197.37%` comp / `269.80%` ann on the same 10 locked OOS folds.
- Best row-metric scaled diagnostic: `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.02_vmdd0.25_lagged_plus_val025_target0.30_cap3.00_rowmetric_scaled` produced `377.94%` comp / `553.50%` ann, but it is **not bar-exact** and is only a sizing stress proxy.
- Deployment label: **research_shadow_only_requires_fresh_forward_shadow_and_bar_exact_rerun**. Clean live allocation remains `0%` until forward shadow/paper fill telemetry passes.

## Why this is not nested and why it is still not live-clean

- Leaf universe only: strict/relaxed efficiency `balanced`, `growth`, `aggressive` leaf rows. Labels containing hybrid/selector/router/meta/static-guarded tokens were excluded.
- Selection inputs: current fold train/validation metrics plus prior completed source-leaf OOS returns. Current fold locked OOS is report-only.
- The rule itself was found after reviewing this historical OOS set. Therefore it is a **post-OOS research variant**, not a clean promotable strategy.
- Cost basis inherits the source 10bps slippage/cost model, but row artifacts have no turnover/RPT/fill telemetry. The execution-cost gate remains blocked for real money.

## Comparator table

| Candidate | Label class | OOS comp | Ann approx | Max OOS MDD | Monthly equity MDD | Positive folds | Latest fold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Best exact diagnostic | shadow/exact-source | 197.37% | 269.80% | 27.69% | 4.50% | 5/10 | 64.80% |
| Best row-metric scaled diagnostic | shadow/approx-sizing | 377.94% | 553.50% | 60.77% | 6.80% | 5/10 | 81.00% |
| Best risk-filtered row-metric scaled | shadow/approx-sizing | 273.90% | 386.75% | 30.39% | 6.75% | 5/10 | 81.00% |
| Canonical lagged best exact | shadow/existing | 61.40% | 77.62% | 29.13% | 3.86% | 4/10 | -3.34% |
| Clean dynamic best | paper-control clean mechanics | 34.39% | 42.57% | 27.69% | 0.00% | 3/10 | 0.00% |
| Strict aggressive leaf | clean leaf source | 32.74% | 45.88% | 14.77% | 14.40% | 4/9 | 49.10% |
| Relaxed aggressive leaf | clean leaf source | 31.62% | 39.05% | 26.47% | 22.01% | 4/10 | 64.80% |

## Best exact diagnostic fold path

| Fold | Branch | Selected leaf | Hist | Val ret | Val MDD | OOS ret | OOS MDD | Scale |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2025-09 | strict_core_cash | cash_strict_core_validation_strength_guard | - | 0.00% | 0.00% | 0.00% | 0.00% | 0.0 |
| 2025-10 | strict_core_cash | cash_strict_core_validation_strength_guard | - | 0.00% | 0.00% | 0.00% | 0.00% | 0.0 |
| 2025-11 | strict_core_scaled | strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna | - | 0.00% | 0.00% | 33.48% | 27.69% | 3.0 |
| 2025-12 | strict_core_cash | cash_strict_core_validation_strength_guard | - | 0.00% | 0.00% | 0.00% | 0.00% | 0.0 |
| 2026-01 | online_leaf | relaxed_efficiency:growth_mdd20_gross8_69_asset_relaxed_efficiency_repair_optuna | 4 | 32.08% | 13.75% | 16.03% | 9.34% | 1.0 |
| 2026-02 | online_leaf | strict_efficiency:aggressive_mdd30_gross10_69_asset_efficiency_repair_optuna | 4 | 52.21% | 14.81% | 8.44% | 11.81% | 1.0 |
| 2026-03 | online_leaf | relaxed_efficiency:balanced_mdd12_gross5_69_asset_relaxed_efficiency_repair_optuna | 6 | 21.65% | 10.74% | -0.05% | 3.93% | 1.0 |
| 2026-04 | online_leaf | relaxed_efficiency:growth_mdd20_gross8_69_asset_relaxed_efficiency_repair_optuna | 7 | 27.29% | 14.01% | -4.46% | 10.06% | 1.0 |
| 2026-05 | online_leaf | relaxed_efficiency:balanced_mdd12_gross5_69_asset_relaxed_efficiency_repair_optuna | 8 | 45.43% | 8.95% | 12.51% | 20.26% | 1.0 |
| 2026-06 | online_leaf | relaxed_efficiency:aggressive_mdd30_gross10_69_asset_relaxed_efficiency_repair_optuna | 9 | 100.20% | 23.33% | 64.80% | 6.69% | 1.0 |

## Interpretation

- The performance lift comes from letting validation conviction have a small vote over the prior-paper-return router: score = last-1 completed leaf OOS return + `0.25 * validation_score` after 4-month warmup, with train MDD <= 50% and validation MDD <= 25%.
- This is theoretically plausible: trend/cross-sectional momentum and volatility/risk-budgeting are documented across futures/asset classes, while limit-order-book/fill work remains the right next execution-control layer. It is not a date/asset hard-code.
- But the huge 2026-06 contribution (`relaxed aggressive` selected, OOS `+64.80%`) makes overfit risk obvious. Treat it as a hypothesis for fresh-forward shadow, not as expected live return.
- The row-metric scaled variants are useful for prioritizing a bar-exact rerun, but not for reporting production performance because scale is approximated from fold-level metrics rather than recomputed on bar returns.

## Next clean action

1. Implement this exact diagnostic as a pre-registered lagged-router spec in the runner, then rerun bar-exact on unchanged historical folds only to verify metric reproduction.
2. Freeze the spec before any new OOS month. Run fresh-forward shadow/paper for at least 1–2 refit months with 10/15/20bps cost grid, turnover/RPT, BBO spread/slippage, partial/reject/cancel telemetry.
3. Only after forward evidence passes: consider paper-control or small-sleeve review. Until then: no real-money deployment.

## External evidence basis checked

- Binance historical market-data archive: https://data.binance.vision/ and bookTicker prefix page https://data.binance.vision/?prefix=data%2Ffutures%2Fum%2Fmonthly%2FbookTicker%2FBTCUSDT%2F
- Binance USDⓈ-M futures fee page / current fee verification entry: https://www.binance.com/en/fee/futureFee
- Moskowitz, Ooi, Pedersen, “Time Series Momentum,” Journal of Financial Economics 2012: https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf
- Moreira & Muir, “Volatility Managed Portfolios,” NBER/JF: https://www.nber.org/papers/w22208
- Bailey et al., “The Probability of Backtest Overfitting”: https://escholarship.org/uc/item/4w1110bb
- DeepLOB / limit-order-book feature literature: https://arxiv.org/abs/1808.03668

## Artifact

- JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/codex_performance_research_20260607/lagged_leaf_router_grid_diagnostic_20260607.json`
- Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/codex_performance_research_20260607/lagged_leaf_router_grid_diagnostic_20260607.md`
