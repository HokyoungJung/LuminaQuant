# Manual overall re-evaluation with latest data included

Generated: 2026-06-28T11:08:43.990214Z

## Correction
Earlier 2026-06-21 cutoff was incorrect. Direct 1m parquet timestamp audit found every symbol in both 85 and 110 universes present through 2026-06-28T10:09:00Z; monthly walk-forward can use completed 30m bars through 2026-06-28T09:30:00Z.

## Evaluation period and method
- aggregate_oos_folds: 2025-09 through 2026-06, 10 monthly OOS folds
- latest_fold_oos_utc: 2026-06-01T00:00:00 through 2026-06-28T09:30:00
- latest_fold_train_utc: 2025-01-01T00:00:00 through 2026-03-31T23:30:00
- latest_fold_validation_utc: 2026-04-01T00:00:00 through 2026-05-31T23:30:00
- Method: monthly walk-forward; previous closed folds unchanged, latest 2026-06 fold rerun and inserted into the full 10-fold aggregate.
- Data: direct 1m parquet audit found 85/85 and 110/110 symbols through 2026-06-28T10:09:00Z; completed 30m WF end is 2026-06-28T09:30:00Z.

## Winner
- comparison: `expanded_110_latest_tail_full`
- strategy: `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled`
- compounded OOS: 159.83%
- annualized approx: 214.51%
- monthly Sharpe approx: 1.881
- profit factor: 23.635
- max intra-fold OOS MDD: 27.69%
- monthly equity MDD: 5.09%
- hit count: 4/10

## Ranking
| rank | comparison | strategy suffix | compounded OOS | ann approx | Sharpe | PF | max OOS MDD | hit |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `expanded_110_latest_tail_full` | `exact_unscaled` | 159.83% | 214.51% | 1.881 | 23.635 | 27.69% | 4/10 |
| 2 | `expanded_110_latest_tail_full` | `fallback_mdd20_cap2` | 138.11% | 183.23% | 1.783 | 21.454 | 23.58% | 4/10 |
| 3 | `historical_85_preregistered` | `exact_unscaled` | 125.13% | 164.81% | 2.506 | 21.148 | 30.47% | 5/10 |
| 4 | `historical_85_preregistered` | `fallback_mdd20_cap2` | 106.31% | 138.47% | 2.637 | 18.669 | 30.47% | 5/10 |

## Latest fold replacements
- `historical_85_preregistered`: June OOS 2026-06-01T00:00:00 -> 2026-06-28T09:30:00, return 24.76%, MDD 30.47%, source `var/reports/manual_reval_20260628_monthly_refit_june_latest_rerun85/june_latest_85_relaxed_efficiency_wf_latest.json`
- `expanded_110_latest_tail_full`: June OOS 2026-06-01T00:00:00 -> 2026-06-28T09:30:00, return 63.01%, MDD 23.58%, source `var/reports/manual_reval_20260628_monthly_refit_june_latest_rerun110/june_latest_110_relaxed_efficiency_wf_latest.json`

## Caveat
This remains backtest/shadow-paper evidence only. The repo artifact explicitly does not imply real-money approval.
