# Profit moonshot live final selection

- generated_at_utc: `2026-05-10T11:25:29.701707Z`
- status: `no_live_promotion`
- recommendation: `no_live_promotion`
- latest_complete_oos_end_date: `2026-05-09`
- winner: `None`

## Comparison rows

| Kind | Name | Leverage | Integer live | Strategy pass | Primary signal | Candidate-derived | Benchmark-only | TV score | OOS return | OOS MDD | OOS return/MDD | Liq | Min buffer | Deployable |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `current_base` | `current_base_tuple_liquidation_aware_2.34273x` | 2.342733 | `False` | `False` | `calendar_primary` | `False` | `True` | 15.817103 | 6.4281% | 0.9293% | 6.916878 | 0 | 9256.942427 | `False` |
| `direct_candidate` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fr_liquidation_aware_5x` | 5.000000 | `True` | `False` | `calendar_primary` | `True` | `False` | 19.882952 | 14.6634% | 1.9646% | 7.463950 | 0 | 8514.066649 | `False` |
| `direct_candidate` | `current_base_tuple_liquidation_aware_5x` | 5.000000 | `True` | `False` | `calendar_primary` | `True` | `False` | 19.244936 | 14.0578% | 1.9584% | 7.178039 | 1 | 8415.811075 | `False` |
| `direct_candidate` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fr_liquidation_aware_4x` | 4.000000 | `True` | `False` | `calendar_primary` | `True` | `False` | 25.856247 | 8.2151% | 5.3796% | 1.527074 | 0 | 9281.183213 | `False` |
| `direct_candidate` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fr_liquidation_aware_3x` | 3.000000 | `True` | `False` | `calendar_primary` | `True` | `False` | 25.144233 | 6.1278% | 4.0786% | 1.502424 | 0 | 9461.147976 | `False` |
| `direct_candidate` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fr_liquidation_aware_2x` | 2.000000 | `True` | `False` | `calendar_primary` | `True` | `False` | 21.240003 | 4.0544% | 2.7489% | 1.474913 | 0 | 9640.417895 | `False` |
| `direct_candidate` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fr_liquidation_aware_4x` | 4.000000 | `True` | `False` | `calendar_primary` | `True` | `False` | 20.094996 | 13.7605% | 3.2515% | 4.232070 | 0 | 8449.334129 | `False` |
| `direct_candidate` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fr_liquidation_aware_5x` | 5.000000 | `True` | `False` | `calendar_primary` | `True` | `False` | 19.875591 | 14.5038% | 1.9435% | 7.462877 | 0 | 8514.066649 | `False` |
| `direct_candidate` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fr_liquidation_aware_5x` | 5.000000 | `True` | `False` | `calendar_primary` | `True` | `False` | 19.392476 | 10.3324% | 6.6883% | 1.544841 | 3 | 8700.760784 | `False` |
| `candidate_portfolio` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls530_ss120_tp600` | 2.342733 | `False` | `False` | `calendar_primary` | `True` | `False` | 16.576134 |  |  |  | 0 |  | `False` |
| `legacy_hybrid_benchmark` | `legacy_hybrid_benchmark` | 2.342733 | `False` | `True` | `non_candidate_row` | `False` | `True` | 19.066699 | 0.1618% | 0.4897% | 0.330422 | 0 |  | `False` |
| `cash` | `cash` | 1.000000 | `True` | `True` | `non_candidate_row` | `False` | `True` | 0.000000 | 0.0000% | 0.0000% |  | 0 |  | `False` |

## Metric explanations

- **total_return**: Split ending equity return before compounding; higher is better after gates.
- **monthlyized_return**: Monthlyized return estimate used for train/validation stability context.
- **annualized_return**: Annualized/CAGR-style return where available.
- **max_drawdown**: Largest peak-to-trough equity loss in the split; lower is safer.
- **return_mdd**: Total return divided by max drawdown; rewards return per drawdown unit.
- **sharpe**: Return per total volatility; higher is better.
- **sortino**: Return per downside volatility; higher is better for asymmetric downside risk.
- **smart_sortino**: Sortino adjusted by strategy-specific tail/quality penalties when available.
- **calmar**: Annualized return divided by max drawdown; live safety ratio.
- **volatility**: Annualized or split-normalized total volatility from source artifact.
- **downside_volatility**: Downside-only volatility where available.
- **positive_period_ratio**: Fraction of positive periods or proxy win-rate where available.
- **fills**: Executed fill count; helps detect starved or overactive sleeves.
- **round_trips**: Completed trade count when available.
- **liquidation_count**: Number of liquidation events in the split.
- **minimum_margin_buffer**: Minimum equity minus conservative margin requirement; must stay positive.
- **minimum_margin_ratio**: Minimum equity divided by margin requirement; higher is safer.
- **maximum_liquidation_event_drawdown**: Worst instantaneous drawdown at liquidation event.
- **maximum_liquidation_equity_loss_fraction**: Worst equity fraction lost at liquidation event.
- **account_wipeout**: Whether an event wiped the whole account; any true value blocks promotion.
- **live_integer_leverage**: Whether the row uses a positive integer leverage supported by the live venue/runner.
- **live_integer_source_leverage**: For hybrid rows, whether every active source candidate also uses positive integer leverage.
- **strategy_validity_passed**: Source-aware thesis/rule validation of the primary signal and source sleeves.
- **source_metadata_present**: Whether candidate rows link to durable research-history and source-search ledger references.
- **memory_max_rss**: Maximum resident set size evidence; must stay below 8 GiB.

## Memory ledger

- under_8gib: `True`
- `artifact_memory_summary` `1`: max_rss_mib=`225.34`, under_8gib=`True`
- `artifact_memory_summary` `2`: max_rss_mib=`56.461`, under_8gib=`True`
- `artifact_memory_summary` `3`: max_rss_mib=`56.594`, under_8gib=`True`
- `artifact_memory_summary` `4`: max_rss_mib=`71.562`, under_8gib=`True`
- `time_log` `var/reports/profit_moonshot_20260501/dynamic_restart_20260510/replay_time.log`: max_rss_mib=`249.406`, under_8gib=`True`
- `time_log` `var/reports/profit_moonshot_20260501/dynamic_restart_20260510/tuning_time.log`: max_rss_mib=`251.977`, under_8gib=`True`
- `time_log` `var/reports/profit_moonshot_20260501/dynamic_restart_20260510/candidate_hybrid_time.log`: max_rss_mib=`251.555`, under_8gib=`True`
- `time_log` `var/reports/profit_moonshot_20260501/live_final_selection_20260510/runner_evidence/data_refresh_time.log`: max_rss_mib=`5069.789`, under_8gib=`True`
- `time_log` `var/reports/profit_moonshot_20260501/live_final_selection_20260510/runner_evidence/liquidation_validation_time.log`: max_rss_mib=`261.203`, under_8gib=`True`
