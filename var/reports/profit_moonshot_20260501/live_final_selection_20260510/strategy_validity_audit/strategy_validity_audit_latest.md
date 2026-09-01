# Profit moonshot strategy-validity audit

- generated_at_utc: `2026-05-10T10:26:15.978327Z`
- status: `pass_with_rejections`
- row_count: `33`
- strategy_invalid_count: `31`
- deployable_valid_count: `0`
- deployable_invalid_count: `0`
- no_new_alpha_search: `True`

## Row audit

| Kind | Name | Deployable | Strategy pass | Primary signal | Reasons |
|---|---|---:|---:|---|---|
| `current_base` | `current_base_tuple_liquidation_aware_2.34273x` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `direct_candidate` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fr_liquidation_aware_5x` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `direct_candidate` | `current_base_tuple_liquidation_aware_5x` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `direct_candidate` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fr_liquidation_aware_4x` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `direct_candidate` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fr_liquidation_aware_3x` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `direct_candidate` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fr_liquidation_aware_2x` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `direct_candidate` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fr_liquidation_aware_4x` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `direct_candidate` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fr_liquidation_aware_5x` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `direct_candidate` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fr_liquidation_aware_5x` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `candidate_portfolio` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h120_ls600_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `candidate_portfolio` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls530_ss120_tp600` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `candidate_portfolio` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `candidate_portfolio` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `candidate_portfolio` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr150_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `candidate_portfolio` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sweakest_thr120_h120_ls620_ss120_tp450` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `candidate_portfolio` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `candidate_portfolio` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `candidate_portfolio` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls580_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `candidate_portfolio` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr150_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `candidate_portfolio` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls560_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `candidate_portfolio` | `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr150_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls600_ss120_tp600` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `candidate_hybrid` | `candidate_hybrid_online_rank_01_candidate_hybrid_input_01_fresh_portfolio_train_val_monthly_` | `False` | `False` | `calendar_primary` | `calendar_fixed_month_alpha, calendar_primary_alpha_unsupported, fixed_asset_calendar_target` |
| `candidate_hybrid` | `candidate_hybrid_online_rank_02_candidate_hybrid_input_07_fresh_portfolio_train_val_monthly_` | `False` | `False` | `unknown` | `strategy_source_row_missing_sleeves` |
| `candidate_hybrid` | `candidate_hybrid_online_rank_03_candidate_hybrid_input_08_fresh_portfolio_train_val_monthly_` | `False` | `False` | `unknown` | `strategy_source_row_missing_sleeves` |
| `candidate_hybrid` | `candidate_hybrid_online_rank_04_candidate_hybrid_input_01_fresh_portfolio_train_val_monthly_` | `False` | `False` | `unknown` | `strategy_source_row_missing_sleeves` |
| `candidate_hybrid` | `candidate_hybrid_online_rank_05_candidate_hybrid_input_07_fresh_portfolio_train_val_monthly_` | `False` | `False` | `unknown` | `strategy_source_row_missing_sleeves` |
| `candidate_hybrid` | `candidate_hybrid_online_rank_06_candidate_hybrid_input_08_fresh_portfolio_train_val_monthly_` | `False` | `False` | `unknown` | `strategy_source_row_missing_sleeves` |
| `candidate_hybrid` | `candidate_hybrid_online_rank_07_candidate_hybrid_input_01_fresh_portfolio_train_val_monthly_` | `False` | `False` | `unknown` | `strategy_source_row_missing_sleeves` |
| `candidate_hybrid` | `candidate_hybrid_online_rank_08_candidate_hybrid_input_07_fresh_portfolio_train_val_monthly_` | `False` | `False` | `unknown` | `strategy_source_row_missing_sleeves` |
| `candidate_hybrid` | `candidate_hybrid_online_rank_09_candidate_hybrid_input_08_fresh_portfolio_train_val_monthly_` | `False` | `False` | `unknown` | `strategy_source_row_missing_sleeves` |
| `candidate_hybrid` | `candidate_hybrid_online_rank_10_candidate_hybrid_input_01_fresh_portfolio_train_val_monthly_` | `False` | `False` | `unknown` | `strategy_source_row_missing_sleeves` |
| `legacy_hybrid_benchmark` | `legacy_hybrid_benchmark` | `False` | `True` | `non_candidate_row` | `` |
| `cash` | `cash` | `False` | `True` | `non_candidate_row` | `` |

## Closure manifest

- `candidate_hybrid` `var/reports/profit_moonshot_20260501/live_final_selection_20260510/candidate_hybrid/candidate_hybrid_latest.json` rows=`12`
- `candidate_portfolio` `var/reports/profit_moonshot_20260501/live_final_selection_20260510/candidate_portfolio/fresh_portfolio_tuning_latest.json` rows=`50`
- `final_selection_json` `var/reports/profit_moonshot_20260501/live_final_selection_20260510/final_decision/profit_moonshot_live_final_selection_latest.json` rows=`33`
- `final_selection_md` `var/reports/profit_moonshot_20260501/live_final_selection_20260510/final_decision/profit_moonshot_live_final_selection_latest.md` rows=`84`
- `legacy_hybrid` `var/reports/profit_moonshot_20260501/live_final_selection_20260510/hybrid_final/hybrid_online_portfolio_latest.json` rows=`9`
- `liquidation_validation` `var/reports/profit_moonshot_20260501/live_final_selection_20260510/liquidation_validation/liquidation_aware_current_base_latest.json` rows=`60`
- `merged_candidate_csv` `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/merged_alpha_v2_candidates.csv` rows=`8821`
- `passing_artifacts` `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/passing_candidate_latest.json` rows=`7`
- `refresh_json` `var/reports/profit_moonshot_20260501/live_final_selection_20260510/data_refresh/data_refresh_latest.json` rows=`6`
- `per_row_per_sleeve_sources` `inline:final_selection.rows.strategy_validity` rows=`87`

## Source pool summary

- available: `True`
- row_count: `8821`
- strategy_valid_rows: `4429`
- calendar_primary_invalid_rows: `4392`
- strategy_valid_success_candidate_count: `0`

### Top strategy-valid success candidates from source pool


### Top strategy-valid candidates before success/liquidation promotion gates

- `fresh_pair_resid_revert_spread_lb24_z150_h48_sc10_st100_tp400_asiaus` family=`residual_pair_reversion_spread` success=`False` train=`0.004366271857272563` val=`4.62382054360333e-05` oos=`-0.0011007060953359682` score=`0.0020832697394866134`
- `fresh_resid_flow_rev_fl12_lb48_z175_imb10` family=`residual_reversion_flow_confirmed` success=`False` train=`0.0` val=`0.0` oos=`-7.177648578149398e-05` score=`0.0`
- `fresh_resid_flow_rev_fl12_lb48_z15_imb10` family=`residual_reversion_flow_confirmed` success=`False` train=`0.0` val=`0.0` oos=`-1.4105663572405724e-05` score=`0.0`
- `fresh_resid_flow_rev_fl12_lb48_z125_imb10` family=`residual_reversion_flow_confirmed` success=`False` train=`0.0` val=`0.0` oos=`-1.4105663572405724e-05` score=`0.0`
- `fresh_resid_flow_rev_fl12_lb24_z175_imb10` family=`residual_reversion_flow_confirmed` success=`False` train=`0.0` val=`0.0` oos=`0.0` score=`0.0`
- `fresh_resid_flow_rev_fl12_lb24_z15_imb10` family=`residual_reversion_flow_confirmed` success=`False` train=`0.0` val=`0.0` oos=`0.0` score=`0.0`
- `fresh_resid_flow_rev_fl12_lb24_z125_imb10` family=`residual_reversion_flow_confirmed` success=`False` train=`0.0` val=`0.0` oos=`-1.9238247264352637e-05` score=`0.0`
- `fresh_funding_oi_fade_lb6_z175_f50_oi0` family=`funding_oi_carry_fade` success=`False` train=`0.0` val=`0.0` oos=`-0.002507104376249125` score=`0.0`
- `fresh_funding_oi_fade_lb6_z175_f100_oi0` family=`funding_oi_carry_fade` success=`False` train=`0.0` val=`0.0` oos=`-0.0009207350007364168` score=`0.0`
- `fresh_funding_oi_fade_lb6_z15_f50_oi0` family=`funding_oi_carry_fade` success=`False` train=`0.0` val=`0.0` oos=`-0.002507104376249125` score=`0.0`
- `fresh_funding_oi_fade_lb6_z15_f100_oi0` family=`funding_oi_carry_fade` success=`False` train=`0.0` val=`0.0` oos=`-0.0009207350007364168` score=`0.0`
- `fresh_funding_oi_fade_lb6_z125_f50_oi0` family=`funding_oi_carry_fade` success=`False` train=`0.0` val=`0.0` oos=`-0.002507104376249125` score=`0.0`
- `fresh_funding_oi_fade_lb6_z125_f100_oi0` family=`funding_oi_carry_fade` success=`False` train=`0.0` val=`0.0` oos=`-0.0009207350007364168` score=`0.0`
- `fresh_funding_oi_fade_lb48_z175_f50_oi0` family=`funding_oi_carry_fade` success=`False` train=`0.0` val=`0.0` oos=`-0.002925957636859655` score=`0.0`
- `fresh_funding_oi_fade_lb48_z175_f100_oi0` family=`funding_oi_carry_fade` success=`False` train=`0.0` val=`0.0` oos=`-0.0013513666515698741` score=`0.0`
- `fresh_funding_oi_fade_lb48_z15_f50_oi0` family=`funding_oi_carry_fade` success=`False` train=`0.0` val=`0.0` oos=`-0.002925957636859655` score=`0.0`
- `fresh_funding_oi_fade_lb48_z15_f100_oi0` family=`funding_oi_carry_fade` success=`False` train=`0.0` val=`0.0` oos=`-0.0013513666515698741` score=`0.0`
- `fresh_funding_oi_fade_lb48_z125_f50_oi0` family=`funding_oi_carry_fade` success=`False` train=`0.0` val=`0.0` oos=`-0.002925957636859655` score=`0.0`
- `fresh_funding_oi_fade_lb48_z125_f100_oi0` family=`funding_oi_carry_fade` success=`False` train=`0.0` val=`0.0` oos=`-0.0013513666515698741` score=`0.0`
- `fresh_funding_oi_fade_lb24_z175_f50_oi0` family=`funding_oi_carry_fade` success=`False` train=`0.0` val=`0.0` oos=`-0.0034336777569358157` score=`0.0`


## Verification evidence

- `uv run pytest -q tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_strategy_validity_audit.py tests/test_profit_moonshot_candidate_hybrid.py tests/test_profit_moonshot_liquidation_aware_validation.py` → 27 passed (`var/reports/profit_moonshot_20260501/live_final_selection_20260510/runner_evidence/targeted_strategy_validity_tests_time.log`)
- `uv run pytest` → 1253 passed (`var/reports/profit_moonshot_20260501/live_final_selection_20260510/runner_evidence/full_pytest_strategy_validity_time.log`)
- `uv run ruff check .` → All checks passed (`var/reports/profit_moonshot_20260501/live_final_selection_20260510/runner_evidence/ruff_strategy_validity_time.log`)
- `python3 -m compileall -q scripts tests` → passed (`var/reports/profit_moonshot_20260501/live_final_selection_20260510/runner_evidence/compileall_strategy_validity_time.log`)
- `git diff --check` → passed (`var/reports/profit_moonshot_20260501/live_final_selection_20260510/runner_evidence/git_diff_check_strategy_validity_time.log`)
