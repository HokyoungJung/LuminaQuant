# Full G005 all-strategy walk-forward alpha selection report

## 결론

- 방식: 월별 refit 시점마다 expanding train + 직전 2개월 validation으로 선별하고, 다음 1개월 locked OOS는 선별 후 report-only로만 평가.
- 대상: G005 supported all-strategy candidates across 30m/1h/4h/1d shards. TONUSDT excluded.
- shard reports loaded: 16; missing: 0.
- unique train/validation-selected candidates: 47; selected fold rows: 70.
- 1h Alpha101FormulaStrategy timeout-filtered fail-closed candidates: 4 per fold; accounted but not selected.
- **train/validation research-selected candidates: 22개**. locked OOS는 diagnostic only이며 선별/랭킹/상태에 쓰지 않았습니다.
- 선택 판정: `research_selected_candidates`.
- 배포 판정: `no_execution_promotion` / `research_only_no_execution`.

## Fold summary

| fold | evaluated | timeout-filtered | accounted | selected | selected OOS ret>0 & sharpe>0 | mean selected val ret | mean selected val sharpe | mean selected OOS ret | mean selected OOS sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WF202604 | 1400 | 4 | 1404 | 28 | 16 | +12.54% | 2.396 | -0.40% | -2.433 |
| WF202605 | 1400 | 4 | 1404 | 13 | 1 | +4.22% | 0.951 | -7.68% | -4.183 |
| WF202606 | 1400 | 4 | 1404 | 12 | 4 | +5.03% | 2.828 | +0.15% | -0.966 |
| WF202607_PARTIAL | 1400 | 4 | 1404 | 17 | 7 | +5.69% | 2.647 | +0.08% | -2.933 |

## Ranked candidates

| rank | candidate_id | alpha | class | tf | selected folds | val ret | val sharpe | OOS ret diagnostic | OOS sharpe diagnostic | OOS MDD diagnostic | status |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `19d07d85cab54789` | alpha101_formula_4h_a011_a011_flow_swing_dir | Alpha101FormulaStrategy | 4h | 3 | +10.68% | 2.944 | +3.07% | -2.576 | +4.86% | research_selected_train_validation |
| 2 | `632d6e864ee01bd8` | pair_spread_4h_fast_cycle_btcusdt_bnbusdt_1.6_0.35 | PairSpreadZScoreStrategy | 4h | 2 | +10.06% | 6.158 | -0.24% | -20.896 | +0.77% | research_selected_train_validation |
| 3 | `a27ef46e91376b7c` | pair_spread_4h_fast_cycle_btcusdt_bnbusdt_1.8_0.45 | PairSpreadZScoreStrategy | 4h | 2 | +9.61% | 5.322 | -0.14% | -22.303 | +0.85% | research_selected_train_validation |
| 4 | `1f1fd241c12f0bc2` | pair_spread_4h_balanced_btcusdt_bnbusdt_1.6_0.35 | PairSpreadZScoreStrategy | 4h | 2 | +9.72% | 5.409 | +0.28% | 0.720 | +0.14% | research_selected_train_validation |
| 5 | `914ff36cba1555ea` | pair_spread_4h_balanced_btcusdt_bnbusdt_2.0_0.50 | PairSpreadZScoreStrategy | 4h | 2 | +9.02% | 5.039 | -0.54% | -23.974 | +0.74% | research_selected_train_validation |
| 6 | `01b6c17e2d0faf32` | adaptive_regime_momentum_4h_profit_reboot_slow_defensive_60_6_0.020 | AdaptiveRegimeMomentumStrategy | 4h | 2 | +14.44% | 2.141 | -3.68% | -2.014 | +8.62% | research_selected_train_validation |
| 7 | `c9c210b24af36839` | profit_moonshot_trend_4h_slow_36_0.014 | ProfitMoonshotTrendStrategy | 4h | 2 | +14.44% | 2.141 | -3.68% | -2.014 | +8.62% | research_selected_train_validation |
| 8 | `81a321b38312a128` | profit_moonshot_breakout_4h_slow_expansion_30_0.008 | ProfitMoonshotBreakoutStrategy | 4h | 2 | +14.44% | 2.141 | -3.68% | -2.014 | +8.62% | research_selected_train_validation |
| 9 | `64ae2306e46aa279` | profit_moonshot_reversion_4h_slow_shock_fade_24_0.850 | ProfitMoonshotReversionStrategy | 4h | 2 | +14.44% | 2.141 | -3.68% | -2.014 | +8.62% | research_selected_train_validation |
| 10 | `78f5a77c2badecfe` | hurst_regime_gated_4h_balanced_lo_48_0.55 | HurstRegimeGatedStrategy | 4h | 2 | +14.44% | 2.141 | -3.68% | -2.014 | +8.62% | research_selected_train_validation |
| 11 | `7e64a5c8f39ca7e8` | flow_share_rotation_4h_balanced_lo | CrossSectionalFlowShareRotationStrategy | 4h | 2 | +14.44% | 2.141 | -3.68% | -2.014 | +8.62% | research_selected_train_validation |
| 12 | `6bb1fdf57477d280` | flow_share_rotation_4h_guarded_ls | CrossSectionalFlowShareRotationStrategy | 4h | 2 | +14.44% | 2.141 | -3.68% | -2.014 | +8.62% | research_selected_train_validation |
| 13 | `b9fefc414f9b57df` | regime_router_confirmed_4h_core_42_56 | RegimeRouterConfirmedRotationStrategy | 4h | 2 | +14.44% | 2.141 | -3.68% | -2.014 | +8.62% | research_selected_train_validation |
| 14 | `942a3646d139b779` | liquidation_cascade_reversion_4h_cascade_swing_0.04_6 | LiquidationCascadeReversionStrategy | 4h | 2 | +14.44% | 2.141 | -3.68% | -2.014 | +8.62% | research_selected_train_validation |
| 15 | `4cf80de7bc75ba7f` | selection_gated_momentum_4h_screened_swing_30_10 | SelectionGatedMomentumStrategy | 4h | 2 | +14.44% | 2.141 | -3.68% | -2.014 | +8.62% | research_selected_train_validation |
| 16 | `a41d21c3ceef9859` | selection_gated_reversion_4h_screened_fade_swing_4_10 | SelectionGatedReversionStrategy | 4h | 2 | +14.44% | 2.141 | -3.68% | -2.014 | +8.62% | research_selected_train_validation |
| 17 | `31c6486faa07ee45` | pair_spread_4h_participation_btcusdt_bnbusdt_1.6_0.35 | PairSpreadZScoreStrategy | 4h | 2 | +4.22% | 2.864 | -0.73% | -2.874 | +0.75% | research_selected_train_validation |
| 18 | `4a8bb636b1474f56` | pair_spread_4h_participation_btcusdt_bnbusdt_1.8_0.45 | PairSpreadZScoreStrategy | 4h | 2 | +3.28% | 2.347 | -0.01% | -0.422 | +0.55% | research_selected_train_validation |
| 19 | `34f87512bdf9b9dc` | pair_spread_1h_exec_tightstop_tp_btcusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | 1h | 2 | +1.69% | 1.204 | +0.08% | -0.264 | +1.41% | research_selected_train_validation |
| 20 | `94c8b5762e0cfbaf` | pair_spread_1h_core_btcusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | 1h | 2 | +1.68% | 1.201 | +0.08% | -0.264 | +1.41% | research_selected_train_validation |
| 21 | `ae745773a979b939` | regime_breakout_1h_trend_ls_48_0.70 | RegimeBreakoutCandidateStrategy | 1h | 2 | +3.63% | 0.758 | -4.13% | 1.774 | +5.75% | research_selected_train_validation |
| 22 | `9fc68016d5aeef0c` | regime_breakout_30m_trend_guarded_48_0.68 | RegimeBreakoutCandidateStrategy | 30m | 2 | +2.91% | 0.861 | -0.77% | -1.139 | +4.08% | research_selected_train_validation |
| 23 | `182385a51fbcc120` | idiosyncratic_volatility_4h_ivol_lo_120_60 | IdiosyncraticVolatilityStrategy | 4h | 1 | +25.22% | 3.560 | +1.42% | 0.596 | +8.41% | selected_once_train_validation |
| 24 | `eb4914335e5c4aec` | composite_trend_stable_30m_stable_ls_exec_trail_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | 30m | 1 | +3.73% | 1.749 | -3.69% | -11.745 | +3.83% | selected_once_train_validation |
| 25 | `886a901633fbf1e7` | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | 30m | 1 | +3.81% | 1.739 | -4.03% | -12.194 | +4.19% | selected_once_train_validation |
| 26 | `b271c1c63cbc2825` | composite_trend_stable_30m_stable_ls_exec_shorthold_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | 30m | 1 | +3.79% | 1.800 | -3.61% | -11.681 | +3.75% | selected_once_train_validation |
| 27 | `5f0de3daa56f9743` | pair_spread_4h_fast_cycle_btcusdt_bnbusdt_2.2_0.55 | PairSpreadZScoreStrategy | 4h | 1 | +2.94% | 2.687 | +0.00% | 0.000 | +0.00% | selected_once_train_validation |
| 28 | `1b67a7f671c3c91a` | composite_trend_stable_30m_stable_ls_crashguard_ls_0.75_0.45_0.20_0.82 | CompositeTrendStrategy | 30m | 1 | +3.46% | 1.661 | -3.25% | -11.306 | +3.46% | selected_once_train_validation |
| 29 | `c01a3510bb5eab15` | pair_spread_1h_exec_tightstop_tp_btcusdt_trxusdt_2.6_0.70 | PairSpreadZScoreStrategy | 1h | 1 | +3.88% | 2.527 | +0.00% | 0.000 | +0.00% | selected_once_train_validation |
| 30 | `f54d2780eb950ba8` | pair_spread_1h_exec_takeprofit_btcusdt_trxusdt_2.6_0.70 | PairSpreadZScoreStrategy | 1h | 1 | +3.88% | 2.527 | +0.00% | 0.000 | +0.00% | selected_once_train_validation |
| 31 | `bc5b361892847da3` | composite_trend_stable_30m_stable_ls_core_ls_0.60_0.45_0.20_0.80 | CompositeTrendStrategy | 30m | 1 | +3.29% | 1.373 | -4.76% | -11.640 | +5.10% | selected_once_train_validation |
| 32 | `1523ae68946b002a` | pair_spread_1h_core_btcusdt_trxusdt_2.6_0.70 | PairSpreadZScoreStrategy | 1h | 1 | +3.88% | 2.527 | +0.00% | 0.000 | +0.00% | selected_once_train_validation |
| 33 | `ab940796e8245b88` | pair_spread_1h_state_atr_btcusdt_trxusdt_2.6_0.70 | PairSpreadZScoreStrategy | 1h | 1 | +3.88% | 2.527 | +0.00% | 0.000 | +0.00% | selected_once_train_validation |
| 34 | `346eaec5b7f79e2d` | pair_spread_1h_state_vwap_btcusdt_trxusdt_2.6_0.70 | PairSpreadZScoreStrategy | 1h | 1 | +3.88% | 2.527 | +0.00% | 0.000 | +0.00% | selected_once_train_validation |
| 35 | `a1912ebf2f4457db` | pair_spread_4h_fast_cycle_btcusdt_bnbusdt_2.0_0.50 | PairSpreadZScoreStrategy | 4h | 1 | +2.50% | 2.679 | +0.00% | 0.000 | +0.00% | selected_once_train_validation |
| 36 | `7da189bdfded8ca7` | carry_trend_factor_rotation_1h_guarded_ls_32_8_0.150 | CarryTrendFactorRotationStrategy | 1h | 1 | +4.60% | 1.808 | +0.55% | 8.518 | +0.65% | selected_once_train_validation |
| 37 | `fa62852b1894fd94` | rolling_breakout_1h_guarded_ls_48_0.002 | RollingBreakoutStrategy | 1h | 1 | +10.74% | 1.696 | -0.60% | -7.648 | +1.40% | selected_once_train_validation |
| 38 | `8d8d50f568a3eb3a` | last_day_liquidity_regime_1h_liquid_momo_ls_24_6_0.012 | LastDayLiquidityRegimeStrategy | 1h | 1 | +3.46% | 2.395 | -3.91% | -4.459 | +5.91% | selected_once_train_validation |
| 39 | `231ff8a644aa4d76` | composite_trend_stable_30m_stable_ls_tefilter_ls_0.60_0.45_0.25_0.80 | CompositeTrendStrategy | 30m | 1 | +1.49% | 0.609 | -3.01% | -12.582 | +3.39% | selected_once_train_validation |
| 40 | `0af8d5a205263172` | deep_research_vol_managed_momentum_crash_gate_1h_balanced_ls_96_0.018 | VolManagedMomentumCrashGateStrategy | 1h | 1 | +2.99% | 1.431 | +1.74% | 31.446 | +0.29% | selected_once_train_validation |

## Locked OOS diagnostic policy

- Locked OOS는 fold별 train+validation 선별이 고정된 뒤 붙인 성능 진단값입니다.
- Locked OOS return/sharpe/MDD는 selection status, rank ordering, repeated-selection 여부에 사용하지 않습니다.

## Safety

- live_execution_enabled: `False`
- paper_execution_enabled: `False`
- testnet_execution_enabled: `False`
- real_money_execution_enabled: `False`
- orders_enabled: `False`
- tonusdt_excluded: `True`

JSON: `/home/hoky/Quants-agent/LuminaQuant/var/reports/latest_alpha_refresh_20260704_full_walkforward/full_all_strategy_walkforward_selection_latest.json`
CSV: `/home/hoky/Quants-agent/LuminaQuant/var/reports/latest_alpha_refresh_20260704_full_walkforward/full_all_strategy_walkforward_selection_latest.csv`
