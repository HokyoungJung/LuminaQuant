# G005 partial walk-forward evaluation attempt report

- status: `in_progress_monitored`
- partial evaluated candidates: `72`
- monitored run: `/home/hoky/Quants-agent/LuminaQuant/var/reports/ultragoal_full_pool_strategy/g005_latest_fold_candidate_eval_monitored` (`bg_1`)

## Top validation among partial evaluated candidates
1. `abnormal_return_continuation_1d_event_ls_solusdt_1.4_2` val_return=0.042435 val_sharpe=1.092 oos_return_report_only=0.040169
2. `last_day_liquidity_regime_1h_liquid_momo_ls_24_6_0.012` val_return=0.034578 val_sharpe=2.395 oos_return_report_only=-0.044118
3. `last_day_liquidity_regime_1d_guarded_lo_1_1_0.006` val_return=0.030983 val_sharpe=2.030 oos_return_report_only=-0.012156
4. `mean_reversion_std_30m_balanced_ls_48_1.80` val_return=0.027860 val_sharpe=0.665 oos_return_report_only=-0.042126
5. `last_day_liquidity_regime_1d_liquid_momo_ls_1_1_0.008` val_return=0.009520 val_sharpe=0.382 oos_return_report_only=-0.038775
6. `mean_reversion_std_30m_guarded_lo_72_2.20` val_return=0.004375 val_sharpe=-0.022 oos_return_report_only=-0.084183
7. `funding_liquidation_crowding_fade_30m_balanced_ls_96_0.85` val_return=0.000000 val_sharpe=0.000 oos_return_report_only=0.000000
8. `funding_liquidation_crowding_fade_30m_guarded_lo_128_1.00` val_return=0.000000 val_sharpe=0.000 oos_return_report_only=0.000000
9. `abnormal_return_continuation_1d_event_lo_trxusdt_1.8_1` val_return=-0.004605 val_sharpe=-2.369 oos_return_report_only=0.000000
10. `abnormal_return_continuation_1d_event_ls_dogeusdt_1.4_2` val_return=-0.004982 val_sharpe=-0.297 oos_return_report_only=0.008717

## Blocker
Full G005 evaluation has not produced final candidate_research_latest.json yet; monitored run must finish before G005 can be checkpointed complete.
