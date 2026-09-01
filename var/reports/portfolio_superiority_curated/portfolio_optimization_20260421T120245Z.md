# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/portfolio_superiority_curated/candidate_research_curated_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 9
- Gross exposure: `60.83%`
- Cash weight: `39.17%`

## Sleeve budgets

- cross_sectional: 18.85%
- event_alpha: 20.54%
- market_neutral: 13.44%
- mean_reversion: 4.00%
- trend: 4.00%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | abnormal_return_continuation_1d_event_ls_ethusdt_1.4_2 | AbnormalReturnContinuationStrategy | event_alpha | 1d | 11.11% | 0.000 | 0.00% | 0.000 | 0.00% |
| 2 | carry_trend_factor_rotation_1h_balanced_lo_24_8_0.200 | CarryTrendFactorRotationStrategy | cross_sectional | 1h | 9.42% | 0.000 | 0.00% | 0.000 | 0.00% |
| 3 | last_day_liquidity_regime_1h_liquid_momo_ls_24_6_0.012 | LastDayLiquidityRegimeStrategy | cross_sectional | 1h | 9.42% | 0.000 | 0.00% | 0.000 | 0.00% |
| 4 | abnormal_return_continuation_1d_event_ls_btcusdt_1.4_2 | AbnormalReturnContinuationStrategy | event_alpha | 1d | 9.42% | 0.000 | 0.00% | 0.000 | 0.00% |
| 5 | pair_spread_1h_exec_tightstop_tp_xauusdt_xagusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 4.72% | 0.000 | 0.00% | 0.000 | 0.00% |
| 6 | pair_spread_1h_exec_tightstop_tp_xptusdt_xpdusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 4.72% | 0.000 | 0.00% | 0.000 | 0.00% |
| 7 | composite_trend_stable_30m_stable_ls_crashguard_ls_0.75_0.45_0.20_0.82 | CompositeTrendStrategy | trend | 30m | 4.00% | 0.000 | 0.00% | 0.000 | 0.00% |
| 8 | pair_spread_1h_exec_tightstop_tp_btcusdt_xauusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 4.00% | 0.000 | 0.00% | 0.000 | 0.00% |
| 9 | mean_reversion_std_30m_guarded_lo_72_2.20 | MeanReversionStdStrategy | mean_reversion | 30m | 4.00% | 0.000 | 0.00% | 0.000 | 0.00% |

JSON: `/home/hoky/Quants-agent/LuminaQuant/var/reports/portfolio_superiority_curated/portfolio_optimization_20260421T120245Z.json`
Latest: `/home/hoky/Quants-agent/LuminaQuant/var/reports/portfolio_superiority_curated/portfolio_optimization_latest.json`

