# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/portfolio_superiority_curated/candidate_research_curated_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 9
- Gross exposure: `125.80%`
- Cash weight: `0.00%`

## Sleeve budgets

- cross_sectional: 28.09%
- event_alpha: 58.06%
- market_neutral: 22.28%
- mean_reversion: 12.75%
- trend: 4.63%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | abnormal_return_continuation_1d_event_ls_ethusdt_1.4_2 | AbnormalReturnContinuationStrategy | event_alpha | 1d | 29.18% | 4.657 | 19.53% | -3.870 | -26.79% |
| 2 | abnormal_return_continuation_1d_event_ls_btcusdt_1.4_2 | AbnormalReturnContinuationStrategy | event_alpha | 1d | 28.88% | -2.815 | -2.40% | -3.260 | -20.37% |
| 3 | carry_trend_factor_rotation_1h_balanced_lo_24_8_0.200 | CarryTrendFactorRotationStrategy | cross_sectional | 1h | 28.09% | 2.528 | 2.61% | -1.190 | -3.27% |
| 4 | pair_spread_1h_exec_tightstop_tp_xauusdt_xagusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 11.14% | -5.850 | -0.85% | 1.559 | 2.14% |
| 5 | pair_spread_1h_exec_tightstop_tp_xptusdt_xpdusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 11.14% | -18.174 | -0.08% | -0.918 | -0.46% |
| 6 | mean_reversion_std_30m_guarded_lo_72_2.20 | MeanReversionStdStrategy | mean_reversion | 30m | 10.43% | 2.375 | 0.04% | -1.330 | -4.53% |
| 7 | composite_trend_stable_30m_stable_ls_crashguard_ls_0.75_0.45_0.20_0.82 | CompositeTrendStrategy | trend | 30m | 2.32% | 0.000 | 0.00% | -0.602 | -0.42% |
| 8 | volcomp_vwap_rev_guarded_15m_guarded_lo_strict_2.40_0.12 | VolCompressionVWAPReversionStrategy | mean_reversion | 15m | 2.32% | 0.000 | 0.00% | -6.961 | -1.51% |
| 9 | composite_trend_stable_1h_stable_lo_guarded_lo_0.75_0.60_0.25_0.95 | CompositeTrendStrategy | trend | 1h | 2.32% | 0.000 | 0.00% | -10.067 | -0.21% |

JSON: `/home/hoky/Quants-agent/LuminaQuant/var/reports/portfolio_superiority_curated/portfolio_optimization_20260421T130301Z.json`
Latest: `/home/hoky/Quants-agent/LuminaQuant/var/reports/portfolio_superiority_curated/portfolio_optimization_latest.json`

