# Execution cost / turnover / theory verification

- Generated UTC: `2026-06-07T07:11:42Z`
- Ultragoal: `G006-independent-cost-turnover-theory-ver`

## Verdict

- Report-only audit: **pass with gaps documented**
- Real-money candidate: **fail**
- Small-sleeve candidate: **fail**

## Why

Required live gate is 10/15/20 bps plus turnover/RPT, capacity/liquidity, and paper fill telemetry. Current evidence is not complete enough for live deployment. Existing high-return labels are historical/shadow/control only.

## Theory gate

Accepted: time-series momentum, lagged volatility-managed exposure, execution-cost-aware implementation. Rejected: OOS-inspired parameters, arbitrary hardcoded rules, or 100% headline-driven tuning.

Sources:
- Time Series Momentum — https://www.aqr.com/insights/research/journal-article/time-series-momentum
- Volatility Managed Portfolios — https://www.nber.org/papers/w22208
- Trading Costs — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3229719
- Backtest overfitting / Pseudo-Mathematics — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659
