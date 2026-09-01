# Profit moonshot integer-leverage audit — 2026-05-09

## Decision
No new candidate was promoted. The current base remains the reference result because every wider integer-leverage challenger failed at least one promotion gate.

## What was checked
- `train_val_monthly_return_budget` now uses an integer train/validation-only leverage grid.
- Continuous leverage needed to hit the monthly floor is diagnostic only.
- Raw/unlevered train and validation monthlyized returns are recorded and gated for new promotions.
- Locked-OOS remains report-only/gate-only; it is never used for selection.
- Heavy runs were serialized with the portfolio follow-up mutex and stayed below 8 GiB.

## Wider top40 result
- Artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/integer_leverage_alpha_v2_top40_20260509/fresh_portfolio_tuning_latest.json`
- Specs: `158,620`; sleeves: `40`; success candidates: `0`.
- Peak RSS: `2523.5234 MiB`; `/usr/bin/time` max RSS `2,584,088 KiB`; wall `22:34.38`.
- Selected by train/validation: integer leverage `3`, train monthly `+2.7510%`, validation monthly `+11.8767%`, raw train `+1.0111%`, OOS return `+6.7836%`; rejected because OOS return/risk did not beat current base.
- Diagnostic best OOS: integer leverage `6`, OOS return `+18.4446%`, MDD `+2.2305%`, return/risk `8.2694`; rejected because raw train was only `+0.8615%`, train Sortino failed, and OOS return/risk did not beat current base.

## Current-base integer stress test
Forced current-base sleeve tuple under integer leverage selected `5x`:
- Train monthly `+3.8443%`; validation monthly `+20.1014%`; OOS monthly `+6.4641%`.
- OOS return `+14.6371%`; MDD `+1.6919%`; return/risk `8.6514`; Sharpe `5.7215`; Sortino `7.4828`; smart Sortino `6.9764`; Calmar `66.2284`.
- Rejected despite attractive OOS because raw train monthly was `+0.9075%` and train Sortino was `1.4814`, below the stricter raw/train-quality audit.

## Leverage verdict
The audit confirms the suspicious behavior: increasing integer leverage can make OOS return look much better, but it is not automatically valid. Rows that looked best after leverage were quarantined when raw train quality, train Sortino, or current-base return/risk gates failed.

Large generated CSVs were pruned from git; compact JSON/markdown/memory/time evidence is retained.
