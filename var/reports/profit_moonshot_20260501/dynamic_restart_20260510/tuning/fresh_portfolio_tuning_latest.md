# Profit moonshot fresh portfolio tuning

Generated: `2026-05-10T11:23:41.435812Z`

## Policy

- Sleeve universe is restricted to train-positive and validation-positive fresh-start candidates.
- Portfolio selection is train/validation-stability primary; locked-OOS is report-only / gate-only.
- `diagnostic_best_oos` is not a deployable selection if it differs from validation selection.
- Selection label: `train_val_validation_only`.
- Locked-OOS label: `locked_oos_report_only`.
- Locked-OOS gate label: `locked_oos_gate_only`.
- Diagnostic quarantine label: `diagnostic_not_promoted`.
- Current-base artifact: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/passing_candidate_latest.json`.
- Current-base status: `loaded`.
- Train/validation stability objective: `frozen_weighted_train_validation_score_v1` (current base `16.576134`).
- No-improvement lifecycle: `no_improvement_current_base_retained`.
- Stable-return floor: train, validation, and locked-OOS monthlyized return `>=2.00%`.
- Train buffer: post-leverage train monthlyized return `>=2.25%` and raw/unlevered train monthlyized return `>=1.00%`.
- Leverage policy: `train_val_monthly_return_budget` uses an integer train/validation-only grid; continuous floor-fitting leverage is diagnostic only.
- MDD budget: locked-OOS max drawdown `≤25.00%`.
- Quality floors: OOS Sharpe `≥2.0`, Sortino `≥3.0`, smart Sortino `≥3.0`, Calmar `≥1.0`.
- Incumbent improvement still requires current-champion return/risk improvement from OOS return `>1.2181%`.

## Runtime guard

- Heavy-run lock: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/portfolio_followup_heavy_run.lock`
- Explicit memory budget: `6979321856` bytes
- RSS summary: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/dynamic_restart_20260510/tuning/_memory_guard/profit_moonshot_fresh_portfolio_tuning_memory_latest.json`

## Summary

- Candidate sleeves considered: `0`
- Portfolio specs evaluated: `0`
- Combo cap per size: `500`; skipped by size: `{}`
- Success candidates: `0`
- Peak RSS: `251.977 MiB`

## Top rows

| rank | name | mode | leverage | success | train | val | locked OOS | OOS monthly | OOS MDD | OOS Sharpe | smart Sortino | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|

