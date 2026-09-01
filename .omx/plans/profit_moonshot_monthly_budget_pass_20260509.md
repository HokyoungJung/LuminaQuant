# Profit moonshot monthly-budget pass plan/result — 2026-05-09

## Objective
Satisfy the clarified user target: roughly `+2%` average monthly return, MDD acceptable up to `25%`, and high Sharpe/Sortino/smart Sortino/Calmar without locked-OOS leakage or standalone diagnostic promotion.

## Implementation
- Added `train_val_monthly_return_budget` allocator to `scripts/research/tune_profit_moonshot_fresh_portfolio.py`.
- The allocator computes leverage from train/validation evidence only, targeting `+2%` monthlyized return while capping train/validation MDD by the `25%` budget.
- Added train/validation quality gates in addition to OOS quality gates:
  - Train Sharpe `>=1.5`, Sortino `>=1.5`, Calmar `>=1.0`.
  - Validation Sharpe `>=3.0`, Sortino `>=3.0`, Calmar `>=3.0`.
  - OOS Sharpe `>=2.0`, Sortino `>=3.0`, smart Sortino `>=3.0`, Calmar `>=1.0`.
- Updated validator to enforce the full return-quality contract.

## Result
Best passing candidate:
`fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls530_ss120_tp600`

Metrics:
- Train `+26.8207%`, monthlyized `+2.0000%`, MDD `+6.9060%`, Sharpe `1.7213`, Sortino `1.5151`, Calmar `3.8842`.
- Validation `+19.9713%`, monthlyized `+9.8490%`, MDD `+6.4935%`, Sharpe `4.0964`, Sortino `4.8859`, Calmar `32.1417`.
- Locked-OOS `+6.8582%`, monthlyized `+3.0883%`, MDD `+0.8198%`, Sharpe `5.6537`, Sortino `7.3961`, smart Sortino `7.1536`, Calmar `53.7350`.

## Evidence
- Replay: `73,465` specs, `8` success candidates, peak RSS `920.51171875 MiB`; `/usr/bin/time` max RSS `942604 KB`.
- Artifacts: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_monthly_budget_v1/`.
- Passing candidate: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/passing_candidate_latest.json`.
- Validator: return-quality and RSS checks pass; final pass awaits full tests/push/CI evidence.

## Final verification closure
- Local: targeted tests `14 passed`; full pytest `1208 passed in 289.27s`; full ruff passed; compileall passed; `git diff --check` passed.
- Source commit: `a1f4e0718975202f416b3e76e358c46a6a3d7ae8` pushed to `private/main`.
- GitHub Actions: `ci` `25594783480` success; `private-ci` `25594783476` success.
- Mission validator: final pass recorded in `.omx/specs/autoresearch-profit-moonshot-alpha-v2/validation_latest.json`.
