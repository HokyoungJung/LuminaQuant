# Profit moonshot alpha-v2 completion summary — 2026-05-09

## Verdict
- Honest passing candidate found via `train_val_target_return_budget` calendar sleeve allocator.
- No standalone non-calendar family was promoted; residual-pair and compression probes remain diagnostic/no-go.
- Locked-OOS remains report-only/gate-only; allocator diagnostics record `uses_locked_oos_for_selection=false`.
- Peak RSS evidence remains below 8 GiB: payload `746.188 MiB`, memory guard `736.219 MiB`.

## Passing candidate
- Artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/passing_candidate_latest.json`
- Portfolio source: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_target_budget_v1/fresh_portfolio_tuning_latest.json`
- Candidate: `fresh_portfolio_train_val_target_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls560_ss120_tp600`
- Mode/leverage: `train_val_target_return_budget` / `0.351825`
- Train return: `+5.5019%`; validation return: `+4.0000%`
- Locked-OOS return: `+1.3397%` > champion `+1.2181%`
- Locked-OOS MDD: `+0.1774%` < shadow gate `+0.1778%`
- Locked-OOS return/risk: `7.5515` > champion `7.3291`
- Locked-OOS Sharpe: `5.4774`; round trips: `27`
- Promotion status: `improved_success_candidate`; failed gates: `[]`

## Search evidence
- Residual pair reversion: 1,296 specs, 0 success; no promotion.
- Residual pair momentum: 432 specs, 0 success; no promotion.
- Compression downside short: 288 specs, 0 success; no promotion.
- Target-budget portfolio: `62970` specs, `38` success candidates, best success is calendar-only.

## Verification status
- Targeted alpha-v2 tests: `9 passed in 0.10s`.
- Full pytest: `1206 passed in 234.23s (0:03:54)`.
- Full ruff, compileall, and git diff check passed. Source Lore commit `b340d38` was pushed to `private/main`; GitHub Actions `ci` run `25593434673` and `private-ci` run `25593434668` completed `success`.
