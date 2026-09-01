# Profit moonshot alpha-v2 target-budget handoff — 2026-05-09

> **Superseded on 2026-05-09:** this file records the old alpha-v2 target-budget result. After the user clarified that return must be stable around `+2%` monthly and MDD can be tolerated up to `25%`, the candidate is no longer a pass. See `.omx/plans/profit_moonshot_return_quality_contract_20260509.md` and `docs/session_handoff_20260509_profit_moonshot_return_quality_reassessment.md`.


## Status
A passing research candidate was found without promoting standalone non-calendar alpha or locked-OOS diagnostics. The candidate is a drawdown/return-budgeted calendar sleeve portfolio selected with train/validation-only inputs.

## Key paths
- Candidate: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/passing_candidate_latest.json`
- Portfolio JSON/CSV/MD: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_target_budget_v1/fresh_portfolio_tuning_latest.json`, `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_target_budget_v1/fresh_portfolio_tuning_candidates.csv`, `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_target_budget_v1/fresh_portfolio_tuning_latest.md`
- Completion summary: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_v2_completion_summary_20260509.md`
- OMX plan/result: `.omx/plans/profit_moonshot_alpha_v2_completion_20260509.md`
- Mission result: `.omx/specs/autoresearch-profit-moonshot-alpha-v2/result.json` (CI/push evidence is updated after GitHub Actions completes)

## Candidate metrics
- Name: `fresh_portfolio_train_val_target_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls560_ss120_tp600`
- Sleeves: `fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600, fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls560_ss120_tp600`
- Mode/leverage: `train_val_target_return_budget` / `0.351825`
- Train: `+5.5019%`; validation: `+4.0000%`
- Locked-OOS: `+1.3397%`; MDD `+0.1774%`; Sharpe `5.4774`; return/risk `7.5515`
- Champion: OOS `+1.2181%`; MDD `+0.1662%`; return/risk `7.3291`
- Gate result: `success_candidate=true`, `promotion_status=improved_success_candidate`, failed gates `[]`.

## Non-calendar/probe disposition
- `residual_pair_v1`: 1,296 specs, 0 success; diagnostic only.
- `residual_pair_momentum_v1`: 432 specs, 0 success; diagnostic only.
- `compression_downside_v1`: 288 specs, 0 success; diagnostic only.
- Prior `portfolio_v1` selected/OOS-ranked diagnostics remain quarantined.

## Verification
- Targeted tests: `9 passed in 0.10s`.
- Full pytest: `1206 passed in 234.23s (0:03:54)`.
- Portfolio target-budget run: 62,970 specs, 38 success candidates, peak RSS `746.188 MiB`; `/usr/bin/time` max RSS observed `764096 KB`.
- Ruff: `uv run --extra dev ruff check .` -> passed.
- Compileall: `python3 -m compileall -q src scripts tests` -> passed.
- Whitespace: `git diff --check` -> passed.
- Source Lore commit `b340d38` pushed to `private/main`.
- GitHub Actions: `ci` `25593434673` success; `private-ci` `25593434668` success.
- Mission validator: updated after CI and expected to pass.

## Risks / directives
- Do not promote `selected_by_validation` if it is `diagnostic_not_promoted`; use `best_success_candidate` for passing row evidence.
- Candidate MDD passes the shadow gate but is slightly above the old champion's MDD; preserve the configured return/risk + shadow-MDD gate semantics unless requirements change.
- Treat locked-OOS strictly as report/gate-only; never use it to choose among candidates.
