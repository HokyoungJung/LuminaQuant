# Profit moonshot alpha-v2 completion plan/result — 2026-05-09

> **Superseded on 2026-05-09:** this file records the old alpha-v2 target-budget result. After the user clarified that return must be stable around `+2%` monthly and MDD can be tolerated up to `25%`, the candidate is no longer a pass. See `.omx/plans/profit_moonshot_return_quality_contract_20260509.md` and `docs/session_handoff_20260509_profit_moonshot_return_quality_reassessment.md`.


## Stop condition
Complete only when the passing candidate is committed/pushed to `private/main`, local verification is green, GitHub Actions `ci` and `private-ci` are green, and the performance-goal evaluator passes.

## Integrated team decision
Team `profit-moonshot-alpha-56afab4e` is `phase=complete` with 13/13 tasks completed and 0 failed. Worker recommendations were reconciled as follows:
- H6 quarantine stays mandatory: high locked-OOS/MDD-failed diagnostics are `diagnostic_not_promoted`.
- Residual-pair reversion, inverse/momentum residual-pair, and compression downside probes are not promotable under train/validation-first evidence.
- The only defensible return-improvement lane was calendar allocator risk budgeting using train/validation only.

## Implemented result
- Added `train_val_target_return_budget` allocator mode to scale additive calendar sleeves to train `>=5%` / validation `>=4%` targets without using locked-OOS selection.
- Added explicit `best_success_candidate` output so the passing row is separate from selected-by-validation diagnostics.
- Passing artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/passing_candidate_latest.json`.
- Best success candidate: `fresh_portfolio_train_val_target_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls590_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls560_ss120_tp600`.
- Train `+5.5019%`, validation `+4.0000%`, locked-OOS `+1.3397%`, locked-OOS MDD `+0.1774%`, locked-OOS return/risk `7.5515`.
- Champion reference: OOS `+1.2181%`, MDD `+0.1662%`, return/risk `7.3291`.
- Note: candidate MDD is below the shadow gate but slightly above the old champion MDD; return/risk ratio is better and all configured gates pass.

## Policy preservation
- `uses_locked_oos_for_selection=false` in allocator diagnostics.
- Locked-OOS is report/gate-only.
- Standalone non-calendar families remain diagnostic/no-go; no OOS-ranked row is promoted.
- Memory guard retained under the 8 GiB cap.

## Verification so far
- Targeted alpha-v2 tests: `9 passed in 0.10s`.
- Full pytest: `1206 passed in 234.23s (0:03:54)`.
- Ruff: `uv run --extra dev ruff check .` -> passed.
- Compileall: `python3 -m compileall -q src scripts tests` -> passed.
- Whitespace: `git diff --check` -> passed.
- Source Lore commit `b340d38` pushed to `private/main`.
- GitHub Actions: `ci` `25593434673` success; `private-ci` `25593434668` success.
- Mission validator: updated after CI and expected to pass.
