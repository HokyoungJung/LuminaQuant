# Session handoff — profit moonshot continued optimization — 2026-05-09

## Status
The validated continuation baseline is the monthly-budget portfolio pass. This handoff is additive and does not rewrite completed historical notes. No final heavy replay was authorized or started in this lane; worker-2 preserved bounded smoke, validator, candidate-metric, and memory evidence only.

## Current baseline candidate
- Candidate artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/passing_candidate_latest.json`
- Source replay: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_monthly_budget_v1/fresh_portfolio_tuning_latest.json`
- Runner/evidence summary: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/runner_evidence_20260509/worker2_runner_evidence.md`
- Candidate name: `fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls620_ss120_tp600__fresh_calendar_trx_takeprofit_sethusdt_thr120_h120_ls620_ss120_tp600__fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_calendar_trx_takeprofit_sethusdt_thr180_h168_ls530_ss120_tp600`
- Mode/leverage: `train_val_monthly_return_budget` / `2.3427334297703024`
- Promotion: `improved_success_candidate`; `success_candidate=true`; failed gates `[]`
- Selection policy: train/validation only; locked-OOS report-only / gate-only; `uses_locked_oos_for_selection=false`

## Metrics to preserve
| Split | Return | Monthlyized | MDD | Sharpe | Sortino | Smart Sortino | Calmar | Round trips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | `+26.8207%` | `+2.0000%` | `+6.9060%` | `1.7213` | `1.5151` | n/a | `3.8842` | `120` |
| Validation | `+19.9713%` | `+9.8490%` | `+6.4935%` | `4.0964` | `4.8859` | n/a | `32.1417` | `41` |
| Locked-OOS | `+6.8582%` | `+3.0883%` | `+0.8198%` | `5.6537` | `7.3961` | `7.1536` | `53.7350` | `33` |

Champion reference: prior locked-OOS return was `+1.2181%`; this candidate's locked-OOS excess is `+5.6401%` absolute return.

## Memory and bounded-run evidence
- Portfolio specs evaluated: `73,465`; success candidates: `8`.
- Payload peak RSS: `920.51171875 MiB`.
- Memory guard: `955,871,232` bytes / `911.58984375 MiB`, under the `8 GiB` mission cap, with `0` soft and `0` hard memory triggers.
- Worker-2 smoke command: `uv run python scripts/minimum_viable_run.py --days 30` -> exit `0`, wall `0:01.06`, max RSS `134272 KB`.
- Worker-2 artifact validator command: `uv run python scripts/research/validate_profit_moonshot_pass_under_8gb.py --result-path .omx/specs/autoresearch-profit-moonshot-alpha-v2/result.json` -> `status=passed`, `passed=true`.

## Historical evidence boundaries
- The target-budget alpha-v2 notes are superseded by the monthly-budget pass for current promotion language:
  - `.omx/plans/profit_moonshot_alpha_v2_completion_20260509.md`
  - `docs/session_handoff_20260509_profit_moonshot_alpha_v2_target_budget.md`
- The stale return-quality validation snapshot is historical and should not override the later pass:
  - `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/return_quality_validation_20260509.md`
- Do not promote standalone diagnostic rows. Portfolio promotion still requires train/validation-only selection plus locked-OOS gate passage.

## Verification state
- Compile/typecheck equivalent: `python3 -m compileall -q src scripts tests` -> pass.
- Targeted tests: `uv run --extra dev pytest -q tests/test_profit_moonshot_pass_under_8gb_validator.py tests/test_profit_moonshot_fresh_portfolio_tuning.py` -> `14 passed in 0.08s`.
- Related lint: `uv run --extra dev ruff check scripts/minimum_viable_run.py scripts/research/validate_profit_moonshot_pass_under_8gb.py tests/test_profit_moonshot_pass_under_8gb_validator.py tests/test_profit_moonshot_fresh_portfolio_tuning.py` -> all checks passed.
- Whitespace/regression check: `git diff --check` -> pass.
- Final integration, full verification, CI push, and any authorized heavy replay remain leader/Ralph responsibilities.

## Next handoff
Use this monthly-budget candidate as the continuation baseline. Search for additional stable return drivers only if they preserve the `+2%` monthlyized train/validation/OOS contract, relaxed `<=25%` locked-OOS MDD budget, high risk-adjusted gates, and train/validation-only selection. If a heavy replay is needed, run exactly one leader-authorized heavy job at a time.

## Final leader/Ralph continuation result — 2026-05-09
- Final outcome: `no_improvement_current_base_retained`; no new challenger was promoted.
- Continued artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/continued_optimization_20260509/fresh_portfolio_tuning_latest.json`.
- Retained-base candidate artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/continued_optimization_20260509/passing_candidate_latest.json`.
- Selection remained train/validation-only with frozen H2 score `frozen_weighted_train_validation_score_v1`; locked-OOS stayed report-only/gate-only.
- Base stability score: `16.576134`; retained base locked-OOS return/risk: `+6.8582%` / `8.365933`.
- Best train/validation challenger score: `26.678861`; it was quarantined because failed gates `['train_sortino_high', 'oos_return_risk_beats_current_champion', 'oos_return_risk_beats_current_base']` (notably current-base OOS return/risk was not improved enough).
- Replay scale: `73465` specs; success candidates `0`; peak RSS `1239.703 MiB`; `/usr/bin/time` max RSS `1269456 KB`.
- Heavy mutex evidence: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/runner_evidence_20260509/leader_heavy_mutex_20260509.json`; status `completed`, overlap `passed`.
- Full local verification: targeted tests `22 passed`; full pytest `1216 passed in 227.75s (0:03:47)`; ruff `All checks passed`; compileall pass; git diff --check pass.
- CI status: pending first push for this continuation commit.
- Push/CI finalization: Lore commit `6556254726db4f70fe8e7eb6d72a6fe9a352c78a` pushed to `private/main`; GitHub Actions `private-ci` run `25596474837` success and `ci` run `25596474834` success.
- Mission validator finalization: `.omx/specs/autoresearch-profit-moonshot-alpha-v2/validation_latest.json` now `status=passed`, `passed=true`, including CI, git push, mutex, and RSS-under-8GiB evidence.
