# Profit moonshot continued optimization plan/result — 2026-05-09

## Objective
Preserve a continuation handoff for the validated profit-moonshot monthly-budget pass while keeping old alpha-v2 target-budget notes historical. This lane is docs/evidence-only: no tuning code, replay code, or final heavy replay was changed or started.

## Current result to carry forward
The current pass is the monthly-budget portfolio candidate, not the superseded target-budget candidate.

- Candidate artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/passing_candidate_latest.json`
- Source replay artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_monthly_budget_v1/fresh_portfolio_tuning_latest.json`
- Runner evidence: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/runner_evidence_20260509/worker2_runner_evidence.md`
- Mode/leverage: `train_val_monthly_return_budget` / `2.3427334297703024`
- Promotion status: `improved_success_candidate`; `success_candidate=true`; failed gates `[]`
- Selection basis: train/validation only; locked-OOS is report-only / gate-only; `uses_locked_oos_for_selection=false`

## Metrics snapshot
- Train: return `+26.8207%`, monthlyized `+2.0000%`, MDD `+6.9060%`, Sharpe `1.7213`, Sortino `1.5151`, Calmar `3.8842`, round trips `120`.
- Validation: return `+19.9713%`, monthlyized `+9.8490%`, MDD `+6.4935%`, Sharpe `4.0964`, Sortino `4.8859`, Calmar `32.1417`, round trips `41`.
- Locked-OOS: return `+6.8582%`, monthlyized `+3.0883%`, MDD `+0.8198%`, Sharpe `5.6537`, Sortino `7.3961`, smart Sortino `7.1536`, Calmar `53.7350`, round trips `33`.
- Current champion reference: locked-OOS return `+1.2181%`; the monthly-budget candidate exceeds it by `+5.6401%` absolute return.

## Evidence and memory
- Portfolio specs evaluated: `73,465`.
- Success candidates: `8`.
- Payload peak RSS: `920.51171875 MiB`.
- Memory guard peak RSS: `911.58984375 MiB` / `955,871,232` bytes; under the explicit `8 GiB` cap.
- Memory soft/hard triggers: `0` / `0`.
- Worker-2 bounded smoke: `uv run python scripts/minimum_viable_run.py --days 30` -> exit `0`, wall `0:01.06`, max RSS `134272 KB`.
- Artifact validator: `uv run python scripts/research/validate_profit_moonshot_pass_under_8gb.py --result-path .omx/specs/autoresearch-profit-moonshot-alpha-v2/result.json` -> `status=passed`, `passed=true`.

## Superseded / historical evidence boundaries
- `.omx/plans/profit_moonshot_alpha_v2_completion_20260509.md` and `docs/session_handoff_20260509_profit_moonshot_alpha_v2_target_budget.md` are historical target-budget evidence. They should not be used as the current pass because the later return-quality contract requires stable `+2%` monthlyized train/validation/OOS return.
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/return_quality_validation_20260509.md` is a stale failed/running snapshot from before the monthly-budget pass. Treat it as historical only.
- Diagnostic or quarantined rows remain non-promoted unless a complete train/validation-selected portfolio passes the same locked-OOS gates.

## Next optimization lane
Continue from the monthly-budget pass by searching for additional stable return drivers, but preserve these gates:

1. Selection uses train/validation evidence only.
2. Locked-OOS remains report-only / gate-only.
3. Train, validation, and locked-OOS monthlyized returns each stay `>= +2.0%`.
4. Locked-OOS MDD remains within the relaxed `<= 25%` budget.
5. Locked-OOS Sharpe `>= 2.0`, Sortino `>= 3.0`, smart Sortino `>= 3.0`, and Calmar `>= 1.0` remain non-negotiable promotion gates.
6. Heavy replay must be explicitly leader-authorized and serialized; otherwise use bounded smoke/artifact validation only.

## Verification recorded for this docs/evidence lane
- `python3 -m compileall -q src scripts tests` -> pass.
- `uv run --extra dev pytest -q tests/test_profit_moonshot_pass_under_8gb_validator.py tests/test_profit_moonshot_fresh_portfolio_tuning.py` -> `14 passed in 0.08s`.
- `uv run --extra dev ruff check scripts/minimum_viable_run.py scripts/research/validate_profit_moonshot_pass_under_8gb.py tests/test_profit_moonshot_pass_under_8gb_validator.py tests/test_profit_moonshot_fresh_portfolio_tuning.py` -> all checks passed.
- `git diff --check` -> pass.

## Team-worker note
Required Task 4 parallel probe was run with three `gpt-5.4-mini` subagents:
- `019e0bc1-109f-7541-8d79-3d1bf99596e0` mapped doc conventions and hazards.
- `019e0bc1-28ea-7031-9597-ce0489dcccdc` summarized current metrics, artifact paths, and stale-evidence risks.
- `019e0bc1-366b-7d20-8eac-1bcff95e494f` returned no usable content; no findings were integrated from that probe.

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
