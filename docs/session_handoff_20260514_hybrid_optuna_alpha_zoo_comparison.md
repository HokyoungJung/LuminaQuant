# Session handoff — Hybrid/Optuna Alpha Zoo comparison (2026-05-14 baseline)

Completed: 2026-05-16 KST
Branch/baseline: `private-main` reset to `private/main` `1c6816fced44d277f6c7112934c9dded65ba710f` before work; baseline preserved as parent.

## Objective

Compare the prior Alpha Zoo strict 6x candidate against repo-local hybrid v3.5/v3.6, hybrid Optuna, tuning/optimization, optimizer/study/best_trial, candidate-hybrid, calendar Optuna, and fresh-portfolio optimization outputs using the strict policy:

- no calendar/month/day/hour entry rules for live promotion;
- current-base/calendar tuple is `hypothesis_reference_only`;
- locked-OOS cannot drive objective/ranking/pruning/sweep expansion/tie-break/selection;
- locked-OOS may be used only after candidate freeze as gate/report;
- return/MDD is diagnostic/report-only, not a hard promotion gate;
- strict deploy lane is separate from diagnostic nonfatal 5x/6x;
- strict live promotion requires zero liquidations, positive margin buffer, OOS MDD <=25%, OOS return above invalid current-base reference, positive risk metrics, and memory <8 GiB.

## Headline decision

**Only live-promotion candidate:** `CryptoFxAlphaZooStateStrategy / alpha_zoo_conservative_exit / strict 6x`.

No hybrid v3.5/v3.6, hybrid Optuna, calendar Optuna, candidate-hybrid, or fresh-portfolio tuning row is live-promotable under the policy. Strong-looking OOS rows that used OOS objective/best-trial ordering or calendar/current-base sleeves are retained only as diagnostic/reference.

## Alpha Zoo strict 6x split evidence

| Split | Period start | Period end | Return | MDD | Return/MDD diagnostic | Sharpe | Sortino | Smart Sortino | Calmar | Trades | Liquidations | Min buffer |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `2025-01-01T00:00:00` | `2025-10-19T13:00:00` | 68.8842% | 29.5651% | 2.329914 | 1.569139 | 1.919776 | 1.481707 | 2.329914 | 1779 | 0 | 9049.125962 |
| validation | `2025-10-22T05:00:00` | `2026-01-28T06:00:00` | 30.1195% | 9.5595% | 3.150734 | 1.552041 | 2.095744 | 1.912882 | 3.150734 | 524 | 0 | 9527.695928 |
| locked_oos | `2026-01-28T07:00:00` | `2026-05-06T23:00:00` | 41.0967% | 13.6667% | 3.007073 | 2.143209 | 2.841936 | 2.500237 | 3.007073 | 540 | 0 | 9572.449083 |

## Hybrid/Optuna/tuning audit summary

- Candidate inventory rows: `15`. Full per-candidate train/validation/locked-OOS metrics and actual split periods are in `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_comparison_latest.md` and `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/candidate_split_performance_latest.csv`.
- Hybrid v3.5/v3.6 clean rows: train/validation selection, OOS report-only; rejected for locked-OOS underperformance versus Alpha Zoo/current-base reference and missing strict liquidation/margin replay.
- Hybrid Optuna `live_guarded` and `train_aware_guarded`: **invalid for live promotion** because those objective profiles consume OOS metrics in `src/lumina_quant/portfolio/hybrid_objective.py` and the inspected artifacts used them.
- Hybrid/tuning `locked_train_val`: clean policy shape but weak locked-OOS performance and missing strict replay; not live-promotable.
- Calendar Optuna: train/validation objective exists, but best-trial/top-trial ordering uses locked-OOS after objective; calendar rules also invalidate live promotion.
- Candidate-hybrid: train/validation selected, but validation liquidation count `1` and calendar/current-base source dependency block strict promotion despite high OOS diagnostics.
- Fresh portfolio calendar/current-base rows: diagnostic/reference only; non-calendar state-distilled row still underperforms Alpha Zoo/current-base and lacks strict replay.

## Strict lane and diagnostic lane

Strict integer recheck was rerun in `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/alpha_zoo_strict_integer_recheck_latest.json`.

- Highest strict zero-liquidation integer: `6.0x`.
- 6x locked-OOS: return `41.0967%`, MDD `13.6667%`, return/MDD diagnostic `3.007073`, Sharpe `2.143209`, Sortino `2.841936`, smart Sortino `2.500237`, Calmar `3.007073`, liquidations `0`, min buffer `9049.125962`.
- Separate diagnostic 5x/6x lane is retained with `promotion_allowed=false`; it must not be used as live-promotion evidence.

## Memory

Max observed peak RSS: `1239.703125 MiB`; pass under 8 GiB: `true`.

## Artifact paths

- JSON report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_comparison_latest.json`
- Markdown report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_comparison_latest.md`
- Split CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/candidate_split_performance_latest.csv`
- Inventory JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_inventory_latest.json`
- Prompt checklist audit: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/prompt_checklist_audit_latest.json`
- Strict integer recheck JSON/MD/time log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/alpha_zoo_strict_integer_recheck_latest.json`, `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/alpha_zoo_strict_integer_recheck_latest.md`, `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/alpha_zoo_strict_integer_recheck_time.log`

## Research history/source ledger

`docs/profit_moonshot_research_history_20260510.md` and `var/reports/.../research_history/` were **not regenerated**: this session reused existing repo-local artifacts and did not add a new global source family or source-ledger chronology.

## Verification status

Local verification passed on 2026-05-16 KST. Raw log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/local_verification_20260516T134741Z.log`:

```bash
uv run --extra dev pytest tests/test_crypto_fx_alpha_zoo.py tests/test_triple_barrier_labeler.py tests/test_edge_calibration.py tests/test_crypto_fx_alpha_zoo_state_strategy.py -q
# 23 passed in 2.14s
uv run --extra dev pytest tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_pass_under_8gb_validator.py -q
# 74 passed in 0.97s
uv run --extra dev pytest -q
# 1307 passed in 431.06s (0:07:11)
uv run --extra dev ruff check .
# All checks passed!
uv run --extra dev python -m compileall -q src scripts tests
# passed
git diff --check
# passed
git diff --cached --check
# passed
```

## Next operator caution

Do not promote any Optuna/hybrid row merely because OOS diagnostics look strong. First ask whether locked-OOS entered objective/ranking/pruning/sweep expansion/tie-break/selection. If yes, keep the row diagnostic/reference only.
