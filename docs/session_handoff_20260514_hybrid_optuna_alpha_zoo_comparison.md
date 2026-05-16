# Session handoff — Hybrid/Optuna Alpha Zoo comparison (2026-05-14 baseline)

Completed: 2026-05-16 KST; corrected literal-hybrid quarantine: 2026-05-17 KST
Branch/baseline: `private-main` reset to `private/main` `1c6816fced44d277f6c7112934c9dded65ba710f` before work; baseline preserved as parent.

## Objective

Compare the prior Alpha Zoo strict 6x candidate against repo-local hybrid v3.5/v3.6, hybrid Optuna, tuning/optimization, optimizer/study/best_trial, and fresh-portfolio runners-up using the strict policy. Candidate-hybrid/calendar Optuna/calendar fresh rows and literal hybrid/hybrid-online/hybrid-tuning rows were scanned to separate quarantine/reference evidence from the non-calendar comparison core:

- no calendar/month/day/hour entry rules for live promotion;
- current-base/calendar tuple is `hypothesis_reference_only`;
- locked-OOS cannot drive objective/ranking/pruning/sweep expansion/tie-break/selection;
- locked-OOS may be used only after candidate freeze as gate/report;
- return/MDD is diagnostic/report-only, not a hard promotion gate;
- strict deploy lane is separate from diagnostic nonfatal 5x/6x;
- strict live promotion requires zero liquidations, positive margin buffer, OOS MDD <=25%, OOS return above invalid current-base reference, positive risk metrics, and memory <8 GiB;
- a live-promotable hybrid must not nest prior literal hybrid / hybrid-online / hybrid-tuning outputs. `portfolio`, `allocator`, `meta`, `static_blend`, and `leverage_sweep` labels are not exclusion triggers by themselves.

## Headline decision

**Only live-promotion candidate:** `CryptoFxAlphaZooStateStrategy / alpha_zoo_conservative_exit / strict 6x`.

No hybrid v3.5/v3.6, hybrid Optuna, hybrid tuning, or candidate-hybrid row is live-promotable. Calendar/current-base rows and literal-hybrid rows are excluded from strict live promotion and retained in quarantine/reference artifacts. The non-hybrid state-distilled fresh-portfolio row is restored to the non-calendar comparison core but remains diagnostic/non-promotable.

## Alpha Zoo strict 6x split evidence

| Split | Period start | Period end | Return | MDD | Return/MDD diagnostic | Sharpe | Sortino | Smart Sortino | Calmar | Trades | Liquidations | Min buffer |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `2025-01-01T00:00:00` | `2025-10-19T13:00:00` | 68.8842% | 29.5651% | 2.329914 | 1.569139 | 1.919776 | 1.481707 | 2.329914 | 1779 | 0 | 9049.125962 |
| validation | `2025-10-22T05:00:00` | `2026-01-28T06:00:00` | 30.1195% | 9.5595% | 3.150734 | 1.552041 | 2.095744 | 1.912882 | 3.150734 | 524 | 0 | 9527.695928 |
| locked_oos | `2026-01-28T07:00:00` | `2026-05-06T23:00:00` | 41.0967% | 13.6667% | 3.007073 | 2.143209 | 2.841936 | 2.500237 | 3.007073 | 540 | 0 | 9572.449083 |

## Hybrid/Optuna/tuning audit summary

- Non-calendar comparison core rows after literal-hybrid exclusion: `2`; calendar/current-base quarantine rows: `5`; literal nested-hybrid quarantine rows: `8`. Comparison-core metrics are in `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/candidate_split_performance_latest.csv`. Literal-hybrid diagnostics are in `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/excluded_nested_hybrid_same_family_quarantine_latest.csv`; calendar/current-base diagnostics are in `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/excluded_calendar_current_base_quarantine_latest.csv`.
- Hybrid v3.5/v3.6 rows: quarantined because inspected candidates are literal hybrid/hybrid-final-selection outputs. Portfolio/allocator/leverage-sweep/static-blend/meta labels are not banned by themselves.
- Hybrid Optuna `live_guarded` and `train_aware_guarded`: quarantined as same-family hybrid optimizer outputs and **invalid for live promotion** because those objective profiles consume OOS metrics in `src/lumina_quant/portfolio/hybrid_objective.py`.
- Hybrid/tuning `locked_train_val`: cleaner train/validation-only policy shape, but still a same-family hybrid-online tuning output and not an atomic-source hybrid candidate; quarantine/reference only.
- Calendar Optuna: excluded from strict core before ranking; train/validation objective exists, but best-trial/top-trial ordering uses locked-OOS after objective and calendar rules invalidate the family.
- Candidate-hybrid: excluded before ranking because its source sleeves include calendar/current-base dependencies and validation liquidation count `1`, not because generic portfolio/allocator labels are banned.
- Fresh portfolio rows: calendar/current-base rows are quarantine/reference only; the non-calendar state-distilled row is restored to the comparison core and remains diagnostic/non-promotable due missing strict liquidation/margin replay and inferior locked-OOS return versus Alpha Zoo strict 6x.

## Strict lane and diagnostic lane

Strict integer recheck was rerun in `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/alpha_zoo_strict_integer_recheck_latest.json`.

- Highest strict zero-liquidation integer: `6.0x`.
- 6x locked-OOS: return `41.0967%`, MDD `13.6667%`, return/MDD diagnostic `3.007073`, Sharpe `2.143209`, Sortino `2.841936`, smart Sortino `2.500237`, Calmar `3.007073`, liquidations `0`, min buffer `9049.125962`.
- Separate diagnostic 5x/6x lane is retained with `promotion_allowed=false`; it must not be used as live-promotion evidence.

## Memory

Max observed peak RSS: `1239.703125 MiB`; pass under 8 GiB: `true`.

## Artifact paths

- JSON report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_comparison_latest.json`
- Corrected non-calendar JSON snapshot: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_comparison_20260517T000000Z_calendar_quarantine_corrected.json`
- Corrected literal-hybrid quarantine JSON snapshot: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_comparison_20260517T015000KST_hybrid_only_quarantine_corrected.json`
- Markdown report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_comparison_latest.md`
- Comparison-core split CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/candidate_split_performance_latest.csv`
- Literal nested-hybrid quarantine CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/excluded_nested_hybrid_same_family_quarantine_latest.csv`
- Calendar/current-base quarantine CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/excluded_calendar_current_base_quarantine_latest.csv`
- Inventory JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_inventory_latest.json`
- Prompt checklist audit: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/prompt_checklist_audit_latest.json`
- Strict integer recheck JSON/MD/time log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/alpha_zoo_strict_integer_recheck_latest.json`, `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/alpha_zoo_strict_integer_recheck_latest.md`, `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/alpha_zoo_strict_integer_recheck_time.log`

## Research history/source ledger

`docs/profit_moonshot_research_history_20260510.md` and `var/reports/.../research_history/` were **not regenerated**: this session reused existing repo-local artifacts and did not add a new global source family or source-ledger chronology.

## Verification status

Local verification was rerun after the 2026-05-17 literal-hybrid-only quarantine correction. Latest raw log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/local_verification_hybrid_only_quarantine_20260517T020000KST.log`:

```bash
# artifact sanity
# comparison core=2, literal-hybrid quarantine=8, calendar quarantine=5; split timestamps populated
uv run --extra dev pytest tests/test_crypto_fx_alpha_zoo.py tests/test_triple_barrier_labeler.py tests/test_edge_calibration.py tests/test_crypto_fx_alpha_zoo_state_strategy.py -q
# 23 passed in 1.19s
uv run --extra dev pytest tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_pass_under_8gb_validator.py -q
# 74 passed in 0.43s
uv run --extra dev pytest -q
# 1308 passed in 289.85s (0:04:49)
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

Do not promote any Optuna/hybrid row merely because OOS diagnostics look strong. First exclude calendar/current-base-derived rows from the core universe entirely. Then exclude literal hybrid/hybrid-online/hybrid-tuning outputs from strict live promotion; do not exclude generic portfolio/allocator/meta/static_blend/leverage_sweep labels by name alone. Then ask whether locked-OOS entered objective/ranking/pruning/sweep expansion/tie-break/selection; if yes, keep the row diagnostic/reference only.


## 2026-05-17 KST — External Hybrid v3.5/v3.6 method applied to fixed A0+P0+E0+S1+S2+S3+S4 inputs

Operator correction incorporated: `/home/hoky/DeepLearning/ensemble_strategies` defines v3.6 as **v3.5 core plus online dynamic default-model/candidate refresh**, not a new candidate universe and not a hybrid-inside-hybrid stack. Evidence checked directly in the external method source:

- `models/hybrid/v3_5.py`: lines 1-8 describe adaptive weights + Optuna; lines 31-49 define the Optuna-tuned defaults/search candidates; lines 311-328 learn train/warmup parameters; lines 397-420 keep the default model fixed while applying rolling weights/high-vol boost.
- `models/hybrid/v3_6.py`: lines 1-9 state the v3.6 delta: Step A `default_model` is dynamically updated online by rolling MAPE while v3.5 defaults/Optuna results are retained; lines 29-30 reuse v3.5 learning; lines 87-105 learn the same parameters; lines 178-223 dynamically refresh only the default model and otherwise use the v3.5 weight/high-vol/bias structure.
- `scripts/compare_v35_v36.py`: lines 1-5 summarize the same delta; lines 49-55 compare v3.5 fixed default vs v3.6 dynamic default.

Repo adaptation now uses fixed input universe `A0 + P0 + E0 + S1 + S2 + S3 + S4` only. No literal prior hybrid/hybrid-online/hybrid-tuning output is an input; no calendar/month/day/hour entry rule is introduced; Optuna objective/selection uses train+validation only. Locked-OOS remains gate/report-only after candidate freeze.

Split periods for this fixed-input experiment:

- locked_oos: `2026-03-01T00:00:00Z` ~ `2026-05-06T23:00:00Z` (1593 rows)
- train: `2025-01-01T00:00:00Z` ~ `2025-12-31T23:00:00Z` (8760 rows)
- validation: `2026-01-01T00:00:00Z` ~ `2026-02-28T23:00:00Z` (1416 rows)

Candidate input split metrics from the reconstructed return-stream experiment:

| Input | Split | Return | MDD | Return/MDD diagnostic | Sharpe | Sortino | Smart Sortino | Calmar | Trades/active hours | Liquidations | Min buffer |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A0 | train | +114.46% | +28.27% | 4.049305 | 1.979653 | 1.131499 | 0.882143 | 4.049305 | 1812 | not_replayed | not_replayed |
| A0 | validation | +19.97% | +13.67% | 1.461080 | 2.668444 | 1.654321 | 1.455414 | 15.249889 | 292 | not_replayed | not_replayed |
| A0 | locked_oos | +20.51% | +6.79% | 3.021741 | 3.993463 | 2.711715 | 2.539336 | 26.368611 | 313 | not_replayed | not_replayed |
| P0 | train | +64.11% | +17.75% | 3.610917 | 1.854506 | 1.719569 | 1.460317 | 3.610917 | 7173 | not_replayed | not_replayed |
| P0 | validation | +26.68% | +5.37% | 4.966077 | 6.497669 | 7.032072 | 6.673577 | 61.776012 | 1262 | not_replayed | not_replayed |
| P0 | locked_oos | +4.52% | +6.97% | 0.647757 | 1.217684 | 1.227542 | 1.147552 | 3.943449 | 1429 | not_replayed | not_replayed |
| E0 | train | +32.17% | +7.89% | 4.077794 | 2.121676 | 1.710879 | 1.585758 | 4.077794 | 5314 | not_replayed | not_replayed |
| E0 | validation | +11.62% | +2.79% | 4.162712 | 5.240997 | 4.732774 | 4.604245 | 34.893482 | 838 | not_replayed | not_replayed |
| E0 | locked_oos | +2.90% | +2.36% | 1.226822 | 1.780358 | 1.646460 | 1.608436 | 7.201675 | 1175 | not_replayed | not_replayed |
| S1 | train | +8.04% | +2.31% | 3.485607 | 2.064400 | 1.661117 | 1.623648 | 3.485607 | 5314 | not_replayed | not_replayed |
| S1 | validation | +2.91% | +0.77% | 3.796269 | 5.095774 | 4.582247 | 4.547448 | 25.328062 | 838 | not_replayed | not_replayed |
| S1 | locked_oos | +0.73% | +0.60% | 1.200019 | 1.748046 | 1.615190 | 1.605489 | 6.707520 | 1175 | not_replayed | not_replayed |
| S2 | train | +7.07% | +2.86% | 2.470000 | 1.818764 | 1.450761 | 1.410400 | 2.470000 | 5309 | not_replayed | not_replayed |
| S2 | validation | +2.91% | +0.77% | 3.796269 | 5.095774 | 4.582247 | 4.547448 | 25.328062 | 838 | not_replayed | not_replayed |
| S2 | locked_oos | +0.68% | +0.60% | 1.136598 | 1.630621 | 1.534185 | 1.525047 | 6.346750 | 1203 | not_replayed | not_replayed |
| S3 | train | +6.90% | +2.86% | 2.412668 | 1.779218 | 1.417786 | 1.378343 | 2.412668 | 5297 | not_replayed | not_replayed |
| S3 | validation | +2.91% | +0.77% | 3.796269 | 5.095774 | 4.582247 | 4.547448 | 25.328062 | 838 | not_replayed | not_replayed |
| S3 | locked_oos | +0.68% | +0.60% | 1.136598 | 1.630621 | 1.534185 | 1.525047 | 6.346750 | 1203 | not_replayed | not_replayed |
| S4 | train | +3.51% | +3.27% | 1.072977 | 1.014829 | 0.764783 | 0.740531 | 1.072977 | 4885 | not_replayed | not_replayed |
| S4 | validation | +1.52% | +0.90% | 1.684443 | 3.832281 | 3.193221 | 3.164658 | 10.840367 | 748 | not_replayed | not_replayed |
| S4 | locked_oos | +0.62% | +0.81% | 0.758295 | 1.514867 | 1.299905 | 1.289404 | 4.228257 | 1062 | not_replayed | not_replayed |

Hybrid Optuna outputs using the corrected external concept:

| Candidate | Train/validation score | Train return | Validation return | Locked-OOS return | Locked-OOS MDD | Return/MDD diagnostic | Sharpe | Sortino | Smart Sortino | Calmar | Liquidations | Min buffer | Deployable success | Rejection reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| hybrid_v3_5_optuna | 70.585 | +47.73% | +13.31% | +8.52% | +1.77% | 4.827936 | 5.259028 | 7.316663 | 7.189734 | 32.173151 | not_replayed | not_replayed | False | dedicated_integrated_margin_replay_required_for_mixed_alpha_state_portfolio_hybrid |
| hybrid_v3_6_optuna | 85.548 | +49.52% | +12.49% | +7.79% | +1.75% | 4.454705 | 4.859674 | 5.991026 | 5.888040 | 29.199963 | not_replayed | not_replayed | False | dedicated_integrated_margin_replay_required_for_mixed_alpha_state_portfolio_hybrid |

Alpha Zoo strict 6x comparison anchor remains superior for live promotion:

| Candidate | Split | Period start | Period end | Return | MDD | Return/MDD diagnostic | Sharpe | Sortino | Smart Sortino | Calmar | Trades | Liquidations | Min buffer |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Alpha Zoo strict 6x | train | `2025-01-01T00:00:00` | `2025-10-19T13:00:00` | +68.88% | +29.57% | 2.329914 | 1.569139 | 1.919776 | 1.481707 | 2.329914 | 1779 | 0 | 9049.125962 |
| Alpha Zoo strict 6x | validation | `2025-10-22T05:00:00` | `2026-01-28T06:00:00` | +30.12% | +9.56% | 3.150734 | 1.552041 | 2.095744 | 1.912882 | 3.150734 | 524 | 0 | 9527.695928 |
| Alpha Zoo strict 6x | locked_oos | `2026-01-28T07:00:00` | `2026-05-06T23:00:00` | +41.10% | +13.67% | 3.007073 | 2.143209 | 2.841936 | 2.500237 | 3.007073 | 540 | 0 | 9572.449083 |

Decision: the fixed-input v3.5/v3.6 Optuna experiments are useful diagnostics but **not live-promotable** yet because the mixed A0/P0/E0/S-sleeve allocator has no dedicated integrated margin replay, so liquidation count and minimum margin buffer are `not_replayed`. Both fixed-input hybrids satisfy train/validation-only selection and have locked-OOS MDD below 25%, but Alpha Zoo strict 6x still dominates live promotion with locked-OOS return `+41.0967%`, zero liquidations, and positive min buffer. The fixed-input hybrids remain report/reference until a dedicated strict zero-liquidation margin replay is implemented.

Artifacts:

- Script: `scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py`
- Regression test: `tests/test_profit_moonshot_hybrid_v35_v36_fixed_inputs.py`
- JSON report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_v35_v36_fixed_inputs_20260517/hybrid_v35_v36_fixed_inputs_latest.json`
- Markdown report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_v35_v36_fixed_inputs_20260517/hybrid_v35_v36_fixed_inputs_latest.md`
- Timestamped latest run: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_v35_v36_fixed_inputs_20260517/hybrid_v35_v36_fixed_inputs_20260516T172901Z.json`
- Peak RSS: `353.754 MiB` (<8 GiB)

Research history/source ledger was not regenerated: this run reused existing repo-local market/research artifacts and added a session-scoped method-adaptation report; it did not introduce a new global data-source family or chronology ledger.

## 2026-05-17 KST — Fixed-input v3.5/v3.6 local verification

The follow-up fixed-input external-method adaptation passed local verification: Alpha Zoo targeted suite 23 passed, profit-moonshot targeted suite 74 passed, full pytest 1311 passed in 223.57s, ruff check, compileall, git diff --check, and git diff --cached --check passed. Logs: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_v35_v36_fixed_inputs_20260517/logs/`.
