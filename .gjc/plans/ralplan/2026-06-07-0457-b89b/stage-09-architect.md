## Summary
G003 survivor manifest/freeze workflow is contract-compliant for the reviewed surfaces. The script freezes candidates from train/validation evidence, keeps locked-OOS as report/gate context outside the survivor manifest hash, writes a dedicated manifest artifact, and leaves real-money disabled. No blocking defect was found; suggested adversarial additions are non-blocking hardening.

## Analysis
- Contract context requires train/validation-only survivor freeze, locked-OOS report/gate attachment only after freeze, frozen manifests as the input to monthly/full WF retest, and real-money exclusion (`.gjc/plans/ralplan/2026-06-07-0457-b89b/pending-approval.md:16-25`, `:84-99`; `.gjc/specs/deep-interview-alpha-strategy-improvement.md:63-91`).
- Selection score and eligibility surfaces are train/validation-only: `_score_row` reads train/validation return, MDD, turnover proxy, and trade counts without locked-OOS fields (`scripts/research/run_alpha_zoo_clean_new_alpha_discovery.py:469-490`); `_eligible_for_policy` rejects any locked-OOS usage flag before delegating to train/validation freeze eligibility (`:555-562`); `_eligible_for_freeze` checks train/validation coverage, positive returns, validation MDD, trade counts, and validation cost proxy (`:592-608`).
- The main discovery report still attaches report-only locked-OOS gates after freeze, as allowed by the contract: `_promotion_gate_report` computes report-only locked-OOS risk/benchmark fields and keeps `ready_for_real=false` and `real_money_execution=false` (`scripts/research/run_alpha_zoo_clean_new_alpha_discovery.py:703-775`).
- The survivor manifest strips locked-OOS keys recursively and builds forward eligibility from freeze selection, train/validation survivor status, train/validation feature sufficiency, holdout contamination flags, and validation cost proxy (`scripts/research/run_alpha_zoo_clean_new_alpha_discovery.py:846-932`). Contamination adds `holdout_used_by_train_validation_path` and prevents `eligible_for_full_wf_retest` (`:876-927`).
- The manifest payload includes only train/validation selection inputs, holdout-use prohibition, fresh-forward requirement, survivor/retest lists, and real-money false; the hash excludes volatile generation time and, because survivor rows are scrubbed before hashing, ignores report-only locked-OOS metric changes (`scripts/research/run_alpha_zoo_clean_new_alpha_discovery.py:935-977`). The run path writes JSON, Markdown, and `clean_new_alpha_survivor_manifest_latest.json` (`:4858-4869`).
- Tests cover the critical adversarial cases: score/selection ignore locked-OOS report fields (`tests/test_alpha_zoo_clean_new_alpha_discovery.py:118-143`), all locked-OOS usage flags fail eligibility (`:1108-1117`), report gates fail closed for missing/high-risk OOS metrics in the main report (`:1120-1158`), manifests exclude locked-OOS keys (`:1256-1279`), manifest hash ignores report-only OOS metric changes (`:1282-1305`), holdout contamination blocks retest (`:1308-1326`), and the manifest artifact is written with real-money disabled (`:1329-1378`).
- Observed caller-run verification evidence is accepted as provided: `uv run pytest tests/test_alpha_zoo_clean_new_alpha_discovery.py` passed 37 tests and `uv run pytest tests/test_alpha_zoo_existing_strategy_reassessment.py` passed 3 tests. I did not run tests, linters, or formatters.

## Root Cause
No active defect. The central risk for this change is hidden locked-OOS leakage from report-only fields into a freeze or retest manifest; the implementation addresses it by using train/validation scoring and eligibility, explicit locked-OOS usage flags, recursive manifest key scrubbing, and a manifest hash over the scrubbed freeze payload.

## Findings
No blocking findings.

Non-blocking adversarial additions worth considering:
- LOW: add a hash perturbation test for `locked_oos_liquidation_count_report_only`, `locked_oos_account_wipeout_count_report_only`, and `feature_coverage.locked_oos`, not just return/MDD. Impact is limited because current manifest construction ignores these fields, but the test would lock the broader report-only promise.
- LOW: add a persisted-artifact recomputation test documenting that `output_path` is intentionally outside `survivor_manifest_sha256`. Impact is traceability only; current behavior is reasonable because output path is an I/O location, not freeze content.
- LOW: add malformed/non-finite locked-OOS risk metric cases for the main discovery report gate. Impact is outside the survivor-manifest hash/selection path, but it would strengthen fail-closed diagnostics for report gates.

## Recommendations
1. Accept G003 for ultragoal checkpointing with no blockers.
2. Keep the survivor manifest as the only handoff surface for full-WF retest candidates; keep locked-OOS report gates in the main discovery report only.
3. Add the non-blocking adversarial tests above in a future hardening pass, especially if external or historical rows start feeding `_survivor_manifest_payload` directly.

## Architectural Status
CLEAR

## Code Review Recommendation
APPROVE

## Trade-offs
| Option | Pros | Cons | Verdict |
| --- | --- | --- | --- |
| Current scrubbed survivor manifest + report-only OOS gates | Preserves clean freeze, makes retest handoff deterministic, keeps OOS diagnostics visible without hash/selection contamination | Main report has separate `can_advance_to_full_wf` wording that must not be confused with manifest retest handoff | Recommended |
| Remove locked-OOS gates from main discovery report | Reduces naming confusion | Loses useful post-freeze risk diagnostics and conflicts with allowed report-gate context | Not recommended |
| Include OOS risk gates in manifest retest eligibility/hash | Stronger post-freeze risk filter | Violates the clean freeze/hash requirement by letting report-only OOS alter manifest state | Reject |
