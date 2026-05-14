# RALPLAN-DR — Profit Moonshot Alpha Zoo Strict Real-Data Policy Restoration (2026-05-14)

## Scope
Restore fail-closed Alpha Zoo promotion policy after the 2026-05-13 relaxation that made return/MDD diagnostic-only. Keep the current-base/calendar tuple as `hypothesis_reference_only`, not a selector or live target. Complete regenerated artifacts only if the train/validation-frozen real-data Alpha Zoo candidate passes the strict policy; otherwise publish a reproducible no-promotion result.

## Principles
1. **No invalid live promotion:** calendar/month/day/hour/current-base tuples remain hypothesis references only.
2. **Train/validation ownership:** selection and edge calibration use only train+validation; locked-OOS is opened after candidate freeze for gate/report only.
3. **Fail-closed deployment:** deployable success requires every strict gate, including OOS return and return/MDD beating the invalid current-base reference.
4. **Lane separation:** strict zero-liquidation lane may promote; nonfatal 5x/6x diagnostic lane can never promote.
5. **Reproducible real-data evidence:** reports must show real data provenance, source coverage, calibration counts, memory <8 GiB, and exact artifact paths.

## Decision Drivers
1. User explicitly re-requires OOS return **and** return/MDD to beat the invalid current-base reference for deployable success.
2. Current code/report evidence shows policy drift: `scripts/research/replay_crypto_fx_alpha_zoo_state.py` sets `return_mdd_hurdle_required=False` and reports `return_mdd_role=diagnostic_report_only`; `tests/test_crypto_fx_alpha_zoo_state_strategy.py` currently locks that relaxed behavior.
3. Existing Alpha Zoo 2026-05-13 strict 6x result beats OOS return but not return/MDD (`3.007073` vs `6.9168776779021455`), so restored strict policy should fail closed unless a new train/validation-selected candidate actually beats both.

## Options
### Option A — Minimal strict-policy restoration + artifact refresh (recommended)
- Change replay policy/tests/docs so return/MDD is a hard strict gate again.
- Re-run existing Alpha Zoo real-data pipeline with current data/artifacts.
- Expected result likely `deployable_success=false` unless new frozen candidate beats both reference metrics.
- Pros: smallest, safest correction; directly addresses user constraint and regression.
- Cons: may not discover new alpha beyond existing candidate set.

### Option B — Broaden candidate zoo before policy restoration
- Expand/search more Alpha Zoo formulas before rerunning strict replay.
- Pros: higher chance of finding a candidate that beats both strict economics gates.
- Cons: increased leakage/overfit risk, broader runtime, more moving parts before fixing a known policy bug.

### Option C — Keep relaxed return/MDD policy and annotate only
- Mark return/MDD diagnostic-only but document the user constraint as unmet.
- Rejected: violates explicit user requirement; would falsely preserve deployable status.

## ADR
- **Decision:** Execute Option A first. Restore return/MDD as a required strict deployable gate, regenerate/finalize artifacts, and report no live promotion unless a train/validation-selected candidate beats current-base reference OOS return and return/MDD.
- **Drivers:** explicit user constraint; current relaxed policy is invalid; minimal correction reduces leakage and scope risk.
- **Alternatives considered:** broaden search first (defer until strict policy is fixed); annotation-only (rejected as noncompliant).
- **Why chosen:** fixes the known deployability bug before any further optimization and preserves Alpha Zoo real-data provenance.
- **Consequences:** current 2026-05-13 strict 6x Alpha Zoo result will become non-deployable if unchanged because return/MDD is below reference; docs/handoff must correct any prior deployable-success claim.
- **Follow-ups:** if fail-closed, a later research lane may broaden train/validation-only candidates, but locked-OOS remains gate/report-only.

## Execution Plan
1. **Lock the regression first**
   - Update/add tests that fail when `deployable_success=true` while `oos_return_mdd_beats_current_base=false`.
   - Reverse the existing relaxed-policy regression in `tests/test_crypto_fx_alpha_zoo_state_strategy.py`.
   - Assert `oos_return_mdd_beats_current_base` appears in strict `performance_gates`, not only diagnostics.
2. **Restore strict replay policy**
   - In `scripts/research/replay_crypto_fx_alpha_zoo_state.py`, set policy metadata to `return_mdd_hurdle_required=True` / hard-gate role.
   - Add `oos_return_mdd_beats_current_base` to strict `performance_gates` and `deployable` calculation.
   - Ensure payload reason, markdown, and `locked_oos_report_only_metrics` no longer describe return/MDD as diagnostic-only for promotion.
3. **Preserve provenance and lane boundaries**
   - Keep `CURRENT_BASE_REFERENCE.role=hypothesis_reference_only`, `selection_target=false`, `promotion_target=false`.
   - Keep candidate selection and calibration counts train/validation-only with `locked_oos_calibration_record_count=0`.
   - Keep diagnostic 5x/6x lane `promotion_allowed=false` regardless of liquidation status or return.
4. **Regenerate real-data artifacts**
   - Re-run screen, ledger/calibration, replay, and summary under an explicitly pinned 2026-05-14 successor artifact directory.
   - Use `--strict-real-data` for both screen and replay so missing OHLCV/source coverage fails closed.
   - Publish strict lane, diagnostic lane, memory summary, source coverage, candidate freeze/provenance, and fail-closed/deployable decision.
   - Do not rely on current script defaults that still point to the 2026-05-13 artifact directory; pass explicit `--output-dir` / `--output`.
   - Add/use an explicit aggregate summary generation path that derives `deployable_success` from replay strict gates, including `oos_return_mdd_beats_current_base`; do not hand-enter promotion status.
5. **Correct documentation/handoff**
   - Update research note/session handoff/active plan/test spec to state the 2026-05-13 relaxed policy is superseded.
   - Edit the existing 2026-05-13 Alpha Zoo handoff with a top-of-file supersession notice and remove/neutralize forward-looking stale instructions such as “Do not use return/MDD as a hard promotion hurdle.”
   - Keep historical metric records readable, but ensure the latest/current handoff points to the 2026-05-14 strict result.
   - If no candidate beats both OOS return and return/MDD, explicitly record `deployable_success=false` and no live promotion.

## Acceptance Criteria
- Strict deployable candidate requires all of:
  - no calendar/month/day/hour entry rules;
  - selected/calibrated from train+validation only;
  - locked-OOS used only after candidate freeze;
  - strict lane liquidation count `0` and minimum margin buffer `>0`;
  - OOS MDD `<=25%`;
  - OOS return `> 0.06428110030664325`;
  - OOS return/MDD `> 6.9168776779021455`;
  - Sharpe, Sortino, smart Sortino, and Calmar strictly positive (`>0.0`) for the current policy pass;
  - peak RSS `<8192 MiB`.
- Any failure above produces `deployable_success=false` with explicit rejection reasons.
- Current-base/calendar tuple appears only as `hypothesis_reference_only`, never as `selection_target` or `promotion_target`.
- Diagnostic 5x/6x lane reports liquidation count, event drawdown, equity loss, recovery, and `promotion_allowed=false`.
- Artifacts include factor screen, candidate outcome ledger, edge calibration, stateful replay, summary JSON/MD, research note, and handoff.

## Verification Path
- Targeted tests:
  - `uv run --extra dev pytest tests/test_crypto_fx_alpha_zoo_state_strategy.py -q`
  - `uv run --extra dev pytest tests/test_crypto_fx_alpha_zoo.py tests/test_triple_barrier_labeler.py tests/test_edge_calibration.py -q`
- Integration tests:
  - `uv run --extra dev pytest tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_pass_under_8gb_validator.py -q`
- Static/full checks:
  - `uv run --extra dev ruff check .`
  - `uv run --extra dev python -m compileall -q src scripts tests`
  - `uv run --extra dev pytest -q`
  - `git diff --check`
- Artifact checks:
  - confirm `locked_oos_calibration_record_count=0`;
  - confirm selected candidate grid exposes only train/validation metrics during selection;
  - confirm strict lane deployable status equals conjunction of all gates including return/MDD;
  - confirm memory summary passes under 8 GiB;
  - confirm docs no longer claim relaxed-policy deployability.
  - negative grep/check regenerated Alpha Zoo docs/artifacts for stale relaxed-policy strings such as `diagnostic_report_only`, `return/MDD is no longer a hard promotion hurdle`, `deployable_success_true_under_revised_policy`, and `live_promotion_candidate_under_revised_gate`.

## Required Artifact Regeneration Commands

Use explicit output paths:

```bash
OUT=var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260514
CACHE=var/cache/profit_moonshot_fresh_start/joined_panel_de62df511cec53df6ad39521.parquet
EXT=var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/external_market_state_20260512/external_market_state_lagged.csv

/usr/bin/time -v uv run --extra dev python scripts/research/run_crypto_fx_alpha_zoo_screen.py \
  --current-tail-cache "$CACHE" \
  --external-state-csv "$EXT" \
  --strict-real-data \
  --output-dir "$OUT" \
  --ledger-output "$OUT/candidate_outcome_ledger_latest.jsonl"

uv run --extra dev python scripts/research/calibrate_crypto_fx_edges.py \
  --ledger "$OUT/candidate_outcome_ledger_latest.jsonl" \
  --output "$OUT/edge_calibration_latest.json"

/usr/bin/time -v uv run --extra dev python scripts/research/replay_crypto_fx_alpha_zoo_state.py \
  --current-tail-cache "$CACHE" \
  --external-state-csv "$EXT" \
  --strict-real-data \
  --calibration "$OUT/edge_calibration_latest.json" \
  --output "$OUT/crypto_fx_alpha_zoo_state_replay_latest.json" \
  --max-leverage 6

uv run --extra dev python scripts/research/write_crypto_fx_alpha_zoo_real_data_summary.py \
  --screen "$OUT/crypto_fx_alpha_zoo_screen_latest.json" \
  --calibration "$OUT/edge_calibration_latest.json" \
  --replay "$OUT/crypto_fx_alpha_zoo_state_replay_latest.json" \
  --output-json "$OUT/crypto_fx_alpha_zoo_real_data_summary_latest.json" \
  --output-md "$OUT/crypto_fx_alpha_zoo_real_data_summary_latest.md"
```

Summary JSON/MD must normalize these strict policy fields across replay and aggregate reports:

- `deployable_success`
- `deployable_success_reason`
- `return_mdd_hurdle_required=true`
- `oos_return_mdd_beats_current_base`
- `current_base_calendar_tuple_role=hypothesis_reference_only`
- `selection_excludes_current_base_calendar_tuple=true`
- no deployable/live-promotion wording when the return/MDD gate fails.

If `scripts/research/write_crypto_fx_alpha_zoo_real_data_summary.py` does not exist at execution start, create it as a small deterministic repo-local summary writer that:

1. reads the screen/calibration/replay JSON artifacts;
2. derives `strict_zero_liquidation_lane`, `diagnostic_nonfatal_5x_6x_lane`, `memory_summary`, `factor_screen`, `edge_calibration`, and `candidate_outcome_ledger` from those artifacts;
3. sets aggregate `deployable_success` from `bool(replay.strict_zero_liquidation_lane.promoted_candidate)` after the replay script has applied all strict gates;
4. copies the promoted-candidate gates and rejection reasons into summary JSON/MD;
5. fails closed if replay policy does not contain `return_mdd_hurdle_required=true` or if `oos_return_mdd_beats_current_base` is missing from strict gates.

## Files/Paths Expected to Be Touched
- `scripts/research/replay_crypto_fx_alpha_zoo_state.py`
- `tests/test_crypto_fx_alpha_zoo_state_strategy.py`
- likely existing tests as needed: `tests/test_profit_moonshot_live_final_selection.py`, `tests/test_profit_moonshot_pass_under_8gb_validator.py`
- `.omx/plans/profit_moonshot_alpha_zoo_real_data_next_plan_20260512.md` or 2026-05-14 successor
- `.omx/plans/test-spec-profit-moonshot-alpha-zoo-real-data-20260513.md` or 2026-05-14 successor
- `.omx/notepad.md`
- `docs/research_note_profit_moonshot_alpha_zoo_real_data_20260512.md` or 2026-05-14 successor
- `docs/session_handoff_*crypto_fx_alpha_zoo_real_data*.md` / `docs/session_handoff_*alpha_zoo*.md`
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260514/`

## Available Agent Types Roster
- `planner`: plan/ADR/test-shape owner.
- `architect`: review policy boundary, leakage risk, lane separation.
- `critic`: reject inconsistent deployable gates or weak acceptance criteria.
- `executor`: implement tests, policy fix, report generation updates.
- `test-engineer`: strengthen targeted/integration regression coverage.
- `verifier`: inspect artifacts and prove gates/docs match policy.
- `code-reviewer`: final diff review for leakage, false promotion, and docs consistency.
- `researcher`: optional only if external data/source-doc questions emerge; not required for this repo-local correction.

## Staffing / Handoff
### Ralph path
Use `$ralph` for a single-owner fix/verify loop when speed and consistency matter:
- reasoning: high for policy/test/doc consistency;
- mandate: tests first, strict policy restoration, artifact regeneration, verification evidence, no live promotion unless both reference gates pass.

### Team path
Use `$team` if parallelizing after this plan is approved:
- Lane 1 (`executor`, medium/high): script policy + markdown payload updates.
- Lane 2 (`test-engineer`, medium): regression tests for strict return/MDD, selection provenance, diagnostic lane.
- Lane 3 (`executor`, medium): artifact regeneration and docs/handoff updates after Lane 1/2 land.
- Lane 4 (`verifier` or `code-reviewer`, high): final artifact/diff audit; verify no relaxed-policy claims remain.
- Team verification must run targeted tests before artifact regeneration, then full/static checks and artifact consistency checks.

## Goal-Mode Follow-up Suggestions
- `$ultragoal`: best default if the user wants a durable tracked completion ledger for policy fix + artifacts.
- `$performance-goal`: not primary; use only if later work focuses on optimizing the Alpha Zoo candidate search/runtime.
- `$autoresearch-goal`: use only for a separate broad research search after this strict-policy correction is complete.

## Stop Rule
Stop when either:
1. a train/validation-selected strict candidate beats both current-base OOS return and return/MDD with zero liquidations, positive buffers, OOS MDD <=25%, positive risk metrics, and artifacts prove it; or
2. all artifacts are regenerated and explicitly fail closed with `deployable_success=false` and no live promotion.
