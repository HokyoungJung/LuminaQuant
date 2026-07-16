# Strategy recovery cross-session handoff — 2026-07-16

## Resume identity

- Repository: `/home/hoky/Quants-agent/LuminaQuant`
- Branch: `recovery/strategy-plan-20260714`
- Current implementation checkpoint: `512d2b804ab05bc1cf023ac4cbea81e0506a8736` (`Checkpoint G019 proof contract repairs`)
- Prior G004 snapshot: `f8ba7f1d`
- Foundation commit: `66c85d5da2edbe42c8e9f359ea59582dd814f997`
- Durable Ultragoal session: `019f603a-0e73-7000-88a7-c94f42950c09`
- Durable brief: `.gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/brief.md`
- Durable goals: `.gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/goals.json`
- Durable ledger: `.gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/ledger.jsonl`
- Machine-readable resume state: [`strategy_recovery_resume_state_20260714.json`](strategy_recovery_resume_state_20260714.json)
- Copy-ready prompt: [`strategy_recovery_new_session_prompt_20260714.md`](strategy_recovery_new_session_prompt_20260714.md)
- Frozen master plan: [`strategy_recovery_master_plan_20260713.md`](strategy_recovery_master_plan_20260713.md)
- Evidence root: `/home/hoky/quants-recovery-runs/20260714T105113Z`
- Writable market snapshot: `/home/hoky/quants-recovery-market/20260714T105113Z/market_parquet`

Stable aggregate objective:

> Complete the durable ultragoal plan in `.gjc/ultragoal/goals.json`, including later accepted/appended stories, under the original brief constraints; use `.gjc/ultragoal/ledger.jsonl` as the audit trail.

Always set `GJC_SESSION_ID=019f603a-0e73-7000-88a7-c94f42950c09` on native `gjc ultragoal` commands. Never run `create-goals` or create a competing plan.

## Stop state at handoff

- All implementation work is stopped.
- The only live subagent, final Cost architect task `97-G019CostArchitectFinal`, was paused before producing a verdict. Do not treat it as review evidence; rerun the Cost architect certification in the new session.
- The inline aggregate goal is paused for the user-requested session transfer.
- Durable G019 remains `active`; resume the inline goal only. Do not run `complete-goals` and do not start G005.
- The latest durable ledger event is `blocker_classified: human_blocked: G019`, recording that only the user can open the replacement session.
- No G019/G018/G004 completion checkpoint exists. No completion claim is authorized.

## Binding safety contract

- Preserve user work and treat all original data roots as read-only.
- Do not use synthetic production data, symbol substitution, pre-listing fills, shortened dates, missing-funding proxies, generic strategy fallbacks, locked-OOS parameter changes, paper/testnet/live orders, or capital allocation.
- Do not run strategy performance, grid search, data download/append, network, order, capital, or scientific execution operations while G019/G018/G004 proof certification is unresolved.
- Do not run Alpha phase preparation, prelock, or historical evaluation while blocker-v2 is STOP.
- Never consume `/home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source`.
- Do not reset, rebase, amend, or push without an explicit user instruction.
- Structural, unauthenticated, malformed, or incomplete evidence is `STOP`; complete authenticated evidence failing economic/statistical gates is `REJECT`/scientific KILL.
- R1/R2 are independently gated. Rank only passers by validation 20bp Calmar, then lower validation MDD, then frozen candidate order. Locked OOS is report/gate only and never selects or retunes.
- Preserve `post_oos_research_variant=true`, `post_oos_augment=false`, augmentation/current-fold-OOS/grid/recompute counts zero/false, generic fallback zero, and the exact two frozen candidate IDs.

## Durable goal state

| Goal | State | Meaning |
|---|---|---|
| G001 | complete | D-01 inventory and immutable source/snapshot evidence complete. |
| G002, G008–G011 | superseded | Replaced by verified G012 chain. |
| G012 | complete | D-01A/D-04/D-05 provider, ownership, receipt, lifecycle, and seal gates complete. |
| G003, G013–G016 | superseded | Replaced by verified terminal G017 chain. |
| G017 | complete | Alpha Rev5.15 handoff at `391000b40717386765bfa39bd212d91c2e3be794`; final CLEAR/APPROVE and PASS. |
| G004 | review_blocked | Original R1/R2 replay/cost-proof story; replacement chain not yet certified. |
| G018 | review_blocked | Dispatch/Router/Cost authentication repair story; replaced by active final Cost blocker G019. |
| G019 | active | Final Cost causality, overflow, SPA, and public-boundary certification blocker. |
| G005–G007 | pending | Do not start until G019 is checkpointed and G018/G004 are superseded from fresh clean evidence. |

## Current committed implementation

Checkpoint `512d2b804ab05bc1cf023ac4cbea81e0506a8736` contains nine changed files:

- `src/lumina_quant/cli/research.py`
- `src/lumina_quant/research/cost_proof.py`
- `src/lumina_quant/research/router_replay.py`
- `src/lumina_quant/strategy_factory/research_runner.py`
- `src/lumina_quant/strategy_factory/strategy_signal_dispatch.py`
- `tests/research/test_cost_proof.py`
- `tests/research/test_router_replay.py`
- `tests/test_strategy_signal_dispatch.py`
- `tests/test_strategy_signal_dispatch_routing.py`

The checkpoint is a resumable implementation snapshot, not a completion checkpoint or performance claim.

### Dispatch closure

- Strict mapped handlers no longer mask actual-engine failure with proxy exposures.
- Final return, turnover, and exposure arrays are shape/finiteness checked.
- NumPy `datetime64[ms]` cadence is handled directly.
- Malformed strict parameters surface typed `StrategySignalDispatchError` with cause.
- Current focused evidence: 38 dispatch tests passed; changed dispatch files passed Ruff/format/diff checks.
- Existing independent dispatch architect evidence was CLEAR/CLEAR/CLEAR, APPROVE (`65-G018DispatchReview`).

### Router closure

- Source-first branch/leaf selection, prior-fold chronology, byte-addressed receipts, transitive engine/data/window identity, source/commit roots, exact profile types, artifact closure, and cost-row ownership are fail-closed.
- Fallback scales are recomputed from authenticated shared train/validation returns.
- PPM domains are field-specific: base signal/base return retain frozen bounds while derived position/return support the accepted 3x range.
- MDD starts at pre-period equity, so an initial loss cannot disappear from drawdown.
- Authenticated public regressions cover scaled negative outputs below -1,000,000 and re-rooted initial-loss stale-scale rejection.
- Current evidence: 57 Router tests passed after formatting; Ruff/format/diff checks passed.
- Final Router architect `95-G018RouterArchitectFinal`: CLEAR/CLEAR/CLEAR, APPROVE, no findings.

### Cost proof closure implemented in G019

The Cost proof now includes:

- exact Router receipt/tape ownership and complete committed artifact consumption;
- authenticated source/profile/market/funding/trial roots and exact ordered rows;
- volume/ADV, tick/step, sqrt impact, funding, cash/inventory/fill-price accounting, segment continuity, endpoint exits, stops, and liquidation reconciliation;
- immediate liquidation causality from authenticated event-state marks rather than unrelated bar extremes;
- carried breach state: after a post-event breach, the next action must be liquidation; residual breach permits consecutive liquidation only;
- full breached terminal liquidation as an explicit zero-position endpoint exit, producing economic `REJECT`, while a healthy liquidation is malformed `STOP`;
- guarded native numeric conversion plus `ArithmeticError` handling at `evaluate_cost_proof_file`;
- whole-family deterministic Hansen-style SPA with 2,000 shared circular-block draws, add-one correction, originally degenerate members fixed at p=1, and positive degenerate resamples from originally nondegenerate members counted conservatively as +infinity;
- public file-boundary and real CLI exit-2 overflow regression with fully re-rooted market/source/search/commit/trusted roots;
- both-candidate, all-10/15/20/30bp authenticated liquidation `REJECT` fixture;
- residual 1% partial-liquidation then non-liquidation STOP fixture that was traced to the carried-breach guard;
- duplicate fixture setup removed.

Current Cost evidence:

- Full Cost suite: 61 passed after the terminal-liquidation/overflow/SPA implementation and formatting.
- After that full run, the residual-breach fixture was strengthened to preserve Router sequence ordering and leave 99% residual exposure; its focused test passed and traced to `_strict_fold` carried-breach return line 2264.
- Healthy-liquidation fixture traced to the immediate-causality return line 2323.
- Because those final test-only strengthening edits occurred after the 61-test run, rerun the full Cost suite and formatter in the new session before review.
- Architect certification 92's remaining findings were implemented, but final Cost architect task 97 was paused before verdict. Cost is not yet certified.

## Verification already observed

```text
uv run pytest -q tests/test_strategy_signal_dispatch.py tests/test_strategy_signal_dispatch_routing.py
# 38 passed

uv run pytest -q tests/research/test_router_replay.py
# 57 passed

uv run pytest -q tests/research/test_cost_proof.py
# 61 passed before final residual-fixture strengthening

uv run pytest -q \
  tests/test_research_profile_activation.py \
  tests/test_research_selection_flags_config.py \
  tests/test_alpha_zoo_69_asset_monthly_refit_walkforward.py \
  -k 'not test_shipped_config_yaml_full_load_is_byte_identical_to_head'
# 79 passed, 1 deselected
```

The excluded config test compares the worktree `config.yaml` with HEAD. The local file is user/environment-managed and differs outside this feature. Do not modify it, reveal its values, or treat that unrelated difference as a product regression.

The following were clean before the last residual-fixture strengthening edits:

- Router Ruff/format/diff checks.
- Cost Ruff/format/diff checks.
- Dispatch Ruff/format/diff checks.
- Repository `git diff --check` immediately before checkpoint commit.

No combined final G018 regression, AI-slop cleaner, full strict three-lane architect review, executor QA/red-team receipt, or G019 quality-gate checkpoint has been completed.

## Exact remaining plan

### Phase 1 — resume and certify G019

1. Verify the branch contains `512d2b804ab05bc1cf023ac4cbea81e0506a8736` and the handoff commit containing this file. Do not rewrite history.
2. Read this handoff, the resume-state JSON, frozen master plan, data-PC runbook, reality audit, durable goals, and durable ledger.
3. Activate `/skill:ultragoal`, set the session ID, inspect `gjc ultragoal status --json`, and resume the paused inline aggregate goal. Durable G019 is already active; do not run `complete-goals`.
4. Confirm the worktree is clean before mutation.
5. Run Ruff format/check and `git diff --check` for the nine checkpoint files. A formatter-only follow-up may be needed for the final residual fixture.
6. Rerun the complete Cost suite (expected 61 tests), Router suite (57), dispatch suite (38), and the 79-test profile/selection regression command above.
7. Rerun the final Cost architect certification from scratch. It must explicitly certify certification-92 items 7, 10, 12, and 13: immediate liquidation causality/sequencing, valid economic REJECT, overflow STOP at file/CLI boundary, conservative degenerate SPA, and non-shallow tests.
8. Any finding must be recorded with `gjc ultragoal record-review-blockers`, repaired by a bounded executor, and the full blocking loop rerun.

### Phase 2 — final G018/G019 completion gate

1. Run the combined focused regression over all dispatch, Router, Cost, profile, selection, and monthly-refit files.
2. Run the internal Ultragoal AI-slop cleaner over exactly the nine changed files. Fix every blocking finding with an executor and rerun until zero blockers.
3. Freeze the post-cleaner change set and rerun verification.
4. Run a fresh architect review across architecture/product/code for the complete dispatch/Router/Cost contract.
5. Run an executor adversarial QA/red-team lane against the frozen set. Parent-owned commands must produce real API/package test-report and CLI-surface artifacts; bare inline evidence is insufficient.
6. Require CLEAR/CLEAR/CLEAR, APPROVE, executor QA passed, full rerun true, empty blockers, and valid artifact references.
7. Checkpoint G019 with strict `--quality-gate-json`. Do not call inline `goal complete`; later durable goals remain.
8. After the G019 receipt exists, mark G018 superseded by completed G019 evidence, then mark G004 superseded by the completed G018/G019 replacement chain. Preserve all historical blocked goals and ledger entries.

### Phase 3 — continue the frozen durable plan

Only after Phase 2 is durably clean:

1. Continue G005: bounded data repair and one-touch R-04/A-03 scientific decisions under source-read-only and no-retuning rules. Existing data/Alpha blockers remain authoritative; classify genuine human-only blockers before pausing.
2. Continue G006 only after R-04 and A-03 are terminal: preregister and execute the bounded follow-up alpha/volatility cycle with complete trial ledgers and locked-OOS report-only semantics.
3. Continue G007 only for champions that pass prior gates: frozen fresh-forward evidence with zero orders and zero capital, 30-day checkpoint, and 60-day terminal PASS/KILL.
4. The aggregate inline goal may be completed only after every required durable goal has a fresh receipt and a fresh final aggregate receipt.

## Durable file integrity at pause

Recorded at `2026-07-16T17:19:11Z`:

- `brief.md`: SHA-256 `faf6f83679e7ce93a8950af4df350fc4a92557d8eaaa40ea17c9c8b918c04e57`, 4,836 bytes.
- `goals.json`: SHA-256 `7b593003e0c6937ec8010b7852022a3e082a9f6c4c23c4111333c17e904b2cd4`, 39,289 bytes.
- `ledger.jsonl`: SHA-256 `6f271bcfbce1580c9b16a947c826417bf45bc743b12e29ea8e07fa880ce9a7aa`, 66,512 bytes, 96 lines.
- Latest ledger event ID: `41bc8670-8d9a-4f2d-b044-3e4d0cf29d93` (`blocker_classified`, `human_blocked`, G019).

These hashes describe the runtime state at pause. Normal resume/steering/checkpoint operations will change them and must append ledger evidence rather than editing runtime files by hand.
