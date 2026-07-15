# Strategy recovery cross-session handoff — 2026-07-15

## Resume identity

- Repository: `/home/hoky/Quants-agent/LuminaQuant`
- Branch: `recovery/strategy-plan-20260714`
- G004 code snapshot: `f8ba7f1d` (`Checkpoint G004 recovery proof infrastructure`)
- Foundation commit: `66c85d5da2edbe42c8e9f359ea59582dd814f997`
- Durable Ultragoal session: `019f603a-0e73-7000-88a7-c94f42950c09`
- Durable goals: `.gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/goals.json`
- Durable ledger: `.gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/ledger.jsonl`
- Machine-readable resume state: [`strategy_recovery_resume_state_20260714.json`](strategy_recovery_resume_state_20260714.json)
- Copy-ready prompt: [`strategy_recovery_new_session_prompt_20260714.md`](strategy_recovery_new_session_prompt_20260714.md)
- Evidence root: `/home/hoky/quants-recovery-runs/20260714T105113Z`
- Writable market snapshot: `/home/hoky/quants-recovery-market/20260714T105113Z/market_parquet`

Stable aggregate objective:

> Complete the durable ultragoal plan in `.gjc/ultragoal/goals.json`, including later accepted/appended stories, under the original brief constraints; use `.gjc/ultragoal/ledger.jsonl` as the audit trail.

Always set `GJC_SESSION_ID=019f603a-0e73-7000-88a7-c94f42950c09` on native `gjc ultragoal` commands. Do not run `create-goals`.

## Binding safety contract

- Preserve user work. Treat every original data root as read-only.
- Never use synthetic production data, symbol substitution, pre-listing fills, shortened dates, missing-funding proxies, fallback strategy proxies, locked-OOS parameter changes, paper/testnet/live orders, or capital allocation.
- Do not run strategy performance, grid-search, data download, network, or order operations while G004/G018 is unresolved.
- Do not run Alpha phase preparation, prelock, or historical evaluation while blocker-v2 is STOP.
- Never consume `/home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source`.
- Do not push either recovery branch without explicit user instruction.
- Structural/incomplete evidence is `STOP`; complete evidence failing scientific gates is `REJECT`/scientific KILL.

## Durable goal state

| Goal | State | Meaning |
|---|---|---|
| G001 | complete | D-01 inventory and immutable source/snapshot evidence complete. |
| G002, G008–G011 | superseded | Replaced by verified G012 chain. |
| G012 | complete | D-01A/D-04/D-05 provider, ownership, receipt, lifecycle, and seal gates complete. |
| G003, G013–G016 | superseded | Replaced by verified terminal G017 chain. |
| G017 | complete | Alpha Rev5.15 handoff at commit `391000b40717386765bfa39bd212d91c2e3be794`; final CLEAR/APPROVE and PASS. |
| G004 | review_blocked | Partial router/dispatch/cost infrastructure is committed at `f8ba7f1d`; it is not accepted proof. |
| G018 | pending review-blocker story | Resume here to close all G004 architecture blockers. |
| G005–G007 | pending | Do not start until G018 resolves and G004 receives clean final gates/checkpoint. |

The inline aggregate goal is paused only for the user-requested session transfer. Resume it in the new session.

## Completed and preserved evidence

### Foundation (G012)

- Commit: `66c85d5da2edbe42c8e9f359ea59582dd814f997`
- Focused gate: 125 passed.
- Full tracked suite: 4475 passed, 20 skipped, 3 xfailed.
- Architect: CLEAR/CLEAR/CLEAR, APPROVE.
- Executor QA: passed.
- Real BTC June remains expected STOP: 874 OHLCV gaps and 18 funding prefix gaps; no append occurred.

### Alpha Rev5.15 terminal chain (G017)

- Alpha branch: `recovery/alpha-max-rev515-alignment-20260714`
- Commit: `391000b40717386765bfa39bd212d91c2e3be794`
- Receipt v5: `/home/hoky/quants-recovery-runs/20260714T105113Z/alpha-max-rev515-alignment-receipt-v5.json`
- Receipt SHA-256: `8687b52180502a11de9fbe317a19d00bb4492c464b3bf33d4eda2437683ca812`
- Manifest SHA-256: `dbe018b066f556152e387f15c84770c7c2bd4e46dbc6ac8bac109e3176e5a036`
- Lock SHA-256: `59d9de230be950761736c24e04af3456e229cf4aa077536167fb7e650a71c339`
- Full/post-cleaner gate: 4927 passed, 36 skipped, 3 xfailed; Ruff passed.
- Architect CLEAR/APPROVE; executor PASS.
- Blocker-v2 remains authoritative STOP: `/home/hoky/quants-recovery-runs/20260714T105113Z/alpha-max-phase-preparation-blocker-v2.json`, SHA `3b8c04b7a6b0a7d99e3d2b0ffb2dc42551c876b9ec5879663529b2b6173a07b2`.

## G004 snapshot at `f8ba7f1d`

This commit is a resumable implementation checkpoint, not a completion checkpoint and not a performance claim.

Implemented surfaces:

1. Default-off strict actual-engine dispatch:
   - `ResearchConfig.route_unmapped_registered_strategies = false`
   - `ResearchConfig.require_actual_engine_routing = false`
   - strict registry wrapper preserves exceptions;
   - strict dispatcher validates aligned arrays, symbols, close prices, timestamps, exposure shape/finiteness, handler mode, and fallback metadata;
   - legacy generic fallback remains unchanged when strict mode is off.
2. Frozen combined profile:
   - `configs/profiles/backtest_cost_realistic.yaml` arms strict routing and candidate-overfit statistics in the single replacement profile.
3. Shared MDD fix:
   - `research_metrics.max_drawdown` includes initial capital peak.
4. Exact two-candidate router manifest validator:
   - ordered R1/R2 IDs and SHA `ddc8996136e70d3847e8270f6165a26992ec8def8439ba6f56e3bcdbdee239b9`;
   - no new grid search, recompute-from-json, post-OOS augmentation, orders, or capital;
   - lifecycle/membership, leaf, handler/registry, cash/mature/scaled parity, duplicate/nonfinite parsing, and zero fallback/OOS-count checks.
5. Read-only `lq research cost-proof` surface:
   - strict JSON/YAML parsing and explicit external bindings;
   - cost ladder 10/15/20/30bp;
   - internal order/fill/position/PnL, funding, impact, grid, exposure, stop, liquidation, MDD, DSR/SPA/PBO, fold robustness, and deterministic selection checks;
   - source-row and provenance scaffolding.

Latest snapshot verification:

```text
uv run pytest -q \
  tests/research/test_cost_proof.py \
  tests/research/test_router_replay.py \
  tests/test_strategy_signal_dispatch.py \
  tests/test_strategy_signal_dispatch_routing.py
# 72 passed in 1.03s

uv run ruff check \
  src/lumina_quant/research/cost_proof.py \
  src/lumina_quant/research/router_replay.py \
  src/lumina_quant/strategy_factory/strategy_signal_dispatch.py \
  tests/research/test_cost_proof.py \
  tests/research/test_router_replay.py \
  tests/test_strategy_signal_dispatch.py \
  tests/test_strategy_signal_dispatch_routing.py
# All checks passed
```

Earlier combined G004 focused gate before the final review-driven hardening was 75 passed. No full-suite or final quality gate was run for `f8ba7f1d`.

## Mandatory G018 blocker plan

Three independent focused architects returned `BLOCK / CHANGES_REQUIRED` (tasks `58-G004CostContractReview`, `59-G004RouterContractReview`, `60-G004DispatchContractReview`). Their full output was session-local, so the actionable findings are preserved here.

### A. Strict dispatch — close first

1. Propagate strictness through mapped pair handlers. `_apply_pair_spread_strategy` currently catches actual-engine failure and substitutes `_pair_spread_fallback_exposures`; strict mode must raise with the original cause instead.
2. Validate final `portfolio_ret`, `turnover`, and aggregate `exposure` shape/finiteness before strict return; finite inputs can currently overflow derived arrays.
3. Derive registry simulation cadence correctly from production `numpy.datetime64[ms]`, not the legacy 60-second exception fallback.
4. Wrap malformed strict candidate/params coercion in `StrategySignalDispatchError` with cause.
5. Add tests for pair-simulator failure, arithmetic overflow, NumPy datetime cadence, malformed params, and the public research call chain.

### B. Router replay — make evidence authoritative and deterministic

1. Do not accept producer-declared branch/label/history/leaves as replay. Bind immutable prior-fold evidence and recompute the frozen warmup/history/lagged-average/train/MDD/validation decision.
2. Require byte-addressable signal/position/engine receipts; hash their actual bytes and bind fold, leaf, symbols, data/window, params, handler/class, and transitive engine dependency identity.
3. Authenticate shared fallback MDD inputs and deterministically recompute both R1 MDD30/cap3 and R2 MDD20/cap2 scales; only deterministic scale consequences may differ.
4. Parse and bind commit/freeze provenance to an out-of-band immutable root; the current source artifact mirrors the manifest and is circular.
5. Make combined-profile parsing recursively finite, closed enough for runtime-consumed fields, and exact-typed (`True` must not equal `1`; purge bar count must be an integer).
6. Convert huge-number/overflow inputs into `STOP` and add adversarial tests.

### C. Cost proof — replace self-consistency with independent proof

1. Bind every cost fold signal/order/execution tape slice to authenticated router execution receipt commitments.
2. Verify source rows against actual market/funding bytes or authenticated row/Merkle receipts; parse and semantically bind the data-contract and cost commit receipts.
3. Pin the exact frozen combined-profile byte digest and exact economically relevant semantics; do not accept arbitrary positive impact coefficients.
4. Authenticate bar volume/ADV. The shipped profile uses per-bar volume when `slippage_adv_quote=0`; fill-supplied volume is currently manipulable.
5. Reconcile cash and inventory from fills so realized fill-price PnL, unrealized mark PnL, fees/linear costs, impact, funding, and equity form one exact ledger.
6. Bind every period label to its declared validation/locked range. Remove implicit free position resets at segment boundaries or represent explicit flattening fills/cost/funding.
7. Derive default stop price from profile/fill and handle entry-bar stop/liquidation with authenticated event order; reject ambiguous OHLC order fail-closed.
8. Authenticate the complete attempted/skipped/failed whole-search trial ledger and bind every trial return row; `raw_trial_count == supplied IDs` is insufficient.
9. Run SPA/max-statistic over the authenticated whole trial family, not independently per selected stream.
10. Add one exploit-shaped test per invariant. Remove the duplicated first-fold `locked_ids` fixture assignment.

### D. Contract decisions that must follow the binding plan

- Master plan R2 explicitly says each candidate must first pass all binding gates; if both pass, select higher validation-20bp Calmar, then lower MDD. Do not silently replace that rule with a different pre-OOS selection algorithm solely because a reviewer suggested it.
- R1/R2 are fixed historical/post-OOS research variants, but `post_oos_augment=false`, candidate count/hash exactly two, and no new variant/search are mandatory. Preserve honest provenance and do not turn this infrastructure into a fresh-alpha claim.
- If stronger scientific constraints make the historical Router incapable of proof, terminate it scientifically in G005; never weaken G018 gates or manufacture evidence.

### E. Verification after implementation

1. Focused tests for each repaired contract and exploit.
2. Combined G004 tests (dispatch, routing, router replay, profile, cost proof, metrics).
3. Changed-file Ruff and format checks, `git diff --check`, AI-slop cleanup.
4. Full tracked regression suite in sanitized environment.
5. Evidence receipts under `/home/hoky/quants-recovery-runs/20260714T105113Z`.
6. Fresh three-lane architect review and executor adversarial QA.
7. Only CLEAR/CLEAR/CLEAR + APPROVE and executor PASS may checkpoint G018 and then G004.

## Exact resume sequence

1. Start in `/home/hoky/Quants-agent/LuminaQuant` and verify branch contains `f8ba7f1d`; do not reset or rewrite it.
2. Read this note, resume-state JSON, master plan, runbook, audit, durable goals, and durable ledger.
3. Activate `/skill:ultragoal`.
4. Set `GJC_SESSION_ID=019f603a-0e73-7000-88a7-c94f42950c09` and run `gjc ultragoal status --json`.
5. Resume the paused inline aggregate goal.
6. Run `gjc ultragoal complete-goals`; it must hand off the review-blocker story G018. Do not start G005.
7. Re-run the 72-test/Ruff snapshot gate before editing to confirm the handoff checkout.
8. Execute G018 in A → B → C → E order, using bounded executor slices and preserving the binding decisions in D.
9. Do not run any performance/data/network/order command during G018.
