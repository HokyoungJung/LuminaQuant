# RALPLAN — StateDistilledRegimeBoostPortfolio

## Scope
Implement a research-only `StateDistilledRegimeBoostPortfolio` path that consumes existing state-distilled candidates and overlays: regime classifier, side bias, volatility-targeted dynamic leverage, conditional booster leverage up to 25x gated by long-term asset volatility/margin safety, and neutral pair overlay. Selection/tuning uses train+validation only; locked-OOS is opened only after candidate freeze for gate/report. No calendar rules. All thresholds/maps/weights/gates tunable. Memory budget: <8 GiB.

## Evidence anchors
- Context: `.omx/context/state-distilled-regime-boost-20260513T133805Z.md`.
- Likely reusable engines: `scripts/research/replay_profit_moonshot_fresh_start.py`, `scripts/research/run_profit_moonshot_liquidation_aware_validation.py`, `scripts/research/replay_crypto_fx_alpha_zoo_state.py`.
- Existing invariants/tests: `tests/test_profit_moonshot_liquidation_aware_validation.py`, `tests/test_crypto_fx_alpha_zoo_state_strategy.py`, `tests/test_profit_moonshot_pass_under_8gb_validator.py`.
- Research docs to update/supersede: `docs/research_note/state_distilled.md`, `docs/research_note/research_note.md`, new handoff doc.

## RALPLAN-DR summary
### Principles
1. Preserve train/validation-only selection; locked-OOS is gate/report-only after freeze.
2. Fail closed on strategy validity: calendar/month/day/hour rules, liquidations, or non-positive margin buffer block strict promotion.
3. Parameterize every research threshold, regime rule, side multiplier, leverage map, booster gate, neutral pair weight, and allocation rule.
4. Bound memory and compute; stream/load minimal artifact columns and cap grids to <8 GiB RSS.
5. Separate strict deployable lane from diagnostic/high-leverage booster lanes in artifacts.

### Decision drivers
1. Avoid leakage/overfit while evaluating richer overlays.
2. Reuse existing liquidation-aware and state-distilled infrastructure instead of introducing parallel semantics.
3. Produce auditable artifacts that explain leverage choice, booster eligibility, and OOS gate outcome.

### Options considered
- **A. New bounded research script reusing existing helpers** — chosen. Low coupling, clear artifact boundary, easiest to enforce train/val freeze and memory guard.
- **B. Extend liquidation validator directly** — rejected as primary because it risks mixing candidate construction/tuning with liquidation validation; still reuse its margin functions where practical.
- **C. Add live strategy class first** — rejected until research proves strict validity; live/deploy lane must follow validated artifact.

## ADR
**Decision:** Add a dedicated research runner, likely `scripts/research/run_state_distilled_regime_boost_portfolio.py`, plus tests and docs. It will load existing state-distilled candidate artifacts, build/tune overlay configs on train+validation, freeze a single strict candidate, then run locked-OOS only as gate/report.

**Drivers:** leakage control, leverage/liquidation safety, memory budget, explainable research artifact.

**Alternatives:** mutate current replay/validator scripts; build live strategy immediately; use locked-OOS in tuning. These are rejected due to leakage, unclear ownership, or premature deployment coupling.

**Consequences:** Some helper extraction may be needed from existing scripts; research artifacts become the source of truth before any live class exists.

**Follow-ups:** If strict gate passes, plan a separate live-opt-in strategy implementation with the frozen config only; if not, keep high-leverage output diagnostic-only.

## Concrete implementation plan
1. **Artifact/data intake**
   - Inventory existing state-distilled outputs under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/` and confirm available fields: candidate id/spec, split, timestamps, symbol, side, returns/equity, OHLCV/high/low, liquidation metadata if present.
   - Define input CLI args for candidate paths, output dir, split names, memory budget, grid limits, and tunable parameter JSON.

2. **Core runner + config model**
   - Add dataclasses/config payloads for: `RegimeClassifierConfig`, `SideBiasConfig`, `VolTargetLeverageConfig`, `BoosterConfig`, `NeutralPairOverlayConfig`, `SelectionConfig`, `MemoryConfig`.
   - Ensure defaults are documented as tunable defaults, not hidden pseudocode thresholds; write full config into every JSON/MD artifact.

3. **Overlay construction**
   - Regime classifier: derive non-calendar state labels from lagged/rolling market/candidate features only.
   - Side bias: map classifier state and candidate side to tunable long/short exposure multipliers.
   - Dynamic leverage: compute per-asset long-term realized volatility and target leverage from tunable target-vol/map bounds.
   - Conditional booster: allow booster leverage up to 25x only when tunable long-term volatility, liquidity, margin-buffer, and liquidation-risk gates pass; otherwise fall back to base dynamic leverage.
   - Neutral pair overlay: add market-neutral pair sleeve with tunable hedge ratio/weight and explicit separation from directional booster exposure.

4. **Selection and locked-OOS discipline**
   - Evaluate grid on train+validation only; selection score must contain no locked-OOS metrics.
   - Freeze selected config/candidate before any locked-OOS replay.
   - Locked-OOS output must show `candidate_freeze_before_locked_oos_gate=true`, `uses_locked_oos_for_selection=false`, and `locked_oos_metrics_visible_during_selection=false`.

5. **Liquidation/margin safety**
   - Reuse/adapt margin model and split gate logic from `run_profit_moonshot_liquidation_aware_validation.py`.
   - Strict promotion requires zero liquidation count and positive min margin buffer across train, validation, and locked-OOS.
   - Booster/high leverage lanes with any liquidation or non-positive buffer remain diagnostic-only.

6. **Memory and artifact reporting**
   - Use `acquire_portfolio_memory_guard` / `memory_policy_payload`; emit peak RSS and memory policy in summary.
   - Prefer lazy/batched reads and selected columns; cap grid size via CLI.
   - Write artifacts under `var/reports/.../alpha_v2/state_distilled_regime_boost_20260513/`: summary JSON, markdown report, grid ledger JSONL/CSV, frozen config JSON, OOS gate JSON, memory log.

7. **Tests**
   - Add `tests/test_profit_moonshot_regime_boost_portfolio.py` covering: no calendar fields/rules; all thresholds from config; train/validation selection ignores OOS poison; freeze before locked-OOS; dynamic leverage/booster cap <=25x; booster disabled by high long-term vol; neutral pair overlay is dollar/market neutral within tolerance; liquidation gates block promotion; memory payload emitted.
   - Add focused regression tests if helper extraction touches existing validator/replay scripts.

8. **Docs and handoff**
   - Update/supersede `docs/research_note/state_distilled.md` and, if Alpha Zoo/current-tail chronology changes, `docs/research_note/research_note.md`.
   - Add `docs/session_handoff_20260513_state_distilled_regime_boost.md` with commands, artifacts, selected config, strict/diagnostic lane decision, CI status.
   - Update research history only if global chronology/source ledger changed; otherwise document why not.

9. **Commit/push/CI handoff**
   - Commit with Lore protocol after tests and artifacts are updated.
   - Push to `private/main`/tracking branch and verify private CI green before declaring done.

## Likely files
- Add: `scripts/research/run_state_distilled_regime_boost_portfolio.py`
- Add: `tests/test_profit_moonshot_regime_boost_portfolio.py`
- Maybe edit: `scripts/research/run_profit_moonshot_liquidation_aware_validation.py` only for safe helper reuse/extraction if needed.
- Maybe edit: `scripts/research/replay_profit_moonshot_fresh_start.py` only if candidate artifact parsing must expose reusable state-distilled helpers.
- Edit docs: `docs/research_note/state_distilled.md`, maybe `docs/research_note/research_note.md`, add `docs/session_handoff_20260513_state_distilled_regime_boost.md`.
- New artifacts: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_regime_boost_20260513/`.

## Acceptance criteria
- Runner exists and can evaluate existing state-distilled candidates with regime, side-bias, dynamic vol-target leverage, conditional <=25x booster, and neutral pair overlay.
- All thresholds/maps/weights/gates are configurable and serialized; tests fail if hidden hardcoded pseudocode thresholds drive selection.
- No calendar/month/day/hour alpha or selection rules are introduced.
- Selection is train+validation only; locked-OOS is opened only after freeze and only for gate/report.
- Strict promotion is false unless zero liquidations and positive min margin buffer hold across all splits.
- Memory peak stays <8 GiB and is recorded in artifacts.
- Research notes/report artifacts state selected candidate, leverage/booster rationale, strict vs diagnostic outcome, and deployability decision.
- Targeted tests, broader relevant tests, commit, push, and CI are complete.

## Verification commands
```bash
# targeted unit/regression
.venv/bin/pytest tests/test_profit_moonshot_regime_boost_portfolio.py -q
.venv/bin/pytest tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_crypto_fx_alpha_zoo_state_strategy.py tests/test_profit_moonshot_pass_under_8gb_validator.py -q

# smoke research run, with bounded grid and explicit memory guard
/usr/bin/time -v .venv/bin/python scripts/research/run_state_distilled_regime_boost_portfolio.py \
  --candidate-root var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2 \
  --output-dir var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_regime_boost_20260513 \
  --max-rss-bytes 8589934592 \
  --grid-limit <bounded_limit> \
  --config <tunable_config.json>

# inspect artifacts for leakage/safety flags
python - <<'PY'
import json, pathlib
p=pathlib.Path('var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_regime_boost_20260513/summary_latest.json')
d=json.loads(p.read_text())
assert d['selection']['uses_locked_oos_for_selection'] is False
assert d['selection']['candidate_freeze_before_locked_oos_gate'] is True
assert d['memory']['peak_rss_bytes'] < 8589934592
assert d['booster']['max_effective_leverage'] <= 25
PY

# style/static as project supports
.venv/bin/python -m compileall scripts/research/run_state_distilled_regime_boost_portfolio.py
# then git status, commit with Lore protocol, push, verify CI
```

## Risks and mitigations
- **Candidate artifacts lack per-trade streams:** first inspect; if absent, reconstruct from existing replay helpers under bounded grid, not from OOS-selected rows.
- **Dynamic leverage introduces liquidation risk:** strict gate blocks promotion; high-leverage lanes are diagnostic unless all buffers/liquidations pass.
- **Overfit via large grid:** cap grid, log trial count, selection inputs, and freeze provenance.
- **Memory blow-up:** lazy/batched reads, grid caps, memory guard, `/usr/bin/time -v` evidence.
- **Helper extraction regression:** keep edits small and protect with existing liquidation/replay tests.


## Critic/Architect hardening addendum — 2026-05-13

Execution must preserve these non-negotiable safeguards:

1. **Immutable freeze provenance**
   - Freeze payload includes selected candidate ids, full selected config, train/validation ledger hash, input artifact hashes, code SHA/dirty marker, grid metadata, search-space hash, and `frozen_at`.
   - Freeze SHA-256 hash is recorded in a separate sidecar/manifest, not inside the hashed payload.
   - Locked-OOS gate artifact records `locked_oos_opened_at` and references the exact freeze path/hash; selected params must be byte-identical to frozen params.
2. **Bounded grid / overfit guard**
   - Default grid cap is 64; hard max is 256. Enforce `evaluated_count <= min(config.grid_limit, 256, product_space_size)`.
   - Emit configured grid size, evaluated count, skipped/pruned count, search-space hash, and selection-score fields.
   - Locked-OOS metrics cannot influence pruning/ranking.
3. **Neutral-pair leakage guard**
   - Pair universe, pair choice, hedge ratio, dispersion trigger, and overlay weights are fit/frozen from lagged train/validation features only.
   - OOS-only-good pairs cannot be selected; pair feature provenance must include as-of/lag statement.
4. **Strict OOS MDD gate**
   - Strict promotion requires locked-OOS max drawdown <=25%, zero liquidations, and positive min margin buffer.
   - Return/MDD ratio remains diagnostic-only and is not a hard promotion gate.

## Staffing / handoff guidance
### Ralph path
Use `$ralph` for single-owner implementation when minimizing coordination overhead matters. Suggested lane: one executor implements runner/tests/docs sequentially, then verifier checks artifacts and CI. Reasoning: executor medium/high, verifier high.

### Team path
Use `$team` if parallelizing:
- `explore`/analyst: artifact schema inventory and current helper map.
- `executor-1`: runner/config/overlay implementation.
- `executor-2`: tests and helper extraction guards.
- `writer`: docs/report/handoff updates.
- `verifier`: leakage, memory, liquidation, artifact, CI review.
Team verification path: targeted tests -> smoke run -> artifact assertions -> docs consistency -> git diff review -> commit/push -> CI.

Available agent types: `explore`, `planner`, `architect`, `executor`, `test-engineer`, `writer`, `verifier`, `code-reviewer`, `critic`.

## Goal-mode suggestion
Use `$performance-goal` if the leader wants durable goal tracking because the task has explicit memory (<8 GiB), bounded-grid, and leverage/risk optimization evaluators. Otherwise `$ultragoal` is acceptable for general implementation tracking; `$ralph` or `$team` remain better for direct delivery.
