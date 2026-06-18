# Pending Approval Plan — LuminaQuant Alpha/Strategy Improvement

Status: pending approval
Source spec: `.gjc/specs/deep-interview-alpha-strategy-improvement.md`
Planner: `.gjc/plans/ralplan/2026-06-07-0457-b89b/stage-02-planner.md`
Architect: `.gjc/plans/ralplan/2026-06-07-0457-b89b/stage-03-architect.md` — WATCH / APPROVE
Critic: `.gjc/plans/ralplan/2026-06-07-0457-b89b/stage-03-critic.md` — APPROVE

## Decision

Proceed, after explicit execution approval, with **Option A: extend the existing clean discovery + monthly walk-forward + artifact portfolio surfaces**. The implementation should broaden alpha search and existing-strategy reassessment without weakening clean research hygiene. The final output remains paper/shadow only unless later fresh-forward and live preflight gates are separately approved.

Do not proceed with a greenfield alpha engine this cycle, and do not treat `shadow70_clean30_v1` as the final answer. `shadow70_clean30_v1` is a benchmark/control and possible shadow-paper diagnostic, not an automatic promotion.

## Decision Drivers

1. **Research trust beats headline PnL**: locked-OOS, post-OOS selectors, lagged-shadow artifacts, and current-fold OOS must not influence selection, thresholds, tie-breaks, correlation, or sizing.
2. **Performance target is strict but two-tiered**: shadow candidates must beat `64.42%` compounded OOS or return/MDD `3.49`; clean/paper candidates must beat `34.39%` compounded OOS. All candidates still need MDD `<=30%`, no liquidation/account wipeout, data sufficiency, and cost/slippage viability.
3. **Maximum breadth requires staged narrowing**: the user wants as many candidates as feasible, so use a 1-2h broad smoke, then full WF only for frozen train/validation survivors.

## Principles

- Locked-OOS is report/gate only after train/validation freeze.
- Breadth first, strict gates second: smoke can be wide; promotion cannot be lenient.
- Implement only real leaf alphas as `Strategy` classes; compose allocation/risk through artifact portfolio manifests.
- Fail closed to cash/no-promotion on missing, stale, unreconciled, weak-data, OOS-contaminated, gross-cap-breaching, or liquidation-risk artifacts.
- Weak-data TradFi remains shadow/research-only; real-money execution remains excluded.

## Alternatives Considered

### Option A — Extend current clean discovery + monthly WF + artifact portfolio surfaces (chosen)

Pros: reuses existing clean selection machinery, monthly WF metrics, registry, portfolio mode, and tests; fastest path to broad smoke + strict-gate survivors; preserves no-promotion as an honest outcome.

Cons: touches large existing surfaces (`run_alpha_zoo_69_asset_monthly_refit_walkforward.py`, `artifact_portfolio_mode.py`) and therefore needs strong schema/freeze/fail-closed tests.

Chosen because it satisfies the approved spec with the fewest new moving parts.

### Option B — Greenfield alpha research engine

Pros: clean first-class schema and isolation. Cons: duplicates WF infrastructure, delays discovery, introduces metric mismatch risk, and still needs all clean invariants re-proven. Rejected unless Option A cannot isolate clean lanes or manifest validation.

### Option C — Portfolio-only shortcut around `shadow70_clean30_v1`

Pros: quick implementable shadow/paper control. Cons: fails the broad reassessment/new-alpha mandate and relies on correlated shadow material. Rejected as primary plan; retain as benchmark/control.

## Execution Plan After Approval

### Phase 1 — Schema and gate contract

Define or extend artifact fields for:
- candidate id, family/source bucket, symbols/timeframes, params, theory rationale;
- data sufficiency, weak-data shadow-only flag, feature coverage;
- `uses_locked_oos_for_objective`, `uses_locked_oos_for_selection`, `uses_locked_oos_for_threshold`, `uses_locked_oos_for_tie_break`, `uses_locked_oos_for_correlation`, `uses_locked_oos_for_sizing`;
- pre-registered search-space hash, train/validation freeze hash, survivor manifest hash;
- benchmark tier, MDD, liquidation/account-wipeout, cost/slippage, telemetry, fresh-forward status;
- promotion status and rejection reasons.

Constants:
- Shadow benchmark: `64.42%` compounded OOS or return/MDD `3.49`.
- Clean/paper benchmark: `34.39%` compounded OOS.
- MDD ceiling: `30%`.
- Fresh-forward minimum: 2 new monthly folds passing benchmark/risk/cost/telemetry.
- `ready_for_real=false` and `real_money_execution=false` until separate approval.

### Phase 2 — Broad smoke and existing-strategy reassessment

Extend/wrap `scripts/research/run_alpha_zoo_clean_new_alpha_discovery.py` to support family grouping, broad smoke budgets, candidate caps, row-output caps, weak-data labels, tried-universe logs, richer rejection reporting, and JSON+Markdown outputs.

Add an existing-strategy smoke lane over registered/runnable strategies from `src/lumina_quant/strategies/registry.py`. Use lenient smoke for coverage, but full-WF promotion only for strict audit pass candidates. Include current top rows and demoted historical rows as controls, not promotion shortcuts.

Outputs must include candidate-level metrics, audit flags, skipped strategies, tried universe, survivors, full-WF promotion candidates, correlation matrix, and rejection reasons.

### Phase 3 — New alpha family expansion

Cover these first-smoke buckets:
- crypto price/volume momentum, reversal, volatility-managed trend, breakout;
- cross-asset BTC/ETH/SOL lead-lag, beta residual, dispersion, cross-sectional residual momentum/reversal;
- funding/OI/taker-flow/BBO/depth/liquidation families using only existing feature coverage;
- existing-winner overlays built from train/validation-only evidence;
- TradFi/equity/commodity/FX-linked families where existing data permits.

Every new family needs theory metadata and feature/data coverage metadata. Weak-data TradFi is forced shadow/research-only.

### Phase 4 — Frozen survivor full WF and promotion gates

Freeze smoke survivors using train/validation evidence only. Send only frozen survivor manifests into monthly refit WF / long run. Attach locked-OOS only after freeze as report/gate evidence.

Promotion gate requires:
- correct two-tier benchmark pass;
- MDD `<=30%`;
- no liquidation/account wipeout;
- data sufficiency;
- cost/slippage viability;
- telemetry coverage;
- no locked-OOS selection/sizing/correlation leakage;
- strict audit pass;
- fresh-forward status clearly marked.

If no candidate passes, publish no-promotion plus best shadow watchlist and retain baseline.

### Phase 5 — Strategy class / registry implementation

Only strict-gate leaf survivors become new `Strategy` classes and registry entries. Do not implement nested/hybrid/meta/portfolio rows as clean leaf strategies. New strategies require explicit non-real tiering/tests; do not rely on default live tier behavior.

Strategy classes must emit real signals, declare data requirements, fail closed when required features are missing, and include signal metadata usable by portfolio composition.

### Phase 6 — Artifact portfolio manifest and risk gates

Implement a manifest-driven portfolio mode alongside existing aliases in `src/lumina_quant/strategies/artifact_portfolio_mode.py`.

Manifest validation must include:
- source artifact paths and sha256;
- generated time/freshness threshold;
- child readiness and reconciliation;
- no current-fold OOS provenance;
- train/validation-only optimizer provenance;
- top-level gross cap and per-leaf netting;
- correlation penalty metadata and input provenance;
- stale/missing/unreconciled/OOS-contaminated/gross-cap breach fail-closed-to-cash behavior.

Treat `shadow70_clean30_v1` as a replay/control benchmark with `2.25x` gross cap, not as final promotion.

### Phase 7 — Reports and handoff artifacts

Emit timestamped JSON+Markdown reports under approved `var/reports/...` paths:
- broad smoke report;
- existing strategy reassessment report;
- full-WF survivor/promotion report;
- no-promotion / shadow watchlist report when applicable;
- correlation report;
- portfolio manifest and replay report;
- live preflight checklist/telemetry report with `ready_for_real=false`.

## Required File Targets

Likely implementation targets after approval:
- `scripts/research/run_alpha_zoo_clean_new_alpha_discovery.py`
- a new or existing `scripts/research/` wrapper for existing-strategy smoke/reassessment
- `scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py` or a wrapper for frozen survivor full WF
- `src/lumina_quant/strategies/<new_leaf>.py` only for strict-gate survivors
- `src/lumina_quant/strategies/registry.py`
- `src/lumina_quant/strategies/artifact_portfolio_mode.py`
- `tests/test_alpha_zoo_clean_new_alpha_discovery.py`
- `tests/unit/test_artifact_portfolio_mode.py`
- focused tests for registry tiering, survivor manifests, and live/real-money vetoes

## Verification Plan

### Unit
- Locked-OOS perturbation leaves selection score, selected survivor, portfolio sizing, and freeze hash unchanged.
- Selector uses train/validation fields only.
- Weak-data TradFi rows are shadow/research-only.
- Report schema includes tried universe, skip reasons, gate constants, feature coverage, freeze hash, and rejection reasons.
- New Strategy classes fail closed on missing feature/data and are explicitly non-real tiered.
- Artifact portfolio manifest fails closed on missing/stale/unreconciled/OOS-contaminated children and gross-cap breach.

### Integration
- Synthetic broad smoke with tiny universe/folds and row-output cap.
- Existing-strategy smoke with mocked/synthetic bars and registry candidates.
- Frozen survivor manifest -> full WF wrapper preserves freeze hash before OOS attachment.
- Manifest replay deterministically reproduces weights, per-leaf netting, gross cap, and cash failover.

### E2E / Research after code tests pass
- Run 1-2h broad smoke over approved candidate families/universe.
- Run full WF only for strict smoke survivors.
- Generate correlation and portfolio manifest reports.
- Start fresh-forward shadow observation only when future monthly folds exist.

### Observability
Reports must expose generated time, data coverage, enabled families, search-space hash, freeze hash, selection input fields, locked-OOS report-only status, benchmark constants, gate matrix, blockers, ready-for-real flags, cost/slippage, trade counts, liquidation/account wipeout flags, feature coverage, and artifact freshness.

## Architect WATCH Constraints Incorporated

- Locked-OOS remains report/gate only.
- Weak-data TradFi remains shadow/research-only.
- Leaf Strategy classes remain separate from portfolio/risk manifests.
- Portfolio sizing uses train/validation only; no locked-OOS or current-fold OOS correlation/tie-break/sizing input.
- Manifest validation fails closed on stale/missing/unreconciled/OOS-contaminated children or gross-cap breach.
- Real-money remains excluded.
- New Strategy survivors require explicit non-real tiering and tests proving `ready_for_real=false` and `real_money_execution=false` until separate approval and fresh-forward evidence.
- No-promotion is acceptable; benchmarks and MDD/liquidation gates must not be weakened.

## Pre-Mortem

1. **Hidden OOS contamination creates a fake winner.** Detection: locked-OOS perturbation tests, explicit selection-input report fields, freeze hash before OOS attachment. Response: quarantine lineage, repair selector/schema, rerun.
2. **Broad smoke exceeds budget and biases coverage.** Detection: runtime and tried-universe coverage report, missing family skip reasons. Response: deterministic family shards and promotion only from audited shards.
3. **Portfolio aggregate looks good but cannot trade safely.** Detection: manifest replay/fail-closed tests, source artifact freshness, child reconciliation, gross/net exposure and correlation reports. Response: cash/no-promotion, not a weaker gate.

## ADR

### Decision
Use an incremental, hard-gated extension of the current clean discovery, monthly WF, strategy registry, and artifact portfolio machinery.

### Drivers
Clean research trust, strict benchmark/risk gates, maximum feasible breadth under a staged budget, and implementation realism.

### Alternatives Considered
- Greenfield engine: cleaner but too slow and duplicates tested infrastructure.
- Portfolio-only shortcut: fast but fails the mandate and over-relies on correlated shadow candidates.
- Option A: chosen because it reuses existing tested surfaces while adding explicit schema/freeze/fail-closed boundaries.

### Why Chosen
Option A gives the best chance to search many candidates while preserving split hygiene and producing implementable artifacts. The architect approved with WATCH constraints, and the critic approved without required changes.

### Consequences
Implementation may still produce no-promotion; that is acceptable. Work must prioritize schema/freeze/fail-closed correctness before broad research runs. Existing large files may need wrapper boundaries to prevent contamination.

### Follow-ups
After explicit execution approval, prefer `ultragoal` for durable goal-tracked implementation. Use `team` only if tmux-based parallel smoke/WF/report workers are explicitly required.

## Pending Approval Boundary

This plan is pending approval. No product source has been edited by RALPLAN, no tests/formatters were run in planning, and no research/backtests were executed. Execution requires separate explicit approval and should start from this pending plan.
