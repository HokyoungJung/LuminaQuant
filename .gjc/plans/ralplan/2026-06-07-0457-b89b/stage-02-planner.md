# RALPLAN Planner Artifact — LuminaQuant Alpha/Strategy Improvement

Status: pending approval / planning only. No product source was edited, no tests or formatters were run, and no research/backtest execution was performed. Stage 1 for the active run already exists with different content, so this planner pass is persisted as stage 2 without overwriting user/workflow history.

## Summary

Implement the approved deep-interview spec `.gjc/specs/deep-interview-alpha-strategy-improvement.md` by broadening clean alpha discovery, reassessing existing registered/runnable strategies, expanding crypto + existing-data TradFi alpha families, and promoting only strict-gate survivors into Strategy classes/registry entries plus artifact portfolio manifests. Required gates are fixed: shadow candidates must beat `64.42%` comp or return/MDD `3.49`; clean/paper candidates must beat `34.39%` comp. All promotions also require MDD `<=30%`, no liquidation/account wipeout, train/validation-only selection, sufficient data, cost/slippage realism, JSON+Markdown reporting, and fresh-forward observation before any real-ready consideration. Real-money execution remains excluded.

Inspected evidence: the spec defines four active components; `current_top_models_20260618.md` provides the shadow and clean benchmarks and says real-money is blocked; `top_strategy_correlation_portfolio_20260618.md` shows top rows are highly correlated and records `70% risk_trim + 30% clean_mdd20` as a shadow/paper diagnostic with `2.25x` gross cap; prior RALPLAN requires locked-OOS report/gate only and candidate freeze before OOS attachment; `run_alpha_zoo_clean_new_alpha_discovery.py` already has train/validation selectors, search hashes, feature coverage, JSON+Markdown output, and `ready_for_real=false`; `artifact_portfolio_mode.py` resolves artifact-backed modes and scales child signal metadata; `registry.py` supports curated and plugin Strategy registration.

## Principles

1. Locked-OOS is report/gate only after train/validation freeze; never objective, pruning, threshold, tie-break, correlation, or portfolio sizing input.
2. Breadth first, strict gates second: 1-2h smoke can be wide, but full WF and implementation are survivor-only.
3. Implement leaf alphas as real Strategy classes/registry entries; compose risk and allocation through artifact portfolio manifests.
4. Fail closed: missing/stale artifacts, weak data, liquidation flags, MDD breach, or benchmark failure mean cash/no-promotion plus rejection reasons.
5. Shadow is not real-ready: weak-data TradFi and post-OOS-inspired hypotheses remain shadow/research-only; fresh-forward is mandatory.

## Top 3 decision drivers

1. Research hygiene: the user rejected locked-OOS/post-hoc trust, so contamination must be technically testable.
2. Benchmark pressure: current clean `34.39%` is weak but near the `30%` MDD ceiling; higher-return `64.42%` shadow evidence is not clean/paper-promotable by itself.
3. Throughput: maximum search breadth is desired, but only a smoke -> full WF -> fresh-forward ladder keeps runtime and overfit risk controlled.

## Options considered

### Option A — Extend current clean discovery + monthly WF + artifact portfolio surfaces (chosen)

Use `run_alpha_zoo_clean_new_alpha_discovery.py` for broad smoke/shortlisting, add existing-strategy smoke around registry/runnable strategies, send strict survivors into monthly refit WF, implement surviving leaf alphas as Strategy classes/registry entries, and add manifest risk gates in `artifact_portfolio_mode.py`.

Pros: reuses existing clean-selection machinery and tests; fits broad smoke then survivor-only execution; handles crypto plus existing-data TradFi; uses existing Strategy and portfolio surfaces; no-promotion stays auditable.

Cons: runner scope must be controlled; artifact portfolio mode already has many aliases and must not regress; runtime must be capped to 1-2h.

Chosen because it satisfies the spec with the fewest new moving parts. Invalidate if representative smoke cannot fit the budget, existing Strategy smoke cannot be isolated from OOS/live side effects, or manifest replay cannot be made deterministic/fail-closed without breaking existing modes.

### Option B — Greenfield alpha research engine

Pros: clean separation from historical code; coherent schema from day one; long-term parallelization potential. Cons: high implementation cost, duplicate monthly WF infrastructure, new metric-mismatch risk, and no immediate trust gain unless all clean invariants are re-proven. Rejected for this iteration unless Option A is invalidated.

### Option C — Implement only `shadow70_clean30_v1`

Pros: fastest implementable shadow/paper artifact; useful gross-cap and replay test bed. Cons: fails the approved mandate for broad reassessment/new families, current components are correlated and partly shadow-only, and it risks treating a diagnostic portfolio as the final answer. Rejected as the primary plan; retain as benchmark/control.

## In scope / out of scope

In scope after approval: broad 1-2h smoke over existing strategies and new crypto + existing-data TradFi candidates; existing-strategy audit; new family expansion across price/volume, cross-asset/residual/dispersion, funding/OI/taker-flow/BBO/depth/liquidation, existing-winner overlays, and data-sufficient TradFi-linked buckets; strict clean gates; two-tier benchmarks; JSON+Markdown reports; Strategy classes for surviving leaves; artifact portfolio manifest with deterministic replay, gross cap, correlation penalty, fail-closed behavior, and `ready_for_real=false`.

Out of scope: product mutation before approval; tests/formatters/research/backtests in planning stage; real-money execution; exchange routing changes; new external vendor/on-chain/news data; locked-OOS tuning or post-OOS meta-selection; sub-minute live execution.

## File-level changes

- `scripts/research/run_alpha_zoo_clean_new_alpha_discovery.py`: add/organize broad-smoke modes, family grouping, runtime/candidate caps, tried-universe coverage, weak-data labels, survivor/promotion decisions, and richer JSON+Markdown rejection reporting while preserving locked-OOS report-only semantics.
- New or existing `scripts/research/` wrapper: enumerate registry/runnable strategies for existing-strategy reassessment, apply lenient smoke audit, record skip/audit flags, and forward only strict survivors to full WF.
- Monthly WF surface, likely `scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py` or a wrapper: accept frozen survivor manifests, attach locked-OOS only after freeze, and emit benchmark/gate status.
- `src/lumina_quant/strategies/<new_leaf>.py` plus `registry.py` or plugin decorators: implement only strict-gate leaf survivors as signal-emitting Strategy classes with appropriate research/shadow/paper tiering; do not implement nested/hybrid/meta rows as clean leaves.
- `src/lumina_quant/strategies/artifact_portfolio_mode.py`: add manifest-driven composition with gross cap, component freshness, source-artifact reconciliation, child availability, cash failover, correlation metadata, and no-current-fold-OOS provenance while preserving existing modes.
- Tests: extend `tests/test_alpha_zoo_clean_new_alpha_discovery.py`, `tests/unit/test_artifact_portfolio_mode.py`, and add focused Strategy/registry tests for new survivors.
- Approved execution reports under `var/reports/...`: timestamped smoke, full WF, survivor/rejection, correlation, and portfolio manifest JSON+Markdown.

## Sequencing and dependencies

### Phase 0 — Approval boundary
Keep this artifact pending approval. Before implementation, re-read the spec, top reports, prior RALPLAN, runner, artifact portfolio mode, registry, and tests. Do not run research/backtests until approval.

### Phase 1 — Schema and gate contract
Define fields for candidate identity, family/source bucket, data sufficiency, weak-data shadow-only status, theory gate, `uses_locked_oos_for_*`, search/freeze hashes, selected-by-freeze, benchmark tier, MDD/liquidation/cost/telemetry gates, promotion status, and rejection reasons. Encode constants: shadow `64.42%` or `3.49`, clean `34.39%`, MDD `30%`, no liquidation/account wipeout, fresh-forward `>=2` new monthly folds.

### Phase 2 — Broad smoke and existing reassessment
Run bounded 1-2h smoke across enabled family buckets with controls for family, symbols, folds, candidates, row output, backend, and workers. Add existing-strategy smoke over registry/runnable strategies, including current top and demoted historical candidates as controls. Record tried universe, skips, audit flags, survivors, full-WF promotion list, and no-promotion outcome. Rank by train+validation only.

### Phase 3 — New alpha expansion
Prioritize: crypto momentum/reversion/vol-managed trend/breakout; cross-asset BTC/ETH/SOL lead-lag and residual/dispersion; derivatives/microstructure using existing funding/OI/taker-flow/BBO/depth/liquidation data; existing-winner overlays derived from train/validation; TradFi-linked candidates only where existing data permits, with weak-data rows forced shadow/research-only. Every family gets theory and feature-coverage metadata.

### Phase 4 — Survivor full WF and promotion
Freeze smoke survivors by train/validation evidence, hash the manifest, send only frozen rows into monthly refit WF/long run, then attach locked-OOS metrics as report/gate evidence. Compute benchmark, return/MDD, Calmar, MDD, liquidation/account-wipeout, cost/slippage, telemetry, and fresh-forward-required status. If none pass, output no-promotion and best shadow watchlist.

### Phase 5 — Strategy implementation
For each strict-gate leaf survivor, implement a real Strategy class with explicit params, data requirements, fail-closed behavior, and signal metadata. Register through existing conventions and tier appropriately. Skip this phase if no strict-gate leaf survives.

### Phase 6 — Portfolio manifest and risk gates
Build final portfolio from frozen leaf components, not locked-OOS optimization. Optimize using train/validation-only return/MDD or Calmar objective with gross cap, MDD `<=30%`, no-liquidation constraint, correlation penalty, and freshness checks. Include `shadow70_clean30_v1` as benchmark/control, not automatic final. Replay deterministically and fail to cash on missing/stale/unreconciled/OOS-contaminated children or gross-cap breach.

### Phase 7 — Reporting and handoff
Emit JSON+Markdown for smoke, existing reassessment, full WF promotions, correlation, survivor/rejection matrix, and portfolio manifest. Include RALPLAN-DR, benchmark constants, `ready_for_real=false`, real-money exclusion, and fresh-forward watchlist.

## Acceptance criteria

- Broad smoke covers as many feasible existing strategies and new crypto + existing-data TradFi candidates as the 1-2h budget permits.
- Smoke output records candidate metrics, audit flags, tried-universe coverage, skip/rejection reasons, survivor list, full-WF promotion list, JSON, and Markdown.
- Full WF candidates are selected only from train/validation smoke evidence and frozen manifests, never locked-OOS evidence.
- Shadow promotion beats `64.42%` comp or return/MDD `3.49` and passes MDD `<=30%`, no liquidation/account wipeout, data, cost/slippage, telemetry, and provenance gates.
- Clean/paper promotion beats `34.39%` comp and passes the same gates.
- Weak-data TradFi remains shadow/research-only until data sufficiency improves.
- Surviving leaf alphas become Strategy classes/registry entries only after theory/audit checks.
- Final composition is an artifact portfolio manifest with deterministic replay, fail-closed behavior, gross cap, correlation penalty, and source-artifact provenance.
- Fresh-forward promotion requires at least two new monthly folds passing benchmark/risk/cost/telemetry; real-money is excluded.
- If no candidate passes, retain baseline and publish no-promotion plus best shadow watchlist.

## Verification

No verification commands were run in planner stage. After approval:

Unit: extend `tests/test_alpha_zoo_clean_new_alpha_discovery.py` for locked-OOS score invariance, selector behavior, feature coverage, pre-registration, report fields, weak-data labels, and policy flags; extend `tests/unit/test_artifact_portfolio_mode.py` for manifest resolution, recursion, missing/stale fail-closed behavior, gross caps, scaling, unbounded-child caps, and source artifacts; add Strategy/registry tests for new leaves.

Integration: synthetic broad smoke with tiny symbols/folds and `--max-candidate-rows-output 0`; existing-strategy smoke with mocked/synthetic bars; frozen survivor -> full WF wrapper with unchanged freeze hash before OOS attachment; portfolio manifest replay with deterministic weights, netting, gross cap, cash failover, and stale/missing rejection.

E2E/research after code tests pass: 1-2h broad smoke over approved universe/families, full WF only for strict smoke survivors, correlation and portfolio manifest report generation, and deferred fresh-forward shadow monitoring when new folds exist.

Observability: reports must expose generated time, search-space hash, freeze hash, enabled families, universe coverage, skips, locked-OOS report-only status, benchmark constants, gate matrix, blockers, `ready_for_real=false`, costs/slippage, trade counts, data freshness, feature coverage, liquidation flags, and artifact freshness.

## Risks and mitigations

1. Runtime blow-up from broad families. Mitigate with candidate caps, family grouping, max folds, max row output, Rust backend diagnostics, workers, and 1-2h smoke budget.
2. Hidden locked-OOS contamination. Mitigate with `uses_locked_oos_for_*` fields, perturbation tests, freeze hash before OOS attachment, and report-only naming.
3. Manifest implementation regresses existing modes. Mitigate by adding manifest support alongside aliases, preserving tests, adding compatibility tests, and failing closed.
4. Weak-data TradFi appears attractive but is not clean. Mitigate with data sufficiency flags and forced shadow/research-only status.
5. No candidate beats gates. Mitigate by accepting no-promotion, retaining baseline, publishing watchlist/rejection reasons, and not lowering gates.

## Pre-mortem

1. High-return candidate wins because locked-OOS was indirectly used in tie-break/correlation selection. Detect via perturbation tests and explicit selection-input report; respond by quarantining lineage and rerunning after repair.
2. Broad smoke exceeds budget and evaluates a biased subset. Detect via runtime/coverage report with missing family skip reasons; respond by deterministic family shards and only promote audited shards.
3. Final portfolio passes aggregate return but cannot be traded safely due stale/missing children or hidden gross/correlation risk. Detect via manifest replay/fail-closed tests, artifact freshness, gross/net exposure report, and correlation matrix; respond with cash/no-promotion.

## Handoff guidance

Use `executor` for runner/schema/report changes, Strategy leaf classes, registry tests, and manifest risk gates. Use `architect` for read-only review if manifest schema or runner split grows. Use `critic` if any shortcut touches locked-OOS or thresholds. Use `team` only for approved persistent parallel smoke/WF/report workers. Use `ultragoal` only if execution becomes a durable multi-day ledger.

## RALPLAN-DR compact summary

Decision: Option A, incremental clean-discovery/monthly-WF/artifact-manifest implementation; reject greenfield rewrite and portfolio-only shortcut.

Drivers: clean research trust; benchmark improvement under MDD/no-liquidation; 1-2h broad-smoke throughput.

Rules: locked-OOS report/gate only; shadow `>64.42%` or return/MDD `>3.49`; clean/paper `>34.39%`; MDD `<=30%`; no liquidation; weak-data TradFi shadow-only; real-money excluded.

Execution: schema/gates -> broad smoke -> existing reassessment -> new alpha expansion -> survivor full WF -> Strategy/registry for leaf survivors -> artifact portfolio manifest -> JSON+Markdown reports and verification.

Status: pending approval; planning-only artifact; no product mutation, tests, or research/backtest execution in planner stage.
