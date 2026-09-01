**APPROVE**

**Justification**: Planner and architect artifacts are actionable. I verified the spec, prior plan, current reports, source surfaces, and tests. The plan preserves train/validation-only selection, locked-OOS report/gate use only, two-tier benchmarks, MDD/no-liquidation gates, weak-data TradFi shadow-only, real-money exclusion, leaf Strategy vs manifest allocation boundary, fail-closed manifest behavior, no-promotion outcome, and WATCH constraints.

**Summary**:
- Clarity: Clear sequencing from schema/gates to smoke, survivor full WF, Strategy leaves, portfolio manifest, and reporting. Representative file targets exist; placeholder `<new_leaf>` is appropriate.
- Verifiability: Strong enough. Unit, integration, e2e/research, and observability checks are concrete, including locked-OOS perturbation, freeze hash invariance, synthetic broad smoke, manifest replay, stale/missing fail-closed, gross cap, and report fields.
- Completeness: Covers spec topology and artifacts. Verified `current_top_models_20260618.md` constants, correlation report `shadow70_clean30_v1` constraints, prior RALPLAN quarantine rules, clean discovery runner, monthly WF leaf filters, registry tier surface, portfolio mode, and relevant tests.
- Big Picture: Option A is a fit: broad research remains in research lanes, full WF is survivor-only, and implementation only follows strict gates. WATCH remains around monthly runner coupling, artifact manifest validation, and registry tier defaults.
- Principle/Option Consistency: No contradiction found. Option C remains benchmark/control only; greenfield stays fallback if isolation or fail-closed replay fails.
- Alternatives Depth: Fair. Greenfield and portfolio-only shortcut are steelmanned enough for this iteration, and invalidation triggers are present.
- Risk/Verification Rigor: Deliberate-mode pre-mortem is present and aligned with risks. Expanded test plan is sufficient before execution.

**File/reference verification**:
- Verified spec `.gjc/specs/deep-interview-alpha-strategy-improvement.md`.
- Verified planner `stage-02-planner.md` and architect `stage-03-architect.md`.
- Verified reports `var/reports/current_top_models/current_top_models_20260618.md` and `var/reports/current_top_models/top_strategy_correlation_portfolio_20260618.md`.
- Verified prior plan `pending-approval.md`.
- Verified source targets `scripts/research/run_alpha_zoo_clean_new_alpha_discovery.py`, `scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py`, `src/lumina_quant/strategies/registry.py`, and `src/lumina_quant/strategies/artifact_portfolio_mode.py`.
- Verified tests `tests/test_alpha_zoo_clean_new_alpha_discovery.py`, `tests/unit/test_artifact_portfolio_mode.py`, registry/plugin tests, and real-money veto tests.

**Representative implementation simulation**:
1. Discovery/schema update: existing clean discovery scoring and tests already ignore locked-OOS report fields and write policy flags; adding tried-universe, weak-data, benchmark, and freeze fields is direct and testable.
2. Existing-strategy smoke/full-WF wrapper: registry APIs expose strategy names/classes and monthly WF already has non-leaf filters; a wrapper can enumerate strategies, record skips, freeze train/validation survivors, then attach locked-OOS only after freeze.
3. Manifest portfolio: current portfolio mode is alias-backed and signal-scaling; a manifest path can be added with source artifact sha, freshness, child readiness, gross cap, current-fold-OOS checks, deterministic replay, and cash failover while preserving alias tests.

**Architect WATCH constraints approved as mandatory execution constraints**:
- Locked-OOS remains report/gate only.
- Weak-data TradFi remains shadow/research-only.
- Leaf Strategy classes stay separate from portfolio/risk manifests.
- Portfolio sizing uses train/validation only; no locked-OOS or current-fold OOS correlation, tie-break, or sizing input.
- Manifest validation fails closed on stale/missing/unreconciled/OOS-contaminated children or gross-cap breach.
- Real-money remains excluded.
- New Strategy/registry survivors must have explicit non-real tiering and tests proving `ready_for_real=false` and `real_money_execution=false` until separate approval and fresh-forward evidence.
- No-promotion is acceptable; benchmarks and MDD/liquidation gates must not be weakened.

**Required fixes**: none.
