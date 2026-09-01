# PRD — Autonomous Portfolio Research Loop Under 8 GiB Memory

## Task
Design and execute an autonomous research loop that explains the current portfolio instability, audits the existing strategy/indicator/tuning/portfolio/backtest stack, researches better methods and new alpha ideas from credible English sources, tests them under the existing LuminaQuant stack, keeps wins, discards losers, logs crashes/failed ideas, and advances the branch without pausing for user confirmation until manually interrupted.

## Current Evidence
- The current 3-sleeve incumbent remains the locked-OOS winner, but its train performance is poor (train total return roughly **-10.18%**) despite positive validation/OOS (`var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_opt/portfolio_optimization_latest.json`).
- Anchored four-sleeve, full strategy-retune, and validation-only full-retune challengers all failed to beat the incumbent; several were materially worse on OOS.
- The focused full-retune suite evaluated **51** candidates under the explicit **8 GiB** budget with peak RSS about **2.99 GiB**, so memory-safe focused experimentation is feasible.
- The repo already has broad strategy coverage that is not yet fully exploited in the portfolio loop: carry/crowding, lead-lag spillover, alpha101, volatility compression reversion, mean reversion, rare-event, micro-range expansion, pair-spread, lag convergence.
- Existing robustness signals and anti-overfit surfaces already exist in code/artifacts: DSR/PBO/SPA-like fields, validation-only selection, exact-window split contract, incumbent-aware freeze mode, explicit train+val-only RollingBreakout gate, and a sequential portfolio search wrapper.
- External credible research directions already identified:
  - Bailey & López de Prado — Deflated Sharpe Ratio
  - Bailey et al. — Probability of Backtest Overfitting
  - López de Prado — Hierarchical Risk Parity
  - Moreira & Muir — Volatility Managed Portfolios
  - Han, Kang, Ryu — stronger crypto time-series momentum than cross-sectional momentum under realistic assumptions
  - Bhambhwani, Korniotis, Delikouras — blockchain/network characteristics explain part of crypto return cross-section
  - Gornall, Rinaldi, Xiao — perpetual futures basis/funding structure matters
  - Tan, Roberts, Zohren — spatio-temporal momentum combining time-series and cross-sectional signals.

## Outcome
Build an **autonomous experiment index/orchestrator** around the current repo that reuses the existing exact-window registry, heavy-run locks, and promotion surfaces rather than introducing a second scheduler/queue. It should:
1. audits and explains the current train instability;
2. upgrades the research methodology where needed;
3. runs a bounded stream of focused experiments on strategies, indicators, alpha sources, and portfolio constructors;
4. logs every experiment outcome (`keep` / `discard` / `crash`) in a durable ledger;
5. only promotes portfolio changes when they beat the incumbent under leakage-safe validation + locked-OOS decision rules;
6. periodically lands verified branch progress and pushes to private git when a milestone is genuinely better and operationally clean.

## Constraints
- Hard total-memory limit: **< 8 GiB** across all active sessions/processes.
- One heavy lane maximum at a time.
- No user confirmation pauses after the loop starts; continue until manually interrupted.
- Use credible English sources; prefer primary sources for research claims.
- Keep changes reviewable; delete or revert dead-end code when experiments are discarded.

## RALPLAN-DR Summary

### Principles
1. **Robustness beats headline OOS.** A candidate with catastrophic train collapse or obvious selection fragility is not production-ready even if one OOS slice looks good.
2. **One heavy experiment at a time.** Memory safety is a hard constraint, not a guideline.
3. **Evidence ledger first.** Every experiment must leave behind a durable row saying what changed, what ran, and whether it was kept, discarded, or crashed.
4. **Prefer reuse before invention.** Mine the existing repo’s unused strategies/indicators and robustness hooks before adding new complexity.
5. **Promote only by verified superiority.** The incumbent stays until a challenger clearly wins on the final locked-OOS rule.

### Decision Drivers
1. Fix the mismatch between train instability and acceptable production confidence.
2. Increase the rate of useful experiments without breaking the 8 GiB ceiling.
3. Expand the idea set beyond the current 3–4 sleeves while preserving evidence quality.

### Viable Options
#### Option A — Build an autonomous audit + experiment loop on top of the existing exact-window / portfolio stack (**Recommended**)
- Audit current code and artifacts.
- Add an experiment ledger and crash handling.
- Use the current exact-window and portfolio surfaces to test existing and new ideas sequentially.
- Pros: fastest path to continuous learning with bounded risk.
- Cons: requires discipline to prevent experiment sprawl.

#### Option B — Keep tuning the same small sleeve set harder
- Pros: minimal new surface area.
- Cons: already diminishing returns; recent full-retune runs worsened OOS materially.

#### Option C — Big-bang architecture rewrite before more experiments
- Pros: could unify everything eventually.
- Cons: too slow, too risky, and not needed before learning whether simpler research improvements work.

### Recommendation
Choose **Option A**.

## ADR
- **Decision:** implement an autonomous research loop anchored on the existing exact-window / freeze / portfolio / decision infrastructure, with a durable experiment ledger, a train-instability audit lane, research-backed methodology upgrades, and a sequential heavy experiment queue.
- **Drivers:** incumbent instability, prior failed retunes, broad unused strategy inventory, strong need for continuous experimentation under 8 GiB.
- **Alternatives considered:** more sleeve-only tuning; full rewrite first.
- **Why chosen:** it maximizes research velocity while preserving operational safety and evidence.
- **Consequences:** introduces more orchestration/reporting code and experiment hygiene rules, but keeps the core trading stack intact.
- **Follow-ups:** once the loop proves stable, broaden to more radical alphas/constructors and private-git milestone pushes.

## Concrete File Targets
### Existing code to inspect and likely touch
- `src/lumina_quant/strategies/*`
- `src/lumina_quant/indicators/*`
- `src/lumina_quant/strategy_factory/candidate_library.py`
- `src/lumina_quant/strategy_factory/research_runner.py`
- `src/lumina_quant/eval/exact_window_suite.py`
- `src/lumina_quant/cli/exact_window.py`
- `src/lumina_quant/workflows/alpha_research_pipeline.py`
- `src/lumina_quant/cli/optimize.py`
- `scripts/run_research_candidates.py`
- `scripts/run_portfolio_optimization.py`
- `scripts/research/search_portfolio_four_sleeve_anchored.py`
- `scripts/research/run_causal_dynamic_portfolio.py`
- `scripts/research/run_causal_overlay_portfolio.py`
- `scripts/research/write_portfolio_max_performance_decision.py`
- `src/lumina_quant/portfolio_split_contract.py`

### New artifacts / ledgers to add
- `var/reports/exact_window_backtests/followup_status/autonomous_research_loop/experiments.tsv`
- `var/reports/exact_window_backtests/followup_status/autonomous_research_loop/research_state_latest.json`
- `var/reports/exact_window_backtests/exact_window_run_registry.jsonl` remains the source of truth for heavy exact-window runs
- `var/reports/exact_window_backtests/followup_status/autonomous_research_loop/ideas_backlog_latest.md`
- `var/reports/exact_window_backtests/followup_status/autonomous_research_loop/stack_audit_latest.md`

### New focused research outputs
- `var/reports/exact_window_backtests/followup_status/autonomous_research_loop/*`
- plus per-experiment bundle / comparison / decision artifacts under that subtree.

## Execution Plan

### Step 1 — Audit the current stack and explain the instability
Build a truthful audit of:
- current incumbent train/val/oos behavior
- strategy-level train collapse contributors
- current selection logic (validation-only, PBO/DSR/SPA fields, candidate-pool fallbacks)
- portfolio optimizer behavior (cap relaxation, clustering, equal-weight convergence)
- backtest execution realism (data alignment, execution timing, stop-loss mechanics, cost assumptions)

Acceptance criteria:
- audit artifact identifies concrete instability drivers with file-backed references and artifact evidence
- audit ranks issues by expected impact on portfolio robustness

### Step 2 — Add a thin experiment index over the existing artifact/lock model
Implement a durable experiment ledger with at least these columns, but keep it as an **index over existing artifacts/registries** rather than a second scheduler or promotion path:
- timestamp
- experiment_id
- hypothesis
- changed files / artifact inputs
- method category (`strategy`, `indicator`, `alpha`, `portfolio`, `backtest`, `validation`)
- status (`keep`, `discard`, `crash`)
- train/val/oos key metrics
- memory evidence path
- notes / crash reason

Crash policy:
- trivial implementation bug → fix and rerun
- fundamentally broken idea → mark `crash` or `discard`, summarize why, and move on

Acceptance criteria:
- every experiment writes a ledger row
- heavy research runs still use the existing exact-window registry / lock model
- portfolio heavy lanes still use the portfolio follow-up heavy lock as the only heavy portfolio gate
- crash handling is deterministic and documented

### Step 3 — Research-backed methodology upgrades
First normalize the memory contract across the current stack, then evaluate targeted upgrades such as:
- explicit 8 GiB budget injection for dynamic/overlay/optimizer lanes (`run_causal_dynamic_portfolio.py`, `run_causal_overlay_portfolio.py`, portfolio search wrappers / optimizer entrypoints), with executable verification that these lanes pass `8 * 1024**3` into `acquire_portfolio_memory_guard` and emit that same explicit budget in their memory artifacts
- stronger train-stability penalties / minimum-train gates
- walk-forward or CPCV-like validation for candidate ranking (bounded, memory-safe approximation if full CPCV is too expensive)
- HRP / risk-parity / volatility-managed portfolio alternatives
- improved concentration / family-correlation controls
- better decision rules for keeping validation winners that are train-fragile

Acceptance criteria:
- each methodology change is isolated as an experiment
- memory-budget propagation is explicit everywhere, not ambient/system-derived, and dynamic/overlay lanes have concrete verification hooks proving the emitted `memory_policy.explicit_budget_bytes` is exactly `8 * 1024**3`
- no change reaches production unless it improves the final decision path

### Step 4 — Expand the candidate universe using existing and new ideas
Priority order:
1. existing unused in-repo strategies/alphas/indicators
   - `perp_crowding_carry`
   - `leadlag_spillover`
   - `alpha101_formula`
   - `candidate_vol_compression_reversion`
   - `rare_event_score`
   - `micro_range_expansion_1s` (only if memory profile stays safe)
   - better pair/basis/carry configurations where coverage allows
2. research-backed additions or adaptations from external sources
   - volatility-managed trend / sleeves
   - residual / cross-sectional / spatio-temporal momentum variants
   - blockchain/network or perpetual-basis features if data feasibility exists
3. combined/ensemble constructions of previous near-misses when justified

Acceptance criteria:
- candidate additions are focused and memory-bounded
- every addition is evaluated and logged, not just added speculatively

### Step 5 — Continuous portfolio construction and promotion loop
For every promising experiment batch:
1. build or refresh sleeve bundle
2. run portfolio search
3. compare against incumbent
4. keep or discard
5. if clearly better and verified, checkpoint branch progress and prepare private-git push

Acceptance criteria:
- no challenger is promoted without passing the final locked-OOS decision rule
- discarded branches leave a clear artifact trail and do not linger as production candidates

### Step 6 — Team execution and Ralph verification
- Team handles light parallel lanes: code audit, source-backed research distillation, candidate manifests, report generation, tests, documentation
- Leader keeps all heavy backtests/searches sequential
- Ralph owns persistence, bug fixing, verification, cleanup, and final milestone push gating

Acceptance criteria:
- team never runs overlapping heavy evaluators
- Ralph closes each milestone with fresh verification and architect sign-off

## Deliberate Pre-Mortem
1. **Experiment sprawl without learning**
   - Failure mode: dozens of runs, no durable insight
   - Mitigation: mandatory TSV ledger + explicit keep/discard/crash rows
2. **Robustness theater**
   - Failure mode: more metrics, same brittle selection behavior
   - Mitigation: treat train collapse and cap relaxation as first-class blockers, not side notes
3. **Memory regressions from “more ideas”**
   - Failure mode: candidate universe expansion breaks the 8 GiB ceiling, or some lanes silently fall back to ambient system memory instead of explicit 8 GiB injection
   - Mitigation: one heavy lane, bounded candidate batches, explicit budget evidence for every heavy run, and code-level budget plumbing across dynamic/overlay/optimizer lanes

## Expanded Test / Verification Plan
### Unit
- experiment ledger writer / parser
- crash logging semantics
- any new ranking/objective helpers
- portfolio-cap and memory-budget propagation

### Integration
- exact-window focused run on bounded candidate batches
- bundle -> portfolio search -> comparison -> decision path
- ledger row emitted for success / discard / crash

### End-to-End
- run at least one full autonomous experiment cycle from audit -> experiment -> decision -> ledger entry
- verify team terminal cleanup and Ralph completion evidence

### Observability
- memory evidence for every heavy run
- artifact paths written into the ledger
- milestone summary written after each batch

## Available Agent Types / Staffing Guidance
- **Roster:** `executor`, `verifier`, `architect`, `debugger`, `test-engineer`, `researcher`, `explore`, `writer`
- **Recommended team headcount:** 3 `executor` workers
- **Lane allocation:**
  1. stack-audit / robustness lane
  2. strategy-alpha expansion lane
  3. portfolio/report/ledger lane
- **Suggested reasoning:** lane 1 `high`, lane 2 `high`, lane 3 `medium`

## Launch Hints
- Team launch hint:
  - `omx team ralph 3:executor "Implement the autonomous portfolio research loop in LuminaQuant: audit instability, add experiment ledger/crash logging, research and test robust strategies/alphas/portfolio methods under 8 GiB, keep winners, discard losers, and prepare verified private-git milestones"`

## Concrete Team -> Ralph Verification Path
1. Team lands audit/ledger/research/code/test surfaces.
2. Leader runs heavy experiment batches sequentially and records outcomes.
3. Ralph fixes trivial failures, reruns verification, updates the ledger, and rejects non-working ideas.
4. Architect sign-off is required before any milestone is treated as complete or pushed privately.
