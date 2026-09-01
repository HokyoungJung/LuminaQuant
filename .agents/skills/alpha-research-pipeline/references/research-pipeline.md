# LuminaQuant Alpha Research Pipeline

Use this as the detailed operating reference after `alpha-research-pipeline/SKILL.md` triggers.

## Phase 0 — Resource, Repo, and Evidence Preflight

1. Confirm repo root: `/home/hoky/Quants-agent/LuminaQuant`.
2. Record git SHA and dirty status in the run manifest.
3. Check compute resources before heavy sweeps. Use the upstream `get-available-resources` pattern if installed; otherwise record CPU/RAM/GPU/disk with local shell commands.
4. Inspect existing exact-window and alpha-discovery artifacts before rerunning expensive jobs:
   - `var/reports/exact_window_backtests/exact_window_run_registry.jsonl`
   - `var/reports/exact_window_backtests/exact_window_backtest_registry_latest.json`
   - `var/reports/exact_window_backtests/followup_status/backtest_log_archive_latest.json`
   - `var/reports/alpha_discovery/`
5. Validate relevant config profiles before execution:
   - `uv run lq config validate`
   - `LQ_CONFIG_PATH=configs/profiles/backtest_cost_realistic.yaml uv run lq config validate` if that profile exists.
6. Keep real-money flags false. Research never promotes to real execution.

## Phase 1 — Scientific Skill Ingestion

Read `references/scientific-agent-skills-audit.md` to choose methods:

- Tier A skills are default methods to use in every serious alpha research loop.
- Tier B skills are conditional tools for specific lanes.
- Tier C skills support ingestion, provenance, reporting, or infrastructure.
- Tier D skills were reviewed and excluded by default.

Do not install/import libraries solely because a skill exists. Treat skills as agent procedures and references. Add runtime dependencies only through normal project review.

## Phase 2 — Candidate Discovery

Create candidate cards before implementation. Use at least four lanes per run unless the user asks for a narrow study:

1. Literature/microstructure/macro: `paper-lookup`, `research-lookup`, `literature-review`, `database-lookup`, `usfiscaldata`.
2. Time-series and econometrics: `aeon`, `statsmodels`, `statistical-analysis`.
3. ML and explainability: `scikit-learn`, `shap`, optionally `pymc`, `timesfm-forecasting`, `umap-learn`.
4. Graph/cross-asset: `networkx`, optionally `torch-geometric`.
5. Execution/cost: `simpy`, `pymoo`, cost stress.
6. Meta-research control: `arbor`, `hypothesis-generation`, `scientific-critical-thinking`, `experimental-design`, `peer-review`.

Each candidate must include:

- falsifiable hypothesis and mechanism
- source skill inspiration
- allowed selection data and forbidden data
- features and parameters
- pre-registered metrics
- disconfirming evidence
- fail-closed conditions
- decision placeholder

## Phase 3 — Pre-Registration and Design

Write `experiment_design.json` before reading any locked/current OOS result.

Required design fields:

- train window
- validation window
- embargo/gap
- lagged completed shadow or fresh-forward reporting window
- locked/current OOS as report-only after selection freeze
- candidate family budget
- benchmark/incumbent
- metrics and cost stress
- rejection thresholds
- missing-data behavior
- run signature

Forbidden after seeing locked/current OOS:

- threshold selection
- sleeve selection
- candidate tie-breaks
- optimizer family or objective changes
- leverage/sizing/weight changes
- data source substitution that changes availability

## Phase 4 — Data Contract

Fail closed if any are unresolved:

- source artifact missing or SHA mismatch
- market data stale for intended window
- symbol coverage differs from declared universe
- exchange status/instrument availability mismatch
- silent forward fill without no-trade materialization
- derived data mislabeled as exact replay
- macro/text/alternative data lacking publication-time provenance
- live-only feature unavailable in backtest or vice versa

## Phase 5 — Implementation Discipline

- Implement small candidates; avoid broad rewrites.
- Prefer existing strategy factory, alpha_zoo, research scripts, and config surfaces.
- Add focused tests for each new feature/candidate.
- Emit live-safe metadata only for paper/shadow candidates:
  - `target_allocation`
  - `max_symbol_exposure_pct`
  - `max_order_value`
- Keep defaults backward-compatible.
- Never flip real-money flags.

## Phase 6 — Evaluation

At minimum collect:

- return at 10/15/20bps
- MDD
- Sharpe, Sortino, Calmar if available
- turnover
- gross exposure
- trade count / win rate / exposure
- benchmark/incumbent delta
- rolling stability
- cost sensitivity
- data blockers
- exact commands and artifacts

Prefer saved artifacts over recomputation for follow-ups.

## Phase 7 — Statistical and Scientific Audit

Use `statistical-analysis` and `scientific-critical-thinking` patterns:

- compare against cash/null/incumbent/negative controls
- account for multiple testing
- emphasize effect size and stability, not p-value alone
- inspect robustness across costs, windows, symbols, and regimes
- run leakage and timestamp audit
- inspect feature importance/SHAP for ML candidates
- reject one-off, tiny-sample, or missing-data wins

## Phase 8 — Optimization

Optimization is allowed only after initial evidence and only on train/validation or lagged completed shadow.

Use Pareto thinking (`pymoo` pattern):

- maximize net return/risk-adjusted return
- minimize MDD and turnover
- cap gross exposure
- maximize stability
- penalize data dependence and implementation fragility

Do not use locked/current OOS to pick a Pareto point.

## Phase 9 — Review and Decision

Required decision files for a completed run:

- `scoreboard.json`
- `scoreboard.md` or equivalent report
- `decision.json`
- updated `quality_gate_receipt.json`

Promotion requires all gates true:

- beats incumbent at 10bps and 20bps on same refreshed run
- 20bps return remains positive
- MDD not materially worse and under cap
- turnover/gross under caps
- no data/provenance/leakage blockers
- focused tests pass
- review lanes clean
- real-money flags false

Otherwise emit `no-promotion` or `shadow-watch` with exact blockers.
