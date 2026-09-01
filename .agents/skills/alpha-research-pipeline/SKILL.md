---
name: alpha-research-pipeline
description: LuminaQuant-specific agent skill for finding, validating, analyzing, researching, and reporting trading alpha. Use when the user asks to discover alpha, improve alpha_zoo, run autonomous alpha research, absorb scientific-agent-skills into quant research, create candidate cards, design walk-forward experiments, audit leakage/overfit, evaluate cost-stressed backtests, or produce promotion/no-promotion decisions. It consolidates the relevant K-Dense scientific-agent-skills methods into a safe local alpha pipeline with pre-registration, candidate registry, statistical validation, data-provenance gates, and real-money-disabled discipline.
---

# Alpha Research Pipeline

Operate from repository root:

```bash
/home/hoky/Quants-agent/LuminaQuant
```

This skill is the local LuminaQuant absorption layer for `K-Dense-AI/scientific-agent-skills`. It was built from a full 147-skill review of upstream commit `9c9bd2e92af12311ecd0c1a643e0931643f9ea04`.

## Load Order

1. Read this `SKILL.md`.
2. Read `references/scientific-agent-skills-audit.md` when choosing which scientific skills matter; it contains the complete 147-skill audit and tiering.
3. Read `references/research-pipeline.md` for the full phase-by-phase workflow.
4. Read `references/candidate-families.md` when generating candidate ideas.
5. Use the bundled scripts instead of recreating boilerplate run artifacts.

## Non-Negotiable Rules

- Keep `ready_for_real=false`, `allow_real_money=false`, and `real_money_execution=false` in all research artifacts.
- Never use locked/current OOS to fit, rank, select thresholds, choose sleeves, choose weights, tune optimizers, or break ties.
- Fail closed on stale/missing data, source SHA mismatch, exchange-status mismatch, data gap, live/backtest feature mismatch, OOS contamination, cost-stress failure, or real-money flags.
- Promotion is never the default. If any gate is uncertain, emit `no-promotion` or `shadow-watch` with exact blockers.
- Every conclusion must cite files, commands, hashes, metrics, and test output.

## Absorbed Skill Tiers

Use the upstream skills as procedures and references, not as automatic library dependencies.

- **Tier A — default alpha pipeline:** `arbor`, `hypothesis-generation`, `scientific-brainstorming`, `scientific-critical-thinking`, `experimental-design`, `statistical-analysis`, `hypogenic`, `paper-lookup`, `research-lookup`, `literature-review`, `database-lookup`, `usfiscaldata`, `aeon`, `statsmodels`, `scikit-learn`, `shap`, `pymoo`, `polars`, `get-available-resources`, `exploratory-data-analysis`, `scientific-visualization`, `peer-review`.
- **Tier B — conditional alpha methods:** `bgpt-paper-search`, `parallel-web`, `exa-search`, `pymc`, `timesfm-forecasting`, `umap-learn`, `scikit-survival`, `networkx`, `torch-geometric`, `pytorch-lightning`, `transformers`, `stable-baselines3`, `pufferlib`, `simpy`, `dask`, `vaex`, `zarr-python`, `optimize-for-gpu`, `geomaster`, `geopandas`, `statistical-power`, `scholar-evaluation`, `what-if-oracle`, `consciousness-council`.
- **Tier C — support/reporting/provenance:** `citation-management`, `pyzotero`, `paperzilla`, `open-notebook`, `xlsx`, `pdf`, `docx`, `pptx`, `markitdown`, `liteparse`, `lamindb`, `autoskill`, `matplotlib`, `seaborn`, `markdown-mermaid-writing`, `scientific-writing`, `scientific-slides`, `scientific-schematics`, `infographics`, `venue-templates`, `market-research-reports`, `modal`, `sympy`, `matlab`, `pi-agent`, poster skills.
- **Tier D — reviewed and excluded by default:** biomedical, chemistry, lab automation, clinical, quantum, astronomy, and most materials skills. Use them only for an explicit alternative-data hypothesis and record why.

The complete list, counts, and rationale live in `references/scientific-agent-skills-audit.md` and `.json`.

## Direct Application

When asked to find alpha now, initialize a run first:

```bash
cd /home/hoky/Quants-agent/LuminaQuant
python3 .agents/skills/alpha-research-pipeline/scripts/init_alpha_research_run.py
```

This creates:

- `var/reports/alpha_discovery/<run_id>/run_manifest.json`
- `candidate_registry.json`
- `experiment_design.json`
- `quality_gate_receipt.json`
- `scientific_skill_audit_snapshot.json`
- `run_plan.md`

Validate the run skeleton:

```bash
python3 .agents/skills/alpha-research-pipeline/scripts/validate_alpha_research_run.py \
  --run-dir var/reports/alpha_discovery/<run_id>
```

Regenerate the upstream skill audit after updating `/tmp/scientific-agent-skills`:

```bash
python3 .agents/skills/alpha-research-pipeline/scripts/audit_scientific_skills.py \
  --skills-dir /tmp/scientific-agent-skills/skills \
  --json-out .agents/skills/alpha-research-pipeline/references/scientific-agent-skills-audit.json \
  --md-out .agents/skills/alpha-research-pipeline/references/scientific-agent-skills-audit.md
```

## Required Pipeline

1. **Preflight:** repo SHA, dirty status, resources, existing run registries, config validation, data freshness.
2. **Candidate discovery:** build 20–50 candidate cards across literature, time-series, econometrics, ML, graph, macro, microstructure, execution, and portfolio/risk lanes.
3. **Pre-registration:** write experiment windows, embargo, benchmarks, metrics, cost stress, rejection thresholds, and forbidden data before results.
4. **Data contract:** verify symbols, timestamps, source artifacts, SHA, exchange status, freshness, and live/backtest feature parity.
5. **Implementation:** small bounded changes only; focused tests first; defaults backward-compatible.
6. **Evaluation:** cost-stressed backtests at 10/15/20bps, incumbent comparison, turnover/gross/MDD/stability metrics.
7. **Statistical audit:** multiple testing, effect sizes, robustness, negative controls, leakage review, SHAP/feature audit for ML.
8. **Optimization:** only after evidence, only train/validation or lagged completed shadow; use Pareto thinking.
9. **Review:** critical-thinking review, peer-review methodology pass, quality gate receipt.
10. **Decision:** `promotion`, `shadow-watch`, or `no-promotion`; default `no-promotion` if any uncertainty remains.

## Existing LuminaQuant Seed Command

The existing LLM alpha manifest generator remains a seed source, not promotion evidence:

```bash
uv run python scripts/research/run_llm_alpha_pipeline.py
```

Treat outputs as candidate inputs only:

- `var/reports/exact_window_backtests/pipeline/alpha_research_pipeline_latest.json`
- `var/reports/exact_window_backtests/pipeline/alpha_research_pipeline_latest.md`

## Completion Contract

A completed alpha research pass must leave a run directory containing candidate registry, experiment design, scoreboard, decision, quality gate receipt, and cited verification output. If those are not present, continue the pipeline or report the exact blocker.
