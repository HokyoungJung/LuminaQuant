# PRD: Profit Moonshot Dynamic Restart and Research-History Ledger

## Objective
Rebuild profit-moonshot live-candidate research from dynamic, state-based strategies only, and preserve a durable research-history/source-ledger artifact so future sessions do not repeat prior searches or reintroduce invalid calendar-primary rules.

## User problem
Previous high-return candidates were later rejected because their apparent edge came from fixed month/asset calendar-primary rules rather than a live-defensible dynamic trading signal. The user wants a rigorous restart that can support live deployment, and a research note that records every strategy, research date, consulted source/history, what was learned, what was used, metrics, pros/cons, and rejection/promotion decisions.

## Scope
### In scope
1. Create/update a detailed research-history note and machine-readable JSON ledger.
2. Include research date, strategy family, hypothesis, rule properties, metrics, advantages/disadvantages, final status, and rejection/promotion reason for all prior and new profit-moonshot strategy research that can be reconstructed from local artifacts.
3. Include a manifest-driven source/search ledger covering every reconstructable local handoff/plan/report from 2026-05-01 through 2026-05-10 plus external references previously consulted or newly consulted; each entry must summarize content and identify what was actually used, while unavailable gaps must be marked `not_reconstructable` with reason.
4. Add tests-first guardrails so calendar-primary, non-integer-leverage, liquidation-unsafe, locked-OOS-leaking, or source-untraceable candidates cannot be promoted.
5. Rebuild/tune/validate dynamic candidate families and dynamic candidate-derived portfolio/hybrid options under conservative live constraints.
6. Produce final live recommendation: best valid candidate/multi-portfolio/hybrid or explicit `no_live_promotion`.
7. Commit/push with Lore commit and verify GitHub Actions `ci` and `private-ci` green.

### Out of scope
- Promoting calendar-primary month/window/asset rules without a separate robust seasonality research program.
- Adding new dependencies.
- Using locked-OOS for selection, threshold tuning, source pruning, or weight optimization.
- Live trading changes or credential-dependent external production actions.

## Functional requirements
1. **Research-history artifact**
   - Write `docs/research_note/research_history.md`.
   - Write `var/reports/profit_moonshot_20260501/research_history/profit_moonshot_research_history_latest.md` and `.json`.
   - Add `.omx/notepad.md` pointer stating future profit-moonshot sessions must read the research history first.
   - The Markdown must be human-readable and grouped by chronology, strategy family, source/search ledger, invalidity lessons, and future-use instructions.
   - The JSON must include `strategy_chronology`, `source_history_inventory`, `source_search_ledger`, `decision_log`, `invalidity_lessons`, `future_session_instructions`, and `generation_metadata`.

2. **Strategy chronology fields**
   Every reconstructed or newly researched strategy family/candidate group must include:
   - `research_date`
   - `strategy_family`
   - `artifact_paths`
   - `hypothesis`
   - `primary_signal_type`
   - `state_variables_or_features`
   - `universe`
   - `timeframe`
   - `split_periods` when known
   - `implementation_files`
   - train/validation/OOS metrics when available
   - leverage and integer-leverage status
   - liquidation count / margin buffer / margin ratio status when available
   - source-ledger references used
   - advantages
   - disadvantages/risks
   - final decision and rejection/promotion reason

3. **Source/search ledger fields**
   Every local history or external reference entry must include:
   - `research_date`
   - `source_type`
   - `query_or_title`
   - `normalized_key`
   - `path_or_url`
   - `content_summary`
   - `what_was_used`
   - `associated_strategy_families`
   - `decision_impact`
   - `staleness_policy`
   - `recheck_before_use`
   - `do_not_repeat_note`

4. **Source/history inventory and duplicate-search guard**
   - Build or consume a `source_history_inventory` manifest for all discoverable profit-moonshot local artifacts and consulted source mentions from 2026-05-01 through 2026-05-10.
   - Every manifest item must map to a `source_search_ledger` entry or an explicit `not_reconstructable` item with reason.
   - Normalize source keys by URL/domain/title/path and search intent.
   - The note must explicitly say which searches should not be repeated unless data coverage, date, exchange API docs, or strategy family changes.
   - Repeated-search clusters must be listed for funding/OI/taker-flow/liquidation docs, Hyperliquid/Tickmill external-venue exploration, and crypto momentum/reversal/risk-factor literature.
   - Repeated mentions sharing a normalized key must collapse to one ledger entry with all associated dates/families retained.

5. **Candidate validity and live-gate contracts**
   - Calendar-primary rules are invalid for live promotion by default.
   - Non-integer leverage is benchmark-only and cannot be promoted.
   - Candidate-derived hybrids inherit the strictest validity/leverage/liquidation status of their source rows.
   - Missing research-history/source-ledger link metadata blocks live promotion.
   - Missing strategy-validity, split metrics, leverage, liquidation, or memory evidence blocks live promotion.

6. **Dynamic restart research**
   - Re-evaluate existing strategy-valid dynamic rows.
   - Add or replay bounded first-principles dynamic families: regime momentum, cross-sectional rank, funding/OI stress, volatility compression, liquidity/taker shock, residual/pair reversion, lead-lag/spillover, and dynamic allocator/hybrid.
   - Select only using train+validation.
   - Report locked-OOS only after selection for gate/report.

7. **Metrics and reporting**
   - For every final candidate and material research family, report train/validation/OOS return, monthlyized return, MDD, return/MDD, Sharpe, Sortino, smart Sortino, Calmar, leverage, liquidation count, minimum margin buffer, minimum margin ratio, memory/RSS, and status.
   - Explain what each metric measures in the final report.

## Non-functional requirements
- Memory: all commands and concurrent lanes must stay below 8 GiB total RSS; heavy backtests run sequentially unless proven safe.
- Reproducibility: report exact commands, input artifact paths, output artifact paths, commit SHAs, and data cutoffs.
- Safety: fail closed when evidence is missing.
- Style: no new dependencies; reuse existing scripts/utilities and patterns.

## Acceptance criteria
- PRD and test spec exist before implementation.
- Targeted tests prove research-history/source-ledger schema completeness and promotion-blocking behavior.
- Calendar-primary and non-integer leverage candidates cannot be promoted.
- Locked-OOS cannot affect selection.
- Research-history Markdown/JSON list prior research dates and consulted references with content summary and usage summary, and pass a no-orphan manifest coverage check for reconstructable 2026-05-01..2026-05-10 history/source items.
- Full validation either recommends a live candidate that passes all gates or records `no_live_promotion` with evidence.
- Verification passes: targeted pytest, full pytest, ruff, compileall, git diff --check, Lore commit/push, GitHub Actions `ci` and `private-ci` green.
