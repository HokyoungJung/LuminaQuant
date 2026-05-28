# RALPLAN-DR: Profit Moonshot Dynamic Restart + Durable Research History

## Status
Draft for Architect/Critic review. This is a planning artifact only; implementation begins only after the PRD and test spec below are approved:

- `.omx/plans/prd-profit-moonshot-dynamic-restart-20260510.md`
- `.omx/plans/test-spec-profit-moonshot-dynamic-restart-20260510.md`

## User objective
Restart profit-moonshot strategy research from first principles after the calendar-primary invalidation. Rebuild and validate a strategy universe that is live-deployable, dynamic/state-based, integer-leverage-only, liquidation-aware, train/validation-selected, locked-OOS report/gate-only, and memory-safe under 8 GiB. Produce a durable research note/history that future sessions must read first so the same flawed calendar rule or duplicate source-search loops do not recur.

## Non-negotiable constraints
1. **No calendar-primary alpha**: fixed or dynamic month/window/asset calendar rules cannot be the primary signal unless separately proven by robust external/mechanistic evidence and out-of-family robustness tests. Current restart assumes no calendar-primary promotion.
2. **Dynamic/state-based primary signal only**: allowed primary families include trend/momentum, breakout/volatility, cross-sectional rank, residual/pair spread, funding/OI/carry/fade, taker/liquidity shock, lead-lag/spillover, and allocator/regime state. Time/session filters may only be secondary risk filters.
3. **Selection integrity**: train+validation may choose candidates; locked-OOS is report-only/gate-only and must not shape thresholds, weights, source pruning, or final rank before the final audit.
4. **Live feasibility**: live candidates require integer leverage, Binance USDⓈ-M-style conservative liquidation replay, funding/fee/slippage/stress buffer, no account-wipeout behavior, and explicit margin-buffer/margin-ratio evidence.
5. **Promotion gates**: train/validation/OOS liquidation counts inside the configured tolerance, all split minimum margin buffer >0, OOS MDD <=25%, OOS return and return/MDD better than current base, and Sharpe/Sortino/smart Sortino/Calmar not degraded.
6. **Memory cap**: every replay/tuning/full validation command must record RSS evidence and stay below 8 GiB across concurrent lanes; heavy backtests run sequentially unless proven safe.
7. **Durable research memory**: the final note must include research dates, strategy properties, metrics, decisions, rejection reasons, and a source/search ledger explaining what was consulted and what was used.

## RALPLAN-DR summary

### Principles
- **Validity before return**: a high-return rule is rejected if its thesis is not live-defensible.
- **Evidence-separated selection**: locked-OOS may veto/report, not optimize.
- **Execution realism**: integer leverage + liquidation/funding/fees/slippage/stress are part of candidate identity, not post-hoc decoration.
- **Research memory compounds**: every strategy and source consultation must become reusable indexed knowledge to avoid repeated searches and forgotten rejection causes.
- **Fail-closed reporting**: missing strategy-validity, leverage, liquidation, split, source, or memory evidence blocks live promotion.

### Decision drivers
1. Avoid another calendar/seasonality pseudo-alpha escaping due to attractive OOS metrics.
2. Find any dynamic candidate or dynamic multi-portfolio/hybrid that survives rigorous train/validation/OOS gates.
3. Make future sessions faster and safer by preserving a structured research ledger with source summaries and “what was actually used.”

### Viable options
| Option | Description | Pros | Cons | Decision |
| --- | --- | --- | --- | --- |
| A. Existing dynamic rescue | Re-rank only already-produced non-calendar rows. | Fastest; minimal new surface. | Prior audit found zero valid successes; may miss dynamic alpha. | Not sufficient alone; use as baseline input. |
| B. First-principles dynamic restart | Build/replay new and existing state-based families, then portfolio/hybrid construction and full liquidation-aware selection. | Best chance of finding valid live candidate; directly addresses calendar flaw. | More compute/time; needs strict memory sequencing. | **Chosen main path.** |
| C. External-alpha expansion first | Start with new external venues/features before internal dynamic restart. | Potentially novel signal. | More data/coverage risk; easy to overfit or repeat searches. | Bounded support lane only; no promotion without same gates. |
| D. No-deploy research-only closeout | Stop after documenting failure modes. | Safest if no candidate passes. | Does not pursue upside. | Valid terminal state if B/C fail gates. |

### Pre-mortem (high-risk deliberate mode)
1. **Calendar leakage returns under a new name**: mitigated by tests for primary-signal classification, per-sleeve validity metadata, and fail-closed final selection.
2. **Locked-OOS contaminates tuning**: mitigated by code/tests that assert OOS fields are absent from candidate ranking/weight optimization and only used in final gate/report.
3. **Research history is too narrative to prevent repeat work**: mitigated by a structured source/search ledger with normalized keys, date, path/URL, content summary, used evidence, linked strategy families, decision impact, and recheck policy.

## Execution plan

### Phase 0 — Grounding and state protection
- Read and preserve current green head and baseline evidence.
- Confirm latest data cutoff and split ranges.
- Use existing context snapshot: `.omx/context/profit-moonshot-restart-dynamic-research-history-20260510T105501Z.md`.
- Do not run implementation/backtests until current PRD + test spec exist.

### Phase 1 — Research-history/source-ledger scaffolding (tests first)
Deliverables:
- `docs/research_note/research_history.md`
- `var/reports/profit_moonshot_20260501/research_history/profit_moonshot_research_history_latest.md`
- `var/reports/profit_moonshot_20260501/research_history/profit_moonshot_research_history_latest.json`
- `.omx/notepad.md` pointer requiring future sessions to read the history first.

Required contents:
- **Strategy chronology**: one row/section per research date and family, including artifact paths, hypothesis, primary signal type, state variables, universe/timeframe, implementation files, train/validation/OOS metrics, leverage/liquidation status, pros/cons, decision, and rejection/promotion reason.
- **Source/history inventory manifest**: first enumerate every reconstructable local profit-moonshot history/source artifact from 2026-05-01 through 2026-05-10 plus every newly consulted source. Each manifest item must either be represented in `source_search_ledger` or explicitly marked `not_reconstructable` with reason.
- **Source/search ledger**: one row/section per consulted source, local history artifact, or repeated-search cluster. Fields: `research_date`, `source_type` (`local_artifact`, `docs`, `official_api_doc`, `paper`, `web_search`, `agent_log`), `query_or_title`, `normalized_key`, `path_or_url`, `content_summary`, `what_was_used`, `associated_strategy_families`, `decision_impact`, `staleness_policy`, `recheck_before_use`, and `do_not_repeat_note`.
- **Reference clusters already known**: Binance funding/OI/taker-flow/liquidation docs; Hyperliquid metadata/candle/funding/fee docs; Tickmill instruments/spreads/swaps; crypto momentum/reversal papers; common crypto risk-factor paper; prior local handoffs/plans/reports from 2026-05-01 through 2026-05-10.
- **Duplicate-search guard**: include normalized URL/domain/title/path keys and search-intent tags so future agents can check the ledger before repeating similar searches. Repeated mentions of the same source/search intent must collapse to a stable normalized key while retaining all associated research dates/strategy families.
- **Invalidity lessons**: calendar-primary month/asset rules are not a live thesis; non-integer leverage is benchmark-only; candidate-derived hybrids inherit source-candidate validity/leverage/liquidation defects.

### Phase 2 — Guardrail tests and fail-closed contracts
Add/extend tests before code changes for:
- Strategy validity classifier rejects calendar-primary/month+asset/seasonality rules regardless of attractive metrics.
- Dynamic state-based primary signals are not overblocked merely for using secondary time/session risk filters.
- Research-history generator emits every required source-ledger field and strategy chronology field.
- Final selection cannot promote candidates missing research-history/source-link metadata, validity status, integer leverage, liquidation evidence, split metrics, or memory evidence.
- Hybrid builder discards non-integer, invalid-source, or liquidation-unsafe source candidates before constructing any live hybrid.
- Locked-OOS cannot be referenced by ranking/tuning functions.

### Phase 3 — Dynamic candidate generation/replay
Allowed candidate families to implement/replay under existing code style:
- Regime-gated time-series momentum.
- Cross-sectional rank rotation across BTC/ETH/SOL/BNB/TRX.
- Funding/OI stress fade or carry momentum where coverage exists.
- Volatility compression breakout/continuation.
- Liquidity/taker-flow exhaustion or shock reversion where data support exists.
- Pair/residual reversion and residual momentum.
- Lead-lag/spillover between large-cap and higher-beta assets.
- Portfolio-state allocator that combines only valid dynamic candidates.

Rules:
- No new dependencies.
- Use train/validation-only tuning and selection.
- Keep heavy sweeps bounded and sequential; record `/usr/bin/time -v` Max RSS.
- Every candidate row must carry primary signal type, feature inputs, strategy validity decision, leverage, liquidation replay, split metrics, and source-ledger links.

### Phase 4 — Portfolio/hybrid construction
- Build candidate portfolios and dynamic hybrids only from strategy-valid, integer-leverage, liquidation-safe candidates.
- Compare single candidate, multi-portfolio, and hybrid options under identical fees/slippage/funding/stress and split gates.
- Report all metrics for train, validation, and OOS: return, monthlyized return, MDD, return/MDD, Sharpe, Sortino, smart Sortino, Calmar, trade counts/exposure if available, liquidation count, minimum margin buffer, minimum margin ratio, leverage, and memory.

### Phase 5 — Final selection and handoff
- If at least one candidate passes all gates, recommend the best live candidate or multi-portfolio/hybrid with rationale.
- If none passes, recommend `no_live_promotion` with the best research candidates quarantined as diagnostics.
- Persist final reports under `docs/session_handoff_*`, `.omx/plans`, `.omx/notepad.md`, and `var/reports/profit_moonshot_20260501/...`.
- Lore commit, push to `private/main`, verify GitHub Actions `ci` and `private-ci` green.

## Acceptance criteria
- Current PRD/test spec exist before execution.
- Research note and JSON include complete strategy chronology, a source/history inventory manifest, and source/search ledger for all reconstructable 2026-05-01..2026-05-10 local history plus newly consulted sources, including research dates and consulted-history summaries.
- A future agent can identify which references were already searched, what they contained, what was used, when to recheck them, and which apparent source/history gaps were explicitly classified as `not_reconstructable`.
- No final candidate uses calendar-primary alpha or non-integer leverage for live promotion.
- Locked-OOS remains gate/report-only.
- All promotion gates and memory caps pass, or final decision is explicitly no-deploy.
- Verification evidence includes targeted tests, full pytest, ruff, compileall, git diff check, commit/push, and CI/private-ci.

## Available agent types and staffing guidance
- `planner`: plan revisions and sequencing.
- `architect`: read-only design review, antithesis, tradeoff/synthesis.
- `critic`: actionability and verification gate.
- `executor`: code/tests/artifacts implementation.
- `test-engineer`: targeted/full test adequacy and leakage-guard review.
- `verifier`: final evidence audit and completion checklist.
- `writer`: research-history/handoff clarity.
- `explore`: fast repo-local mapping only.

Recommended `$team` staffing after plan approval:
- Worker 1 (`executor`/writer responsibility): research-history generator/artifacts and source-ledger content; owns `docs/research_note/research_history.md`, `var/reports/.../research_history/*`, and relevant tests.
- Worker 2 (`executor` responsibility): strategy-validity/live-gate hardening for history/source metadata and hybrid source filtering; owns final-selection/hybrid code and tests.
- Worker 3 (`executor` or `test-engineer` responsibility): dynamic-candidate replay/tuning orchestration and memory-safe evidence; owns bounded run commands and metrics artifacts.

Launch hint after approval:
```bash
omx team 3:executor "Execute the approved profit-moonshot dynamic restart + research-history plan under <8GiB, tests-first, with no locked-OOS selection and no calendar-primary alpha."
```

Ralph follow-up:
- Use a separate Ralph loop only after team lanes finish or stall, for single-owner final verification, deslop, commit/push, and CI green confirmation.

## ADR
**Decision**: Restart from dynamic/state-based strategy research while creating a durable research-history/source-ledger artifact that future sessions must read first.

**Drivers**: The previous winning calendar-primary candidate was theoretically invalid; existing valid dynamic rows had zero passing successes; repeated source searches waste time and risk inconsistent interpretation.

**Alternatives considered**: existing-only re-rank, external-first expansion, or no-deploy closeout. Existing-only is too shallow, external-first is too coverage-risky, and no-deploy is only acceptable after a bounded dynamic restart fails.

**Consequences**: More tests/artifacts up front, heavier but bounded backtesting, and a stricter final-selection gate. The live recommendation may still be no promotion.

**Follow-ups**: Keep `docs/research_note/research_history.md` updated on every future profit-moonshot research pass; add `.omx/notepad.md` pointer; treat missing source-ledger links as a promotion blocker.
