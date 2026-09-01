# RALPLAN-DR — Profit moonshot live final selection 2026-05-10

## Outcome
Run the final profit-moonshot live-selection pass after data refresh: tune/revalidate the candidate tuple family, build candidate-derived portfolio and hybrid/multi-portfolio comparison artifacts, select a live-deployable option only if it clears strict safety/performance gates, then commit/push with CI green evidence.

## Context and baselines
- Workdir: `/home/hoky/Quants-agent/LuminaQuant`.
- Starting head: `cb10f1e53f5c8dea9be4b8dcdac0f284c6c244dd` on `private/main`.
- Preserve performance baseline: `02f4520cf906f48089b8852c2651a0f1e4bd0c1c`.
- Prior source commit: `ff1786da0233d2e39acab8c310e0cf3e2a2b0891`.
- Prior preferred candidate: `integer_audit_diagnostic_quarantine_04`, leverage `5x`, zero liquidations, OOS return `+14.6634%`, OOS MDD `1.9646%`, OOS return/MDD `7.4640` on prior data ending `2026-05-06`.
- Binance USDⓈ-M docs checked for the conservative margin model:
  - Exchange trading rules and liquidation fee: https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Exchange-Information
  - Notional/leverage bracket fields including maintenance margin ratio: https://developers.binance.com/docs/derivatives/usds-margined-futures/account/rest-api/Notional-and-Leverage-Brackets

## RALPLAN-DR summary

### Principles
1. Locked-OOS is report-only/gate-only and must never enter ranking, tuning, family expansion, tie-breaks, or final selection scoring.
2. Safety gates precede returns: liquidation tolerance, no account wipeout, positive margin buffer, OOS MDD <= 25%, and risk metrics must pass before promotion.
3. Updated-tail robustness is mandatory: stale `2026-05-06` defaults may not be used after data refresh unless refresh proves no newer complete data exists.
4. Memory safety is a hard execution contract: one heavy job at a time, max workers 1 unless a memory lease proves safe, and each heavy command must record max RSS <8 GiB.
5. Final live decision must compare like-for-like artifacts under an explicit schema and state whether hybrid rows are candidate-derived or benchmark-only.

### Decision drivers
1. Does the prior preferred 5x tuple remain robust after the latest complete tail is included?
2. Can a candidate-derived static portfolio, multi-portfolio blend, or hybrid allocator improve live risk-adjusted performance without leakage?
3. Can the pipeline complete reproducibly under the 8 GiB cap with full local and GitHub CI verification?

### Options
| Option | Use | Pros | Cons | Decision |
|---|---|---|---|---|
| A. Staged single-heavy-lane final selection | Data refresh, retune, liquidation validation, comparison, reporting sequentially | Lowest OOM/leakage risk; easiest audit | Slower wall time | Chosen |
| B. Team-assisted light lanes only | Use team for read-only/report/test planning while leader runs one heavy job | Some throughput gain | Must prevent concurrent heavy jobs | Allowed after PRD/test-spec |
| C. Broad new-alpha expansion | Add new families before final selection | Highest possible upside | Overfit, runtime, leakage risk; not final-pass scope | Rejected/deferred |
| D. Promote prior tuple without refresh | Skip update and accept prior result | Fast | Violates user request and stale-tail risk | Rejected |

## Final execution architecture

### Phase 0 — Planning gate
Create and keep current:
- `.omx/plans/prd-profit-moonshot-live-final-selection-20260510.md`
- `.omx/plans/test-spec-profit-moonshot-live-final-selection-20260510.md`

### Phase 1 — Data refresh and current-tail cutoff
1. Run data refresh with `--max-workers 1` and `/usr/bin/time -v`.
2. Write artifacts under `var/reports/profit_moonshot_20260501/live_final_selection_20260510/data_refresh/`.
3. Derive `latest_complete_oos_end_date` from refresh output:
   - Take required symbols for the candidate universe: BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, TRX/USDT, plus any symbols present in the selected candidate/hybrid artifacts.
   - For each required symbol, read `after_ohlcv_max_utc` from refresh output.
   - Use the minimum timestamp across required symbols.
   - If the minimum timestamp is not at or after `23:59:00Z`, set the cutoff to the previous UTC date; otherwise set it to that UTC date.
   - If refresh fails/geofences/rate-limits, keep the last complete committed date and mark the data-refresh artifact as not final-signoff-source-of-truth.
4. Every downstream command must pass `--oos-end-date <latest_complete_oos_end_date>` or equivalent split args explicitly.

### Phase 2 — Candidate-derived portfolio retune
Run bounded train/validation-only tuning from refreshed candidate inputs. Outputs go to:
`var/reports/profit_moonshot_20260501/live_final_selection_20260510/candidate_portfolio/`.

Required metadata:
- `selection_policy.uses_locked_oos_for_selection=false`
- train/validation ranking inputs only
- selected sleeves, leverage, split metrics, family quotas/correlation caps
- max RSS and command ledger

### Phase 3 — Liquidation-aware validation
Run conservative Binance USDT perpetual-like validation over current-base, prior preferred, candidate-derived winners, integer grid 1x..6x, and top train/validation challengers.

Margin/cost assumptions must include maintenance margin, taker fee, slippage, funding buffer, stress buffer, liquidation fee/event impact, and account-wipeout flag.

Tiny liquidation tolerance can pass only when all are true:
- total liquidation count <= 1
- per-split liquidation count <= 1
- max liquidation event drawdown <= 0.5%
- max event equity-loss fraction <= 0.5%
- no account wipeout
- every split minimum margin buffer > 0
- final OOS return and return/MDD beat current base
- zero-liquidation candidate is preferred when otherwise comparable

### Phase 4 — Candidate-derived comparison interface
Implement a concrete final-selection writer before claiming hybrid superiority. The planned script path is `scripts/research/write_profit_moonshot_live_final_selection.py`; the mandatory regression test path is `tests/test_profit_moonshot_live_final_selection.py`.

Concrete CLI contract:
```bash
.venv/bin/python scripts/research/write_profit_moonshot_live_final_selection.py \
  --refresh-json var/reports/profit_moonshot_20260501/live_final_selection_20260510/data_refresh/data_refresh_latest.json \
  --candidate-portfolio-json var/reports/profit_moonshot_20260501/live_final_selection_20260510/candidate_portfolio/fresh_portfolio_tuning_latest.json \
  --liquidation-json var/reports/profit_moonshot_20260501/live_final_selection_20260510/liquidation_validation/liquidation_aware_current_base_latest.json \
  --legacy-hybrid-json var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/portfolio_hybrid_online_current/hybrid_online_portfolio_latest.json \
  --output-dir var/reports/profit_moonshot_20260501/live_final_selection_20260510/final_decision
```

Required source artifacts by row kind:
- `current_base`: from `--liquidation-json.current_base` or equivalent current-base replay object.
- `direct_candidate`: from `--liquidation-json.reselected_candidate`, `--liquidation-json.forced_5x`, and/or top `candidate_results` that pass gates.
- `candidate_portfolio`: from `--candidate-portfolio-json.best_success_candidate`, `selected_best_candidate`, and `selected_by_train_val_stability`.
- `candidate_multi_portfolio`: from candidate portfolio artifact allocator modes when the row is train/validation-derived and contains multiple sleeves/portfolios; otherwise omit and record `not_available`.
- `candidate_hybrid`: excluded for this pass unless the implementation adds a real profit-moonshot candidate-stream-to-hybrid adapter in the same branch with tests.
- `legacy_hybrid_benchmark`: from `--legacy-hybrid-json`, always `candidate_derived=false`, `benchmark_only=true`, and ineligible for candidate-derived live promotion.
- `cash`: synthetic benchmark row.


Canonical final comparison row schema:
```json
{
  "name": "string",
  "kind": "direct_candidate|candidate_portfolio|candidate_multi_portfolio|candidate_hybrid|legacy_hybrid_benchmark|current_base|cash",
  "source_artifact": "path",
  "candidate_derived": true,
  "benchmark_only": false,
  "selection_policy": {
    "selection_inputs": ["train", "validation"],
    "locked_oos": "report_only_gate_only",
    "uses_locked_oos_for_selection": false
  },
  "splits": {
    "train": {},
    "validation": {},
    "oos": {}
  },
  "liquidation": {},
  "memory": {},
  "decision_gates": {},
  "metrics_explanation": {}
}
```

Allowed rows:
1. `current_base` replay baseline.
2. `direct_candidate`: liquidation-aware single selected sleeve tuple.
3. `candidate_portfolio`: static candidate-derived portfolio from train/validation tuning.
4. `candidate_multi_portfolio`: blend/ensemble across train/validation-approved candidates using train/validation-only weights.
5. `candidate_hybrid`: only if profit-moonshot candidate streams are actually adapted into the hybrid allocator schema.
6. `legacy_hybrid_benchmark`: allowed only with `candidate_derived=false` and `benchmark_only=true`; it may inform risk context but cannot be declared candidate-derived.
7. `cash`: zero-return fallback for risk table completeness.

Selection rule:
- Rank candidate-derived contenders by the frozen train/validation-only stability score `frozen_weighted_train_validation_score_v1`, reusing `scripts/research/tune_profit_moonshot_fresh_portfolio.py::_train_val_stability_score_from_components` semantics.
- Exact formula: `35*min(train_monthlyized_return,0.06) + 45*min(validation_monthlyized_return,0.12) + 0.40*train_sharpe + 0.60*validation_sharpe + 0.35*train_sortino + 0.55*validation_sortino + 0.20*min(train_calmar,20) + 0.30*min(validation_calmar,60) - 35*train_max_drawdown - 45*validation_max_drawdown - 0.15*max(0, leverage-current_base_leverage) - 0.25*max(0, sleeve_count-current_base_sleeve_count)`.
- Locked-OOS applies only as a post-selection gate/report check.
- Benchmark-only rows are not eligible to win the candidate-derived live promotion decision.
- Final chosen row must be gate-passing on OOS; if the train/validation winner fails OOS safety, report rejection and pick the next train/validation-ranked gate-passing row or recommend no promotion.

### Phase 5 — Metrics/reporting
Final JSON/MD artifacts go under:
`var/reports/profit_moonshot_20260501/live_final_selection_20260510/final_decision/`.

Required metrics per row and split:
- total return
- monthlyized/annualized return where supported
- max drawdown
- return/MDD
- Sharpe
- Sortino
- smart Sortino
- Calmar
- volatility/downside volatility
- win rate / positive period ratio
- fill/trade count
- fees, slippage, funding estimates where available
- liquidation count and event count
- min margin buffer and min margin ratio
- max liquidation event drawdown/equity loss
- account wipeout flag
- memory max RSS and guard status
- train/validation score, OOS gate status, live recommendation status

Each metric must have a short explanation in the final report.

### Phase 6 — Verification, commit, push, CI
Sequential verification only:
1. Targeted tests for liquidation, candidate tuning, final comparison/report, hybrid adapter if changed.
2. Full pytest with `/usr/bin/time -v`.
3. `ruff check .`.
4. `python -m compileall -q src scripts tests`.
5. `git diff --check`.
6. Commit using Lore protocol.
7. Push to `private/main`.
8. Confirm GitHub Actions `ci` and `private-ci` green for pushed head.

## ADR
**Decision:** Use a staged, single-heavy-lane final-selection pipeline with an explicit candidate-derived comparison interface and benchmark-only labeling for legacy hybrid rows.

**Drivers:** live-use safety, no-OOS-leakage, current-tail robustness, memory cap, and reproducible CI-backed evidence.

**Alternatives rejected:** concurrent heavy team execution, broad new-alpha search, promotion without fresh data, unlabeled legacy-hybrid comparison.

**Consequences:** slower but safer; may recommend no promotion; any hybrid winner must be proven candidate-derived rather than legacy benchmark-only.

## Team/Ralph handoff guidance
- Team mode can run after PRD/test-spec exist, but leader must permit only one heavy job at a time.
- Suggested team lanes: executor for comparison interface/reporting, test-engineer for tests/memory ledger, architect/verifier for leakage and final evidence. Heavy refresh/backtests remain leader-sequenced or mutex-guarded.
- Ralph loop owns final persistence after team results: verify prompt-to-artifact checklist, run full verification, commit/push/CI, and close state.

## Stop condition
Stop only when the final recommendation artifact, handoff docs/notepad/plans, local verification, Lore commit/push, and GitHub Actions green evidence all exist. If no row is deployable, stop with a documented `no live promotion` recommendation instead of weakening gates.
