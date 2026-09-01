# PRD — Profit moonshot live final selection 2026-05-10

## Objective
Deliver a final, live-ready recommendation for the profit-moonshot strategy family after updating data, retuning/optimizing, comparing candidate-derived portfolio and hybrid/multi-portfolio variants, and verifying all safety/performance gates under an 8 GiB memory cap.

## Users / stakeholders
- Live trading operator deciding whether to deploy the current-base, the prior 5x candidate, a candidate-derived portfolio, a candidate-derived hybrid/multi-portfolio, or no promotion.
- Future agents/operators who need reproducible evidence, commands, metrics explanations, and CI status.

## Scope
### In scope
1. Refresh/audit latest market and support-data tail.
2. Derive and record the latest complete OOS end date from refreshed data.
3. Retune candidate-derived portfolios using train/validation data only.
4. Revalidate direct candidates and candidate portfolios with liquidation-aware replay.
5. Build a normalized final comparison artifact across direct candidate, candidate portfolio, candidate multi-portfolio/blend, candidate hybrid if implemented, legacy hybrid benchmark, current base, and cash fallback.
6. Report broad metrics with short explanations.
7. Recommend the live option or explicitly recommend no promotion.
8. Update `.omx/notepad.md`, `.omx/plans`, `docs/session_handoff_*`, and `var/reports/profit_moonshot_20260501/live_final_selection_20260510/**`.
9. Run targeted tests, full pytest, ruff, compileall, diff-check, commit with Lore protocol, push, and verify GitHub Actions `ci` + `private-ci` green.

### Out of scope
- Placing live orders or changing exchange/account configuration.
- Adding broad new alpha families unless all existing candidates fail and only as a separately planned research task.
- Using locked-OOS to tune, rank, expand, or tie-break candidates.
- Increasing total active session/heavy-run memory beyond 8 GiB.

## Baselines and references
- Preserve current pushed head lineage: `cb10f1e53f5c8dea9be4b8dcdac0f284c6c244dd`.
- Preserve performance baseline: `02f4520cf906f48089b8852c2651a0f1e4bd0c1c`.
- Previous preferred candidate: `integer_audit_diagnostic_quarantine_04`, leverage `5x`.
- Prior artifacts: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_tolerant_retune_20260510/*`.
- Official docs for margin assumptions:
  - Binance USDⓈ-M exchange info: `/fapi/v1/exchangeInfo`.
  - Binance USDⓈ-M notional/leverage brackets: `/fapi/v1/leverageBracket`.

## Functional requirements

### FR1 — Data refresh and cutoff
- Run refresh with `--max-workers 1` by default.
- Write JSON/MD/RSS artifacts under `var/reports/profit_moonshot_20260501/live_final_selection_20260510/data_refresh/`.
- Compute `latest_complete_oos_end_date` from the minimum `after_ohlcv_max_utc` across required symbols.
- Do not use stale script defaults (`2026-05-06`) after refresh unless refresh proves no later complete date exists.
- Store the derived cutoff in final decision metadata and pass it explicitly into downstream commands.

### FR2 — Candidate retuning
- Retune static candidate-derived portfolios using only train/validation inputs.
- Preserve family/correlation/sleeve caps unless PRD/test-spec updates explicitly justify different values.
- Emit selection metadata proving `uses_locked_oos_for_selection=false`.

### FR3 — Liquidation-aware replay
- Compare current-base 2.3427x, forced/current-base integer grid 1x..6x, prior preferred 5x, and retuned candidate-derived portfolio winners.
- Conservative margin model must include maintenance margin, taker fee, slippage, funding, stress buffer, liquidation fee/event impact, and account-wipeout flag.
- Split summaries must include liquidation count, event count, minimum margin buffer, minimum margin ratio, and event impact metrics.

### FR4 — Tiny liquidation tolerance
A row with liquidation events may be considered only if all are true:
- total liquidations <= 1,
- per split liquidations <= 1,
- max liquidation event drawdown <= 0.5%,
- max event equity loss fraction <= 0.5%,
- account wipeout is false,
- every split minimum margin buffer > 0,
- OOS MDD <= 25%,
- OOS return and return/MDD beat current-base replay.
Zero-liquidation rows are preferred over tiny-liquidation rows when risk-adjusted performance is comparable.

### FR5 — Candidate-derived comparison interface
- Implement and run `scripts/research/write_profit_moonshot_live_final_selection.py`.
- Add mandatory tests in `tests/test_profit_moonshot_live_final_selection.py`.
- Produce one canonical comparison JSON/MD artifact.
- Every row must declare:
  - `kind`,
  - `candidate_derived`,
  - `benchmark_only`,
  - source artifact,
  - train/validation/OOS metrics,
  - selection policy,
  - liquidation/margin fields,
  - memory fields,
  - gate status,
  - metric explanations.
- A legacy hybrid artifact may be included only as `legacy_hybrid_benchmark` with `candidate_derived=false` and `benchmark_only=true` unless a true profit-moonshot adapter feeds candidate streams into the hybrid allocator.
- For this pass, `candidate_hybrid` is excluded unless the branch adds a real profit-moonshot candidate-stream-to-hybrid adapter plus tests; legacy hybrid remains benchmark-only.
- Required inputs: refresh JSON, candidate portfolio JSON, liquidation-aware JSON, optional legacy hybrid benchmark JSON, and output directory.
- Final ranking must use the frozen `frozen_weighted_train_validation_score_v1` train/validation-only formula used by `tune_profit_moonshot_fresh_portfolio.py::_train_val_stability_score_from_components`; locked-OOS must not enter the score.

### FR6 — Final recommendation
- Choose the best deployable row from train/validation-ranked contenders after OOS gate checks.
- If no row passes, recommend `no_live_promotion` and preserve the current-base/incumbent.
- Include rejected alternatives and why each failed or lost.
- Include operator guidance for live deployment/readiness, not actual order placement.

### FR7 — Metrics explanations
The final report must explain at least:
- return, monthlyized/annualized return,
- max drawdown,
- return/MDD,
- Sharpe,
- Sortino,
- smart Sortino,
- Calmar,
- volatility/downside volatility,
- positive-period ratio/win rate where available,
- fills/trades,
- fees/slippage/funding/stress buffers,
- liquidation count/event count,
- minimum margin buffer and ratio,
- max liquidation event drawdown/equity loss,
- account wipeout,
- memory max RSS.

### FR8 — Memory and lifecycle
- Keep all heavy jobs sequential.
- Use `/usr/bin/time -v` for heavy commands.
- Record max RSS for refresh, tuning, liquidation replay, hybrid/comparison, full pytest.
- Do not run full pytest concurrently with backtests or refresh.
- Hard fail or reduce scope if a heavy command approaches/exceeds 8 GiB.

### FR9 — Verification and delivery
- Add/adjust tests before implementation changes when behavior is not already locked.
- Run targeted tests and full verification.
- Commit with Lore protocol and push to `private/main`.
- Verify GitHub Actions `ci` and `private-ci` green.

## Acceptance criteria
1. PRD, test spec, and RALPLAN-DR plan artifacts exist.
2. Latest complete OOS end date is derived from refreshed data and recorded.
3. Candidate and hybrid/multi-portfolio comparisons are normalized by `scripts/research/write_profit_moonshot_live_final_selection.py` and clearly label benchmark-only rows.
4. Final selected row, if any, passes all safety/performance gates.
5. Locked-OOS never affects selection ranking.
6. All required metrics and explanations are included.
7. Every heavy command has max RSS evidence <8 GiB.
8. Targeted tests, full pytest, ruff, compileall, and diff-check pass.
9. Lore commit is pushed and `ci`/`private-ci` are green.

## Risks and mitigations
- **OOS leakage:** add poisoned-OOS/ranking tests; inspect final artifacts for selection policy.
- **Stale data default:** require explicit cutoff from refresh metadata in downstream commands.
- **False hybrid superiority:** require candidate-derived flag and benchmark-only labels; do not select legacy benchmark as candidate-derived.
- **OOM/session termination:** one heavy command at a time, max workers 1, memory ledger.
- **Exchange/API refresh blocker:** document geofence/rate-limit/error; use latest committed complete data only if refresh cannot extend and mark source-of-truth limitation.

## Deliverables
- `.omx/plans/profit_moonshot_live_final_selection_20260510.md`
- `.omx/plans/prd-profit-moonshot-live-final-selection-20260510.md`
- `.omx/plans/test-spec-profit-moonshot-live-final-selection-20260510.md`
- `var/reports/profit_moonshot_20260501/live_final_selection_20260510/**`
- `docs/session_handoff_20260510_profit_moonshot_live_final_selection.md`
- `.omx/notepad.md` update
- Lore commit and pushed CI-green head
