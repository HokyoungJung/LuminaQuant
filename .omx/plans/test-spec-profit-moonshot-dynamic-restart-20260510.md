# Test Spec: Profit Moonshot Dynamic Restart and Research-History Ledger

## Tests-first requirements
Implementation must start by adding failing/covering tests before production code or artifact generation changes.

## Targeted unit tests
1. **Research-history schema**
   - Add `tests/test_profit_moonshot_research_history.py` or equivalent.
   - Assert Markdown and JSON outputs are generated from a minimal fixture.
   - Assert JSON contains `strategy_chronology`, `source_history_inventory`, `source_search_ledger`, `decision_log`, `invalidity_lessons`, `future_session_instructions`, and `generation_metadata`.
   - Assert each strategy entry has all required chronology fields.
   - Assert each source/search entry has all required ledger fields.
   - Assert duplicate-search fields (`normalized_key`, `do_not_repeat_note`, `staleness_policy`, `recheck_before_use`) are non-empty, and repeated source/search clusters collapse to stable normalized keys while retaining all associated dates/families.

2. **Research-history source coverage**
   - Assert known local artifacts from 2026-05-01 through 2026-05-10 appear in `source_history_inventory` and are mapped to the ledger when reconstructable.
   - Assert every inventory item has either a ledger reference or `not_reconstructable=true` with a non-empty reason.
   - Assert known external/reference clusters appear: Binance funding/OI/taker-flow/liquidation, Hyperliquid metadata/candles/funding/fees, Tickmill instruments/spreads/swaps, crypto momentum/reversal/risk-factor literature.
   - Assert each source entry explains `content_summary` and `what_was_used` separately.
   - Assert no-orphan coverage: every ledger entry refers back to at least one inventory item or newly consulted source record.

3. **Promotion fail-closed gates**
   - Extend final-selection tests so missing source-ledger/research-history metadata blocks `deployable_candidate`.
   - Assert calendar-primary alpha still blocks promotion even with strong metrics.
   - Assert non-integer leverage blocks live promotion and is benchmark-only.
   - Assert missing liquidation split metrics, margin buffer, or memory evidence blocks promotion.

4. **Hybrid source inheritance**
   - Assert hybrid construction discards or quarantines source rows that are calendar-invalid, non-integer, liquidation-unsafe, or missing source-ledger metadata.
   - Assert a hybrid cannot promote unless all active source candidates pass live-source gates.

5. **Locked-OOS selection guard**
   - Assert selection/ranking/tuning functions do not read OOS metrics when choosing thresholds, sleeves, weights, or active source rows.
   - Assert locked-OOS fields are only used by final gate/report code paths.

6. **Dynamic strategy validity**
   - Assert dynamic residual/funding/OI/flow/trend/cross-sectional/compression/pair primary signals pass classifier when time/session features are only secondary filters.
   - Assert calendar/window/month/day primary signal types fail unless explicitly marked as externally justified and separately robustness-tested.

## Integration/smoke tests
- Generate the research-history Markdown/JSON from existing local artifacts and validate schema plus manifest no-orphan completeness.
- Run strategy-validity audit with research-history/source-link metadata required.
- Run bounded dynamic candidate replay/tuning on a small fixture or capped input to prove command path and output schema.
- Run final selection on fixture artifacts to prove no calendar/non-integer/liquidation-unsafe/source-untraceable candidate can promote.

## Full validation commands
Run and record output/RSS where applicable:
- Targeted pytest for all changed tests.
- Dynamic replay/tuning commands under `/usr/bin/time -v` or equivalent RSS evidence.
- Candidate portfolio/hybrid build under `/usr/bin/time -v` or equivalent RSS evidence.
- Final selection/report writer.
- Full `uv run --extra dev pytest -q`.
- `uv run --extra dev ruff check .`.
- `python3 -m compileall -q src scripts tests`.
- `git diff --check`.
- Commit/push and verify GitHub Actions `ci` and `private-ci` success.

## Memory acceptance
- Every recorded Max RSS must be <8 GiB.
- Heavy backtests must not run concurrently unless summed RSS evidence stays below 8 GiB.

## Final report acceptance
- Final report explains every metric shown.
- Final report lists all final candidates and material rejected families with train/validation/OOS metrics when available.
- Final report states whether live deployment is recommended; if yes, identify exact strategy/portfolio/hybrid and why; if no, state why no promotion is safer.
