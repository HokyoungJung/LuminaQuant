# Test Spec: Profit Moonshot Strategy Validity Audit

## Unit/targeted tests
- `tests/test_profit_moonshot_strategy_validity_audit.py`
  - fixed `calendar_rotation` with `calendar_long_symbol`/`calendar_short_symbol` and month lists is rejected.
  - fixed calendar sleeve names such as `fresh_calendar_trx_takeprofit_sethusdt_...` produce `strategy_validity_gate=false` and reasons including calendar/fixed-symbol seasonality.
  - non-calendar state-driven specs such as residual/flow/funding/cross-sectional rules are not rejected solely for using the universe.
  - artifact row expansion audits all row sleeves and returns candidate-level aggregate gate.
- Extend `tests/test_profit_moonshot_live_final_selection.py`
  - strategy validity failure blocks `deployable_candidate` even when liquidation/performance gates pass.
  - report-only rows still include validity reasons without being promotable.

## Integration/smoke
- Run strategy-validity audit script on latest live-final-selection sources.
- Run liquidation-aware validation/retune with validity-filtered seeds if implemented.
- Rebuild final selection with strategy-validity artifact wired in.

## Regression/full checks
- Targeted pytest for new/modified tests.
- Full `pytest`.
- `ruff check .`.
- `python3 -m compileall -q .`.
- `git diff --check`.
- CI/private-ci after push.

## Memory evidence
- Capture `/usr/bin/time -v` or existing runner evidence for audit/retune/final-selection commands; all max RSS under 8 GiB.


## Additional mandatory validity tests
- `closure_manifest` completeness: each audited source item has `path`, `artifact_kind`, `row_count`, and `source_role`; final selection/candidate portfolio/liquidation/candidate hybrid/merged CSV/current-base source roles are represented when files exist.
- Fail-closed metadata: a final-selection row missing `strategy_validity` or missing required strategy-validity fields cannot set `deployable_candidate=true`.
- Calendar-primary invalidation: every `calendar_rotation` and `calendar_spread` primary-alpha spec fails without robust external/mechanistic evidence, even if the asset target is dynamically selected.
- Non-overblocking dynamic signals: residual, funding/OI, flow, adaptive/trend, cross-sectional rank, compression/volatility, and pair residual/momentum specs pass the strategy-validity classifier when they only use time/session/day fields as secondary filters.
- No new search handoff: tests/reporting assert the strategy-validity run uses existing artifacts/CSV only and emits `no_live_promotion` rather than launching broad alpha generation when no candidate remains.

### Closure manifest parity tests
- Assert `closure_manifest.source_role` covers all applicable roles: `final_selection_json`, `final_selection_md`, `liquidation_validation`, `candidate_portfolio`, `candidate_hybrid`, `merged_candidate_csv`, `current_base`, `passing_artifacts`, and per-row/per-sleeve traced source coverage for every final row kind.
- Assert every closure manifest entry has non-empty `path`, `artifact_kind`, `row_count`, and `source_role`.
- Assert optional/missing artifacts appear in `missing_optional_sources` with a reason; they must not disappear silently.
