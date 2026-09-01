## Summary
The fixed G004 manifest path is substantially safer than the prior blocker state: default and dynamic manifest modes route beside aliases, invalid source artifacts fail closed to cash, and caller-run focused suites passed (41 targeted tests, then 83 final focused tests). Red-team review is not clear, because malformed child handling still has fail-open shapes that can produce live components despite the manifest fail-closed contract.

## Analysis
Evidence reviewed only in the requested files. `artifact_manifest_mode` is registered in both the strategy alias surface and live-selection support (`src/lumina_quant/strategies/artifact_portfolio_mode.py:171`, `src/lumina_quant/live_selection.py:80`), and dynamic `manifest:` references are accepted by live support (`src/lumina_quant/live_selection.py:134-136`) and resolved through `_manifest_definition_from_path` (`src/lumina_quant/strategies/artifact_portfolio_mode.py:2169-2179`). The fail-closed helper returns no components with `cash_weight=1.0` and records `manifest_fail_closed_to_cash=true` (`artifact_portfolio_mode.py:285-294`). Source artifact controls require path existence, non-empty SHA, SHA match, freshness, and ready flags (`artifact_portfolio_mode.py:305-324`). Manifest/child OOS and real-money flags are explicitly rejected (`artifact_portfolio_mode.py:367-380`, `artifact_portfolio_mode.py:432-448`), and leaf gross contributes to aggregate gross before manifest gross-cap enforcement (`artifact_portfolio_mode.py:465-481`).

The focused tests cover the main happy path and several adversarial paths: valid manifest resolution (`tests/unit/test_artifact_portfolio_mode.py:1282-1295`), child OOS contamination (`:1299-1312`), source SHA mismatch/missing (`:1315-1336`), child optimizer OOS (`:1339-1356`), blank strategy class malformed child (`:1359-1367`), gross cap including leaf gross (`:1371-1400`), and default/dynamic live support with missing default manifest cash mode (`:1403-1412`). Runtime/backtest integration coverage exists for portfolio mode normalization and live runtime config (`tests/unit/test_backtest_live_portfolio_mode_resolution.py:16-99`). Caller evidence says `uv run pytest tests/unit/test_artifact_portfolio_mode.py tests/unit/test_backtest_live_portfolio_mode_resolution.py` passed 41 tests, and the final focused suite passed 83 tests.

## Root Cause
The remaining risk is that manifest validation filters or normalizes malformed child rows before deciding validity, rather than treating any malformed child shape as an invalid manifest. That contradicts fail-closed semantics: a partially malformed manifest can still yield live components if at least one valid child remains, and symbol shape errors can be converted into bogus component symbols instead of failing closed.

## Findings
1. HIGH — `src/lumina_quant/strategies/artifact_portfolio_mode.py:409-414`: malformed non-object child entries are silently dropped by `[item for item in list(payload.get("children") or []) if isinstance(item, dict)]`. Impact: a manifest containing one malformed child plus one valid child can still produce live components, so a malformed child does not fail closed. Fix: require `children` to be a non-empty list and return `_manifest_fail_closed_definition(..., reason="child_invalid:<index>")` on the first non-dict child instead of filtering it out; add a regression test with `["bad", valid_child]` proving cash-only output.

2. HIGH — `src/lumina_quant/strategies/artifact_portfolio_mode.py:511-527`: manifest child validation delegates to `_component_from_row`, which only rejects missing `strategy_class` and converts `symbols` with `list(row.get("symbols") or [])`. Impact: malformed symbol payloads such as a scalar string become per-character symbols and still instantiate a live child; empty symbols also pass. Fix: add manifest-specific child schema validation before `_component_from_row`: require non-empty list/tuple of non-empty symbol strings, mapping params, positive numeric weight/leaf gross, and fail closed on invalid types. Keep `_component_from_row` permissive if legacy artifact rows rely on it, but do not reuse it as the only manifest schema guard.

## Recommendations
1. Change manifest child parsing to fail on any non-dict child rather than dropping malformed entries.
2. Add manifest-specific schema checks for symbols, params, numeric weight/leaf gross, source id, and readiness fields before component creation.
3. Add focused regression tests for non-object child, scalar/empty symbols, missing `portfolio_ready`, stale source artifact, unreconciled source artifact, and explicit manifest/child real-money flags. The real-money and stale/unreconciled code paths exist, but the reviewed tests do not toggle those adversarial cases.

## Architectural Status
BLOCK

## Code Review Recommendation
REQUEST CHANGES

## Trade-offs
- Strict manifest schema validation preserves the fail-closed safety contract and is the recommended option; it may require adapting malformed historical manifests before live use.
- Continuing permissive normalization minimizes compatibility churn but leaves the live-composition boundary unable to prove that malformed children never produce components.
