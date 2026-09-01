## Summary
G004 is close on the named happy path and several safety gates, but the fail-closed boundary is not complete. One implementation path can accept an invalid manifest with a malformed child row and still instantiate other live components, so checkpoint status is BLOCK pending a fail-closed fix and regression coverage.

## Analysis
Evidence inspected only in the four target files. Caller-run evidence was accepted as reported: `uv run pytest tests/unit/test_artifact_portfolio_mode.py tests/unit/test_backtest_live_portfolio_mode_resolution.py` passed 37 tests, and the final focused suite passed 79 tests. No tests, linters, or formatters were run in this review.

Implementation evidence: `artifact_portfolio_mode.py:344-482` loads a manifest, rejects missing and unreadable files, rejects top-level real-money flags, rejects forbidden OOS provenance, verifies source artifact existence, optional sha256, max age, readiness, child readiness, child OOS flags, child source reconciliation, leaf gross caps, netting group caps, and aggregate gross caps. `artifact_portfolio_mode.py:2148-2157` wires both `artifact_manifest_mode` and `manifest:<path>` into the resolver. `live_selection.py:134-156` accepts the default mode and dynamic manifest prefix through live runtime config.

Test evidence: `test_artifact_portfolio_mode.py:1198-1347` covers a valid manifest, child OOS contamination, source sha mismatch, gross cap breach, and default missing manifest cash mode. `test_backtest_live_portfolio_mode_resolution.py:14-103` covers bare and wrapped live mode acceptance, source sleeve expansion, backtest routing through live runtime config, and runtime symbols for an existing mode.

## Root Cause
The manifest parser attempts to be permissive when collecting children. It filters non-object child entries instead of treating them as manifest invalidity, which conflicts with the G004 fail-closed contract.

## Findings
- HIGH, `artifact_portfolio_mode.py:406`: malformed child entries are silently discarded by `children = [item for item in list(payload.get("children") or []) if isinstance(item, dict)]`. A manifest with one valid child plus a non-object child can still return live components, even though invalid manifests must fail closed to cash. Fix by validating that `children` is a list or tuple and every entry is a dict; any malformed entry should return `_manifest_fail_closed_definition` with a specific reason. Add a regression test using a valid child plus a malformed child and assert components empty and cash weight 1.0.
- MEDIUM, `test_artifact_portfolio_mode.py:1204-1206`: real-money disabled is implemented at `artifact_portfolio_mode.py:361-365`, but target tests only exercise false real-money flags. Add parameterized adversarial coverage for `real_money_execution`, `allow_real_money`, and `ready_for_real` set true, asserting `manifest_real_money_enabled` and cash fail-close.

## Recommendations
1. Block checkpoint until malformed manifest children fail closed and a regression test covers mixed valid plus malformed child arrays.
2. Add real-money enabled fail-close tests for all accepted top-level real-money flags.
3. Consider adding manifest-specific live runtime config coverage for `resolve_portfolio_mode_runtime_config("artifact_manifest_mode")` and a `manifest:<tmp_path>` reference so dynamic/default live support is directly covered through the live entrypoint.

## Architectural Status
BLOCK

## Code Review Recommendation
REQUEST CHANGES

## Trade-offs
- Strict schema validation: strongest safety boundary and aligns with fail-closed mode; may reject previously tolerated malformed research artifacts.
- Permissive filtering: preserves compatibility with dirty artifacts but violates invalid-manifest fail-closed semantics and hides data quality defects.
