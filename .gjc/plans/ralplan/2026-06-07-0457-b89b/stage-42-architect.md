## Summary
G004 wires manifest-driven portfolio composition beside aliases and live dynamic `manifest:` support, and the reviewed tests exercise the happy path plus several key fail-closed cases. Remaining blockers prevent a clean ultragoal checkpoint: source-artifact validation can still raise instead of returning the cash definition for some invalid source paths, and required adversarial coverage is missing for real-money, stale source, and unreconciled source cases.

## Analysis
- Manifest surface is present: `DEFAULT_ARTIFACT_PORTFOLIO_MANIFEST_PATH`, `MANIFEST_PORTFOLIO_MODE_PREFIX`, global gross cap, and forbidden OOS keys are defined in `src/lumina_quant/strategies/artifact_portfolio_mode.py:82-97`; `artifact_manifest_mode` is listed with aliases at `src/lumina_quant/strategies/artifact_portfolio_mode.py:118-176`.
- Fail-closed return shape is explicit: `_manifest_fail_closed_definition` records a reason, sets `manifest_fail_closed_to_cash`, returns no components, and sets `cash_weight=1.0` in `src/lumina_quant/strategies/artifact_portfolio_mode.py:283-295`.
- Dynamic and default manifest resolution is wired before legacy aliases: `artifact_manifest_mode` resolves through the default manifest path and `manifest:` resolves the supplied path in `src/lumina_quant/strategies/artifact_portfolio_mode.py:2199-2209`.
- Provenance gates cover manifest-level and child-level OOS, optimizer, correlation, readiness, and real-money flags in `src/lumina_quant/strategies/artifact_portfolio_mode.py:371-388` and `src/lumina_quant/strategies/artifact_portfolio_mode.py:432-468`.
- Source artifact validation checks collection shape, id/path, existence, sha presence/match, max age, staleness, and readiness in `src/lumina_quant/strategies/artifact_portfolio_mode.py:298-331`, then requires children to reconcile to a source id in `src/lumina_quant/strategies/artifact_portfolio_mode.py:473-477`.
- Gross-cap handling uses `gross_cap`, `leaf_gross`, leaf cap, netting group cap, and cumulative gross in `src/lumina_quant/strategies/artifact_portfolio_mode.py:405-497`; this directly addresses the leaf gross cap regression.
- Live selection includes `artifact_manifest_mode` in `SUPPORTED_LIVE_PORTFOLIO_MODES` and accepts any normalized `manifest:` token in `src/lumina_quant/live_selection.py:12-80` and `src/lumina_quant/live_selection.py:134-166`.
- Reviewed unit coverage includes valid manifest resolution and source-artifact recording in `tests/unit/test_artifact_portfolio_mode.py:1285-1299`; child OOS fail-close in `tests/unit/test_artifact_portfolio_mode.py:1302-1315`; source sha mismatch and missing sha in `tests/unit/test_artifact_portfolio_mode.py:1318-1339`; nested optimizer OOS in `tests/unit/test_artifact_portfolio_mode.py:1342-1359`; bad child correlation in `tests/unit/test_artifact_portfolio_mode.py:1362-1380`; malformed collections and child shape cases in `tests/unit/test_artifact_portfolio_mode.py:1383-1426`; gross cap and leaf gross cap cases in `tests/unit/test_artifact_portfolio_mode.py:1429-1458`; default and dynamic support assertions in `tests/unit/test_artifact_portfolio_mode.py:1461-1468`; and live/backtest runtime resolution in `tests/unit/test_backtest_live_portfolio_mode_resolution.py:16-103`.
- Caller-run evidence says `uv run pytest tests/unit/test_artifact_portfolio_mode.py tests/unit/test_backtest_live_portfolio_mode_resolution.py` passed 44 tests, and the final focused suite passed 86 tests. I did not run tests, linters, or formatters.

## Root Cause
The implementation added explicit validation branches for known manifest defects, but the source-artifact path still performs file IO through `_file_sha256` and `stat` without an `is_file` guard or exception-to-cash boundary. The test set verifies the second blocker fixes, but does not yet adversarially lock the real-money, stale-source, and unreconciled-source gates required by the contract.

## Findings
1. Severity HIGH, `src/lumina_quant/strategies/artifact_portfolio_mode.py:249-255` and `src/lumina_quant/strategies/artifact_portfolio_mode.py:313-331`: `_validate_manifest_source_artifacts` calls `_file_sha256(source_path)` after only `exists()`. A manifest can point `source_artifacts[].path` at a directory or another non-readable existing path, causing `Path.open("rb")` or `stat` to raise instead of returning a cash-only `PortfolioModeDefinition`. Impact: invalid manifests do not always fail closed to cash. Fix: require `source_path.is_file()` before hashing, catch IO/OSError around hash/stat, return a source-artifact unreadable reason, and add a regression test.
2. Severity HIGH, `tests/unit/test_artifact_portfolio_mode.py:1194-1468`: the reviewed tests never flip manifest-level or child-level `real_money_execution`, `allow_real_money`, or `ready_for_real` to true. Source code has gates at `src/lumina_quant/strategies/artifact_portfolio_mode.py:371-377` and `src/lumina_quant/strategies/artifact_portfolio_mode.py:448-454`, but the acceptance request explicitly calls for adversarial real-money-disabled coverage. Fix: add fail-close tests for at least one manifest-level and one child-level real-money flag.
3. Severity MEDIUM, `tests/unit/test_artifact_portfolio_mode.py:1194-1468`: no test asserts `source_artifact_stale:*` or `child_source_unreconciled:*`, despite source gates at `src/lumina_quant/strategies/artifact_portfolio_mode.py:323-330` and `src/lumina_quant/strategies/artifact_portfolio_mode.py:473-477`. Impact: future regressions could allow stale or unreconciled artifacts to emit live components without caller-run evidence catching it. Fix: add stale max-age and unknown child `source_artifact_id` regression tests.

## Recommendations
1. Block checkpoint until `_validate_manifest_source_artifacts` converts non-file and unreadable source paths into cash fail-closed definitions.
2. Add adversarial tests for manifest and child real-money flags.
3. Add adversarial tests for stale source artifacts and unreconciled child source ids.
4. Keep the existing valid/OOS/sha/optimizer/correlation/malformed/gross/default/dynamic/backtest tests; they cover the fixed paths well.

## Architectural Status
BLOCK

## Code Review Recommendation
REQUEST CHANGES

## Trade-offs
- Strict source path validation: stronger fail-closed behavior and clearer operator diagnostics, at the cost of rejecting unusual non-file artifact providers. The manifest contract is file-backed JSON, so this is the safer option.
- Test-only additions for real-money/stale/unreconciled: low implementation cost and high regression value, with minimal runtime impact.
