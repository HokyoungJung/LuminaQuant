## Summary
The G004 manifest mode implementation is architecturally sound and fail-closes invalid manifests to cash before producing live components. Caller-provided evidence reports `uv run pytest tests/unit/test_artifact_portfolio_mode.py tests/unit/test_backtest_live_portfolio_mode_resolution.py` passed 46 tests and the final focused suite passed 88 tests; I did not rerun tests per review constraints.

## Analysis
- Manifest routing is explicit: `artifact_manifest_mode` resolves the default manifest path, while `manifest:` dynamically resolves caller-supplied manifests before any alias expansion (`src/lumina_quant/strategies/artifact_portfolio_mode.py:2205-2216`). Live support mirrors that with `artifact_manifest_mode` in `SUPPORTED_LIVE_PORTFOLIO_MODES` and `token.startswith("manifest:")` support (`src/lumina_quant/live_selection.py:12-80`, `src/lumina_quant/live_selection.py:134-160`).
- The fail-closed primitive returns `components=()`, `cash_weight=1.0`, and durable reason metadata (`src/lumina_quant/strategies/artifact_portfolio_mode.py:283-295`). Missing or unreadable manifests also route through that primitive (`src/lumina_quant/strategies/artifact_portfolio_mode.py:364-375`).
- Root manifest and child controls block real-money flags, forbidden OOS keys, optimizer provenance, and correlation provenance before component construction (`src/lumina_quant/strategies/artifact_portfolio_mode.py:377-395`, `src/lumina_quant/strategies/artifact_portfolio_mode.py:438-478`).
- Source artifact validation requires list/object shape, file existence, file source, non-empty SHA, SHA equality, freshness, and readiness before any child can reconcile to that source (`src/lumina_quant/strategies/artifact_portfolio_mode.py:303-337`). Child source IDs must reconcile with validated source artifacts (`src/lumina_quant/strategies/artifact_portfolio_mode.py:479-483`).
- Child shape validation occurs before the zero-weight skip, so malformed zero-weight children still fail closed (`src/lumina_quant/strategies/artifact_portfolio_mode.py:484-513`). Gross exposure uses `max(abs(weight), leaf_gross)` and checks aggregate gross against the capped manifest gross (`src/lumina_quant/strategies/artifact_portfolio_mode.py:499-518`).
- Tests cover valid manifest resolution, OOS-contaminated child, source SHA mismatch/missing, non-file source, nested child optimizer OOS, bad child correlation, malformed collections/scalar child shapes, malformed zero-weight child, gross breach including leaf gross, default missing manifest cash mode, and dynamic/default live support (`tests/unit/test_artifact_portfolio_mode.py:1203-1504`, `tests/unit/test_backtest_live_portfolio_mode_resolution.py:1-80`).

## Root Cause
Not applicable; the reviewed implementation fixes the prior risk by centralizing manifest validation ahead of component creation and reusing a single fail-closed cash definition for all invalid-manifest exits.

## Findings
- LOW — `tests/unit/test_artifact_portfolio_mode.py:1285-1504`: the target tests do not include explicit negative cases for stale source artifacts, unreconciled child `source_artifact_id`, or manifest/child real-money flags set to true. The implementation has direct guards for those branches, so this is not a blocker, but adding those adversarial tests would reduce regression risk around the highest-risk live-safety contract.

## Recommendations
1. Keep the current implementation as the G004 checkpoint baseline.
2. Add follow-up unit tests for stale source artifacts, unreconciled child source IDs, and root/child real-money true flags to make the code-backed safety guarantees test-backed as well.
3. Keep dynamic manifest routing constrained to `manifest:` plus `artifact_manifest_mode`; do not add broader live-mode fallbacks.

## Architectural Status
WATCH

## Code Review Recommendation
COMMENT

## Trade-offs
- Current explicit manifest validator: simple, fail-closed, easy to audit; requires manually adding tests for every new safety branch.
- Schema-library validation: stronger declarative shape guarantees; more dependency and migration overhead for a small live-safety surface.
