## Summary
Reviewed only the four requested G004 files. The manifest path now fails closed to cash for invalid inputs, validates regular source artifacts with sha, freshness, and readiness gates, enforces strict child provenance and shape before component construction, registers live-selection support, and keeps real-money execution disabled. No blocking findings were identified.

## Analysis
Spec compliance: artifact_portfolio_mode.py registers artifact_manifest_mode with the portfolio aliases at lines 170-171 and resolve_portfolio_mode_definition routes both artifact_manifest_mode and manifest-prefixed paths through _manifest_definition_from_path at lines 2205-2215. live_selection.py registers artifact_manifest_mode in SUPPORTED_LIVE_PORTFOLIO_MODES at lines 12-80, accepts manifest-prefixed references at lines 134-136, and feeds supported portfolio modes into the shared runtime resolver at lines 139-160.

Fail-closed behavior: _manifest_fail_closed_definition returns no components, cash_weight 1.0, and explicit manifest_fail_closed metadata at artifact_portfolio_mode.py lines 283-295. _manifest_definition_from_path uses that path for missing or unreadable manifests, real-money flags, top-level OOS contamination, invalid optimizer provenance, invalid correlation provenance, source-artifact failures, malformed children collections, empty children, malformed child rows, gross-cap breaches, and no positive validated children at lines 363-536.

Source artifact gates: _validate_manifest_source_artifacts requires source_artifacts to be a sequence of objects, requires id and path, resolves relative paths against the manifest directory, rejects missing and non-file paths, requires sha256, compares the file sha, requires positive max_age_hours freshness, rejects stale artifacts, and requires both ready and portfolio_ready to be exactly True at artifact_portfolio_mode.py lines 299-349.

Child strictness: child validation occurs before the non-positive weight skip, so zero-weight malformed children still fail closed. Readiness, no-current-fold-OOS provenance, train-validation optimizer provenance, real-money flags, forbidden OOS flags, optimizer provenance, correlation provenance, source-artifact reconciliation, strategy_class, non-scalar symbols, params shape, leaf gross caps, and netting group caps are all checked before _component_from_row can build a live component at artifact_portfolio_mode.py lines 435-536.

Gross-cap semantics: leaf_gross defaults from weight but is applied independently through leaf_gross, leaf_gross_cap, netting_group_gross_cap, and portfolio gross accumulation using max(abs(weight), leaf_gross). This prevents understated weights from bypassing the manifest gross cap at artifact_portfolio_mode.py lines 494-519.

Tests reviewed: test_artifact_portfolio_mode.py covers valid manifest resolution, OOS fail-closed, sha mismatch, missing sha, directory source rejection, child optimizer and correlation provenance rejection, malformed source and child collections, scalar child shapes, malformed child strategy_class, zero-weight malformed child validation, gross-cap breach, leaf-gross gross-cap breach, and default missing manifest cash fallback at lines 1278-1504. test_backtest_live_portfolio_mode_resolution.py covers bare and wrapped live portfolio mode support, source sleeve expansion, shared backtest live runtime resolution, and runtime symbols at lines 14-100.

## Root Cause
Not applicable. This was a final read-only validation sweep after fixes.

## Findings
No blocking findings.

LOW advisory - tests/unit/test_artifact_portfolio_mode.py: the reviewed tests do not directly exercise every implemented fail-closed branch, including stale source artifacts, missing freshness, source not ready, unreconciled child source_artifact_id, manifest or child real-money flags, child leaf_gross_cap breach, and netting_group_gross_cap breach. The source code implements these guards, so this is not blocking, but adding table rows for those branches would reduce regression risk.

## Recommendations
1. Approve the G004 artifact portfolio manifest fail-closed mode as implemented.
2. Add the advisory branch-coverage rows in a follow-up test-only cleanup before the next manifest schema expansion.
3. Keep manifest-prefixed live support in live_selection.py and artifact_portfolio_mode.py synchronized when adding future aliases.

## Architectural Status
CLEAR

## Code Review Recommendation
APPROVE

## Trade-offs
- Current fail-closed resolver: simple operator-safe behavior, invalid manifests become cash without propagating live components; downside is callers must inspect manifest_fail_closed_reason for diagnostics.
- Raising exceptions on invalid manifests: stronger failure visibility; downside is worse live safety because runtime selection could crash instead of cashing out.
- Duplicated live support sets: explicit and easy to audit; downside is drift risk, mitigated by existing live-selection tests and the advisory recommendation.
