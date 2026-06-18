## Summary
G004 remains BLOCKED. The manifest route implements many required guardrails, but two fail-closed gaps remain in artifact_portfolio_mode.py: nonpositive children can bypass malformed and leaf gross validation while other children still go live, and source artifact hashing exceptions escape instead of converting to a cash-only definition.

## Analysis
Reviewed only the requested files:
- /home/hoky/Quants-agent/LuminaQuant/src/lumina_quant/strategies/artifact_portfolio_mode.py
- /home/hoky/Quants-agent/LuminaQuant/src/lumina_quant/live_selection.py
- /home/hoky/Quants-agent/LuminaQuant/tests/unit/test_artifact_portfolio_mode.py
- /home/hoky/Quants-agent/LuminaQuant/tests/unit/test_backtest_live_portfolio_mode_resolution.py

Positive evidence:
- Manifest modes are routed beside aliases: artifact_manifest_mode and manifest: are resolved in artifact_portfolio_mode.py lines 2199-2209, and live_selection.py line 136 accepts manifest: references.
- Real-money flags fail closed at manifest and child level in artifact_portfolio_mode.py lines 371-377 and 448-454.
- Top-level and child OOS flags, optimizer provenance, and correlation provenance are checked before component creation in artifact_portfolio_mode.py lines 378-389 and 455-472.
- Source artifacts require list shape, id/path, sha256, freshness via max_age_hours plus file mtime, and exact ready/portfolio_ready in artifact_portfolio_mode.py lines 303-330.
- Child readiness is exact True in artifact_portfolio_mode.py lines 431-435.
- Positive children apply leaf_gross to the portfolio gross cap in artifact_portfolio_mode.py lines 481-497.

Blocking gaps:
- artifact_portfolio_mode.py lines 478-480 skip children with weight <= 0.0 before schema validation at lines 499-511 and before leaf_gross validation at lines 481-497. A manifest with one valid positive child and a zero-weight malformed or gross-breaching child will still return live components instead of cash. That violates malformed-child cash fail-close and the broader invalid-manifest fail-closed contract.
- artifact_portfolio_mode.py lines 314-320 call exists() and then _file_sha256(source_path), but validation exceptions are not caught by _manifest_definition_from_path lines 391-397. A source_artifacts entry pointing at an existing directory or unreadable path raises out of resolution instead of returning a cash-only PortfolioModeDefinition. That violates invalid-manifest fail-closed behavior at the source-artifact boundary.

Test coverage evidence:
- tests/unit/test_artifact_portfolio_mode.py lines 1383-1415 cover malformed collection and scalar child shapes for a single positive child.
- tests/unit/test_artifact_portfolio_mode.py lines 1417-1426 cover malformed positive child cash fail-close.
- tests/unit/test_artifact_portfolio_mode.py lines 1429-1458 cover gross-cap and leaf_gross cap behavior for positive children.
- The requested tests do not cover zero-weight malformed siblings, zero-weight leaf_gross breaches, source artifact unreadable/directory paths, source staleness, or source readiness failures.

No tests, linters, or formatters were run per assignment constraints.

## Root Cause
Validation order treats nonpositive children as ignorable before proving that their manifest entries are well formed and within safety limits. The fail-closed exception boundary covers manifest JSON loading but not later source-artifact validation and hashing.

## Findings
1. Severity: HIGH. File/reference: src/lumina_quant/strategies/artifact_portfolio_mode.py lines 478-511. Impact: malformed or leaf-gross-breaching nonpositive children can coexist with a valid positive child and live components will still be produced. Fix: validate every child schema, readiness/provenance, source reconciliation, and leaf_gross bounds before any weight-based skip; only skip component instantiation after the child has passed safety validation, or fail nonpositive weights explicitly if the manifest schema forbids them.
2. Severity: HIGH. File/reference: src/lumina_quant/strategies/artifact_portfolio_mode.py lines 314-320 and 391-397. Impact: an invalid source artifact path that exists but cannot be hashed can raise instead of returning cash, so invalid manifests do not consistently fail closed. Fix: require source_path.is_file(), wrap source artifact stat/hash operations in OSError/ValueError handling, and convert those failures to manifest_fail_closed_definition reasons such as source_artifact_unreadable:<id>.

## Recommendations
1. Move child structural and leaf_gross validation ahead of the weight <= 0.0 continue, or remove support for nonpositive child entries.
2. Extend source artifact validation to fail closed for non-file, unreadable, and stat/hash exceptions.
3. Add focused tests for zero-weight malformed sibling, zero-weight leaf_gross breach with another valid child, source artifact not_ready, source artifact stale, and source artifact path as a directory.
4. Add a runtime-config smoke test for manifest: references through live_selection.resolve_portfolio_mode_runtime_config.

## Architectural Status
BLOCK

## Code Review Recommendation
REQUEST CHANGES

## Trade-offs
- Strictly reject nonpositive child weights: simpler fail-closed behavior and less ambiguity, but manifests cannot carry disabled leaves.
- Validate then skip nonpositive children: preserves disabled-leaf annotations while ensuring malformed or unsafe entries still close to cash.
- Catch all source validation exceptions locally: stronger live safety and consistent cash fallback, with slightly more explicit error-reason plumbing.
