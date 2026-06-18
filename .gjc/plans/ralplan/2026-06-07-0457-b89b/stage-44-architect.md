## Summary
G004 manifest driven artifact portfolio composition now meets the reviewed fail closed contract. Invalid or unsafe manifests resolve to cash only, manifest aliases and live selection support are wired, and no blocker was found in the four target files.

## Analysis
- Manifest routing and live selection are present: artifact_manifest_mode is in the strategy alias set at src/lumina_quant/strategies/artifact_portfolio_mode.py:171, the resolver handles both default artifact_manifest_mode and manifest: paths at src/lumina_quant/strategies/artifact_portfolio_mode.py:2205-2215, live_selection includes artifact_manifest_mode at src/lumina_quant/live_selection.py:80, and supports_live_portfolio_mode accepts normalized aliases plus manifest: prefixes at src/lumina_quant/live_selection.py:134-136.
- Cash fail closed behavior is centralized: _manifest_fail_closed_definition returns components empty, cash_weight 1.0, and explicit reason/source markers at src/lumina_quant/strategies/artifact_portfolio_mode.py:283-295. Missing and unreadable manifests also enter that path at src/lumina_quant/strategies/artifact_portfolio_mode.py:366-375.
- Top level manifest safety gates are fail closed: real money flags are blocked at src/lumina_quant/strategies/artifact_portfolio_mode.py:377-383, forbidden OOS provenance at 384-387, optimizer provenance at 388-390, and correlation provenance at 392-395.
- Source artifacts are mandatory file backed inputs: source_artifacts must be a list of objects with id/path at src/lumina_quant/strategies/artifact_portfolio_mode.py:303-313, the source must exist and be a file at 314-317, sha256 must be present and match with OSError mapped to source_artifact_unreadable at 318-326, max_age_hours/freshness and stale checks run at 327-333, and ready plus portfolio_ready are required at 334-335.
- Child validation is fail closed before component creation: child collection/object shape is checked at src/lumina_quant/strategies/artifact_portfolio_mode.py:423-436; child readiness and train/validation provenance markers at 438-453; child real money flags at 454-460; OOS contamination at 461-464; child optimizer and correlation provenance at 465-478; source artifact reconciliation at 479-483; strategy, symbols, and params shape at 484-498.
- Gross exposure is based on declared child gross, not only portfolio weight: leaf_gross and leaf_gross_cap are enforced at src/lumina_quant/strategies/artifact_portfolio_mode.py:499-503, netting group caps at 505-510, and the manifest gross cap adds max(abs(weight), leaf_gross) before permitting a component at 512-517.
- The strategy runtime is compatible with cash fail closed definitions: ArtifactPortfolioModeStrategy builds child strategies only by iterating definition.components at src/lumina_quant/strategies/artifact_portfolio_mode.py:2345-2377, so a fail closed empty component tuple leaves no live child path.
- Tests in the reviewed files exercise the core product contract: valid manifest resolution and source artifact recording at tests/unit/test_artifact_portfolio_mode.py:1288-1299; child OOS, SHA mismatch/missing, directory source, child optimizer provenance, and child correlation provenance fail closed at 1302-1397; malformed collections and child shapes at 1400-1458; weight gross and leaf gross cap behavior at 1461-1490; default alias, manifest: support, and missing default manifest cash at 1493-1502. Live and backtest mode normalization/runtime support are covered in tests/unit/test_backtest_live_portfolio_mode_resolution.py:16-22 and 86-103.
- Tests, linters, and formatters were not run by this review per instruction; this conclusion is based on static inspection of the four target files and caller supplied focused test context.

## Root Cause
No remaining defect was identified. The design uses explicit manifest validation before child component materialization, with a single fail closed cash definition for unsafe states.

## Findings
None.

## Recommendations
1. Approve G004 for the ultragoal checkpoint.
2. Keep future manifest extensions on the same fail closed validation boundary: validate provenance, file source, freshness, readiness, reconciliation, real money flags, and gross exposure before _component_from_row is called.

## Architectural Status
CLEAR

## Code Review Recommendation
APPROVE

## Trade-offs
- Returning cash instead of raising on invalid manifests keeps live routing safe and observable through source_artifacts reason markers, at the cost of requiring monitoring to alert on fail closed reasons.
- Supporting both artifact_manifest_mode and manifest: paths preserves default operator ergonomics while allowing explicit file based validation runs.
