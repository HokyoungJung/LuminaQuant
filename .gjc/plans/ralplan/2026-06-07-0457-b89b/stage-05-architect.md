## Summary
Reviewed the four requested G004 files read-only. Most manifest controls are present, including manifest/default routing, provenance checks, source artifact freshness/hash/ready gates, real-money fail-close flags, and leaf-gross based gross caps, but two fail-closed gaps remain.

Recommendation: request changes before the ultragoal checkpoint because invalid source artifact paths can escape the cash fail-close boundary, and malformed zero/non-positive-weight child rows are pruned before full schema validation.

## Analysis
- Manifest-driven composition is wired beside aliases: resolve_portfolio_mode_definition() handles artifact_manifest_mode and dynamic manifest: paths before alias resolution in src/lumina_quant/strategies/artifact_portfolio_mode.py:2167-2210, while existing aliases continue through _alias_rows() / _expand_reference() in src/lumina_quant/strategies/artifact_portfolio_mode.py:1071-1125 and src/lumina_quant/strategies/artifact_portfolio_mode.py:2033-2152.
- Live selection recognizes both static modes and manifest paths: supports_live_portfolio_mode() accepts registered modes or manifest: references in src/lumina_quant/live_selection.py:134-136, and resolve_portfolio_mode_runtime_config() builds the ArtifactPortfolioModeStrategy runtime config from the resolved definition in src/lumina_quant/live_selection.py:139-159.
- The normal manifest fail-close return is correct: _manifest_fail_closed_definition() returns no components and cash_weight=1.0 in src/lumina_quant/strategies/artifact_portfolio_mode.py:283-296.
- Root and child OOS/provenance gates are materially present: _manifest_correlation_ok() requires a dict, ready is True, non-empty source without oos/locked, train/validation-only selection inputs, and no forbidden OOS keys in src/lumina_quant/strategies/artifact_portfolio_mode.py:334-345; _manifest_optimizer_ok() rejects missing/non-dict provenance, non-train/validation inputs, and forbidden OOS keys in src/lumina_quant/strategies/artifact_portfolio_mode.py:347-354; manifest and child callers enforce these gates in src/lumina_quant/strategies/artifact_portfolio_mode.py:378-389 and src/lumina_quant/strategies/artifact_portfolio_mode.py:459-472.
- Source artifact descriptor gates are present for path/id, sha, max-age, staleness, ready, and portfolio_ready in src/lumina_quant/strategies/artifact_portfolio_mode.py:298-332, but the implementation hashes any existing path before proving it is a regular readable file, and _manifest_definition_from_path() does not catch source validation exceptions after src/lumina_quant/strategies/artifact_portfolio_mode.py:392-397.
- Child safety gates are mostly present: child readiness, portfolio readiness, no-current-fold-OOS, train/validation optimizer marker, real-money flags, forbidden OOS keys, optimizer/correlation provenance, and source reconciliation are checked in src/lumina_quant/strategies/artifact_portfolio_mode.py:431-477.
- Gross cap now uses child gross exposure: leaf_gross is derived independently from weight and accumulated via gross += max(abs(weight), leaf_gross) in src/lumina_quant/strategies/artifact_portfolio_mode.py:478-497; unit coverage asserts leaf_gross breaches fail closed in tests/unit/test_artifact_portfolio_mode.py:1440-1458.
- Existing tests statically cover valid manifest resolution, OOS fail-close, source sha/missing sha, child optimizer/correlation provenance, malformed collection/positive-child shapes, positive malformed child rows, gross/leaf-gross breaches, default manifest support, and live-selection/backtest alias routing in tests/unit/test_artifact_portfolio_mode.py:1276-1472 and tests/unit/test_backtest_live_portfolio_mode_resolution.py:17-103. Per assignment, tests/linters/formatters were not run.

## Root Cause
The fail-closed design is implemented as explicit error-string returns for many expected validation failures, but not all validation operations are total/no-throw. Source artifact hashing/statting happens outside a broad fail-closed boundary, and child schema validation is ordered after the non-positive-weight pruning branch.

## Findings
1. HIGH — src/lumina_quant/strategies/artifact_portfolio_mode.py:311-317, src/lumina_quant/strategies/artifact_portfolio_mode.py:392-397 — Source artifact validation can raise or hang instead of fail-closing to cash. _validate_manifest_source_artifacts() checks exists() and then calls _file_sha256(source_path); _file_sha256() opens and streams the path. A manifest that points a source artifact at a directory, unreadable file, FIFO, or special device can raise or block before _manifest_definition_from_path() returns a cash definition, because only manifest JSON reading is inside the defensive try. Impact: malformed source artifacts violate the invalid-manifest fail-closed contract and can prevent live selection from resolving safely. Fix: require source_path.is_file() before hashing, reject non-regular paths with an explicit fail-closed reason, catch OSError/RuntimeError around hash/stat/freshness checks, and/or wrap the whole source-artifact validation call in a fail-closed exception boundary that preserves the reason.

2. MEDIUM — src/lumina_quant/strategies/artifact_portfolio_mode.py:478-512 — Malformed non-positive-weight child rows are skipped before schema validation. The code computes weight and continues when weight <= 0.0 before checking strategy_class, symbols, and params. A manifest can therefore contain one valid positive child plus one malformed zero/negative-weight child and still return live components, despite the product contract that invalid/malformed manifests fail closed to cash. Fix: validate child schema for every child object before the weight-pruning branch, or explicitly define and test that disabled child rows are allowed only when they still satisfy the manifest schema.

## Recommendations
1. Make source artifact validation total and fail-closed: check regular-file status before hashing, catch filesystem exceptions, and preserve diagnostic fail-close reasons in source_artifacts.
2. Move manifest child schema checks ahead of weight <= 0.0 pruning, then add tests for malformed zero-weight child rows alongside a valid positive child.
3. Add focused tests for non-regular/unreadable source artifact paths, stale source artifacts, source readiness flags, child source unreconciled, and child/root real-money flags; current tests cover many positive-weight and provenance cases but not these failure surfaces.

## Architectural Status
BLOCK

## Code Review Recommendation
REQUEST CHANGES

## Trade-offs
- Strict fail-close on any malformed child row: safer and matches the stated contract; may reject manifests containing disabled draft sleeves unless producers keep disabled rows schema-valid.
- Allow disabled malformed rows: more permissive for research artifacts, but it weakens the live manifest boundary and requires an explicit product exception plus tests.
- Source is_file() plus exception-to-cash returns: small implementation cost and strong live-safety behavior.
- Broadly catching all source-validation exceptions only at the caller: simpler but can hide which artifact failed unless the reason includes artifact id/type.
