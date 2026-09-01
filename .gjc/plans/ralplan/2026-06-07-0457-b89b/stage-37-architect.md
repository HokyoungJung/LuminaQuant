## Summary
G004 is still BLOCKED. The manifest path is present beside aliases and the happy-path plus several fail-closed checks are covered, but malformed manifest and child shapes can still escape the cash-only boundary or be silently coerced into live components.

## Analysis
Reviewed only the requested files:
- `/home/hoky/Quants-agent/LuminaQuant/src/lumina_quant/strategies/artifact_portfolio_mode.py`
- `/home/hoky/Quants-agent/LuminaQuant/src/lumina_quant/live_selection.py`
- `/home/hoky/Quants-agent/LuminaQuant/tests/unit/test_artifact_portfolio_mode.py`
- `/home/hoky/Quants-agent/LuminaQuant/tests/unit/test_backtest_live_portfolio_mode_resolution.py`

Positive evidence: `live_selection.py` includes `artifact_manifest_mode` and accepts `manifest:` references; `resolve_portfolio_mode_definition` routes `artifact_manifest_mode` and `manifest:` into `_manifest_definition_from_path`; manifest source artifacts require path, sha256, freshness, ready and portfolio_ready; manifest and child real-money flags are vetoed; optimizer and correlation provenance are required; OOS flags are vetoed; source IDs must reconcile; `leaf_gross` contributes to portfolio gross via `max(abs(weight), leaf_gross)`.

Blocking evidence: after JSON read, the manifest parser still trusts multiple manifest-controlled shapes. `list(...)` over scalar `source_artifacts`, `children`, or optimizer `selection_inputs` can raise outside the fail-closed boundary. Child collection handling also filters out non-object children and `_component_from_row` accepts malformed symbol shapes or raises unhandled conversion errors.

## Root Cause
The fail-closed boundary is implemented around file readability and selected semantic checks, but not around full manifest schema validation. The loader still relies on permissive Python coercions and filtering instead of explicit schema checks for externally supplied manifest fields.

## Findings
1. Severity: HIGH. File/reference: `artifact_portfolio_mode.py:287-407`. Impact: malformed manifest collection fields can raise `TypeError` instead of returning cash-only fail-close, violating invalid-manifest fail-closed routing. Fix suggestion: explicitly require `source_artifacts`, `children`, and optimizer `selection_inputs` to be list or tuple before iterating, or add a narrow post-read fail-closed boundary that converts expected shape errors into `manifest_fail_closed_to_cash` with diagnostic reasons.

2. Severity: HIGH. File/reference: `artifact_portfolio_mode.py:407-515`. Impact: malformed child entries can be silently dropped or coerced, allowing a manifest with a non-object child plus a valid child to still emit live components; string `symbols` can become per-character symbols, empty symbols are accepted, and non-mapping `params` can crash outside the `except ValueError` block. Fix suggestion: require every child entry to be an object, validate `strategy_class`, `symbols`, and `params` shapes before constructing a component, and catch expected `TypeError` and `ValueError` as `child_invalid:<id>`.

3. Severity: MEDIUM. File/reference: `artifact_portfolio_mode.py:415-418`. Impact: child `portfolio_ready` is not mandatory; missing, null, or string values pass as long as `ready is True`, unlike source artifacts which require exact true readiness. Fix suggestion: require `child.get("portfolio_ready") is True` or document the optional semantics and add tests.

## Recommendations
1. Add strict manifest schema guards before semantic validation: manifest top-level object fields, source artifact list entries, children list entries, child symbols list, params mapping, and provenance selection input lists.
2. Expand the malformed-child tests in `test_artifact_portfolio_mode.py` beyond empty `strategy_class`: non-object child mixed with valid child, scalar `children`, scalar `source_artifacts`, scalar selection inputs, string symbols, empty symbols, and non-mapping params.
3. Align child readiness with source-artifact readiness unless optional child `portfolio_ready` is intentional.

## Architectural Status
BLOCK

## Code Review Recommendation
REQUEST CHANGES

## Trade-offs
- Explicit schema validation: more code and more negative tests, but deterministic fail-closed behavior and clear diagnostics.
- Broad catch-all around manifest parsing: smaller patch, but can hide programming errors and weaken diagnostics.
- Current permissive coercion: concise, but unsafe for live-routing manifests and not compliant with fail-closed requirements.
