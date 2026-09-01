# Test Spec: LuminaQuant dashboard approval refactor + Next migration under 8GB

## Verification goals
- Prove the exact-window parity slice is complete and stable.
- Prove migrated dashboard routes follow a consistent Python-contract/web-runtime pattern.
- Prove touched Streamlit/service seam extraction preserves behavior.
- Prove the hybrid launcher/docs contract remains correct.
- Keep all heavy verification sequential and under the 8GB cap.

## Expanded test plan
### Unit
- `apps/dashboard_web/tests/python-bridge.test.ts`
- any new exact-window web/unit tests added in this wave
- `tests/test_dashboard_exact_window_service.py`
- `tests/test_dashboard_bridge_contract.py`
- `tests/test_dashboard_cli.py`
- any new regression tests for the workflow-jobs section extraction seam
- `tests/test_exact_window_dashboard_loader.py` to preserve exact-window Streamlit compatibility
- `tests/test_dashboard_view_switch.py` for the touched app-level composition seam

### Integration
- `uv run pytest tests/test_dashboard_exact_window_service.py tests/test_dashboard_bridge_contract.py tests/test_dashboard_overview_service.py tests/test_dashboard_risk_health_service.py tests/test_dashboard_workflow_jobs_payload_service.py tests/test_dashboard_cli.py -q`
- touched Streamlit/dashboard tests for the extracted seam, explicitly including `tests/test_dashboard_view_switch.py`, and narrower new tests if the seam changes

### Web build/runtime
- `npm run test` in `apps/dashboard_web`
- `npm run lint` in `apps/dashboard_web`
- `npm run typecheck` in `apps/dashboard_web`
- `npm run build` in `apps/dashboard_web`

### E2E-style / route-manifest checks
- confirm Next build output includes `/`, `/workflows`, `/risk-health`, `/exact-window`, and `/api/python/dashboard/*` routes
- confirm launcher contract via `uv run python -m lumina_quant.cli.dashboard --mode next --print-contract` and existing CLI assertions
- confirm touched Streamlit route regression still supports `Dashboard View -> Exact-Window Suite` / workflow-jobs composition path via `tests/test_exact_window_dashboard_loader.py` and `tests/test_dashboard_view_switch.py`

### Broader repo safety
- `uv run ruff check src/lumina_quant/dashboard tests/test_dashboard_exact_window_service.py`
- if executable Python code beyond the dashboard contract layer changes, expand to repo-standard ruff scope
- `python3 -m py_compile` on new/changed Python files
- additional repo-wide checks only after targeted gates are green

### Observability / operational verification
- verify route availability in Next build output (`/`, `/workflows`, `/risk-health`, `/exact-window`, and corresponding `/api/python/dashboard/*` routes)
- verify launcher contract output via `uv run python -m lumina_quant.cli.dashboard --mode next --print-contract` and preserve default/auto Streamlit behavior via existing CLI tests
- verify docs/readme changed text matches actual hybrid migration behavior

## Failure gates
- Missing or broken exact-window route/API/service = fail
- Broken read-only route-contract consistency across touched migrated surfaces = fail
- Launcher contract regression, including auto/default no longer resolving to Streamlit, = fail
- `tests/test_dashboard_cli.py` regression = fail
- `tests/test_exact_window_dashboard_loader.py` regression = fail
- `tests/test_dashboard_view_switch.py` regression on the touched workflow-jobs composition seam = fail
- Any heavy verification overlap or memory-policy violation = fail
- Any post-cleanup regression = fail

## Execution order
1. Run narrow targeted tests on the current WIP exact-window slice.
2. After each refactor slice, rerun only the affected narrow Python/web checks.
3. When feature/refactor changes settle, run sequential web lint/typecheck/build/test.
4. Run sequential targeted Python integration/regression suites.
5. If broader executable paths changed materially, run broader repo checks sequentially.
6. Run deslop cleanup on changed files only.
7. Re-run required targeted web/Python gates after cleanup.
8. Perform final cold approval reassessment.

## Memory-safe execution notes
- Maximum team worker count: 1.
- No concurrent `npm run build` with full `pytest`.
- No concurrent broad repo checks.
- Leader owns all heavy verification sequentially.
- Prefer targeted suites first to avoid unnecessary memory load.

## Approval evidence checklist
- [ ] Exact-window route/API/service/runtime slice complete
- [ ] Read-only route-contract pattern across touched migrated surfaces is consistent
- [ ] Workflow-jobs Streamlit seam extraction landed with preserved behavior
- [ ] Dashboard CLI/launcher contract still passes, including default/auto Streamlit behavior (`tests/test_dashboard_cli.py`)
- [ ] Web tests/lint/typecheck/build pass
- [ ] Exact-window Streamlit compatibility still passes (`tests/test_exact_window_dashboard_loader.py`)
- [ ] Workflow-jobs/app composition regression still passes (`tests/test_dashboard_view_switch.py`)
- [ ] Targeted Python dashboard contract suites pass
- [ ] Post-deslop reruns pass
- [ ] Final cold review can justify APPROVE
