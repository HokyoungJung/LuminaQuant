# PRD: LuminaQuant dashboard approval refactor + Next migration under 8GB

## Problem statement
The LuminaQuant dashboard migration is real but still structurally split across a large Streamlit monolith (`apps/dashboard/app.py`), emerging Python contract services (`src/lumina_quant/dashboard/*.py`), and an unfinished Next.js frontend (`apps/dashboard_web/`). The current state proves the migration direction, but approval risk remains because functionality is still spread across weakly unified seams, route parity is partial, and some newly introduced parity slices are implemented as one-off bridges rather than a cleanly modular contract family. The user explicitly wants a broad refactor wave that preserves all working behavior, strengthens OOP/service boundaries, improves file/module organization, fixes bugs, and finishes with an APPROVE-grade assessment while keeping the total execution posture under 8GB.

## Desired outcome
- Preserve all currently working dashboard and CLI behavior while continuing the React/Next migration.
- Keep `auto`/default launch behavior on the Streamlit path during this wave; Next remains an explicit migration mode, not the new default.
- Land another approval-grade refactor wave focused on dashboard migration seams, code splitting, and explicit service/module boundaries.
- Consolidate the Python-backed dashboard contract layer so migrated routes follow the same pattern instead of bespoke implementations.
- Keep the execution/verification posture memory-safe: <=1 active team worker plus leader-owned sequential heavy verification, always below the effective 8GB system cap.
- Finish with strong verification evidence and a cold reassessment that can credibly return APPROVE.

## Scope
### In scope
1. Complete and harden the exact-window parity slice already started in the working tree.
2. Refactor the Next dashboard bridge/server/runtime layer into a more modular, **read-only route-contract convention** across overview, workflow jobs read path, risk-health, and exact-window, while keeping mutating workflow control paths separate and explicitly out of the shared abstraction scope.
3. Tighten the dashboard launch contract so `uv run lq dashboard` remains explicit, tested, migration-aware, and default-Streamlit in this wave.
4. Extract exactly one additional safe responsibility seam out of `apps/dashboard/app.py`: the workflow-jobs section rendering/composition path around `_render_workflow_jobs_section`, preserving public behavior through thin wrappers/helpers.
5. Add/refresh targeted tests and docs for the migration/refactor wave.
6. Run sequential verification, cleanup/deslop, and final cold approval reassessment.

### Out of scope
- Full replacement of every Streamlit dashboard panel in one cutover.
- Deleting Streamlit or removing compatibility fallback in this wave.
- New dependencies or a broad repo-wide rewrite outside the dashboard/migration/refactor hotspots.
- Concurrent heavy verification or >1 active team worker.

## Evidence and current facts
- Current repo branch: `private-main`.
- Existing migration artifacts already exist:
  - `.omx/plans/prd-dashboard-react-nextjs-migration.md`
  - `.omx/plans/test-spec-dashboard-react-nextjs-migration.md`
- Existing approval refactor artifacts already exist:
  - `.omx/plans/prd-luminaquant-full-refactor-approve-8gb.md`
  - `.omx/plans/test-spec-luminaquant-full-refactor-approve-8gb.md`
- Launcher contract already supports Streamlit/Next switching in `src/lumina_quant/cli/dashboard.py` and tests cover launch-mode env + cwd behavior in `tests/test_dashboard_cli.py`.
- The Streamlit monolith remains large (`apps/dashboard/app.py` ~3830 LOC), while service extraction has begun in `apps/dashboard/services/` and Python contract services now exist in `src/lumina_quant/dashboard/`.
- Next dashboard surfaces exist for overview, workflow jobs, risk-health, and a new exact-window route WIP under `apps/dashboard_web/`.
- Memory policy remains binding: overall execution must stay below 8GB, with no concurrent heavy verification.

## User stories
### US-001 Unified contract-backed migration slices
As a maintainer, I want all migrated dashboard routes to use a consistent Python-contract/server/runtime pattern so future slices are incremental and maintainable instead of bespoke.

**Acceptance criteria**
- Exact-window parity follows the same bridge/service pattern as other routes.
- Shared route-contract conventions are explicit in touched web/Python bridge files.
- Targeted tests cover route availability and payload normalization.

### US-002 Streamlit monolith pressure reduction on touched paths
As a maintainer, I want the touched migration-related logic extracted from `apps/dashboard/app.py` into focused services so the monolith shrinks along active parity boundaries.

**Acceptance criteria**
- At least one touched dashboard responsibility cluster becomes thinner in `app.py` and clearer in service code.
- Existing public behavior and call points remain preserved.
- Regression tests protect the touched seam.

### US-003 Safe migration-aware launcher contract
As an operator, I want `uv run lq dashboard` to remain explicit and safe during migration so I can still launch the dashboard with either Streamlit or Next mode without ambiguity.

**Acceptance criteria**
- Launcher contract remains tested and docs stay accurate.
- `auto`/default mode remains on the Streamlit path in this wave.
- Next mode remains isolated to `apps/dashboard_web` while Streamlit compatibility still works.
- No regression in current CLI behavior.

### US-004 Approval-grade, 8GB-safe verification
As the repo owner, I want the refactor and migration wave verified under the 8GB policy with clear evidence so the work is approval-worthy on constrained hardware.

**Acceptance criteria**
- <=1 active team worker plus leader verification lane.
- No concurrent heavy verification.
- Targeted web/Python checks pass before full regression.
- Final reassessment can cite green evidence and improved structure.

## RALPLAN-DR summary
### Principles
1. Preserve working behavior first; migrate behind explicit contracts, not rewrites.
2. Prefer vertical slice completion plus seam cleanup over broad speculative churn.
3. Keep memory safety as a first-class execution constraint.
4. Unify patterns where they already exist instead of adding new one-off abstractions.
5. Approval requires evidence, diagnosability, and maintainable boundaries—not just green tests.

### Decision drivers
1. Highest approval leverage per changed line.
2. Lowest regression risk under 8GB-constrained execution.
3. Clearer long-term migration architecture for future parity slices.

### Viable options
#### Option A — Vertical-slice completion + contract unification (recommended)
- **Pros:** Builds directly on current WIP; preserves momentum; adds real user-visible parity; keeps scope bounded; high approval signal for migration architecture.
- **Cons:** Leaves some broader Streamlit monolith debt for later waves; may require careful seam selection to avoid under-refactoring.

#### Option B — Streamlit-first service extraction before more Next parity
- **Pros:** Shrinks the monolith earlier; may simplify later parity work; strong structural cleanup story.
- **Cons:** Delays migration-visible progress; broader risk surface in `app.py`; easier to drift into a large refactor without approval-grade deliverables.

### Recommended decision
Use **Option A**: finish/harden the current exact-window parity slice and unify the Python/Next dashboard contract pattern, while taking one additional safe extraction/refactor seam from the touched dashboard migration paths.

### Deliberate pre-mortem
1. **Failure scenario: parity slice lands but remains bespoke.**
   - Risk: another isolated bridge increases maintenance debt instead of reducing it.
   - Mitigation: require a shared contract/service pattern across all migrated routes touched in this wave.
2. **Failure scenario: app.py extraction destabilizes Streamlit behavior.**
   - Risk: approval fails because the refactor changes behavior in a fragile monolith.
   - Mitigation: restrict extraction to migration-touched seams with regression tests before broad movement.
3. **Failure scenario: memory-safe plan is violated during verification.**
   - Risk: tmux worker/build/test pressure causes OOM and invalidates the workflow.
   - Mitigation: one worker only, sequential heavy verification, no overlapping full pytest/build/benchmark runs.

## ADR
- **Decision:** Continue the dashboard migration with an approval-focused vertical slice: exact-window parity completion + contract pattern unification + one safe Streamlit/dashboard seam extraction, all verified under the 8GB policy.
- **Drivers:** approval leverage, regression safety, memory safety, migration maintainability.
- **Alternatives considered:**
  - broader Streamlit-first decomposition wave
  - stop at the exact-window slice only without broader pattern cleanup
- **Why chosen:** it balances visible migration progress with structural cleanup, while keeping scope controllable and evidence strong enough for approval.
- **Consequences:** some large-monolith debt remains for future waves, but the migration pattern becomes clearer and safer.
- **Follow-ups:** additional panel parity waves, deeper `app.py` decomposition, eventual primary-entrypoint switch when parity and operational confidence are sufficient.

## Execution plan
1. **Lock the touched baseline and complete the current exact-window slice**
   - Confirm the current exact-window WIP compiles/tests cleanly.
   - Fill any missing route/API/runtime/docs gaps so the slice is complete and not half-integrated.
   - Acceptance: exact-window web route/API/service/tests all exist and pass targeted checks.

2. **Unify the read-only dashboard contract pattern across migrated routes**
   - Normalize **read-only** bridge/server/runtime structures across these touched files only: `apps/dashboard_web/lib/python-bridge.ts`, `apps/dashboard_web/lib/python-bridge-server.ts`, `apps/dashboard_web/lib/workflow-jobs-server.ts`, `apps/dashboard_web/lib/risk-health-server.ts`, and `apps/dashboard_web/lib/exact-window-server.ts`.
   - Keep payload producers route-local on the Python side (`src/lumina_quant/dashboard/overview_service.py`, `workflow_jobs_service.py` read path, `risk_health_service.py`, `exact_window_service.py`) unless a tiny shared helper clearly reduces duplication without widening scope.
   - Treat `apps/dashboard_web/app/api/python/dashboard/workflow-jobs/control/route.ts` and Python `control_workflow_job` mutation logic as explicitly **out of Step 2 commonization scope**.
   - Acceptance: touched migrated read paths follow a consistent contract pattern and targeted web tests remain green.

3. **Extract one named Streamlit/service seam: workflow-jobs section rendering/composition**
   - Thin `apps/dashboard/app.py` by moving `_render_workflow_jobs_section` and its tightly related workflow-jobs composition helpers into `apps/dashboard/services/workflow_jobs.py`, leaving thin wrappers/import-level compatibility in `app.py`.
   - Do not widen this extraction to unrelated workflow control, market, or risk tabs.
   - Acceptance: the workflow-jobs section path is thinner in `app.py`, behavior is preserved, and regression tests cover the seam.

4. **Tighten launcher/docs/test alignment for hybrid migration mode**
   - Keep `uv run lq dashboard` explicit and tested for Streamlit vs Next mode.
   - Preserve `auto`/default launch behavior on Streamlit in this wave.
   - Refresh docs to reflect the real hybrid migration state rather than a placeholder state.
   - Acceptance: launcher tests pass and touched docs accurately describe current routes/contracts.

5. **Run memory-safe sequential verification and cleanup**
   - Run targeted Python/web tests after each slice.
   - Run lint/typecheck/build/targeted pytest, then broader repo checks sequentially.
   - Run deslop cleanup only on changed files, then re-run required gates.
   - Acceptance: all touched gates green, no memory-policy violation, and cold reassessment can credibly return APPROVE.

## Follow-up staffing guidance
### Available-agent-types roster
- `planner`
- `architect`
- `critic`
- `executor`
- `verifier`
- `code-simplifier`
- `test-engineer`

### Recommended team staffing
- **Team:** `1:executor`
  - Reasoning: `medium` to `high`
  - Role: implement the approved dashboard migration/refactor slice in one controlled lane
  - Why only 1 worker: minimizes RAM pressure and merge churn under the 8GB cap
- **Leader lane:** local orchestration + targeted inspection + integration review
- **Ralph follow-up lane:** sequential verification/fix/cleanup loop after team terminal completion
  - Roles used: `architect` for approval review, `code-simplifier` for deslop/cleanup review if needed, leader-owned verification

### Launch hints
- Team execution hint:
  - `omx team 1:executor "Implement the approved LuminaQuant dashboard approval refactor + Next migration wave under 8GB; complete exact-window parity, unify dashboard bridge patterns, extract one safe app.py seam, preserve launcher behavior, and leave verification notes."`
- Ralph follow-up hint:
  - `omx ralph "Verify/fix/clean the approved LuminaQuant dashboard approval refactor + Next migration wave under 8GB"`

### Team verification path
1. Worker completes code + commits/checkpoints.
2. Leader reviews resulting diffs and runs narrow targeted tests.
3. Leader performs sequential heavy verification (lint/typecheck/build/full targeted Python suites, then broader gates if required).
4. Architect performs cold review for approval.
5. Ralph-style cleanup/deslop + post-cleanup reruns close the wave.
