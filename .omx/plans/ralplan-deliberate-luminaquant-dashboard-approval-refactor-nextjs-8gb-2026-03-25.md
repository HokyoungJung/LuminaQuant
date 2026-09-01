# RALPLAN-DR Deliberate Plan: LuminaQuant dashboard approval refactor + Next migration under 8GB (2026-03-25)

## Principles
1. Preserve current dashboard behavior until parity is explicitly verified.
2. Prefer vertical migration slices plus service extraction over a broad flag-day rewrite.
3. Put explicit Python payload contracts in front of every touched Next surface.
4. Keep the active execution posture <=8GB by limiting team concurrency and running heavy verification sequentially.
5. Refactor only where behavior is protected by tests or immediately protected before edits.

## Decision drivers
1. The user explicitly wants broad refactoring plus continued React/Next migration, but behavior must be preserved.
2. `apps/dashboard/app.py` remains a hotspot, so migration without seam cleanup is unlikely to feel approval-grade.
3. Hardware is memory-constrained, so execution must avoid multi-worker or concurrent-heavy verification patterns.

## Viable options
| Option | Summary | Pros | Cons |
| --- | --- | --- | --- |
| A | Whole-dashboard rewrite now: aggressively decompose Streamlit, expand Next parity, and flip launch defaults quickly | Highest architectural cleanup potential | Too risky for behavior preservation; high OOM/regression risk; unlikely to stay approval-safe in one wave |
| B | Dashboard-focused vertical wave: finish exact-window parity, consolidate Python dashboard payload services, extract one more safe `app.py` seam, preserve hybrid launch contract | Best approval/risk balance; concrete user-visible progress; compatible with <=8GB execution | Streamlit remains temporarily; some monolith debt survives for later waves |
| C | Refactor Streamlit internals only and defer further Next work | Lowest short-term UI churn | Conflicts with the user's Next migration direction; weaker evidence that the migration path is real |

## Recommended decision
Choose **Option B**.

## Invalidated alternatives
- **Reject A**: architectural ambition is too high for one low-memory wave and endangers behavior preservation.
- **Reject C**: fails the explicit React/Next direction and would weaken the migration proof.

## Deliberate pre-mortem
1. **Exact-window parity lands, but payload/schema drift breaks later builds**
   - Mitigation: keep Python payload services explicit and covered by unit tests; prefer narrow JSON summaries over raw artifact passthrough.
2. **`app.py` extraction changes Streamlit behavior subtly**
   - Mitigation: lock touched paths with targeted dashboard tests before extraction; preserve wrapper call points.
3. **Team execution causes OOM or noisy parallel verification**
   - Mitigation: use only `omx team 1:executor`, keep full verification on the leader/Ralph lane, and run heavy checks sequentially.

## 5-step execution plan
### Step 1 — Behavior lock + plan gate
- Adopt the current exact-window WIP as the first active slice.
- Ensure the new PRD/test-spec files above are the canonical execution gate for this wave.
- Run/extend targeted dashboard tests so touched parity and extraction paths are behavior-locked before wider edits.

**Acceptance criteria**
- PRD + test-spec files exist at the new canonical paths.
- Current exact-window WIP has targeted tests covering payload/service + frontend contract assumptions.
- Execution scope is explicitly limited to dashboard migration/refactor hotspots.

### Step 2 — Consolidate Python dashboard payload layer
- Finish and normalize `src/lumina_quant/dashboard/` so touched Next routes (`overview`, `workflow-jobs`, `risk-health`, `exact-window`) all follow explicit, testable service patterns.
- Remove avoidable duplication in server bridge adapters where possible without changing behavior.

**Acceptance criteria**
- Touched payload modules expose clear JSON-oriented service functions.
- Frontend/server bridge code is thinner and more regular on the touched path.
- Targeted Python + frontend tests pass after this slice.

### Step 3 — Safe Streamlit seam extraction
- Extract one more safe dashboard state/query responsibility from `apps/dashboard/app.py` into service code on the touched path.
- Preserve public call points with thin wrappers/imports so Streamlit behavior stays stable.

**Acceptance criteria**
- `app.py` is thinner on at least one meaningful touched seam.
- Empty-result vs failure behavior is explicit and regression-tested.
- No loss of current Streamlit functionality on migrated/non-migrated surfaces.

### Step 4 — Hybrid launch/docs hardening
- Reconfirm `src/lumina_quant/cli/dashboard.py` hybrid launch behavior and related docs.
- Ensure the Next route inventory and fallback story are accurately documented.

**Acceptance criteria**
- CLI/launch tests pass.
- Dashboard docs/readme accurately describe the hybrid period.
- Build output and route inventory show the expected Next/API surfaces.

### Step 5 — Team execute, Ralph verify/fix, deslop, approve
- Launch one executor worker via `omx team` for the main implementation lane.
- Keep leader-owned sequential verification, deslop, and final reassessment in the Ralph lane.
- Iterate on any failing verification until all targeted and repo-wide gates are green and the wave is approval-ready.

**Acceptance criteria**
- Team run reaches terminal clean completion with evidence.
- Sequential frontend + Python + repo-wide verification passes.
- ai-slop-cleaner pass on changed files succeeds and post-deslop reruns remain green.
- Final reassessment can defend APPROVE.

## Available-agent-types roster
- `planner`
- `architect`
- `critic`
- `executor`
- `test-engineer`
- `verifier`
- `code-simplifier`
- `build-fixer`

## Follow-up staffing guidance
### Team lane
- **Headcount:** 1 worker total
- **Role:** `executor`
- **Reasoning:** high
- **Responsibility:** implement the approved dashboard migration/refactor slice under the new PRD/test-spec, commit progress incrementally, and run only lane-local targeted checks.
- **Launch hint:** `omx team 1:executor "implement approved LuminaQuant dashboard approval refactor + Next migration wave under 8GB; own touched dashboard_web + src/lumina_quant/dashboard + safe app.py seam extraction; run only local targeted checks"`

### Ralph follow-up lane
- **Owner:** leader / Ralph
- **Reasoning:** high
- **Responsibility:** integrate/fix any remaining issues, run sequential heavy verification, perform deslop on changed files, rerun regressions, and close with approval reassessment.
- **Launch hint:** `omx ralph --no-deslop "final verification/fix loop for approved LuminaQuant dashboard approval refactor + Next migration wave under 8GB"`
  - Then run the mandatory deslop + post-deslop reruns locally in the leader lane if the standard Ralph wrapper is not used directly.

### Team verification path
1. Team worker runs only targeted local checks for its touched files.
2. Leader inspects `omx team status <team>` and mailbox evidence until terminal completion.
3. Leader shuts the team down only after `pending=0`, `in_progress=0`, `failed=0`.
4. Ralph/leader runs sequential frontend verification.
5. Ralph/leader runs targeted Python verification.
6. Ralph/leader runs repo-wide static/regression gates.
7. Ralph/leader runs deslop on changed files and repeats required checks.
8. Architect-style cold reassessment decides APPROVE / ITERATE.

## ADR
### Decision
Execute a dashboard-focused vertical migration/refactor wave: stabilize exact-window parity, consolidate Python dashboard payload services, extract one additional safe `app.py` seam, preserve the hybrid launch contract, and verify sequentially under the 8GB cap.

### Drivers
- Broad refactor is requested, but behavior preservation is mandatory.
- Approval risk currently centers on dashboard migration seams and remaining monolith pressure.
- The hardware cap punishes multi-lane heavy execution.

### Alternatives considered
- Whole-dashboard rewrite/cutover now.
- Streamlit-only cleanup with minimal further Next progress.

### Why chosen
This option maximizes approval-relevant architectural improvement while staying within memory, risk, and verification constraints.

### Consequences
- Streamlit remains part of the hybrid dashboard story for now.
- Some deeper monolith debt is intentionally deferred to later waves.
- The repo gets a clearer dashboard service boundary and stronger migration proof.

### Follow-ups
- Later waves can continue with additional Streamlit seam extraction after each parity slice is green.
- Default launch mode should change only after a larger parity checklist is fully proven.
- If approval is still blocked after this wave, the next target should be the highest-friction remaining `app.py` seam or launch-contract inconsistency identified by final review.
