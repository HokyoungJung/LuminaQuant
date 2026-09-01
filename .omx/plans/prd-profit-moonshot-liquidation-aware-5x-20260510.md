# PRD — Profit moonshot liquidation-aware 5x validation — 2026-05-10

## Decision intent
Determine whether the current-base sleeve tuple at integer `5x` leverage is a deployable improvement versus the preserved current base `2.3427334297703024x`, after conservative Binance USDT perpetual-style liquidation validation.

## RALPLAN-DR summary
### Principles
1. Preserve train/validation-only selection; locked-OOS is report-only/gate-only.
2. Treat liquidation safety as a hard deployability gate, not a cosmetic metric.
3. Use conservative fallback margin assumptions when exact exchange tier data is unavailable.
4. Keep generated evidence reproducible under the global `<8 GiB` memory guard.
5. Prefer a bounded current-base validation path over broad candidate search.

### Decision drivers
1. Whether `5x` has zero liquidation events and positive margin buffers across train/validation/OOS.
2. Whether `5x` improves OOS return and return/MDD versus current base without violating MDD/risk-quality gates.
3. Whether selection remains train/validation-only even if OOS is reported as a final gate.

### Viable options
- Option A — Add a dedicated liquidation-aware current-base validation script. Pros: bounded, reproducible, avoids destabilizing broad tuner. Cons: separate artifact path must be wired/documented.
- Option B — Embed liquidation simulation inside the full portfolio tuner. Pros: single pipeline. Cons: higher memory/runtime and broader regression surface.

Chosen: Option A plus small shared gate additions in tuner/validator where appropriate. Option B is rejected for this session because the requirement is to validate a known tuple and preserve a green baseline with minimal churn.

## Scope
- Add focused unit tests first for liquidation threshold breach, split liquidation/margin recording, OOS selection isolation, and promotion blocking on liquidation/non-positive buffer.
- Implement a conservative liquidation-aware replay for the current-base sleeve tuple over integer leverages `1..6`, highlighting forced `5x` and the preserved current-base leverage.
- Record margin model assumptions and source references in artifacts.
- Write JSON/Markdown reports and session handoff.
- Run targeted tests, full pytest, ruff, compileall, and `git diff --check`.
- Commit with Lore protocol, push to `private/main`, and verify GitHub Actions `ci` and `private-ci` green.

## Out of scope
- Authenticated Binance account tier lookup.
- Live deployment or live-order configuration changes.
- Broad alpha search beyond the current-base tuple and integer leverage grid.

## Acceptance criteria
- `liquidation_count == 0` for train/validation/OOS on any promoted candidate.
- `minimum_margin_buffer > 0` and `minimum_margin_ratio > 0` for all splits.
- OOS MDD `<= 25%`.
- OOS return and OOS return/MDD beat current base.
- Sharpe, Sortino, smart Sortino, and Calmar are recorded and acceptable.
- Peak RSS evidence remains below `8 GiB`.
- Locked-OOS fields are explicitly report-only/gate-only and never used in leverage selection.

## ADR
Decision: implement a dedicated liquidation-aware validation artifact lane for the current-base tuple.
Drivers: safety-first liquidation gate, train/validation selection integrity, minimal change risk.
Alternatives considered: full tuner integration; rejected due runtime and diff risk.
Consequences: deployability evidence lives under `liquidation_aware_*`; future broad search can reuse the helpers.
Follow-ups: if `5x` passes, decide separately whether to update mission result/promoted candidate; if it fails, retain current base or select the best train/validation-safe integer leverage.

## Available agent-types roster and staffing guidance
- `executor`: implementation lane for script/tests/docs.
- `test-engineer`: evidence/regression lane for test shape and command execution.
- `architect`: final sign-off lane for margin-model and selection-boundary review.
- `explore`: quick codebase/file lookup only.

Team launch hint: `omx team 2:executor "profit moonshot liquidation-aware validation: one read-only model/test review lane and one artifact/report review lane; do not edit without leader assignment"`.
Ralph follow-up: single-owner completion loop owns final integration, verification, Lore commit, push, and CI polling.
Goal-mode suggestion: `$performance-goal` would be the best durable follow-up if later optimizing margin-model runtime or replay speed.
