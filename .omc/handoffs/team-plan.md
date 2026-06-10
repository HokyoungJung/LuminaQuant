# Handoff: team-plan/team-prd → team-exec

- **Decided**: Execute approved consensus plan `.omc/plans/quants-agent-overhaul-consensus.md` (Status: APPROVED, Critic round-2). Option A strangler refactor with Phase 2 as clean-room compute island. Phase order: 0 (baseline) → 0.5 (Stack Lift: py>=3.14 + latest deps, local full-suite hard gate) → 1–7, gated sequentially. Tolerance contract: rtol 1e-8 default, divergences need docs/divergences/*.md. Perf gates are report-and-justify; correctness gates are hard-blocking. AGENTS.md (not AGENT.md). Team: 3 executors, tasks #1–#11 (phase-aligned), deps enforce gates.
- **Rejected**: Option B clean-room full rewrite (oracle runs too late; retained as documented fallback, trigger: >40% of Phase 1 fighting old seams). Binance Spot, always-on L2 collection (non-goals).
- **Risks**: 3.14 compat for ta-lib/psycopg/cudf-polars/optuna/numba (blocking in Phase 0.5, fallback = highest compatible pin + docs/divergences/stack-lift.md); empty var/data → tick fixture must be captured or first-run-is-golden documented; alpha_zoo/raw_first are LIVE code (61/23 refs) — per-deletion grep+import-graph+green-suite trio required.
- **Files**: Plan + spec + open-questions under .omc/plans/ and .omc/specs/. Baselines will land in /home/hoky/Quants-agent/baseline/.
- **Remaining**: All execution (tasks #1–#11), then team-verify with verifier + code-reviewer + security-reviewer (live-trading touches real money).
- **Git policy**: user approved commits/push. Workers commit phase-aligned on branch `refactor/overhaul` after each phase gate passes; push happens in Phase 7 (#11).

## Phase 4 Finding — walk-forward silent -999 sentinels (logged by worker-2, Phase 0b)

**Finding**: The legacy walk-forward harness (`src/lumina_quant/optimization/walkers.py` +
`evaluate_metrics_backend`) silently emits `-999.0` as a sentinel Sharpe value when a
val/test window is shorter than the MA warmup required by the selected `long_window`
parameter (e.g. `long_window=120` in a 90-day window).  No exception, no warning, no
log line — the -999 propagates into golden output without any indication that the
metrics are degenerate.

**Evidence**: `baseline/golden/walk_forward_results.json` (variant A, commit `a111e29`),
folds 1 & 3 val/test metrics = -999.0. Fold 2 (best `lw=80`) is unaffected because
80 < 90-day window.

**Correct oracle**: `baseline/golden/walk_forward_results_warmup.json` (variant B) shows
real metrics for all folds when 120-bar warmup context is prepended before each
val/test window.

**Phase 4 MUST-DO**: The new walk-forward implementation (`Phase 4: Backtest & validation`)
MUST raise a loud error (exception or structured error result, never -999) when an
eval window is too short to warm up the requested indicator.  The -999 path in
`evaluate_metrics_backend` is the exact class of silent-failure the overhaul exists to
eliminate.  Divergence from variant A folds 1 & 3 val/test is a **documented improvement
divergence** — see `docs/divergences/` procedure; no rtol 1e-8 obligation on those
cells.  Divergence from variant B is a **correctness regression** and hard-blocks Phase 4
gate.
