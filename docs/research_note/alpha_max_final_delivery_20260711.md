# Alpha-Max Revision 5.14 — Final Local Delivery (2026-07-11)

## Status and authority

This document supersedes the implementation-status sections of
`alpha_max_independent_checkpoint_20260711.md`. The approved experiment design,
frozen configuration, trial inventory, causal rules, gates, and no-claim policy
remain unchanged.

The repository implementation and local integrity controls are complete enough
to hand to the data-bearing PC. The actual market-data replay has **not** been
run on this machine. Consequently this delivery makes no alpha, return,
robustness, confirmation, promotion, or capital-allocation claim.

## Repository update decision

The working branch is `feat/alpha-max-20260710`. The latest fetched
`private/main` (`49bdd52a` at the local delivery audit) is an ancestor of the
branch and was intentionally retained because the operator explicitly required
the repository to be updated before continuation.

That repository update does not redefine the frozen experiment identity:

- the prior trial family is still read from the exact immutable
  `252910e54e280cc593365484cbc99d6ca87893f9` Git blob;
- the current 21-node registry and incumbent-resolution audit are embedded in
  the frozen config and hash checked;
- runtime code does not read `.omx`, `.omc`, report-latest aliases, profiles,
  YAML, or ambient `LQ_*` configuration;
- newer repository history is not admitted as a trial, selector input, or
  historical-evaluation input.

Thus the user-directed source update supersedes the earlier clean-ancestry
handoff constraint, while the experiment's trial/data/runtime isolation remains
fail-closed and tested.

## Completed implementation

The final tree provides:

1. **Strict causal data binding** — descriptor-bound config, manifest, raw and
   feature inventories; exact adjacent feature roots; raw-first access; causal
   funding points; exact admitted-symbol ownership; no ambient fallback.
2. **Native-clock adapters** — completed 1d trend, atomic full-cross-section 1d
   near-high, and completed 4h carry. The near-high aggregator collects every
   admitted symbol for a key before closing or declaring a true expired
   omission.
3. **Two-engine phase boundary** — indicator-only warmup, atomic native
   finalization, deterministic indicator capsules, fresh flat economic scorer,
   and no warmup cash/order/fill/funding leakage.
4. **Exact native observability** — every actual-run receipt seals adapter class,
   native timeframe, admitted-symbol completed keys/counts/last keys, atomic
   barrier keys and symbol coverage, and the exact completed/barrier keys added
   by finalization. Missing, regressed, failed, partial, unbound, or mismatched
   coverage fails before scoring.
5. **Execution economics** — immutable positive-fill pricing traces, applied-fill
   attribution, separate no-fill attempts, causal funding settlement,
   liquidation evidence, full-event equity/MDD/ruin tracking, and exact
   reconciliation.
6. **Frozen experiment orchestration** — 21 matrix rows, 17 executable rows,
   10/15/20/30 bps cells, common seeds, exact train admission, allocation and
   capped rounding, validation-only gates/ranking, immutable prelock champion,
   report-only exposed historical leader, and singular terminal precedence.
7. **Physical process separation** — distinct prelock and one-touch historical
   CLIs, absent-output ownership, `SEALED.json` written last, immutable inventory
   checks, duplicate-completion refusal, and historical inability to mutate the
   prelock tree.
8. **Independent observability export** — descriptor-bound re-read of every
   sealed artifact with inventory and historical/prelock binding checks; NaN,
   symlink, atomic replacement, post-verification mutation, unsealed-extra, and
   unrelated-root attacks fail closed.
9. **Legacy-neutral seams** — ordinary/non-Alpha execution, funding, resolver,
   manifest ordering, and portfolio behavior remain unchanged when the new
   strict inputs are absent.

## Local verification boundary

Local tests use deterministic replay stubs for the expensive 680/816 physical
market schedules because this machine does not own the complete phase roots.
They exercise the real public CLIs, filesystem lifecycle, seals, matrix loops,
selection/terminal logic, 68 Alpha constructor activations, and hostile boundary
controls, but they are not a substitute for the data-PC market replay.

The local process ledger covers P01-P26. P23-P25 exercise the real descriptor
open and engine-activation identities in a child process, and P26 crosses the
public prelock CLI into the embedded incumbent-audit preflight. P11 covers the
production row/cost/fold control with deterministic replay data and is expressly
not evidence that the market schedules ran against the frozen phase roots.

## Final local verification ledger

- Frozen Section 13 command: `144 passed in 82.04s`; maximum RSS
  `355,796 KiB`.
- Clean-environment full pytest: `4,849 passed, 20 skipped, 3 xfailed in
  236.77s`.
- Full branch-coverage run: the same `4,849 passed, 20 skipped, 3 xfailed`;
  total `80%`, financial-core `83%` (floor `70%`), and
  live/exchanges/core `71%` (floor `65%`).
- Public CLI/process boundary suite: `50 passed`; P01-P26 present in the sealed
  coverage ledger.
- Raw-first/market-window contract suite: `78 passed`; numeric golden suite:
  `8 passed`; GPU contract suite: `24 passed`, followed by a successful strict
  GPU query with CPU/GPU row parity on the local NVIDIA device.
- Dashboard: `86 passed`, with lint, typecheck, and production build also
  passing. Native Rust/Python backends built successfully.
- Benchmark/8 GiB gate: `14,765.39 bars/s`, peak RSS `259.61 MiB`, no OOM
  signature, overall `PASS` against the `7.2 GiB` RSS limit.
- `uv lock --check`, repository-wide Ruff lint/format, compileall,
  `git diff --check`, all three architecture gates, hardcoded-value audit
  (`2,314` baselined, `0` new), and documentation verification passed.
- Independent architecture review finished `CLEAR`. Independent code review
  reported no remaining critical, high, or medium production issue after this
  ledger was added.

## Data-PC-only remaining action

The sole performance-bearing next action is the no-discretion procedure in
`alpha_max_data_pc_runbook_20260711.md` after verifying
`alpha_max_final_sha256_20260711.txt` on the exact pushed commit.

The data PC must provide all frozen raw and feature phase roots, run the prelock
process once, independently verify its immutable bundle, then run the physically
separate exposed historical process once. Missing data, a failed gate, a
consumed completion identity, or any seal mismatch is a terminal failure for
that output identity; it is never permission to change the experiment.

## No-claim and allocation decision

- Real-money allocation: **0%**.
- Paper/testnet/live approval: **none**.
- Fresh confirmation: **not run and still mandatory**.
- Historical exposed evaluation: report-only even if its gates pass.
- Return/MDD, turnover/RPT, capacity, and historical leadership cannot authorize
  selection or deployment.

A passing exposed historical report would still require a genuinely fresh,
uninspected future/withheld interval under a new predeclared protocol and an
explicit operator decision before any allocation review.

## Normative handoff files

- `docs/research_note/alpha_max_final_delivery_20260711.md` — current local
  completion and claim boundary.
- `docs/research_note/alpha_max_data_pc_runbook_20260711.md` — exact data-PC
  execution and audit procedure.
- `docs/research_note/alpha_max_final_sha256_20260711.txt` — normative source,
  test, config, plan, and handoff checksum manifest.
- `docs/research_note/alpha_max_independent_checkpoint_20260711.md` — historical
  implementation checkpoint only.
