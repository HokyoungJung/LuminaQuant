# Alpha-Max recovery resume plan — 2026-08-02

## Objective

Complete the existing durable Ultragoal recovery: acquire the full official Alpha-Max source in a wholly fresh authenticated run, validate it, compact the existing WAL, authorize source/canonical conflicts, atomically publish one complete canonical generation, verify all existing loaders/features/funding/backtest surfaces, run regressions, clean temporary artifacts, and close the Git/Ultragoal audit trail.

This plan continues durable session `019fad7d-536a-7000-b794-52ccaa961746`, goal `G001`. It does not create a new plan or goal.

## Non-negotiable invariants

- Use only official source archives. Never synthesize 1-second rows or reconstruct them from lower-resolution bars.
- Never reuse or publish any bytes from failed/disqualified/interrupted runs, including `6fefca...` and `47eeac...`.
- A failed attempt is evidence, not a reason to idle: diagnose, repair, and start a wholly fresh safe attempt under new IDs/keys/roots when the failure is autonomously resolvable.
- Preserve 20 GiB Windows C reserve and cgroup/OOM containment.
- Keep `data/market_parquet` unchanged until a fresh run is signed `SUCCEEDED` and every prepublication gate passes.
- Apply WAL before conflict authorization/publication.
- Publication activation is one top-level atomic generation exchange; readers may observe only complete old or complete new generations.
- Real-money execution remains excluded.
- Never reset, stash, revert, delete, or overwrite unrelated user work.

## Phase 0 — Rehydrate and verify the handoff

1. Set `GJC_SESSION_ID=019fad7d-536a-7000-b794-52ccaa961746` for every native Ultragoal command.
2. Read in full:
   - `alpha_max_recovery_session_handoff_20260802.md`;
   - this plan;
   - `alpha_max_recovery_new_session_prompt_20260802.md`;
   - the session-specific Ultragoal brief, goals, and ledger;
   - the interruption receipt and signed terminal receipt.
3. Verify branch `recovery/strategy-plan-20260714`, a clean worktree, HEAD equal to Git ref `recovery-session-handoff-20260802`, and ancestry from `f518dc7bed5cb416dba27886b2337f4a31ea7650`.
4. Run `gjc ultragoal status --json`; require G001 active/incomplete and the final ledger events to describe the user-directed pause.
5. Resume the inline aggregate goal only after the durable state matches.
6. Verify every prior acquisition/publication/monitor unit is inactive and no matching process is running.
7. Verify Windows sleep is AC 900/DC 600 seconds before restarting work.
8. Verify canonical identity `[2096,195868,493,7]` and no publication/generation mutation since the handoff.

Acceptance: exact handoff identities match, no work process is active, Git is clean, and no invalid output is eligible.

## Phase 1 — Bind final code and fresh controls

1. Treat `f518dc7b` as the approved product implementation ancestor. The handoff commit adds only documentation/session evidence.
2. Mechanically pin all four recovery controls to the final handoff HEAD:
   - `verify_fresh_acquisition_47ee.py`;
   - `transition_canonical_wal_47ee.py`;
   - `generate_conflict_authorization_47ee.py`;
   - `verify_published_generation_47ee.py`.
3. Re-run Ruff format/check, all four `--help` probes, focused publisher/terminal/storage tests, `git diff --check`, and `systemd-analyze --user verify` for generated units.
4. Generate cryptographically new run ID, request ID, authority/acquisition/phase/one-touch keys, control root, output root, evidence root, telemetry root, and publication root.
5. Do not copy the old request, receipt, credential, socket, owner marker, source, report, manifest, or phase state.
6. Build and read back production-equivalent units with exact credential, cgroup, sandbox, command, checkout, and digest bindings.
7. Disable Windows sleep only after the fresh capacity gate and record the original/restored values.

Acceptance: a clean final checkout and wholly fresh control topology pass byte/digest/permission/unit readback; no network or canonical mutation has occurred.

## Phase 2 — Fresh official acquisition

1. Start authority, telemetry, then observer under the validated units.
2. Keep one live ZIP at a time.
3. For every archive record URL, HTTP metadata, bytes, digest, archive entry identity, parser version, source row count, output row count, duplicate/order statistics, partition identity, durable provenance receipt, and deletion receipt.
4. Recheck Windows free space, source/scratch usage, memory/swap, and OOM counters throughout the run.
5. Maintain a quiet terminal/fault monitor; monitor timeout must never be confused with acquisition termination.
6. On an actual failure, preserve signed terminal evidence, retire private keys, delete non-reusable staging only after a no-clobber cleanup receipt, diagnose the root cause, and start a new fresh topology without waiting for redundant approval.

Exact success totals:

- raw partitions: `415`;
- raw rows: `1,066,681,730`;
- funding partitions: `12,347`;
- funding rows: `39,569`;
- total parquet files represented by the fresh source contract: `12,762`;
- months: 43 each for ADA/AVAX/BNB/BTC/DOGE/ETH/SOL/TRX/XRP and 28 for TON.

Acceptance: signed terminal state `SUCCEEDED`, successful composite telemetry, no OOM/kill/limit equality failure, no live archive, and exact source/report/manifest/provenance totals.

## Phase 3 — Independent fresh staging audit

1. Run the hardened fresh acquisition verifier from the exact clean pinned checkout.
2. Validate terminal request digest, signature, authority/observer cgroup captures, zero resource events, exact tree, exact manifest, exact provenance and no-clobber PASS receipt.
3. Inspect the PASS receipt, file mode, digest, source identity, and sealed totals independently.
4. Stop fresh acquisition capacity guards only after the PASS receipt is sealed.

Acceptance: one immutable fresh-audit PASS receipt binds the exact successful source and terminal evidence. Canonical remains unchanged.

## Phase 4 — WAL transition

1. Confirm no canonical reader/writer is active.
2. Run the authenticated WAL rehearsal under shared locks.
3. Validate all ten WALs as strict 64-byte records; reject any corruption/misalignment.
4. Capture exact root identity, predecessor inventory digest, per-symbol record counts, duplicate last-write-wins resolution, projected outputs, and capacity.
5. Execute only with the rehearsal root identity and predecessor inventory digest under the exclusive transition lock.
6. Require a durable no-clobber journal and exact post-inventory; verify all ten WALs are empty.

Known planning total: `3,967,207` WAL records across ten symbols, 2026-06-28 through 2026-07-02.

Acceptance: executed WAL receipt binds pre/post inventory and exact clean checkout; every WAL is empty; canonical data reflects the authorized compaction and remains readable.

## Phase 5 — Conflict authorization and capacity audit

1. Recompute conflicts from the fresh successful source after WAL execution.
2. Generate the v2 conflict authorization using the terminal authority identity and one identity-bound private-key fd.
3. Require exact bindings to the fresh audit, terminal receipt, WAL receipt, source tree/report, current canonical inventory, run/request IDs, and conflict decisions.
4. Measure exact current canonical bytes, fresh source bytes, candidate generation estimate, transient budget, old-generation retention, and Windows host free space.
5. Require the 20 GiB reserve through candidate creation, fsync, exchange, rollback window, and old-generation cleanup.

Planning reference only: 56 raw conflict partitions and 649,585 rows; funding overlaps had zero conflicts. Recompute rather than trusting these numbers.

Acceptance: conflict receipt and publication capacity receipt are immutable, exact, and based on the fresh source/current canonical/WAL result.

## Phase 6 — Atomic publication

1. Link/load the verified publication guard and publisher units generated for the fresh IDs.
2. Start the capacity guard, then publisher.
3. Require candidate generation completeness, manifest/data/listing consistency, independent inode identities, durability, and one top-level exchange.
4. Verify publisher terminal resource evidence and canonical identity transition.
5. Invoke the exact same publisher inputs again; require successful idempotent replay and unchanged active generation identity.

Acceptance: canonical is one complete new generation, predecessor handling matches the signed protocol, replay is idempotent, and no mixed/empty reader view was possible.

## Phase 7 — Post-publication integration

Run the fail-closed post-publication verifier with a unique no-clobber receipt. It must independently exercise:

- raw 1-second OHLCV loader;
- existing downsampling;
- funding queries;
- feature-point queries/calculation;
- chunked and panel loading;
- representative legacy backtest;
- generation/manifest and phase receipts;
- exact final row/symbol/month totals.

Acceptance: all observable existing framework surfaces consume the published generation without a parallel data abstraction or fallback.

## Phase 8 — Regression, cleanup, and completion

1. Run focused publisher/terminal/storage tests and the broad repository suite.
2. Run Ruff format/check and `git diff --check`.
3. Report the known frozen ignored `uv.lock` mismatch honestly if it remains; never modify the ignored lock merely to pass the assertion.
4. Remove only temporary units, sockets, private credentials, scratch, failed non-reusable sources, and old generation according to signed cleanup receipts.
5. Preserve final canonical, successful source/provenance required by policy, terminal/publication/WAL/conflict/post-verification evidence, public keys, and capacity history.
6. Restore Windows sleep AC 900/DC 600 seconds.
7. Update the research handoff, machine-readable resume/completion state, and durable Ultragoal ledger with exact paths and SHA-256 values.
8. Commit all product/document changes, require clean status, run the final Ultragoal review/checkpoint, and complete G001 plus the inline aggregate only when every deliverable is evidenced.

Acceptance: no blocker, clean Git, stopped temporary services, restored sleep, final receipts/digests recorded, and durable goal completion verified from the current state.
