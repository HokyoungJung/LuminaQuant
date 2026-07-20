# Strategy recovery cross-session handoff — G058 — 2026-07-20

## Resume identity

- Repository: `/home/hoky/Quants-agent/LuminaQuant`; branch: `recovery/strategy-plan-20260714`.
- Implementation checkpoint: `b349cb57596a44d9e7e4a68519d0ddb586f97dc3`; it is incomplete.
- Handoff-content commit: `__HANDOFF_CONTENT_COMMIT__` (the first commit containing this handoff must descend from that checkpoint).
- Authoritative Ultragoal session: `019f603a-0e73-7000-88a7-c94f42950c09`; durable active goal: **G058**. G056, G036, and G037 are blocked.
- Portable bootstrap: this file, `strategy_recovery_resume_state_20260714.json`, and `strategy_recovery_new_session_prompt_20260714.md`. The canonical same-machine state is `.gjc`; `ledger.jsonl` is its audit trail.

Stable aggregate objective:

> Complete the durable ultragoal plan in `.gjc/ultragoal/goals.json`, including later accepted/appended stories, under the original brief constraints; use `.gjc/ultragoal/ledger.jsonl` as the audit trail.

The durable handoff annotation is next; pause the inline goal only after that annotation. No completion claim is authorized. Explicitly forbid `create-goals` and `complete-goals`.

## Durable seals and stop state

| Durable file | SHA-256 | Bytes | Lines |
|---|---|---:|---:|
| `brief.md` | `faf6f83679e7ce93a8950af4df350fc4a92557d8eaaa40ea17c9c8b918c04e57` | 4836 | 22 |
| `goals.json` | `a0fb91089f27262196db7314257019c2edebee14e2a0a82000d67c1f35e6046a` | 137951 | 1322 |
| `ledger.jsonl` | `8cf64435db08254e4f99afcefa7902b668f7626d89ca1bb19f36c48e3d1e2c5a` | 319528 | 350 |

Latest pre-stop event is `goal_checkpointed` G058, `78f41109-94ae-436a-9723-5bc9089ce446`, at `2026-07-20T12:58:36.033Z`. Refresh the final ledger event and all durable-file hashes once more after the stop annotation and pause. All workers and monitors are terminal; no acquisition process runs. There is no eligible receipt, phase, prelock, historical, candidate, forward, order, or capital operation.

## Preserved G056 v4 evidence — terminal and never reusable

- Controller: `/home/hoky/quants-recovery-runs/G056-full-acquisition-controller-20260719-v4`
- Source: `/home/hoky/quants-external-data/alpha-max-g056-full-source-v4`
- Report: `/home/hoky/quants-recovery-runs/G056-full-acquisition-report-20260719-v4`
- Failed/ineligible at raw `139/415` through BTCUSDT 2023-09; funding `3834/12347`; `2625436018` source bytes; `3974` files; failure `archive_trade_order_invalid`; zero orders and capital.
- Descriptor sequence 10 SHA-256: `70b1c5d421a1f28fbce4943210db7467b058e96d0e27d59599cd3005b3d891b9`.

V4 is historical scientific evidence only. After the code change it must never be reused, continued, mutated, copied into, or treated as execution authority.

## G058 root cause, repair, and proof

The official Binance futures-UM BTCUSDT 2023-10 archive has SHA-256 `d3fe5fa477d68d6730248d634e1bd37ae4838839d78709ef355d9d9c6749fea4`, `492720741` bytes, and `38272235` rows. It has `3988367` adjacent ID regressions and timestamp regressions, zero equal adjacent IDs, and consists of interleaved valid ordered streams. G058 repaired the exact digest-bound allowlist and wrapper pin.

Implementation identities: acquirer `864b397ee0a26cad1e4be67431c9d6e2929280a06c59966a5f57712634a2c7ad`; wrapper `db5eab028edf78e0d816e6838a50c65f910cf122701428ca7704dcea5b5e5ee9`; acquirer test `2e3c71142263decb13c0dfed680ff0e4f71c7a1b497f3923b1312625bb56ce6f`; wrapper test `7b711ecc998d28628d4c5956f46002258d03da615ef071e8e4b7a0b73723aeb5`.

Two canonical runs produced `2678400` rows, first `[1696118404732,1862715434]`, last `[1698796799559,1900987668]`, carry `34651.4`, frame SHA `890b0e591990fbabf35f323f7987547d99cdc62416cba826ca601393b0b34f79`, and byte-identical proof SHA `9902947934a9df52685db0b5198c69d9f57f6b54836b923bb804a0bab0387b27`. There was no scratch residue; observed use was about 520 MiB RSS and 134–136 seconds. Evidence: `/home/hoky/quants-recovery-runs/G058-order-repro-20260720/{futures-order-scan.json,canonical-proof-run1.json,canonical-proof-run2.json}`. The unrelated spot ZIP there is not scientific evidence.

Current verification passed: Ruff, pycompile, and git diff check; focused `185 passed +33 subtests`; integrated `329 passed +33 subtests`; sanitized full `5331 passed +33 subtests, 20 skipped, 1 explicit unrelated deselection, 3 xfailed`.

## Incomplete quality gate and blocked terminal authority

`27-G058AiSlopCleaner` is **BLOCKED only on the stale resume descriptor**. `29-G058ResumeDescriptorFix` corrected it, but the cleaner has not rerun. Its missing exact production-tuple test was advisory and remains unimplemented.

Architect `28-G056TerminalVerifierReview` returned **BLOCK/REQUEST CHANGES** on the external terminal verifier. The files `/home/hoky/quants-recovery-runs/G056-full-acquisition-controller-20260719-v4/g056_boundary_primitives.py`, `g056_boundary_verifier.py`, and rejected `post_acquisition_gate.py` are not execution authority and must never run. Critical/high blockers: wrong topology (runbook has 14 fences and explicit no-execute language; verifier assumes 11/fences 4–11); no first-operation forbidden-root/role-disjointness proof; caller-self-authenticated authority/process receipts; assertion-only O_EXCL/non-TTY/at-most-one/publication/zero-order facts; shallow phase semantics; observability not sealed-payload bound; outcome/readback/seal disagreement allowed; and caller-self-bound manifest authority.

## Only future acquisition roots

These roots do not exist and must not start yet:

- `/home/hoky/quants-external-data/alpha-max-g058-full-source-v5`
- `/home/hoky/quants-recovery-runs/G058-full-acquisition-report-20260720-v5`
- `/home/hoky/quants-recovery-runs/G058-full-acquisition-controller-20260720-v5`

Create them only after G058 cleaner rerun, post-cleaner verification, fresh acquirer architect `CLEAR/CLEAR/CLEAR APPROVE`, and executor QA/red-team pass.

## Exact continuation order

1. Verify Git/session/durable hashes/process absence; `goal get`, then `goal resume`.
2. Rerun cleaner on repaired descriptor/code, resolve only blockers, rerun Ruff/pycompile/focused/integrated/sanitized full.
3. Obtain fresh architect and executor QA for G058.
4. Before wrapper/phase work, replace/rebuild and independently CLEAR the blocked terminal authority—never execute current external drafts.
5. Create/freeze fresh v5 roots and launch official complete acquisition with real Python-child monitoring and periodic repository descriptors.
6. Run `validate_complete` and offline `--verify-eligible`.
7. Strict G058 checkpoint, then reactivate G056 explicitly without `complete-goals`.
8. Six authenticated phase roots and one-touch only after terminal authority CLEAR.
9. Strict G056 checkpoint/supersede predecessor chain.
10. G036, G037, final aggregate audit/receipt, then inline completion only at the very end.

If another official archive fails ordering, preserve v5 and create an explicit evidence-backed replacement subgoal; never broaden automatically.

## Binding references and quarantine

Original binding references remain `docs/research_note/strategy_recovery_master_plan_20260713.md`, `docs/research_note/data_pc_strategy_recovery_runbook_20260713.md`, and `docs/audits/strategy_reality_audit_20260713.md`.

`/home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source` and `/home/hoky/Quants-agent/Quants-agent-alpha-max-data-pc` are lexical `--forbidden-root` strings only: never inspect, stat, hash, traverse, read, copy, or use either root. Preserve official-only/no synthetic, substitution, prelisting, date-shift, retune, or locked-OOS selection; zero orders/capital; `uv` Python; profile-first; Rust/native only after a material exact-equivalence benchmark; and no reset, rebase, amend, or push.
