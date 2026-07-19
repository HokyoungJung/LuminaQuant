/skill:ultragoal

Resume the G056 cross-session handoff in `/home/hoky/Quants-agent/LuminaQuant`. The implementation checkpoint is `5bdc3b378cc477a939428d4455462ba651dc024b` (`Checkpoint G056 acquisition and raw durability work`), on branch `recovery/strategy-plan-20260714`. It is incomplete: do not make any completion claim. The handoff-document commit is unknown; the commit containing the three handoff files must descend from that checkpoint.

First read all three handoff files: `docs/research_note/strategy_recovery_resume_state_20260714.json`, `docs/research_note/strategy_recovery_session_handoff_20260714.md`, and `docs/research_note/strategy_recovery_new_session_prompt_20260714.md`; then read `docs/research_note/strategy_recovery_master_plan_20260713.md`, `docs/research_note/data_pc_strategy_recovery_runbook_20260713.md`, `docs/audits/strategy_reality_audit_20260713.md`, and authoritative durable files `.gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/{brief.md,goals.json,ledger.jsonl}`. Set `GJC_SESSION_ID=019f603a-0e73-7000-88a7-c94f42950c09`, run `gjc ultragoal status --json`, and verify Git ancestry, clean state, durable seals, and that no process is running. Call `goal get`, then `goal resume`. Only if no inline goal is visible in the genuinely new session, create it with this exact stable objective: `Complete the durable ultragoal plan in .gjc/ultragoal/goals.json, including later accepted/appended stories, under the original brief constraints; use .gjc/ultragoal/ledger.jsonl as the audit trail.` Never run `create-goals`, create a competing plan, or run `complete-goals`.

The durable session is `019f603a-0e73-7000-88a7-c94f42950c09`; durable G056 remains active/incomplete. Latest durable event is `human_blocked` `44a0ea42-0861-41b6-a74d-f18c04bfd507` at `2026-07-19T13:53:05.860Z`. Verify seals exactly as recorded in the JSON and handoff. The portable bootstrap is these three documents; `.gjc` is canonical same-machine state and `ledger.jsonl` is the audit trail.

Proceed in this exact order:
1. Resume exact durable session and inline goal.
2. Close scratch and same-open authentication review; run Ruff and focused tests via `uv`.
3. Run a real allowed-v3-provenance artifact two-run deterministic/bounded benchmark and the already-ordered BTCUSDT 2024-01 byte regression; admitted output SHA `ac99f6439d544901db0c09d8d2e7ad7ffcd9d328a2de73e7930231a849e2b1e8`.
4. Freeze acquirer, repin wrapper/tests, and run focused, integrated, and full gates.
5. Run mandatory cleaner, then fresh architect CLEAR/CLEAR/CLEAR APPROVE and executor QA/red-team.
6. Run fresh v4 complete acquisition with periodic repository descriptors, `validate_complete`, and offline verify-eligible.
7. Run six authenticated phase roots and exact one-touch runbook fences 4-11 after wrapper.
8. Create strict G056 checkpoint, then explicitly supersede predecessors.
9. Run G036 C-00..C-05/C-06 candidate contracts and execution.
10. Run G037 fresh-forward.
11. Perform current-state final aggregate audit/receipt/goal complete.

Preserve the stop state: all detached workers/monitors are terminal; no acquirer/network acquisition runs; v4 never started; no eligible receipt; no phase/prelock/historical/candidate/forward/order/capital operation. V2/v3 are immutable. Fresh v4 roots are allowed only after approval: `/home/hoky/quants-external-data/alpha-max-g056-full-source-v4`, `/home/hoky/quants-recovery-runs/G056-full-acquisition-report-20260719-v4`, and `/home/hoky/quants-recovery-runs/G056-full-acquisition-controller-20260719-v4`. Never mutate/reuse/copy into v2/v3; advance again after any code change following a started v4.

Treat `/home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source` and `/home/hoky/Quants-agent/Quants-agent-alpha-max-data-pc` as quarantined lexical strings: never inspect, stat, hash, traverse, read, copy, or use them; they may appear only as `--forbidden-root` argv. Preserve official-only, no synthetic/substitute/prelisting/date shift/retune/locked-OOS-selection, zero order/capital, `uv` Python, profile-first only, Rust/native only after material exact-equivalence benchmark, and no reset/rebase/amend/push.
