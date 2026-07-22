/skill:ultragoal

Resume the stopped, incomplete G061 session in `/home/hoky/Quants-agent/LuminaQuant` on branch `recovery/strategy-plan-20260714`.

Set this on every native Ultragoal command:

```bash
export GJC_SESSION_ID=019f603a-0e73-7000-88a7-c94f42950c09
```

Do not create a plan or goals. Never run `create-goals` or `complete-goals`. Existing master plan, runbook, G060/G061 resume plan, and durable Ultragoal state are canonical.

Read these ten files in full before action:

1. `docs/research_note/strategy_recovery_resume_state_20260714.json`
2. `docs/research_note/strategy_recovery_session_handoff_20260714.md`
3. `docs/research_note/strategy_recovery_new_session_prompt_20260714.md`
4. `docs/research_note/g060_g061_terminal_authority_v3_resume_plan_20260722.md`
5. `docs/research_note/strategy_recovery_master_plan_20260713.md`
6. `docs/research_note/data_pc_strategy_recovery_runbook_20260713.md`
7. `docs/audits/strategy_reality_audit_20260713.md`
8. `.gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/brief.md`
9. `.gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/goals.json`
10. `.gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/ledger.jsonl`

Required Git state:

- implementation checkpoint: `475f3f2ebe37994f574dc970e1b3fa9563da8009`
- handoff-content commit: `__HANDOFF_CONTENT_COMMIT__`
- branch HEAD must be clean and descend from both commits
- historical commits remain ancestry only; do not reset/rebase/amend/push

Before any work:

1. Verify branch, clean worktree, HEAD ancestry from both required commits, and committed file identities from the resume state.
2. Verify durable seals exactly:
   - brief `faf6f83679e7ce93a8950af4df350fc4a92557d8eaaa40ea17c9c8b918c04e57`, 4836 bytes, 22 lines
   - goals `68c3a07e68bb8c3b1864399e0324d8fc3a0ba441e5c6d0417c03feb576c8a667`, 144365 bytes, 1400 lines
   - ledger `ee73088dd033baa68b67b90a25692af185cd7c1bf81c819b4211dfd4f81ee483`, 345608 bytes, 376 lines
3. Verify latest ledger event `7523ed6e-93fb-41f8-a937-18897ac3de8f`, classification `human_blocked`, goal G061.
4. Verify no worker, monitor, acquisition, terminal-authority, phase, one-touch, prelock, historical, order, or capital process is running.
5. Do not inspect either quarantined root for any reason. They may appear only as lexical strings:
   - `/home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source`
   - `/home/hoky/Quants-agent/Quants-agent-alpha-max-data-pc`
6. Verify future v5 roots remain absent without touching quarantined roots. Do not create them.
7. Run `GJC_SESSION_ID=019f603a-0e73-7000-88a7-c94f42950c09 gjc ultragoal status --json`; it must report durable G061 active/incomplete and G060 review-blocked.
8. Call `goal get`, then `goal resume`. Only if a genuinely new thread has no visible inline goal may you create this exact objective:

`Complete the durable ultragoal plan in .gjc/ultragoal/goals.json, including later accepted/appended stories, under the original brief constraints; use .gjc/ultragoal/ledger.jsonl as the audit trail.`

Current durable state:

- G061 is active and incomplete.
- G060 is review-blocked by Architect 67's twelve integration findings.
- G056, G036, and G037 are blocked.
- Counts: complete 12, active 1, blocked 3, review-blocked 15, superseded 30, pending/failed 0.
- Inline aggregate goal was paused only for this user-owned session transition.
- No G061 target or external-root execution occurred.

G061 objective:

`Replace the rejected delegated integration with one interoperable typed wire contract; bind request identities to the one-direction external manifest and real files; prevalidate roots and keys before the first O_EXCL mutation; enforce exact launch/start/exit/recovery state, semantic child-0 gates, canonical acquisition/A-02/sealed one-touch artifacts, immutable before-after snapshots, no-bytecode bounded cwd, secure key provenance, and no-launch recovery; rerun the full G060 cleaner/review/QA gate before any external root or target execution.`

Implementation checkpoint `475f3f2ebe37994f574dc970e1b3fa9563da8009` contains the typed terminal policy/config, secure key creator, no-launch authority, sole-launch observer, and focused tests. Exact post-stop identities are in the resume JSON. Frozen target identities remain:

- acquirer `b440d79899a4ed60e18decfcd8bc2656d2de012189f03572a8be65f90cd24978`
- phase wrapper `054163d23e8d2f1446b225e281472bcc563ac76f06aa47552cc5f3953b7c4dd9`

Stopped-snapshot verification:

- Ruff format/check passed
- `py_compile` passed
- focused terminal suite: `135 passed`
- acquisition/phase/one-touch/prelock/historical execution: none

This is not completion. The mandatory final cleaner rerun, integrated/sanitized full verification, architect review, executor QA/red-team, and strict G061 checkpoint remain pending. The implementation is not execution authority.

Continue in this exact order:

1. Recheck the committed identities and rerun the focused 135-test terminal suite under `uv` and a network namespace.
2. Run the mandatory internal ai-slop-cleaner fragment on only the G061 changed files. Fix blocking findings only with an executor; rerun the focused suite and cleaner until blocking findings are zero.
3. Run integrated terminal/acquirer verification and sanitized full repository verification without executing any target. Preserve and report the established intentional shipped-YAML HEAD deselection if still applicable; never hide other failures.
4. Freeze the post-cleaner change set.
5. Run a fresh architect review across architecture/product/code and obtain `CLEAR/CLEAR/CLEAR APPROVE` on that exact snapshot.
6. Run a fresh executor QA/red-team lane on the same snapshot. It must produce real package/API/CLI artifact evidence and attempt invalid framing, signatures, replay, path identities, symlink ancestors, quarantine paths, launch ordering, crash recovery, clearance, receipt, phase snapshot, sealed inventory/readback, and no-launch authority behavior. Never execute the actual acquirer, wrapper, prelock, or historical targets.
7. If either review lane finds a blocker, record durable replacement work and repeat the full cleaner/verification/review loop. Do not checkpoint G061 and do not complete the inline goal.
8. When and only when all gates are clean, strictly checkpoint G061 with structured quality-gate JSON.
9. Explicitly supersede or resolve G060 using the fresh completed G061 receipt. Do not run `complete-goals`.
10. Only after G060 terminal authority is independently cleared, explicitly reactivate G056, create/freeze fresh v5 roots, and launch official complete acquisition with real Python-child monitoring and repository descriptor refresh.
11. Run `validate_complete` and offline `--verify-eligible`; then create six authenticated phase roots and run one-touch under the cleared authority.
12. Complete G056, then execute genuinely diverse G036 and G037.
13. Perform a current-state final aggregate audit, create a fresh aggregate receipt, and only then complete the inline goal.

Preserve the official BTCUSDT 2023-10 proof:

- futures-UM archive SHA `d3fe5fa477d68d6730248d634e1bd37ae4838839d78709ef355d9d9c6749fea4`
- bytes `492720741`
- trades `38272235`
- ID/timestamp regressions `3988367`
- canonical rows `2678400`
- frame SHA `890b0e591990fbabf35f323f7987547d99cdc62416cba826ca601393b0b34f79`
- byte-identical proof SHA `9902947934a9df52685db0b5198c69d9f57f6b54836b923bb804a0bab0387b27`
- evidence root `/home/hoky/quants-recovery-runs/G058-order-repro-20260720`
- unrelated spot ZIP is not scientific evidence

V4 is immutable failed/ineligible evidence only. Never reuse, continue, mutate, copy into, or treat it as authority. Preserve official-only data, no synthesis/substitution/prelisting/date shifts/retuning/locked-OOS selection, zero exchange orders/capital, Python through `uv`, profile-first execution, and native/Rust only after a material exact-equivalence benchmark. Never stage or force-add the ignored local `uv.lock`.

If another official archive fails ordering, preserve v5 and create an explicit evidence-backed replacement subgoal. Never broaden canonical sorting automatically.

Scope clarification: the exact `HokyoungJung/Market-Cap-Weighted-Indices` repository is not directly integrated. Existing TopCap and turnover/flow-share work is adjacent, not equivalent. Any later port needs point-in-time constituent/capitalization evidence and must remain research/shadow-only until clean walk-forward and cost/funding gates pass. The current pipeline work is correctness/provenance hardening, not performance tuning.
