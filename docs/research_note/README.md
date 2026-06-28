# research_note

Canonical research-note directory. Use this stable name instead of strategy/date-specific document names.

## Files

- `research_note.md` — current cumulative strategy research note and research journal. New entries go at the top in latest-first order.
- `research_history.md` — global research inventory/source ledger. Regenerate/update only when source families or global chronology change.
- `state_distilled.md` — archived predecessor note for the state-distilled non-calendar lane.

## Update rule

1. Prepend the newest research diary entry to `research_note.md` under **Research journal — latest first**.
2. Keep strategy names, artifact families, dates, and decisions inside the entry body, not in the filename.
3. Keep session checkpoints in `.omx/notepad.md` and detailed handoffs in `docs/session_handoff_*.md`.
4. Keep large generated evidence under `var/reports/`.
5. If the global source ledger changes, update `research_history.md` and the matching `var/reports/.../research_history/` artifacts, or explicitly document why regeneration was unnecessary.

## Latest diary index

- 2026-06-28 KST — 최신 데이터 포함 전체 WF 재평가 정정; 85/110 universes to 2026-06-28T10:09Z, full 10-fold ranking updated.
- 2026-06-21 KST — H35 executable shadow/testnet adoption checkpoint, fresh-forward evidence, and live decision template.

- 2026-05-28 KST — standardized live refits on refreshed data, latest 8 complete weeks as validation, Optuna full-parameter tuning, and train+validation final refit.
- 2026-05-28 KST — canonicalized research-note paths to `docs/research_note/` and added stable latest-first diary rules.
- 2026-05-27 KST — live `MARKET_WINDOW` hot-path optimized with a trusted internal canonical-row fast path; real-money gates remain false.
- 2026-05-27 KST — Python-wrapped Rust live state-signal acceleration added for deterministic Alpha Zoo state machines.
- 2026-05-27 KST — repo-wide format/Rust hygiene baseline and cleanup contract refreshed.
