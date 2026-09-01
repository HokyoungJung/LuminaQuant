# Profit moonshot dynamic restart full non-calendar result — 2026-05-10

Decision: `no_live_promotion`.

## Executed plan result

- Calendar-primary families were excluded from the dynamic restart replay.
- Full non-calendar strategy family replay covered `4429` specs.
- Replay survivors: `0`.
- Portfolio success candidates: `0`.
- Candidate-derived hybrid accepted live-source rows: `0`.
- Final deployable rows: `0`.
- Memory cap: all measured commands were below 8 GiB RSS.

## Artifacts

- Replay: `var/reports/profit_moonshot_20260501/dynamic_restart_full_noncalendar_20260510/replay/fresh_start_overhaul_replay_latest.json`
- Tuning: `var/reports/profit_moonshot_20260501/dynamic_restart_full_noncalendar_20260510/tuning/fresh_portfolio_tuning_latest.json`
- Candidate hybrid: `var/reports/profit_moonshot_20260501/dynamic_restart_full_noncalendar_20260510/candidate_hybrid/candidate_hybrid_latest.json`
- Final decision: `var/reports/profit_moonshot_20260501/dynamic_restart_full_noncalendar_20260510/final_decision/profit_moonshot_live_final_selection_latest.json`
- Handoff: `docs/session_handoff_20260510_profit_moonshot_dynamic_restart_full_noncalendar.md`
- Research history: `docs/research_note/research_history.md`
  - corrected scope: `2026-03-01..2026-05-10`
  - semantic git research commits included: `278`
  - artifact inventory included: `2384`
  - total inventory/ledger rows: `2667` / `2666`
  - strategy chronology entries: `15`

## Guardrails retained

- locked-OOS is report-only/gate-only;
- live leverage must be integer;
- calendar-primary live promotion is rejected;
- source-history/source-search provenance is required;
- liquidation/margin gates remain fail-closed;
- heavy jobs run sequentially under 8 GiB.
