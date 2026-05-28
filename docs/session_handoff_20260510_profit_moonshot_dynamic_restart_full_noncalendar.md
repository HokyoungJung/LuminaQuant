# Session handoff — profit moonshot dynamic restart / full non-calendar audit — 2026-05-10

## Final conclusion

**No live profit-moonshot strategy is selected.** After the calendar-primary defect was recorded, the restart lane replayed the full non-calendar/dynamic family set and still found **0 replay survivors**, **0 portfolio success candidates**, and **0 deployable final-selection rows**. The safe live action remains `no_live_promotion`.

This handoff is intentionally conservative: locked-OOS is report-only/gate-only, source-history/source-search provenance is required, calendar-primary rows are rejected, non-integer leverage is rejected, and all heavy commands stayed far below the 8 GiB RSS cap.

## Scope and data contract

- Universe: `BTC/USDT`, `ETH/USDT`, `SOL/USDT`, `BNB/USDT`, `TRX/USDT`.
- Split windows inherited from the current profit-moonshot artifacts:
  - train: `2025-01-01..2025-12-31`
  - validation: `2026-01-01..2026-02-28`
  - locked-OOS: `2026-03-01..2026-05-09`
- Timeframe/inputs: refreshed raw-first Binance market data through the current complete tail; replay runner builds stateful candidate bars from the fresh-start data panel.
- Selection rule: train/validation only. Locked-OOS is never a ranking, tuning, expansion, or tie-break input.
- Excluded from live promotion: calendar-primary `calendar_rotation` / `calendar_spread` rows and any source row without research-history/source-search metadata.

## Strategy families replayed from first principles

Full non-calendar family allowlist:

`residual_reversion`, `cross_momentum`, `residual_momentum`, `funding_carry_fade`, `funding_carry_momentum`, `funding_oi_carry_fade`, `flow_momentum`, `flow_exhaustion_fade`, `flow_imbalance_persistence`, `flow_imbalance_exhaustion`, `residual_reversion_flow_confirmed`, `adaptive_trend`, `adaptive_trend_fade`, `cross_sectional_sharpe_rank`, `cross_sectional_sharpe_reversal`, `residual_pair_reversion_spread`, `residual_pair_momentum_spread`, `compression_breakout`, `compression_breakout_fade`, `compression_expansion_downside_short`.

A smaller 120-spec probe was also run first as an OOM-safe smoke; it also produced 0 survivors and 0 success candidates.

## Full non-calendar replay evidence

Artifact root: `var/reports/profit_moonshot_20260501/dynamic_restart_full_noncalendar_20260510/`

| Stage | Key artifact | Result | RSS evidence |
|---|---|---:|---:|
| Replay | `replay/fresh_start_overhaul_replay_latest.json` | `spec_count=4429`, `replay_survivor_count=0`, `success_candidate_count=0` | `288.621 MiB` |
| Portfolio tuning | `tuning/fresh_portfolio_tuning_latest.json` | `portfolio_spec_count=140`, `success_candidate_count=0` | `270.043 MiB` |
| Candidate-derived hybrid | `candidate_hybrid/candidate_hybrid_latest.json` | `status=no_live_source_candidates`, accepted source rows `0/8` | `252.457 MiB` |
| Final decision | `final_decision/profit_moonshot_live_final_selection_latest.json` | `status=no_live_promotion`, deployable rows `0/14` | `34.125 MiB` |

## Best diagnostic portfolio row — not promoted

The best train/validation diagnostic from portfolio tuning was not a live candidate:

- name: `fresh_portfolio_validation_return_risk_weight_fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all__fresh_pair_resid_revert_spread_lb24_z150_h72_sc10_st100_tp240_asiaus`
- leverage: integer `1x`
- sleeves: two residual pair mean-reversion spread specs
- train return / MDD / Sharpe / Sortino / Calmar: `+0.1220%` / `0.2071%` / `0.5511` / `0.4441` / `0.5893`
- validation return / MDD / Sharpe / Sortino / Calmar: `+0.0882%` / `0.0526%` / `2.3339` / `3.0069` / `10.4044`
- locked-OOS return / MDD / Sharpe / Sortino / Calmar: `+0.0782%` / `0.0536%` / `2.0904` / `2.4011` / `8.0362`
- decision: **diagnostic_not_promoted**
- main failures: train/validation monthly returns far below required 2% floors, train Sharpe/Sortino/Calmar weak, OOS return and return/MDD do not beat the current base, and train/validation stability score does not beat current base.

## Candidate-derived hybrid result

The dynamic restart tuning generated no promotable source candidates. The hybrid runner therefore wrote a fail-closed artifact instead of crashing:

- `status=no_live_source_candidates`
- raw source rows considered: `8`
- live-source accepted rows: `0`
- discarded reasons include `calendar_primary_source_invalid`, `research_history_source_metadata_missing`, and for current-base forced `5x`, `liquidation_source_unsafe`.

This is correct behavior: candidate-derived hybrids must not silently reuse invalid calendar/liquidation/source-metadata-unsafe sleeves.

## Final live decision

Final decision artifact: `var/reports/profit_moonshot_20260501/dynamic_restart_full_noncalendar_20260510/final_decision/profit_moonshot_live_final_selection_latest.json`

- `recommendation=no_live_promotion`
- `winner=null`
- `memory_ledger.under_8gib=true`
- `deployable_count=0`
- locked-OOS remains report-only/gate-only

## Research-history update

Research history was regenerated after this run:

- `docs/research_note/research_history.md`
- `var/reports/profit_moonshot_20260501/research_history/profit_moonshot_research_history_latest.json`
- `var/reports/profit_moonshot_20260501/research_history/profit_moonshot_research_history_latest.md`

The regenerated ledger was corrected from May-only scope to `2026-03-01..2026-05-10`.
It now includes semantic git history from the March/April predecessor research lanes plus local artifacts:

- `git_commit_count=278`
- `artifact_inventory_count=2384`
- `inventory_count=2667`
- `ledger_count=2666`
- `strategy_chronology=15`

The chronology now covers raw-first/live-data foundations, exact-window/timeframe sweeps,
dynamic portfolio/walk-forward work, strict latest-tail validation, regime/pair challengers,
portfolio-superiority/leverage work, hybrid reboot, production carry/trend lanes,
late-April live-equivalent filtering, and the May profit-moonshot restart.

## Verification

- Full non-calendar replay: exit `0`, `4429` specs, max RSS `295548 kB`.
- Portfolio tuning: exit `0`, max RSS `276524 kB`.
- Candidate hybrid fail-closed artifact: exit `0`, max RSS `258516 kB`.
- Final decision rebuild: exit `0`, max RSS `34944 kB`.
- Research-history regeneration: exit `0`, max RSS `192972 kB`.
- Targeted leader regression before full replay: `20 passed`, max RSS `174872 kB`.

## Next instructions

1. Do not trade or promote any current profit-moonshot row.
2. Treat the old calendar-primary high-return rows as invalid for live unless a separate pre-registered seasonal thesis is created and tested.
3. Future research should start from the regenerated research-history artifact before searching again.
4. Any future candidate must show positive train/validation evidence before locked-OOS reporting, carry integer leverage, pass conservative liquidation replay, include source-history/source-search references, and stay under 8 GiB RSS.
