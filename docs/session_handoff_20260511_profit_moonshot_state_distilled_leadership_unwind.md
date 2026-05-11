# Session handoff — profit moonshot state-distilled leadership/unwind — 2026-05-11

## User correction handled
The calendar/month rule remains rejected. This pass built a causal state proxy instead of resurrecting `fresh_calendar_*` sleeves.

## What changed
- Added tests proving the new state-distilled signal is dynamic and calendar-free, including a same-state January/June check and a crowded-laggard short check.
- Added `state_distilled_leadership_unwind` signal generation to fresh-start replay.
- Generated `both` and `longonly` specs only; standalone `shortonly` spec generation was removed because it selected train/validation artifacts that failed locked-OOS.
- Replayed the new family and ran liquidation-aware validation with locked-OOS report/gate-only.

## Strategy mechanics
The rule ranks non-BTC symbols by causal state:

1. compute slow return/residual-z and fast momentum;
2. require cross-sectional residual spread/rank gap;
3. for longs, require positive leader return, positive fast return, residual-z above threshold, optional positive taker-flow;
4. for shorts inside `both`, require negative laggard return/fast return/residual-z plus bearish market, negative flow, or crowding context;
5. choose long leader versus short laggard by state score, never by month/day/calendar bucket.

## Key results
Replay artifact: `state_distilled_leadership_unwind_20260511/fresh_start_overhaul_replay_latest.json`.

- spec_count `648`; train/validation-positive `113`; replay_survivor `0`; success `0`; peak RSS `253.992 MiB`.
- Best raw train/validation state-distilled row: `fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600`
  - train +8.0437%, validation +2.9051%, OOS +0.7380%, OOS MDD 0.6042%, OOS Sharpe 1.7795.

Liquidation artifact: `liquidation_aware_state_distilled_20260511/liquidation_aware_current_base_latest.json`.

- selected train/validation retune: `fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600` at `4.0x`, deployable `False`.
- 4x strict-zero row:
  - train: return +32.9431%, MDD 9.4768%, Sharpe 1.9463, Sortino 2.0182, Calmar 3.4766, liq 0, min buffer 9740.5206, min margin ratio 177.5034, fills 104, round trips 52
  - validation: return +11.6925%, MDD 3.1028%, Sharpe 4.9606, Sortino 5.9849, Calmar 31.6786, liq 0, min buffer 9959.0876, min margin ratio 204.1618, fills 22, round trips 11
  - oos: return +2.4722%, MDD 2.5328%, Sharpe 1.5131, Sortino 1.8815, Calmar 5.6787, liq 0, min buffer 9875.3540, min margin ratio 208.3866, fills 16, round trips 8
- 5x diagnostic: OOS +3.0887%, MDD 3.1589%, Sharpe 1.5194, train liquidations 2, wipeouts 0.
- 6x diagnostic: OOS +3.7036%, MDD 3.7832%, Sharpe 1.5251, train liquidations 4, wipeouts 0.

## Final conclusion
A legitimate non-calendar strategy now exists and reproduces part of the calendar tuple's shape, but it is research-only today:

- strict 4x has zero liquidations and positive train/validation/OOS, but OOS return/risk does not beat the current-base replay;
- 5x/6x are non-wipeout diagnostics but have train liquidation events, so they remain blocked under strict promotion;
- current-base calendar tuple remains invalid regardless of its performance.

## Verification completed so far
- Targeted tests: `uv run --extra dev pytest -q tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_liquidation_aware_validation.py` → 34 passed.
- Replay: exit 0, max RSS 260088 KiB.
- Liquidation-aware validation: exit 0, max RSS 273124 KiB.

- Full verification: targeted profit-moonshot suite 54 passed; full pytest 1264 passed in 264.61s / wall 4:01.78, max RSS 2782136 KiB; ruff check ., compileall scripts/tests/src, and git diff --check passed.

## Next recommended work
Use this family as a research seed. Next improvement should try portfolio-level combinations of state-distilled + residual-pair signals selected only by train/validation, not locked-OOS, and keep calendar-primary sleeves blocked.
