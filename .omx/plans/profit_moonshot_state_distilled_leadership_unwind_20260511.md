# Profit moonshot state-distilled leadership/unwind build — 2026-05-11

## Goal
Build a legitimate non-calendar strategy that can reproduce the rejected calendar tuple's useful shape without month/day/fixed seasonal rules.

## Implemented strategy
`state_distilled_leadership_unwind` in `scripts/research/replay_profit_moonshot_fresh_start.py`.

- Inputs: cross-sectional return/residual-z rank, fast/slow momentum, taker-flow imbalance, market regime, optional OI/funding crowding.
- Exclusions: no month, day-of-month, entry-hour calendar, calendar long/short month, or fixed seasonal TRX/ETH target.
- BTC is used as market/risk anchor, not as this family's tradable alt target.
- Candidate specs generated: `both` and `longonly`; dedicated standalone `shortonly` generation was removed after it proved train/validation-overfit. The signal helper still supports shorting a crowded laggard inside `both` mode.

## Best train/validation-selected state-distilled candidate
`fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600` at strict zero-liquidation `4x`:

- train: return +32.9431%, MDD 9.4768%, Sharpe 1.9463, Sortino 2.0182, Calmar 3.4766, liq 0, min buffer 9740.5206, min margin ratio 177.5034, fills 104, round trips 52
- validation: return +11.6925%, MDD 3.1028%, Sharpe 4.9606, Sortino 5.9849, Calmar 31.6786, liq 0, min buffer 9959.0876, min margin ratio 204.1618, fills 22, round trips 11
- oos: return +2.4722%, MDD 2.5328%, Sharpe 1.5131, Sortino 1.8815, Calmar 5.6787, liq 0, min buffer 9875.3540, min margin ratio 208.3866, fills 16, round trips 8
- Strategy validity: `True`; liquidation gates all strict-zero: `True`.

## 5x/6x diagnostic, respecting user's non-wipeout comment
Same candidate diagnostics:

- `5x`: train: return +34.3497%, MDD 15.5461%, Sharpe 1.6520, Sortino 1.7007, Calmar 2.2098, liq 2, min buffer 9675.6508, min margin ratio 141.0255, fills 106, round trips 53; validation: return +14.7888%, MDD 3.8719%, Sharpe 4.9632, Sortino 5.9860, Calmar 34.8329, liq 0, min buffer 9948.8594, min margin ratio 162.4580, fills 22, round trips 11; oos: return +3.0887%, MDD 3.1589%, Sharpe 1.5194, Sortino 1.8899, Calmar 5.7681, liq 0, min buffer 9844.1925, min margin ratio 166.0417, fills 16, round trips 8. Account wipeouts: `0`; strict promotion blocked by train liquidations `2`.
- `6x`: train: return +34.8783%, MDD 22.3058%, Sharpe 1.4128, Sortino 1.4505, Calmar 1.5639, liq 4, min buffer 9610.7624, min margin ratio 120.0425, fills 104, round trips 52; validation: return +17.9560%, MDD 4.6383%, Sharpe 4.9658, Sortino 5.9869, Calmar 38.3707, liq 0, min buffer 9938.6286, min margin ratio 134.6668, fills 22, round trips 11; oos: return +3.7036%, MDD 3.7832%, Sharpe 1.5251, Sortino 1.8976, Calmar 5.8558, liq 0, min buffer 9813.0310, min margin ratio 137.8046, fills 16, round trips 8. Account wipeouts: `0`; strict promotion blocked by train liquidations `4`.

## Decision
Created a real, calendar-free research strategy and validated it under liquidation-aware replay. It is **not a strict deployable replacement yet** because locked-OOS return and return/MDD do not beat the current-base replay, and 5x/6x introduce train liquidations even though no account wipeout occurred.

- Strict selected non-calendar row: 4x, zero liquidations, OOS +2.4722%, MDD 2.5328%, return/MDD 0.976107.
- Current-base reference replay: OOS +6.4281%, MDD 0.9293%, return/MDD 6.916878; still strategy-invalid due calendar-primary sleeves.
- Artifact decision remains `no_live_promotion_strategy_validity_failed`; calendar-primary current-base remains blocked.

## Artifacts
- Replay: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_leadership_unwind_20260511/`
- Liquidation-aware validation: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_aware_state_distilled_20260511/`
- Replay counts: spec_count `648`, train/validation-positive candidates `113`, survivor `0`, success `0`, peak RSS `253.992 MiB`.
- Liquidation validation: evaluated `480` integer results; deployable `0`; max RSS recorded in run output `273124 KiB`.

## Verification
- Full verification: targeted profit-moonshot suite 54 passed; full pytest 1264 passed in 264.61s / wall 4:01.78, max RSS 2782136 KiB; ruff check ., compileall scripts/tests/src, and git diff --check passed.
