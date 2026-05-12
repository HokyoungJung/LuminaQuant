# Profit moonshot fresh-start overhaul replay

Generated: `2026-05-12T13:17:21.698895Z`
OOS end date: `2026-05-06`

## Intent

- 기존 ETH shock-reversion incumbent/leadlag/context-wrapper를 쓰지 않고 raw-first data에서 새로 출발했다.
- 신규 후보군: cross-sectional residual reversal, cross-sectional momentum, adaptive trend, cross-sectional Sharpe/rank selector, funding-carry fade, funding+OI carry fade, taker-flow persistence/exhaustion, non-calendar state-distilled leadership/unwind, non-calendar crowded-leadership unwind v2, non-calendar TRX state-momentum proxy, non-calendar TRX/ETH state-relative-strength spread, calendar rotation, calendar-conditioned veto/day-window sleeves, TRX/ETH calendar spread, compression breakout.
- Replay는 one-position, fee/slippage, 10% bar-volume fill cap, cooldown, stop/take/max-hold, 0.8% target allocation, $175 max order를 강제한다.

## Gate policy

- Success requires OOS return > `+1.2181%`, OOS MDD < `0.1778%`, OOS Sharpe > `1.0`, liquidations `0`, and positive train/val.
- Replay survivor는 full live-equivalent raw-first backtest 후보일 뿐이며, sub-1 Sharpe는 성공이 아니다.

## Result

- Specs evaluated: `972`
- Replay survivors: `0`
- Success candidates: `0`
- Peak RSS: `270.031 MiB`

## Top candidates/failures

| rank | name | family | survivor | success | train ret | val ret | OOS ret | OOS MDD | OOS Sharpe | OOS trips | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_calendar_teacher_state_longonly_lb168_fast72_z075_ret30_h168_fl2_xr125` | `calendar_teacher_state_similarity` | False | False | -2.2436% | -0.2643% | -0.1379% | +0.1461% | -4.807518 | 9 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 2 | `fresh_calendar_teacher_state_longonly_lb168_fast72_z075_ret60_h168_fl2_xr125` | `calendar_teacher_state_similarity` | False | False | -2.2505% | -0.2643% | -0.1379% | +0.1461% | -4.807518 | 9 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 3 | `fresh_calendar_teacher_state_longonly_lb168_fast72_z075_ret90_h168_fl2_xr125` | `calendar_teacher_state_similarity` | False | False | -2.2510% | -0.2643% | -0.1463% | +0.1531% | -5.123687 | 9 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 4 | `fresh_calendar_teacher_state_longonly_lb168_fast72_z075_ret30_h168_fl0_xr125` | `calendar_teacher_state_similarity` | False | False | -2.2389% | -0.2643% | -0.1530% | +0.1613% | -5.652796 | 10 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 5 | `fresh_calendar_teacher_state_longonly_lb168_fast72_z075_ret60_h168_fl0_xr125` | `calendar_teacher_state_similarity` | False | False | -2.2554% | -0.2643% | -0.1530% | +0.1613% | -5.652796 | 10 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 6 | `fresh_calendar_teacher_state_longonly_lb168_fast72_z075_ret90_h168_fl0_xr125` | `calendar_teacher_state_similarity` | False | False | -2.2838% | -0.2643% | -0.1614% | +0.1682% | -5.993795 | 10 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 7 | `fresh_calendar_teacher_state_longonly_lb336_fast72_z075_ret30_h168_fl2_xr125` | `calendar_teacher_state_similarity` | False | False | -1.6757% | -0.2257% | -0.1771% | +0.1771% | -7.230268 | 7 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 8 | `fresh_calendar_teacher_state_longonly_lb336_fast72_z075_ret60_h168_fl2_xr125` | `calendar_teacher_state_similarity` | False | False | -1.6503% | -0.2257% | -0.1771% | +0.1771% | -7.230268 | 7 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 9 | `fresh_calendar_teacher_state_longonly_lb336_fast72_z075_ret90_h168_fl2_xr125` | `calendar_teacher_state_similarity` | False | False | -1.6777% | -0.2257% | -0.1771% | +0.1771% | -7.230268 | 7 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 10 | `fresh_calendar_teacher_state_longonly_lb336_fast72_z075_ret30_h120_fl2_xr125` | `calendar_teacher_state_similarity` | False | False | -2.3995% | -0.3066% | -0.1771% | +0.1771% | -7.230391 | 7 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 11 | `fresh_calendar_teacher_state_longonly_lb336_fast72_z075_ret60_h120_fl2_xr125` | `calendar_teacher_state_similarity` | False | False | -2.3669% | -0.3066% | -0.1771% | +0.1771% | -7.230391 | 7 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 12 | `fresh_calendar_teacher_state_longonly_lb336_fast72_z075_ret90_h120_fl2_xr125` | `calendar_teacher_state_similarity` | False | False | -2.3740% | -0.3292% | -0.1771% | +0.1771% | -7.230391 | 7 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 13 | `fresh_calendar_teacher_state_longonly_lb336_fast72_z075_ret30_h120_fl0_xr125` | `calendar_teacher_state_similarity` | False | False | -2.4221% | -0.3104% | -0.1816% | +0.1816% | -7.690033 | 7 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 14 | `fresh_calendar_teacher_state_longonly_lb336_fast72_z075_ret30_h168_fl0_xr125` | `calendar_teacher_state_similarity` | False | False | -1.7776% | -0.2254% | -0.1816% | +0.1816% | -7.690033 | 7 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 15 | `fresh_calendar_teacher_state_longonly_lb336_fast72_z075_ret60_h120_fl0_xr125` | `calendar_teacher_state_similarity` | False | False | -2.4294% | -0.3104% | -0.1816% | +0.1816% | -7.690033 | 7 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |

## Decision

- No fresh-start candidate earned a full live-equivalent slot; do not promote or backtest a random vector-only shape.
- Blocked/failed families remain recorded in CSV/JSON with failed gates and top reject reasons.
