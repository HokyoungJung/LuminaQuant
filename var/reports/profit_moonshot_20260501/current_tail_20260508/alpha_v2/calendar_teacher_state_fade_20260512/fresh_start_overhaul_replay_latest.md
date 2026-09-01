# Profit moonshot fresh-start overhaul replay

Generated: `2026-05-12T13:20:10.460426Z`
OOS end date: `2026-05-06`

## Intent

- 기존 ETH shock-reversion incumbent/leadlag/context-wrapper를 쓰지 않고 raw-first data에서 새로 출발했다.
- 신규 후보군: cross-sectional residual reversal, cross-sectional momentum, adaptive trend, cross-sectional Sharpe/rank selector, funding-carry fade, funding+OI carry fade, taker-flow persistence/exhaustion, non-calendar state-distilled leadership/unwind, non-calendar crowded-leadership unwind v2, non-calendar TRX state-momentum proxy, non-calendar TRX/ETH state-relative-strength spread, calendar rotation, calendar-conditioned veto/day-window sleeves, TRX/ETH calendar spread, compression breakout.
- Replay는 one-position, fee/slippage, 10% bar-volume fill cap, cooldown, stop/take/max-hold, 0.8% target allocation, $175 max order를 강제한다.

## Gate policy

- Success requires OOS return > `+1.2181%`, OOS MDD < `0.1778%`, OOS Sharpe > `1.0`, liquidations `0`, and positive train/val.
- Replay survivor는 full live-equivalent raw-first backtest 후보일 뿐이며, sub-1 Sharpe는 성공이 아니다.

## Result

- Specs evaluated: `324`
- Replay survivors: `0`
- Success candidates: `0`
- Peak RSS: `268.234 MiB`

## Top candidates/failures

| rank | name | family | survivor | success | train ret | val ret | OOS ret | OOS MDD | OOS Sharpe | OOS trips | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_calendar_teacher_fade_both_lb336_fast72_z075_ret90_h168_fl0_xr125` | `calendar_teacher_state_fade` | False | False | -4.3620% | -0.8479% | -0.5818% | +0.5818% | -12.910173 | 27 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 2 | `fresh_calendar_teacher_fade_both_lb336_fast72_z075_ret60_h168_fl0_xr125` | `calendar_teacher_state_fade` | False | False | -4.3557% | -0.8177% | -0.5916% | +0.5916% | -12.905490 | 27 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 3 | `fresh_calendar_teacher_fade_both_lb336_fast72_z075_ret30_h168_fl0_xr125` | `calendar_teacher_state_fade` | False | False | -4.3805% | -0.8235% | -0.6166% | +0.6166% | -12.804195 | 26 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 4 | `fresh_calendar_teacher_fade_both_lb336_fast72_z075_ret60_h168_fl2_xr125` | `calendar_teacher_state_fade` | False | False | -4.1918% | -0.9010% | -0.6241% | +0.6241% | -14.679649 | 27 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 5 | `fresh_calendar_teacher_fade_both_lb336_fast72_z050_ret30_h168_fl0_xr125` | `calendar_teacher_state_fade` | False | False | -4.4396% | -0.8768% | -0.6245% | +0.6245% | -14.832814 | 31 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 6 | `fresh_calendar_teacher_fade_both_lb336_fast72_z075_ret30_h168_fl2_xr125` | `calendar_teacher_state_fade` | False | False | -4.2277% | -0.9068% | -0.6309% | +0.6309% | -14.496312 | 26 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 7 | `fresh_calendar_teacher_fade_both_lb336_fast72_z050_ret90_h168_fl0_xr125` | `calendar_teacher_state_fade` | False | False | -4.1556% | -0.8465% | -0.6345% | +0.6345% | -13.418149 | 29 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 8 | `fresh_calendar_teacher_fade_both_lb336_fast72_z050_ret60_h168_fl0_xr125` | `calendar_teacher_state_fade` | False | False | -4.1928% | -0.8873% | -0.6356% | +0.6356% | -13.409363 | 29 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 9 | `fresh_calendar_teacher_fade_both_lb336_fast72_z075_ret90_h168_fl2_xr125` | `calendar_teacher_state_fade` | False | False | -4.1909% | -0.8868% | -0.6388% | +0.6388% | -14.880813 | 27 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 10 | `fresh_calendar_teacher_fade_both_lb336_fast72_z075_ret60_h168_fl2` | `calendar_teacher_state_fade` | False | False | -4.2195% | -0.8853% | -0.6477% | +0.6477% | -14.572793 | 27 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 11 | `fresh_calendar_teacher_fade_both_lb336_fast72_z035_ret30_h168_fl0_xr125` | `calendar_teacher_state_fade` | False | False | -4.8605% | -0.8503% | -0.6486% | +0.6486% | -15.682464 | 32 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 12 | `fresh_calendar_teacher_fade_both_lb336_fast72_z050_ret60_h168_fl0` | `calendar_teacher_state_fade` | False | False | -4.1846% | -0.8729% | -0.6527% | +0.6527% | -15.166752 | 30 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 13 | `fresh_calendar_teacher_fade_both_lb336_fast72_z075_ret30_h168_fl2` | `calendar_teacher_state_fade` | False | False | -4.2096% | -0.8911% | -0.6546% | +0.6546% | -14.412534 | 26 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 14 | `fresh_calendar_teacher_fade_both_lb336_fast72_z050_ret90_h168_fl0` | `calendar_teacher_state_fade` | False | False | -4.1475% | -0.8283% | -0.6593% | +0.6593% | -14.229828 | 29 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |
| 15 | `fresh_calendar_teacher_fade_both_lb336_fast72_z075_ret90_h168_fl2` | `calendar_teacher_state_fade` | False | False | -4.2242% | -0.8711% | -0.6625% | +0.6625% | -14.771409 | 27 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow,oos_sharpe_gt_1` |

## Decision

- No fresh-start candidate earned a full live-equivalent slot; do not promote or backtest a random vector-only shape.
- Blocked/failed families remain recorded in CSV/JSON with failed gates and top reject reasons.
