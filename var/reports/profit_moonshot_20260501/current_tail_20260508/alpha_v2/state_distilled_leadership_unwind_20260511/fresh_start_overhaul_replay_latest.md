# Profit moonshot fresh-start overhaul replay

Generated: `2026-05-11T13:31:41.168854Z`
OOS end date: `2026-05-06`

## Intent

- 기존 ETH shock-reversion incumbent/leadlag/context-wrapper를 쓰지 않고 raw-first data에서 새로 출발했다.
- 신규 후보군: cross-sectional residual reversal, cross-sectional momentum, adaptive trend, cross-sectional Sharpe/rank selector, funding-carry fade, funding+OI carry fade, taker-flow persistence/exhaustion, non-calendar state-distilled leadership/unwind, non-calendar TRX state-momentum proxy, non-calendar TRX/ETH state-relative-strength spread, calendar rotation, calendar-conditioned veto/day-window sleeves, TRX/ETH calendar spread, compression breakout.
- Replay는 one-position, fee/slippage, 10% bar-volume fill cap, cooldown, stop/take/max-hold, 0.8% target allocation, $175 max order를 강제한다.

## Gate policy

- Success requires OOS return > `+1.2181%`, OOS MDD < `0.1778%`, OOS Sharpe > `1.0`, liquidations `0`, and positive train/val.
- Replay survivor는 full live-equivalent raw-first backtest 후보일 뿐이며, sub-1 Sharpe는 성공이 아니다.

## Result

- Specs evaluated: `648`
- Replay survivors: `0`
- Success candidates: `0`
- Peak RSS: `253.992 MiB`

## Top candidates/failures

| rank | name | family | survivor | success | train ret | val ret | OOS ret | OOS MDD | OOS Sharpe | OOS trips | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_state_distilled_longonly_lb72_fast24_z100_ret180_h168_ls620_ss0_tp600` | `state_distilled_leadership_unwind` | False | False | -1.9941% | -0.9736% | +1.0531% | +0.4086% | 3.213511 | 8 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 2 | `fresh_state_distilled_both_lb72_fast24_z075_ret60_h168_ls590_ss100_tp450` | `state_distilled_leadership_unwind` | False | False | -5.2048% | -3.0621% | +1.0115% | +0.6164% | 2.857708 | 11 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 3 | `fresh_state_distilled_both_lb72_fast24_z075_ret120_h168_ls590_ss100_tp450` | `state_distilled_leadership_unwind` | False | False | -5.2048% | -3.0510% | +1.0115% | +0.6164% | 2.857708 | 11 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 4 | `fresh_state_distilled_longonly_lb168_fast72_z075_ret60_h168_ls620_ss0_tp450` | `state_distilled_leadership_unwind` | False | False | +1.8894% | -0.7268% | +0.9761% | +0.2322% | 4.587476 | 8 | `val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 5 | `fresh_state_distilled_longonly_lb168_fast72_z075_ret120_h168_ls620_ss0_tp450` | `state_distilled_leadership_unwind` | False | False | +1.8972% | -0.7268% | +0.9761% | +0.2411% | 4.556583 | 8 | `val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 6 | `fresh_state_distilled_longonly_lb72_fast24_z050_ret180_h168_ls620_ss0_tp600` | `state_distilled_leadership_unwind` | False | False | -2.3114% | -1.4615% | +0.9752% | +0.4086% | 2.940311 | 9 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 7 | `fresh_state_distilled_longonly_lb72_fast24_z075_ret180_h168_ls620_ss0_tp600` | `state_distilled_leadership_unwind` | False | False | -1.3554% | -1.4615% | +0.9726% | +0.4086% | 2.934980 | 9 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 8 | `fresh_state_distilled_longonly_lb168_fast72_z050_ret60_h168_ls620_ss0_tp600_fl3` | `state_distilled_leadership_unwind` | False | False | -0.9835% | -2.0674% | +0.9712% | +0.3276% | 3.556935 | 5 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 9 | `fresh_state_distilled_longonly_lb168_fast72_z050_ret120_h168_ls620_ss0_tp600_fl3` | `state_distilled_leadership_unwind` | False | False | -2.2654% | -2.0674% | +0.9712% | +0.3276% | 3.556935 | 5 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 10 | `fresh_state_distilled_longonly_lb168_fast72_z050_ret180_h168_ls620_ss0_tp600_fl3` | `state_distilled_leadership_unwind` | False | False | -1.2332% | -2.1110% | +0.9712% | +0.3276% | 3.556935 | 5 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 11 | `fresh_state_distilled_longonly_lb168_fast72_z050_ret120_h120_ls620_ss0_tp450` | `state_distilled_leadership_unwind` | False | False | -0.8229% | -1.2432% | +0.9670% | +0.6999% | 2.894263 | 13 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 12 | `fresh_state_distilled_longonly_lb168_fast72_z050_ret60_h120_ls620_ss0_tp450` | `state_distilled_leadership_unwind` | False | False | -0.8117% | -1.2432% | +0.9670% | +0.6999% | 2.902220 | 13 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 13 | `fresh_state_distilled_longonly_lb168_fast72_z050_ret60_h120_ls620_ss0_tp600` | `state_distilled_leadership_unwind` | False | False | -1.5496% | -1.3348% | +0.9509% | +0.6606% | 2.654111 | 12 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 14 | `fresh_state_distilled_longonly_lb168_fast72_z050_ret120_h120_ls620_ss0_tp600` | `state_distilled_leadership_unwind` | False | False | -2.8399% | -1.3348% | +0.9509% | +0.6606% | 2.654111 | 12 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 15 | `fresh_state_distilled_longonly_lb168_fast72_z050_ret180_h120_ls620_ss0_tp600` | `state_distilled_leadership_unwind` | False | False | -2.8323% | -1.5822% | +0.9509% | +0.6606% | 2.654111 | 12 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |

## Decision

- No fresh-start candidate earned a full live-equivalent slot; do not promote or backtest a random vector-only shape.
- Blocked/failed families remain recorded in CSV/JSON with failed gates and top reject reasons.

