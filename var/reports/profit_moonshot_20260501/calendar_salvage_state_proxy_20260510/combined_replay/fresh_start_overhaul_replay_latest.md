# Profit moonshot fresh-start overhaul replay

Generated: `2026-05-10T13:47:28.021275Z`
OOS end date: `2026-05-09`

## Intent

- 기존 ETH shock-reversion incumbent/leadlag/context-wrapper를 쓰지 않고 raw-first data에서 새로 출발했다.
- 신규 후보군: cross-sectional residual reversal, cross-sectional momentum, adaptive trend, cross-sectional Sharpe/rank selector, funding-carry fade, funding+OI carry fade, taker-flow persistence/exhaustion, non-calendar TRX state-momentum proxy, non-calendar TRX/ETH state-relative-strength spread, calendar rotation, calendar-conditioned veto/day-window sleeves, TRX/ETH calendar spread, compression breakout.
- Replay는 one-position, fee/slippage, 10% bar-volume fill cap, cooldown, stop/take/max-hold, 0.8% target allocation, $175 max order를 강제한다.

## Gate policy

- Success requires OOS return > `+1.2181%`, OOS MDD < `0.1778%`, OOS Sharpe > `1.0`, liquidations `0`, and positive train/val.
- Replay survivor는 full live-equivalent raw-first backtest 후보일 뿐이며, sub-1 Sharpe는 성공이 아니다.

## Result

- Specs evaluated: `7092`
- Replay survivors: `0`
- Success candidates: `0`
- Peak RSS: `321.875 MiB`

## Top candidates/failures

| rank | name | family | survivor | success | train ret | val ret | OOS ret | OOS MDD | OOS Sharpe | OOS trips | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_state_trx_mom_lb168_z075_ret120_h96_ls620_ss100_tp240` | `state_momentum_proxy` | False | False | -1.0327% | +0.7533% | +1.3339% | +0.4353% | 4.873412 | 13 | `train_positive,oos_mdd_beats_shadow` |
| 2 | `fresh_state_trx_mom_lb168_z075_ret120_h96_ls590_ss100_tp240` | `state_momentum_proxy` | False | False | -1.0132% | +0.7643% | +1.3014% | +0.4313% | 4.799967 | 13 | `train_positive,oos_mdd_beats_shadow` |
| 3 | `fresh_state_trx_longonly_lb168_z075_ret60_h168_ls800_tp600` | `state_momentum_proxy` | False | False | -1.3038% | -0.2374% | +1.2496% | +0.2412% | 5.922207 | 5 | `train_positive,val_positive,oos_mdd_beats_shadow` |
| 4 | `fresh_state_trx_longonly_lb168_z075_ret120_h168_ls800_tp600` | `state_momentum_proxy` | False | False | -1.1292% | -0.2374% | +1.2496% | +0.2412% | 5.879793 | 5 | `train_positive,val_positive,oos_mdd_beats_shadow` |
| 5 | `fresh_state_trx_longonly_lb168_z075_ret60_h168_ls800_tp450` | `state_momentum_proxy` | False | False | -1.1461% | -0.1643% | +1.2279% | +0.1690% | 6.401440 | 6 | `train_positive,val_positive` |
| 6 | `fresh_state_trx_longonly_lb168_z075_ret120_h168_ls800_tp450` | `state_momentum_proxy` | False | False | -0.9714% | -0.1643% | +1.2279% | +0.1690% | 6.313641 | 6 | `train_positive,val_positive` |
| 7 | `fresh_state_trx_longonly_lb168_z050_ret60_h168_ls800_tp450` | `state_momentum_proxy` | False | False | -1.5832% | -0.1100% | +1.2103% | +0.1690% | 6.307961 | 6 | `train_positive,val_positive,oos_return_beats_incumbent` |
| 8 | `fresh_state_trx_longonly_lb168_z050_ret120_h168_ls800_tp450` | `state_momentum_proxy` | False | False | -1.4806% | -0.1100% | +1.2103% | +0.1690% | 6.221512 | 6 | `train_positive,val_positive,oos_return_beats_incumbent` |
| 9 | `fresh_state_trx_longonly_lb168_z100_ret120_h168_ls800_tp450` | `state_momentum_proxy` | False | False | -0.4513% | -0.1942% | +1.1697% | +0.1690% | 6.337289 | 5 | `train_positive,val_positive,oos_return_beats_incumbent` |
| 10 | `fresh_state_trx_longonly_lb168_z100_ret60_h168_ls800_tp450` | `state_momentum_proxy` | False | False | -0.3088% | -0.1942% | +1.1697% | +0.1690% | 6.435316 | 5 | `train_positive,val_positive,oos_return_beats_incumbent` |
| 11 | `fresh_state_trx_longonly_lb168_z075_ret150_h168_ls800_tp600` | `state_momentum_proxy` | False | False | -0.5970% | -0.2374% | +1.1669% | +0.3039% | 5.450468 | 5 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 12 | `fresh_state_trx_longonly_lb168_z050_ret150_h168_ls800_tp600` | `state_momentum_proxy` | False | False | -1.4894% | -0.0656% | +1.1604% | +0.2987% | 5.416854 | 5 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 13 | `fresh_state_trx_longonly_lb72_z050_ret120_h168_ls800_tp450` | `state_momentum_proxy` | False | False | +0.4134% | -0.0306% | +1.1578% | +0.1999% | 5.106289 | 7 | `val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 14 | `fresh_state_trx_mom_lb168_z075_ret180_h96_ls620_ss100_tp240` | `state_momentum_proxy` | False | False | -1.2220% | +0.7533% | +1.1534% | +0.4365% | 4.445713 | 12 | `train_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 15 | `fresh_state_trx_longonly_lb168_z075_ret150_h168_ls800_tp450` | `state_momentum_proxy` | False | False | -0.3990% | -0.1643% | +1.1468% | +0.1690% | 5.819563 | 5 | `train_positive,val_positive,oos_return_beats_incumbent` |

## Decision

- No fresh-start candidate earned a full live-equivalent slot; do not promote or backtest a random vector-only shape.
- Blocked/failed families remain recorded in CSV/JSON with failed gates and top reject reasons.
