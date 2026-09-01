# Profit moonshot fresh-start overhaul replay

Generated: `2026-05-10T11:29:29.792486Z`
OOS end date: `2026-05-09`

## Intent

- 기존 ETH shock-reversion incumbent/leadlag/context-wrapper를 쓰지 않고 raw-first data에서 새로 출발했다.
- 신규 후보군: cross-sectional residual reversal, cross-sectional momentum, adaptive trend, cross-sectional Sharpe/rank selector, funding-carry fade, funding+OI carry fade, taker-flow persistence/exhaustion, calendar rotation, calendar-conditioned veto/day-window sleeves, TRX/ETH calendar spread, compression breakout.
- Replay는 one-position, fee/slippage, 10% bar-volume fill cap, cooldown, stop/take/max-hold, 0.8% target allocation, $175 max order를 강제한다.

## Gate policy

- Success requires OOS return > `+1.2181%`, OOS MDD < `0.1778%`, OOS Sharpe > `1.0`, liquidations `0`, and positive train/val.
- Replay survivor는 full live-equivalent raw-first backtest 후보일 뿐이며, sub-1 Sharpe는 성공이 아니다.

## Result

- Specs evaluated: `4429`
- Replay survivors: `0`
- Success candidates: `0`
- Peak RSS: `288.621 MiB`

## Top candidates/failures

| rank | name | family | survivor | success | train ret | val ret | OOS ret | OOS MDD | OOS Sharpe | OOS trips | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_pair_resid_revert_spread_lb24_z050_h48_sc30_st100_tp400_postfund` | `residual_pair_reversion_spread` | False | False | -3.5153% | -0.5208% | +0.6779% | +0.2428% | 4.499169 | 27 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 2 | `fresh_pair_resid_revert_spread_lb24_z050_h72_sc30_st60_tp240_postfund` | `residual_pair_reversion_spread` | False | False | -2.7935% | -0.2047% | +0.6716% | +0.1610% | 5.371250 | 24 | `train_positive,val_positive,oos_return_beats_incumbent` |
| 3 | `fresh_pair_resid_revert_spread_lb24_z075_h48_sc30_st100_tp400_postfund` | `residual_pair_reversion_spread` | False | False | -3.3558% | -0.5874% | +0.6453% | +0.2725% | 4.285253 | 27 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 4 | `fresh_pair_resid_revert_spread_lb24_z050_h72_sc30_st100_tp240_postfund` | `residual_pair_reversion_spread` | False | False | -1.8823% | -0.3540% | +0.6354% | +0.1606% | 4.859954 | 20 | `train_positive,val_positive,oos_return_beats_incumbent` |
| 5 | `fresh_pair_resid_revert_spread_lb24_z075_h72_sc30_st60_tp240_all` | `residual_pair_reversion_spread` | False | False | -2.5112% | -0.4165% | +0.5369% | +0.1549% | 4.299440 | 24 | `train_positive,val_positive,oos_return_beats_incumbent` |
| 6 | `fresh_pair_resid_revert_spread_lb24_z050_h72_sc30_st100_tp240_asiaus` | `residual_pair_reversion_spread` | False | False | -2.1820% | +0.0922% | +0.5349% | +0.2046% | 3.867026 | 21 | `train_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 7 | `fresh_pair_resid_revert_spread_lb24_z150_h120_sc30_st100_tp240_postfund` | `residual_pair_reversion_spread` | False | False | -1.9284% | -0.0814% | +0.5252% | +0.1400% | 5.467341 | 10 | `train_positive,val_positive,oos_return_beats_incumbent` |
| 8 | `fresh_pair_resid_revert_spread_lb24_z150_h72_sc30_st100_tp400_postfund` | `residual_pair_reversion_spread` | False | False | -2.2239% | -0.2539% | +0.4915% | +0.1200% | 4.807005 | 11 | `train_positive,val_positive,oos_return_beats_incumbent` |
| 9 | `fresh_pair_resid_revert_spread_lb24_z050_h48_sc20_st100_tp400_postfund` | `residual_pair_reversion_spread` | False | False | -2.4297% | -0.3424% | +0.4509% | +0.1623% | 4.493443 | 27 | `train_positive,val_positive,oos_return_beats_incumbent` |
| 10 | `fresh_pair_resid_revert_spread_lb24_z050_h72_sc20_st60_tp240_postfund` | `residual_pair_reversion_spread` | False | False | -1.6799% | +0.0292% | +0.4474% | +0.1076% | 5.374852 | 24 | `train_positive,oos_return_beats_incumbent` |
| 11 | `fresh_pair_resid_revert_spread_lb24_z075_h48_sc20_st100_tp400_postfund` | `residual_pair_reversion_spread` | False | False | -1.6414% | -0.0836% | +0.4296% | +0.1817% | 4.280435 | 27 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |
| 12 | `fresh_pair_resid_revert_spread_lb24_z050_h72_sc20_st100_tp240_postfund` | `residual_pair_reversion_spread` | False | False | -1.7578% | -0.1916% | +0.4223% | +0.1081% | 4.859739 | 20 | `train_positive,val_positive,oos_return_beats_incumbent` |
| 13 | `fresh_pair_resid_revert_spread_lb24_z150_h120_sc30_st100_tp120_postfund` | `residual_pair_reversion_spread` | False | False | -0.7180% | -0.0206% | +0.4147% | +0.0731% | 6.109168 | 11 | `train_positive,val_positive,oos_return_beats_incumbent` |
| 14 | `fresh_pair_resid_revert_spread_lb24_z150_h72_sc30_st100_tp240_postfund` | `residual_pair_reversion_spread` | False | False | -2.6282% | -0.0671% | +0.3976% | +0.1427% | 4.165612 | 13 | `train_positive,val_positive,oos_return_beats_incumbent` |
| 15 | `fresh_pair_resid_revert_spread_lb24_z100_h72_sc30_st60_tp240_all` | `residual_pair_reversion_spread` | False | False | -3.3944% | -0.2985% | +0.3693% | +0.2456% | 2.668350 | 23 | `train_positive,val_positive,oos_return_beats_incumbent,oos_mdd_beats_shadow` |

## Decision

- No fresh-start candidate earned a full live-equivalent slot; do not promote or backtest a random vector-only shape.
- Blocked/failed families remain recorded in CSV/JSON with failed gates and top reject reasons.

