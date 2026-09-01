# Profit moonshot fresh-start overhaul replay

Generated: `2026-05-09T05:19:54.000280Z`
OOS end date: `2026-05-06`

## Intent

- 기존 ETH shock-reversion incumbent/leadlag/context-wrapper를 쓰지 않고 raw-first data에서 새로 출발했다.
- 신규 후보군: cross-sectional residual reversal, cross-sectional momentum, adaptive trend, cross-sectional Sharpe/rank selector, funding-carry fade, funding+OI carry fade, taker-flow persistence/exhaustion, calendar rotation, calendar-conditioned veto/day-window sleeves, TRX/ETH calendar spread, compression breakout.
- Replay는 one-position, fee/slippage, 10% bar-volume fill cap, cooldown, stop/take/max-hold, 0.8% target allocation, $175 max order를 강제한다.

## Gate policy

- Success requires OOS return > `+1.2181%`, OOS MDD < `0.1778%`, OOS Sharpe > `1.0`, liquidations `0`, and positive train/val.
- Replay survivor는 full live-equivalent raw-first backtest 후보일 뿐이며, sub-1 Sharpe는 성공이 아니다.

## Result

- Specs evaluated: `432`
- Replay survivors: `0`
- Success candidates: `0`
- Peak RSS: `251.863 MiB`

## Top candidates/failures

| rank | name | family | survivor | success | train ret | val ret | OOS ret | OOS MDD | OOS Sharpe | OOS trips | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_pair_resid_mom_spread_lb48_z200_h24_sc05_st100_asiaus` | `residual_pair_momentum_spread` | False | False | -0.0858% | -0.0309% | +0.0046% | +0.0101% | 0.780211 | 3 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 2 | `fresh_pair_resid_mom_spread_lb48_z200_h24_sc05_st60_asiaus` | `residual_pair_momentum_spread` | False | False | -0.0757% | -0.0517% | -0.0037% | +0.0130% | -0.851927 | 3 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 3 | `fresh_pair_resid_mom_spread_lb12_z100_h24_sc05_st60_asiaus` | `residual_pair_momentum_spread` | False | False | -0.5210% | -0.0619% | -0.0088% | +0.0660% | -0.406177 | 30 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 4 | `fresh_pair_resid_mom_spread_lb48_z200_h24_sc05_st100_all` | `residual_pair_momentum_spread` | False | False | +0.0011% | -0.0314% | -0.0089% | +0.0121% | -1.295280 | 3 | `val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 5 | `fresh_pair_resid_mom_spread_lb48_z200_h48_sc05_st60_asiaus` | `residual_pair_momentum_spread` | False | False | -0.0862% | -0.0574% | -0.0116% | +0.0209% | -2.490906 | 3 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 6 | `fresh_pair_resid_mom_spread_lb48_z200_h72_sc05_st60_asiaus` | `residual_pair_momentum_spread` | False | False | -0.0722% | -0.0492% | -0.0116% | +0.0209% | -2.490906 | 3 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 7 | `fresh_pair_resid_mom_spread_lb48_z200_h48_sc05_st100_asiaus` | `residual_pair_momentum_spread` | False | False | -0.1057% | -0.0150% | -0.0181% | +0.0274% | -2.266638 | 3 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 8 | `fresh_pair_resid_mom_spread_lb48_z200_h72_sc05_st100_asiaus` | `residual_pair_momentum_spread` | False | False | -0.1027% | -0.0255% | -0.0181% | +0.0274% | -2.266638 | 3 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 9 | `fresh_pair_resid_mom_spread_lb12_z075_h24_sc05_st60_asiaus` | `residual_pair_momentum_spread` | False | False | -0.8401% | -0.0704% | -0.0211% | +0.0842% | -0.902820 | 39 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 10 | `fresh_pair_resid_mom_spread_lb48_z200_h48_sc05_st60_all` | `residual_pair_momentum_spread` | False | False | -0.0593% | -0.0570% | -0.0245% | +0.0245% | -5.183383 | 3 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 11 | `fresh_pair_resid_mom_spread_lb48_z200_h72_sc05_st60_all` | `residual_pair_momentum_spread` | False | False | -0.0365% | -0.0567% | -0.0245% | +0.0245% | -5.183383 | 3 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 12 | `fresh_pair_resid_mom_spread_lb48_z150_h72_sc05_st60_all` | `residual_pair_momentum_spread` | False | False | -0.2703% | -0.1290% | -0.0295% | +0.0419% | -2.147333 | 12 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 13 | `fresh_pair_resid_mom_spread_lb48_z150_h24_sc05_st100_asiaus` | `residual_pair_momentum_spread` | False | False | -0.1345% | -0.0718% | -0.0300% | +0.0424% | -2.582673 | 10 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 14 | `fresh_pair_resid_mom_spread_lb48_z200_h48_sc05_st100_all` | `residual_pair_momentum_spread` | False | False | -0.0483% | -0.0302% | -0.0300% | +0.0332% | -3.416199 | 3 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 15 | `fresh_pair_resid_mom_spread_lb48_z200_h72_sc05_st100_all` | `residual_pair_momentum_spread` | False | False | -0.0380% | -0.0260% | -0.0300% | +0.0332% | -3.416199 | 3 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |

## Decision

- No fresh-start candidate earned a full live-equivalent slot; do not promote or backtest a random vector-only shape.
- Blocked/failed families remain recorded in CSV/JSON with failed gates and top reject reasons.
