# Profit moonshot fresh-start overhaul replay

Generated: `2026-05-07T11:01:45.594094Z`
OOS end date: `2026-05-06`

## Intent

- 기존 ETH shock-reversion incumbent/leadlag/context-wrapper를 쓰지 않고 raw-first data에서 새로 출발했다.
- 신규 후보군: cross-sectional residual reversal, cross-sectional momentum, funding-carry fade, taker-flow momentum/exhaustion, compression breakout.
- Replay는 one-position, fee/slippage, 10% bar-volume fill cap, cooldown, stop/take/max-hold, 0.8% target allocation, $175 max order를 강제한다.

## Gate policy

- Success requires OOS return > `+0.8284%`, OOS MDD < `0.1778%`, OOS Sharpe > `1.0`, liquidations `0`, and positive train/val.
- Replay survivor는 full live-equivalent raw-first backtest 후보일 뿐이며, sub-1 Sharpe는 성공이 아니다.

## Result

- Specs evaluated: `1219`
- Replay survivors: `0`
- Success candidates: `0`
- Peak RSS: `2547.137 MiB`

## Top candidates/failures

| rank | name | family | survivor | success | train ret | val ret | OOS ret | OOS MDD | OOS Sharpe | OOS trips | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_resid_rev_lb24_z175_h72_asia_us` | `residual_reversion` | False | False | -0.5135% | -0.0113% | +0.2021% | +0.0504% | 0.908931 | 22 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 2 | `fresh_resid_rev_lb24_z175_h48_asia_us` | `residual_reversion` | False | False | -0.7756% | -0.0428% | +0.1882% | +0.0422% | 0.885538 | 25 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 3 | `fresh_xs_mom_lb12_z10_h96` | `cross_momentum` | False | False | -0.6332% | +0.0886% | +0.1307% | +0.0559% | 0.478401 | 26 | `train_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 4 | `fresh_resid_rev_lb24_z175_h72_all` | `residual_reversion` | False | False | -0.6996% | +0.1220% | +0.1296% | +0.0371% | 0.659074 | 21 | `train_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 5 | `fresh_resid_rev_rvcap5146_lb48_z125` | `residual_reversion` | False | False | -0.4194% | +0.1745% | +0.1077% | +0.0679% | 0.612466 | 26 | `train_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 6 | `fresh_resid_rev_lb24_z15_h72_all` | `residual_reversion` | False | False | -0.7848% | +0.0881% | +0.1068% | +0.0443% | 0.489801 | 26 | `train_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 7 | `fresh_xs_mom_lb12_z15_h96` | `cross_momentum` | False | False | -0.3781% | +0.0144% | +0.1040% | +0.0619% | 0.408831 | 23 | `train_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 8 | `fresh_resid_rev_rvcap3156_lb12_z175` | `residual_reversion` | False | False | +0.0228% | +0.0347% | +0.1040% | +0.0262% | 0.869259 | 16 | `oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 9 | `fresh_xs_mom_lb6_z125_h48` | `cross_momentum` | False | False | -0.6401% | +0.0175% | +0.1006% | +0.1328% | 0.346960 | 40 | `train_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 10 | `fresh_resid_rev_rvcap3156_lb12_z125` | `residual_reversion` | False | False | -0.0896% | -0.0999% | +0.0864% | +0.0665% | 0.527684 | 22 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 11 | `fresh_resid_rev_lb24_z20_h48_asia_us` | `residual_reversion` | False | False | -0.6849% | +0.0506% | +0.0840% | +0.0469% | 0.435424 | 22 | `train_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 12 | `fresh_resid_rev_rvcap4654_lb48_z125` | `residual_reversion` | False | False | -0.4279% | +0.1090% | +0.0777% | +0.0735% | 0.465101 | 25 | `train_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 13 | `fresh_resid_rev_rvcap5146_lb48_z15` | `residual_reversion` | False | False | -0.2001% | +0.1814% | +0.0684% | +0.0569% | 0.389066 | 25 | `train_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 14 | `fresh_resid_rev_lb24_z15_h72_asia_us` | `residual_reversion` | False | False | -0.6688% | -0.1370% | +0.0672% | +0.0667% | 0.283489 | 26 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 15 | `fresh_resid_rev_rvcap4654_lb24_z175` | `residual_reversion` | False | False | -0.0341% | +0.0232% | +0.0608% | +0.0618% | 0.361771 | 24 | `train_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |

## Decision

- No fresh-start candidate earned a full live-equivalent slot; do not promote or backtest a random vector-only shape.
- Blocked/failed families remain recorded in CSV/JSON with failed gates and top reject reasons.

