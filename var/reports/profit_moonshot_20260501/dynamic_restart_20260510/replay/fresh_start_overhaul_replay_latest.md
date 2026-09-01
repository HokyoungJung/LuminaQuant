# Profit moonshot fresh-start overhaul replay

Generated: `2026-05-10T11:23:19.297941Z`
OOS end date: `2026-05-09`

## Intent

- 기존 ETH shock-reversion incumbent/leadlag/context-wrapper를 쓰지 않고 raw-first data에서 새로 출발했다.
- 신규 후보군: cross-sectional residual reversal, cross-sectional momentum, adaptive trend, cross-sectional Sharpe/rank selector, funding-carry fade, funding+OI carry fade, taker-flow persistence/exhaustion, calendar rotation, calendar-conditioned veto/day-window sleeves, TRX/ETH calendar spread, compression breakout.
- Replay는 one-position, fee/slippage, 10% bar-volume fill cap, cooldown, stop/take/max-hold, 0.8% target allocation, $175 max order를 강제한다.

## Gate policy

- Success requires OOS return > `+1.2181%`, OOS MDD < `0.1778%`, OOS Sharpe > `1.0`, liquidations `0`, and positive train/val.
- Replay survivor는 full live-equivalent raw-first backtest 후보일 뿐이며, sub-1 Sharpe는 성공이 아니다.

## Result

- Specs evaluated: `120`
- Replay survivors: `0`
- Success candidates: `0`
- Peak RSS: `249.406 MiB`

## Top candidates/failures

| rank | name | family | survivor | success | train ret | val ret | OOS ret | OOS MDD | OOS Sharpe | OOS trips | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_flow_imb_persist_fl3_px3_imb15_h48` | `flow_imbalance_persistence` | False | False | +0.0000% | -0.0128% | +0.0025% | +0.0038% | 0.893961 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 2 | `fresh_flow_imb_persist_fl3_px3_imb15_h12` | `flow_imbalance_persistence` | False | False | +0.0000% | -0.0147% | +0.0009% | +0.0052% | 0.235186 | 2 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 3 | `fresh_flow_imb_persist_fl3_px3_imb15_h24` | `flow_imbalance_persistence` | False | False | +0.0000% | -0.0105% | +0.0009% | +0.0052% | 0.235186 | 2 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 4 | `fresh_flow_imb_persist_fl3_px6_imb15_h48` | `flow_imbalance_persistence` | False | False | +0.0000% | -0.0128% | +0.0003% | +0.0060% | 0.093972 | 2 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 5 | `fresh_flow_imb_persist_fl3_px12_imb15_h48` | `flow_imbalance_persistence` | False | False | +0.0000% | -0.0128% | +0.0003% | +0.0060% | 0.093972 | 2 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 6 | `fresh_flow_imb_persist_fl3_px6_imb15_h12` | `flow_imbalance_persistence` | False | False | +0.0000% | -0.0147% | -0.0012% | +0.0076% | -0.296334 | 3 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 7 | `fresh_flow_imb_persist_fl3_px6_imb15_h24` | `flow_imbalance_persistence` | False | False | +0.0000% | -0.0105% | -0.0012% | +0.0076% | -0.296334 | 3 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 8 | `fresh_flow_imb_persist_fl3_px12_imb15_h12` | `flow_imbalance_persistence` | False | False | +0.0000% | -0.0147% | -0.0012% | +0.0076% | -0.296334 | 3 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 9 | `fresh_flow_imb_persist_fl3_px12_imb15_h24` | `flow_imbalance_persistence` | False | False | +0.0000% | -0.0105% | -0.0012% | +0.0076% | -0.296334 | 3 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 10 | `fresh_flow_imb_persist_fl3_px3_imb15_h6` | `flow_imbalance_persistence` | False | False | +0.0000% | -0.0155% | -0.0068% | +0.0117% | -1.278437 | 3 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 11 | `fresh_flow_imb_persist_fl3_px6_imb15_h6` | `flow_imbalance_persistence` | False | False | +0.0000% | -0.0155% | -0.0090% | +0.0141% | -1.646893 | 4 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 12 | `fresh_flow_imb_persist_fl3_px12_imb15_h6` | `flow_imbalance_persistence` | False | False | +0.0000% | -0.0155% | -0.0090% | +0.0141% | -1.646893 | 4 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 13 | `fresh_flow_imb_persist_fl3_px12_imb10_h48` | `flow_imbalance_persistence` | False | False | -0.1590% | -0.0229% | -0.0309% | +0.0365% | -2.522631 | 11 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 14 | `fresh_flow_imb_persist_fl3_px3_imb10_h48` | `flow_imbalance_persistence` | False | False | -0.1752% | -0.0341% | -0.0346% | +0.0424% | -3.019720 | 10 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |
| 15 | `fresh_flow_imb_persist_fl3_px12_imb10_h12` | `flow_imbalance_persistence` | False | False | -0.1525% | -0.0348% | -0.0456% | +0.0530% | -3.946975 | 13 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1` |

## Decision

- No fresh-start candidate earned a full live-equivalent slot; do not promote or backtest a random vector-only shape.
- Blocked/failed families remain recorded in CSV/JSON with failed gates and top reject reasons.

