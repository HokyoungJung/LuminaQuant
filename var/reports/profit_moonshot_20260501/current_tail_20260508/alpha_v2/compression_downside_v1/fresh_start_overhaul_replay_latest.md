# Profit moonshot fresh-start overhaul replay

Generated: `2026-05-09T05:20:18.697168Z`
OOS end date: `2026-05-06`

## Intent

- 기존 ETH shock-reversion incumbent/leadlag/context-wrapper를 쓰지 않고 raw-first data에서 새로 출발했다.
- 신규 후보군: cross-sectional residual reversal, cross-sectional momentum, adaptive trend, cross-sectional Sharpe/rank selector, funding-carry fade, funding+OI carry fade, taker-flow persistence/exhaustion, calendar rotation, calendar-conditioned veto/day-window sleeves, TRX/ETH calendar spread, compression breakout.
- Replay는 one-position, fee/slippage, 10% bar-volume fill cap, cooldown, stop/take/max-hold, 0.8% target allocation, $175 max order를 강제한다.

## Gate policy

- Success requires OOS return > `+1.2181%`, OOS MDD < `0.1778%`, OOS Sharpe > `1.0`, liquidations `0`, and positive train/val.
- Replay survivor는 full live-equivalent raw-first backtest 후보일 뿐이며, sub-1 Sharpe는 성공이 아니다.

## Result

- Specs evaluated: `288`
- Replay survivors: `0`
- Success candidates: `0`
- Peak RSS: `253.211 MiB`

## Top candidates/failures

| rank | name | family | survivor | success | train ret | val ret | OOS ret | OOS MDD | OOS Sharpe | OOS trips | failed gates |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `fresh_compression_downside_short_lb6_thr140_c55_h12_sc05_st60_tp120` | `compression_expansion_downside_short` | False | False | -0.0712% | -0.0041% | -0.0026% | +0.0026% | -3.118093 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 2 | `fresh_compression_downside_short_lb6_thr140_c55_h12_sc05_st60_tp250` | `compression_expansion_downside_short` | False | False | -0.0712% | -0.0041% | -0.0026% | +0.0026% | -3.118093 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 3 | `fresh_compression_downside_short_lb6_thr140_c55_h12_sc05_st100_tp120` | `compression_expansion_downside_short` | False | False | -0.0712% | -0.0041% | -0.0026% | +0.0026% | -3.118093 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 4 | `fresh_compression_downside_short_lb6_thr140_c55_h12_sc05_st100_tp250` | `compression_expansion_downside_short` | False | False | -0.0712% | -0.0041% | -0.0026% | +0.0026% | -3.118093 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 5 | `fresh_compression_downside_short_lb6_thr140_c55_h24_sc05_st60_tp120` | `compression_expansion_downside_short` | False | False | -0.0585% | -0.0041% | -0.0026% | +0.0026% | -3.118093 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 6 | `fresh_compression_downside_short_lb6_thr140_c55_h24_sc05_st60_tp250` | `compression_expansion_downside_short` | False | False | -0.0585% | -0.0041% | -0.0026% | +0.0026% | -3.118093 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 7 | `fresh_compression_downside_short_lb6_thr140_c55_h24_sc05_st100_tp120` | `compression_expansion_downside_short` | False | False | -0.0585% | -0.0041% | -0.0026% | +0.0026% | -3.118093 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 8 | `fresh_compression_downside_short_lb6_thr140_c55_h24_sc05_st100_tp250` | `compression_expansion_downside_short` | False | False | -0.0585% | -0.0041% | -0.0026% | +0.0026% | -3.118093 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 9 | `fresh_compression_downside_short_lb6_thr200_c55_h12_sc05_st60_tp120` | `compression_expansion_downside_short` | False | False | -0.0373% | +0.0000% | -0.0026% | +0.0026% | -3.118093 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 10 | `fresh_compression_downside_short_lb6_thr200_c55_h12_sc05_st60_tp250` | `compression_expansion_downside_short` | False | False | -0.0373% | +0.0000% | -0.0026% | +0.0026% | -3.118093 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 11 | `fresh_compression_downside_short_lb6_thr200_c55_h12_sc05_st100_tp120` | `compression_expansion_downside_short` | False | False | -0.0373% | +0.0000% | -0.0026% | +0.0026% | -3.118093 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 12 | `fresh_compression_downside_short_lb6_thr200_c55_h12_sc05_st100_tp250` | `compression_expansion_downside_short` | False | False | -0.0373% | +0.0000% | -0.0026% | +0.0026% | -3.118093 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 13 | `fresh_compression_downside_short_lb6_thr200_c55_h24_sc05_st60_tp120` | `compression_expansion_downside_short` | False | False | -0.0312% | +0.0000% | -0.0026% | +0.0026% | -3.118093 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 14 | `fresh_compression_downside_short_lb6_thr200_c55_h24_sc05_st60_tp250` | `compression_expansion_downside_short` | False | False | -0.0312% | +0.0000% | -0.0026% | +0.0026% | -3.118093 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |
| 15 | `fresh_compression_downside_short_lb6_thr200_c55_h24_sc05_st100_tp120` | `compression_expansion_downside_short` | False | False | -0.0312% | +0.0000% | -0.0026% | +0.0026% | -3.118093 | 1 | `train_positive,val_positive,oos_return_beats_incumbent,oos_sharpe_gt_1,oos_trades_not_starved` |

## Decision

- No fresh-start candidate earned a full live-equivalent slot; do not promote or backtest a random vector-only shape.
- Blocked/failed families remain recorded in CSV/JSON with failed gates and top reject reasons.
